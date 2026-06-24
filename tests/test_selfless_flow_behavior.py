import os
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from torch import nn
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import FlowMatchingHead, Qwen3ForCausalLM, Qwen3Model


def tiny_qwen3_config():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config.mask_token_id = 7
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    return config


class CaptureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        X0_hidden_states,
        XT_hidden_states,
        attention_mask,
        **kwargs,
    ):
        self.calls.append(
            {
                "x0": X0_hidden_states.detach().clone(),
                "xt": None if XT_hidden_states is None else XT_hidden_states.detach().clone(),
                "attention_mask": attention_mask,
            }
        )
        return X0_hidden_states, XT_hidden_states


class FakeLinear:
    def __init__(self, dtype=torch.float32):
        self.weight = torch.empty(1, dtype=dtype)


class FakeFinalLayer:
    def __init__(self):
        self.linear = FakeLinear()


class FakeFlowHead:
    def __init__(self, latent_dim):
        self.final_layer = FakeFinalLayer()
        self.latent_dim = latent_dim
        self.sample_calls = []

    def sample(self, z, num_steps, temperature, sample_method=None):
        self.sample_calls.append(
            {
                "z": z.detach().clone(),
                "num_steps": num_steps,
                "temperature": temperature,
                "sample_method": sample_method,
            }
        )
        return torch.zeros(z.shape[0], self.latent_dim, device=z.device, dtype=z.dtype)


class FakeInnerModel:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "calculate_likelihood": kwargs["calculate_likelihood"],
                "sigma": kwargs["attention_mask"],
                "image_latent_mask": kwargs["image_latent_mask"].detach().clone(),
                "image_latents": kwargs["image_latents"].detach().clone(),
                "token_types": kwargs["token_types"].detach().clone(),
            }
        )
        input_ids = kwargs["X0_input_ids"]
        hidden = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden_size,
            device=input_ids.device,
            dtype=kwargs["image_latents"].dtype,
        )
        return types.SimpleNamespace(last_hidden_state=hidden)


class SelflessFlowBehaviorTest(unittest.TestCase):
    def test_flow_head_scales_time_and_passes_condition_through(self):
        head = FlowMatchingHead(
            target_channels=4,
            z_channels=8,
            width=8,
            depth=1,
            time_scale=123.0,
        )
        captured = {}

        def capture_time(_, args, __):
            captured["time"] = args[0].detach().clone()

        def capture_condition(_, args, __):
            captured["condition"] = args[0].detach().clone()

        time_hook = head.time_embed.register_forward_hook(capture_time)
        cond_hook = head.cond_embed.register_forward_hook(capture_condition)
        try:
            x_t = torch.randn(3, 4)
            t = torch.tensor([0.0, 0.5, 1.0])
            z = torch.randn(3, 8) * 17.0 + 5.0
            pred = head.predict_velocity(x_t, t, z)
        finally:
            time_hook.remove()
            cond_hook.remove()

        self.assertEqual(tuple(pred.shape), (3, 4))
        self.assertTrue(torch.allclose(captured["time"], t * 123.0))
        self.assertTrue(torch.allclose(captured["condition"], z))

    def test_training_likelihood_uses_xt_all_mask_and_x0_real_content(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        capture = CaptureLayer()
        model.layers = nn.ModuleList([capture])
        model.norm = nn.Identity()
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

        output = model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            token_types=token_types,
            image_latents=image_latents,
            calculate_likelihood=True,
        )

        call = capture.calls[-1]
        mask_embeds = model.embed_tokens(
            torch.full_like(input_ids, config.mask_token_id)
        )
        self.assertTrue(torch.allclose(call["xt"], mask_embeds))

        expected_x0 = model.embed_tokens(input_ids)
        expected_x0[:, 1:3] = model.image_latent_proj(image_latents[:, 1:3])
        self.assertTrue(torch.allclose(call["x0"], expected_x0))

        self.assertTrue(torch.allclose(output.last_hidden_state, mask_embeds))

    def test_single_stream_unfilled_image_tokens_use_same_mask_embedding(self):
        config = tiny_qwen3_config()
        model = Qwen3Model(config)
        capture = CaptureLayer()
        model.layers = nn.ModuleList([capture])
        model.norm = nn.Identity()
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4]])
        token_types = torch.tensor([[0, 1, 1, 2]])
        image_latents = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        image_latent_mask = torch.tensor([[False, True, False, False]])

        model(
            X0_input_ids=input_ids,
            attention_mask=object(),
            token_types=token_types,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            calculate_likelihood=False,
        )

        call = capture.calls[-1]
        self.assertIsNone(call["xt"])

        expected_x0 = model.embed_tokens(input_ids.masked_fill(token_types == 1, config.mask_token_id))
        expected_x0[:, 1] = model.image_latent_proj(image_latents[:, 1])
        self.assertTrue(torch.allclose(call["x0"], expected_x0))
        self.assertTrue(torch.allclose(call["x0"][:, 2], model.embed_tokens(torch.tensor([config.mask_token_id]))))

    def test_single_stream_helper_uses_single_stream_masks_and_original_sigma(self):
        latent_dim = 4
        hidden_size = 8
        dummy = types.SimpleNamespace(
            config=types.SimpleNamespace(image_tokens_per_img=4),
            model=FakeInnerModel(hidden_size=hidden_size),
            flow_head=FakeFlowHead(latent_dim=latent_dim),
        )

        input_ids = torch.tensor([[10, 11, 7, 7, 7, 7, 12, 13]])
        token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 2]])
        sigma = torch.tensor([[0, 1, 4, 5, 3, 6, 2, 7]])
        spans = [(0, 2, 6)]

        with patch("utils.utils.get_selfless_mask", side_effect=lambda sigma, seq_len, device: sigma.detach().clone()):
            generated = Qwen3ForCausalLM.sample_image_latents_single_stream(
                dummy,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=latent_dim,
                flow_steps=3,
                flow_temperature=0.5,
                parallel_rate=1,
            )

        self.assertEqual(tuple(generated.shape), (1, latent_dim, 2, 2))
        self.assertEqual(len(dummy.model.calls), 4)
        self.assertEqual(len(dummy.flow_head.sample_calls), 4)

        for call in dummy.model.calls:
            self.assertFalse(call["calculate_likelihood"])
            self.assertTrue(torch.equal(call["sigma"], sigma))

        expected_masks = [
            [False, False, False, False],
            [False, False, True, False],
            [True, False, True, False],
            [True, True, True, False],
        ]
        for call, expected in zip(dummy.model.calls, expected_masks):
            self.assertEqual(call["image_latent_mask"][0, 2:6].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
