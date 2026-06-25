import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from torch import nn
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import FlowMatchingHead, ImageLatentProjector, Qwen3ForCausalLM, Qwen3Model
from utils.dataset_combined_flow import CombinedBatchDataLoader, TextArrowDataset, collate_text_arrow
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset, collate_imagenet_flow_cache


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


class FakeTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = 100 + len(self.vocab)
            ids.append(self.vocab[word])
        return ids


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
    def test_full_imagenet_cache_collate_trains_boi_and_eos_but_not_eoi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(16, dtype=torch.float16).view(2, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1, 2])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text(
                '{"img_id": 1, "synset": "n00000001"}\n'
                '{"img_id": 2, "synset": "n00000002"}\n'
            )
            (root / "mapping.txt").write_text(
                "n00000001 test class, alternate name\n"
                "n00000002 other class\n"
            )

            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=FakeTokenizer(),
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_template="a photo of {class_name}",
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=16,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=16)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        sigma = batch["sigma"][0]
        image_latents = batch["image_latents"][0]

        prompt_len = 5
        image_start = prompt_len + 1
        eoi_pos = image_start + 4
        eos_pos = eoi_pos + 1

        self.assertEqual(input_ids[prompt_len].item(), 11)
        self.assertEqual(input_ids[eoi_pos].item(), 12)
        self.assertEqual(input_ids[eos_pos].item(), 14)

        self.assertEqual(labels[prompt_len].item(), 11)
        self.assertEqual(labels[eoi_pos].item(), -100)
        self.assertEqual(labels[eos_pos].item(), 14)
        self.assertTrue(torch.equal(labels[image_start:eoi_pos], torch.full((4,), -100)))

        self.assertTrue(torch.equal(token_types[image_start:eoi_pos], torch.ones(4, dtype=torch.uint8)))
        self.assertTrue(torch.equal(image_latents[image_start:eoi_pos], latents[0]))

        self.assertEqual(sigma[prompt_len].item(), prompt_len)
        self.assertEqual(sigma[eoi_pos].item(), prompt_len + 1)
        self.assertTrue(torch.all(sigma[image_start:eoi_pos] > sigma[eoi_pos]))
        self.assertTrue(torch.all(sigma[eos_pos] > sigma[image_start:eoi_pos]))

    def test_full_imagenet_cache_collate_supports_image_first_templates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(8, dtype=torch.float16).view(1, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text('{"img_id": 1, "synset": "n00000001"}\n')
            (root / "mapping.txt").write_text("n00000001 test class\n")

            tokenizer = FakeTokenizer()
            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=tokenizer,
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_templates=["{image} this image shows {class_name}"],
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=16,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=16)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        sigma = batch["sigma"][0]

        image_start = 1
        eoi_pos = image_start + 4
        suffix_start = eoi_pos + 1
        eos_pos = suffix_start + 5

        self.assertEqual(input_ids[0].item(), 11)
        self.assertEqual(input_ids[eoi_pos].item(), 12)
        self.assertEqual(input_ids[eos_pos].item(), 14)
        self.assertTrue(torch.equal(token_types[suffix_start:eos_pos], torch.zeros(5, dtype=torch.uint8)))
        self.assertTrue(torch.equal(labels[suffix_start:eos_pos], input_ids[suffix_start:eos_pos]))
        self.assertEqual(labels[eoi_pos].item(), -100)
        self.assertTrue(torch.all(sigma[image_start:eoi_pos] > sigma[eoi_pos]))
        self.assertTrue(torch.all(sigma[suffix_start:eos_pos] > sigma[image_start:eoi_pos].max()))
        self.assertTrue(torch.all(sigma[eos_pos] > sigma[suffix_start:eos_pos]))

    def test_full_imagenet_cache_uses_category_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latents = torch.arange(8, dtype=torch.float16).view(1, 4, 2)
            torch.save({"latents": latents, "img_ids": torch.tensor([1])}, root / "latents.pt")
            (root / "manifest.jsonl").write_text('{"img_id": 1, "synset": "n00000001"}\n')
            (root / "mapping.txt").write_text("n00000001 tabby cat\n")
            (root / "prompts.yaml").write_text(
                "fallback_templates:\n"
                "  - 'fallback object {image}'\n"
                "groups:\n"
                "  - name: cats\n"
                "    keywords: ['cat']\n"
                "    templates:\n"
                "      - 'cat specific prefix {class_name} {image} cat specific suffix'\n"
            )

            tokenizer = FakeTokenizer()
            dataset = ImageNetFlowCacheDataset(
                cache_path=str(root / "latents.pt"),
                tokenizer=tokenizer,
                boi_token_id=11,
                eoi_token_id=12,
                mask_token_id=13,
                eos_token_id=14,
                image_tokens_per_img=4,
                image_latent_dim=2,
                manifest_jsonl=str(root / "manifest.jsonl"),
                prompt_templates_path=str(root / "prompts.yaml"),
                synset_mapping_path=str(root / "mapping.txt"),
                max_seq_length=32,
            )

            batch = collate_imagenet_flow_cache([dataset[0]], pad_to_length=32)

        input_ids = batch["input_ids"][0]
        labels = batch["labels"][0]
        token_types = batch["token_types"][0]
        text_tokens = input_ids[token_types == 0].tolist()
        text_labels = labels[(token_types == 0) & (labels != -100)].tolist()

        expected = tokenizer.encode("cat specific prefix tabby cat cat specific suffix", add_special_tokens=False)
        self.assertEqual(text_tokens, expected)
        self.assertEqual(text_labels, expected)
        self.assertEqual(
            batch["input_ids"][0, 0].item(),
            tokenizer.encode("cat", add_special_tokens=False)[0],
        )

    def test_image_latent_projector_handles_bfloat16_weights(self):
        projector = ImageLatentProjector(latent_dim=4, hidden_size=8, projector_width=16).to(dtype=torch.bfloat16)
        latents = torch.randn(5, 4, dtype=torch.float32)
        out = projector(latents)
        self.assertEqual(tuple(out.shape), (5, 8))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_text_arrow_dataset_slices_and_collates_as_text_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            HFDataset.from_dict({"input_ids": [list(range(20)), list(range(100, 112))]}).save_to_disk(root)
            dataset = TextArrowDataset(
                tokenized_path=str(root),
                max_seq_length=8,
                pad_token_id=0,
                seed=123,
            )
            batch = collate_text_arrow([dataset[0], dataset[1]], pad_token_id=0, pad_to_multiple_of=8)

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertTrue(torch.equal(batch["token_types"], torch.zeros(2, 8, dtype=torch.uint8)))
        self.assertTrue(torch.equal(batch["sigma"][0], torch.arange(8)))
        self.assertTrue(torch.equal(batch["labels"], batch["input_ids"]))
        self.assertEqual(batch["pack_stats"].tolist(), [16, 0, 0, 8])

    def test_combined_loader_mixes_whole_batches(self):
        image_batch = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "token_types": torch.ones(2, 6, dtype=torch.uint8),
            "sigma": torch.zeros(2, 6, dtype=torch.long),
            "labels": torch.full((2, 6), -100, dtype=torch.long),
            "image_latents": torch.zeros(2, 6, 4),
        }
        text_batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long) * 2,
            "token_types": torch.zeros(2, 8, dtype=torch.uint8),
            "sigma": torch.arange(8).unsqueeze(0).expand(2, 8),
            "labels": torch.ones(2, 8, dtype=torch.long) * 2,
        }
        image_loader = DataLoader([image_batch, image_batch], batch_size=None)
        text_loader = DataLoader([text_batch, text_batch], batch_size=None)
        combined = CombinedBatchDataLoader(
            image_loader=image_loader,
            text_loader=text_loader,
            text_batch_ratio=1.0,
            seed=1,
            mode="train",
        )

        batches = list(combined)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(batch["batch_source"] == "text" for batch in batches))
        self.assertTrue(all("image_latents" not in batch for batch in batches))
        self.assertTrue(all(torch.equal(batch["token_types"], torch.zeros_like(batch["token_types"])) for batch in batches))

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
