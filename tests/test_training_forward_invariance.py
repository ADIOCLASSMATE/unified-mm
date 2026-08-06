import pytest
import torch
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.utils import get_selfless_mask


def _tiny_training_model(device: torch.device):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 4
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_input_noise_strength = 0.0
    config.image_uncond_prob = 0.0
    config.backbone_attention_output_gate = "per_head_identity_sigmoid"
    config.use_flex_attention = True
    return Qwen3ForCausalLM(config).to(
        device=device,
        dtype=torch.bfloat16,
    ).train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_disabling_debug_stats_preserves_training_forward_and_gradients():
    device = torch.device("cuda")
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    model = _tiny_training_model(device)

    input_ids = torch.tensor(
        [[21, 11, 8, 8, 8, 8, 12, 2]],
        device=device,
        dtype=torch.long,
    )
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        device=device,
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [[0.0, 0.0, 0.75, 0.25, 1.0, 0.5, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    image_latents = torch.zeros(
        1,
        8,
        4,
        device=device,
        dtype=torch.float32,
    )
    image_latents[:, 2:6].normal_()
    image_span_table = torch.tensor(
        [[0, 0, 2, 6, 1]],
        device=device,
        dtype=torch.long,
    )
    image_local_positions = torch.tensor(
        [[-1, -1, 0, 1, 2, 3, -1, -1]],
        device=device,
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    attention_mask = get_selfless_mask(
        sigma=sigma,
        seq_len=input_ids.shape[1],
        device=device,
        input_ids=input_ids,
        token_types=token_types,
        boi_token_id=11,
    )

    def run(record_flow_stats: bool):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(987)
        torch.cuda.manual_seed_all(987)
        output = model(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            token_types=token_types,
            image_span_table=image_span_table,
            image_local_positions=image_local_positions,
            image_latents=image_latents,
            flow_sigma=sigma,
            record_flow_stats=record_flow_stats,
            record_backbone_gate_stats=record_flow_stats,
            backbone_gate_stats_level="summary",
            return_logits=False,
        )
        output.loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return (
            output.loss.detach().clone(),
            output.last_hidden_state.detach().clone(),
            gradients,
        )

    with_stats = run(True)
    without_stats = run(False)

    torch.testing.assert_close(without_stats[0], with_stats[0], rtol=0, atol=0)
    torch.testing.assert_close(without_stats[1], with_stats[1], rtol=0, atol=0)
    assert without_stats[2].keys() == with_stats[2].keys()
    for name in without_stats[2]:
        torch.testing.assert_close(
            without_stats[2][name],
            with_stats[2][name],
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )
