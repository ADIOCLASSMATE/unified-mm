import torch
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.utils import get_selfless_mask


def _tiny_model():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=True,
    )
    values = {
        "mask_token_id": 7,
        "image_mask_token_id": 8,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "image_latent_dim": 4,
        "image_tokens_per_img": 4,
        "image_flow_width": 32,
        "image_flow_depth": 1,
        "image_flow_num_sampling_steps": "10",
        "image_flow_batch_mul": 1,
        "image_flow_time_sampling": "uniform",
        "image_input_noise_strength": 0.0,
        "lambda_text": 0.05,
        "lambda_image": 1.0,
        "use_flex_attention": True,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return Qwen3ForCausalLM(config).train()


def test_pure_text_forward_skips_flow_compute_but_connects_zero_gradients():
    model = _tiny_model()

    def forbidden_flow_forward(*_args, **_kwargs):
        raise AssertionError("pure-text microbatch executed the flow head")

    model.image_flow_head.forward = forbidden_flow_forward
    input_ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
    sigma = torch.arange(4, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    labels[:, 0] = -100
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = get_selfless_mask(
        sigma=sigma,
        seq_len=input_ids.shape[1],
        device=input_ids.device,
        input_ids=input_ids,
        token_types=token_types,
        boi_token_id=11,
        segment_ids=segment_ids,
    )

    output = model(
        X0_input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        token_types=token_types,
        flow_sigma=sigma,
        image_span_table=torch.empty(0, 5, dtype=torch.long),
        image_loss_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        compute_text_loss=True,
        compute_image_loss=False,
        return_logits=False,
    )

    assert torch.isfinite(output.loss)
    assert output.per_modality_count["text_tokens"].item() == 3
    assert output.per_modality_count["image_tokens"].item() == 0
    assert output.per_modality_loss["image_loss"].item() == 0.0
    output.loss.backward()

    skipped_parameters = [
        *model.image_flow_condition_proj.parameters(),
        *model.image_flow_head.parameters(),
    ]
    assert skipped_parameters
    assert all(parameter.grad is not None for parameter in skipped_parameters)
    assert all(
        torch.count_nonzero(parameter.grad).item() == 0
        for parameter in skipped_parameters
    )
