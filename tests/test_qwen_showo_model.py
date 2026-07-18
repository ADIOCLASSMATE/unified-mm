import types

import torch
from torch.nn import functional as F
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from models.modeling_model.modeling_qwen_showo import (
    QwenShowOForCausalLM,
    official_showo_ranking_temperature,
)


def tiny_config(
    *,
    vocab_size: int = 16,
    image_offset=None,
    image_vocab_size: int = 8,
):
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.image_vocab_size = image_vocab_size
    config.image_loss_chunk_size = 1
    config.lambda_image = 1.0
    config.lambda_text = 0.0
    if image_offset is not None:
        config.image_offset = image_offset
        config.image_mask_token_id = image_offset + image_vocab_size
    return config


def omni_mask(length: int):
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool))
    # Treat positions [2, 4) as a bidirectional image span for this test.
    allowed[2:4, :4] = True
    mask = torch.full((1, 1, length, length), torch.finfo(torch.float32).min)
    mask.masked_fill_(allowed[None, None], 0.0)
    return mask


def test_official_showo_temperature_schedule_is_cumulative():
    temperatures = [
        official_showo_ranking_temperature(1.0, step, 4)
        for step in range(4)
    ]

    assert temperatures == [0.75, 0.375, 0.09375, 0.0]
    # A non-cumulative linear schedule would be 0.5 at step 1.
    assert temperatures[1] != 0.5


def test_base_checkpoint_shape_is_preserved_until_explicit_vocab_configuration():
    config = tiny_config(image_offset=18)
    model = QwenShowOForCausalLM(config)
    original_embedding = model.get_input_embeddings().weight.detach().clone()

    # __init__ must retain the base-Qwen shape so from_pretrained can load it.
    assert model.get_input_embeddings().num_embeddings == 16

    mask_token_id = model.configure_image_vocabulary(
        image_mask_token_id=26,
        image_loss_chunk_size=2,
        resize_embeddings=True,
    )

    assert mask_token_id == 26
    assert model.config.image_offset == 18
    assert model.config.image_vocab_size == 8
    assert model.config.image_mask_token_id == 26
    assert model.config.image_loss_chunk_size == 2
    assert model.config.vocab_size == 27
    assert model.get_input_embeddings().num_embeddings == 27
    assert model.get_output_embeddings().out_features == 27
    assert torch.equal(
        model.get_input_embeddings().weight[:16],
        original_embedding,
    )
    assert not hasattr(model, "image_embed_tokens")
    assert not hasattr(model, "image_lm_head")


def test_base_qwen_from_pretrained_loads_before_vocabulary_expansion(tmp_path):
    config = tiny_config()
    base_model = HFQwen3ForCausalLM(config)
    base_model.save_pretrained(tmp_path)

    load_config = Qwen3Config.from_pretrained(tmp_path)
    load_config.image_offset = 18
    load_config.image_vocab_size = 8
    load_config.image_mask_token_id = 26
    model = QwenShowOForCausalLM.from_pretrained(
        tmp_path,
        config=load_config,
    )

    assert model.get_input_embeddings().num_embeddings == 16
    model.configure_image_vocabulary(
        image_offset=18,
        image_vocab_size=8,
        image_mask_token_id=26,
        resize_embeddings=True,
    )
    assert model.get_input_embeddings().num_embeddings == 27


def test_same_position_image_loss_uses_full_unified_vocab_in_chunks():
    torch.manual_seed(0)
    config = tiny_config()
    model = QwenShowOForCausalLM(config)
    image_mask_token_id = model.configure_image_vocabulary(
        image_offset=16,
        image_vocab_size=8,
    )
    model.train()

    input_ids = torch.tensor([[1, 2, image_mask_token_id, image_mask_token_id, 3]])
    token_types = torch.tensor([[0, 2, 1, 1, 2]])
    labels = torch.full_like(input_ids, -100)
    labels[0, 2:4] = torch.tensor([4, 5])  # Raw MAGVIT codes are accepted.
    attention_mask = omni_mask(input_ids.shape[1])

    output = model(
        input_ids=input_ids,
        token_types=token_types,
        attention_mask=attention_mask,
        labels=labels,
    )

    image_hidden = output.last_hidden_state[token_types == 1]
    unified_targets = torch.tensor([20, 21])
    manual_logits = model.unified_logits(image_hidden)
    manual_loss = F.cross_entropy(manual_logits, unified_targets)
    manual_correct = (manual_logits.argmax(dim=-1) == unified_targets).sum()

    assert output.logits is None
    assert output.image_token_count.item() == 2
    assert output.image_token_correct.item() == manual_correct.item()
    assert torch.allclose(output.loss, manual_loss, atol=1.0e-6)
    assert torch.allclose(
        output.per_modality_loss["image_loss"],
        manual_loss.detach(),
        atol=1.0e-6,
    )

    output.loss.backward()
    assert model.get_output_embeddings().weight.grad is not None


def test_forward_accepts_raw_or_offset_image_ids_and_arbitrary_4d_additive_mask():
    torch.manual_seed(1)
    config = tiny_config()
    model = QwenShowOForCausalLM(config)
    model.configure_image_vocabulary(image_offset=16, image_vocab_size=8)
    model.eval()

    raw_input = torch.tensor([[1, 2, 3, 4, 5]])
    unified_input = raw_input.clone()
    unified_input[0, 2:4] += 16
    token_types = torch.tensor([[0, 2, 1, 1, 2]])
    attention_mask = omni_mask(raw_input.shape[1])

    raw_output = model(
        input_ids=raw_input,
        token_types=token_types,
        attention_mask=attention_mask,
    )
    unified_output = model(
        input_ids=unified_input,
        token_types=token_types,
        attention_mask=attention_mask,
        return_logits=True,
    )

    assert raw_output.last_hidden_state.shape == (1, 5, config.hidden_size)
    assert unified_output.logits.shape == (1, 5, 25)
    assert torch.allclose(
        raw_output.last_hidden_state,
        unified_output.last_hidden_state,
        atol=1.0e-6,
    )
    expected_image_logits = F.linear(
        raw_output.last_hidden_state[:, 2:4],
        model.get_output_embeddings().weight[16:24],
    )
    assert torch.allclose(
        model.image_logits(raw_output.last_hidden_state[:, 2:4]),
        expected_image_logits,
    )


def test_maskgit_returns_raw_codes_and_uses_official_cfg_formula():
    config = tiny_config(image_vocab_size=4)
    model = QwenShowOForCausalLM(config)
    image_mask_token_id = model.configure_image_vocabulary(
        image_offset=16,
        image_vocab_size=4,
    )

    def fake_forward(self, input_ids=None, **kwargs):
        del kwargs
        batch, length = input_ids.shape
        hidden = torch.zeros(batch, length, self.config.hidden_size)
        # Conditional rows start with 1; unconditional rows start with 0.
        hidden[..., 0] = (input_ids[:, :1] == 1).float()
        output = CausalLMOutputWithPast(logits=None)
        output["last_hidden_state"] = hidden
        return output

    def fake_image_logits(self, hidden_states):
        is_conditional = hidden_states[..., 0] > 0.5
        logits = torch.full((*hidden_states.shape[:-1], 4), -100.0)
        logits[..., 0] = torch.where(
            is_conditional,
            torch.full_like(logits[..., 0], -100.0),
            torch.full_like(logits[..., 0], 100.0),
        )
        logits[..., 1] = torch.where(
            is_conditional,
            torch.full_like(logits[..., 1], 100.0),
            torch.full_like(logits[..., 1], -100.0),
        )
        return logits

    model.forward = types.MethodType(fake_forward, model)
    model.image_logits = types.MethodType(fake_image_logits, model)

    input_ids = torch.tensor([[1, 2, image_mask_token_id, 3]])
    uncond_input_ids = torch.tensor([[0, 2, image_mask_token_id, 3]])
    token_types = torch.tensor([[0, 2, 1, 2]])
    image_token_mask = token_types == 1
    codes = model.generate_image_tokens_maskgit(
        input_ids=input_ids,
        token_types=token_types,
        attention_mask=omni_mask(4),
        uncond_input_ids=uncond_input_ids,
        uncond_token_types=token_types,
        uncond_attention_mask=omni_mask(4),
        image_token_mask=image_token_mask,
        timesteps=1,
        guidance_scale=1.0,
        temperature=0.0,
        sample_seeds=torch.tensor([123]),
    )

    # (1+s)*cond - s*uncond strongly selects conditional code 1.
    assert codes.shape == (1, 1)
    assert codes.dtype == torch.long
    assert codes.item() == 1


def test_maskgit_cosine_remasking_resolves_all_tokens_with_generator_list():
    config = tiny_config(image_vocab_size=4)
    model = QwenShowOForCausalLM(config)
    image_mask_token_id = model.configure_image_vocabulary(
        image_offset=16,
        image_vocab_size=4,
    )

    def fake_forward(self, input_ids=None, **kwargs):
        del kwargs
        batch, length = input_ids.shape
        hidden = torch.zeros(batch, length, self.config.hidden_size)
        hidden[..., 0] = torch.arange(length).float()
        output = CausalLMOutputWithPast(logits=None)
        output["last_hidden_state"] = hidden
        return output

    def fake_image_logits(self, hidden_states):
        desired = (hidden_states[..., 0].long() - 2) % 4
        logits = torch.full((*hidden_states.shape[:-1], 4), -100.0)
        logits.scatter_(-1, desired.unsqueeze(-1), 100.0)
        return logits

    model.forward = types.MethodType(fake_forward, model)
    model.image_logits = types.MethodType(fake_image_logits, model)

    input_ids = torch.tensor(
        [[1, 2, image_mask_token_id, image_mask_token_id,
          image_mask_token_id, image_mask_token_id, 3]]
    )
    token_types = torch.tensor([[0, 2, 1, 1, 1, 1, 2]])
    generator = torch.Generator().manual_seed(7)
    codes = model.generate_image_tokens_maskgit(
        input_ids=input_ids,
        token_types=token_types,
        image_token_mask=token_types == 1,
        timesteps=3,
        temperature=1.0,
        generators=[generator],
    )

    assert torch.equal(codes, torch.tensor([[0, 1, 2, 3]]))
    assert int(codes.min()) >= 0
    assert int(codes.max()) < 4
