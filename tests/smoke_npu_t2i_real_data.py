"""One-NPU T2I-only forward/backward smoke on the published ImageNet data."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import get_selfless_mask, load_model_tokenizer


CONFIG_PATH = Path(
    "configs/selfless/imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml"
)


def build_dataset(config, tokenizer) -> ImageNetFlowCacheDataset:
    params = config.dataset.params
    return ImageNetFlowCacheDataset(
        cache_path=params.cache_path,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        image_tokens_per_img=params.image_tokens_per_img,
        image_latent_dim=params.image_latent_dim,
        manifest_jsonl=params.manifest_jsonl,
        conditioning_mode=params.conditioning_mode,
        caption_jsonl=params.caption_jsonl,
        caption_list_key=params.caption_list_key,
        caption_list_text_key=params.caption_list_text_key,
        caption_path_key=params.caption_path_key,
        caption_id_key=params.caption_id_key,
        caption_validation_index=params.caption_validation_index,
        t2i_prompt_validation_index=params.t2i_prompt_validation_index,
        caption_sequence_modes=params.caption_sequence_modes,
        synthetic_text_index_manifest=params.synthetic_text_index_manifest,
        caption_t2i_prefix=params.caption_t2i_prefix,
        caption_i2t_prefix=params.caption_i2t_prefix,
        caption_include_original=params.caption_include_original,
        cache_caption_tokens=False,
        max_seq_length=params.max_seq_length,
        model_context_length=params.model_context_length,
        caption_manifest_sha256=params.caption_manifest_sha256,
        max_samples=32,
        seed=config.training.seed,
        emit_audit_metadata=False,
    )


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    torch.manual_seed(424242)
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    config = OmegaConf.load(CONFIG_PATH)
    # Keep the formal initialization and loss, but bound flow duplication for
    # a single smoke step. This does not alter any parameter shape.
    config.model.image_flow_batch_mul = 1
    config.model.image_uncond_prob = 0.0
    model, tokenizer = load_model_tokenizer(
        config=config,
        model_dtype=torch.bfloat16,
    )
    model = model.to(device).train()
    dataset = build_dataset(config, tokenizer)
    dataset.set_training_indices(range(len(dataset)))

    rows = []
    prompt_indices = []
    for epoch in range(13):
        dataset.set_epoch(epoch)
        row = dataset[0]
        if row["task_mode"] != "t2i":
            raise AssertionError(f"non-T2I task leaked into dataset: {row['task_mode']}")
        if int(row["caption_count"]) != 12:
            raise AssertionError(f"expected 12 T2I prompts, got {row['caption_count']}")
        prompt_indices.append(int(row["caption_index"]))
        rows.append(row)
    if set(prompt_indices[:12]) != set(range(12)):
        raise AssertionError(f"first prompt cycle is not without replacement: {prompt_indices}")
    if prompt_indices[12] != prompt_indices[0]:
        raise AssertionError(f"13th epoch did not begin the next prompt cycle: {prompt_indices}")
    if any(left == right for left, right in zip(prompt_indices, prompt_indices[1:])):
        raise AssertionError(f"adjacent epochs reused a prompt: {prompt_indices}")

    row = rows[0]
    batch = collate_imagenet_flow_cache(
        [row],
        pad_to_length=int(config.dataset.params.pad_to_length),
        pad_to_multiple_of=int(config.dataset.params.pad_to_multiple_of),
    )
    payload = {
        key: value.to(device, non_blocking=False)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    attention_mask = get_selfless_mask(
        sigma=payload["sigma"],
        seq_len=payload["input_ids"].shape[1],
        device=device,
        input_ids=payload["input_ids"],
        token_types=payload["token_types"],
        boi_token_id=int(config.model.boi_token_id),
    )
    output = model(
        X0_input_ids=payload["input_ids"],
        labels=payload["labels"],
        attention_mask=attention_mask,
        position_ids=payload["position_ids"],
        token_types=payload["token_types"],
        image_latents=payload["image_latents"],
        image_local_positions=payload["image_local_positions"],
        image_span_table=payload["image_span_table"],
        image_loss_mask=payload["image_loss_mask"],
        flow_sigma=payload["sigma"],
        record_flow_stats=False,
        use_cache=False,
    )
    losses = output.per_modality_loss
    counts = output.per_modality_count
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError("real-data T2I loss is not finite")
    if int(counts["text_tokens"].item()) != 0:
        raise AssertionError(f"T2I-only row produced text targets: {counts}")
    if int(counts["image_tokens"].item()) != 256:
        raise AssertionError(f"real T2I target count changed: {counts}")
    if not bool(
        torch.allclose(output.loss, losses["image_loss"], atol=1.0e-5, rtol=1.0e-5)
    ):
        raise AssertionError("T2I-only loss is not exactly the image-flow loss")

    output.loss.backward()
    gradients = {
        "flow_head": model.image_flow_head.net.final_layer.linear.weight.grad,
        "backbone": model.model.layers[0].self_attn.q_proj.weight.grad,
        "prompt_embedding": model.model.embed_tokens.weight.grad,
    }
    for label, gradient in gradients.items():
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise AssertionError(f"non-finite or missing {label} gradient")
    torch.npu.synchronize()
    report = {
        "checkpoint": str(config.model.model_path),
        "task_modes": batch["task_modes"],
        "prompt_indices_epochs_0_to_12": prompt_indices,
        "t2i_variants": int(row["caption_count"]),
        "serialized_length": int(row["serialized_length"]),
        "loss": float(output.loss.detach().cpu()),
        "image_loss": float(losses["image_loss"].cpu()),
        "text_tokens": int(counts["text_tokens"].cpu()),
        "image_tokens": int(counts["image_tokens"].cpu()),
        "npu": torch.npu.get_device_name(0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("REAL DATA T2I-ONLY NPU SMOKE PASS")


if __name__ == "__main__":
    main()
