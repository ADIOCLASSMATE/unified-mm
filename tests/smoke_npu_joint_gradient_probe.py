"""One-NPU smoke for task-separated joint gradient measurement."""

from __future__ import annotations

import json

import torch
import torch_npu  # noqa: F401

from tests.smoke_npu_joint_caption import joint_batch, tiny_config
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.joint_gradient_probe import measure_gradient_probe_batch
from utils.utils import get_selfless_mask


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    payload = joint_batch(device)
    model = Qwen3ForCausalLM(tiny_config()).to(
        device=device,
        dtype=torch.bfloat16,
    ).train()
    attention_mask = get_selfless_mask(
        sigma=payload["sigma"],
        seq_len=payload["input_ids"].shape[1],
        device=device,
        input_ids=payload["input_ids"],
        token_types=payload["token_types"],
        boi_token_id=11,
    )

    def forward_losses():
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
            return_per_modality_loss_graph=True,
            use_cache=False,
        )
        return output.per_modality_loss_graph, output.per_modality_count

    result = measure_gradient_probe_batch(
        model,
        forward_losses,
        special_token_ids=[7, 8, 11, 12],
        reset_seed=lambda: torch.npu.manual_seed_all(424242),
    )
    if not all(parameter.grad is None for parameter in model.parameters()):
        raise AssertionError("probe failed to clear gradients")
    torch.npu.synchronize()
    print(json.dumps(result, indent=2, sort_keys=True))
    print("JOINT GRADIENT PROBE NPU SMOKE PASS")


if __name__ == "__main__":
    main()
