#!/usr/bin/env python3
"""Compare repeated and shared content using Ascend BF16 attention/backward."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch_npu  # noqa: F401
from models.modeling_model.modeling_selfless_flow import X0ContentFlowLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.npu.is_available() and torch.npu.device_count() == 16
    torch.npu.set_device(0)
    torch.manual_seed(927)
    reference = X0ContentFlowLoss(
        target_channels=16, z_channels=1024, width=1280, depth=3,
        num_sampling_steps=10, grad_checkpointing=True,
        image_tokens_per_img=256, flow_head_attention_contract="xlnet_content_diagonal",
    )
    with torch.no_grad():
        for block in reference.net.blocks:
            block.adaLN_modulation[-1].weight.normal_(0, 0.005)
            block.adaLN_modulation[-1].bias.normal_(0, 0.02)
        reference.net.final_layer.linear.weight.normal_(0, 0.01)
        reference.net.final_layer.linear.bias.normal_(0, 0.01)
    reference = reference.to(device="npu:0", dtype=torch.bfloat16).train()
    shared = copy.deepcopy(reference)
    batch, tokens, repeats = 2, 256, 4
    target = torch.randn(batch, tokens, 16, device="npu:0")
    content = target + 0.03 * torch.randn_like(target)
    query_condition = torch.randn(batch, tokens, 1024, device="npu:0", dtype=torch.bfloat16)
    content_condition = torch.randn_like(query_condition)
    positions = torch.stack([torch.randperm(tokens, device="npu:0") for _ in range(batch)])
    sigma = torch.stack([torch.randperm(tokens, device="npu:0") for _ in range(batch)]).float()
    state = reference.sample_training_state(target.repeat(repeats, 1, 1))
    results = []
    for model, use_shared in ((reference, False), (shared, True)):
        z = query_condition.detach().clone().requires_grad_()
        c = content_condition.detach().clone().requires_grad_()
        x = content.detach().clone().requires_grad_()
        loss = model(
            target=target.repeat(repeats, 1, 1), z=z.repeat(repeats, 1, 1),
            sigma=sigma.repeat(repeats, 1), image_positions=positions.repeat(repeats, 1),
            context_latents=x if use_shared else x.repeat(repeats, 1, 1),
            context_conditions=c if use_shared else c.repeat(repeats, 1, 1),
            training_state=state, record_stats=False,
        )
        loss.backward()
        torch.npu.synchronize()
        results.append((float(loss.item()), {"query_condition": z.grad, "content_condition": c.grad, "content": x.grad}))
    loss_error = abs(results[0][0] - results[1][0]) / max(abs(results[0][0]), 1e-12)
    errors = {}
    for (name, expected), (actual_name, actual) in zip(reference.named_parameters(), shared.named_parameters(), strict=True):
        assert name == actual_name
        if expected.grad is None:
            assert actual.grad is None
            continue
        assert actual.grad is not None and bool(torch.isfinite(actual.grad).all().item()), name
        denominator = expected.grad.float().norm().clamp_min(1e-12)
        errors[name] = float(((actual.grad.float() - expected.grad.float()).norm() / denominator).item())
    for name, expected in results[0][1].items():
        actual = results[1][1][name]
        errors[name] = float(((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item())
    worst = max(errors, key=errors.get)
    report = {"schema": "shared_flow_content_npu_parity_v1", "reference_loss": results[0][0],
              "shared_loss": results[1][0], "relative_loss_error": loss_error,
              "worst_gradient": worst, "max_relative_gradient_l2_error": errors[worst],
              "gradient_errors": errors, "dtype": "bfloat16", "device": "Ascend 910B",
              "layout": shared.net.last_training_batch_layout,
              "passed": loss_error < 0.002 and errors[worst] < 0.03}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "gradient_errors"}), flush=True)
    assert report["passed"], "shared content exceeds BF16 parity tolerance"


if __name__ == "__main__":
    main()
