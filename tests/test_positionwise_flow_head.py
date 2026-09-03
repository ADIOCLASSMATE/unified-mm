import torch

from models.modeling_model.image_flow_loss_positionwise import (
    PositionwiseFlowLoss,
)


def _head() -> PositionwiseFlowLoss:
    torch.manual_seed(17)
    head = PositionwiseFlowLoss(
        target_channels=4,
        z_channels=8,
        depth=3,
        width=32,
        num_sampling_steps=10,
        image_tokens_per_img=4,
    )
    with torch.no_grad():
        for block in head.net.res_blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
            block.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        head.net.final_layer.adaLN_modulation[-1].weight.normal_(0.0, 0.1)
        head.net.final_layer.adaLN_modulation[-1].bias.normal_(0.0, 0.1)
        head.net.final_layer.linear.weight.normal_(0.0, 0.1)
        head.net.final_layer.linear.bias.normal_(0.0, 0.1)
    return head


def test_positionwise_contract_has_no_context_or_cache():
    head = _head()
    assert head.net.position_contract() == {
        "schema": "positionwise_flow_head_v1",
        "architecture": "positionwise_adaln_mlp",
        "cross_token_attention": False,
        "uses_content_latents": False,
        "uses_image_position": False,
    }
    assert head.empty_latent_mixer_cache(batch_size=2, capacity=4) is None
    assert not hasattr(head, "append_latent_mixer_cache")
    assert not any("attn" in name or "cross" in name for name, _ in head.named_parameters())


def test_token_output_is_invariant_to_other_tokens_and_context():
    head = _head()
    torch.manual_seed(23)
    x = torch.randn(2, 4, 4)
    t = torch.rand(2, 4)
    z = torch.randn(2, 4, 8)
    baseline = head.velocity(
        x,
        t,
        z,
        context_latents=torch.randn(2, 4, 4),
        context_mask=torch.ones(2, 4, 4, dtype=torch.bool),
    )

    changed_x = x.clone()
    changed_z = z.clone()
    changed_x[:, 1:] += 1000.0
    changed_z[:, 1:] -= 1000.0
    changed = head.velocity(
        changed_x,
        t,
        changed_z,
        context_latents=torch.randn(2, 4, 4) * 1.0e6,
        context_mask=torch.zeros(2, 4, 4, dtype=torch.bool),
    )
    torch.testing.assert_close(baseline[:, 0], changed[:, 0])


def test_vectorized_sequence_matches_flattened_tokens():
    head = _head().eval()
    torch.manual_seed(29)
    x = torch.randn(2, 4, 4)
    t = torch.rand(2, 4)
    z = torch.randn(2, 4, 8)
    sequence = head.velocity(x, t, z)
    flattened = head.velocity(
        x.reshape(-1, 4),
        t.reshape(-1),
        z.reshape(-1, 8),
    ).reshape_as(sequence)
    torch.testing.assert_close(sequence, flattened)


def test_sampling_reuses_exact_initial_noise_and_is_finite():
    head = _head().eval()
    torch.manual_seed(31)
    z = torch.randn(5, 8)
    initial_noise = torch.randn(5, 4)
    first = head.sample(
        z,
        initial_noise=initial_noise,
        solver="heun",
        num_steps=2,
    )
    torch.manual_seed(999)
    second = head.sample(
        z,
        initial_noise=initial_noise,
        solver="heun",
        num_steps=2,
    )
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()
