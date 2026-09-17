import copy

import pytest
import torch
from omegaconf import OmegaConf

from test_y import tiny_model, batch
from scripts.evaluate_single_stream_fid_is import (
    canonical_y_reveal_order, class_balanced_screening_subset, build_inception_score_split_plan,
)


def test_mar_mask_support_mean_and_default_unchanged():
    model = tiny_model()
    model.config.image_tokens_per_img = 256
    count = 8192
    types = torch.ones(count, 256, dtype=torch.long)
    spans = torch.zeros(count, 4, dtype=torch.long)
    spans[:, 0] = torch.arange(count)
    torch.manual_seed(7)
    baseline = model._sample_visible(types, types.bool(), spans)
    model.config.y_mask_distribution = "cosine"
    torch.manual_seed(7)
    assert torch.equal(model._sample_visible(types, types.bool(), spans), baseline)
    model.config.y_mask_distribution = "mar_truncnorm"
    model.config.y_empty_visible_prob = 0.
    torch.manual_seed(7)
    mask = ~model._sample_visible(types, types.bool(), spans)
    ratio = mask.float().mean(-1)
    assert (ratio >= .7).all() and (ratio <= 1).all()
    assert ratio.mean().item() == pytest.approx(.869, abs=.004)
    assert ratio.eq(1).float().mean() < .03


@pytest.mark.parametrize("schedule", ["constant", "linear"])
@pytest.mark.parametrize("cfg", [1., 3.5])
def test_reveal_guidance_is_constant_inside_each_flow_solve(monkeypatch, schedule, cfg):
    model = tiny_model().eval()
    b = batch()
    observed = []
    original = model.image_flow_head.sample
    def sample(*args, **kwargs):
        observed.append((kwargs["cfg"], kwargs["cfg_schedule"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(model.image_flow_head, "sample", sample)
    out, trace = model.generate("t2i", input_ids=b["X0_input_ids"], token_types=b["token_types"],
        sigma=b["flow_sigma"], spans=[(0, 2, 6)], flow_cfg=cfg, flow_num_steps=2,
        reveal_steps=2, reveal_cfg_schedule=schedule, return_trace=True)
    expected = [cfg, cfg] if schedule == "constant" else [1+(cfg-1)*.5, 1+(cfg-1)*.75]
    assert [x[0] for x in observed] == expected
    assert all(x[1] == "constant" for x in observed)
    assert trace["reveal_cfg_values"] == expected and torch.isfinite(out).all()


def test_screening_orders_and_classes_are_paired():
    all_orders = canonical_y_reveal_order([0, 1, 5], 42, 256)
    assert torch.equal(all_orders[[0, 2]], canonical_y_reveal_order([0, 5], 42, 256))
    class Dataset:
        img_ids = torch.arange(12)
        synsets = {i: f"class-{i//4}" for i in range(12)}
        def __len__(self):
            return 12
    subset = class_balanced_screening_subset(Dataset(), 2, 42)
    assert subset.indices == class_balanced_screening_subset(Dataset(), 2, 42).indices
    _, plan = build_inception_score_split_plan(subset, samples=6, splits=2)
    assert plan["class_count"] == 3 and plan["samples_per_class_min"] == 2
    assert plan["classes_per_split"] == [3, 3]


def test_marmask_recipe_only_changes_mask_and_run_identity():
    from utils.y_protocol import expected_config as base
    from utils.y_marmask_protocol import expected_config, CONFIG, validate_joint_dit_config
    original, changed = base(), expected_config()
    assert validate_joint_dit_config(OmegaConf.load(CONFIG))["optimizer_steps"] == 2000
    normalized = copy.deepcopy(changed)
    for key in ("project", "name", "identity"):
        normalized.experiment[key] = original.experiment[key]
    normalized.model.y_empty_visible_prob = original.model.y_empty_visible_prob
    del normalized.model.y_mask_distribution
    normalized.training.stop_after_steps = original.training.stop_after_steps
    normalized.evaluation.checkpoint = original.evaluation.checkpoint
    assert OmegaConf.to_container(normalized) == OmegaConf.to_container(original)
