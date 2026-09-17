import pytest
import torch
from torch.utils.data import Subset

from scripts.evaluate_single_stream_fid_is import (
    IS_SPLIT_ASSIGNMENT_STRATIFIED,
    build_inception_score_split_plan,
    is_formal_flow_protocol,
)
from scripts.image_evaluation_metrics import InceptionScoreMoments
from scripts.summarize_unified_evaluation import (
    validate_t2i_generation_protocol,
    validate_t2i_is_protocol,
)


class _Dataset:
    def __init__(self, classes: int, samples_per_class: int):
        image_ids = []
        self.synsets = {}
        for class_index in range(classes):
            synset = f"n{class_index:08d}"
            for within_class_index in range(samples_per_class):
                image_id = class_index * 1000 + within_class_index
                image_ids.append(image_id)
                self.synsets[image_id] = synset
        self.img_ids = torch.tensor(image_ids, dtype=torch.long)
        self.dataset_split = "val"

    def __len__(self):
        return int(self.img_ids.numel())


def test_stratified_is_plan_balances_every_class_in_every_split():
    dataset = _Dataset(classes=1000, samples_per_class=50)
    split_ids, plan = build_inception_score_split_plan(
        dataset,
        samples=50000,
        splits=10,
    )

    assert len(split_ids) == 50000
    assert plan == {
        "assignment": IS_SPLIT_ASSIGNMENT_STRATIFIED,
        "samples": 50000,
        "splits": 10,
        "class_stratified": True,
        "source_dataset_split": "val",
        "within_class_order": "ascending_stable_image_id",
        "class_count": 1000,
        "samples_per_class_min": 50,
        "samples_per_class_max": 50,
        "samples_per_split": [5000] * 10,
        "classes_per_split": [1000] * 10,
        "samples_per_class_per_split_min": 5,
        "samples_per_class_per_split_max": 5,
    }
    assert is_formal_flow_protocol(
        shared_real_count=50000,
        samples=50000,
        is_splits=10,
        fid_feature=2048,
        is_split_plan=plan,
    )


def test_stratified_is_assignment_is_independent_of_dataset_order():
    dataset = _Dataset(classes=3, samples_per_class=6)
    original_ids, _ = build_inception_score_split_plan(
        dataset,
        samples=18,
        splits=3,
    )
    reverse_order = list(reversed(range(len(dataset))))
    reversed_ids, _ = build_inception_score_split_plan(
        Subset(dataset, reverse_order),
        samples=18,
        splits=3,
    )

    original_by_image_id = {
        int(dataset.img_ids[row]): original_ids[row]
        for row in range(len(dataset))
    }
    reversed_by_image_id = {
        int(dataset.img_ids[base_row]): reversed_ids[loader_row]
        for loader_row, base_row in enumerate(reverse_order)
    }
    assert reversed_by_image_id == original_by_image_id


def test_inception_score_moments_accepts_explicit_split_ids():
    moments = InceptionScoreMoments.zeros(
        splits=2,
        classes=3,
        device=torch.device("cpu"),
    )
    moments.update(
        torch.tensor(
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 0.0, 8.0],
            ]
        ),
        [0, 1, 0, 1],
    )

    assert moments.count.tolist() == [2, 2]
    mean, std, scores = moments.compute()
    assert mean > 1.9
    assert std == pytest.approx(0.0, abs=1e-12)
    assert scores[0] == pytest.approx(scores[1])


def test_inception_score_moments_rejects_invalid_split_ids():
    moments = InceptionScoreMoments.zeros(2, 3, torch.device("cpu"))
    with pytest.raises(ValueError, match="must match the logits batch"):
        moments.update(torch.zeros(2, 3), [0])
    with pytest.raises(ValueError, match="must be in"):
        moments.update(torch.zeros(1, 3), [2])


def _formal_t2i_protocol():
    return {
        "metric_protocol": {
            "is_split_assignment": IS_SPLIT_ASSIGNMENT_STRATIFIED,
            "is_split_plan": {
                "assignment": IS_SPLIT_ASSIGNMENT_STRATIFIED,
                "source_dataset_split": "val",
                "samples": 50000,
                "splits": 10,
                "class_count": 1000,
                "samples_per_class_min": 50,
                "samples_per_class_max": 50,
                "samples_per_split": [5000] * 10,
                "classes_per_split": [1000] * 10,
                "samples_per_class_per_split_min": 5,
                "samples_per_class_per_split_max": 5,
            },
        }
    }


def test_summary_accepts_only_balanced_formal_stratified_is():
    payload = _formal_t2i_protocol()
    assert validate_t2i_is_protocol(payload, profile="formal") == (
        payload["metric_protocol"]["is_split_plan"]
    )


def _formal_generation_metrics():
    split_values = [2.0] * 10
    payload = _formal_t2i_protocol()
    payload.update(
        {
            "schema": "selfless_imagenet_val_t2i_fid_is_v2",
            "runtime_hashing_enabled": False,
            "project_formal_protocol": True,
            "leaderboard_comparable_to_adm_dit": False,
            "split": "val",
            "real_source": "cached_original_imagenet_val",
            "samples_requested": 50_000,
            "samples_evaluated": 50_000,
            "strategies": {
                "spatial_halton": {
                    "count": 50_000,
                    "fid": 5.0,
                    "inception_score_mean": 2.0,
                    "inception_score_std": 0.0,
                    "inception_score_splits": split_values,
                    "generation_samples_per_second": 20.0,
                }
            },
        }
    )
    payload["metric_protocol"].update(
        {
            "protocol_name": (
                "imagenet_val_fid50k_torch_fidelity_stratified_is"
            ),
            "reference_distribution": "imagenet_val_50000",
            "comparison_scope": "same_protocol_only",
            "not_adm_dit_reason": (
                "validation_reference_and_pytorch_torch_fidelity_extractor"
            ),
            "fid_reducer": "symmetric_eigendecomposition",
            "fid_computed": True,
            "is_std": "population",
            "is_splits": 10,
        }
    )
    return payload


def test_generation_summary_accepts_complete_project_formal_protocol():
    plan, selected = validate_t2i_generation_protocol(
        _formal_generation_metrics(), profile="formal"
    )

    assert plan["samples"] == 50_000
    assert selected["fid"] == 5.0
    assert selected["inception_score_splits"] == [2.0] * 10


@pytest.mark.parametrize(
    ("field", "invalid", "match"),
    [
        ("leaderboard_comparable_to_adm_dit", True, "comparability"),
        ("samples_evaluated", 49_999, "sample counts"),
    ],
)
def test_generation_summary_rejects_invalid_formal_contract(
    field, invalid, match
):
    payload = _formal_generation_metrics()
    payload[field] = invalid

    with pytest.raises(ValueError, match=match):
        validate_t2i_generation_protocol(payload, profile="formal")


def test_generation_summary_rejects_inconsistent_is_moments():
    payload = _formal_generation_metrics()
    payload["strategies"]["spatial_halton"]["inception_score_mean"] = 3.0

    with pytest.raises(ValueError, match="mean is inconsistent"):
        validate_t2i_generation_protocol(payload, profile="formal")


@pytest.mark.parametrize("trace_name,head_calls,error", [
    ("joint", 20, None),
    ("joint", 10, "solver-specific"),
    ("spatial_halton", 20, "cache"),
])
def test_joint_checkpoint_summary_uses_and_validates_its_whole_image_trace(
    tmp_path, monkeypatch, trace_name, head_calls, error
):
    import json
    from types import SimpleNamespace
    from scripts import summarize_unified_evaluation as summary
    from utils.evaluation_model_source import EvaluationModelSource

    checkpoint = tmp_path / "hf_model-final-ema"
    source = EvaluationModelSource(checkpoint, "hf_final_ema", 31800, 64, {})
    root = tmp_path / "evaluation"
    validation_root = root / "validation"
    captions = validation_root / "validation_i2t_captions/step-00031800/captions.jsonl"
    captions.parent.mkdir(parents=True)
    captions.write_text('{}\n')
    images = validation_root / "validation_flow_images"
    images.mkdir()
    (images / "step-00031800-joint.png").touch()
    t2i = _formal_generation_metrics()
    t2i.update(weight_source=source.kind, model_path=str(checkpoint),
               evaluation_model_source=source.report(), model_source_load=source.report())
    validation_keys = (
        "loss", "loss_i2t", "loss_t2i", "ppl_text", "weighted_contribution_i2t",
        "weighted_contribution_t2i", "weighted_contribution_total",
    )
    payloads = {
        "evaluation_run.json": dict(complete=True, global_step=31800,
            checkpoint=str(checkpoint), weight_source=source.kind, imagenet_split="val",
            runtime_hashing_enabled=False, dual_stream_attention_contract="xlnet_content_diagonal"),
        "validation_metrics_step_31800.json": {"metrics": {f"val/{key}": 1.0 for key in validation_keys}},
        "metrics.json": t2i,
        "validation_generation_step_31800.json": dict(
            architecture_variant="selfless_joint_dit", generation_entry="model.generate", use_cache=False,
            strategies={trace_name: dict(backbone_kv_cache_enabled=False,
                generation_mode="joint_dit_full_image_flow", solver="heun", steps=10,
                backbone_calls=1, flow_head_calls=head_calls)}),
    }
    monkeypatch.setattr(summary, "parse_args", lambda: SimpleNamespace(
        checkpoint=checkpoint, output_root=root, profile="formal"))
    monkeypatch.setattr(summary, "resolve_evaluation_model_source", lambda _: source)
    monkeypatch.setattr(summary, "load_json", lambda path: payloads[path.name])
    if error:
        with pytest.raises(ValueError, match=error):
            summary.main()
    else:
        summary.main()
        result = json.loads((root / "evaluation_summary.json").read_text())
        assert result["complete"] is True
        assert result["qualitative_artifacts"]["backbone_kv_cache_enabled"] is False
        assert result["t2i_fid_is"]["strategy"] == "spatial_halton"
