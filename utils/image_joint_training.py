"""B-matched T2I/I2T joint training, with no ClimbMix initialization or cursor."""

from __future__ import annotations

from omegaconf import OmegaConf

from utils.experiment_registry import experiment_identity

CONFIG = "configs/selfless/unified_b_t2i_i2t_matched_ascend32.yaml"
REFERENCE = "configs/selfless/unified_b_t2i_only_matched_ascend16.yaml"
RUN = "unified-b-x0content-0p6b-t2i-i2t-bmatched-s42-r1"
HEAD_PARAMETERS = 164_072_976
TOTAL_PARAMETERS = 761_189_904


def validate_config(config) -> dict:
    """Compare the complete recipe against the already qualified only control."""
    expected = OmegaConf.load(REFERENCE)
    expected.experiment.project = expected.experiment.name = RUN
    expected.evaluation.checkpoint = f"output/{RUN}/hf_model-final-ema"
    expected.dataset.params.schedule = ["t2i", "i2t"]
    source = OmegaConf.to_container(expected.dataset.params.sources.t2i, resolve=True)
    source.pop("pad_to_length_schedule")
    expected.dataset.params.sources = {"t2i": source, "i2t": source}
    expected.training.total_batch_size = 2048
    expected.training.physical_tokens_per_optimizer_step = 1_048_576
    expected.training.target_physical_tokens = 100_049_879_040
    actual = OmegaConf.to_container(config, resolve=True)
    identity = experiment_identity(RUN, actual)
    actual["experiment"].pop("identity", None)

    def differences(value, reference, prefix=""):
        if isinstance(value, dict) and isinstance(reference, dict):
            paths = []
            for key in sorted(value.keys() | reference.keys()):
                path = f"{prefix}.{key}" if prefix else key
                paths.extend([path] if key not in value or key not in reference else
                             differences(value[key], reference[key], path))
            return paths
        return [] if value == reference else [prefix]

    changed = differences(actual, OmegaConf.to_container(expected, resolve=True))
    if changed:
        raise ValueError("image joint arm differs from B-matched recipe: " + ", ".join(changed))
    return {"schema": "unified_b_image_joint_matched_v1", "run_project": RUN,
            "identity": identity, "reference_config": REFERENCE,
            "world_size": 32, "nodes": 2, "npu_per_node": 16,
            "active_sources": ["t2i", "i2t"], "climbmix_loaded": False,
            "gradient_accumulation_steps": 2, "per_rank_micro_batch": 32,
            "global_rows_per_task_per_step": 1024, "max_train_steps": 95415,
            "physical_positions_per_task_per_step": 524288,
            "physical_positions_per_step": 1048576,
            "cumulative_positions_per_task": 50024939520,
            "total_physical_positions": 100049879040,
            "head_parameters": HEAD_PARAMETERS, "total_parameters": TOTAL_PARAMETERS,
            "matches_b_image_task_exposure": True,
            "matches_b_source_gradient_fraction": False}
