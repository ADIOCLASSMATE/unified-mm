from pathlib import Path

from omegaconf import OmegaConf

from utils import evaluation_paths as paths


def test_training_results_are_separate_and_sweep_arms_remain_distinct(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "EVALUATION_ROOT", tmp_path / "output/evaluation")
    first = paths.training_validation_root(tmp_path / "output/sweep-one/arm-a")
    second = paths.training_validation_root(tmp_path / "output/sweep-two/arm-a")
    assert first == tmp_path / "output/evaluation/training-validation/sweep-one/arm-a"
    assert first != second
    assert not first.is_relative_to(tmp_path / "output/sweep-one")


def test_explicit_standalone_result_path_does_not_return_to_training_directory():
    config = OmegaConf.create({"experiment": {"output_dir": "/training/run", "validation_output_dir": "/evaluation/run/validation"}})
    assert paths.validation_output_dir(config) == Path("/evaluation/run/validation")
    config.experiment.output_dir = "/evaluation/replay/wave-0"
    config.experiment.validation_output_dir = "/evaluation/replay/wave-0"
    assert paths.validation_output_dir(config) == Path("/evaluation/replay/wave-0")


def test_old_sweep_results_remain_readable_after_relocation(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "EVALUATION_ROOT", tmp_path / "output/evaluation")
    run = tmp_path / "output/sweep/arm"
    run.mkdir(parents=True)
    old = run / "validation_metrics_step_955.json"
    old.write_text('{"metrics": {"val/loss": 1.2}}')
    assert paths.training_validation_file(run, old.name) == old
    new = paths.training_validation_root(run) / old.name
    new.parent.mkdir(parents=True)
    old.rename(new)
    assert paths.training_validation_file(run, old.name) == new
