import ast
from pathlib import Path
import re

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLING_STEPS = 10


def test_all_selfless_configs_default_to_ten_sampling_steps():
    config_paths = sorted((REPO_ROOT / "configs/selfless").glob("*.yaml"))
    assert config_paths
    for path in config_paths:
        config = OmegaConf.load(path)
        assert int(config.model.image_flow_num_sampling_steps) == 10, path
        assert int(config.evaluation.sampling_steps) == 10, path


def test_all_python_sampling_step_literals_are_ten():
    roots = ("models", "pretrain", "scripts", "tests", "utils")
    for root in roots:
        for path in (REPO_ROOT / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    value = node.value
                    for target in targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and target.attr == "image_flow_num_sampling_steps"
                            and isinstance(value, ast.Constant)
                        ):
                            assert int(value.value) == DEFAULT_SAMPLING_STEPS, path
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if (
                        keyword.arg == "num_sampling_steps"
                        and isinstance(keyword.value, ast.Constant)
                    ):
                        assert (
                            int(keyword.value.value) == DEFAULT_SAMPLING_STEPS
                        ), path
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    continue
                if node.args[0].value != "--sampling_steps":
                    continue
                defaults = [
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "default"
                ]
                assert len(defaults) == 1, path
                assert isinstance(defaults[0], ast.Constant), path
                assert int(defaults[0].value) == DEFAULT_SAMPLING_STEPS, path


def test_all_shell_sampling_step_literals_are_ten():
    for path in (REPO_ROOT / "script").rglob("*.sh"):
        source = path.read_text(encoding="utf-8")
        for value in re.findall(r"--sampling_steps\s+(\d+)", source):
            assert int(value) == DEFAULT_SAMPLING_STEPS, path
        for value in re.findall(r"SAMPLING_STEPS:-([0-9]+)", source):
            assert int(value) == DEFAULT_SAMPLING_STEPS, path


def test_all_model_fallbacks_default_to_ten_sampling_steps():
    model_paths = (
        "models/modeling_model/modeling_selfless_flow.py",
        "models/modeling_model/modeling_positionwise_flow.py",
    )
    pattern = re.compile(
        r'getattr\(\s*config,\s*"image_flow_num_sampling_steps",\s*"10"\s*,?\s*\)'
    )
    for relative_path in model_paths:
        path = REPO_ROOT / relative_path
        assert pattern.search(path.read_text(encoding="utf-8")), path
