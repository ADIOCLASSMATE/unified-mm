from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_selfless_flow_selected_eval_defaults() -> None:
    config = OmegaConf.load(REPO_ROOT / "configs/ablation/imagenet_flow_100c_80ep.yaml")
    assert float(config.experiment.validation_flow_cfg) == 3.5
    assert config.evaluation.checkpoint.endswith("hf_model-final-ema")
    assert config.evaluation.model_dtype == "bf16"
    assert float(config.evaluation.cfg) == 3.5
    assert int(config.evaluation.parallel_rate) == 1
    assert int(config.evaluation.sampling_steps) == 100
    assert config.evaluation.flow_solver == "heun"
    assert config.evaluation.strategies == "spatial_halton"

    launcher = _read("script/ablation/evaluate_imagenet_flow_100c.sh")
    assert '${CFG:-3.5}' in launcher
    assert '${MODEL_DTYPE:-bf16}' in launcher
    assert '${PARALLEL_RATE:-1}' in launcher
    assert "hf_model-final-ema" in launcher
    assert "fid_is_selected_cfg3p5_ema" in launcher


def test_showo_selected_eval_defaults_and_cfg_mapping() -> None:
    config = OmegaConf.load(REPO_ROOT / "configs/ablation/qwen_showo_vq_100c_80ep.yaml")
    guidance_scale = float(config.evaluation.guidance_scale)
    common_cfg_scale = float(config.evaluation.common_cfg_scale)
    assert guidance_scale == 11.75
    assert common_cfg_scale == 12.75
    assert common_cfg_scale == 1.0 + guidance_scale

    launcher = _read("script/ablation/evaluate_qwen_showo_vq_100c.sh")
    assert '${GUIDANCE_SCALE:-11.75}' in launcher
    assert "hf_model-final" in launcher
    assert "fid_is_selected_w12p75_s11p75" in launcher

    evaluator = _read("scripts/evaluate_qwen_showo_fid_is.py")
    assert 'parser.add_argument("--guidance_scale", type=float, default=11.75)' in evaluator
