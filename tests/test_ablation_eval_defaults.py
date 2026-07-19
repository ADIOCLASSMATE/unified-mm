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


def test_token_mlp_ablation_matches_flow_budget_and_changes_only_the_head_architecture() -> None:
    baseline = OmegaConf.load(REPO_ROOT / "configs/ablation/imagenet_flow_100c_80ep.yaml")
    token_mlp = OmegaConf.load(
        REPO_ROOT / "configs/ablation/imagenet_flow_token_mlp_100c_80ep.yaml"
    )

    assert token_mlp.model.image_flow_head_arch == "token_mlp"
    assert bool(token_mlp.model.image_flow_zero_init_gate)
    assert "image_flow_latent_mixer_heads" not in token_mlp.model
    assert "image_flow_latent_mixer_dropout" not in token_mlp.model
    assert "image_flow_latent_mixer_zero_init_gate" not in token_mlp.model

    baseline_model = OmegaConf.to_container(baseline.model, resolve=True)
    token_model = OmegaConf.to_container(token_mlp.model, resolve=True)
    for key, value in baseline_model.items():
        if key.startswith("image_flow_latent_mixer_"):
            continue
        assert token_model[key] == value

    assert OmegaConf.to_container(token_mlp.dataset, resolve=True) == OmegaConf.to_container(
        baseline.dataset, resolve=True
    )
    assert OmegaConf.to_container(token_mlp.optimizer, resolve=True) == OmegaConf.to_container(
        baseline.optimizer, resolve=True
    )
    assert OmegaConf.to_container(token_mlp.lr_scheduler, resolve=True) == OmegaConf.to_container(
        baseline.lr_scheduler, resolve=True
    )
    assert OmegaConf.to_container(token_mlp.training, resolve=True) == OmegaConf.to_container(
        baseline.training, resolve=True
    )

    launcher = _read("script/ablation/evaluate_imagenet_flow_token_mlp_100c.sh")
    assert "imagenet_flow_token_mlp_100c_80ep.yaml" in launcher
    assert "selfless-flow-token-mlp-ablation-imagenet100-80ep" in launcher
    assert 'CFG="3.5"' in launcher


def test_parameter_matched_token_mlp_changes_only_head_capacity() -> None:
    token_mlp = OmegaConf.load(
        REPO_ROOT / "configs/ablation/imagenet_flow_token_mlp_100c_80ep.yaml"
    )
    parameter_matched = OmegaConf.load(
        REPO_ROOT
        / "configs/ablation/imagenet_flow_token_mlp_param_matched_100c_80ep.yaml"
    )

    assert parameter_matched.model.image_flow_head_arch == "token_mlp"
    assert parameter_matched.model.image_flow_mlp_ratio == 4.5
    assert bool(parameter_matched.model.image_flow_zero_init_gate)
    assert "image_flow_latent_mixer_heads" not in parameter_matched.model
    assert "image_flow_latent_mixer_dropout" not in parameter_matched.model
    assert "image_flow_latent_mixer_zero_init_gate" not in parameter_matched.model

    token_model = OmegaConf.to_container(token_mlp.model, resolve=True)
    matched_model = OmegaConf.to_container(parameter_matched.model, resolve=True)
    token_model.pop("image_flow_mlp_ratio")
    matched_model.pop("image_flow_mlp_ratio")
    assert matched_model == token_model

    for section in ("dataset", "optimizer", "lr_scheduler", "training"):
        assert OmegaConf.to_container(
            parameter_matched[section], resolve=True
        ) == OmegaConf.to_container(token_mlp[section], resolve=True)

    launcher = _read(
        "script/ablation/evaluate_imagenet_flow_token_mlp_param_matched_100c.sh"
    )
    assert "imagenet_flow_token_mlp_param_matched_100c_80ep.yaml" in launcher
    assert "selfless-flow-token-mlp-param-matched-ablation-imagenet100-80ep" in launcher
    assert 'CFG="3.5"' in launcher


def test_width1936_token_mlp_changes_only_head_width() -> None:
    token_mlp = OmegaConf.load(
        REPO_ROOT / "configs/ablation/imagenet_flow_token_mlp_100c_80ep.yaml"
    )
    width1936 = OmegaConf.load(
        REPO_ROOT / "configs/ablation/imagenet_flow_token_mlp_width1936_100c_80ep.yaml"
    )

    assert width1936.model.image_flow_head_arch == "token_mlp"
    assert int(width1936.model.image_flow_width) == 1936
    assert int(width1936.model.image_flow_depth) == 8
    assert float(width1936.model.image_flow_mlp_ratio) == 1.0
    assert bool(width1936.model.image_flow_zero_init_gate)
    assert "image_flow_latent_mixer_heads" not in width1936.model
    assert "image_flow_latent_mixer_dropout" not in width1936.model
    assert "image_flow_latent_mixer_zero_init_gate" not in width1936.model

    token_model = OmegaConf.to_container(token_mlp.model, resolve=True)
    width1936_model = OmegaConf.to_container(width1936.model, resolve=True)
    token_model.pop("image_flow_width")
    width1936_model.pop("image_flow_width")
    assert width1936_model == token_model

    for section in ("dataset", "optimizer", "lr_scheduler", "training"):
        assert OmegaConf.to_container(
            width1936[section], resolve=True
        ) == OmegaConf.to_container(token_mlp[section], resolve=True)

    launcher = _read("script/ablation/evaluate_imagenet_flow_token_mlp_width1936_100c.sh")
    assert "imagenet_flow_token_mlp_width1936_100c_80ep.yaml" in launcher
    assert "selfless-flow-token-mlp-width1936-ablation-imagenet100-80ep" in launcher
    assert 'CFG="3.5"' in launcher


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
