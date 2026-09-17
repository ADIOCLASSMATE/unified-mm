"""31,800-update ablations derived from their frozen 100B recipes."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架"
MAX_STEPS = 31800
WARMUP_STEPS = 199
DECAY_STEPS = 7950
VAL_EVERY = 3180
EMA_DECAY = 0.9997
RUN_REVISION = "r2"
ARMS = {
    "y": ("unified_y", "y", "Y"),
    "s2-single": ("unified_s2_single", "s2_single", "S2-single"),
    "s2-text2stream": ("unified_s2_single_text_two_stream", "s2_single_text_two_stream", "S2-single + text two-stream"),
    "z": ("unified_z", "z", "Z"),
    "z-b": ("unified_z_b_head", "z_b_head", "Z + B head"),
    "b-s2": ("unified_b_s2_modulation", "b_s2_modulation", "B + S2 modulation"),
}


def config_path(arm, *, base=False):
    stem = ARMS[arm][0]
    return f"configs/selfless/{stem}_{'100b' if base else '33b'}_ascend64.yaml"


def expected_config(arm):
    if arm == "y":
        from utils.y_protocol import expected_config as y_config
        return y_config()
    config = OmegaConf.load(ROOT / config_path(arm, base=True))
    if arm in ("s2-single", "s2-text2stream"):
        from utils.showo2_unified_protocol import validate_s2_config
        validate_s2_config(config)
    elif arm == "b-s2":
        from utils.b_s2_modulation_protocol import validate_b_s2_modulation_config
        validate_b_s2_modulation_config(config)
    else:
        from utils.joint_experiments import joint_experiment_protocol
        joint_experiment_protocol(arm).validate_joint_dit_config(config)
    run = str(config.experiment.project).replace("-100b-", "-33b-").rsplit("-", 1)[0] + f"-{RUN_REVISION}"
    config.experiment.project = run
    config.experiment.name = run
    config.experiment.identity = dict(id=f"{ARMS[arm][1]}_33b_{RUN_REVISION}", label=f"{ARMS[arm][2]} · 33B",
                                     group=config.experiment.identity.group, purpose="formal")
    config.experiment.resume_from_checkpoint = "none"
    config.experiment.val_every = VAL_EVERY
    config.experiment.validation_generation = dict(enabled=True, weights="raw", samples=16, seed=42,
        prompt_file="configs/protocols/unified_qualitative_prompts_v1.json",
        cfg=3.5, steps=10, solver="heun", vae_module_root="public/code/mar",
        vae_path="public/vae/mar-kl16/kl16.ckpt", vae_scaling_factor=0.2325)
    config.training.max_train_steps = MAX_STEPS
    config.training.stop_after_steps = MAX_STEPS
    config.training.target_text_tokens = MAX_STEPS * int(config.training.nominal_text_targets_per_step_64npu)
    config.lr_scheduler.params.warmup_steps = WARMUP_STEPS
    config.lr_scheduler.params.decay_steps = DECAY_STEPS
    # Preserve N * (1 - decay), and therefore the EMA timescale relative to
    # the proportionally shortened WSD decay phase (rounded to four digits).
    config.training.ema_decay = EMA_DECAY
    config.evaluation.checkpoint = f"output/{run}/hf_model-final-ema"
    return config


def validate_short_config(arm, config):
    expected = expected_config(arm)
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(expected, resolve=True):
        raise ValueError(f"{arm}: short ablation differs from the budget/EMA/validation recipe")
    return dict(schema="short_ablation_31800_v1", arm=arm, project=PROJECT,
                run_project=str(config.experiment.project), world_size=64,
                optimizer_steps=MAX_STEPS, warmup_steps=WARMUP_STEPS, decay_steps=DECAY_STEPS,
                decay_start=MAX_STEPS - DECAY_STEPS, val_every=VAL_EVERY,
                validation_steps=list(range(VAL_EVERY, MAX_STEPS + 1, VAL_EVERY)),
                nominal_text_targets=int(config.training.target_text_tokens),
                ema_decay=float(config.training.ema_decay),
                validation_generation=OmegaConf.to_container(config.experiment.validation_generation, resolve=True))


def asset_preflight(arm, python, *, assets):
    # Model/data are exact matches to the checked 100B recipe. Its asset checks
    # are reused after validating the actual short config in the launcher.
    if arm in ("s2-single", "s2-text2stream"):
        command = [python, "scripts/validate_showo2_unified.py", "--config", config_path(arm, base=True)]
    elif arm == "b-s2":
        command = [python, "scripts/validate_b_s2_modulation.py"]
    else:
        command = [python, "scripts/validate_z.py", "--experiment", arm]
    command += ["--require-npu-count", "16"]
    if assets:
        command.append("--assets")
    return command
