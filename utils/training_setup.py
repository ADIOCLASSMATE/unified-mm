"""Optimizer and schedule construction for Selfless-Flow training."""
from torch.optim import AdamW

from utils.selfless_flow_optimizer import (
    learning_rate_for_parameter,
    optimizer_parameter_role,
    weight_decay_for_parameter,
)
from utils.wsd_schedule import get_wsd_schedule


def build_optimizer_and_scheduler(model, config, logger):
    optimizer_config = config.optimizer.params

    # Use lower LR for the pretrained backbone and higher LR for continuous-
    # image modules. Decay flow-head matrix weights while keeping biases,
    # normalization parameters, and the small input projectors decay-free.
    base_lr = float(optimizer_config.learning_rate)
    backbone_lr = float(optimizer_config.get("backbone_learning_rate", base_lr))
    flow_lr = float(optimizer_config.get("flow_learning_rate", base_lr))
    projector_lr = float(optimizer_config.get("projector_learning_rate", flow_lr))
    special_token_lr = float(optimizer_config.get("special_token_learning_rate", projector_lr))
    global_weight_decay = float(optimizer_config.weight_decay)
    flow_weight_decay = float(
        optimizer_config.get("flow_weight_decay", global_weight_decay)
    )
    grouped = {}
    optimizer_role_numel = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        weight_decay = weight_decay_for_parameter(
            name,
            global_weight_decay=global_weight_decay,
            flow_weight_decay=flow_weight_decay,
        )
        role = optimizer_parameter_role(name)
        optimizer_role_numel[role] = optimizer_role_numel.get(role, 0) + int(
            param.numel()
        )
        learning_rate = learning_rate_for_parameter(
            name,
            backbone_lr=backbone_lr,
            flow_lr=flow_lr,
            projector_lr=projector_lr,
            special_token_lr=special_token_lr,
        )
        key = (learning_rate, weight_decay)
        grouped.setdefault(key, []).append(param)

    optimizer_grouped_parameters = [
        {"params": params, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), params in grouped.items()
    ]
    logger.info(
        "Optimizer LRs: "
        f"backbone={backbone_lr:g}, image_token_embedder/image_flow_condition_proj={projector_lr:g}, "
        f"image_flow_head={flow_lr:g}; "
        f"special_tokens={special_token_lr:g}; "
        f"weight_decay={global_weight_decay:g}, "
        f"flow_weight_decay={flow_weight_decay:g}"
    )
    tied_embedding = model.lm_head.weight is model.model.embed_tokens.weight
    if not tied_embedding:
        raise RuntimeError(
            "Joint training expects lm_head.weight and embed_tokens.weight to be tied"
        )
    logger.info(
        "Optimizer parameter coverage: "
        + ", ".join(
            f"{role}={optimizer_role_numel.get(role, 0):,}"
            for role in (
                "backbone",
                "tied_lm_head_embedding",
                "image_projector",
                "flow_head",
            )
        )
        + "; lm_head/embed_tokens tied=true; special_token_learning_rate "
        "applies to the complete tied matrix"
    )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        num_decay_steps=config.lr_scheduler.params.decay_steps,
        num_training_steps=config.training.max_train_steps,
        min_lr_ratio=config.lr_scheduler.params.min_lr_scale
    )
    logger.info(
        "WSD schedule: "
        f"warmup_steps={int(config.lr_scheduler.params.warmup_steps)}, "
        f"stable_steps={int(config.training.max_train_steps) - int(config.lr_scheduler.params.warmup_steps) - int(config.lr_scheduler.params.decay_steps)}, "
        f"warmdown_steps={int(config.lr_scheduler.params.decay_steps)}, "
        f"min_lr_scale={float(config.lr_scheduler.params.min_lr_scale):g}"
    )
    return optimizer, lr_scheduler
