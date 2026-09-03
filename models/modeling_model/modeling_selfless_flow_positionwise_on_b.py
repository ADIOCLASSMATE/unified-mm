"""Ablation F: parameter-matched position-wise flow head on baseline B.

F preserves B's XLNet-style content diagonal, query stream, backbone,
training data, source schedule, loss weights, RF estimator, and optimizer
contract.  Its only modeling change is replacing the contextual flow head with
an AdaLN MLP independently applied to every image latent token.

The formal 0.6B recipe uses width 1936.  At depth eight this gives 163,828,208
flow-head parameters versus 164,072,976 for B's width-1280 contextual head, a
0.149% difference while keeping the matrix width aligned to 16 for Ascend.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .image_flow_loss_positionwise import PositionwiseFlowLoss
from .modeling_selfless_flow import (
    Qwen3ForCausalLM as ContextualQwen3ForCausalLM,
)
from .modeling_selfless_flow import Qwen3Model, Qwen3PreTrainedModel
from .modeling_selfless_flow_positionwise_on_b_generation import (
    PositionwiseFlowOnBGenerationMixin,
)


POSITIONWISE_ON_B_ARCHITECTURE = "positionwise_flow_head_on_b"
POSITIONWISE_ON_B_ATTENTION_CONTRACT = "xlnet_content_diagonal"
POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT = "not_applicable"


class SelflessFlowPositionwiseOnBConfig(Qwen3Config):
    """Checkpoint identity for the isolated F implementation."""

    model_type = "selfless_flow_positionwise_on_b"


AutoConfig.register(
    SelflessFlowPositionwiseOnBConfig.model_type,
    SelflessFlowPositionwiseOnBConfig,
)


def contextual_flow_head_parameter_count(
    *,
    latent_dim: int,
    condition_dim: int,
    width: int,
    depth: int,
) -> int:
    """Exact parameter count of the repository contextual flow head."""

    latent_dim = int(latent_dim)
    condition_dim = int(condition_dim)
    width = int(width)
    depth = int(depth)
    # Shared input/time/condition/final modules.
    shared = (
        3 * width * width
        + width * (condition_dim + 2 * latent_dim + 262)
        + latent_dim
    )
    # Four attention projections, ratio-1 MLP, three affine norms, and 6-way
    # AdaLN modulation in each ContextualFlowBlock.
    per_block = 12 * width * width + 18 * width
    return int(shared + depth * per_block)


class PositionwiseFlowOnBQwen3ForCausalLM(
    PositionwiseFlowOnBGenerationMixin,
    ContextualQwen3ForCausalLM,
):
    """Baseline-B backbone plus an isolated, parameter-matched flow MLP."""

    config_class = SelflessFlowPositionwiseOnBConfig
    model_type = SelflessFlowPositionwiseOnBConfig.model_type
    architecture_variant = POSITIONWISE_ON_B_ARCHITECTURE

    def __init__(self, config):
        contract = str(
            getattr(
                config,
                "dual_stream_attention_contract",
                POSITIONWISE_ON_B_ATTENTION_CONTRACT,
            )
        ).strip().lower()
        if contract != POSITIONWISE_ON_B_ATTENTION_CONTRACT:
            raise ValueError(
                "ablation F must be built on baseline B's "
                f"{POSITIONWISE_ON_B_ATTENTION_CONTRACT!r} contract, got "
                f"{contract!r}"
            )
        flow_head_contract = str(
            getattr(
                config,
                "flow_head_attention_contract",
                POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT,
            )
        ).strip().lower()
        if flow_head_contract != POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT:
            raise ValueError(
                "ablation F has no cross-token flow-head attention and requires "
                "flow_head_attention_contract='not_applicable', got "
                f"{flow_head_contract!r}"
            )

        # Construct the shared backbone directly so a 164M contextual head is
        # never allocated and discarded at startup.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = int(config.vocab_size)
        self.image_latent_dim = int(getattr(config, "image_latent_dim", 4))
        self.image_flow_batch_mul = int(
            getattr(config, "image_flow_batch_mul", 1)
        )
        self.training_objective = str(
            getattr(config, "training_objective", "selfless_dual_stream")
        ).strip().lower()
        if self.training_objective != "selfless_dual_stream":
            raise ValueError(
                "ablation F requires training_objective="
                f"'selfless_dual_stream', got {self.training_objective!r}"
            )
        self.lambda_text = float(getattr(config, "lambda_text", 0.0))
        self.lambda_image = float(getattr(config, "lambda_image", 1.0))
        if self.lambda_text < 0.0 or self.lambda_image < 0.0:
            raise ValueError("lambda_text and lambda_image must be non-negative")
        if self.lambda_text == 0.0 and self.lambda_image == 0.0:
            raise ValueError("lambda_text and lambda_image cannot both be zero")

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.image_flow_condition_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )
        self.image_flow_head = PositionwiseFlowLoss(
            target_channels=self.image_latent_dim,
            z_channels=config.hidden_size,
            width=int(getattr(config, "image_flow_width", 1936)),
            depth=int(getattr(config, "image_flow_depth", 8)),
            num_sampling_steps=getattr(
                config,
                "image_flow_num_sampling_steps",
                "10",
            ),
            grad_checkpointing=bool(
                getattr(config, "image_flow_grad_checkpointing", False)
            ),
            time_scale=float(
                getattr(config, "image_flow_time_scale", 1000.0)
            ),
            time_sampling=getattr(
                config,
                "image_flow_time_sampling",
                "logit_normal",
            ),
            logit_mean=float(getattr(config, "image_flow_logit_mean", 0.0)),
            logit_std=float(getattr(config, "image_flow_logit_std", 1.0)),
            time_eps=float(getattr(config, "image_flow_time_eps", 1.0e-4)),
            uniform_mix=float(
                getattr(config, "image_flow_time_uniform_mix", 0.1)
            ),
            solver=getattr(config, "image_flow_solver", "heun"),
            image_tokens_per_img=int(
                getattr(config, "image_tokens_per_img", 256)
            ),
        )

        self.positionwise_flow_head_parameter_count = sum(
            parameter.numel() for parameter in self.image_flow_head.parameters()
        )
        reference_width = getattr(
            config,
            "positionwise_reference_flow_width",
            None,
        )
        reference_depth = getattr(
            config,
            "positionwise_reference_flow_depth",
            None,
        )
        if reference_width is None or reference_depth is None:
            self.reference_contextual_flow_head_parameter_count = int(
                self.positionwise_flow_head_parameter_count
            )
        else:
            self.reference_contextual_flow_head_parameter_count = (
                contextual_flow_head_parameter_count(
                    latent_dim=self.image_latent_dim,
                    condition_dim=int(config.hidden_size),
                    width=int(reference_width),
                    depth=int(reference_depth),
                )
            )
            relative_error = abs(
                self.positionwise_flow_head_parameter_count
                - self.reference_contextual_flow_head_parameter_count
            ) / float(self.reference_contextual_flow_head_parameter_count)
            max_error = float(
                getattr(
                    config,
                    "positionwise_max_parameter_relative_error",
                    0.005,
                )
            )
            if relative_error > max_error:
                raise ValueError(
                    "ablation F flow-head parameter mismatch exceeds the "
                    f"configured tolerance: relative_error={relative_error:.6f}, "
                    f"max={max_error:.6f}"
                )

        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_modules()


__all__ = [
    "POSITIONWISE_ON_B_ARCHITECTURE",
    "POSITIONWISE_ON_B_ATTENTION_CONTRACT",
    "POSITIONWISE_ON_B_FLOW_HEAD_ATTENTION_CONTRACT",
    "PositionwiseFlowOnBQwen3ForCausalLM",
    "SelflessFlowPositionwiseOnBConfig",
    "contextual_flow_head_parameter_count",
]
