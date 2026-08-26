import importlib.util
import sys
import types

import pytest

# The model module imports torch_npu for its runtime kernels.  Constructor-only
# tests do not execute those kernels, and the login node has no CANN runtime.
sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))
_real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: (
    None if name == "torch_npu" else _real_find_spec(name, *args, **kwargs)
)
try:
    from transformers import Qwen3Config

    from models.modeling_model.modeling_positionwise_flow import (
        PositionwiseFlowQwen3ForCausalLM,
    )
finally:
    importlib.util.find_spec = _real_find_spec


def _tiny_config(*, lambda_text=0.0, lambda_image=1.0):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 2
    config.image_flow_num_sampling_steps = "10"
    config.lambda_text = lambda_text
    config.lambda_image = lambda_image
    return config


def test_positionwise_model_initializes_inherited_loss_weights():
    model = PositionwiseFlowQwen3ForCausalLM(
        _tiny_config(lambda_text=0.0, lambda_image=1.0)
    )

    assert model.lambda_text == pytest.approx(0.0)
    assert model.lambda_image == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("lambda_text", "lambda_image", "message"),
    ((-0.1, 1.0, "non-negative"), (0.0, 0.0, "cannot both be zero")),
)
def test_positionwise_model_rejects_invalid_loss_weights(
    lambda_text,
    lambda_image,
    message,
):
    with pytest.raises(ValueError, match=message):
        PositionwiseFlowQwen3ForCausalLM(
            _tiny_config(
                lambda_text=lambda_text,
                lambda_image=lambda_image,
            )
        )
