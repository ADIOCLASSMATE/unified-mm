import copy
import json

import pytest

from scripts.build_evaluation_report import matrix_sweep_data
from scripts.sweep_unified_t2i_sampling import advance, evaluator_command
from utils.image_order_strategies import checkpoint_generation_contract


def test_model_owned_checkpoint_and_config_are_used_in_each_command():
    protocol = {"python": "/python", "real_stats": "/reference", "inception_weights": "/inception",
        "models": {"a": {"config": "/a.yaml", "model_source": "/a/ema"},
                   "f": {"config": "/f.yaml", "model_source": "/f/ema"}}}
    for mid in protocol["models"]:
        argv = evaluator_command(protocol, {"model": mid, "strategy": "confidence_stability", "cfg": 2., "steps": 10}, "/out")
        assert argv[argv.index("--model_source") + 1] == f"/{mid}/ema"
        assert argv[argv.index("--config") + 1] == f"/{mid}.yaml"
        assert argv[argv.index("--strategies") + 1] == "confidence_stability"


def test_generation_contract_keeps_legacy_dynamic_and_positionwise_semantics():
    legacy = checkpoint_generation_contract({"architecture_variant": "selfless_contextual"})
    assert legacy["flow_condition"] == "backbone_xt_shared_query_content"
    dynamic = checkpoint_generation_contract({"architecture_variant": "dynamic_xt",
        "dynamic_xt_flow_condition_contract": "backbone_xt_query_backbone_x0_content"})
    assert dynamic["flow_condition"] == "backbone_xt_query_backbone_x0_content"
    positionwise = checkpoint_generation_contract({"architecture_variant": "positionwise_flow_head_on_b"})
    assert positionwise["flow_condition"] == positionwise["flow_attention"] == "not_applicable"


def test_fixed_matrix_never_creates_a_cfg_or_heun_search():
    state = {"phase": "matrix", "status": "running", "tasks": [{"status": "done"}, {"status": "running"}]}
    advance(state, {})
    assert state["status"] == "running"
    state["tasks"][1]["status"] = "done"
    advance(state, {})
    assert state["phase"] == "matrix" and state["status"] == "complete"
    assert len(state["tasks"]) == 2


def test_matrix_report_rejects_missing_duplicate_and_wrong_checkpoint_arms(tmp_path):
    directory = tmp_path / "matrix"
    directory.mkdir()
    mids = ["b_x0", "e_on_b"]
    models = [{"id": mid, "checkpoint": f"/weights/{mid}/ema", "source": {"global_step": 95415},
               "label": mid, "metrics": {}} for mid in mids]
    protocol = {"schema": "unified_t2i_ablation_matrix_v1", "cfg_fixed": 2, "heun_fixed": 10,
        "models": {m["id"]: {"model_source": m["checkpoint"], "checkpoint_step": 95415, "label": m["label"],
            "native_strategy": "spatial_halton" if m["id"] == "b_x0" else "sequential"} for m in models}}
    pairs = [(m, s) for m in mids for s in ["spatial_halton", "confidence_stability"]] + [("e_on_b", "sequential")]
    state = {"phase": "matrix", "status": "running", "tasks": [{"id": f"{m}-{s}", "model": m, "strategy": s,
        "cfg": 2, "steps": 10, "status": "pending"} for m, s in pairs]}
    selection = {"models": dict.fromkeys(mids, {}), "matrix_sweeps": [{"root": "matrix"}]}

    def render(current_state=state, current_protocol=protocol):
        (directory / "protocol.json").write_text(json.dumps(current_protocol))
        (directory / "state.json").write_text(json.dumps(current_state))
        return matrix_sweep_data(tmp_path, selection, models)

    result = render()[0]
    assert result["completed"] == 0 and result["conclusion"] is None
    selection["models"]["s2_single"] = {}
    assert render()[0]["total"] == 5  # New evaluations do not change a frozen study's coverage.
    del selection["models"]["b_x0"]
    with pytest.raises(ValueError, match="unselected"):
        render()
    selection["models"]["b_x0"] = {}
    for mutation in [state["tasks"][:-1], [*state["tasks"][:-1], state["tasks"][0]]]:
        with pytest.raises(ValueError, match="coverage"):
            render({**state, "tasks": mutation})
    wrong = copy.deepcopy(protocol)
    wrong["models"]["b_x0"]["model_source"] = "/weights/b-flowdiag/ema"
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        render(current_protocol=wrong)
    with pytest.raises(ValueError, match="unfinished"):
        render({**state, "status": "complete"})
    expanded_protocol = {**protocol, "matrix_strategies": ["spatial_halton", "confidence_stability", "random"]}
    expanded_state = {**state, "tasks": [*state["tasks"], *[
        {"id": f"{mid}-random", "model": mid, "strategy": "random", "cfg": 2, "steps": 10, "status": "pending"}
        for mid in mids]]}
    assert render(expanded_state, expanded_protocol)[0]["total"] == 7
    with pytest.raises(ValueError, match="coverage"):
        render(state, expanded_protocol)
