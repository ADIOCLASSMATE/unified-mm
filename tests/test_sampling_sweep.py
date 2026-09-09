"""CPU checks for expensive sweep scheduling and fail-closed result reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import signal

from scripts.sweep_unified_t2i_sampling import advance, task, validate_metrics, worker
from scripts.submit_unified_t2i_sweep import expand_pending_boundary, missing_npu_node
from utils.evaluation_image_subset import evenly_spaced_image_indices


PROTOCOL = {"cfg_steps": 10, "cfg_limit": 12.0, "heun_steps": [5, 10, 20, 50, 100]}


def completed(cfg, fid, inception):
    arm = task(cfg, 10, "cfg")
    arm.update(status="done", result={"fid": fid, "is": inception})
    return arm


class SamplingSweepTests(unittest.TestCase):
    def test_parallel_boundary_extension_keeps_spacing_and_does_not_repeat(self):
        state = {"status": "running", "phase": "cfg", "tasks": [completed(0.5, 3, 100), completed(1, 4, 200)]}
        advance(state, PROTOCOL)
        protocol = {**PROTOCOL, "cfg_extension_points": 4}
        self.assertTrue(expand_pending_boundary(state, protocol))
        self.assertEqual(sorted(a["cfg"] for a in state["tasks"] if a["status"] == "pending"),
                         [0, 1.5, 2, 2.5, 3])
        self.assertFalse(expand_pending_boundary(state, protocol))
        self.assertEqual(state["phase"], "cfg")

    def test_node_exclusion_requires_missing_mounts_and_actual_placement(self):
        node = "infra-gpu-npu-259.host.shzhisuan.com"
        events = [{"reason": "FailedScheduling", "message": f"Can possibly be assigned to {node}"}]
        empty = {"devices": [], "driver_libraries": []}
        self.assertIsNone(missing_npu_node(empty, events))
        events.append({"reason": "Scheduled", "message": f"Successfully assigned namespace/pod to {node}"})
        self.assertEqual(missing_npu_node(empty, events), node)
        self.assertIsNone(missing_npu_node({**empty, "devices": ["/dev/davinci0"]}, events))
        self.assertIsNone(missing_npu_node({}, events))

    def test_one_failed_job_does_not_abort_independent_sweep_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol.json").write_text(json.dumps({"source_repo": temporary, "arm_timeout_hours": 12}))
            state = {"status": "running", "phase": "cfg", "tasks": [task(2, 10, "cfg"), task(3, 10, "cfg")]}
            (root / "state.json").write_text(json.dumps(state))
            process = Mock(returncode=1)
            process.poll.return_value = 1
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                with patch("scripts.sweep_unified_t2i_sampling.evaluator_command", return_value=["false"]), \
                     patch("scripts.sweep_unified_t2i_sampling.subprocess.Popen", return_value=process):
                    with self.assertRaisesRegex(ValueError, "evaluator exited 1"):
                        worker(root, task_id=state["tasks"][0]["id"])
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            actual = json.loads((root / "state.json").read_text())
            self.assertEqual(actual["status"], "running")
            self.assertEqual([arm["status"] for arm in actual["tasks"]], ["failed", "pending"])

    def test_saved_samples_cover_full_dataset_and_all_rank_partitions(self):
        indices = evenly_spaced_image_indices(50000, 64)
        self.assertEqual(len(set(indices)), 64)
        self.assertEqual((indices[0], indices[-1]), (0, 49999))
        self.assertEqual(set(i % 16 for i in indices), set(range(16)))
        self.assertEqual(evenly_spaced_image_indices(32, 64), list(range(32)))

    def test_separate_objective_winners_get_all_requested_steps(self):
        state = {"status": "running", "phase": "cfg", "tasks": [
            completed(1, 9, 100), completed(1.5, 3, 200),
            completed(2, 4, 240), completed(2.5, 6, 220)]}
        advance(state, PROTOCOL)
        self.assertEqual(state["phase"], "heun")
        self.assertEqual(state["cfg_selection"]["best_fid"]["cfg"], 1.5)
        self.assertEqual(state["cfg_selection"]["best_is"]["cfg"], 2)
        for cfg in (1.5, 2):
            self.assertEqual({a["steps"] for a in state["tasks"] if a["cfg"] == cfg}, {5, 10, 20, 50, 100})
        self.assertEqual(len(state["tasks"]), 12)  # Ten steps reused for both CFGs.
        for arm in state["tasks"]:
            if arm["status"] == "pending":
                arm.update(status="done", result={"fid": 2 + arm["cfg"], "is": 220 + arm["steps"]})
        advance(state, PROTOCOL)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["heun_selection"]["best_is"]["best_is"]["steps"], 100)

    def test_order_phase_waits_for_every_arm_and_selects_actual_metric_winners(self):
        first = {**completed(2, 4.2, 222), "id": "halton", "phase": "order", "strategy": "spatial_halton"}
        second = {**completed(2, 4.0, 220), "id": "confidence", "phase": "order", "strategy": "confidence_cfg", "status": "running"}
        state = {"status": "running", "phase": "order", "tasks": [first, second]}
        advance(state, PROTOCOL)
        self.assertNotIn("order_selection", state)
        second["status"] = "done"
        advance(state, PROTOCOL)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["order_selection"]["best_fid"]["id"], "confidence")
        self.assertEqual(state["order_selection"]["best_is"]["id"], "halton")
        self.assertEqual(len(state["tasks"]), 2)

    def test_no_step_selection_before_every_cfg_result(self):
        state = {"status": "running", "phase": "cfg", "tasks": [completed(1, 4, 150), task(1.5, 10, "cfg")]}
        before = copy.deepcopy(state)
        advance(state, PROTOCOL)
        self.assertEqual(state, before)

    def test_extend_both_winning_boundaries(self):
        state = {"status": "running", "phase": "cfg", "tasks": [completed(1, 3, 100), completed(1.5, 4, 150)]}
        advance(state, PROTOCOL)
        self.assertEqual(state["phase"], "cfg")
        self.assertEqual([a["cfg"] for a in state["tasks"] if a["status"] == "pending"], [0.5, 2.0])

    def test_boundary_limit_is_not_reported_as_optimum(self):
        state = {"status": "running", "phase": "cfg", "tasks": [completed(11.5, 4, 100), completed(12, 3, 150)]}
        advance(state, PROTOCOL)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("cfg_selection", state)

    def test_one_joint_winner_does_not_duplicate_step_runs(self):
        state = {"status": "running", "phase": "cfg", "tasks": [
            completed(0, 8, 100), completed(0.5, 3, 200), completed(1, 5, 150)]}
        advance(state, PROTOCOL)
        self.assertEqual(len(state["tasks"]), 7)

    def test_existing_metric_from_another_cfg_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps({"schema": "selfless_imagenet_val_t2i_fid_is_v2",
                "project_formal_protocol": True, "runtime_hashing_enabled": False,
                "samples_requested": 50000, "samples_evaluated": 50000,
                "split": "val", "seed": 42, "batch_size": 4096, "cfg": 3.5}))
            with self.assertRaisesRegex(ValueError, "cfg != 4.0"):
                validate_metrics(path, PROTOCOL, task(4.0, 10, "cfg"))


if __name__ == "__main__":
    unittest.main()
