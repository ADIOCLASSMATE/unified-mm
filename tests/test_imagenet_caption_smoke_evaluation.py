import argparse
import json

from scripts.evaluate_imagenet_caption_smoke import (
    RuntimeLog,
    TextProgress,
    log_progress,
)


def test_runtime_log_flushes_sampled_progress_and_terminal_events(tmp_path):
    log_path = tmp_path / "evaluation.runtime.jsonl"
    args = argparse.Namespace(log_every=2)

    with RuntimeLog(log_path, "clip") as runtime_log:
        args.runtime_log = runtime_log
        for completed in range(1, 6):
            log_progress(
                args,
                "clip_images",
                completed,
                5,
                img_id=completed * 10,
            )
        runtime_log.emit("run_completed", exit_code=0)

    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "run_started",
        "progress",
        "progress",
        "progress",
        "progress",
        "run_completed",
    ]
    progress = [row for row in rows if row["event"] == "progress"]
    assert [row["completed"] for row in progress] == [1, 2, 4, 5]
    assert all(row["stage"] == "clip_images" for row in progress)
    assert len({row["run_id"] for row in rows}) == 1


def test_dependency_free_progress_bar_reaches_total(capsys):
    values = list(
        TextProgress(
            range(3),
            desc="CLIP evaluation",
            unit="image",
        )
    )

    assert values == [0, 1, 2]
    stderr = capsys.readouterr().err
    assert "CLIP evaluation" in stderr
    assert "3/3 image" in stderr
