from pathlib import Path

import pytest

from scripts.validate_ascend_imagenet100_final_run import ordered_rng_state_paths


def test_rng_shards_are_ordered_by_numeric_rank(tmp_path: Path) -> None:
    for rank in range(16):
        (tmp_path / f"random_states_{rank}.pkl").touch()

    paths = ordered_rng_state_paths(tmp_path)

    assert [path.name for path in paths] == [
        f"random_states_{rank}.pkl" for rank in range(16)
    ]


def test_rng_shards_must_cover_every_rank(tmp_path: Path) -> None:
    for rank in range(16):
        if rank != 7:
            (tmp_path / f"random_states_{rank}.pkl").touch()

    with pytest.raises(RuntimeError, match="non-contiguous RNG shard ranks"):
        ordered_rng_state_paths(tmp_path)
