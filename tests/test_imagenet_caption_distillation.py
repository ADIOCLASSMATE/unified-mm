import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.distill_imagenet_captions import (
    DISTILL_RECORD_SCHEMA,
    FailureCircuitBreaker,
    MERGED_SCHEMA,
    canonical_json,
    classify_request_error,
    content_text,
    model_request_options,
    open_response_db,
    parse_caption_response,
    resolve_model_concurrency,
    resolve_model_rate_limits,
    response_has_exact_caption_count,
    should_record_circuit_outcome,
    run_coverage,
    run_merge,
    run_validate,
    sha256_bytes,
    sha256_file,
    sha256_text,
    validate_generation_models,
    word_tokens,
)


def test_parse_caption_response_requires_exact_count_without_quality_retries():
    response = json.dumps(
        {
            "captions": [
                {
                    "text": "A blue bird rests quietly on a narrow branch beneath soft daylight."
                },
                {
                    "text": "Green leaves frame a small perched animal against a distant sky."
                },
            ]
        }
    )
    captions = parse_caption_response(
        f"```json\n{response}\n```",
        expected_count=2,
        min_words=8,
        max_words=20,
        max_jaccard=0.8,
    )
    assert [caption["caption_index"] for caption in captions] == [0, 1]
    assert all(caption["word_count"] >= 8 for caption in captions)

    over_target = json.dumps(
        {
            "captions": [
                {
                    "text": "A detailed blue bird remains quietly on a narrow branch while soft daylight reveals layered feathers and green leaves throughout the distant natural background."
                },
                {
                    "text": "From a low viewpoint the small animal occupies one side of the frame as branches cross a softly blurred sky behind its compact silhouette."
                },
            ]
        }
    )
    accepted = parse_caption_response(
        over_target,
        expected_count=2,
        min_words=8,
        max_words=10,
        max_jaccard=0.8,
    )
    assert all(caption["word_count"] > 10 for caption in accepted)

    short_or_duplicate = json.dumps(
        {"captions": [{"text": "A tiny bird."}, {"text": "A distant tree."}]}
    )
    accepted = parse_caption_response(
        short_or_duplicate,
        expected_count=2,
        min_words=8,
        max_words=20,
        max_jaccard=0.0,
    )
    assert [caption["word_count"] for caption in accepted] == [3, 3]

    with pytest.raises(ValueError, match="expected exactly 2 captions"):
        parse_caption_response(
            json.dumps({"captions": [{"text": "One usable caption."}]}),
            expected_count=2,
            min_words=8,
            max_words=20,
            max_jaccard=0.0,
        )

    duplicate = json.dumps({"captions": [{"text": captions[0]["text"]}] * 2})
    assert (
        len(
            parse_caption_response(
                duplicate,
                expected_count=2,
                min_words=8,
                max_words=20,
                max_jaccard=0.0,
            )
        )
        == 2
    )

    empty = json.dumps({"captions": [{"text": ""}, {"text": "A tree."}]})
    with pytest.raises(ValueError, match="empty"):
        parse_caption_response(
            empty,
            expected_count=2,
            min_words=8,
            max_words=20,
            max_jaccard=0.8,
        )


def test_known_text_models_are_rejected_as_vision_teachers():
    for model in (
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "glm-5.2",
        "MiniMax-M2.5",
        "MiniMax/MiniMax-M2.7",
        "doubao-embedding-vision-251215",
        "doubao-seedance-1-0-pro-fast-251015",
        "doubao-seedream-5-0-260128",
    ):
        with pytest.raises(ValueError, match="not a multimodal"):
            validate_generation_models([model], allow_unverified=False)
    for model in ("qwen3.6-flash", "qwen3-vl-embedding", "qwen-image-2.0"):
        with pytest.raises(ValueError, match="API generation is disabled"):
            validate_generation_models([model], allow_unverified=False)
    validate_generation_models(
        [
            "kimi-k2.6",
            "MiniMax-M3",
            "doubao-seed-2-1-turbo-260628",
        ],
        allow_unverified=False,
    )


def test_model_request_options_apply_fast_provider_constraints():
    args = argparse.Namespace(
        temperature=None,
        thinking="auto",
        thinking_budget_tokens=1024,
    )
    assert model_request_options("qwen3.6-flash", args) == {
        "temperature": 0.8,
        "thinking": {"type": "disabled"},
    }
    assert model_request_options("kimi-k2.6", args) == {
        "temperature": 0.6,
        "thinking": {"type": "disabled"},
    }
    assert model_request_options("MiniMax-M3", args) == {"temperature": 0.8}
    assert model_request_options("kimi-k3", args) == {"temperature": 1.0}


def test_model_concurrency_requires_explicit_limits_for_each_model():
    models = ["qwen3.6-flash", "kimi-k2.6", "MiniMax-M3"]
    assert resolve_model_concurrency(None, models) is None
    assert resolve_model_concurrency(
        [
            "qwen3.6-flash=64",
            "kimi-k2.6=48",
            "MiniMax-M3=16",
        ],
        models,
    ) == {
        "qwen3.6-flash": 64,
        "kimi-k2.6": 48,
        "MiniMax-M3": 16,
    }
    with pytest.raises(ValueError, match="missing MiniMax-M3"):
        resolve_model_concurrency(
            ["qwen3.6-flash=64", "kimi-k2.6=48"],
            models,
        )
    with pytest.raises(ValueError, match="unselected model"):
        resolve_model_concurrency(["qwen3.7-flash=8"], models)
    with pytest.raises(ValueError, match="positive"):
        resolve_model_concurrency(
            ["qwen3.6-flash=0", "kimi-k2.6=48", "MiniMax-M3=16"],
            models,
        )


def test_model_rate_limits_are_conservative_and_explicit_overrides_are_complete():
    assert resolve_model_rate_limits(None, ["MiniMax-M3"]) == {"MiniMax-M3": 4.0}
    assert resolve_model_rate_limits(
        ["MiniMax-M3=3.5"], ["MiniMax-M3"]
    ) == {"MiniMax-M3": 3.5}
    with pytest.raises(ValueError, match="missing MiniMax-M3"):
        resolve_model_rate_limits(
            ["kimi-k2.6=0.8"], ["kimi-k2.6", "MiniMax-M3"]
        )


def test_empty_provider_content_and_rolling_failure_circuit_are_safe():
    assert content_text(SimpleNamespace(content=None)) == ""
    circuit = FailureCircuitBreaker(["MiniMax-M3"], window=4, failure_rate=0.5)
    assert circuit.record("MiniMax-M3", success=True) is None
    assert circuit.record("MiniMax-M3", success=False) is None
    assert circuit.record("MiniMax-M3", success=False) is None
    opened = circuit.record("MiniMax-M3", success=True)
    assert opened is not None
    assert opened["failures"] == 2
    assert circuit.is_open("MiniMax-M3")


def test_image_level_failures_do_not_poison_provider_circuit():
    assert should_record_circuit_outcome(
        success=True,
        error_type=None,
        previous_error_type="empty_response",
    )
    assert not should_record_circuit_outcome(
        success=False,
        error_type="empty_response",
        previous_error_type=None,
    )
    assert not should_record_circuit_outcome(
        success=False,
        error_type="empty_response",
        previous_error_type="empty_response",
    )
    assert not should_record_circuit_outcome(
        success=False,
        error_type="malformed_response",
        previous_error_type="malformed_response",
    )
    assert not should_record_circuit_outcome(
        success=False,
        error_type="image_read",
        previous_error_type=None,
    )
    assert should_record_circuit_outcome(
        success=False,
        error_type="timeout",
        previous_error_type="timeout",
    )
    assert (
        classify_request_error(
            "OverloadedError: Error code: 529 - 当前服务集群负载较高，请稍后重试"
        )
        == "overloaded_529"
    )


def _caption(text: str, index: int) -> dict:
    return {
        "caption_index": index,
        "text": text,
        "word_count": len(word_tokens(text)),
        "text_sha256": sha256_text(text),
    }


def test_response_completion_requires_exact_nonempty_caption_count():
    exact = {"captions": [_caption("One caption.", 0), _caption("Two caption.", 1)]}
    assert response_has_exact_caption_count(exact, 2)
    assert not response_has_exact_caption_count(
        {"captions": [exact["captions"][0]]}, 2
    )
    assert not response_has_exact_caption_count(
        {"captions": [*exact["captions"], _caption("Three caption.", 2)]}, 2
    )


def test_merge_and_validate_preserve_manifest_identity(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    originals_path = tmp_path / "originals.jsonl"
    output_root = tmp_path / "public-output"
    images = tmp_path / "images"
    rows = []
    originals = []
    for index in range(2):
        image_id = f"n00000001_{index + 1}"
        relative_path = f"n00000001/{image_id}.JPEG"
        image_path = images / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{index}".encode())
        rows.append(
            {
                "img_id": index + 1,
                "source_path": str(image_path),
                "synset": "n00000001",
            }
        )
        originals.append(
            {
                "img_id": index + 1,
                "id": image_id,
                "path": relative_path,
                "synset": "n00000001",
                "recaption_short": f"Original caption number {index + 1}.",
            }
        )
    manifest_path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    originals_path.write_text(
        "".join(canonical_json(row) + "\n" for row in originals),
        encoding="utf-8",
    )

    fingerprint = "test-fingerprint"
    response_dir = output_root / "responses" / fingerprint
    response_dir.mkdir(parents=True)
    manifest_digest = sha256_file(manifest_path)
    run = {
        "dataset": "imagenet100",
        "request_fingerprint": fingerprint,
        "manifest_sha256": manifest_digest,
        "models": ["kimi-k3"],
        "captions_per_model": 2,
        "min_words": 5,
        "max_words": 20,
        "max_jaccard": 1.0,
    }
    (response_dir / "run.json").write_text(json.dumps(run) + "\n", encoding="utf-8")
    connection = open_response_db(response_dir / "part-00000-of-00001.sqlite3")
    for index, (manifest_row, original_row) in enumerate(zip(rows, originals)):
        image_bytes = Path(manifest_row["source_path"]).read_bytes()
        record = {
            "schema": DISTILL_RECORD_SCHEMA,
            "request_fingerprint": fingerprint,
            "manifest_sha256": manifest_digest,
            "manifest_index": index,
            "img_id": manifest_row["img_id"],
            "id": original_row["id"],
            "path": original_row["path"],
            "source_path": manifest_row["source_path"],
            "synset": manifest_row["synset"],
            "image_sha256": sha256_bytes(image_bytes),
            "model": "kimi-k3",
            "captions": [
                _caption(
                    f"A detailed subject occupies the center with texture number {index + 1}.",
                    0,
                ),
                _caption(
                    f"Soft daylight surrounds a distinct scene viewed from angle {index + 1}.",
                    1,
                ),
            ],
        }
        connection.execute(
            "INSERT INTO responses VALUES (?, ?, ?)",
            (manifest_row["img_id"], "kimi-k3", canonical_json(record)),
        )
    connection.commit()
    connection.close()

    coverage_path = output_root / "coverage.json"
    coverage_args = argparse.Namespace(
        dataset="imagenet100",
        manifest=str(manifest_path),
        output_root=str(output_root),
        response_dir=str(response_dir),
        coverage_output=str(coverage_path),
        sample_limit=10,
    )
    assert run_coverage(coverage_args) == 0
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["status"] == "complete"
    assert coverage["missing_groups"] == 0
    assert coverage["present_unique_groups"] == 2

    merged_path = output_root / "captions" / "merged.jsonl"
    merge_args = argparse.Namespace(
        dataset="imagenet100",
        manifest=str(manifest_path),
        original_captions=str(originals_path),
        output_root=str(output_root),
        response_dir=str(response_dir),
        merged_output=str(merged_path),
        allow_incomplete=False,
        overwrite=False,
    )
    assert run_merge(merge_args) == 0

    merged_rows = [
        json.loads(line)
        for line in merged_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["img_id"] for row in merged_rows] == [1, 2]
    assert all(row["schema"] == MERGED_SCHEMA for row in merged_rows)
    assert all(row["caption_count"] == 3 for row in merged_rows)
    assert all(row["captions"][0]["source"] == "original" for row in merged_rows)

    validate_args = argparse.Namespace(
        dataset="imagenet100",
        manifest=str(manifest_path),
        output_root=str(output_root),
        merged_output=str(merged_path),
        verify_images=True,
        image_root=None,
    )
    assert run_validate(validate_args) == 0

    response_db = response_dir / "part-00000-of-00001.sqlite3"
    connection = open_response_db(response_db)
    raw = connection.execute(
        "SELECT record_json FROM responses WHERE img_id=1 AND model='kimi-k3'"
    ).fetchone()[0]
    partial = json.loads(raw)
    partial["captions"] = partial["captions"][:1]
    connection.execute(
        "UPDATE responses SET record_json=? WHERE img_id=1 AND model='kimi-k3'",
        (canonical_json(partial),),
    )
    connection.commit()
    connection.close()

    assert run_coverage(coverage_args) == 1
    incomplete = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert incomplete["status"] == "incomplete"
    assert incomplete["missing_groups"] == 1
    assert incomplete["invalid_caption_count_groups"] == 1
