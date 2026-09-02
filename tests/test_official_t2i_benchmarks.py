import csv
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluate_official_t2i_benchmarks import (
    parse_dpgbench_results,
    run_summary,
    summarize_geneval_rows,
)
from scripts.generate_official_t2i_benchmarks import _build_t2i_item
from scripts.official_t2i_benchmarks import (
    CLEANFID_COMMIT,
    CLEANFID_VERSION,
    DPGBENCH_COMMIT,
    DPGBENCH_PROMPTS,
    GENEVAL_COMMIT,
    GENEVAL_IMAGES_PER_PROMPT,
    GENEVAL_PROMPTS,
    GENEVAL_TAG_COUNTS,
    MJHQ_CATEGORY_COUNTS,
    MJHQ_COMMIT,
    MJHQ_PROMPTS,
    BenchmarkPrompt,
    load_dpgbench_prompts,
    load_geneval_prompts,
    load_mjhq_prompts,
)
from utils.imagenet_flow_batching import collate_imagenet_flow_cache


def _geneval_prompts() -> list[BenchmarkPrompt]:
    prompts = []
    for tag, count in GENEVAL_TAG_COUNTS.items():
        for _ in range(count):
            index = len(prompts)
            metadata = {"tag": tag, "prompt": f"prompt {index}"}
            prompts.append(
                BenchmarkPrompt(
                    index=index,
                    prompt_id=f"{index:05d}",
                    prompt=metadata["prompt"],
                    category=tag,
                    metadata=metadata,
                )
            )
    return prompts


def test_geneval_loader_and_official_task_average(tmp_path: Path):
    repository = tmp_path / "geneval"
    prompt_dir = repository / "prompts"
    prompt_dir.mkdir(parents=True)
    expected = _geneval_prompts()
    (prompt_dir / "evaluation_metadata.jsonl").write_text(
        "".join(json.dumps(prompt.metadata) + "\n" for prompt in expected),
        encoding="utf-8",
    )
    prompts = load_geneval_prompts(repository, verify_revision=False)
    assert len(prompts) == GENEVAL_PROMPTS

    rows = []
    for prompt in prompts:
        correct = prompt.category == "single_object"
        for sample in range(GENEVAL_IMAGES_PER_PROMPT):
            rows.append(
                {
                    "filename": str(
                        tmp_path
                        / prompt.prompt_id
                        / "samples"
                        / f"{sample:05d}.png"
                    ),
                    "tag": prompt.category,
                    "correct": correct,
                }
            )
    metrics = summarize_geneval_rows(rows, prompts)
    assert metrics["overall"] == pytest.approx(1 / 6)
    assert metrics["task_scores"]["single_object"] == 1.0
    assert metrics["image_accuracy"] == pytest.approx(80 / GENEVAL_PROMPTS)
    assert metrics["prompt_accuracy_any_of_four"] == pytest.approx(
        80 / GENEVAL_PROMPTS
    )


def test_dpgbench_loader_and_complete_result_parser(tmp_path: Path):
    repository = tmp_path / "ELLA"
    prompt_dir = repository / "dpg_bench" / "prompts"
    prompt_dir.mkdir(parents=True)
    csv_path = repository / "dpg_bench" / "dpg_bench.csv"
    fieldnames = ["item_id", "text"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(DPGBENCH_PROMPTS):
            prompt_id = str(index)
            prompt = f"prompt {index}"
            writer.writerow({"item_id": prompt_id, "text": prompt})
            (prompt_dir / f"{prompt_id}.txt").write_text(prompt, encoding="utf-8")
    prompts = load_dpgbench_prompts(repository, verify_revision=False)
    assert len(prompts) == DPGBENCH_PROMPTS

    result_path = tmp_path / "dpg-results.txt"
    result_path.write_text(
        "".join(
            f"/images/{prompt.prompt_id}.png, 0.5, 0.5, 0.5, 0.5, 0.5\n"
            for prompt in prompts
        )
        + "L1 category scores:\n\tentity: 50.0\n"
        + "L2 category scores:\n\tentity - whole: 50.0\n"
        + "Image path: /images\nDPG-Bench score: 50.0\n",
        encoding="utf-8",
    )
    metrics = parse_dpgbench_results(
        result_path, {prompt.prompt_id for prompt in prompts}
    )
    assert metrics["dpgbench"] == 0.5
    assert metrics["evaluated_grids"] == DPGBENCH_PROMPTS
    assert metrics["l1_category_scores"] == {"entity": 0.5}


def test_mjhq_loader_requires_full_balanced_metadata(tmp_path: Path):
    repository = tmp_path / "MJHQ-30K"
    repository.mkdir()
    metadata = {}
    for category, count in MJHQ_CATEGORY_COUNTS.items():
        for index in range(count):
            prompt_id = f"{category}-{index:04d}"
            metadata[prompt_id] = {
                "prompt": f"{category} prompt {index}",
                "category": category,
            }
    (repository / "meta_data.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    prompts = load_mjhq_prompts(repository, verify_revision=False)
    assert len(prompts) == MJHQ_PROMPTS
    assert {prompt.category for prompt in prompts} == set(MJHQ_CATEGORY_COUNTS)


def test_t2i_benchmark_item_uses_training_serialization_and_image_span():
    class Tokenizer:
        eos_token_id = 2

        @staticmethod
        def encode(text, add_special_tokens):
            assert text == "Generate an image: official prompt"
            assert add_special_tokens is False
            return [10, 11]

    config = SimpleNamespace(
        model=SimpleNamespace(
            image_tokens_per_img=4,
            image_latent_dim=3,
            boi_token_id=20,
            mask_token_id=21,
            eoi_token_id=22,
        )
    )
    item = _build_t2i_item(
        "official prompt",
        tokenizer=Tokenizer(),
        config=config,
        prompt_prefix="Generate an image:",
        task_index=7,
        image_sigma_order="sequential",
        seed=42,
    )
    assert item["input_ids"].tolist() == [10, 11, 20, 21, 21, 21, 21, 22, 2]
    assert item["token_types"].tolist() == [0, 0, 2, 1, 1, 1, 1, 2, 2]
    assert item["image_loss_mask"].nonzero().flatten().tolist() == [3, 4, 5, 6]
    batch = collate_imagenet_flow_cache([item])
    assert batch["image_span_table"].tolist() == [[0, 0, 3, 7, 7]]
    assert torch.equal(
        batch["sigma"][0], torch.tensor([0, 1, 2, 4, 5, 6, 7, 3, 8])
    )


def test_summary_requires_one_checkpoint_and_frozen_official_protocols(
    tmp_path: Path,
):
    model_source = {"kind": "hf_final_ema", "path": "/model", "global_step": 10}
    components = {
        "geneval": {
            "schema": "official_t2i_benchmark_metrics_v1",
            "complete": True,
            "benchmark": "geneval",
            "metric_unit": "unit_interval",
            "primary_metric": "overall",
            "official_evaluator": {"commit": GENEVAL_COMMIT},
            "model_source": model_source,
            "metrics": {"overall": 0.4},
        },
        "dpgbench": {
            "schema": "official_t2i_benchmark_metrics_v1",
            "complete": True,
            "benchmark": "dpgbench",
            "metric_unit": "unit_interval",
            "primary_metric": "dpgbench",
            "official_evaluator": {"commit": DPGBENCH_COMMIT},
            "model_source": model_source,
            "metrics": {"dpgbench": 0.6},
        },
        "mjhq": {
            "schema": "official_t2i_benchmark_metrics_v1",
            "complete": True,
            "benchmark": "mjhq",
            "metric_unit": "fid_distance_lower_is_better",
            "primary_metric": "fid",
            "official_dataset": {"commit": MJHQ_COMMIT},
            "official_evaluator": {
                "commit": CLEANFID_COMMIT,
                "version": CLEANFID_VERSION,
            },
            "model_source": model_source,
            "leaderboard_comparable_at_1024px": False,
            "metrics": {"fid": 18.0},
        },
    }
    paths = {}
    for name, component in components.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(component), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "summary.json"
    run_summary(Namespace(**paths, output=output))
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["primary_metrics"] == {
        "geneval_overall": 0.4,
        "dpgbench": 0.6,
        "mjhq_fid": 18.0,
    }
    assert summary["mjhq_leaderboard_comparable_at_1024px"] is False
