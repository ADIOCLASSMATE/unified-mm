#!/usr/bin/env python3
"""Inventory the ablation and 512px release from their actual manifests."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
from statistics import median

import yaml

REPO = Path(__file__).resolve().parents[1]


def audit(repo: Path, root: Path) -> dict:
    files = {}

    def identity(path):
        path = Path(path)
        stat = path.stat()
        return {"path": os.path.relpath(path, repo), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def read(path):
        path = Path(path)
        before = identity(path)
        raw = path.read_bytes()
        assert before == identity(path), f"source changed: {path}"
        files[before["path"]] = {**before, "sha256": hashlib.sha256(raw).hexdigest()}
        return yaml.safe_load(raw) if path.suffix == ".yaml" else json.loads(raw)

    def rows(path):
        path = Path(path)
        before, digest = identity(path), hashlib.sha256()
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                yield json.loads(raw)
        assert before == identity(path), f"source changed: {path}"
        files[before["path"]] = {**before, "sha256": digest.hexdigest()}

    def location(label, path, note=""):
        path = repo / path
        assert path.exists(), path
        return {"label": label, "path": os.path.relpath(path, repo),
                "absolute": str(path.resolve()), "href": os.path.relpath(path, root), "note": note}

    old_config_path = repo / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
    new_config_path = repo / "configs/selfless/unified_b_x0_images512_v1_ascend64.yaml"
    old_config, new_config = read(old_config_path), read(new_config_path)
    old = old_config["dataset"]["params"]
    new = new_config["dataset"]["params"]
    release = (repo / new["image"]["manifest_jsonl"]).parent
    work = repo / "public/data_preparation/unified_b_512_sol_v1"
    receipt = read(work / "final_dataset_completion.json")
    assert receipt["state"] == "completed" and receipt["records"] == new["image"]["expected_records"]
    assert receipt["configuration"]["sha256"] == files[os.path.relpath(new_config_path, repo)]["sha256"]
    checks = {}
    for name, source in receipt["checks"].items():
        checks[name] = read(source["path"])
        assert files[os.path.relpath(source["path"], repo)]["sha256"] == source["sha256"], name
    legacy = read(root / "data-provenance/audit.json")
    caption_manifest = read(repo / "public/datasets/imagenet1k_synthetic_v1/captions/manifest.json")
    t2i_manifest = read(repo / "public/datasets/imagenet1k_synthetic_v1/t2i/manifest.json")
    contract = read(release / "publication.json")["contract"]
    prepared = read(work / "source_supply/prepare_status.json")
    scope = read(work / "scope.json")

    sources, buckets, purpose = Counter(), Counter(), Counter()
    annotations = defaultdict(Counter)
    views, synsets = Counter(), Counter()
    keys, identities, imagenet_ids = set(), set(), set()
    ordered = []
    for row in rows(release / "manifest.jsonl"):
        key, source = row["key"], row["source"]
        assert key not in keys and (source, row["source_id"]) not in identities
        assert row["split"] == "train" and row["image_size"] == 512
        keys.add(key)
        identities.add((source, row["source_id"]))
        sources[source] += 1
        bucket = row["selection_bucket"]
        buckets[bucket] += 1
        group = ("Open Images · TextOCR 场景文字" if bucket == "textocr_readable" else
                 "Open Images · 关系 / 属性" if source == "openimages" else
                 "WikiArt · 实际艺术图像" if source == "wikiart" else
                 "ImageNet · 物体 / 细粒度类别" if source == "imagenet" else
                 {"counting": "PixMo · 计数选图", "ocr": "PixMo · 文字选图",
                  "style": "PixMo · 风格选图"}.get(bucket, "PixMo · 关系 / 动作 / 其他"))
        purpose[group] += 1
        views[row["view_version"]] += 1
        observations = row.get("observations", {})
        counts = [x["count"] for x in observations.get("counts", [])
                  if type(x.get("count")) is int and x["count"] > 0]
        flags = {"relations": bool(observations.get("relations")),
                 "counts": bool(counts), "count_ge_2": any(x >= 2 for x in counts),
                 "count_ge_5": any(x >= 5 for x in counts),
                 "visible_text": bool(observations.get("visible_text")),
                 "uncertainties": bool(row.get("uncertainties")),
                 "upsampled": bool(row.get("upsampled"))}
        annotations[source].update(k for k, value in flags.items() if value)
        if source == "imagenet":
            imagenet_ids.add(row["source_id"])
            synsets[row["source_id"].split("_")[0]] += 1
        ordered.append((key, source))
    assert dict(sources) == receipt["source_counts"]
    assert files[os.path.relpath(release / "manifest.jsonl", repo)]["sha256"] == checks["all_training_loader_rows"]["manifest_sha256"]

    decisions, source_decisions = Counter(), defaultdict(Counter)
    generators = {"i2t": Counter(), "t2i": Counter()}
    word_counts = {"i2t": [], "t2i": []}
    examples = {}
    caption_rows = rows(release / "captions.jsonl")
    prompt_rows = rows(release / "t2i.jsonl")
    for index, (item, caption, prompt) in enumerate(zip_longest(ordered, caption_rows, prompt_rows)):
        assert item is not None and caption is not None and prompt is not None
        key, source = item
        assert caption["manifest_index"] == index and prompt["image_id"] == "train/" + key
        assert len(caption["captions"]) == len(prompt["model_result"]["prompts"]) == 1
        provenance = caption["provenance"]
        assert provenance == prompt["provenance"]
        assert provenance["finalizer_model"] == "gpt-5.6-sol" and provenance["finalizer_effort"] == "low"
        decision = provenance["decision"]
        decisions[decision] += 1
        source_decisions[source][decision] += 1
        texts = {"i2t": caption["captions"][0]["text"], "t2i": prompt["model_result"]["prompts"][0]["prompt"]}
        for task, text in texts.items():
            model = provenance["generator_models"][task]
            if decision == "generate":
                assert model == "gpt-5.6-sol"
            generators[task][model] += 1
            word_counts[task].append(len(text.split()))
        examples.setdefault(source, {"key": key, "decision": decision, **texts})
    assert dict(decisions) == checks["strict_text"]["decisions"]

    train_overlap, old_classes, train_records = set(), Counter(), 0
    for row in rows(repo / old["image"]["manifest_jsonl"]):
        train_records += 1
        old_classes[row["synset"]] += 1
        source_id = Path(row["source_path"]).stem
        if source_id in imagenet_ids:
            train_overlap.add(source_id)
    val_ids = {Path(row["source_path"]).stem for row in rows(repo / old["image"]["validation"]["manifest_jsonl"])}
    assert train_records == old["image"]["expected_records"] == legacy["captions"]["train"]["records"]
    assert len(train_overlap) == len(imagenet_ids) and not (val_ids & imagenet_ids)
    print(f"Verified {train_records:,} ablation images and {len(keys):,} new-release pairs", flush=True)

    def word_summary(values):
        values = sorted(values)
        return {"mean": round(sum(values) / len(values), 2), "median": median(values),
                "p95": values[int((len(values) - 1) * .95)], "max": values[-1]}

    def config_summary(config):
        params = config["dataset"]["params"]
        return {"schedule": params["schedule"],
                "micro_batch_sizes": {k: v["micro_batch_size"] for k, v in params["sources"].items()},
                "image_tokens": params["image"]["image_tokens_per_img"],
                "image_sequence_length": params["image"]["pad_to_length"],
                "image_latent_dim": params["image"]["image_latent_dim"],
                "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
                "stop_after_steps": config["training"].get("stop_after_steps"),
                "max_train_steps": config["training"]["max_train_steps"]}

    paths = {"shared": [location("纯文本 · 两套配置共用", "public/ClimbMix", "训练匹配 *.jsonl；100 个分片，未新合成纯文本。"),
                         location("纯文本固定验证 manifest", "public/datasets/climbmix_validation_v1/manifest.json", "400 条固定记录；仅从开训起排除这些记录时才构成留出集。")],
             "ablation": [location("ImageNet 原图 · 当前可访问路径", "public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC", "旧 train manifest 保留 /inspire/dataset/imagenet/v1 前缀；本机真实原图位于此 public 路径，训练直接读取缓存。"),
                          location("图像身份与 train / val 映射", "public/datasets/imagenet_full"),
                          location("256px train posterior", old["image"]["cache_path"]),
                          location("256px val posterior", old["image"]["validation"]["cache_path"]),
                          location("I2T 七字幕文件 · 训练选其中六条", old["image"]["caption_jsonl"]),
                          location("T2I 原始合成分片", "public/datasets/imagenet1k_synthetic_v1/t2i/shards"),
                          location("train 文本随机读取索引", old["image"]["synthetic_text_index_manifest"]),
                          location("val 文本随机读取索引 · 两套共用", old["image"]["validation"]["synthetic_text_index_manifest"]),
                          location("val I2T 描述 · 两套共用", old["image"]["validation"]["caption_jsonl"]),
                          location("256px FID 参考统计", old_config["evaluation"]["real_stats_path"])],
             "new": [location("最终合成文本 / 索引目录", release, "只把此完整 release 作为训练入口；历史 snapshot 不叠加计数。"),
                     *[location(label, new["image"][key]) for label, key in (
                         ("512px 训练图像 manifest", "manifest_jsonl"), ("I2T · 每图一条", "caption_jsonl"),
                         ("随机读取文本索引", "synthetic_text_index_manifest"), ("外部 posterior 索引", "cache_path"))],
                     location("T2I · 每图一条", release / "t2i.jsonl"),
                     location("独立图像与 latent 根目录", receipt["image_pool"]),
                     location("下载原图分片", Path(receipt["image_pool"]) / "raw_supply"),
                     location("补充图像 512px 分片", Path(receipt["image_pool"]) / "prepared_supply"),
                     location("本地 ImageNet 精选图像分片", Path(receipt["image_pool"]) / "imagenet_100k"),
                     location("最终 posterior 行映射", Path(receipt["image_pool"]) / "vae_releases/sol_100k_plus_corners_v1/posterior.rows.pt"),
                     location("512px 独立验证缓存", new["image"]["validation"]["cache_path"]),
                     location("512px FID 参考统计", new_config["evaluation"]["real_stats_path"])],
             "evidence": [location("旧消融数据配置", old_config_path), location("512px 数据验收配置 · 0.6B", new_config_path),
                          location("新数据最终验收报告", work / "final_dataset_completion.json"),
                          location("全量图像 / 文本 / 缓存校验", receipt["checks"]["full_images_and_posteriors"]["path"]),
                          location("全量训练 loader 校验", receipt["checks"]["all_training_loader_rows"]["path"]),
                          location("本次统计脚本", Path(__file__))]}
    result = {"schema": "evaluation_dataset_inventory_v1", "complete": True,
              "audited_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "public_root": str((repo / "public").resolve()), "files": list(files.values()), "paths": paths,
              "ablation": {"records": train_records, "classes": len(old_classes), "validation_records": len(val_ids),
                           "i2t_targets": train_records * 6, "i2t_stored": caption_manifest["caption_rows"],
                           "t2i_targets": train_records * t2i_manifest["prompts_per_image"],
                           "config": config_summary(old_config)},
              "new": {"records": len(keys), "sources": dict(sources), "selection_buckets": dict(buckets),
                      "purpose_groups": dict(purpose), "decisions": dict(decisions),
                      "source_decisions": dict(source_decisions), "generators": generators,
                      "annotations_by_source": dict(annotations), "views": dict(views),
                      "imagenet_classes": len(synsets), "imagenet_class_counts": dict(synsets),
                      "imagenet_train_identity_overlap": len(train_overlap), "imagenet_val_identity_overlap": 0,
                      "non_imagenet_records": len(keys) - len(imagenet_ids),
                      "word_counts": {k: word_summary(v) for k, v in word_counts.items()},
                      "examples": examples, "contract": contract, "config": config_summary(new_config),
                      "downloads": receipt["download_counts"], "prepared": prepared, "scope": scope,
                      "loader_audit": checks["all_training_loader_rows"],
                      "reused_imagenet_pairs": receipt["reused_imagenet_pairs"]},
              "counting_notes": ["Source and selection buckets count each released image once.",
                                 "Annotation flags overlap and are teacher observations, not independent ground truth.",
                                 "Old/new overlap uses ImageNet source IDs; cross-dataset visual uniqueness is not established.",
                                 "One pair contains one I2T caption and one T2I prompt; only these texts enter the current loaders."]}
    destination = root / "data-provenance/dataset-inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "output/evaluation")
    args = parser.parse_args()
    report = audit(REPO, args.root.resolve())
    print(json.dumps({k: report["new"][k] for k in ("records", "sources", "purpose_groups", "decisions", "views", "annotations_by_source", "word_counts", "imagenet_classes")}, ensure_ascii=False))
