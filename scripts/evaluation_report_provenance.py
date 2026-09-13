"""Evidence and readable dataset lineage for the local evaluation report."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


def collect_provenance(repo: Path, root: Path):
    from scripts.distill_imagenet_captions import build_caption_prompt
    spec = json.loads((repo / "configs/protocols/evaluation_data_sources.json").read_text())
    if spec.get("schema") != "unified_evaluation_data_sources_v1":
        raise ValueError("unknown dataset provenance schema")
    audit_path = root / "data-provenance/audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if audit:
        if not audit.get("complete") or audit.get("schema") != "evaluation_data_provenance_audit_v1":
            raise ValueError("incomplete dataset provenance audit")
        for source in audit["files"]:
            path = repo / source["path"]
            stat = path.stat()
            if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(f"dataset changed since provenance audit: {path}; rerun scripts/audit_evaluation_data_provenance.py")
    result = json.loads(json.dumps(spec))
    # Record the existing download revisions, without reading corpus payloads
    # or recomputing content hashes. Repository identity was recovered from the
    # original download list and is preserved in evaluation_data_sources.json.
    metadata = sorted((repo / "public/ClimbMix/.cache/huggingface/download").glob("part*.jsonl.metadata"))
    result["climbmix_download"] = {"repository": "OptimalScale/ClimbMix",
        "url": spec["training"][0]["source_url"], "metadata_files": len(metadata),
        "revisions": dict(Counter(p.read_text().splitlines()[0] for p in metadata)),
        "source_identity_evidence": "Original local download list /tmp/climbmix_aria2.input pointed to https://huggingface.co/datasets/OptimalScale/ClimbMix/resolve/main/",
        "metadata_root": os.path.relpath(repo / "public/ClimbMix/.cache/huggingface/download", root)}
    for section in ("training", "evaluation"):
        for row in result[section]:
            row["evidence"] = [{"label": p, "path": os.path.relpath(repo / p, root)} for p in row["evidence"]]
            for entry in row["evidence"]:
                if not (root / entry["path"]).is_file():
                    raise FileNotFoundError(entry["label"])
    qwen = json.loads((repo / "configs/caption_farm/imagenet1k_qwen36_35b_a3b_fp8.json").read_text())
    t2i = json.loads((repo / "public/datasets/imagenet1k_synthetic_v1/t2i/manifest.json").read_text())
    captions = json.loads((repo / "public/datasets/imagenet1k_synthetic_v1/captions/manifest.json").read_text())
    result.update(qwen_protocol=qwen["caption"], styles=t2i["styles"], caption_manifest=captions,
                  minimax_template=build_caption_prompt(3, 32, 60, "{class_hint}"),
                  audit_path="data-provenance/audit.json" if audit else None,
                  audit={k: v for k, v in audit.items() if k != "files"} if audit else None,
                  document="data-provenance/README.md")
    result["protocol_links"] = [{"label": label, "path": os.path.relpath(repo / path, root)} for label, path in (
        ("Qwen 三个提示模板与采样配置", "configs/caption_farm/imagenet1k_qwen36_35b_a3b_fp8.json"),
        ("Qwen 实际图像请求实现", "caption_farm/worker.py"),
        ("MiniMax 三字幕模板与校验实现", "scripts/distill_imagenet_captions.py"),
        ("T2I 发布 manifest / 12 风格", "public/datasets/imagenet1k_synthetic_v1/t2i/manifest.json"),
        ("train/val 对齐审计", "public/datasets/imagenet1k_synthetic_v1/alignment/audit_report.json"),
        ("训练数据、采样和 loss 协议", "configs/protocols/unified_ablation_100b_ascend64.yaml"))]
    inventory_path = root / "data-provenance/dataset-inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if inventory.get("schema") != "evaluation_dataset_inventory_v1" or not inventory.get("complete"):
            raise ValueError("incomplete dataset inventory")
        for source in inventory["files"]:
            stat = (repo / source["path"]).stat()
            if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError("dataset inventory changed; rerun scripts/audit_evaluation_dataset_inventory.py: " + source["path"])
        result["dataset_inventory"] = {k: v for k, v in inventory.items() if k != "files"}
        result["dataset_inventory_path"] = "data-provenance/dataset-inventory.json"
        old, new = inventory["ablation"], inventory["new"]
        result["large_scale_data_protocol"] = {
            "scope": "full_imagenet_train_plus_corner_case_images",
            "status": "initial_release_complete_full_imagenet_processing_incomplete",
            "imagenet_candidates": old["records"],
            "current_corner_case_images": new["non_imagenet_records"],
            "combined_candidates_before_further_exclusions": old["records"] + new["non_imagenet_records"],
            "imagenet_already_ready": new["imagenet_train_identity_overlap"],
            "imagenet_remaining": old["records"] - new["imagenet_train_identity_overlap"],
            "image_size": 512, "image_tokens": 1024,
            "initial_release_is_final_large_scale_pool": False,
            "source_sampling_weights": "not_yet_frozen",
        }
    plan_path = repo / "configs/protocols/unified_b_non_imagenet_2m_v1.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("schema") != "unified_b_non_imagenet_quota_plan_v1":
            raise ValueError("unknown non-ImageNet quota plan")
        if sum(row["target"] for row in plan["sources"]) != plan["target_unique_non_imagenet_images"]:
            raise ValueError("initial source targets do not sum to the non-ImageNet baseline")
        if not plan["target_is_minimum"] or plan["source_targets_are_hard_caps"]:
            raise ValueError("source targets are coverage baselines, not acceptance caps")
        if len({row["id"] for row in plan["sources"]}) != len(plan["sources"]):
            raise ValueError("duplicate source quota IDs")
        document = repo / plan["document"]
        if not document.is_file():
            raise FileNotFoundError(document)
        result["non_imagenet_2m_plan"] = {
            **plan, "document_href": os.path.relpath(document, root),
            "config_href": os.path.relpath(plan_path, root),
        }
        if "large_scale_data_protocol" in result:
            result["large_scale_data_protocol"]["superseded_by"] = "non_imagenet_2m_plan"
    return result


def export_provenance(root: Path, data: dict, write):
    write(root / "data-provenance/summary.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    origin = data["original_caption"]
    lines = ["# 消融数据协议与大规模训练准备协议", "",
        "本说明对应当前统一消融的实际训练配置与已发布数据。模型名称与协议版本按保存记录报告。", "",
        f"Original caption 来自 [{origin['dataset']}]({origin['url']})；公开数据卡及论文 Appendix E 指明生成模型为 **{origin['model']}**。",
        origin["protocol"], "", origin["training_usage"], "",
        "图像来自 ImageNet 官方 train / val；项目缓存为 MAR KL16 posterior，每图 256 个 latent token、每 token 16 通道。train 与 val 使用独立文件和索引。", ""]
    inventory = data.get("dataset_inventory")
    if inventory:
        old, new = inventory["ablation"], inventory["new"]
        target = data["large_scale_data_protocol"]
        lines[2:2] = ["## 两套训练数据协议", "",
            f"统计时间：{inventory['audited_at']}。下文旧合成协议仍对应消融实验；新数据使用单独的 512px 协议。", "",
            f"**正式大规模训练范围：完整 ImageNet train {old['records']:,} 张 + 补齐 corner cases 的图库。** 目前已有非 ImageNet 补充图 {new['non_imagenet_records']:,} 张，按来源身份计算合并候选 {target['combined_candidates_before_further_exclusions']:,} 张，最终以去重、评测排除与验收后数量为准。", "",
            f"当前 {new['records']:,} 张是已验收的首轮发布，不是正式大规模训练的完整图库。首轮 ImageNet {new['imagenet_train_identity_overlap']:,} 张来自早期约 100K 校准 / 复用验证批次；剩余 {target['imagenet_remaining']:,} 张尚未完成新协议处理，不是经过全库筛选后被淘汰。", "",
            "| 数据 | 图像数 | I2T 目标 | T2I 目标 | 图像 token |", "| --- | ---: | ---: | ---: | ---: |",
            f"| 旧消融 / ImageNet-1K | {old['records']:,} | {old['i2t_targets']:,} | {old['t2i_targets']:,} | 256 |",
            f"| 新 512px 首轮发布（已完成） | {new['records']:,} | {new['records']:,} | {new['records']:,} | 1,024 |", "",
            "新集每图一条 I2T 和一条忠实 T2I 文本。新生成只用 GPT-5.6-sol low；复用文本保留原作者，并由 sol 复核。",
            f"ImageNet 原图与旧训练集重合 {new['imagenet_train_identity_overlap']:,} 张；另有 {new['non_imagenet_records']:,} 张来自其他图库。目录独立不代表原图身份完全独立。", "",
            "| 新集图库 | 图片数 | 占比 |", "| --- | ---: | ---: |",
            *[f"| {source} | {count:,} | {count / new['records']:.2%} |" for source, count in new["sources"].items()], "",
            "完整 ImageNet 作为候选底座逐图处理，排除损坏、明确重复及评测重叠并记录原因；不再用 100K 数量上限替代全库。已有 512px 结果按原图身份复用；其余图片重新固定 512px 视图并编码，旧 256px latent 不能直接拼入。旧六字幕及 faithful_photo 可作为 sol 复核候选，不合格则重写。",
            "当前 512px YAML 仍只读取首轮 release，完整 ImageNet 合并发布尚未生成。正式训练应在完整发布验收后切换入口。保留全图库与来源采样权重分别定义，不能用删除 ImageNet 图片代替提高 corner case 曝光。两套均沿用 ClimbMix；现有网页分数仍属于旧消融。",
            "旧图像 microbatch=16，新配置=4；两者每图 256/1,024 token，因此每步图像位置数相同，但新配置图次只有四分之一，比较时必须单列图次与算力。",
            "新缓存与全量 loader 已验收，尚无新模型训练结果。512px FID 使用独立参考统计，不与旧 256px 数值直接混排。", "",
            f"[完整统计 JSON]({data['dataset_inventory_path'].split('/')[-1]}) · [网页分布、能力分析与完整路径](../index.html#sources)", ""]
        for group, entries in inventory["paths"].items():
            lines += [f"## 路径 / {group}", ""]
            lines += [f"- {entry['label']}：`{entry['absolute']}`。{entry['note']}" for entry in entries]
        lines += ["", "## 旧消融数据的详细来源与协议", ""]
    plan = data.get("non_imagenet_2m_plan")
    if plan:
        lines[2:2] = ["## 2026-09-13 当前计划：ImageNet 之外至少 200 万原图，优先复用文本", "",
            f"**目标至少 {plan['target_unique_non_imagenet_images']:,} 张非 ImageNet 合格原图**，200 万基线加 ImageNet 原始 train 为 {plan['combined_target_before_imagenet_exclusions']:,} 张。已有 {plan['existing_non_imagenet_release_images']:,} 张非 ImageNet 首轮产物计入该目标；目标尚未实现。各来源配额不是上限，有价值的额外合格项继续保留，训练分布通过采样权重控制。", "",
            "先按能力分桶并连接已有 caption/prompt，再完成候选下载、去重与 512px 验收；仅缺失或错误样本调用 SII API。合格 caption 可以同时作为 I2T/T2I 文本，不逐图调用 sol 生成或审核。", "",
            "| 非 ImageNet 来源归属 | 起始合格原图目标，非上限 | 文本处理 |", "| --- | ---: | --- |",
            *[f"| {row['name']} | {row['target']:,} | {row['reuse']} |" for row in plan["sources"]], "",
            "原图互斥归属，多种标注可共存；不足时以同能力候选补位并记录变更，超过配额的优质图可以继续保留。JourneyDB 保留条件池，模型 512px 前置质量验收尚未完成；上一版全来源下载已暂停并保留断点。", "",
            f"[完整选择依据、精选资源与执行方案](../{plan['document_href']}) · [机器可读协议](../{plan['config_href']})", "",
            "**以下为旧消融和 2026-09-12 首轮准备记录。** 其中 sol 合成、滚动补图和当时的候选范围只描述该历史阶段，不作为 9 月 13 日新方案的执行指令。", ""]
    for section, label in (("training", "训练与验证"), ("evaluation", "评测与网页样本")):
        lines += [f"## {label}", ""]
        for row in data[section]:
            source = f"[{row['source']}]({row['source_url']})" if row.get("source_url") else row["source"]
            lines += [f"### {row['usage']}", "", source + "。" + row["size"] + "。", "", row["protocol"], "",
                "依据：" + " · ".join(f"[{e['label']}](../{e['path']})" for e in row["evidence"]), ""]
    lines += ["## 合成模型与协议", ""]
    for key, title in (("qwen", "Qwen 合成 caption"), ("minimax", "MiniMax 合成 caption"), ("t2i", "T2I 合成 prompts")):
        lines += [f"### {title}", "", data["synthesis_notes"][key], ""]
    lines += ["### Qwen 原始提示与参数", "", "```json", json.dumps(data["qwen_protocol"], ensure_ascii=False, indent=2), "```", "",
              "### MiniMax 模板（当前源码默认参数示例，非历史请求快照）", "", "```text", data["minimax_template"], "```", "",
              "### T2I 风格（存储顺序）", "", ", ".join(data["styles"]), ""]
    audit = data.get("audit")
    if audit:
        lines += ["## 全量来源统计", "", f"检查时间：{audit['audited_at']}。逐条读取 indexed train/val 与 caption 文件；没有重新生成数据或计算哈希。", "",
                  "| 数据 | 模型 / 来源 | 协议 | 数量 |", "| --- | --- | --- | ---: |"]
        for split, block in audit["t2i"].items():
            for row in block["models"]:
                lines.append(f"| T2I {split} | {row['model']} / {row['reasoning_effort']} | {row['prompt_version']} | {row['records']:,} 图 |")
        for split, block in audit["captions"].items():
            for row in block["sources"]:
                lines.append(f"| Caption {split} | {row['source']} / {row['model'] or '原始行未标注模型'} | {row['prompt_version'] or '—'} | {row['captions']:,} 条 |")
        lines += ["", "完整逐来源计数及原始样本见 [audit.json](audit.json)。", ""]
    lines += ["## 证据边界", "", "合成记录能证明生成模型与保存的协议版本；不能保证每条描述均准确。12 风格文本与对应原图的风格可能不同。",
              "ClimbMix、基座模型预训练与外部 benchmark 之间未在本次执行语义去重；本项目的 ImageNet train/val 独立不等于所有上游数据无重叠。",
              "MiniMax 历史请求快照与 T2I 原始完整合成指令未在当前数据目录找到；具体缺失项已在上文标注。", ""]
    write(root / data["document"], "\n".join(lines))
