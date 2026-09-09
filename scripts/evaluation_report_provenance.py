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
    return result


def export_provenance(root: Path, data: dict, write):
    write(root / "data-provenance/summary.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    origin = data["original_caption"]
    lines = ["# 训练、评测数据与文本合成来源", "",
        "本说明对应当前统一消融的实际训练配置与已发布数据。模型名称与协议版本按保存记录报告。", "",
        f"Original caption 来自 [{origin['dataset']}]({origin['url']})；公开数据卡及论文 Appendix E 指明生成模型为 **{origin['model']}**。",
        origin["protocol"], "", origin["training_usage"], "",
        "图像来自 ImageNet 官方 train / val；项目缓存为 MAR KL16 posterior，每图 256 个 latent token、每 token 16 通道。train 与 val 使用独立文件和索引。", ""]
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
