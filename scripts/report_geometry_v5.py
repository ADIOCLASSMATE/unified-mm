"""Build a Chinese evidence table report only after all formal V5 result audits pass."""

import argparse
import json
from pathlib import Path

from utils.research.geometry_v5_assets import RUN, emit

NAMES = {
    "b_native": "B",
    "f_native": "F",
    "dinov2_qwen": "DINOv2 + Qwen",
    "mae_qwen": "MAE + Qwen",
    "siglip": "SigLIP 内容均值",
    "janusflow_understanding": "JanusFlow 理解",
    "janusflow_generation": "JanusFlow 生成",
    "showo2_understanding": "Show-o2 理解",
    "showo2_generation": "Show-o2 生成",
    "siglip_native": "SigLIP 原生 pooler（单列参考）",
}


def read(path):
    return json.loads(Path(path).read_text())


def ci(point, bounds):
    return f"{point:.3f} [{bounds[0]:.3f}, {bounds[1]:.3f}]"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def endpoint(data, setting, family, mode="centered_euclidean", selected=False):
    rows = [
        r
        for r in data[setting]
        if r["readout"]
        == ("native_endpoint" if setting == "siglip_native" else "content_mean")
        and r["family"] == family
        and r["mode"] == mode
        and r["fit_points"] == (600 if family == "imagenet" else 8192)
        and r["endpoint"] == ("dev_selected" if selected else "fixed")
        and (selected or r["requested_dimension"] == 32)
    ]
    assert len(rows) == 1
    return rows[0]


def main(root):
    contract = read(root / "comparison-contract.json")
    data, audits = {}, []
    for setting in contract["settings"]:
        audit = read(root / "audits" / f"results-{setting}.json")
        assert audit["status"] == "passed"
        audits.append(audit)
        data[setting] = [read(p) for p in (root / "endpoints" / setting).glob("*.json")]
    assert sum(a["layer_readout_rows"] for a in audits) == 1171
    assert read(root / "audits/summary-tables.json")["status"] == "passed"
    assert read(root / "audits/paired-model-differences.json")["status"] == "passed"
    figures = read(root / "figures/index.json")
    assert not figures["partial_preflight"] and set(figures["settings"]) == set(
        contract["settings"]
    )
    differences = read(root / "paired-model-differences-v5.json")
    sections = [
        "# 跨模型逐层语义几何 V5：完整数值报告",
        "本报告由正式机器结果自动生成。覆盖 14 个路径／提示设置、1171 个层对×读出组合；模型均冻结，没有新增训练。统计区间是给定权重、源域拟合和开发集选择的条件区间，不含重新训练或重新选层的不确定性。",
        "## 先区分三种证据",
        "kNN/RSA/CKA 回答关系结构是否对应；留出正交映射回答仅旋转／反射能否泛化；ARO 回答这种映射是否保留属性绑定与关系信息。这三项不是同一个问题，不能用较高 CKA 替代后两项，更不能推出 head 只负责采样。",
        "ImageNet：1000 类、32000 图、12000 模板，映射类别 600/200/200 fit/dev/test；COCO：11776 场景、58909 caption，512/8192/1024/2048 cal/fit/dev/test；ARO：451 属性＋417 关系，868 张已确认 COCO-ID 互斥原图。同场景先平均原始 caption 特征，再做中心化／球面化。",
        "## 固定最终层的共享关系结构",
        "以下是共同内容均值；SigLIP 训练过的原生 pooler 另作参考。几何样本为 ImageNet 200 测试类、COCO 前 512 测试场景；括号内为 199 次身份置换的 95% 零分布分位数，不是置信区间。",
    ]
    for family in ("imagenet", "coco"):
        rows = []
        for setting, name in NAMES.items():
            ep = endpoint(data, setting, family)
            source = read(ep["test_source"])
            geo = source["families"][family]["centered_euclidean"]["geometry"][
                "primary"
            ]
            cells = [
                f"{geo['scores'][m]:.3f} ({geo['null_summary'][m]['q95']:.3f})"
                if geo["valid"]
                else "无有效方差"
                for m in ("linear_cka", "rsa_spearman", "knn_10")
            ]
            rows.append([name, *cells])
        sections += [
            f"### {family.upper()}",
            table(
                [
                    "模型／路径",
                    "CKA（置换 q95）",
                    "RSA（置换 q95）",
                    "kNN@10（置换 q95）",
                ],
                rows,
            ),
        ]
    sections += [
        "![逐层关系曲线](figures/layers-common-centered_euclidean.png)",
        "## 严格留出与跨域旋转：固定最终层、共同 32 维",
        "每侧独立 fit-only PCA、整体 RMS 单位和正交矩阵均冻结；不逐轴白化，不在目标域重新中心化或拟合。R² 允许负值；负值不等于完全无语义，应同时看打乱对照和关系指标。不同模型的 R² 分母各属自己的表示空间。",
    ]
    for mode in ("centered_euclidean", "unit_sphere"):
        rows = []
        for setting, name in NAMES.items():
            cells = []
            for family in ("imagenet", "coco"):
                row = endpoint(data, setting, family, mode)
                for split in ("test", "transfer_test"):
                    cells.append(
                        ci(
                            row[split]["paired"]["r2"],
                            row[split]["bootstrap"]["r2_95_interval"],
                        )
                        if row["valid"]
                        else "不可拟合"
                    )
            rows.append([name, *cells])
        sections += [
            f"### {mode}",
            table(["模型／路径", "IN→IN", "IN→COCO", "COCO→COCO", "COCO→IN"], rows),
        ]
    sections += [
        "![旋转与迁移](figures/mapping-fixed.png)",
        "## B 相对其他模型：配对差值而非独立区间目测",
        "这里只列预定主对照：共同内容均值、最终层、欧氏 32 维。每个身份抽样同时用于两模型，正值表示 B 的 R² 更高；95% 区间为逐项区间，不是全实验同时置信区间。",
    ]
    contrast_rows = []
    for row in differences["rows"]:
        if row["primary_contrast"]:
            contrast_rows.append(
                [
                    NAMES[row["setting_b"]],
                    row["family"],
                    ci(row["test"]["a_minus_b_r2"], row["test"]["a_minus_b_r2_95"]),
                    ci(
                        row["transfer_test"]["a_minus_b_r2"],
                        row["transfer_test"]["a_minus_b_r2_95"],
                    ),
                ]
            )
    sections.append(
        table(["B 减去", "拟合源域", "留出 ΔR²", "跨域 ΔR²"], contrast_rows)
    )
    sections += [
        "## 开发集选择与方差覆盖",
        "仅在源域 dev 上、分别按预定读出选择层与 32/128/512 维。下面只展示内容均值欧氏模式；其余读出／模式／拟合规模保存在 endpoints/。低维成功不代表丢弃方向没有语义，低秩或病态的 full 拟合不构成全空间同构证明。",
    ]
    selected_rows = []
    for setting, name in NAMES.items():
        for family in ("imagenet", "coco"):
            row = endpoint(data, setting, family, selected=True)
            if row["valid"]:
                selected_rows.append(
                    [
                        name,
                        family,
                        f"{row['pair']['layer_x']} / {row['pair']['layer_y']}",
                        row["dimension"],
                        f"{row['test']['variance_retained_x']:.3f} / {row['test']['variance_retained_y']:.3f}",
                        ci(
                            row["test"]["paired"]["r2"],
                            row["test"]["bootstrap"]["r2_95_interval"],
                        ),
                        row["rotation_identified"],
                    ]
                )
            else:
                selected_rows.append([name, family, "不可选择", "—", "—", "—", False])
    sections += [
        table(
            [
                "模型／路径",
                "源域",
                "图层 / 文层",
                "维度",
                "测试方差保留 图/文",
                "测试 R²",
                "旋转可识别",
            ],
            selected_rows,
        ),
        "![维度敏感性](figures/mapping-dimensions.png)",
        "![拟合规模曲线](figures/coco-fit-size.png)",
        "## 统一模型：内容与原生任务读出不能混同",
        "下表固定 COCO8192 源域、欧氏模式；每个预定读出各自用源域 dev 选择层与共同维度，没有用测试结果挑选读出。固定最终层 32 维同时保留。原生 query、assistant boundary 和生成槽位不是同一种池化；这张表是读出敏感性诊断，不取代共同内容均值主对照。",
    ]
    unified = (
        "b_native",
        "f_native",
        "janusflow_understanding",
        "janusflow_generation",
        "showo2_understanding",
        "showo2_generation",
    )
    readout_rows = []
    for setting in unified:
        for readout in contract["settings"][setting]["readouts"]:
            selected = next(
                r
                for r in data[setting]
                if r["readout"] == readout
                and r["family"] == "coco"
                and r["fit_points"] == 8192
                and r["mode"] == "centered_euclidean"
                and r["endpoint"] == "dev_selected"
            )
            fixed = next(
                r
                for r in data[setting]
                if r["readout"] == readout
                and r["family"] == "coco"
                and r["fit_points"] == 8192
                and r["mode"] == "centered_euclidean"
                and r["endpoint"] == "fixed"
                and r["requested_dimension"] == 32
            )
            fixed_r2 = (
                f"{fixed['test']['paired']['r2']:.3f}" if fixed["valid"] else "无效"
            )
            if not selected["valid"]:
                readout_rows.append(
                    [NAMES[setting], readout, fixed_r2, "无效", "—", "—"]
                )
                continue
            readout_rows.append(
                [
                    NAMES[setting],
                    readout,
                    fixed_r2,
                    f"{selected['pair']['layer_x']} / {selected['pair']['layer_y']}, d={selected['dimension']}",
                    ci(
                        selected["test"]["paired"]["r2"],
                        selected["test"]["bootstrap"]["r2_95_interval"],
                    ),
                    ci(
                        selected["transfer_test"]["paired"]["r2"],
                        selected["transfer_test"]["bootstrap"]["r2_95_interval"],
                    ),
                ]
            )
    sections += [
        table(
            [
                "模型／路径",
                "读出",
                "固定最终层测试 R²",
                "dev 选图层 / 文层、维度",
                "留出 R² [95% CI]",
                "COCO→IN R² [95% CI]",
            ],
            readout_rows,
        ),
        "## ARO：困难负例功能检验",
        "下表固定最终层、欧氏 32 维，映射来自 COCO8192。真正应关注的是与任务内图像打乱相比的增益，不能只看是否略高于 50%。更换文本表述、语言先验等会影响此类成绩；没有在 ARO 拟合或选层。",
    ]
    for task, task_name in (
        ("aro_vg_attribution", "属性绑定（451）"),
        ("aro_vg_relation", "关系（417）"),
    ):
        for metric in ("cosine", "distance"):
            rows = []
            for setting, name in NAMES.items():
                ep = endpoint(data, setting, "coco")
                if not ep["valid"]:
                    rows.append([name, "不可拟合", "—", "—"])
                    continue
                row = ep["aro"][task]
                point, null = row["paired_fit"][metric], row["shuffled_image"][metric]
                rows.append(
                    [
                        name,
                        ci(point, row["bootstrap"][metric]["accuracy_95"]),
                        f"{null:.3f}",
                        ci(
                            point - null,
                            row["bootstrap"][metric][
                                "advantage_over_shuffled_image_95"
                            ],
                        ),
                    ]
                )
            sections += [
                f"### {task_name}／{metric}",
                table(
                    [
                        "模型／路径",
                        "正确率 [95% CI]",
                        "打乱图像",
                        "相对打乱增益 [95% CI]",
                    ],
                    rows,
                ),
            ]
    sections += [
        "![ARO 余弦功能读出](figures/aro-cosine.png)",
        "![ARO 距离功能读出](figures/aro-distance.png)",
        "## 噪声、读出与数值稳健性",
        "robustness/ 保留 B/F 的 sigma1、sigma2、VAE 后验均值，以及两种 flow 模型的两个额外固定噪声种子和图像中间时间。源域所有参数冻结。B/F 的 sigma 干预还移动首图像 query 空间位置，因此不是纯顺序干预。输入固定噪声没有语义，不能把舍入误差当作早期语义。",
    ]
    for setting in (
        "b_native",
        "f_native",
        "janusflow_generation",
        "showo2_generation",
    ):
        sections.append(f"![{NAMES[setting]} 扰动](figures/robustness-{setting}.png)")
    robust_rows = []
    for setting in (
        "b_native",
        "f_native",
        "janusflow_generation",
        "showo2_generation",
    ):
        for readout in ("content_mean", "native_task"):
            ep = next(
                r
                for r in data[setting]
                if r["readout"] == readout
                and r["family"] == "coco"
                and r["fit_points"] == 8192
                and r["mode"] == "centered_euclidean"
                and r["endpoint"] == "dev_selected"
            )
            if not ep["valid"]:
                robust_rows.append([NAMES[setting], readout, "无效", "—", "—", "—"])
                continue
            robust = read(
                root
                / "robustness"
                / setting
                / (Path(ep["sample_statistics"]).stem + ".json")
            )
            by_profile = {
                r["profile"]: r for r in robust["rows"] if r["target_family"] == "coco"
            }
            profiles = (
                ("native_sigma1", "native_sigma2", "native_mean")
                if setting in {"b_native", "f_native"}
                else ("seed1", "seed2", "image_midpoint")
            )
            robust_rows.append(
                [
                    NAMES[setting],
                    readout,
                    f"{ep['pair']['layer_x']} / {ep['pair']['layer_y']}, d={ep['dimension']}",
                    *[
                        ci(
                            by_profile[p]["error_reduction_over_original_denominator"],
                            by_profile[p]["error_reduction_95"],
                        )
                        for p in profiles
                    ],
                ]
            )
    sections += [
        "下面使用各读出在 COCO 源域 dev 选定的映射，在预定 512 个 COCO 测试场景上检验扰动。数值是除以同一原始目标分母的误差减少量，负值表示误差增大；不是两次各自归一化的 R² 相减。B/F 三列对应 sigma1、sigma2、后验均值；flow 三列对应额外种子1、额外种子2、图像 t=0.5。所有源域参数冻结。",
        table(
            [
                "模型／路径",
                "读出",
                "源 dev 选层、维度",
                "sigma1／seed1 [95% CI]",
                "sigma2／seed2 [95% CI]",
                "后验均值／图像中间时间 [95% CI]",
            ],
            robust_rows,
        ),
    ]
    sections += [
        "### 事后解释：固定坐标失败不等于关系结构消失",
        "为解释 Show-o2 的异常大误差，另对四组受扰模型、两种读出、固定32／源 dev 选择共48个 COCO512 条件做残差分解。没有重新拟合、重定位预测或修改正式分数。下面只展示触发此解释的三个条件；平均残差项与中心化残差项之和等于总误差增加，符号与上表误差减少量相反。",
    ]
    decomposition = read(root / "audits/perturbation-error-decomposition.json")
    assert decomposition["status"] == "passed" and len(decomposition["rows"]) == 48
    assert decomposition["posthoc_explanatory"]
    assert not any(
        decomposition[key]
        for key in ("target_refit", "predictions_recentered", "formal_results_modified")
    )
    decomposition_rows = []
    for readout, profile in (
        ("native_task", "seed1"),
        ("native_task", "seed2"),
        ("content_mean", "image_midpoint"),
    ):
        row = next(
            r
            for r in decomposition["rows"]
            if r["setting"] == "showo2_generation"
            and r["endpoint"] == "dev_selected"
            and r["readout"] == readout
            and r["profile"] == profile
        )
        axis = "x" if profile == "image_midpoint" else "y"
        raw = row["raw_change"][axis]
        decomposition_rows.append(
            [
                f"{readout} / {profile}",
                f"{row['pair']['layer_x']} / {row['pair']['layer_y']}, d={row['dimension']}",
                f"{row['mean_residual_error_increase_over_native_denominator']:.6f}",
                f"{row['centered_residual_error_increase_over_native_denominator']:.6f}",
                f"{100 * raw['translation_fraction_of_delta_energy']:.6f}%",
                f"{raw['centered_deformation_relative_rms']:.3f}",
            ]
        )
    sections += [
        table(
            [
                "Show-o2 生成条件",
                "源 dev 选图层 / 文层、维度",
                "平均残差误差增加／原分母",
                "中心化残差误差增加／原分母",
                "原始变化能量中的平移比例",
                "中心化变形／原语义 RMS",
            ],
            decomposition_rows,
        ),
        "生成槽位更换噪声时，几乎全部变化是整体平移，跨模态 CKA 和近邻关系基本保留；图像中间时间则还包含明显结构变形。不能只根据巨大负值就声称语义全部消失，也不能根据平移占比高就忽略相对原语义信号仍很大的变形。[完整48条残差分解](audits/perturbation-error-decomposition.json)保留独立重算证据；这不是目标适配后的新成绩。扰动图仅在大动态范围面板使用明确标注的 symlog 轴，[-1,1]内线性，负值不裁剪。",
    ]
    sections += [
        "Show-o2 正式使用 FP32 计算和存储，旧 BF16 完整产物归档；修订只依据 cal 数值检查。B/F neutral 末 token 的局部 BF16 批量敏感性单独披露，不据此提出强架构结论。全部 cal 检查、精度版本和模型权重逐参数核验见 audits/ 与 model-verification/。",
        "## 训练差异与结论边界",
        "B/F 不是从零的无教师纯压缩模型：有 Qwen/MAR VAE 预训练和配对 ImageNet 条件生成。B/F 的文本 query mask ID=151669，图像 query mask ID=151672，是不同 token；native query 本来就是异构读出。F 的数据、优化器、调度和训练配置与 B 一致，但独立训练，flow head 参数量差约 0.149%。这个对照比外部模型更接近架构消融，仍不是多个训练种子的因果验证。",
        "JanusFlow 使用 SigLIP 理解编码器和 REPA 中间表示对齐；Show-o2 的 VAE 语义分支先接受 SigLIP 蒸馏。它们不能被称为仅靠无教师 flow loss 的基线。SigLIP 是显式图文对齐正对照；DINOv2/MAE＋原始 Qwen 是无新图文连接器的单模态关系结构参考，不是随机负对照。参数量、预训练语料、教师、指令训练与输入分辨率不同，排名不能独立归因于架构。详见[方法文档](../../docs/CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)及其中的原论文来源。",
        "本实验允许回答共享关系结构和可泛化子空间是否存在、B 在哪些预定条件下优于或弱于参照；不能仅凭这些结果证明 head 完全不做语义计算，也不能把独立单模态模型的对应结构说成 B 特有现象。",
        "## 可复现产物与审计入口",
        "原始与修订特征、所有层结果、无效／负结果、拟合矩阵、身份级统计、开发集选择、扰动、所有 readout 图表均保留。主索引：geometry-v5.csv、analysis-index-v5.json、dev-selected-geometry-v5.json、layer-search-null-v5.json、paired-model-differences-v5.json、figures/index.json。",
        "结果覆盖审计为 audits/results-*.json；模型间配对统计逐项重算见 audits/paired-model-differences.json；数据／裁剪／身份、B-V4 精确复现和权重来源分别见 audits/samples-and-preprocessing.json、audits/b-v4-parity.json、audits/model-sources-and-training.json。测试见 audits/tests-v3-v4-v5.xml。计算资源停止证据单独保留于 resource-cleanup.json；本报告本身不代替资源状态或完整目标验收。",
    ]
    path = root / "RESULTS_ZH.md"
    path.write_text("\n\n".join(sections) + "\n")
    emit("v5_chinese_numeric_report_written", path=str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    main(parser.parse_args().output_dir)
