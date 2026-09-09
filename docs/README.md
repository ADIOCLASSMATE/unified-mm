# 文档索引

当前正式模型是 **A、B**；原 A_x0/B_x0 后缀停用于展示。
先读[实验命名与定义](EXPERIMENTS.md)，日常查看[统一评测总览](../output/evaluation/index.html)。
C–F 单列为“B 上的消融”；旧 A/B 和旧单任务对照保留为历史实验。

## 当前训练与评测

- [A/B 与 C–F 的 100B 训练合同](../configs/protocols/unified_ablation_100b_ascend64.yaml)
- [评测结构与统一输出目录](EVALUATION_STRUCTURE.md)
- [训练中下游验证](TRAINING_DOWNSTREAM_VALIDATION.md)
- [仓库正确性与重构审计（2026-09-09）](REPO_AUDIT_20260909.md)
- [重构实施、回退点与开发机验收](REPO_REFACTOR_20260909.md)
- [原生图像理解评测](PRETRAINING_NATIVE_EVALUATION.md)
- [评测协议审计](EVALUATION_PROTOCOL_AUDIT.md)
- [GenEval / DPG-Bench / MJHQ 官方评测](OFFICIAL_T2I_BENCHMARKS.md)
- [理解 benchmark 选择依据](IMAGE_UNDERSTANDING_BENCHMARK_SELECTION.md)

## B 的采样实验

- [CFG / Heun 扫描](B_SAMPLING_SWEEP.md)
- [固定参数后的解码顺序比较](B_ORDER_SWEEP.md)
- [跨模型固定 CFG / 顺序矩阵（含历史消融）](UNIFIED_MATRIX_CFG2_ORDER.md)

## 表征研究

- V5：[结果](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)、[方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)、[协议](CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md)、[复现](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)、[完成审计](CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md)
- V4：[结果](B_GEOMETRY_V4_RESULTS_20260907.md)、[协议](B_GEOMETRY_PROTOCOL_V4.md)
- V3：[结果](B_GEOMETRY_V3_RESULTS_20260907.md)、[协议](B_GEOMETRY_PROTOCOL_V3.md)
- V2：[结果](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md)、[协议](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md)
- [最初的输入 embedding 与表征诊断](B_REPRESENTATION_PROBE_20260907.md)

这些版本回答不同问题；V5 仍使用部分 V2–V4 的公共实现。较旧版本不是可直接删除的重复文件。

## 历史实验与修正依据

- [早期 ImageNet 架构消融](ABLATION_CONCLUSIONS.md)
- [ImageNet-100 超参数](IMAGENET100_HYPERPARAMETER_CONCLUSION.md)
- [ImageNet-1K 800-epoch 合同](IMAGENET1K_800EP_PRETRAINING.md)
- [Caption 联合训练结论](IMAGENET1K_CAPTION_JOINT_CONCLUSION.md)
- [早期平台与 ImageNet 执行记录](archive/IMAGENET_EXECUTION_NOTES.md)
- [B 验证缓存审计](B_X0CONTENT_VALIDATION_CACHE_AUDIT_20260905.md)
- [文本 P1 修正](TEXT_P1_CORRECTION_20260905.md)

历史分数保留其当时的采样配置和协议标签；当前默认值以仓库 README 与正式评测协议为准。
