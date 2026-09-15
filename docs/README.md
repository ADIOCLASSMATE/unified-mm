# 文档索引

## 项目入口

| 内容 | 主文档 |
| --- | --- |
| 安装与常用命令 | [项目首页](../README.md) |
| 从统一多模态预测方式的观察到双流设计 | [双流架构的设计动机](DUAL_STREAM_DESIGN_MOTIVATION.md) |
| A/B、C–F、only与规模实验定义 | [实验](EXPERIMENTS.md) |
| 训练预算、启动、恢复、checkpoint | [训练](TRAINING.md) |
| 语料、caption、latent与索引 | [数据](DATA.md) |
| 当前数据合成入口、连接配置、复用与失败兜底 | [DATA_SYNTHESIS](DATA_SYNTHESIS.md) |
| Long 原文复用、专项补缺与定向 DeepSeek 修复 | [复用优先 2026-09-15](B_REUSE_FIRST_20260915.md) |
| B 大规模训练的数据缺口与合成方案（历史诊断） | [I2T / T2I 数据建议（2026-09-11）](B_DATA_SCALING_RECOMMENDATION_20260911.md) |
| 当前大规模数据选择：非 ImageNet 至少 200 万图，优先复用文本 | [来源、精选资源与复用协议（2026-09-13）](B_2M_NON_IMAGENET_DATA_PLAN_20260913.md) |
| 512px Codex sol 首轮合成与独立目录（历史） | [9 月 12 日首轮图文流水线](B_CODEX_IMAGE_SYNTHESIS_20260912.md) |
| 图片来源与旧 Qwen → Codex 方案记录 | [历史图文合成方案](B_IMAGE_TEXT_SYNTHESIS_V1_20260911.md) |
| 评测目录、命令与报告 | [评测入口](EVALUATION_STRUCTURE.md) |
| NPU资源、项目与平台操作 | [Inspire](../INSPIRE.md) |
| 协作与维护约定 | [CLAUDE.md](../CLAUDE.md) |

## 训练验证

| 文档 | 内容 |
| --- | --- |
| [统一loss](UNIFIED_LOSS_VALIDATION.md) | 各任务分母、调度权重和输出 |
| [训练中下游评测](TRAINING_DOWNSTREAM_VALIDATION.md) | EMA快评、任务子集和时间预算 |
| [ClimbMix固定验证](CLIMBMIX_VALIDATION.md) | 采样、记录排除和缓存 |

## 评测与采样

| 文档 | 内容 |
| --- | --- |
| [评分协议](EVALUATION_PROTOCOL_AUDIT.md) | 文本、图像理解、生成指标定义 |
| [原生理解评测](PRETRAINING_NATIVE_EVALUATION.md) | 任务、资产和完整运行入口 |
| [CFG / Heun扫描](B_SAMPLING_SWEEP.md) | B采样参数选择 |
| [生成顺序](B_ORDER_SWEEP.md) | Halton、random、局部速度评分 |
| [统一生成矩阵](UNIFIED_MATRIX_CFG2_ORDER.md) | 9模型、28组CFG2/Heun10评测 |
| [官方生成评测](OFFICIAL_T2I_BENCHMARKS.md) | GenEval、DPG-Bench、MJHQ-30K |

## S2与语义分支

| 文档 | 内容 |
| --- | --- |
| [四组实验设计](SHOWO2_UNIFIED_ABLATION_DESIGN.md) | B、B+SigLIP、S2-single/dual与Show-o2阶段训练 |
| [B+SigLIP](B_SIGLIP_UNIFIED_ABLATION.md) | 实现、参数与infra |
| [S2 infra](S2_INFRA_20260910.md) | RF4、checkpointing、吞吐和恢复验证 |
| [Z](Z_EXPERIMENT.md) | 同图 sigma 相同、双流 backbone 一次前向、单流 DiT 联合去噪，Heun10（20 次 head） |

## 表征研究

| 版本 | 协议 / 方法 | 结果与复现 |
| --- | --- | --- |
| V1：输入与backbone | 合并在结果页 | [结果与命令](B_REPRESENTATION_PROBE_20260907.md) |
| V2：提示、检索与任务读出 | [协议](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md) | [结果](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md) |
| V3：类别留出与旋转泛化 | [协议与命令](B_GEOMETRY_PROTOCOL_V3.md) | [结果](B_GEOMETRY_V3_RESULTS_20260907.md) |
| V4：扩样本、原型与扰动 | [协议](B_GEOMETRY_PROTOCOL_V4.md) | [结果](B_GEOMETRY_V4_RESULTS_20260907.md) |
| V5：跨模型比较 | [协议](CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md)、[方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md) | [结果](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)、[复现](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)、[产物清单](CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md) |

## 维护记录

| 文档 | 内容 |
| --- | --- |
| [2026-09-10实验审查](../output/repo-audit/20260910-experiment-design/REVIEW_ZH.md) | 预算、结果、验证口径和实现差异 |
| [2026-09-09仓库审查](REPO_AUDIT_20260909.md) | 五项已复现问题与处理 |
| [2026-09-09重构](REPO_REFACTOR_20260909.md) | 模块职责、保存事务和16-NPU验证 |
| [历史B生成缓存](B_X0CONTENT_VALIDATION_CACHE_AUDIT_20260905.md) | 时间嵌入缓存修复 |
| [历史文本P1修正](TEXT_P1_CORRECTION_20260905.md) | 字符归一化、WinoGrande与修正后分数 |

## 历史ImageNet实验

| 文档 | 内容 |
| --- | --- |
| [架构消融](ABLATION_CONCLUSIONS.md) | 位置、flow head和gate |
| [ImageNet-100](IMAGENET100_HYPERPARAMETER_CONCLUSION.md) | 学习率扫描 |
| [ImageNet-1K 800 epoch](IMAGENET1K_800EP_PRETRAINING.md) | class训练、position-wise和sequential结果 |
| [Caption / T2I联合训练](IMAGENET1K_CAPTION_JOINT_CONCLUSION.md) | LR和文本权重选择 |
| [运行索引](archive/IMAGENET_EXECUTION_NOTES.md) | 历史硬件与80/400-epoch T2I入口 |

当前配置集中在项目入口各页；历史文档保留当时的配方、指标和产物路径。

- [B512 Long 主库与分离存储](B_BLIP3O_LONG_DATA_PLAN_20260913.md)：完整 Long 下载、专项补充、项目空间图文与 global_user posterior 缓存。
