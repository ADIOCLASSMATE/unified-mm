# 文本评测 P1 修正（2026-09-05）

适用：历史 A/B step 95,415 final EMA。不是新 x0content 重训练结果。

## 修正后的主指标（%）

| 任务 | A | B | 说明 |
| --- | ---: | ---: | --- |
| ARC-Easy | 65.9512 | 65.7828 | 原始选项字符数归一化 |
| ARC-Challenge | 38.6519 | 37.7133 | 从旧 B 较高改为 A 较高 |
| HellaSwag | 55.7757 | 55.7060 | 原始选项字符数归一化 |
| PIQA | 72.9597 | 73.2862 | 从旧 A 较高改为 B 较高 |
| OpenBookQA | 37.6000 | 36.6000 | 原始选项字符数归一化 |
| BoolQ | 64.1284 | 61.2232 | 主指标 accuracy 不变 |
| MMLU | 36.1427 | 36.3743 | 主指标 57-subject macro accuracy 不变 |
| WinoGrande | 56.8272 | 57.7743 | 16×910B 全量重新推理，各 1,267 条 |
| 八任务 macro | 53.5046 | 53.0575 | 全部使用修正后主指标，A 高 0.4471 个百分点 |

`acc_norm` 分母为 `len(original_choice)`，不是 token 数或 UTF-8 字节数；
编码时添加的分隔空格不计入分母。WinoGrande 改为
`log P(shared_suffix | prefix + option)`，不再评分候选项自身。
同位置 query-stream 打分及 A/B 的 attention contract 不变。

依据是冻结的 lm-eval
[指标实现](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/api/task.py)
及 [WinoGrande 预处理](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/tasks/winogrande/preprocess_winogrande.py)。

## 结果与备份

以下两个目录的逐样本记录、metrics、summary 已原地更新：

- `output/evaluation/unified-a-0p6b/final-ema-v9-20260902/text/`
- `output/evaluation/unified-b-0p6b/final-ema-v9-20260902/text/`

旧文件保留在各自 `legacy-before-p1/`；逐任务变化记录为 `p1_correction.json`。
目录名中的 v9 是历史产物身份，不表示修正后仍使用旧文本协议。
当前文本协议为 v3、sample/metrics 为 v2、summary/run/rank marker 为 v4，
完整评测配置为 v10。旧分片、旧完整 summary 被续跑/归档/趋势门禁拒绝。
两个文本 summary 均已恢复 `complete=true`、`pending_tasks=[]`，通过正式文本门禁。
WinoGrande 独立重评原始产物为各模型目录的 `winogrande-p1-v3-20260905/`，
每模型 16 个 rank 分片严格覆盖 1,267 个样本，均无 token 边界调整或上下文截断。

历史 B checkpoint 的现存路径为
`output/unified-b-flow-head-no-diagonal-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema`；
备份和 `p1_correction.json.checkpoint` 保留当时路径作为来源记录，当前 summary
及 `resolved_checkpoint` 指向现存路径。合并时显式传入 `--relocated_checkpoint`，
校验 EMA step、类型、state key 数、浮点 dtype 和 attention contract；不改用
当前新 B checkpoint 补历史 B 的 WinoGrande。

## 验证与后续

- 针对性测试：30 passed，包含 A/B 同位置后缀打分、字符分母、旧协议拒绝、
  重算数据对齐、备份、重复执行、WinoGrande 合并和 checkpoint 迁移校验。
- 可运行 CPU suite：457 passed、2 skipped；排除了本机缺少 `tbe` 的两个
  NPU 专用模块。没有将其记作通过。
- WinoGrande 已在 `dev-wjx-ascend` 以 BF16、16 ranks 完成 A/B 正式重评。
- 对 A/B 各 34,507 条最终记录逐条复核 ID、label、字符分母、有限 LL、
  raw/normalized argmax 和正确性；两套结果均通过 `validate_text_summary(formal=True)`。
- 飞书目标为《Unified-MM：模型设计、现有证据与下一步研究计划》第 8.4/8.5 节，
  使用修正后的完整主表、协议说明与数值结论。

八任务 macro 仅为内部跨任务摘要；上述 A/B 差异未做多 seed 或显著性检验，
不能据此宣称具有统计显著性或跨模态总冠军。

本次没有修改训练、验证、图像生成或多模态评测逻辑。
