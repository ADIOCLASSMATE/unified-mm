# 历史 A/B 文本评测：P1 修正

适用 shared-condition 历史 A/B 的 step95415 final EMA。当前模型身份见 [实验定义](EXPERIMENTS.md)，评分规则见 [评测协议](EVALUATION_PROTOCOL_AUDIT.md)。

## 修正后结果（%）

| 任务 | 历史 A | 历史 B | 说明 |
| --- | ---: | ---: | --- |
| ARC-Easy | 65.9512 | 65.7828 | 原始选项字符数归一化 |
| ARC-Challenge | 38.6519 | 37.7133 | 字符归一化 |
| HellaSwag | 55.7757 | 55.7060 | 原始选项字符数归一化 |
| PIQA | 72.9597 | 73.2862 | 字符归一化 |
| OpenBookQA | 37.6000 | 36.6000 | 原始选项字符数归一化 |
| BoolQ | 64.1284 | 61.2232 | 主指标 accuracy 不变 |
| MMLU | 36.1427 | 36.3743 | 主指标 57-subject macro accuracy 不变 |
| WinoGrande | 56.8272 | 57.7743 | 16×910B 全量重新推理，各 1,267 条 |
| 八任务 macro | 53.5046 | 53.0575 | 全部使用修正后主指标，A 高 0.4471 个百分点 |

`acc_norm` 使用 `len(original_choice)`，编码新增空格不计入分母。WinoGrande 评分为 `log P(shared_suffix | prefix + option)`。依据固定 lm-eval revision `b954108c9baaaa934b4ad842033b31a97ee30816`。

## 产物

两个文本目录已更新逐样本记录、metrics 和 summary：

- `output/evaluation/unified-a-0p6b/final-ema-v9-20260902/text/`
- `output/evaluation/unified-b-0p6b/final-ema-v9-20260902/text/`

旧文件在各自 `legacy-before-p1/`，变化记录为 `p1_correction.json`。目录名v9保留历史身份；修正后的文本协议v3、sample/metrics v2、summary/run/rank marker v4，完整评测配置v10。

历史 B 的当前 checkpoint 路径为 `output/unified-b-flow-head-no-diagonal-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema`，summary 的 `resolved_checkpoint` 已对应迁移路径。

## 验证

WinoGrande 在16×910B上重评，每模型1267条；每模型全部34507条文本记录核对ID、label、分母、LL、argmax及正确性，正式文本完整性检查通过。定向测试30 passed；可运行CPU suite为457 passed / 2 skipped。

八任务macro为任务均值，此处为单seed结果。原始规则见 [lm-eval指标](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/api/task.py)和 [WinoGrande预处理](https://github.com/EleutherAI/lm-evaluation-harness/blob/b954108c9baaaa934b4ad842033b31a97ee30816/lm_eval/tasks/winogrande/preprocess_winogrande.py)。
