# 实验命名与定义

当前正式模型只有 **A、B**，分别对应原 `A_x0`、`B_x0`。
正文、图表和新实验说明统一写 A/B，不再附加 `_x0`、X0-content、新版或指定版。
**C–F 均是在正式 B 上的消融**；旧 A/B 及旧分支单独归入历史实验。

## 正式 A / B

两者都从 Qwen3-0.6B-Base 开始，使用 Selfless two-stream backbone 和
dynamic dual-stream contextual flow head（8 层、宽 1280）。Flow query
条件来自 backbone XT hidden，flow content 条件来自 backbone X0 hidden，
即 `backbone_xt_query_backbone_x0_content`。这里 X0 是共同的方法定义，
不再作为模型名称后缀。

| 项目 | A | B（基线） |
| --- | --- | --- |
| Backbone query attention | `sigma_kv < sigma_q` | `sigma_kv < sigma_q` |
| Backbone content attention | `sigma_kv < sigma_q` | `sigma_kv <= sigma_q` |
| Flow-head query attention | `sigma_kv < sigma_q` | `sigma_kv < sigma_q` |
| Flow-head content attention | `sigma_kv < sigma_q` | `sigma_kv <= sigma_q` |
| Backbone / flow attention contract | `selfless_strict` | `xlnet_content_diagonal` |
| 图像训练顺序 | random | random |

A/B 的主比较只改变 backbone 和 flow head 的 content 对角线可见性。
共同训练设置是 64×910B、seed 42、100B 名义文本目标、95,415 步，任务调度为
`[climbmix, t2i, climbmix, i2t]`。详细数值及 C–F 的实现字段由
[100B 训练合同](../configs/protocols/unified_ablation_100b_ascend64.yaml) 维护。

```bash
# 正式 A
bash script/selfless/pretraining_unified_ablation_a_0p6b_formal_ascend64.sh
# 正式 B
bash script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh
```

## B 上的消融：C–F

C–F 都以正式 B 为参照，只改变各自指定因素，按完整 100B 合同从相同预训练
权重开始训练。“在 B 上”表示方法与实验设置以 B 为基线，不是从 B 的已训练
checkpoint 续训。

| 名称 | 相对 B 的改动 |
| --- | --- |
| C · B + 文本 AR | 文本改为单流、按物理位置 next-token AR，图像路径沿用 B |
| D · B + Dynamic-XT | T2I 的 backbone query 接收带时间嵌入的 `x_t`，每次 ODE 速度计算刷新条件；训练 r4，完整评测选修复后的 r2 |
| E · B + sequential | 图像训练及原生生成均按 sequential 顺序 |
| F · B + position-wise head | 保留 B 的 backbone，flow head 换成参数量匹配的逐位置 AdaLN MLP，无跨 token attention 或 content stream |

保留结果 ID `c_on_b`、`d_on_b`、`e_on_b`、`f_on_b`。对应启动脚本为
`script/selfless/pretraining_unified_ablation_{c,d,e,f}_on_b_0p6b_formal_ascend64.sh`。
报告将它们归入“B 上的消融”，与旧版本区分；选择该组时同时显示 B 基线。

## 历史实验与辅助实验

历史实验按“历史 + 原字母 + 改动”展示，不再让旧版本占用裸名 A/B。
此处只汇总差异；精确架构、训练任务和权重始终以各自 checkpoint/config 为准。

| 历史身份 ID | 显示名称 / 定义 |
| --- | --- |
| `a_legacy` | 历史 A · shared-condition：严格注意力，flow query/content 共用 XT-query 条件 |
| `b_flowdiag` | 历史 B · shared-condition / flow 对角线：backbone 和 flow content 都含对角线，共用 XT-query 条件 |
| `b_no_flowdiag` | 历史 B · shared-condition / flow 无对角线：backbone content 含对角线，flow content 严格，共用 XT-query 条件 |
| `c_on_a_legacy` | 历史 C · 旧 A + 文本 AR：基于 shared-condition 旧 A 的早期对照 |
| `a_caption_only` / `b_caption_only` | 历史 A/B · caption-only：仅 I2T 训练 |
| `a_text_only` / `b_text_only` | 历史 A/B · text-only：仅 ClimbMix 训练 |

其他旧单任务配置按实际任务标记为历史对照。B 的 flow depth 16/30 实验归入
深度扩展，1.7B LR sweep 归入历史调参；都不占用正式 A/B 的名称或默认视图。
早期 ImageNet-only 架构研究单独见[历史结论](ABLATION_CONCLUSIONS.md)。

## 名称、身份与结果的对应

[实验登记表](../configs/protocols/experiment_registry.json) 是显示名称和分组的唯一来源；
`main` 仅包含 A/B，`ablation` 包含 C–F，`legacy` 包含旧版本。
报告构建时用登记表更新旧 manifest 的显示信息。
`formal` purpose 表示该运行按完整实验记录，并不表示它仍是当前研究主线。

| 正式名 | 已保存的结果 ID | 已保存的训练目录（`output/` 下） |
| --- | --- | --- |
| A | `a_x0` | `unified-a-x0content-0p6b-100b-imagenet-split-s42-r1` |
| B | `b_x0` | `unified-b-x0content-0p6b-100b-imagenet-split-s42-r1` |

这些 ID 和目录是历史结果的关联键，保留原值；它们不是当前展示名。
尤其不能把正式 A/B 指向不含 `x0content` 的旧 `unified-a/b-0p6b-...` 目录。
已有 checkpoint、结果数值、冻结协议、历史报告文件名和来源校验均保留可追溯性。

[评测总览](../output/evaluation/index.html) 的首页结论、指标、训练曲线和定性样例
默认展示正式 A/B；B 上的 C–F 消融、历史实验通过筛选查看。采样扫描写作“B · CFG / Heun”或
“B · 解码顺序”，属于同一模型的推理设置实验，不再增加模型字母。
生成分数必须连同 CFG、步数、顺序和权重版本比较，详见[评测结构](EVALUATION_STRUCTURE.md)。
