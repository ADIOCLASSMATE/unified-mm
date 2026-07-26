# Selfless-Flow Qwen Backbone × DF1 Flow Head 联合消融 Proposal

Status: **active; seed-42 execution authorized**

Written: 2026-07-25

Scope: balanced ImageNet-100，单 seed 架构选择；不自动扩展到多 seed 或 full ImageNet。

## 1. 决策问题

前两轮实验分别收敛出两个关闭的 runtime 接口：

- Qwen image backbone：`E2-Q1 / E2-Q0 / E2b-Q0`；
- dynamic dual-stream flow head：`DF1-FH0 / DF1-FH4`。

单独完成两个消融并不能保证各自的默认项组合后仍是最优系统。backbone 和 flow head
都处理二维位置，二者可能互补、重复或相互干扰。本轮因此运行完整 `3×2` factorial，
回答：

> 在固定 `DF1` dynamic dual-stream 架构下，哪一个 Qwen image-backbone position
> contract 与哪一个 flow-head position contract 组成质量足够好、概念最纯净且最适合
> scaling 的最终模型架构？

本轮重新训练全部六个 cell。旧的 `E2-Q1 × DF1-FH0/FH4` seed-42 checkpoint 只作为
外部 sanity anchor，不进入主 selector；它们来自上一轮冻结源码，不能与 fresh Q0
cell 构成 matched causal comparison。

## 2. 候选架构

### 2.1 Qwen image backbone

| Backbone | Observed latent additive 2D | Mask query additive 2D | Qwen image attention |
| --- | ---: | ---: | --- |
| `E2-Q1` | 无 | 有 | row/column 2D RoPE |
| `E2-Q0` | 无 | 无 | row/column 2D RoPE |
| `E2b-Q0` | 有 | 无 | row/column 2D RoPE |

三者都没有 stage embedding、S2D、sequence-1D image RoPE 或额外 layout branch。

### 2.2 DF1 flow head

| Flow head | Query/content additive 2D | Flow attention |
| --- | ---: | --- |
| `DF1-FH0` | 有 | 无 RoPE |
| `DF1-FH4` | 无 | row/column 2D RoPE |

两者共享同一个 `DF1` attention、MLP、AdaLN、dynamic content update、strict
`sigma[k] < sigma[q]` mask 和参数预算，只改变完整 position contract。

## 3. 主矩阵

全部 cell 使用 seed `42`，均 fresh training：

| Cell | Backbone | Flow head | 概念角色 |
| --- | --- | --- | --- |
| `E2-Q1__DF1-FH0` | `E2-Q1` | `DF1-FH0` | 当前两个默认项的 matched reference |
| `E2-Q1__DF1-FH4` | `E2-Q1` | `DF1-FH4` | mask-query additive + pure-RoPE head |
| `E2-Q0__DF1-FH0` | `E2-Q0` | `DF1-FH0` | pure-RoPE Qwen + additive-only head |
| `E2-Q0__DF1-FH4` | `E2-Q0` | `DF1-FH4` | **全链路 pure 2D RoPE 首选** |
| `E2b-Q0__DF1-FH0` | `E2b-Q0` | `DF1-FH0` | observed additive control |
| `E2b-Q0__DF1-FH4` | `E2b-Q0` | `DF1-FH4` | observed additive + pure-RoPE head |

`E2-Q0__DF1-FH4` 的核心吸引力不是预设它一定有最低 FID，而是职责最统一：
content/query 都不注入 additive absolute position，两个 attention stack 都用相同类型的
row/column 2D RoPE 表达空间关系。它没有随分辨率增长的 learned position table，也没有
stage/layout 分支，是最容易推广到更大 token grid 的候选。

## 4. 固定协议

六个 cell 除两个架构因素外必须完全一致：

- 数据：balanced ImageNet-100，固定 `115K/10K` split；
- base model：固定 digest 的 `Qwen3-0.6B-Base`；
- latent：`16×16` 个 KL16 token，每 token width `16`；
- 训练：`35,920` optimizer steps，global batch `256`，bf16，seed `42`；
- optimizer/LR/WSD schedule、EMA、augmentation 和 dataloader order 完全一致；
- image modules 和新增 special-token rows 使用 module-keyed paired initialization；
- flow head：`DF1`、depth `8`、width `1280`、heads `8`、MLP ratio `1.0`；
- 评测：EMA final checkpoint，10K validation samples，seed `42`；
- 生成：BF16 model forward，CFG `3.5`，constant schedule，100-step Heun，
  `parallel_rate=1`，`spatial_halton`；
- FID：共享 original-ImageNet Inception statistics；IS 使用固定 10 splits；
- VAE decode 与 flow integration 保持 FP32。

每个 run 必须绑定同一个不可变 runtime-source manifest。训练前和评测前各验证一次；
任何源码漂移都使该 run 失败，不能继续产出可比较 metrics。

## 5. 有效性门槛

一个 cell 只有同时满足以下条件才进入排序：

1. 自然完成到 step `35,920`，产生 final EMA `model.safetensors`；
2. config fingerprint、runtime-source manifest 和声明的 cell 完全一致；
3. training seed、dataloader shuffle seed 都为 `42`；
4. architecture report 精确匹配 backbone、`DF1` 与 `FH0/FH4`；
5. 10K 样本全部生成，generated latent finite rate 为 `1.0`；
6. official FID/IS protocol 为真，model BF16、VAE/integrator FP32；
7. 没有覆盖既有输出、手工删样本、`nan_to_num` 或失败 run 的部分结果。

若任一 cell 无效，先按相同 source/config 重试该 cell；矩阵不完整时不得选择 winner。

## 6. 预注册 selector

### 6.1 质量非劣集合

在六个有效 cell 中令：

- `FID_best` 为最低 FID；
- `IS_best` 为最高 IS。

质量非劣集合定义为同时满足：

\[
\mathrm{FID} \le \mathrm{FID}_{best}+0.50
\]

以及

\[
\mathrm{IS} \ge \mathrm{IS}_{best}-1.00.
\]

这两个 margin 与此前三-seed backbone 波动量级相当，避免让 seed-42 的微小数值差覆盖
明确的架构简洁性偏好。它们不是显著性声明。

### 6.2 概念与 scaling tie-break

在质量非劣集合内按以下顺序选择：

1. `E2-Q0__DF1-FH4`；
2. `E2-Q1__DF1-FH4`；
3. `E2-Q0__DF1-FH0`；
4. `E2-Q1__DF1-FH0`；
5. `E2b-Q0__DF1-FH4`；
6. `E2b-Q0__DF1-FH0`。

顺序先奖励“无 observed-content additive”，再奖励更少的 additive 注入点和 pure-RoPE
flow head。`E2b` 两项保留为重要 control，但 observed content 同时承载 content 与
absolute position，不作为长期默认的优先形态。

如果质量非劣集合意外为空，则只在 FID/IS Pareto frontier 中选择最低 FID；概念顺序仅
用于完全相同或数值舍入后相同的结果。

### 6.3 报告但不作为 gate

- generation samples/s、peak memory；
- final validation flow MSE；
- `FH4-FH0` 在每个 backbone 下的 paired effect；
- backbone effect 在 `FH0` 和 `FH4` 下是否改变符号；
- 旧 `E2-Q1 × DF1-FH0/FH4` 结果与 fresh anchor 的漂移。

吞吐必须报告，但本轮不把 FH4 当前实现的速度劣势设为淘汰条件；这里选择的是长期架构
接口，kernel 优化与 cache 实现可以后续独立处理。

## 7. 结果解释边界

- 这是 matched seed-42 system-level screen，不是多 seed 显著性结论；
- selector 可以决定本版默认架构，但论文中的稳定性表述仍需后续独立 confirmation；
- 本任务不会自动提交 seeds `43/44/45`，也不会自动启动 full ImageNet；
- 如果 winner 只以小于上述 margin 的数值优势胜出，最终表述应强调概念选择，而不是
  声称绝对质量最优。

## 8. 执行与产物

正式训练使用六个独立 `8×H100` Job、priority `4`、固定镜像
`docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1`。首选项目最多放两个
Job，其余在 live quota/availability 合法时路由到备用项目。每个 Job 内先训练再评测，
输出互不覆盖。

关键产物：

```text
configs/ablation/backbone_flow_head_joint/screen/*.yaml
output/backbone_flow_head_joint_ablation/evidence/runtime_source_manifest.json
output/backbone_flow_head_joint_ablation/evidence/matrix_manifest.json
output/<run>/hf_model-final-ema/model.safetensors
output/<run>/fid_is_cfg3p5_10k_ema/metrics.json
output/backbone_flow_head_joint_ablation/evidence/summary_seed42.json
```

终态后必须核验六份 checkpoint、metrics、fingerprint、样本数与 architecture report，
再运行 selector 并更新本 proposal 的状态与最终 runtime 默认值。
