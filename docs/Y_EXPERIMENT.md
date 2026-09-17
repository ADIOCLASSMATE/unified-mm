# Y：随机可见集合与逐 token flow 生成

Y 主方案采用随机可见图像 content、全 mask query 和共享逐位置 AdaLN MLP flow head；只监督未知图像 token。实现见 [modeling_y.py](../models/modeling_model/modeling_y.py)，64 卡配方见 [unified_y_33b_ascend64.yaml](../configs/selfless/unified_y_33b_ascend64.yaml)。整图 DiT 重建不属于 Y 主定义，质量和理解收益尚待正式实验评测。

## 目标与判断

保留 Z 的双流 backbone 和 I2T 完整图像双向 attention。T2I 中随机提供已知图像集合 V，未知集合 U 的 mask query 读取文本和 V，经 backbone 输出每个位置的条件 h_i，再由逐 token flow head 学习条件分布。

这与推理时“已有部分 clean latent，其余位置待生成”的状态直接对应。生成每轮只采样将要提交的一组 token，已生成值保持固定，cosine 控制新增数量。训练和生成的可见性、head 输入与目标一致；训练条件来自真实图像，推理条件来自历史生成，仍存在通常的条件分布差异，不能声称完全消除了 exposure bias。

Y 的首要假设是：由 backbone 承担图像上下文建模，head 专注单个连续 latent 的条件采样，能够兼顾 I2T 理解、分轮生成和计算效率。质量及理解收益均待验证。

## 与现有模型的关系

| 模型 | 图像 backbone 条件 | Head | 生成方式 |
| --- | --- | --- | --- |
| Z | 同图同 sigma，图像 query 不读取自身图像 content | 全双向 DiT | 固定一次 backbone 条件，整图联合去噪 |
| F（B 上的逐位置 head） | B 的随机序，每个目标读取更早 token，各位置的可见集合不同 | 逐位置 AdaLN MLP | 沿用 B 的顺序与生成状态机 |
| Y | 同一轮全部未知 query 读取同一个 V；V 内双向 | 逐位置 AdaLN MLP | 每轮刷新可见集合的 backbone 条件，只采样本轮新增位置 |

代码中 B 的原始 head 是 contextual Transformer；已有的纯逐 token head 位于 F / positionwise 分支。Y 复用 [PositionwiseFlowMLP / PositionwiseFlowLoss](../models/modeling_model/image_flow_loss_positionwise.py) 的结构和 flow 求解器，不能直接沿用 F 的随机序 attention 或 KV cache。

MAR 的原始逐 token head 使用 diffusion loss。Y 借鉴其“backbone 条件 + 逐 token 条件采样”的分工，保留项目现有 rectified flow。MAR 的 MAE decoder 允许 mask 位置彼此 self-attention；Y 保留 query 从 content 读取 K/V 的两流结构，不引入 query-to-query attention，因此并非完整复刻 MAR。

## Backbone attention

设目标图像位置集合为 I，可见集合为 V，未知集合 U = I \ V。P 为合法前置上下文，包括原有逻辑次序中的 BOI/EOI；未来文本和其他 packed segment 不属于 P。

| 路径 | 输入 | 允许读取 |
| --- | --- | --- |
| T2I content V | 已知图像 latent 投影 | P 和全部 V，V 内双向并包含自身 |
| T2I content U | 无效槽或不含真实值的占位输入 | 不作为任何有效位置可读的 K/V |
| T2I image query | 所有图像位置仍为 learned mask + 原有位置编码 | P 和全部 V |
| T2I flow loss | 仅 U 的 query hidden 作为条件 | 仅监督 U，不重建 V |
| I2T image content | 完整图像 latent | 图像块内双向，保留原有跨块次序 |
| I2T caption query | 现有同位置 mask query | 完整图像与更早文本，不读目标及未来文本 |
| 纯文本 | 现有两流输入 | 原有 query 严格早于、content 包含自身的规则 |

图像 query 可在全网格上计算，但只取 U 的输出作为训练条件。V 的 query 即使读取了同位置图像内容，也不参与图像 loss，不引入复制目标。只监督 U 仍可经 query -> content attention 将梯度传回 V 的投影和 backbone。

### Sigma 与有效位置

相同 content sigma 可以实现 V 内双向；但当前 `get_selfless_mask` 使用 `sigma_kv < sigma_q` 或 `<=`，直接设 `sigma=-1` 反而可能使该位置被所有后续位置读取。因此必须显式区分 content 有效位置和 query 有效位置。

对每幅 T2I 目标图分配逻辑 rank r，在不越过后续文本的前提下：

```text
content_sigma[V] = r
content_valid[V] = True
content_valid[U] = False
query_sigma[I] = r + delta       # delta > 0
query_valid[I] = True

content_allowed[q,k] = same_segment(q,k)
                       & content_valid[q] & content_valid[k]
                       & (content_sigma[k] <= content_sigma[q])
query_allowed[q,k] = same_segment(q,k)
                     & query_valid[q] & content_valid[k]
                     & (content_sigma[k] < query_sigma[q])
```

也可直接构造等价的布尔 attention 真值表。r 与 delta 必须按 block 分配，不能盲目在原有整数 sigma 上加一而与后续文本碰撞。`-1` 可以作为存储哨兵，但不是 attention 规则本身。

不能将 U 的 segment_id 设为 -1 并沿用两流共用的有效性判断，否则 U 的 query 也会被删除。所有 U query 必须保持有效；无效 content 行应移除或采用数值安全的占位计算，输出不可作为有效 K/V，不能让全屏蔽行产生 NaN。

当前 Z 的 `joint_image_sigma` 会覆盖集合 rank，生成还会清除全部目标图像条件；Y 必须有独立的 attention 和生成分支。I2T 分支继续采用完整图像可见性。

### 多层信息隔离

从第一层起，V content 不能读取 U 的真实 latent；P、BOI/EOI、后续文本也不能成为间接中转。未知真实值只进入其独立 flow head 的标准加噪输入和监督目标。

随机遮挡只作用于 T2I 目标图。其他已观测 context 图像和 I2T 图像保持完整。按显式任务元数据路由，不以 train/eval 状态猜测任务。

## T2I 训练

一次训练抽一个 V，不展开多轮生成。首版建议的集合分布如下，属于可调实验参数，并非 MAR 原始训练分布：

- 10% 样本 V 为空，明确训练纯文本条件下的生成起点。
- 90% 样本抽 s ~ Uniform[0,1)，设未知数量 m = clip(floor(N cos(pi s / 2)), 1, N)，随机选择其余 N-m 个位置作为 V。
- 始终保留至少一个未知位置；完整图像理解由 I2T 覆盖。位置采样保留原有二维空间坐标，不按生成顺序重排位置编码。

在不同已知比例下，所有 U 共享同一个可见集合：

```text
h = backbone(P, x_V, image_mask_queries, attention(V))
z_i = condition_proj(h_i)                       # i in U

epsilon_i ~ Normal(0, I)
t_i ~ existing_flow_time_distribution
x_t_i = (1 - t_i) * epsilon_i + t_i * x_i
v_pred_i = MLP(x_t_i, t_i, z_i)
v_target_i = x_i - epsilon_i

L_image_per_image = mean_{i in U, channels} (v_pred_i - v_target_i)^2
L_image = mean_images(L_image_per_image)
```

全部位置使用同一个 MLP 的参数，head 无跨 token attention。位置和图像上下文已经编码在 z_i 中。由于 token 独立建模，训练无需整图共享 t；首版沿用已有 positionwise head 的逐 token 时间采样。RF4 共享一次集合与 backbone 条件，四份噪声/时间独立。

可以只 gather U 后执行 head，或者计算全网格并屏蔽 V 的 loss；前者节约 head 工作量。按每图 U 均值再平均，避免随机集合较大的图像隐式占更高权重。已有 PositionwiseFlowLoss 的 masked reduction 是全 batch 有效 token 均值，Y 需明确适配该归一化，不能直接声称二者相同。

I2T 保持完整图像输入和 caption CE，按当前任务定义不增加图像 flow loss；ClimbMix 路径、任务权重和 RF4 沿用基线。图像输入微噪声增强与 Z/F 对齐，验证生成关闭；该增强不表示已学会语义纠错。

训练集合通过训练进程的 torch RNG 抽样，随训练推进改变；沿用现有 checkpoint 的随机状态保存与恢复。模型支持显式 `y_visible_mask`，用于固定可见比例的依赖检查和对照。训练中定性生成按 prompt seed 固定 reveal order 和逐位置初始噪声，不依赖当时的全局 RNG。

## 分轮生成

设总 token 数为 N，外层生成轮数为 K，内层 flow 步数为 S。预先抽随机位置顺序，V_0 为空。第 k 轮完成后的剩余数量为：

```text
M_k = floor(N * cos(pi * k / (2K)))
本轮新增数量 = M_{k-1} - M_k
```

采用取整保护：非末轮至少推进一个位置，末轮填满余下位置，首版限制 1 <= K <= N。随机顺序决定选谁，cosine 决定选多少。

每轮执行：

1. 将当前 V 的已生成 latent 输入 content；剩余位置的图像 query 为 mask，重新计算 backbone 条件。
2. 由 cosine 与固定随机顺序选出本轮新增集合 A，A 是 U 的子集。
3. 只取 z_A，从独立噪声开始，使用共享 MLP 和 S 步 flow solver，并行采样 A。
4. 将采样结果提交到 V，旧 V 的 latent 数值保持不变；未选中的 U 不做 head 采样。
5. 重复直到所有位置完成，再用 VAE 解码。

训练可监督全部 U，生成只采样 A，不构成 head 输入不一致：给定 h 后各位置的条件采样彼此独立，某个位置无需其他未知位置的 noisy latent。为避免歧义，“逐 token head”不表示“一轮只能生成一个 token”，A 内可以全并行。

一次生成中，各位置只提交一次。若每轮使用同样的 Heun S，head 总处理量约为 2SN 个 token 次速度求值，而不是整图 head 每轮处理全 N 的 2SKN；这不包含 CFG 分支翻倍、backbone、padding 与运行时开销。实际调用次数仍约为 2SK 次，需同时报告处理 token 数和墙钟时间。

Backbone 通常前向 K 次。V 内全双向使新增 token 改变旧 V 的隐状态，不能直接复用 B/F 的增量图像 KV；每轮重新编码 V，而每轮内部所有 flow 步复用同一个 h。安全的文本前缀缓存可作为后续优化，先验证无缓存路径。

CFG 两分支使用相同 V，分别保留/去除文本条件。无文本分支要从 content 编码阶段去除文本影响，防止经 V 隐状态间接读 prompt；V 为空也须保持数值安全。

### 轮数的能力边界

同一轮的新增 token 在给定文本和 V 后条件独立：

```text
p(x_A | P, x_V) = product_{i in A} p_i(x_i | h_i(P, x_V))
```

更多轮次通过逐轮更新 V 引入依赖。K=1 在接口上成立，但此时所有 token 仅凭文本条件独立采样，不保证整图一致性或质量接近多轮生成。Cosine 前期小组、后期大组的调度值得验证，不能预设增加轮数必然改善所有指标。

本方案保留已生成 token，不具有自动修改旧 token 的能力。若以后研究纠错，可单独设计将一部分旧 token 重新遮挡后再生成的实验，并记录额外计算；不把它混入首版 Y。

## 联合 T2I 与 I2T 的判断

Y 的分工使图像上下文主要通过 backbone 的双向 content 编码进入 h；I2T 文本 query 同样依赖这套图像编码，因此在结构上更直接共享图像条件表征。

这不保证 I2T 一定胜过 DiT。逐位置 head 可能要求 backbone 承担更多语义和布局信息，也可能增加图像训练对文本参数的压力。必须同时看 I2T CE、ImageNet 分类、SugarCrepe/ARO 及纯文本指标。

只监督 U 可以避免将已知图像直接重建视为生成进步，又保留对 V content 的梯度。它符合“保留已生成结果并扩展”的推理目标；无需为重建已知位置分配 flow 监督预算。

整图 DiT 或 DiT 仅未知区域条件去噪也能正确对齐训练/推理，不能把选择 MLP 的理由写成“DiT 必然不对齐”。Y 选择 MLP 的具体理由是：支持任意新增组的独立采样、无需整图 head 反复计算、与现有逐位置模块兼容，并使 backbone/head 分工明确。

## 实验与评测

首版实现 Y，保留 Z/F 为参考。Z/F 与 Y 同时存在 head 或 attention 差异，不能直接把结果归因于单个因素。若继续检验 head 的价值，再增加保持 Y backbone、仅在 U 去噪并固定 V 的 DiT 对照；它不属于首版 Y。

沿用 [Z 33B 配置](../configs/selfless/unified_z_33b_ascend64.yaml) 的数据、任务曝光、基座 step 0 初始化、优化器、验证与 EMA。head 复用 F 的 8 层、宽 1936 设置，有 163,828,208 参数，总模型 760,945,136 参数（词表特殊 token 扩展前），head 比 B 低 0.149%。默认 8 轮 reveal、每轮 Heun10，验证使用固定 prompt 的 raw 生成及 EMA 下游评分。

| 维度 | 必须报告 |
| --- | --- |
| 训练 | 已知比例分布、每图 U 数、未知位置 flow MSE、各任务 CE、有效监督 token 与图像曝光 |
| 生成 | K in {1,4,8,16,32,64} 的质量/速度曲线；初筛后再做足量指标 |
| 求解器 | 外层 K、内层 S、Euler/Heun、CFG、backbone/head 调用次数 |
| 成本 | head 实际处理 token 数、backbone 时间、head 时间、显存、端到端耗时 |
| 理解 | 固定完整图像条件下的 I2T CE、分类和已有图文理解协议 |
| 文本 | 沿用当前纯文本评测，观察联合训练的能力保留 |
| 可复现 | checkpoint/raw/EMA、prompt、位置顺序、集合与噪声 seed |

固定 prompt 与位置顺序，建议按图像位置预生成噪声 bank，使不同 K 使用同一位置的初始噪声。保持 S 相同可近似固定 MLP token 次工作量，但 backbone 计算随 K 增加；不能把不同 K 直接称为总计算等预算。

## 实现与验收要求

CPU 测试见 [test_y.py](../tests/test_y.py)，设备验收复用共享训练、恢复、raw/EMA 及验证流程：

- Attention 真值表：V content 全双向；U content 对任何有效 query/content 均不可读；全部 U query 可读 V，未来文本/其他 packed segment 不可读。
- 多层依赖：修改 clean x_U 不改变 backbone 条件 h_U；修改 x_V 能改变 h_U；只监督 U 时 V 的输入投影及 backbone 仍得到梯度。
- MLP 独立性：固定 h，改变一个位置的 noisy latent 不影响其他位置输出；整批采样与子集采样在相同条件/噪声下数值一致。
- Loss：只监督 U，每图归一化明确；V=空、单 token、N-1 token 均有效，无零分母、NaN 或未知信息旁路。
- I2T：完整图像 attention 与 Z 对齐；未来 caption token 不影响更早预测，文本和图像理解缓存与完整前向一致。
- 生成：K=1、K=N 及中间值都无重复/遗漏提交；旧 latent 保持不变；非新增位置不调用 head。
- 缓存与 CFG：每轮更新图像条件，无陈旧双向图像 KV；CFG 无文本分支没有经 V 泄漏的 prompt 信息。
- RF4、AMP、checkpoint/EMA、恢复与验证种子保持协议一致。Y 使用独立架构身份与输出目录，避免被旧 F/Z 分支静默加载。

## 运行入口

```bash
# CPU 模型与协议检查
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python scripts/validate_z.py --experiment y
.venv/bin/python scripts/launch_short_ablation.py --arm y --smoke --dry-run

# 固定 16 卡开发机上的完整验收；label 必须唯一
bash script/selfless/pretraining_y_ascend64.sh --smoke-suite --validation \
  --label <unique-label> --output-dir output/experiments/y/<smoke-directory>

# 正式四节点 Job 入口
bash script/selfless/pretraining_y_ascend64.sh
```

正式项目为“随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架”，配置协议位于 [y_protocol.py](../utils/y_protocol.py)。正式输出为 `output/unified-y-0p6b-33b-imagenet-split-s42-r1`。

Python 生成接口通过 `model.generate("t2i", ..., reveal_steps=K, flow_num_steps=S)` 分别控制外层和内层步数。FID/IS 评测入口支持 `--reveal_steps K --sampling_steps S --strategies random`；不向旧模型传递 Y 专用的 reveal 参数。

### 2026-09-17 设备验收与正式提交

CPU 全仓检查 1126 项通过、2 项跳过；最后的空集合 loss 与采样默认值补充检查 19 项通过，ruff 与 diff 检查通过。[16 卡设备报告](../output/experiments/y/smoke-20260917-r1/report.json)确认训练至 step 5、恢复至 step 6，两轮完整验证（step 2/4）及 raw/EMA 重载后图像、文本和 I2T 生成全部通过。逐步图像生成每次执行 8 次 backbone、160 次 head 前向。开发 Notebook 已停机并保留对象。

正式 Job `umm-y-33b-64-0917-r1` 已提交到上述随机序语言建模项目，4 节点 × 16 张 910B、31,800 updates，从基座 step 0 开始。[提交回执](../output/experiments/y/formal-20260917-r1/submission.json)、[运行记录](../output/experiments/y/formal-20260917-r1/launch_record.json)和冻结源码保存在同一目录；提交成功不代表训练已经完成，平台状态以实时查询为准。

11:51 UTC 已确认全部 4 个 worker 为 Running，64 卡训练到 step 10，loss 0.6423、图像 flow loss 1.9936。四节点资产检查均通过，启动日志无异常退出；[rank 0 训练日志](../output/unified-y-0p6b-33b-imagenet-split-s42-r1/prelaunch_audit/node-0/training.log)持续记录后续进度。

## Step-2000 采样筛选与 masking 对照

`script/selfless/screen_y_sampling_ascend16.sh --model-source <checkpoint-2000> --output-dir <fresh-dir>` 使用 checkpoint 内的 EMA，在 16 卡上筛选 K={8,20,32,64} × reveal CFG={constant,linear}。固定 S=10/Heun、temperature=1、CFG 上限=3.5。每类固定取两张 ImageNet-val 图像，共 2000 张；IS 使用两个分层 split。初始噪声与 reveal 顺序按样本独立设种子，跨配置保持一致。FID 对照现有 50k 原图统计，但标记为小样本筛选，不能直接当成正式 FID50k。报告同时保存 K/S、CFG 分支数、backbone/head 调用次数、head token 求值数和耗时。

`reveal_cfg_schedule=linear` 按 MAR 的下一轮已补全比例提升 CFG；`flow_cfg_schedule` 仍单独控制 ODE 时间调度，筛选时保持 constant。

`script/selfless/pretraining_y_marmask_ascend64.sh` 启动独立的 Y-MARmask。其 [配置](../configs/selfless/unified_y_marmask_33b_ascend64.yaml)只改变 mask 分布：N(1,0.25²) 截断到 [0.7,1]，ceil 得到 token 数，取消额外 10% 全 mask 混合。其余模型、数据、随机 seed、batch、优化器、EMA 和 31,800-step LR 曲线均保持 Y 的设置；`stop_after_steps=2000` 用于等步数对照，不能把 LR 日程压缩到 2000 步。自然取整仍会产生少量全 mask 样本。

Flow 统计采集已改为在指定 optimizer step 的任务 microbatch 上开启，避免只在 GA4 最后的 I2T microbatch 采集而错过 T2I。新增日志为 `train/flow/unknown_fraction`、`train/flow/mask_below_70_fraction` 和 `train/flow/full_mask_fraction`。此诊断改动不改变梯度，已启动的原 Y 使用其冻结源码继续运行。

### 2026-09-17 筛选结果

完整 step-2000 checkpoint 已独立保留于 `output/experiments/y/step2000-preserved/checkpoint-2000`，避免原训练的 checkpoint 轮换影响复查。本次使用 EMA；[完整结果](../output/evaluation/y-step2000-screen-20260917-r2/summary.json)与[结果表](../output/evaluation/y-step2000-screen-20260917-r2/summary.md)保存每组指标和调用成本。

| K | Constant FID | Linear FID | Constant 生成秒数（2000 张 / 16 卡） |
| --- | --- | --- | --- |
| 8 | 347.36 | 349.18 | 23.16 |
| 20 | 347.44 | 348.72 | 48.99 |
| 32 | 347.23 | 348.56 | 74.89 |
| 64 | 347.28 | 348.67 | 141.07 |

这是早期 checkpoint 的小样本筛选。增加 K 尚未带来明显收益，constant 的数值略好于 linear；暂保留 K=8 + constant 作为低成本设置，不据此推断收敛后的最佳配置。K=64 的生成耗时约为 K=8 的 6 倍。以上均非正式 FID50k。

对照 Job `umm-y-marmask-2k-64-0917-r1` 已按顺序在筛选成功后提交，使用 4×16 张 910B，从基座训练到 step 2000。训练冻结源码对应 commit `3b208fa`；[提交回执](../output/experiments/y/ablation-20260917-r1/marmask-submission.json)和[源码记录](../output/experiments/y/ablation-20260917-r1/source_record.json)保留复现信息。原 Y 训练不停止。设备 smoke 已验证训练、恢复、raw/EMA 重载及生成通过；正式训练状态和 mask 统计见其训练日志。

## 代码与研究依据

- [Z backbone 与生成](../models/modeling_model/modeling_joint_dit.py)。
- [当前 sigma attention](../utils/utils.py) 的 get_selfless_mask。
- [两流 backbone](../models/modeling_model/modeling_selfless_flow.py)。
- [已有逐位置 flow head](../models/modeling_model/image_flow_loss_positionwise.py)与[F 模型接入](../models/modeling_model/modeling_selfless_flow_positionwise_on_b.py)。
- [T2I/I2T 数据定义](../utils/dataset_imagenet_flow_cache.py)。
- [MAR 论文](https://arxiv.org/abs/2406.11838)与[本地官方实现](../public/code/mar/models/mar.py)：参考连续 token 的条件采样分工与 cosine reveal；Y 仍使用项目自己的两流 backbone 和 rectified flow。
