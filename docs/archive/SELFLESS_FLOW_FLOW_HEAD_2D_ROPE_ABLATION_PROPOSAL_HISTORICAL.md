# Selfless-Flow Contextual Flow Head：2D RoPE 位置编码消融（历史归档）

Status: **completed and closed**; written 2026-07-21, protocol revised and
closed 2026-07-24. The matched seed-42 `FH0--FH4` screen is complete. No
multi-seed confirmation is planned, and this document no longer authorizes jobs.

## 0. 最终结果与决定

| ID | 变化 | FID ↓ | IS ↑ | Final-step flow MSE ↓ |
| --- | --- | ---: | ---: | ---: |
| **FH0** | additive query + additive context；无 flow-head RoPE | **25.0669** | **61.5120** | 0.688217 |
| FH1 | FH0 + row/column 2D RoPE | 25.2985 | 60.3427 | 0.687393 |
| FH2 | FH0 去掉 context additive | 27.3799 | 57.4373 | 0.688646 |
| FH3 | FH2 + row/column 2D RoPE | 25.4672 | 60.7045 | 0.690614 |
| FH4 | FH3 再去掉 query additive | 26.1535 | 60.1577 | **0.686453** |

`FH0` 是 FID/IS 的唯一 Pareto point。关键 FID contrast 为：

- `FH1-FH0 = +0.2316`：保留 additive paths 时，2D RoPE 没有改善；
- `FH2-FH0 = +2.3131`：直接移除 context additive 明显恶化；
- `FH3-FH2 = -1.9127`：RoPE 能修复一部分被移除的空间信息，但仍落后 FH0；
- `FH3-FH1 = +0.1687`：使用 RoPE 时，移除 context additive 仍没有收益；
- `FH4-FH3 = +0.6863`：再移除 query absolute position 继续恶化。

最终 flow MSE 与生成指标不一致：FH4 的 MSE 最低但 FID/IS 更差，因此不能用
velocity MSE 为这些 position variants 排名。该 screen 只是一组 seed-42 机制证据，
不支持跨 seed 稳定性主张；但没有任何候选在主指标上优于 FH0，故不存在值得自动进入
confirmation 的 RoPE 方案。

**最终决定：保留 FH0 position contract，停止 FH1--FH4，不运行 seeds
43/44/45。** 后续 head 搜索转向信息交互架构，而不是继续增加 position encoding。
机器可读 selector 中的 `FH0/FH1/FH3` 是 screen 前规则产生的审计记录，不是继续提交
confirmation 的授权。

以下正文保留原预注册设计，用于审计当时的实验问题、实现边界和决策规则。

## 1. 研究问题与执行位置

本 proposal 回答：当 Qwen backbone 已使用 row/column 2D RoPE 时，contextual flow head
的 noisy-query-to-clean-context cross-attention 是否也应使用 row/column 2D RoPE，以及
flow head 中 absolute query identity 与 relative spatial relation 应如何分工？

执行顺序固定为：

```text
image-backbone ablation archive
    -> choose one retained backbone
    -> 本 flow-head 2D-RoPE ablation
    -> make a provisional flow-head position choice
    -> caption × initialization ablation
```

本实验不能与 image backbone 因子同时变化。启动前必须从
[`SELFLESS_FLOW_IMAGE_TOKEN_EMBEDDER_ABLATION_PROPOSAL.md`](../SELFLESS_FLOW_IMAGE_TOKEN_EMBEDDER_ABLATION_PROPOSAL.md)
选择并冻结一个离散的 `image_backbone_variant`：

- `E2-Q1`：默认和主实验；
- `E2-Q0`：可选的无 additive control；
- `E2b-Q0`：可选的 observed-position control。

三个 variant 都固定 direct latent grid、row/column 2D backbone RoPE、无 stage、
无 S2D。主矩阵只在 `E2-Q1` 上运行；若要检验 flow-head 结论对 backbone 的稳健性，
可以把**同一套**选定的 flow-head systems 复制到另外两个 retained variants，但必须
作为独立 secondary study 报告。不得恢复旧的独立 stage/observed/mask/RoPE/layout
开关，也不得引入第四种 backbone。历史 E2-Q1 confirmation 结果直接复用，不为选择
本实验 backbone 而重训；本 proposal 自身改变 flow head，因此其正式 cells 仍需按新
协议训练。

## 2. 当前 contextual flow-head 契约

当前实现位于 `models/modeling_model/image_flow_loss.py`。固定架构为：

- `ContextualFlowTransformerHead`；
- depth 8、width 1280、8 heads、head dimension 160；
- MLP ratio 1.0、zero-initialized AdaLN residual gates；
- noisy latent query `x_t` 与 visible clean-latent context `z_j` 先投影到 width 1280；
- query 和 context 都加同一 fixed 2D sin/cos position embedding；
- 每个 block 只有 query-to-context cross-attention 和 MLP，没有 query self-attention；
- cross-attention Q/K/V 目前都不使用 RoPE；
- K/V 由 clean context 构造，并可在同一次 ODE solve 中缓存；
- context set 继续严格满足 `sigma_j < sigma_i`；
- flow timestep `t_i` 与 outer reveal order `sigma_i` 是不同变量。

对 query position `p_i=(r_i,c_i)` 和 context position `p_j=(r_j,c_j)`，当前 head 为：

\[
\begin{aligned}
h_i^q &= W_x x_{t,i}+P_{xy}(p_i),\\
h_j^c &= W_x z_j+P_{xy}(p_j),\\
q_i &= W_q\operatorname{LN}(h_i^q),\\
k_j &= W_k\operatorname{LN}(h_j^c),\\
v_j &= W_v\operatorname{LN}(h_j^c).
\end{aligned}
\]

因此当前 baseline 把绝对坐标直接混入 query、key 和 value 的输入 hidden，并没有显式
建模二维 relative phase。

## 3. 位置因素的原子分解

定义三个二值开关：

- `A_q`：是否向 projected noisy query hidden 加 fixed additive `P_xy`；
- `A_c`：是否向 projected clean-context hidden 加 fixed additive `P_xy`；
- `R_f`：是否对 flow-head cross-attention 的 projected Q/K 使用 row/column 2D RoPE。

统一公式为：

\[
\begin{aligned}
h_i^q &= W_xx_{t,i}+\mathbb 1_{A_q}P_{xy}(p_i),\\
h_j^c &= W_xz_j+\mathbb 1_{A_c}P_{xy}(p_j),\\
\hat q_i &= W_q\operatorname{LN}(h_i^q),\\
\hat k_j &= W_k\operatorname{LN}(h_j^c),\\
q_i &= R_f(p_i)\hat q_i,\\
k_j &= R_f(p_j)\hat k_j,\\
v_j &= W_v\operatorname{LN}(h_j^c).
\end{aligned}
\]

其中 `R_f=0` 表示 identity。任何模式下都不旋转 V，不旋转 residual hidden，也不把
position 加到 timestep/condition modulation `y=t_embed(t_i)+cond_embed(c_i)`。

这三个开关分别对应：

| 信息 | 候选路径 | 作用 |
| --- | --- | --- |
| noisy query 的 absolute identity | `A_q P_xy` | 指明当前 velocity 对应哪个 latent site |
| clean context 的 absolute content-position mixing | `A_c P_xy` | 把 context 坐标直接混入 K/V 输入 |
| query-context relative 2D relation | flow-head Q/K 2D RoPE | 让 attention score依赖行列位移 |

## 4. 预注册假设

- **H1：relative geometry。** 在 `A_c` 固定时，`R_f=1` 改善 flow head 对可见 clean
  latents 的空间利用；
- **H2：context content-position decoupling。** 当 `R_f=1` 时移除 `A_c`，让 V 保持纯
  clean-latent content，可能优于 query/context 都 additive；
- **H3：query identity。** 即使使用 2D RoPE，保留 `A_q=1` 仍可能必要，尤其在零个或
  一个 visible context token 的 early reveal 阶段；
- **H4：backbone/head consistency is not guaranteed。** backbone 2D RoPE 有效不代表
  flow head 2D RoPE 必然有效，因为前者做 mixed text-image sequence modeling，后者只做
  noisy-query-to-clean-image-context cross-attention；
- **H5：hybrid candidate。** `A_q=1,A_c=0,R_f=1` 最符合“absolute query identity +
  relative relation + clean value path”的信息分工。

## 5. 正式实验矩阵

核心矩阵使用五个 systems：

| ID | `A_q` | `A_c` | `R_f` | 作用 |
| --- | ---: | ---: | ---: | --- |
| `FH0` | 1 | 1 | 0 | 当前 additive-only baseline |
| `FH1` | 1 | 1 | 1 | 在当前 additive paths 上原子增加 2D RoPE |
| `FH2` | 1 | 0 | 0 | 只移除 context additive 的原子 control |
| `FH3` | 1 | 0 | 1 | proposed hybrid：query absolute + Q/K relative |
| `FH4` | 0 | 0 | 1 | flow-head pure-RoPE stress test |

`FH0/FH1/FH2/FH3` 在固定 `A_q=1` 下构成 `A_c × R_f` 的完整 `2×2`：

\[
\begin{aligned}
\Delta_{R_f}(A_c=1)&=FID(FH1)-FID(FH0),\\
\Delta_{R_f}(A_c=0)&=FID(FH3)-FID(FH2),\\
\Delta_{A_c}(R_f=0)&=FID(FH2)-FID(FH0),\\
\Delta_{A_c}(R_f=1)&=FID(FH3)-FID(FH1).
\end{aligned}
\]

`FH4-FH3` 在 `A_c=0,R_f=1` 下只移除 flow-head query additive position，用于测试
absolute query identity。没有运行全部 `2^3` 八个组合，因此结论必须限制在上述 contrasts，
不能声称三个因素的全局 factorial optimum。

特别注意：`FH4` 只表示 **flow head 内部** 不使用 additive position。冻结的 backbone
condition `c_i` 仍可能携带 absolute/spatial 信息；如果 image-embedder winner 保留
mask-query `P_xy`，`FH4` 更不能被称为 end-to-end pure RoPE。

## 6. Row/column 2D RoPE 的精确定义

### 6.1 Dimension layout

当前 head dimension 为 `1280/8=160`。每个 head 固定分配：

- 80 dimensions 给 row rotary pairs；
- 80 dimensions 给 column rotary pairs；
- 每个 axis 的维数均为偶数；
- 频率构造、interleaving convention 与冻结 backbone 的 image-side row/column helper
  保持一致，坐标直接取 latent grid 的 row/column index；
- 不引入 partial rotary ratio、learned frequency、NTK scaling 或第三个 axis。

如果 future flow-head width/head count 改变导致 head dimension 不满足合法拆分，config
构造直接失败，不能静默少旋转一部分维度。

### 6.2 Operation order

每个 `ContextualFlowBlock` 中固定顺序为：

```text
query hidden -> query norm -> cross_q projection -> split heads -> rotate by query (row,col)
context hidden -> context norm -> cross_k projection -> split heads -> rotate by context (row,col)
context hidden -> context norm -> cross_v projection -> split heads -> no rotation
```

RoPE 在 linear projection 和 head split 之后、attention score 之前应用。FlexAttention 与
SDPA fallback 必须接收完全相同的 rotated Q/K。

### 6.3 Context cache

inference K/V cache 的 contract 为：

- K 在写入 cache 前按各 context local position 旋转；
- V 保持未旋转；
- cache metadata 记录 flow position mode、grid/layout、dtype 和 local position hash；
- query Q 在每次 ODE evaluation 中按 query position 旋转；
- 同一次 ODE solve 可复用 rotated K/V，不得重复旋转已经 cached 的 K；
- cache mode/digest 不匹配时 fail fast，不能把 additive-only cache 传给 RoPE head。

### 6.4 Empty/single context

- 零 context 时继续跳过 cross-attention；RoPE 不凭空提供 context；
- 单 context 时 attention weight 仍恒为 1，2D relative phase不能通过 softmax weight区分
  query；
- 因此 `FH4` 的 early-stage 风险必须单独报告，不能只给 aggregate FID；
- flow-head condition `c_i`、noisy latent `x_t` 和 backbone 可能仍携带位置相关信息，故
  `FH4` 不是数学上的必然失败。

## 7. 固定项与禁止混入的变化

所有 `FH*` runs 必须固定：

- `image_backbone_variant=E2-Q1`（主矩阵）以及同一个 frozen architecture manifest；
- 同一个 Qwen3-0.6B-Base initialization；
- 同一个 ImageNet-100 train/validation membership；
- direct `256×16` image latent layout；
- flow head depth、width、heads、MLP ratio、dropout、gating 和 parameter initialization；
- flow time sampling、uniform mix、loss reduction、context mask 和 `image_flow_batch_mul`；
- optimizer、module-wise LR、global batch、steps、EMA 和 seed 42；
- sampler、Heun NFE、CFG、temperature、parallel rate 和 evaluator prompts/noise；
- backbone variant。尤其不能在同一个 `FH*` cell 内把 `E2-Q1` 改成 `Q=0`。

每个 run 都端到端重新训练相同架构中的全部预注册 trainable modules。“固定 flow head”在
这里指除 position mode 外的 architecture/hyperparameters 固定，不是跨 run 复用一套
flow-head weights。

不允许同时：

- 增加 flow-head self-attention；
- 改用 token-only MLP head；
- 改 depth/width/head count；
- 旋转 V 或 residual stream；
- 改 visible context set 或把未来 clean latents 泄漏给 query；
- 改 flow timestep/reveal-stage 定义；
- 引入 learnable position gain、relative bias 或 local attention window。
- 恢复 stage/S2D/sequence-1D backbone 接口，或拼装 retained set 之外的 backbone。

## 8. 训练与评测协议

### 8.1 Single-seed mechanism screen

1. `FH0--FH4` 全部使用 seed 42 做 screen；
2. 五个 systems 必须共享 initialization schema、data order、训练预算和 evaluator
   protocol，以 paired contrasts 解释 `A_q/A_c/R_f` 的机制影响；
3. 当前阶段不运行 seeds `{43,44,45}`，不生成新的 confirmation Job，也不以 legacy
   selector artifact 作为继续提交 confirmation 的授权；
4. 已停止的 confirmation Job、其 config 和 partial checkpoints 只作为可恢复的执行记录，
   不进入本 proposal 的结果汇总、排名或 architecture decision；
5. seed 42 只支持 provisional choice。小幅 FID/IS 差异不得表述为稳定 winner；没有明确且
   机制一致的收益时，保留实现更简单的 `FH0` 作为当前默认值；
6. 如未来论文主张、full-scale 训练或分辨率外推结论需要统计稳健性，必须另行批准 matched
   multi-seed protocol，不从本 proposal 自动续跑。

如果 `FH0` 与已完成的 image-embedder winner run 在 config、code、seed、initialization、
training/evaluator manifest 和 implementation contract 上完全一致，可以通过 artifact
digest 复用；任一字段不同都必须重训 `FH0`。

### 8.2 Primary metrics

- fixed-CFG 10K FID（primary）；
- IS（guardrail/Pareto）；
- train/validation flow velocity MSE；
- train images/s、sampling wall time、peak memory 和 K/V-cache memory；
- 参数量应完全相同；若不同则 implementation invalid。

### 8.3 Mechanism diagnostics

按 visible context count 分 bucket：

```text
0, 1, 2--4, 5--16, 17--64, 65+
```

每个 bucket 报告：

- flow loss/velocity MSE；
- cross-attention gate magnitude；
- attention entropy 与 query-context spatial distance；
- conditional/unconditional velocity delta；
- generated latent RMS、finite rate 和最终 image quality diagnostics。

同时按 early/middle/late reveal stage 报告。若 `FH3` 的收益只在 context 较多时出现、而
`FH4` 只在 0/1 context 显著恶化，这将直接支持 absolute identity 与 relative relation
的分工假设。

## 9. 建议配置接口

以下仅描述未来实现，本轮不修改代码：

```yaml
model:
  image_flow_head_arch: contextual
  image_flow_query_position_mode: additive_2d    # additive_2d | none
  image_flow_context_position_mode: additive_2d  # additive_2d | none
  image_flow_rope_mode: none                     # none | row_col_2d
  image_flow_rope_axis_dims: [80, 80]
  image_flow_rope_rotate_value: false
```

validator 必须：

- 验证 axis dims 之和等于 head dimension 且各自为偶数；
- 把三个 position flags 写入 config fingerprint、run slug、checkpoint 与 metrics；
- 拒绝 `rotate_value=true`；
- 拒绝在主矩阵中改变 `image_backbone_variant=E2-Q1`；
- 对 `FH0--FH4` 做精确 ID-to-config mapping，禁止名称与实际 flags 不一致。

## 10. Repo 实施落点（未来工作，不在本轮实现）

### `models/modeling_model/image_flow_loss.py`

- `ContextualFlowTransformerHead` 分离 query/context additive-position modes；
- `ContextualFlowBlock.prepare_cross_cache` 接收 context positions，并在 K projection 后
  选择性应用 2D RoPE；
- `_cross_attention` 在 Q projection 后用 query positions 旋转 Q；
- cached K 是 already-rotated K，V 永远不旋转；
- FlexAttention 与 SDPA fallback 共享同一 rotary helper 和 cache contract；
- 空 context 与 all-masked rows 沿用当前 finite-safe behavior。

### `models/modeling_model/image_position_utils.py`

- 提供可复用、无参数的 per-head row/column rotary frequency/coordinate builder；
- 明确 backbone hidden-size rotary 与 flow-head head-dim rotary 的 shape contract；
- buffer 使用 FP32 确定性构造，再转换为实际 Q/K dtype；
- checkpoint materialization 后按 config 确定性重建非 persistent buffers。

### training/evaluator/summarizer

- 从 image spans 传递 query/context local positions；
- 记录 cache position-mode metadata 与 digest；
- 汇总 `A_q/A_c/R_f` contrasts、context-count buckets 和 stage buckets；
- 任何 NaN/Inf 都 fail fast 并定位首个产生位置，不能 clamp、`nan_to_num` 或跳过样本。

## 11. 必须通过的测试

### Numerical geometry

- 同一 position 下 Q/K dot product 在共同 rotary 后保持不变；
- 只改变 row 时 column phase 不变，反之亦然；
- query/context 同时平移时 relative attention score 在 tolerance 内不变；
- V、residual hidden、time embedding 和 condition embedding 不受 RoPE 修改；
- head dimension 160 精确拆成 row 80 / column 80。

### Mode isolation

- `FH0` 与当前 flow head forward/loss/generation 回归等价；
- `FH1-FH0` 代码路径只增加 Q/K rotation；
- `FH2-FH0` 只移除 context additive lookup/addition；
- `FH3-FH2` 只增加 Q/K rotation；
- `FH4-FH3` 只移除 query additive lookup/addition；
- 所有 variants 的 learned parameter names/shapes/counts 完全相同。

### Cache and attention

- cached 与 uncached inference outputs 在 tolerance 内一致；
- K 只旋转一次，V 不旋转；
- query/context position permutation 后 cache 与 mask 仍对齐；
- FlexAttention 与 SDPA fallback 输出/gradient 在 tolerance 内一致；
- 0/1/multiple context、部分 masked context 和 multi-query paths 全部 finite；
- strict `sigma_context < sigma_query` 不变，无未来 latent leakage。

### Provenance

- checkpoint/config/metrics 都记录 `A_q/A_c/R_f` 和 implementation contract；
- summary 拒绝 ID 与 flags 不一致、cache contract 缺失或 architecture digest 不同的 run；
- summary 只接收 `FH0--FH4` 的 matched seed-42 screen；
- seeds 43/44/45 的 partial confirmation artifacts 必须标记为 deferred/excluded，不能与
  seed 42 聚合。

## 12. 决策规则与结论边界

- 若 seed 42 上 `FH1` 优于 `FH0`，说明 flow-head relative 2D geometry 在保留 additive
  paths 时有值得后续验证的候选收益，但不能称为稳定收益；
- 若 `FH3` 优于 `FH1` 且 `FH2` 不优于 `FH0`，说明移除 context additive 的收益依赖
  2D RoPE，是明确 interaction；
- 若 `FH2` 与 `FH3` 都优于对应 `A_c=1` cell，说明 context value path 更适合保持纯内容；
- 若 `FH4` 只在 early/0--1 context bucket 恶化，支持保留 flow-query absolute identity；
- 若所有 RoPE systems 都不改善，保留 `FH0`，不能因 backbone 使用 2D RoPE 就宣称 head
  也应保持形式一致；
- 若 seed-42 差异较小或不同指标方向不一致，优先保留当前更简单、已有 cache 实现的
  `FH0`，并把其它 systems 记录为未确认候选，而不是继续自动扩展 seeds。

本实验只比较固定 contextual head 内的位置表达，不能重新回答 contextual cross-attention
是否优于 token-only MLP，也不能证明 pure RoPE 在所有 flow architectures 中无效。固定
`256×256` screen 也不能排除 2D RoPE 在分辨率、宽高比或序列长度外推中的潜在收益。

完成本 proposal 后，provisional choice 的完整 position contract 与 digest 写回 experiment
manifest；它可以作为下一阶段实验的当前默认值，但不能标记为 multi-seed-confirmed
winner。随后可以启动
[`SELFLESS_FLOW_CAPTION_AND_BACKBONE_INIT_ABLATION_PROPOSAL.md`](../SELFLESS_FLOW_CAPTION_AND_BACKBONE_INIT_ABLATION_PROPOSAL.md)
中的数据/初始化实验。
