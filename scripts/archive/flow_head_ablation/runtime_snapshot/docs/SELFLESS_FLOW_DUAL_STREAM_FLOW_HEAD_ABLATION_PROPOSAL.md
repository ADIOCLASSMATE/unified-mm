# Selfless-Flow Dynamic Dual-Stream × 2D Position Flow Head 消融 Proposal

Status: **revised; matched single-seed architecture × position screen**

Written: 2026-07-24

Scope: seed-42 screen only；不包含多 seed confirmation、full ImageNet 或其他自动扩展实验。

## 1. 研究问题与核心方案

当前 contextual flow head 的 query 在每个 block 中 cross-attend 一份不随深度更新的
clean-latent context，然后把 attention 输出直接加回 query residual。旧实验已经证明：
完全去掉 clean-latent interaction、退化成 token-only MLP 会损失生成质量；但只在这条
静态 cross-attention 上修改 additive position 或加入 2D RoPE，也没有找到优于 baseline
的候选。

本轮改为回答一个更本质的问题：

> flow head 是否应该像 Selfless Qwen backbone 一样，把 clean content 与 noisy query
> 视为同一个共享网络中的两种流，而不是把 content 当作每层都不更新的外部 K/V？

同时检验一个可能与 architecture 耦合的位置问题：

> static-context head 上无收益的 2D RoPE，在逐层更新的 dynamic content memory 中是否
> 会产生不同作用？加入 RoPE 与移除 additive position 各自贡献多少？

主方案是 **dynamic dual-stream AdaLN-DiT**：

- content stream 输入与 Qwen backbone 共用同一份小噪声 latent；
- query stream 输入为每个位置独立采样 flow time 和 noise 后的 `x_t`；
- 两条流共享 input projection、attention、MLP、AdaLN 和 output projection；
- content 与 query 的 Q 都由各自 hidden state 产生；
- 两条流的 K/V 都只来自 content stream；
- 两条流都使用相同的 strict sigma-causal mask `sigma_k < sigma_q`；
- content stream 在每层更新，成为随深度演化的 causal memory；
- velocity prediction 和 flow loss 只读取 query stream；
- 推理时不维护两个完整序列，只运行一个 active query，并复用逐层 content K/V cache。

位置因素采用三段式 controlled ladder，保证两次改动不落在同一个 contrast 中：

- `FH0`：query/content additive 2D position；无 RoPE；
- `FH1`：保留两侧 additive，并加入 row/column 2D RoPE；
- `FH4`：在 `FH1` 基础上移除两侧 additive，成为 pure 2D RoPE。

本设计不引入 stage conditioning，不增加 stream-ID embedding，也不让 query 产生 K/V。
所有 architecture × position cell 暂时只运行 seed 42。

## 2. 已有证据与为何转向信息交互

### 2.1 Context interaction 是必要的

旧 ImageNet-100 flow-head architecture ablation 使用相同的 10K evaluation protocol：

| Flow head | Head parameters | Head 内 token interaction | FID ↓ | IS ↑ |
| --- | ---: | --- | ---: | ---: |
| Contextual baseline | 164.073M | clean-latent cross-attention | **26.0110** | **59.5362** |
| Token-only MLP, ratio 1.0 | 72.210M | 无 | 27.9774 | 56.9485 |
| Token-only MLP, ratio 4.5 | 163.996M | 无 | 26.4404 | 58.3860 |
| Token-only MLP, width 1936 | 163.828M | 无 | 26.9315 | 57.3898 |

参数匹配的 ratio-4.5 token-only head 仍比 contextual baseline 差 `0.4294` FID 和
`1.1502` IS。说明收益不只是参数量；flow head 需要显式读取已经可见的 clean latent。

### 2.2 Static head 的位置结论不能替代 architecture × position interaction

在冻结 `E2-Q1` backbone 后，matched seed-42 `FH0--FH4` screen 得到：

| ID | Flow-head position change | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: |
| **FH0** | additive query + additive context；无 RoPE | **25.0669** | **61.5120** |
| FH1 | FH0 + row/column 2D RoPE | 25.2985 | 60.3427 |
| FH2 | FH0 去掉 context additive | 27.3799 | 57.4373 |
| FH3 | FH2 + row/column 2D RoPE | 25.4672 | 60.7045 |
| FH4 | FH3 再去掉 query additive | 26.1535 | 60.1577 |

FH0 是 static head 中唯一的 FID/IS Pareto point。RoPE 能部分修复移除 additive
context 后的退化，但不能超过原始 additive baseline；flow MSE 也不能正确预测 FID
排名。不过 dynamic dual stream 会逐层更新 content Q/K，因此上述 static 结果不能排除
architecture × position interaction。本轮复用 `FH0/FH1/FH4` 作为三个 DF0 锚点，
并在 DF1/DF2 下匹配这三种 position contract。完整历史记录见
[`archive/SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL_HISTORICAL.md`](archive/SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL_HISTORICAL.md)。

## 3. DF0：当前 static-context baseline

固定图像位置 `i`，clean target 为 `z_i`，Qwen backbone 输出的同位置 condition 为
`c_i`。训练时已有一份与 backbone 共用的小噪声 content latent：

\[
\tilde z_i=z_i+\eta_i,\qquad \eta_i\sim\mathcal N(0,\alpha^2I),\quad \alpha=0.01.
\]

query flow sample 为：

\[
\epsilon_i\sim\mathcal N(0,I),\qquad
t_i\sim p(t),\qquad
x_{t_i,i}=(1-t_i)\epsilon_i+t_i z_i.
\]

对 position contract `p`，DF0 的初始 hidden 为：

\[
C_i^0=W_{\mathrm{in}}\tilde z_i+A_pP_{xy}(i),\qquad
Q_i^0=W_{\mathrm{in}}x_{t_i,i}+A_pP_{xy}(i).
\]

`FH0/FH1` 使用 `A_p=1`，`FH4` 使用 `A_p=0`；`FH1/FH4` 在 attention 中对 Q/K
应用 row/column 2D RoPE，FH0 不应用。该位置因素不改变 DF0 的 static-context 定义。

每个 block 都从同一个不更新的 `C^0` 独立投影 K/V。只有 query 被 AdaLN attention
和 MLP 更新：

\[
Q^{l+1}=F_l(Q^l;\operatorname{KV}_l(C^0),\,T(t)+W_c c,\,M_\sigma).
\]

其中：

\[
M_\sigma[i,j]=\mathbf 1[\sigma_j<\sigma_i].
\]

canonical DF0 对应已完成的 `FH0 seed-42`：

| Artifact | Value |
| --- | --- |
| run | `output/selfless-flow-fhpos-fh0-s42` |
| FID / IS | `25.0669 / 61.5120` |
| final validation flow MSE | `0.688217` |
| flow-head parameters | `164,072,976` |
| initial-state SHA256 | `2d1a03416bef06958d3f3036623cb1042ee4048741b58a836ad5b61d9b1e89b6` |
| config fingerprint | `1cb03333953240e0ec6db0013b3ce2b0c490a4bd79901226de02e3e8ff2b33c4` |

position screen 另外复用两个 matched seed-42 DF0 锚点：

| Cell | Position contract | FID ↓ | IS ↑ | 训练状态 |
| --- | --- | ---: | ---: | --- |
| `DF0-FH0` | additive；无 RoPE | **25.0669** | **61.5120** | 复用 |
| `DF0-FH1` | additive + 2D RoPE | 25.2985 | 60.3427 | 复用 |
| `DF0-FH4` | pure 2D RoPE；无 additive | 26.1535 | 60.1577 | 复用 |

**DF0-FH0/FH1/FH4 均禁止重新训练。** 新 screen 直接复用已有 checkpoint、metrics 和
provenance；若 evaluator contract 更新，只允许重评已有 checkpoint。
历史 `E2-Q1` seeds 43/44/45 的 `24.9615 ± 0.4231` FID 只作为跨 seed 背景参考；
它们来自不同的冻结 source revision，不能替代 DF0/DF1 的 matched seed-42 contrast。

## 4. DF1：完整共享的 dynamic dual stream

### 4.1 输入与 conditioning

DF1 复用 DF0 的 input projection。令 `A_p∈{0,1}` 表示 position contract `p`
是否启用 additive 2D position：

\[
C_i^0=W_{\mathrm{in}}\tilde z_i+A_pP_{xy}(i),\qquad
Q_i^0=W_{\mathrm{in}}x_{t_i,i}+A_pP_{xy}(i).
\]

`FH0/FH1` 取 `A_p=1`，`FH4` 取 `A_p=0`。是否旋转 attention Q/K 由独立的
`R_p` 决定：`FH0` 为 identity，`FH1/FH4` 为 row/column 2D RoPE。

content 不重新采样第二份小噪声，而是直接复用 Qwen backbone 已构造的
`\tilde z` tensor。flow target 始终是 clean `z`。两条流使用同一个 timestep embedder
和 condition projection：

\[
y_i^C=T(1)+W_c c_i,\qquad
y_i^Q=T(t_i)+W_c c_i.
\]

`t=1` 表示 data endpoint；`\alpha=0.01` 仍只是 content augmentation，不把它重新解释成
另一个 flow time。每个 query position 的 `t_i` 与 `\epsilon_i` 独立采样，和当前
FlowLoss 完全一致。

### 4.2 共享 block

令 `H_s^l` 表示第 `l` 层的 stream hidden，`s∈{C,Q}`。同一个 block 先为两条流构造
AdaLN-modulated query：

\[
\begin{aligned}
U_s^l &= \operatorname{Modulate}
  (\operatorname{LN}_q(H_s^l);y_s),\\
q_s^l &= R_p(i)\,W_q^l U_s^l.
\end{aligned}
\]

K/V 只从当前层 content hidden 产生：

\[
k_C^l=R_p(i)\,W_k^l\operatorname{LN}_{kv}(C^l),\qquad
v_C^l=W_v^l\operatorname{LN}_{kv}(C^l).
\]

RoPE 只作用于 Q/K，V 始终不旋转。content/query 使用相同的 `R_p` 与相同二维位置；
因此 DF1 的 endpoint-consistency argument 在 `FH0/FH1/FH4` 下均成立。

两条流使用同一个 attention operator、同一个 output projection 和同一个 strict mask：

\[
\begin{aligned}
a_C^l&=\operatorname{Attn}(q_C^l,k_C^l,v_C^l;M_\sigma),\\
a_Q^l&=\operatorname{Attn}(q_Q^l,k_C^l,v_C^l;M_\sigma),\\
\bar C^l&=C^l+g_{\mathrm{attn}}(y_C)\,W_o^la_C^l,\\
\bar Q^l&=Q^l+g_{\mathrm{attn}}(y_Q)\,W_o^la_Q^l.
\end{aligned}
\]

随后两条流分别通过同一个 AdaLN-MLP：

\[
\begin{aligned}
C^{l+1}&=\bar C^l+
g_{\mathrm{mlp}}(y_C)\operatorname{MLP}_l(
\operatorname{Modulate}(\operatorname{LN}_{mlp}(\bar C^l);y_C)),\\
Q^{l+1}&=\bar Q^l+
g_{\mathrm{mlp}}(y_Q)\operatorname{MLP}_l(
\operatorname{Modulate}(\operatorname{LN}_{mlp}(\bar Q^l);y_Q)).
\end{aligned}
\]

所有 `Wq/Wk/Wv/Wo`、Norm、AdaLN 和 MLP 参数在 content/query 之间完全共享。content
stream 不新增专属 block，不使用 stop-gradient。虽然没有 content-side direct loss，
query velocity loss 会通过 K/V 和 content recurrence 训练 content representations。

### 4.3 为什么 residual 能保留 query 自身信息

当可见 context 数为 0 时，attention residual 定义为零；当可见 context 数为 1 时，
softmax 权重恒为 1。两种情况下 query 都不会消失：

\[
Q^{l+1}=Q^l+\text{gated attention}+\text{gated MLP}.
\]

`x_t` 的自身信息始终通过 query residual 传播。单 key softmax 只限制“如何在多个
context token 间选择”，并不把 query hidden 稀释成 context value。

### 4.4 Endpoint consistency

在关闭小噪声、令 `Q^0=C^0` 且 `t=1` 时，有 `y_Q=y_C`。由于两条流使用相同的
Q projection、相同 K/V、相同 mask、相同 residual gates 和相同 MLP，可逐层归纳：

\[
Q^l=C^l,\quad \forall l.
\]

因此 clean endpoint 与 content memory 不是两套任意参数化。训练中的 `0.01` 小噪声只让
该等价关系成为邻域约束，提升对推理期 generated-latent 误差的鲁棒性。

### 4.5 无 target leakage

对任意 `i`，query `Q_i` 只能读取 `C_j,\sigma_j<\sigma_i`。content `C_j` 自身也只从
更早的 `C_k,\sigma_k<\sigma_j` 更新。由归纳可知：

\[
C_j^l=f_l(\tilde z_{\{k:\sigma_k\le \sigma_j\}}),
\]

所以 `Q_i` 的所有 K/V 依赖都严格来自 `\sigma_j<\sigma_i` 的可见 latent；`z_i`
和未来 latent 不可能经 content recurrence 绕过 mask。实现必须对每一层都使用
strict `<`，不能在 content stream 上改成 `<=`。

## 5. DF2：attention-only content update control

DF2 与 DF1 完全相同，但 content stream 跳过每层的 MLP residual：

\[
C^{l+1}=\bar C^l.
\]

query stream 仍执行完整 attention + MLP。DF2 的作用不是候选默认，而是区分：

- 只要让 content 在层间做 causal attention 更新是否已经足够；
- DF1 的 content-side MLP 和完整 endpoint-consistent block 是否带来额外收益；
- 更轻的 cache insertion 计算是否能保留 DF1 的主要收益。

DF2 与 DF0/DF1 使用相同的 learned parameter tensors；MLP 仍被 query stream 使用，
因此 learned parameter count 不变。DF2 不满足完整的 `Q=C` endpoint equality，应作为
机制/效率 control 单独解释。

## 6. 正式实验矩阵

architecture factor：

| ID | Content state | Content block | Query K/V source | 参数共享 |
| --- | --- | --- | --- | --- |
| **DF0** | 所有层固定为 `C^0` | 无 | static `C^0` | n/a |
| **DF1** | 逐层更新 | shared attention + shared MLP | dynamic `C^l` | 全共享 |
| **DF2** | 逐层更新 | shared attention only | dynamic `C^l` | attention 全共享 |

position factor：

| ID | Query additive | Content additive | Row/column 2D RoPE | 解释 |
| --- | ---: | ---: | ---: | --- |
| **FH0** | on | on | off | canonical additive |
| **FH1** | on | on | on | 只增加 RoPE |
| **FH4** | off | off | on | 从 FH1 只移除 additive；pure RoPE |

完整 matched seed-42 matrix：

| Architecture | FH0 | FH1 | FH4 |
| --- | --- | --- | --- |
| **DF0 static** | 复用已有 artifact | 复用已有 artifact | 复用已有 artifact |
| **DF1 full dual** | 新训练 | 新训练 | 新训练 |
| **DF2 attention-only** | 新训练 | 新训练 | 新训练 |

在固定 position contract `p∈{FH0,FH1,FH4}` 时，architecture estimands 为：

\[
\begin{aligned}
\Delta_{\mathrm{dual}}(p) &= m(DF1,p)-m(DF0,p),\\
\Delta_{\mathrm{attn\ only}}(p) &= m(DF2,p)-m(DF0,p),\\
\Delta_{\mathrm{content\ MLP}}(p) &= m(DF1,p)-m(DF2,p).
\end{aligned}
\]

在固定 architecture `d∈{DF0,DF1,DF2}` 时，两次位置改动分别估计：

\[
\begin{aligned}
\Delta_{\mathrm{RoPE|add}}(d)
  &=m(d,FH1)-m(d,FH0),\\
\Delta_{\mathrm{-add|RoPE}}(d)
  &=m(d,FH4)-m(d,FH1),\\
\Delta_{\mathrm{pure\ RoPE}}(d)
  &=m(d,FH4)-m(d,FH0).
\end{aligned}
\]

对 `d∈{DF1,DF2}`，architecture × position interaction 为：

\[
\begin{aligned}
I_{\mathrm{RoPE|add}}(d)
  &=\Delta_{\mathrm{RoPE|add}}(d)
    -\Delta_{\mathrm{RoPE|add}}(DF0),\\
I_{\mathrm{-add|RoPE}}(d)
  &=\Delta_{\mathrm{-add|RoPE}}(d)
    -\Delta_{\mathrm{-add|RoPE}}(DF0).
\end{aligned}
\]

这是有明确顺序的三段式 controlled ladder，而不是完整 `additive on/off × RoPE on/off`
四格 factorial。刻意不加入“无 additive、无 RoPE”的无二维位置信号 cell；因此报告
的是上述 conditional effects，而不是脱离条件的 additive/RoPE main effect。

本轮不加入 untied content/query towers。untied 设计会增加约一套 block 参数，并同时改变
capacity、optimization 和 interaction form，不能作为主矩阵中的原子比较。

## 7. 训练与 loss

所有 image positions 在一个 batch 中并行构造：

1. 从 clean target `z` 构造一次 `\tilde z=z+\eta`，并与 Qwen backbone content input
   精确复用同一 tensor；
2. 对每个位置独立采样 `t_i` 和 `\epsilon_i`；
3. 构造所有 `x_{t_i,i}`，形成 query stream；
4. content/query 都使用同一个 `M_\sigma[i,j]=[\sigma_j<\sigma_i]`；
5. 只从 `Q^L` 经过 final AdaLN layer 得到 velocity：

   \[
   \hat v_i=W_{\mathrm{out}}\operatorname{AdaLN}_{final}(Q_i^L;y_i^Q);
   \]

6. loss 只计算：

   \[
   \mathcal L=\frac1N\sum_i
   \left\|\hat v_i-(z_i-\epsilon_i)\right\|_2^2.
   \]

content hidden 不加 reconstruction loss、distillation loss 或 auxiliary loss。若 DF1
失败，不能在同一 screen 中事后加入这些 loss 修补。

## 8. 单流增量推理与 cache contract

训练需要双流，是为了并行计算所有 query；推理只存在“历史 content cache + 当前 active
query”，不构造两份完整 256-token hidden states。

### 8.1 ODE solve

生成位置 `i` 时：

- `Q_i^0=W_in x_t+A_pP_xy(i)`；
- 每个 ODE evaluation 只推进 query hidden；
- 第 `l` 层读取已生成 positions 的 cached `K_C^l/V_C^l`；
- `FH1/FH4` 的 active query 按当前位置应用一次 2D RoPE；
- 当前 `x_t` 不写入 cache，避免不同 ODE time 之间污染；
- empty context 时 attention 分支输出精确零且保持 finite。

因此 100-step Heun 不会重复计算历史 content tower。

### 8.2 Commit generated latent

ODE 完成得到 `\hat z_i` 后，将它作为一个新的 clean content token只运行一次：

1. 构造 `C_i^0=W_in\hat z_i+A_pP_xy(i)`；
2. 在第 `l` 层先用已有 earlier-content cache 更新 `C_i^l`；
3. 当前 token 不读取自己的 K/V；
4. 将由 pre-update `C_i^l` 得到的 `K_i^l/V_i^l` 追加到第 `l` 层 cache；`FH1/FH4`
   的 K 在 commit 时只旋转一次，V 不旋转；
5. 继续得到 `C_i^{l+1}`，直到所有层 cache 均完成追加。

正式评测固定 `parallel_rate=1`，避免同一批 commit token 之间的可见性语义不明确。

### 8.3 CFG

DF1/DF2 的 content recurrence 使用 `c_i`，因此 conditional 与 unconditional branch
从第二层起可能产生不同 content states。实现必须维护两个逻辑 cache，并让每个 query
只读取对应 branch 的 K/V；不能把 conditional content cache 错用于 unconditional
velocity。第一层可在严格等价时共享物理 storage，但这只是实现优化，不能改变数值。

## 9. 固定项

DF0/DF1/DF2 必须固定：

- `image_backbone_variant=E2-Q1`；
- ImageNet-100 balanced membership、80 epochs、seed 42 和 data order；
- Qwen3-0.6B-Base initialization；
- direct `256×16` KL latent layout；
- flow width 1280、depth 8、8 heads、MLP ratio 1.0；
- zero-initialized AdaLN gates 和 final projection；
- flow-time sampling、`uniform_mix`、loss reduction 和 `image_flow_batch_mul`；
- position contract 只能是 `FH0/FH1/FH4`，其 additive/RoPE 映射严格按第 6 节；
- `FH1/FH4` 只旋转 Q/K，不旋转 V，使用相同 row/column axis split；
- optimizer、learning rates、global batch、EMA 和 checkpoint cadence；
- CFG 3.5、100-step Heun、temperature 1.0、`spatial_halton`、
  `parallel_rate=1`；
- 相同 validation prompts、initial noises、original-ImageNet FID stats 和 10K samples。

明确禁止：

- stage conditioning 或 stream-ID embedding；
- query self-attention、query-to-query K/V 或 query 写 cache；
- content stream 使用 `sigma_k<=sigma_q`；
- 为 content 单独增加 weights、Norm、AdaLN 或 auxiliary head；
- 使用 `FH0/FH1/FH4` 以外的 position 组合、relative bias 或 local window；
- 重训任一 DF0 position anchor，或用历史 E2-Q1 三 seed 均值替换 matched
  DF0 seed-42 metric；
- 在本 screen 中追加 seed 43/44/45 或根据中间 FID 改变实验矩阵。

## 10. Pairing 与参数公平性

所有 DF1/DF2 × FH0/FH1/FH4 cell 应复用 DF0 的 learned tensor schema：

- input/condition/time projections；
- per-block `q/k/v/o`、Norm、AdaLN 和 MLP；
- final AdaLN 与 velocity projection。

新增的是 tensor execution path 和 cache state，不是 learned weights。验收要求：

- flow-head learned parameter count 精确等于 DF0 的 `164,072,976`；
- parameter names/shapes 与 DF0 一致；
- seed 42 initial-state tensor hash 与 DF0 一致；
- 六个 dynamic cell 的 initial-state hash 相同；
- position contract 只改变固定 additive buffer 的使用和 Q/K 执行路径，不增加参数；
- 若为了实现必须改变任何 learned tensor，proposal 必须先修订，不能把结果称为
  parameter-matched。

## 11. 指标与机制诊断

Primary：

- 10K FID；
- IS 及其 10-split standard deviation。

Guardrails：

- final validation flow velocity MSE；
- `t∈{0.1,0.5,0.9}` 的 `x_0` estimate latent MSE/RMS；
- train images/s、peak training memory；
- 10K sampling wall time、samples/s 和 per-layer cache memory；
- non-finite count 必须为 0。

按 visible context count 分 bucket：

```text
0, 1, 2--4, 5--16, 17--64, 65+
```

每个 bucket 报告 query velocity MSE、content/query attention gate、MLP gate、attention
entropy 和 spatial distance。DF1 额外报告：

- `||C^{l+1}-C^l|| / ||C^l||`；
- content attention/MLP update RMS；
- content/query hidden cosine at `t≈1`；
- conditional/unconditional content-cache divergence。

这些诊断用于解释，不覆盖 FID/IS selector。任何 NaN/Inf 都必须定位第一个产生位置，
不能用 clamp、`nan_to_num` 或丢弃 samples 继续汇总。

## 12. Screen 与决策规则

本 proposal 只定义一个 matched seed-42 architecture × position screen：

- DF0-FH0/FH1/FH4：直接读取已有 artifact，训练 jobs 数为 0；
- DF1-FH0/FH1/FH4：3 个新训练 run；
- DF2-FH0/FH1/FH4：3 个新训练 run；
- checkpoint 完成后即可分别启动评测，不等待另一个 run 全部结束。

总计只新增 6 个 seed-42 training cells，不运行第二个 seed。以 canonical `DF0-FH0`
的 `FID=25.0669, IS=61.5120` 为绝对质量锚点。任一 dynamic candidate 通过 screen 的
必要条件为：

\[
FID\le24.5669\quad\text{且}\quad IS\ge61.0120,
\]

即至少改善 `0.5` FID，且 IS 下降不超过 `0.5`。同时：

- sampling wall time不得超过 DF0 的 `1.20×`；
- 训练/推理必须全程 finite；
- cache/uncached parity 和 no-leakage tests 必须通过。

position effects 与 interaction 按第 6 节完整报告，即使对应 cell 未通过绝对门槛。
若多个 dynamic candidate 过线：

- 最低 FID 与其他候选相差超过 `0.25` 时选择最低 FID；
- FID 相差不超过 `0.25` 时优先 IS 更高者；
- FID 相差不超过 `0.25` 且 IS 相差不超过 `0.5` 时，优先 sampling throughput 更高者；
- 仍无法区分时，优先 learned/serving contract 更简单的 `DF2`，position contract
  按 `FH0`、`FH1`、`FH4` 的顺序优先。

若都不过线，保留 `DF0-FH0`。本轮输出只能称为 single-seed mechanism screen；
无论结果是否改善，都不自动触发 seeds 43/44/45、full ImageNet、caption 或
initialization 实验，也不把单 seed winner 宣称为稳健结论。

## 13. 建议配置接口

主接口使用两个受限枚举，而不是可自由拼装的 additive/RoPE flags：

```yaml
model:
  image_flow_head_variant: "DF1"  # DF0 | DF1 | DF2
  image_flow_position_variant: "FH1"  # FH0 | FH1 | FH4
```

architecture 精确映射：

| Variant | Architecture | Content MLP |
| --- | --- | ---: |
| DF0 | contextual static K/V | n/a |
| DF1 | dynamic shared dual stream | on |
| DF2 | dynamic shared dual stream | off for content only |

position 精确映射：

| Variant | Query additive | Content additive | 2D RoPE |
| --- | ---: | ---: | ---: |
| FH0 | on | on | off |
| FH1 | on | on | on |
| FH4 | off | off | on |

checkpoint/config/metrics 必须记录 variant、cache schema、strict-mask digest、
shared-noise provenance、position contract/digest、parameter schema hash 和
initial-state hash。名称与实际行为不一致时 fail fast。

## 14. 实现落点与必须通过的测试

实现与正式实验必须满足以下契约。

### Flow head

- 新增 shared dual-stream block execution，但复用现有 learned modules；
- training forward 同时推进 `C/Q`，只输出 query velocity；
- inference 提供逐 token content-cache append；
- content K/V 按 layer 缓存，active query 永不写 cache；
- gradient checkpointing 覆盖双流且不改变随机数消费。

### Numerical / semantic tests

- **DF0 regression**：旧 checkpoint 的 forward、loss、sampling 和 cache 完全等价；
- **shared parameter test**：DF1/DF2 无 content-only learned parameter；
- **endpoint equality**：关闭 content noise、设置 `Q^0=C^0,t=1` 时 DF1 每层
  `Q^l=C^l`；
- **no-leakage test**：修改 `z_i` 或 future `z_k` 不改变 `Q_i`；对应 forbidden
  gradients 为零；
- **independent-time test**：改变 `t_i` 不影响其他 query 的 timestep condition；
- **shared-noise test**：content head 与 Qwen backbone 读取同一 noised tensor；
- **loss-routing test**：direct loss 只来自 query output，但 content path收到间接梯度；
- **empty/single-context test**：输出与梯度 finite，query residual 保留；
- **cache parity**：full-sequence training-style forward 与 incremental cached inference
  在同一 inputs/conditions 下数值一致；
- **CFG cache test**：conditional/unconditional caches 不串支路；
- **position ladder test**：FH0 只使用 additive；FH1 同时使用 additive/RoPE；FH4
  不读取 additive buffer 且只对 Q/K 应用 RoPE；
- **RoPE cache test**：cached K 只旋转一次、active Q 每次按当前位置旋转、V 不旋转；
- **parameter/init test**：全部 architecture × position cell 的 count、schema 和
  seed-42 initialization 满足第 10 节。

## 15. 结论边界

若 DF1 改善，只能说明“在当前 E2-Q1、256-token、8-layer flow tower、指定 position
contract 和 ImageNet-100 seed-42 protocol 下，动态共享 content memory 优于静态
cross-attention context”。若 interaction 为正，只能说明 static head 的 position
结论不能直接外推到该 dynamic architecture。
它不能单独证明：

- 任意分辨率或更大模型都获益；
- shared weights 一定优于参数更多的 untied towers；
- content noise `0.01` 已是最优；
- 相同 position effect 会推广到其他 seed、分辨率或 dual-stream architecture。

本轮同时检验信息交互形式及其与位置编码的 conditional interaction，但只提供单 seed
机制证据。是否追加多 seed confirmation、scaling 或 content-noise 实验需另立 proposal。
