# Selfless-Flow Image Token Embedder：历史原子级架构消融 Proposal

Status: completed; the strict selector retains `E4b` as the recommended architecture.
Revised 2026-07-22 after the Q1-reuse amendment and final Q0 evaluation.

## 1. 目标与结论

本轮消融只回答一个问题：在 Selfless-Flow 中，连续图像 latent 应该怎样成为
Qwen backbone 的 token，才能得到更好的生成架构？

Seed-42 screen 原先包含三个架构假设：

1. **Stage query coordinate**：把外层 image-token reveal stage 显式坐标化到图像
   query；
2. **Row/column factorized 2D RoPE**：图像 Q/K 使用行列分解的二维 RoPE，
   observed image hidden/value path 不再混入 additive spatial position；
3. **Lossless space-to-depth**：使用纯 reshape/permutation 将相邻 `2×2` latent
   合并为一个 token，不丢失任何 VAE latent 元素。该分支只作为历史 screen 记录，
   已退出 confirmation。

2026-07-21 的设计复核发现原矩阵遗漏了第四个原子问题：在 backbone 已使用
row/column 2D RoPE 时，mask query 是否仍需要 additive absolute `P_xy`。原 `E2/E4`
只移除了 observed latent 的 additive position，mask query 始终保留 `P_xy`，因此现有
结果没有回答这个问题。本文新增 `Q` 因子作为 **prospective post-confirmation
extension**；它不追溯修改既有 screen/confirmation manifest，也不能被表述为原始
预注册矩阵的一部分。

Screen 阶段先做原子消融，再做组合消融。它们不能被打包成一个
“Qwen3.5-style module”，因为我们借鉴的只是其**不同模态共享多轴坐标槽**的思想，
而不是照搬 Qwen3.5 的视觉塔、T/H/W 语义、partial rotary ratio 或 patch merger。

2026-07-20 的确认范围决策是：**不再把 S2D 当作合法的 image-embedder 因子**。
虽然 S2D/D2S 在数值上可逆，但 `f=2` 会同时把 reveal 原子从单个 latent site 改成
`2×2` block、把外层生成步数从 256 改成 64，并把 contextual flow head 的
latent input/output width 从 16 改成 64。它因此没有在固定 flow-head 接口下只改变
image embedder，不能回答本轮问题。正式 confirmation 固定
`image_space_to_depth_factor=1`，只比较 stage、observed additive position 与
row/column RoPE。

最终候选架构写成：

\[
\begin{aligned}
z' &= \operatorname{S2D}_f(z),\\
h^{X0}_{\mathrm{image}} &= W_z z' + b_z,\\
h^{Q}_{\mathrm{image}} &= e_{\mathrm{mask}}
    + \mathbb{1}_{Q}P_{xy}(r,c)
    + \mathbb{1}_{\mathrm{stage}}P_s(s),\\
Q'_{\mathrm{image}},K'_{\mathrm{image}}
    &= \operatorname{RoPE}_{row,col}(Q_{\mathrm{image}},K_{\mathrm{image}}),\\
V'_{\mathrm{image}} &= V_{\mathrm{image}}.
\end{aligned}
\]

上式保留 screen 的历史定义，其中曾取 `f∈{1,2}`；confirmation 中固定 `f=1`，
即 `z'=z`。原 confirmation 的活跃开关只控制 `P_s`、observed additive position 和
row/column RoPE，并隐式固定 `Q=1`；新增 extension 才比较 `Q∈{0,1}`。

所有 image-embedder 变体统一使用当前 baseline 的 contextual flow head。对第 `i`
个待生成 latent token，其 velocity prediction 为

\[
v_i=f_\theta\!\left(x_{t,i},t_i,c_i,\mathcal C_i\right),\qquad
\mathcal C_i=\{(z_j,p_j)\mid \sigma_j<\sigma_i\},
\]

其中 `c_i` 是同位置 Qwen hidden condition，`\mathcal C_i` 是严格早于该 query 的
clean-latent context。flow head 在每个 block 中让 noisy latent query cross-attend
这些 visible clean latents，再经过 timestep/condition 调制的 MLP。flow-head
architecture 不是本 proposal 的实验因子。

## 2. 本轮不做什么

- 不迁移到 Qwen3.5 backbone；继续以当前 Qwen3 baseline 为模型锚点。
- 不在 image embedder 或 Qwen backbone 中引入新的视觉 encoder、cross-attention
  adapter、patch merger 或额外 resampler；baseline contextual flow head 自带的
  clean-latent cross-attention 保持不变。
- 不增加 dynamic gate、learnable stage scalar 或按层可训练 position scale。
- 不恢复 `z_proj_ln`，也不对输入 latent 做 LayerNorm/RMS=1 normalization。
- 不把 flow matching 的连续时间 `t` 当作 generation stage。
- 不再把 flow-head architecture 纳入本轮矩阵。已有消融已选定 baseline
  `contextual` head；所有 run 均固定相同的 head class、depth、width、MLP ratio、
  attention heads、gating 和 optimization protocol。这里“固定”只指架构与超参数：
  每个 run 都从同一初始化规则重新训练 flow-head weights，绝不冻结或跨 run 复用其
  权重。只有 `S2D` 必然引起的 latent input/output width 与 token-count 派生形状
  可以改变。

因此本矩阵估计的是 image-embedder intervention 经过同协议端到端重新优化后的
**system-level total effect**，不是“冻结同一套 flow-head weights 后”的局部前向效应。
后者既会偏向 baseline，且在 `D` 改变 latent width 时也没有形状兼容性。

## 3. 当前 baseline 契约

image embedder/backbone 当前实现在
`models/modeling_model/modeling_selfless_flow.py`，flow head 实现在
`models/modeling_model/image_flow_loss.py`：

- 原始 VAE latent 为 `16×16×16`，按空间展平为 `256×16`；
- observed latent token：

  \[
  h^{X0}=W_z z+b_z+gP_{xy};
  \]

- unfilled X0 image slot 与训练时 XT image query：

  \[
  h^{Q}=e_{mask}+gP_{xy};
  \]

- `P_xy` 是固定二维 sin/cos buffer，`g` 是现有的 position gain；
- 所有文本和图像 token 在 backbone attention 中共用 flatten sequence 的 1D
  Qwen RoPE；
- Q/K 在 attention 内部经过 Qwen 的 q/k RMSNorm，但 image embedder 输出本身
  **没有** LayerNorm；
- `sigma` 目前只决定 strict visibility：`sigma[kv] < sigma[q]`；
- baseline flow head 为 8-layer、width 1280、8-head、MLP ratio 1.0 的
  `ContextualFlowTransformerHead`；
- noisy latent query 和 visible clean-latent context 都先投影并加入 flow head
  自己的 fixed 2D sin/cos position embedding；
- 每个 flow block 以 `time_embed(t_i)+cond_embed(c_i)` 做 AdaLN modulation，query
  cross-attend 满足 `sigma_j<sigma_i` 的 clean latent keys/values，再通过 MLP；
- 推理时 context 是当前已经填充的 clean latents，query/context local positions
  显式传给 flow head；clean-context K/V 可在同一次 ODE solve 内缓存；
- flow head 的连续时间 `t_i` 独立采样，描述 query latent 的 flow trajectory；它
  与外层 reveal stage 不同，但 velocity prediction 不是 token-context independent。

选择该 head 的依据是同协议架构消融：baseline 得到 FID 26.0110、
IS 59.5362，而参数扩展后的 token-only `width=1936, ratio=1.0` 仍只有
FID 26.9315、IS 57.3898。扩宽能恢复部分差距，但不能替代 clean-latent
cross-attention，因此本 proposal 不再假设 flow head 完全无 cross-attention。

历史 baseline 结果只有在 flow head、embedder 配置、训练预算和评测协议均完全相同
时才可复用。否则必须把本矩阵的 `E0` 从同一个 Qwen checkpoint 重新训练。

## 4. 统一的信息分工

新架构应明确区分五种信息，避免把所有内容都直接相加到 image hidden state：

| 信息 | 表达路径 | 作用 |
| --- | --- | --- |
| latent content | observed X0 residual/value path | 图像局部内容与幅值 |
| absolute query identity | mask/query seed 中的固定 `P_xy` | 指明当前要预测哪个空间位置 |
| relative spatial relation | image Q/K 的 row/column 2D RoPE | 控制 token 间的二维 attention geometry |
| generation progress | query-only fixed `P_stage` | 告诉 query 当前已有多少图像上下文 |
| flow time | flow head 的 timestep embedding | 描述当前 query 从 noise 到 data 的连续路径；head 同时读取 visible clean-latent context |

本文所说“image hidden state 只保留幅值”不是把 latent 压成一个 scalar norm，
也不是把它归一化到 RMS=1。它表示 observed latent 的 residual/value representation
保留完整的多通道内容向量 `Wz+b`，但不再通过加法把空间坐标混入该向量。
坐标主要作用于 query seed 和 Q/K phase。

## 5. 消融 A：Generation stage 坐标化到 query

### 5.1 Stage 的准确语义

当前 repo 中的 `sigma` 是 token reveal permutation/order；flow head 中的 `t` 是
rectified-flow interpolation time。二者不可混用。

对第 `i` 个 image query，先计算 image-local reveal rank：

\[
r_i=\sum_{j}\mathbf{1}
\left[token\_type_j=image\;\land\;\sigma_j<\sigma_i\right].
\]

再归一化为：

\[
s_i=\frac{r_i}{\max(N_{image}-1,1)}\in[0,1].
\]

不能直接使用 raw `sigma_i`，因为 raw sigma 含 prompt、BOI/EOI 和 suffix 的 offset，
会随文本长度变化。

### 5.2 训练与推理对齐

- **训练 XT query**：每个 image query 按其 image-local rank 得到自己的 `s_i`；
- **训练 X0 masked slot**：若某位置使用 mask/query seed，也使用同一 stage 定义；
- **推理**：当前轮所有尚未填充的候选 query 使用
  `s = n_filled / max(N_image - 1, 1)`；
- 同一轮并行填充的 token 使用相同 stage，不能用人为的 batch 内顺序区分；
- `parallel_rate=1` 的正式比较与训练的单 token reveal 语义最直接对齐。

### 5.3 编码方式

主实验采用固定 1D sin/cos stage embedding，并且只加到 image query seed：

\[
h^Q=e_{mask}+P_{xy}+P_s(s).
\]

- observed latent `Wz+b` 不接收 `P_s`；
- text token 不接收 `P_s`；
- `P_s` 不进入 flow head timestep 分支或 clean-latent K/V path；它只能通过
  same-position backbone condition `c_i` 间接影响 baseline flow head；
- 不新增 learnable scalar、MLP 或 dynamic gate；
- 令 `u=255s`，复用 repo 的标准 1D sin/cos frequency bank，并使用
  `PE(u)-PE(0)` 使 stage 0 不改变 baseline query；
- 用 `u∈{0,...,255}` 上的全局 RMS 将该 buffer 一次性校准到当前 balanced
  positional component 的初始化 RMS；校准值是固定 buffer，不是 parameter；
- 为兼容 `f=1` 与 `f=2`，stage 的相位只由 normalized progress 决定，不能由
  64/256 的 raw token index 决定。

本轮不把 stage 默认设为第三个 Q/K RoPE axis。Qwen3.5 的 T 是内容时间坐标，
而这里的 stage 是生成算法状态。若以后测试 stage-RoPE，必须使用“预留相同 rotary
维度但 stage 恒为零”的 control，避免把 H/W rotary capacity 的变化误判为 stage 收益。

### 5.4 为什么值得单独验证

Strict `sigma[kv] < sigma[q]` mask 已经隐式改变了 backbone query 能看到的上下文
集合；baseline contextual flow head 还显式读取同一 reveal state 下的 visible clean
latents。两条路径理论上都可能从 context cardinality 推断进度。显式 `P_s` 的作用是
让 backbone query 直接知道全局 reveal fraction；它可能改善不同上下文密度下的
校准，也可能只是重复信息。所以 stage 是独立假设，不能作为二维 RoPE 的默认组成
部分。

## 6. 消融 B：Row/column factorized 2D RoPE

### 6.1 Coordinate rule

构造两个 rotary coordinate slots，而不是为文本额外增加一个专属轴：

- text/special token 的一维 running position `p`：

  \[
  (p_{row},p_{col})=(p,p);
  \]

- image token 位于 grid `(r,c)`，image span 的 sequence anchor 为 `a`：

  \[
  (p_{row},p_{col})=(a+r_{phys},a+c_{phys}).
  \]

对 `f=1`，`r_phys=r, c_phys=c`；对 `f=2`，使用原始 16×16 latent lattice
上的 block anchor：`r_phys=2r, c_phys=2c`。这样 S2D 前后的二维距离仍处在同一
物理坐标尺度。

`a` 只建立 text-image 的共同 coordinate frame；在 image-image relative phase 中
会相消。attention visibility 仍完全由 sigma mask 决定，不能用 RoPE coordinate
替代 causal/reveal order。

对 mixed sequence 使用一个 running coordinate cursor，而不是把 image grid 当作
256 个一维位置：

1. text/special span 长度为 `L` 时，依次使用 `(p,p)`，cursor 增加 `L`；
2. 遇到 image span 时，以当前 cursor 为 `a`，按上式放置整个 grid；
3. image span 结束后，cursor 增加 canonical spatial extent `16`，后续 EOI/text
   从新 cursor 继续；
4. `f=1` 与 `f=2` 都使用 extent 16，因此 S2D 不改变 text-image coordinate scale；
5. 纯文本序列的 running position 与当前 Qwen sequence position 完全相同。

### 6.2 Frequency allocation

复用当前 Qwen RoPE 的原始 `inv_freq`，按 rotary frequency pair 交错分配 row/column：

```text
frequency-pair index: 0   1   2   3   4   5   ...
coordinate source:    row col row col row col ...
```

不使用“前一半低频给 row、后一半高频给 column”的 contiguous split，避免两个轴
获得系统性不同的频谱。

当 text token 使用 `(p,p)` 时，每个 frequency pair 看到的仍是同一个 `p`，因此
纯文本 attention 的 rotary phase 应严格退化为当前 pretrained Qwen 的 1D RoPE。
这是实现必须满足的兼容性测试，而不是仅凭直觉假设。

### 6.3 Attention path

对 image token：

\[
\begin{aligned}
Q'_{X0} &= R_{row,col}(Q_{X0}),\\
Q'_{XT} &= R_{row,col}(Q_{XT}),\\
K'_{X0} &= R_{row,col}(K_{X0}),\\
V'_{X0} &= V_{X0}.
\end{aligned}
\]

也就是说：

- X0 query、XT query 和 X0 key 都使用相应 token 的二维 coordinate；
- value 不旋转；
- residual hidden state 不旋转；
- observed latent embedding 改为严格的 `Wz+b`，不再加 `P_xy`；
- 不增加 image-side normalization。

这里的 Q/K/V 均指 Qwen backbone attention。`R` 不修改 baseline flow head 的
cross-attention，也不把 row/column RoPE 移植到 flow head；flow head 继续使用自身
原有的 fixed 2D sin/cos query/context position embedding。否则实验会同时改变
image embedder 和 flow head，无法归因。

### 6.4 为什么 query 仍保留 absolute `P_xy`

二维 RoPE 只改变 pairwise Q/K relation，不能完全替代 query identity：

- 没有可见 image key 时，image-image RoPE 没有作用对象；
- 只有一个可见 key 时，softmax 权重恒为 1，单个 relative score 无法区分输出位置；
- Selfless-Flow 的早期生成阶段正好频繁出现这两种情况。

因此 proposed geometry 是：

- observed content/value hidden：无 additive position；
- mask/query seed：保留 fixed absolute `P_xy`；
- Q/K relation：使用 row/column 2D RoPE。

这不是重复编码：前者解决“我要预测哪里”，后者解决“我与上下文的二维关系是什么”。

### 6.5 遗漏补充：mask query 完全移除 additive `P_xy`

上一节给出了保留 query `P_xy` 的机制动机，但动机不能替代实验。原实现中
`image_observed_position_mode=none` 只让 observed latent 变成 `W_z z+b_z`；
`embed_mask` 仍无条件构造：

\[
h^Q=e_{mask}+gP_{xy}+\mathbb 1_{stage}P_s.
\]

因此 `E2/E4` 不是“完全使用 2D RoPE”的 query 实验。新增因子定义为：

- `Q=1`（现有默认）：`h^Q=e_mask+gP_xy+optional(P_stage)`；
- `Q=0`（新增）：`h^Q=e_mask+optional(P_stage)`，不 lookup、不缩放、也不加任何
  absolute spatial embedding；
- local `(row,col)` 仍必须传给 backbone attention，用于 row/column 2D RoPE；
- observed-latent position mode、stage mode、sigma topology 和 image coordinates 均不变；
- flow head 仍保留 baseline 的 fixed 2D query/context additive position。本 extension
  只测试 **backbone mask-query seed**，不是 end-to-end pure-RoPE system；
- 不增加 learned position token、bias、gate 或其他补偿路径。

`Q=0` 的机制风险是：二维 RoPE 只作用于 pairwise Q/K phase。零个 visible image key
时没有 image-image relation；只有一个 visible image key 时 attention softmax 权重恒为
1，relative score 不能通过权重区分目标位置。另一方面，text keys、stage signal、noisy
flow query 以及 flow-head absolute position 仍可能提供足够锚点，所以该实验必须实测，
不能预设必然失败。

预注册机制假设为：

- `Q=0` 的损失若主要集中在 early reveal-stage buckets，支持 absolute query identity
  对稀疏 image context 必要；
- 若 `Q=0` 不降反升，说明 backbone 内 row/column relation 与下游 flow-head position 已
  足以定位，mask hidden 无需混入 absolute position；
- 若影响只在有/无 stage 的变体间不同，说明 `P_stage` 与 `P_xy` 存在 interaction，但
  stage 本身仍不能被解释为空间坐标。

## 7. 已退出 confirmation 的消融 C：`space-to-depth`

本节仅保留 seed-42 screen 与已有实现的可复现定义，不再把它作为有效候选空间。
“无损”只说明张量元素可逆，并不说明实验干预是原子的：S2D 改变了生成决策粒度、
context reveal trajectory、backbone 序列长度，以及 flow head 的输入/输出接口。
因此它与“固定 flow head 后消融 image embedder”的研究问题不相容。

### 7.1 精确定义

原始 latent layout：

```text
[B, H=16, W=16, C=16] -> 256 tokens × 16 channels
```

使用 factor `f=2` 后：

```text
[B, H=8, W=8, C=64] -> 64 tokens × 64 channels
```

映射固定为：

\[
y[b,h,w,c\cdot4+\Delta r\cdot2+\Delta c]
=x[b,2h+\Delta r,2w+\Delta c,c],
\]

其中 `Δr,Δc∈{0,1}`。逆变换 `depth-to-space` 使用完全相反的 index mapping。

该操作只能由 `view/reshape + permute` 构成：

- 无卷积；
- 无 pooling；
- 无平均；
- 无 nonlinear activation；
- 无 LayerNorm/RMSNorm；
- 无 learned merger；
- 对 FP32/BF16/FP16 都应满足 round-trip `torch.equal(x, D2S(S2D(x)))`。

### 7.2 数据与模型边界

canonical cache 继续保存原始 `256×16` latent，避免复制数据和改变 validation
membership。dataset 在构造 image span 前按以下顺序处理：

1. 将缓存恢复为 `[16,16,16]` 的空间布局；
2. 在 canonical layout 上做 horizontal flip augmentation；
3. 若 `f=2`，执行 lossless S2D；
4. flatten 为 `64×64`，再构造 64 个 image slots。

模型配置由 layout 派生，而不是允许互相矛盾的自由组合：

```text
f=1: image_tokens_per_img=256, image_latent_dim=16
f=2: image_tokens_per_img=64,  image_latent_dim=64
```

生成和 validation decode 时，在送入 VAE decoder 前执行 D2S，恢复
`16×16×16`。FID/IS 始终基于同一 VAE decoder 和同一像素分辨率。

### 7.3 Flow head 语义

S2D 后，一个 flow target 是一个完整的 `2×2×16=64` 维 block。历史 screen 中，
架构与超参数固定但逐 run 重训的 baseline contextual flow head 预测该 noisy block
的 velocity，同时读取已经可见的 clean blocks：

\[
v_i=f_\theta\!\left(
  x_{t,i},t_i,c_i,
  \{(z'_j,p'_j)\mid \sigma_j<\sigma_i\}
\right),
\qquad x_{t,i},z'_j\in\mathbb{R}^{64}.
\]

训练时，context mask 继续严格使用变换后 64-token permutation 上的
`sigma_j<sigma_i`；推理时，context 只包含当前已填充的 clean blocks。query 和
context position 都使用 8×8 block-local index，并由
`image_tokens_per_img=64` 重建 baseline flow head 的 fixed 2D position buffer。
`S2D` 不关闭或新增 flow-head cross-attention，只改变其 latent granularity 和派生
input/output shape。

S2D 会同时改变 token granularity、backbone sequence length、embedder input width
和 flow output width，这是该架构假设的真实含义。必须额外报告参数量、训练吞吐和
sampling wall time，不能把它描述成纯计算优化；但不能为了“参数严格相等”再加入
额外 learned bottleneck，因为那会破坏本消融的简洁性。

## 8. Screen 矩阵与正式 confirmation 范围

历史 screen 定义三个 proposal-level 因素：

- `S`：query stage coordinate；
- `R`：row/column 2D RoPE + observed content hidden 去 additive position；其中
  `R_a` 表示只去 additive position，`R_b` 表示只换 RoPE；
- `D`：lossless S2D factor 2。

历史 `2^3` factorial matrix：

| ID | S | R | D | 目的 |
| --- | ---: | ---: | ---: | --- |
| `E0` | 0 | 0 | 0 | 当前 image token embedder baseline |
| `E1` | 1 | 0 | 0 | stage 的原子主效应 |
| `E2` | 0 | 1 | 0 | 2D relation/content-position 解耦的原子主效应 |
| `E3` | 0 | 0 | 1 | lossless S2D 的原子主效应 |
| `E4` | 1 | 1 | 0 | stage × composite `R` system interaction |
| `E5` | 1 | 0 | 1 | stage × S2D interaction |
| `E6` | 0 | 1 | 1 | composite `R` × S2D system interaction |
| `E7` | 1 | 1 | 1 | 完整 proposal |

另加两个不属于 `2^3` 主矩阵、但用于拆解复合因素 `R` 的原子 control：

| ID | observed additive `P_xy` | backbone RoPE | 目的 |
| --- | --- | --- | --- |
| `E2a` | 去除 | 原 1D RoPE | 只测 content-position 解耦 |
| `E2b` | 保留 | row/column 2D | 只测二维 relation |

因此 `E2` 是 `E2a` 与 `E2b` 两个子因子的 **joint setting**，而不是默认可加的
`E2a + E2b`。只有比较 `E0/E2a/E2b/E2` 后，才能判断 `R` 的收益来自哪个子改动，
以及两者是否有 interaction。

历史 screen 按两阶段执行：

1. 第一阶段运行 `E0/E1/E2a/E2b/E2/E3`，同时观察三个 screen 因子并拆开 `R`；
2. 第二阶段运行 `E7` 并补齐 `E4/E5/E6`，以诊断 `R×D` 等 system interaction。

第一阶段只是调度 wave，不是淘汰 interaction candidates 的 gate。现有十点把
“去 observed additive position”和“启用 row/column RoPE”在主矩阵中捆成 `R`，所以它
严格回答的是“这十个预注册系统中谁最好”。若 `E2a/E2b` 一正一负、任一原子 control
优于 joint `E2`，或其差异落在近优区间，则继续补齐六个 partial-`R` 组合
（`S+E2a`、`S+E2b`、`D+E2a`、`D+E2b`、`S+D+E2a`、`S+D+E2b`），把搜索扩展为
四个二值开关的完整 `2^4`。未补齐时，结论不得表述成四开关空间的全局最佳。

触发后使用如下固定 ID，避免训练产物和汇总脚本对组合名称产生歧义：

| ID | S | 去 observed additive (`R_a`) | row/column RoPE (`R_b`) | D |
| --- | ---: | ---: | ---: | ---: |
| `E4a` | 1 | 1 | 0 | 0 |
| `E4b` | 1 | 0 | 1 | 0 |
| `E6a` | 0 | 1 | 0 | 1 |
| `E6b` | 0 | 0 | 1 | 1 |
| `E7a` | 1 | 1 | 0 | 1 |
| `E7b` | 1 | 0 | 1 | 1 |

原十点结果使用 `full` 汇总模式；触发后的完整 16 点结果使用 `expanded` 模式，并报告
`S/R_a/R_b/D` 的四因素主效应和 interaction。两种模式的命名是为了保留已有产物兼容性，
不是把十点矩阵称为四开关全空间。

### 8.1 2026-07-20 confirmation scope amendment

Expanded seed-42 screen 按预注册 selector 产生的原始确认集合为
`E0/E1/E2b/E2/E3/E4b/E4/E6b/E6/E7a`。在观察任何 confirmation metrics 前，
确认集合缩为以下六个 `factor=1` 变体：

| ID | stage | 去 observed additive | row/column RoPE |
| --- | ---: | ---: | ---: |
| `E0` | 0 | 0 | 0 |
| `E1` | 1 | 0 | 0 |
| `E2b` | 0 | 0 | 1 |
| `E2` | 0 | 1 | 1 |
| `E4b` | 1 | 0 | 1 |
| `E4` | 1 | 1 | 1 |

移除 `E3/E6b/E6/E7a`，原因是它们都启用了 S2D，而不是根据 confirmation
结果做事后筛选。范围变更记录在
`output/image_embedder_ablation/confirmation_scope_d1_only.json`。原始 screen
manifest 不改写：训练 checkpoint 继续绑定原 manifest digest；最终汇总必须同时校验
原 manifest、scope amendment、被保留的六个 ID、被移除的四个 ID，以及 amendment
早于 confirmation metrics 的声明。

正式机制分析以四开关 matched contrast 为准；不能把 `E4` 简写成纯粹的
`S×2D-RoPE` 效应，也不能把 `E6` 简写成纯粹的 `2D-RoPE×S2D` 效应，因为两者都把
`R_a` 与 `R_b` 同时打开。

不能只比较 `E0` 与 `E7`：即使 full model 变好，也无法知道收益来自哪个结构；
full model 变差时也无法识别互相抵消的正负效应。

### 8.2 2026-07-21 missing-factor extension：`Q`

既有 ID 全部保持原义并隐式表示 `Q=1`。新增实验不重命名 checkpoint，也不改写原
confirmation scope。六变体、三 seed 的正式 confirmation 已完成；`E2b/E2/E4b/E4`
均优于 `E0` 的 nominal mean FID，但 stage 没有形成稳定的独立收益。结合“指标接近时
偏好无 stage”的先验，Q follow-up 固定 `stage=0`，只在 `E2b/E2` 上做
observed-additive × mask-query-additive 的完整 `2×2`：

| 分析 cell | parent | stage | observed additive | backbone 2D RoPE | mask-query additive `P_xy` | 权威数据来源 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `E2b-Q1` | `E2b` | 0 | 1 | 1 | 1 | 直接复用旧 `E2b` seeds 43/44/45 |
| `E2b-Q0` | `E2b` | 0 | 1 | 1 | 0 | fresh Q0 seeds 43/44/45 |
| `E2-Q1` | `E2` | 0 | 0 | 1 | 1 | 直接复用旧 `E2` seeds 43/44/45 |
| `E2-Q0` | `E2` | 0 | 0 | 1 | 0 | fresh Q0 seeds 43/44/45 |

不新增 `E0-Q0/E1-Q0`：它们仍使用 sequence-1D RoPE，不能回答“mask query 完全使用
2D RoPE”的目标问题。若未来研究无 additive query 在 1D backbone 下的表现，应另立
问题，不能混入本 extension。

不再增加 seed-42 Q-screen。`E2b-Q1/E2-Q1` 只是汇总层的分析别名，分别直接指向旧
confirmation 的 `E2b/E2` seeds `{43,44,45}`，不要求复制或重新训练 Q1 checkpoint。
这里复用的是既有 Q1 checkpoint/metrics 作为控制，不是以 Q1 权重初始化 Q0。fresh
workload 只有 `E2b-Q0/E2-Q0 × seeds {43,44,45}` 共 6 个 run；每个 Q0 run 仍按同一
base-model、module-keyed initialization、DataLoader order 与 augmentation contract
独立端到端训练，固定架构的 contextual flow head 也随 run 重新训练。

旧 Q1 的准入门是：legacy 缺省字段必须与显式
`image_mask_position_mode=additive_2d` 在 checkpoint load、X0/XT mask forward 和
generation 路径上兼容；旧 confirmation 与 fresh Q0 还必须使用相同 seeds 以及相同
canonical evaluation noise/sample contracts。旧 Q1 metrics 保持不可变，通过独立的
hash-bound reuse map 进入分析，不能回填新字段伪装成 fresh Q-factor 产物。
历史模型源码 Git blob `00d1215378…`（raw SHA256 `cfd4056dff1e…`）与 Q0 注册源码
blob `a3ca758658…`（raw SHA256 `41e845cd5375…`）的静态 diff 也必须进入 bridge：差异只
允许新增默认 `additive_2d` 的 mask-mode 开关，并把旧版无条件执行的同一 lookup/add
包进 Q1 分支；该静态证据与 bitwise regression 缺一不可。

这项 extension 是在原 confirmation 完成后新增的 prospective study。历史 Q1 绑定
`output/image_embedder_ablation/confirmation_d1_summary.json`、原 run/metrics/checkpoint
哈希与历史 source manifest；fresh Q0 则绑定 Q-factor resolved-config、training
provenance、checkpoint、metrics、runtime source manifest 和 evaluator RNG contract。
因此四格分析不能宣称来自 byte-identical post-change source，而是通过显式
legacy-Q1 equivalence bridge 接受的 seed-aligned cross-source comparison。一般 runtime
source drift 仍一律拒绝，只有精确限定且可机器验证的等价性豁免例外。

**2026-07-22 execution amendment.** 被本更正取代的 four-fresh-cell 执行曾产生三份完整
`E2b-Q1` final EMA 与 metrics；它们全部标记为 superseded/out-of-scope，不得替代旧
`E2b` Q1 controls。误启动的 `E2-Q1` seeds 43/44/45 已停止，只有中途 checkpoint，均无
step-35920、final EMA 或 metrics；这些部分产物不得计入完成数、效应估计或候选排序。

该 amendment 另有一个与 legacy-Q1 bridge 相互独立的 evaluation-only source-equivalence
waiver：它只适用于 `E2-Q0` seeds 44/45、evaluation seed 42 和
`models/modeling_model/modeling_selfless_flow.py`。预注册 source manifest 为
`5be769a7f2f5d01b3749844caec79044d135db0f5f54be3cfc4328aad72b0f04`；该文件从
`41e845cd5375f50edb6733985763dd81d006c8d9816a4abc14fb143c50e7fd92` 变为
`e1fb61dc12bab86158912d6f85467e98974bd28645769b94f995f79370b8e5b3`，差异被限定为
formatting、import reordering 与 unused-import cleanup。冻结 CPython 3.12 bytecode
`d9ec0fa3a3ea3545a15e25ec8cbea939d480213bf722996468ae1508858b3c6f` 的 99 个 code
objects 与当前源码逐项比对；98 个非-module executable bodies 和 exception-table graph
均等价。该 waiver 不修改模型执行、不允许覆盖 metrics，并拒绝任何其他文件、SHA、
Python 版本、ID 或 seed 漂移。机器可读证据位于
`output/image_mask_position_ablation/source_drift_waiver/evaluation_source_equivalence.json`。

六份 Q0 metrics 全部产出后，还必须先生成版本化的
`configs/ablation/image_mask_position_q0_metrics_attestation_v1.json`，逐 run 固定 analysis
ID、training seed、唯一 metrics 路径和 raw SHA256；bridge 在该 manifest 缺失、self
digest 不一致或其 raw SHA 未被代码固定时一律 fail closed。`E2-Q0` seeds 44/45 的条目
还必须绑定实际 evaluation job 名、上述 waiver、专用 launcher、validation shim、冻结
bytecode 以及冻结/当前模型源码哈希；其余四个 Q0 条目则显式声明 registered source、
无 waiver、无 sidecar。这样有限但被事后改写的 FID/IS 也会被拒绝，而不只是拦截
`NaN/Inf`。该文件是 raw-hash pinned 的研究产物完整性记录，并结合 `inspire job wait`
的 terminal success 使用；它不冒充平台签名的 execution receipt。

每个 pair 报告：

\[
\Delta_Q(E2b,s)=FID(E2b\text{-}Q0_{\rm fresh},s)-FID(E2b_{\rm reused},s),
\]

以及

\[
\Delta_Q(E2,s)=FID(E2\text{-}Q0_{\rm fresh},s)-FID(E2_{\rm reused},s).
\]

同时报告 interaction `[E2-Q0 - E2] - [E2b-Q0 - E2b]`，IS 同理；这些跨 source
差值都标为 descriptive。最终先看 seed-aligned FID/IS；若
`E2-Q0` 相对 nominal best 的 mean FID 绝对差不超过 `0.5`、mean IS 绝对差不超过
`1.0`，按预先声明的简洁性偏好选择无 stage、无 observed additive、无 mask-query
additive 的 `E2-Q0`。单 seed 的 nominal 小数点差异不能改写该规则。

## 9. 固定的训练与评测协议

### 9.1 必须保持一致

- 同一个 Qwen3-0.6B-Base 初始化；
- 同一个固定但随各 run 重新训练的 baseline contextual flow-head architecture：
  depth 8、width 1280、
  8 attention heads、MLP ratio 1.0、zero-initialized residual gates；
- 同一 ImageNet-100 train/validation membership；
- 同一 global batch、optimizer steps、EMA、optimizer、LR schedule 和 seed；
- 同一 latent scaling factor、per-element input noise 和 CFG dropout；
- 无 `z_proj_ln`，无 runtime latent normalization；
- 相同 flow solver、NFE、temperature 与 CFG schedule；
- 正式评测均使用 10K validation prompts 和相同 real statistics。

Confirmation 固定 `image_space_to_depth_factor=1`，因此六个候选的 latent width、
image token 数、外层 reveal 步数与 flow-head input/output shape 完全相同。历史 S2D
screen 的速度只能作为被拒绝系统变体的诊断，不能参与正式架构选择或速度 Pareto。

Q extension 继续冻结上述全部协议，尤其不修改 flow-head position encoding。两个 Q0
cell fresh 重训，两个 Q1 cell 直接来自旧 confirmation；四个分析 cell 都使用 seeds
`{43,44,45}` 与相同 canonical evaluation seed/noise/sample manifests。Q0 与 Q1 的差值是
通过 legacy compatibility 与 source-equivalence gate 接受的 seed-aligned cross-source
estimate，必须标注 source revision confounding，不能称为 byte-identical same-source
causal contrast。legacy 缺字段与显式 `additive_2d` 必须先通过 bitwise regression。

### 9.2 FID/IS 选择规则

沿用 `docs/IMAGENET100_ABLATION.md` 的正式架构消融协议：

- EMA checkpoint；
- BF16 model forward；
- 100-step Heun；
- `spatial_halton`；
- `parallel_rate=1`；
- 所有主矩阵与 `E2a/E2b` 统一使用 baseline 已选定的 CFG=3.5，不为每个 architecture 重新
  sweep CFG；
- 以该 fixed-CFG FID/IS 选择 image-embedder architecture，避免把 inference tuning
  混入训练期架构差异；
- 只有最终胜出的候选可以在架构选择完成后补一个局部 CFG sweep，用于报告其自身
  FID/IS Pareto frontier；该 sweep 不回头改变本轮 architecture ranking。

第一轮固定 training seed 42 做架构筛选。最终 confirmation 不复用筛选 seed：`E0`
与所有近优、nominal FID/IS Pareto 或有显著速度优势的候选使用完全相同的独立
training seeds（预注册为 43/44/45）。决策报告逐 seed 的配对
`ΔFID = FID(candidate)-FID(E0)`、mean、sample SD、95% CI 和胜场；不能把不同 seed set
的非配对均值直接排序，也不能只给单次 nominal winner 补 seed。

为避免看完 seed-42 后选择性确认，expanded screen 的 confirmation set 原先预注册为 `E0`
加上下列条件的并集：nominal FID 距离 seed-42 最优不超过 1.0；位于 FID↓/IS↑ Pareto
frontier；或位于 FID↓/10K-throughput↑ Pareto frontier 且速度至少为 `1.5×E0`。汇总时
必须保存这个 screen-derived candidate manifest，seeds 43/44/45 对 manifest 中所有
候选使用完全相同的集合；不能只补 nominal winner。随后、且在任何 confirmation
metrics 产生前登记的 scope amendment 只删除其中全部 S2D 候选；最终六个候选仍必须
使用完全相同的 seeds 43/44/45。汇总器不得用缩小后的集合替换 checkpoint-bound 的
原 manifest digest，而应在原 provenance 之上额外校验 amendment。

固定 CFG=3.5 的 ranking 估计的是“共同固定 inference protocol 下的较优架构”。若要
声称各架构在自身最佳推理配置下的 best-achievable 结果，应在 confirmation 后对所有
finalists 使用同一个预注册 CFG grid；不能只给单 seed winner sweep 后反过来改变架构排名。

### 9.3 额外记录

每个 run 还应记录：

- resolved training-protocol fingerprint、training seed、final global step，以及最终
  checkpoint/EMA/HF artifact provenance；
- 相同参数形状的 cell 记录初始化 hash、样本顺序 hash 与 evaluator 随机流；如果架构
  改变了 RNG 消耗顺序，必须显式记录，不能仅凭相同 seed 宣称逐参数初始化配对；
- trainable/total parameter count；
- image sequence length 与 latent width；
- train images/s、tokens/s、peak memory；
- 10K sampling wall time；
- projected latent RMS、query RMS、backbone hidden RMS；
- 按 stage bucket 分组的 flow loss/velocity MSE；
- generation trace 中的 stage、position 和 fill order。

这些统计用于解释机制，不能替代 FID/IS 做最终选择。

## 10. 建议的配置接口

实现层为复现历史 screen 仍暴露四个原子开关；proposal-level 的 `R` 由其中两个
开关联合定义。正式 confirmation 强制 factor 1：

```yaml
model:
  image_query_stage_mode: none        # none | fixed_sincos
  image_observed_position_mode: additive_2d  # additive_2d | none
  image_mask_position_mode: additive_2d      # additive_2d | none；legacy/default Q1
  image_rope_mode: sequence_1d        # sequence_1d | row_col_2d
  image_space_to_depth_factor: 1      # confirmation 必须为 1；2 仅供历史复现
```

`image_mask_position_mode` 必须与 `image_observed_position_mode` 分开。前者只控制
mask/query seed，后者只控制 visible observed-latent content path。fresh Q0 extension
validator 要求 `image_mask_position_mode=none` 时
`image_rope_mode=row_col_2d`，并把 `Q0` 写入 run slug、config fingerprint 和 metrics。
历史 Q1 维持原 confirmation slug/metrics，不复制进 Q-factor 目录。

派生字段：

```text
image_tokens_per_img = 256 / factor^2
image_latent_dim     = 16  * factor^2
image_grid_side      = 16  / factor
```

不建议再暴露可独立设置的 `image_tokens_per_img=64`、`image_latent_dim=16` 等非法
组合。loader、model、flow head 和 decoder 必须从同一个 layout contract 推导形状。

flow head 不是第四个开关，而是所有实验必须满足的固定 invariant：

```yaml
model:
  image_flow_head_arch: contextual
  image_flow_depth: 8
  image_flow_width: 1280
  image_flow_mlp_ratio: 1.0
  image_flow_latent_mixer_heads: 8
  image_flow_latent_mixer_zero_init_gate: true
```

## 11. Repo 中的实施落点

### `models/modeling_model/modeling_selfless_flow.py`

- 将 `ImageTokenEmbedder.embed_latents` 拆成明确的 observed content path 与 query
  path；
- `row_col_rope` 下 observed latent 只返回 `z_proj(z)`；
- `embed_mask` 接收 local `(row,col)`、独立的 `image_mask_position_mode` 与可选
  normalized stage；`none` 时只跳过 additive lookup/addition，不能丢掉供 RoPE 和
  flow head 使用的 local position；
- 生成 2-axis multimodal position ids；
- `Qwen3RotaryEmbedding` 支持 interleaved row/column phase；
- `apply_rotary_pos_emb` 对 X0 Q、XT Q、X0 K 使用同一 token coordinate，V 不变；
- single-stream generation 调用 `self.model` 时传入 `current_sigma` 或已计算的
  image-local stage，保证训练/推理一致。
- 保持 `FlowLoss(head_arch="contextual")` 与 `_flow_context` 路径开启；每次 ODE
  solve 都把已填充 clean latents、context mask、query positions 和 context positions
  传给 baseline flow head。

### `models/modeling_model/image_flow_loss.py`

- 保留现有 `ContextualFlowTransformerHead`、`ContextualFlowBlock`、strict training
  context mask 和 inference clean-context K/V cache；
- 不在 `S/R/D` 配置中选择 `TokenFlowMLPHead`，也不增加新的 flow-head variant；
- `f=1/2` 只派生 `target_channels=16/64` 与
  `image_tokens_per_img=256/64`，其余 head hyperparameters 不变。

### `models/modeling_model/image_position_utils.py`

- 增加 row/column coordinate builder；
- 增加 pure-text `(p,p)` compatibility helper；
- 增加固定 stage sin/cos builder；
- 所有 coordinate buffer 使用 FP32 构造，再转换到 model dtype。

### 建议新增 `models/modeling_model/image_latent_layout.py`

- `space_to_depth_2d(latents, factor)`；
- `depth_to_space_2d(latents, factor)`；
- shape/layout validation；
- 不包含 parameter 或 module state。

### `utils/dataset_imagenet_flow_cache.py`

- cache shape validation 以 canonical `256×16` 为准；
- augmentation 在 canonical layout 上完成；
- sequence 构造前按 layout factor 变换 latent 和 image slot 数；
- sigma permutation 在变换后的 64/256 个 image token 上生成。

### `pretrain/train_selfless_flow.py` 与 evaluator

- 从 factor 推导 target shape；
- 将同一 layout 下的 clean target/context latents 一并传给 contextual flow head；
- `f=2` 时将 flow-head query/context positions 和 fixed 2D position buffer 派生为
  64-token 8×8 block grid；
- validation/sample decode 前执行 D2S；
- FID/IS metadata 记录三个 architecture flags 与派生 shape；
- 输出目录和 experiment name 必须包含 `S/R/D` ID，禁止覆盖历史结果。
- fresh Q0 metadata 额外记录 `image_mask_position_mode`、base variant 和 `Q0`；`Q0`
  目录不得复用或覆盖既有 `E2b/E2/E4b/E4` 产物。旧 Q1 metrics 保持不可变，由
  hash-bound reuse/equivalence manifest 关联，不能为补写 Q 字段而改动旧文件。

### Stage checkpoint-load invariant

`image_stage_embed` 是由 config 唯一确定的 fixed sin/cos table，不属于 learned
checkpoint state。HF `from_pretrained` 的 meta-device materialization 可能把
`persistent=False` buffer 变成未初始化 storage，因此模型在加载 checkpoint 后必须显式、
确定性地重建该 table，并在第一次使用前验证其 shape/device。不能用 `nan_to_num`、clamp
或跳过坏样本替代重建。

2026-07-19 的 E1/E7 初次评测触发了这一问题：首个非有限值来自 generation step 1 的
`image_mask_embed.stage_lookup`，而不是 flow head、CFG 合成或 Heun ODE；旧 E1/E7 FID/IS
因此无效，必须在修复后重评。评测仍保留 generated-latent finite assertion 作为故障检测，
但该 assertion 不是数值修复。修复后在 dev-wjx 复跑与正式 rank 相同的 local
batch 64、`parallel_rate=1`、Heun-100、CFG 3.5 路径，E1/E7 各 64 个样本均有限；
这只是实现健康检查，不能代替 10K 正式指标。

所有 `S=1` 正式 metrics 必须记录
`fixed_sincos_nonpersistent_rebuild_v1` implementation contract；汇总器拒绝缺少该
provenance 的 stage 结果。正式 evaluator 还必须在 clamp/写 metrics 前验证 decoded
image、Inception feature/logit、FID/IS 和 latent diagnostics 全部有限；任何非有限值都让
job 失败，不能写成 `null` 后继续。已有合法 `metrics.json` 的目录默认禁止静默复用；
这不禁止本文显式声明、哈希绑定并通过 compatibility gate 的历史 Q1 control reuse。
最终 bridge 汇总器必须拒绝 superseded 的 fresh `qf-e2b-q1` 完整产物和误启动的
`qf-e2-q1` 部分产物。

## 12. 必须通过的测试

### Stage

- prompt 长度变化不改变相同 image-local rank 的 stage；
- raw sigma 带 offset 时仍得到 `[0,1]` 的正确 image-local stage；
- observed latent embedding 对 stage 完全不敏感；
- XT mask query 与 inference X0 unfilled query 使用同一 stage encoding；
- 同一 parallel generation round 的候选 stage 相同；
- flow timestep `t` 的代码路径不受影响；
- baseline flow head 的 visible clean-latent context set 不因是否启用 `P_s` 而改变。
- 模拟 checkpoint materialization 后的损坏/未初始化 non-persistent stage buffer 时，
  第一次使用必须重建出精确 fixed table，且不得改动 learned projector、position buffer
  或 stage scale。
- 已就绪模型再经过 `.to()` / `.to_empty()` 等 storage 变换时，ready 状态必须自动失效，
  不允许依赖调用方手工清 flag；下一次使用必须重建出相同的 fixed table。
- tiny HF `save_pretrained` → `from_pretrained(low_cpu_mem_usage=True)` round-trip 后，
  stage query 输出必须有限，且重建 fixed table 不得改变任何 learned/persistent state。
- `S=1` formal result 缺少 stage-buffer implementation contract 时，汇总必须拒绝；
  confirmation seed set 必须精确为预注册的 `{43,44,45}`。

### 2D RoPE

- 所有 text coordinate 为 `(p,p)` 时，新 RoPE 与旧 1D RoPE 数值一致；
- 同 row token 的相对 phase 只随 column difference 改变，反之亦然；
- image anchor 平移不改变 image-image relative phase；
- Q/K 旋转但 V 与 residual hidden 不旋转；
- observed image embedding 不含 additive `P_xy`；
- `Q=1` reference 中，query 在零个或一个 visible image key 时仍因 absolute `P_xy`
  可区分位置；
- `R` 不改变 flow-head cross-attention、context mask 或其 fixed 2D position buffer。

### Mask-query absolute position (`Q` extension)

- legacy 缺字段的 `Q=1` 与显式 `additive_2d` 在 `embed_mask` forward、checkpoint load
  和 generation behavior 上 bitwise/tolerance-compatible；这是旧 Q1 直接复用的准入门；
- `Q=0` 时不同 local positions 的 raw mask seed 在 stage 相同条件下完全相同，且代码不
  lookup、不 scale、不 add `image_pos_embed`；
- `Q=0` 仍生成并传递合法 `(row,col)` coordinates，backbone Q/K 的 row/column RoPE
  phase随位置变化；
- `Q=0,S=1` 只保留 stage additive encoding，不存在 spatial-position 泄漏；
- observed latent embedding 不受 `image_mask_position_mode` 影响；
- flow-head query/context position、strict sigma mask、visible context set 和 K/V cache
  不受 `Q` 影响；
- 零个、一个和多个 visible image keys 的 train/inference forward 均 finite；
- config/summary 拒绝把 `Q0` 与 `sequence_1d` 注册为本 extension 的正式变体；
- fresh Q0 provenance 必须包含 base ID、`Q`、seed、implementation contract、
  training/evaluator manifests 和 checkpoint digest；reused Q1 保留原 confirmation
  provenance，并由独立 reuse/equivalence manifest 绑定，不要求篡改旧 metrics 补写 Q 字段。

### S2D

以下测试只保证历史 S2D 产物的实现可复现，不构成把 S2D 纳入 confirmation 的依据：

- `D2S(S2D(x))` 在 FP32/BF16/FP16 上逐元素 `torch.equal`；
- 固定 index pattern 验证 subpixel channel order；
- canonical horizontal flip 后再 S2D 与预期空间位置一致；
- factor 1 为 identity；
- factor 2 的 shape 严格为 `256×16 -> 64×64`；
- contextual flow head 在 factor 2 下接收 64 维 noisy query 与 64 维 clean context，
  并输出 64 维 velocity；
- training context mask 在 64-token layout 上仍严格满足 `sigma_j<sigma_i`；
- inference flow context 只包含当前已填充 block，且 query/context positions 均合法；
- 生成结果 D2S 后严格恢复 VAE 所需的 `16×16×16`；
- gradient 能通过 reshape/permute 正常回传。

### Regression

- `E0` 的 forward、loss 和 generation behavior 不变；
- `E0/E1/E2a/E2b/E2/E3/E4a/E4b/E4/E5/E6a/E6b/E6/E7a/E7b/E7` 均满足
  `image_flow_head.head_arch == "contextual"` 且
  `uses_latent_mixer == True`，不存在静默回退到 `token_mlp` 的路径；
- pure-text batch behavior 不变；
- strict `sigma[kv] < sigma[q]` mask 不变；
- repo 中不存在重新启用 `z_proj_ln` 或 runtime latent RMS normalization 的路径。

## 13. 结果解释与下一步决策

Seed-42 的完整 factorial screen 可按主效应和 interaction 做诊断：

\[
\Delta_S(R,D)=FID(S=1,R,D)-FID(S=0,R,D),
\]

`R`、`D` 同理。FID 越低越好，因此负值表示收益。

- 若 `S` 在所有 `(R,D)` 下都接近零，说明 strict context topology 已足以表达
  reveal progress；删除 stage，保持架构简洁。
- 若 `R` 稳定改善，说明 position 应主要存在于 query identity 与 Q/K phase，而不应
  混进 observed latent hidden。
- `D` 相关差异只描述改变生成原子、序列长度和 flow-head 接口后的历史 system-level
  结果，不再解释为 image embedder 的主效应，也不参与 confirmation selector。
- 若只有 `E7` 改善，说明存在强 interaction，不能把三项收益分别宣称为独立贡献。
- 若 FID nominally 接近但 IS/速度明显不同，应保留 Pareto candidates 并补 seed，
  不依据单次小数点差异立即定架构。

原 confirmation 已完成 `E0/E1/E2b/E2/E4b/E4` 的三 seed 评测。权威 Q1 controls 为：
`E2b/Q1` FID `25.246329 ± 0.300965`、IS `61.580519 ± 0.490959`；`E2/Q1` FID
`24.961536 ± 0.423134`、IS `61.325307 ± 0.343701`（均为三 seed mean ± sample SD）。

fresh Q0 结果为：`E2b-Q0` FID `25.221092 ± 0.086807`、IS
`61.793315 ± 0.839019`；`E2-Q0` FID `25.490735 ± 0.251393`、IS
`60.715986 ± 0.483666`。Q extension 的权威分析集因此完整地由 6 个 reused Q1 results
和 6 个 fresh Q0 results 组成；没有等待或使用 12 个 fresh matched runs，superseded
fresh Q1 和误启动的部分 checkpoint 均未进入汇总。

跨 source 的 seed-aligned 描述性差值显示：在 `E2b` 上移除 mask-query additive 的
`Q0-Q1` 为 FID `-0.025237`、IS `+0.212795`，基本持平；在 `E2` 上则为 FID
`+0.529200`、IS `-0.609321`，nominally 变差。后者不能表述为同源码 causal Q effect，
但足以按预注册决策门拒绝自动切换到完全无 additive 的 `E2-Q0`。

最终 strict selector 的 nominal best 是 `E4b`：FID `24.917306`、IS `61.577339`。
`E2-Q0` 相对它为 FID `+0.573429`、IS `-0.861353`；IS 落在 `1.0` margin 内，FID
却超过 `0.5` margin `0.073429`，所以不满足 simplicity rule，正式推荐保留 `E4b`：
fixed-sincos stage、observed additive 2D position、mask-query additive 2D position、
row/column 2D RoPE、`image_space_to_depth_factor=1`，以及固定架构但逐 run 重训的
contextual flow head。不能在看到结果后放宽 `0.5` 门槛。

若未来把部署简洁性设为高于当前硬门槛的目标，`E2-Q1` 是应保留的 no-stage 次选：
它相对 `E4b` 仅为 FID `+0.044229`、IS `-0.252032`，且去掉 observed additive；但它仍
保留 mask-query additive，因此不是“完全无 additive”方案，也不是本轮 strict winner。
所有结论限于当前候选空间与固定 CFG=3.5；下一轮 flow-head position ablation 应围绕
`E4b` 主方案，并可把 `E2-Q1` 作为简洁性 control。

机器可读最终汇总位于
`output/image_mask_position_ablation/legacy_bridge_summary.json`（raw SHA256
`d1b169530c00e60dcee85afdbbd84e257999337e16fd22cd6d78755ad7c9750f`），12-run 明细位于
同目录的 `legacy_bridge_runs.csv`。Q0 metrics attestation raw SHA256 为
`30491a0d5a4e24cbe44a523fa0b644a2a423ed248996f393faeda10049ac9cf3`；两项最终评测的
blocking-wait terminal records 位于 `evaluation_job_terminal_receipts.json`。

汇总 JSON 中 seed-42 的 `best_by_fid` / `ranking_by_fid` 只表示 nominal screen，不能
替代上述 confirmation selector；即便 exact FID tie 在展示排序中用 IS 排列，也不构成
最终架构的统计 tie-break。

特别地，`D` 会改变随机初始化所消费的 RNG 数量；若 DataLoader shuffle 继续隐式使用
全局 RNG，则相同整数 seed 也不保证 `D=0/1` 看到相同 sample order。因此 seed-42 只作
screen，不把跨 `D` 的单次差值称为严格 paired estimate。启动 confirmation 前必须给
DataLoader 使用独立、显式 seed 的 generator，并把实际 order contract/hash 写入产物；
E0 与全部 finalists 随后用同一新协议重训。

实现已预留 `training.dataloader_shuffle_seed`：seed-42 screen 保持 unset 以免在矩阵中途
改变协议；confirmation 的每个 run 必须将它显式设为该 run 的 training seed。该 generator
同时控制 shuffle 与 worker base seed，不依赖模型初始化之后的全局 RNG state。

## 14. 思想来源而非实现依赖

Qwen3.5 的公开实现将 multimodal RoPE 分成多坐标分量，文本 span 将同一个 1D
position 复制到各坐标槽，视觉 token 才使用真实 T/H/W grid；其频率组织也采用
interleaved MRoPE。我们只借鉴“**共享坐标槽、按模态解释坐标、位置作用于 Q/K**”
这一原则。

- [Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Official Qwen3.5-9B configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)

本 proposal 的 stage 语义、历史 lossless S2D 实现、two-stream Selfless attention 和 query/content
分工均由当前生成任务自身决定，不是 Qwen3.5 组件的复刻。
