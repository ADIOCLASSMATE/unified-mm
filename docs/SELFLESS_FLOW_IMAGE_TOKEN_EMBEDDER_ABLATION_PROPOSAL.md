# Selfless-Flow Image Token Embedder：原子级架构消融 Proposal

Status: proposed; revised 2026-07-19 after the flow-head architecture ablation.

## 1. 目标与结论

本轮消融只回答一个问题：在 Selfless-Flow 中，连续图像 latent 应该怎样成为
Qwen backbone 的 token，才能得到更好的生成架构？

Proposal 包含三个相互独立的架构假设：

1. **Stage query coordinate**：把外层 image-token reveal stage 显式坐标化到图像
   query；
2. **Row/column factorized 2D RoPE**：图像 Q/K 使用行列分解的二维 RoPE，
   observed image hidden/value path 不再混入 additive spatial position；
3. **Lossless space-to-depth**：使用纯 reshape/permutation 将相邻 `2×2` latent
   合并为一个 token，不丢失任何 VAE latent 元素。

三项改动必须先做原子消融，再做组合消融。它们不能被打包成一个
“Qwen3.5-style module”，因为我们借鉴的只是其**不同模态共享多轴坐标槽**的思想，
而不是照搬 Qwen3.5 的视觉塔、T/H/W 语义、partial rotary ratio 或 patch merger。

最终候选架构写成：

\[
\begin{aligned}
z' &= \operatorname{S2D}_f(z),\\
h^{X0}_{\mathrm{image}} &= W_z z' + b_z,\\
h^{Q}_{\mathrm{image}} &= e_{\mathrm{mask}} + P_{xy}(r,c)
    + \mathbb{1}_{\mathrm{stage}}P_s(s),\\
Q'_{\mathrm{image}},K'_{\mathrm{image}}
    &= \operatorname{RoPE}_{row,col}(Q_{\mathrm{image}},K_{\mathrm{image}}),\\
V'_{\mathrm{image}} &= V_{\mathrm{image}}.
\end{aligned}
\]

其中 `f∈{1,2}`。三个二值开关分别控制 `P_s`、二维 RoPE/content-position
解耦和 `S2D_2`。

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
  `contextual` head；`E0`--`E7` 均冻结相同的 head class、depth、width、MLP ratio、
  attention heads、gating 和 optimization protocol。只有 `S2D` 必然引起的
  latent input/output width 与 token-count 派生形状可以改变。

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

## 7. 消融 C：完全无损的 `space-to-depth`

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

S2D 后，一个 flow target 是一个完整的 `2×2×16=64` 维 block。冻结的
baseline contextual flow head 预测该 noisy block 的 velocity，同时读取已经可见的
clean blocks：

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

## 8. 三因素实验矩阵

定义：

- `S`：query stage coordinate；
- `R`：row/column 2D RoPE + observed content hidden 去 additive position；
- `D`：lossless S2D factor 2。

完整的 `2^3` factorial matrix：

| ID | S | R | D | 目的 |
| --- | ---: | ---: | ---: | --- |
| `E0` | 0 | 0 | 0 | 当前 image token embedder baseline |
| `E1` | 1 | 0 | 0 | stage 的原子主效应 |
| `E2` | 0 | 1 | 0 | 2D relation/content-position 解耦的原子主效应 |
| `E3` | 0 | 0 | 1 | lossless S2D 的原子主效应 |
| `E4` | 1 | 1 | 0 | stage × 2D RoPE interaction |
| `E5` | 1 | 0 | 1 | stage × S2D interaction |
| `E6` | 0 | 1 | 1 | 2D RoPE × S2D interaction |
| `E7` | 1 | 1 | 1 | 完整 proposal |

推荐两阶段执行：

1. 第一阶段先跑 `E0/E1/E2/E3`，确认三个原子主效应；
2. 第二阶段至少跑 `E7`。若目标是可靠确定最优架构，而不是只验证 full model，
   则补齐 `E4/E5/E6`，因为特别是 `R×D` 很可能存在强 interaction。

不能只比较 `E0` 与 `E7`：即使 full model 变好，也无法知道收益来自哪个结构；
full model 变差时也无法识别互相抵消的正负效应。

## 9. 固定的训练与评测协议

### 9.1 必须保持一致

- 同一个 Qwen3-0.6B-Base 初始化；
- 同一个已冻结的 baseline contextual flow-head architecture：depth 8、width 1280、
  8 attention heads、MLP ratio 1.0、zero-initialized residual gates；
- 同一 ImageNet-100 train/validation membership；
- 同一 global batch、optimizer steps、EMA、optimizer、LR schedule 和 seed；
- 同一 latent scaling factor、per-element input noise 和 CFG dropout；
- 无 `z_proj_ln`，无 runtime latent normalization；
- 相同 flow solver、NFE、temperature 与 CFG schedule；
- 正式评测均使用 10K validation prompts 和相同 real statistics。

S2D 的序列更短，因此“相同训练样本/optimizer steps”是主质量控制；同时单独报告
实际 token 数、FLOPs proxy 和 wall time，不宣称它与 baseline 等计算量。

### 9.2 FID/IS 选择规则

沿用 `docs/IMAGENET100_ABLATION.md` 的正式架构消融协议：

- EMA checkpoint；
- BF16 model forward；
- 100-step Heun；
- `spatial_halton`；
- `parallel_rate=1`；
- `E0`--`E7` 统一使用 baseline 已选定的 CFG=3.5，不为每个 architecture 重新
  sweep CFG；
- 以该 fixed-CFG FID/IS 选择 image-embedder architecture，避免把 inference tuning
  混入训练期架构差异；
- 只有最终胜出的候选可以在架构选择完成后补一个局部 CFG sweep，用于报告其自身
  FID/IS Pareto frontier；该 sweep 不回头改变本轮 architecture ranking。

第一轮可使用单 seed 做架构筛选；最终候选至少应与 `E0` 一起补多 seed，避免把小于
run-to-run variance 的差异解释成架构结论。

### 9.3 额外记录

每个 run 还应记录：

- trainable/total parameter count；
- image sequence length 与 latent width；
- train images/s、tokens/s、peak memory；
- 10K sampling wall time；
- projected latent RMS、query RMS、backbone hidden RMS；
- 按 stage bucket 分组的 flow loss/velocity MSE；
- generation trace 中的 stage、position 和 fill order。

这些统计用于解释机制，不能替代 FID/IS 做最终选择。

## 10. 建议的配置接口

只暴露三个正交的架构开关：

```yaml
model:
  image_query_stage_mode: none        # none | fixed_sincos
  image_position_mode: additive_2d    # additive_2d | row_col_rope
  image_space_to_depth_factor: 1      # 1 | 2
```

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
- `embed_mask` 接收 local `(row,col)` 与可选 normalized stage；
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

## 12. 必须通过的测试

### Stage

- prompt 长度变化不改变相同 image-local rank 的 stage；
- raw sigma 带 offset 时仍得到 `[0,1]` 的正确 image-local stage；
- observed latent embedding 对 stage 完全不敏感；
- XT mask query 与 inference X0 unfilled query 使用同一 stage encoding；
- 同一 parallel generation round 的候选 stage 相同；
- flow timestep `t` 的代码路径不受影响；
- baseline flow head 的 visible clean-latent context set 不因是否启用 `P_s` 而改变。

### 2D RoPE

- 所有 text coordinate 为 `(p,p)` 时，新 RoPE 与旧 1D RoPE 数值一致；
- 同 row token 的相对 phase 只随 column difference 改变，反之亦然；
- image anchor 平移不改变 image-image relative phase；
- Q/K 旋转但 V 与 residual hidden 不旋转；
- observed image embedding 不含 additive `P_xy`；
- query 在零个或一个 visible image key 时仍因 absolute `P_xy` 可区分位置；
- `R` 不改变 flow-head cross-attention、context mask 或其 fixed 2D position buffer。

### S2D

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
- `E0`--`E7` 均满足 `image_flow_head.head_arch == "contextual"` 且
  `uses_latent_mixer == True`，不存在静默回退到 `token_mlp` 的路径；
- pure-text batch behavior 不变；
- strict `sigma[kv] < sigma[q]` mask 不变；
- repo 中不存在重新启用 `z_proj_ln` 或 runtime latent RMS normalization 的路径。

## 13. 结果解释与下一步决策

完整 factorial 结果应按主效应和 interaction 解读：

\[
\Delta_S(R,D)=FID(S=1,R,D)-FID(S=0,R,D),
\]

`R`、`D` 同理。FID 越低越好，因此负值表示收益。

- 若 `S` 在所有 `(R,D)` 下都接近零，说明 strict context topology 已足以表达
  reveal progress；删除 stage，保持架构简洁。
- 若 `R` 稳定改善，说明 position 应主要存在于 query identity 与 Q/K phase，而不应
  混进 observed latent hidden。
- 若 `D` 改善，说明以局部 `2×2` block 为生成原子比逐 latent site 更合适；同时要看
  它是否依赖 2D RoPE 才成立。
- 若只有 `E7` 改善，说明存在强 interaction，不能把三项收益分别宣称为独立贡献。
- 若 FID nominally 接近但 IS/速度明显不同，应保留 Pareto candidates 并补 seed，
  不依据单次小数点差异立即定架构。

最终选择 minimum-FID 且经多 seed 验证的最简 Pareto architecture，再围绕该架构安排
下一轮 backbone/attention ablation。

## 14. 思想来源而非实现依赖

Qwen3.5 的公开实现将 multimodal RoPE 分成多坐标分量，文本 span 将同一个 1D
position 复制到各坐标槽，视觉 token 才使用真实 T/H/W grid；其频率组织也采用
interleaved MRoPE。我们只借鉴“**共享坐标槽、按模态解释坐标、位置作用于 Q/K**”
这一原则。

- [Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 Transformers implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Official Qwen3.5-9B configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)

本 proposal 的 stage 语义、lossless S2D、two-stream Selfless attention 和 query/content
分工均由当前生成任务自身决定，不是 Qwen3.5 组件的复刻。
