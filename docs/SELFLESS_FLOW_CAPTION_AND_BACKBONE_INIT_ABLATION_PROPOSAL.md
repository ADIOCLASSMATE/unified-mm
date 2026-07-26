# Selfless-Flow Caption Conditioning × Backbone Initialization 消融 Proposal

Status: proposed and blocked; written 2026-07-21. This document specifies experiments
only. It does not authorize implementation, training, or evaluation jobs.

## 1. 研究问题与一句话结论

本轮消融回答两个彼此相关的问题：

1. 在完全相同的 ImageNet 图像上，把短 class condition 换成完整的图像 caption，是否能
   改善生成质量、细粒度图文一致性和条件利用率？
2. 这种 richer text supervision 是否会放大 Qwen language-backbone pretraining 相对
   fully-from-scratch training 的优势？

核心实验是一个 `caption payload × backbone initialization` 的 `2×2` factorial。
四个 cell 固定 `image_backbone_variant=E2-Q0` 和默认 `DF1-FH4` flow head，只改变
条件文本与 Qwen backbone 初始化。最重要的统计量不是某个 cell 的单次 FID，而是
caption 收益在 pretrained 与 scratch 下是否存在稳定 interaction。

## 2. 严格执行顺序与启动门槛

本 proposal **不得与架构搜索并行启动正式训练**。执行顺序固定为：

```text
Image-backbone ablation archive
    -> flow-head screens archived
    -> freeze E2-Q0 + DF1-FH4 as the factorial baseline
    -> caption × initialization ablation
```

正式数据/初始化训练只有在以下条件全部满足后才能启动：

1. [`SELFLESS_FLOW_IMAGE_TOKEN_EMBEDDER_ABLATION_PROPOSAL.md`](SELFLESS_FLOW_IMAGE_TOKEN_EMBEDDER_ABLATION_PROPOSAL.md)
   已归档结果，并把 runtime 收敛为 `E2-Q1 / E2-Q0 / E2b-Q0`；主实验固定默认
   `E2-Q0`；
2. [`SELFLESS_FLOW_HEAD_BASELINE.md`](SELFLESS_FLOW_HEAD_BASELINE.md)
   已固定 active runtime 为 `DF1-FH0 / DF1-FH4`，本 factorial 使用默认
   `DF1-FH4`；
3. `image_backbone_variant=E2-Q0`、flow-head architecture/position mode、
   flow-head depth/width 和
   inference protocol 已写入 immutable architecture manifest；
4. manifest 包含 config digest、代码 commit、选择指标、训练 seeds 和有效 checkpoint；
5. 四个数据/初始化 cell 都从该 architecture manifest 构造，禁止为任一 cell 单独修改
   架构或 flow head。

`E2-Q1` 和 `E2b-Q0` 只允许用于独立的 backbone-robustness replication；不能并入主
`2×2` 形成事后挑选。不得恢复 stage、S2D、sequence-1D image RoPE 或独立
observed/mask-position 开关。`DF1-FH0` 同样只能作为单独、完整复制四个 cell 的
flow-head robustness replication，不能在主 `2×2` 内逐 cell 挑选。旧 E2-Q1 的
backbone confirmation 不重训；这里的四个 cell 因为改变
caption/initialization/data protocol，仍是新的正式训练。

在上述门槛前只允许完成数据审计、token-length profile、packing 等价性测试和吞吐基准；
这些工作不得查看任何本轮生成质量指标，也不得据此选择模型 cell。

## 3. 因素、estimand 与预注册假设

定义两个二值因素：

- `C=0`：canonical ImageNet class name 作为 condition payload；
- `C=1`：对应图像的完整 `recaption_short` 作为 condition payload；
- `I=0`：加载固定 Qwen3-0.6B-Base backbone 权重；
- `I=1`：相同 backbone config 完全随机初始化。

预注册假设：

- **H1：caption main effect。** 在相同图像曝光量下，`C=1` 提升细粒度图文一致性；
- **H2：pretraining main effect。** 在固定训练预算下，`I=0` 收敛更快；
- **H3：positive interaction。** caption 相对 class 的收益在 pretrained backbone 下更大，
  因为 class-only condition 的语言熵过低，不能充分调用语言预训练；
- **H4：class controllability guardrail。** caption training 不应显著损害标准 class-prompt
  ImageNet FID、IS 和 classifier accuracy；
- **H5：sample-efficiency distinction。** 固定预算结果只支持预训练的 sample-efficiency
  结论；只有额外的 scratch-to-convergence 实验才能支持 best-achievable 结论。

FID 越低越好。对任一指标 `m`，报告：

\[
\begin{aligned}
\Delta_C^{\mathrm{pre}} &= m(C=1,I=0)-m(C=0,I=0),\\
\Delta_C^{\mathrm{scratch}} &= m(C=1,I=1)-m(C=0,I=1),\\
\Delta_I^{\mathrm{class}} &= m(C=0,I=1)-m(C=0,I=0),\\
\Delta_I^{\mathrm{caption}} &= m(C=1,I=1)-m(C=1,I=0),\\
\Gamma_{C\times I} &= \Delta_C^{\mathrm{pre}}-\Delta_C^{\mathrm{scratch}}.
\end{aligned}
\]

不能只比较 `caption+pretrained` 与 `class+scratch`；该对比同时改变两个因素，无法归因。

## 4. 数据集与 membership 契约

### 4.1 ImageNet 与 CSFM caption 的关系

CSFM-ImageNet1K-Caption 提供 ImageNet-1K 图像对应的 caption metadata，而不是第二份
可独立拼接的图像数据。正式数据集通过 `path` 和 `id` 将 `recaption_short` 一一关联到
本地 ImageNet 图像/latent：

- [CSFM-ImageNet1K-Caption dataset card](https://huggingface.co/datasets/junwann/CSFM-ImageNet1K-Caption)
- [Better Source, Better Flow paper](https://arxiv.org/abs/2602.05951)

使用 caption metadata 不等于复现论文中的 condition-dependent source distribution。
本 proposal 的实验因素仅是 condition text 与 backbone initialization；flow source
distribution 保持不变。

### 4.2 正式 membership

- 正式实验使用同一份 ImageNet-1K train membership 和同一份 validation membership；
- 四个 cell 只使用同时具有合法图像 latent、manifest row 和非空 caption 的交集；
- `caption_missing_policy=error`，禁止对 caption cell 静默回退到 class name；
- 任何缺失、重复、path/id 冲突都必须在训练前失败；
- 若确需排除坏样本，必须在查看任何训练结果前生成共同 exclusion manifest，四个 cell
  使用完全相同的剩余图像；
- ImageNet-100 可以用于 loader smoke，但不能替代正式 ImageNet-1K factorial 结论。

caption join manifest 至少记录：`img_id`、relative source path、synset、caption source
row、原始 UTF-8 文本 hash、token-id hash 和最终 serialized length。manifest 在四个
cell 启动前冻结并记录 SHA-256。

### 4.3 Caption 质量审计

模型生成的 recaption 可能包含错配、幻觉、OCR 文本或过度细节。正式训练前必须发布：

- caption 字符数和 token 数的 min/median/p90/p95/p99/max；
- 空 caption、重复 caption、重复 image id、path/id 不一致的数量；
- 每个 class 的 coverage；
- 固定随机种子的人工抽检集合及错误类别统计；
- 可选的冻结图文相似度诊断，但不能根据 validation generation metric 选择过滤阈值。

主实验优先保留原始完整 caption distribution。若质量问题迫使过滤，过滤规则必须在训练前
锁定，并同时约束 class arm 的 image membership。

## 5. Condition serialization：只改变 payload

主实验只做 text-to-image，不混入 image-to-text task。每个样本的逻辑结构为：

```text
fixed_t2i_prefix + payload + <|boi|> + 256 image slots + <|eoi|> + <eos>
```

其中：

- `fixed_t2i_prefix` 在四个 cell 中逐 token 相同，且只使用一个固定模板；
- `C=0` 的 `payload` 是 canonical class name；
- `C=1` 的 `payload` 是未经截断的完整 `recaption_short`；
- 不随机抽取 prompt templates，因为 template sampling 会引入第三个数据因素；
- `caption_sequence_modes=[t2i]`，不混入 `i2t`；
- `label_text=false`、`lambda_text=0`，与 image-flow 架构实验一致；
- CFG dropout 作用于完整 text condition，概率与架构 winner 完全一致。

建议冻结的模板为：

```text
Generate an image matching this description: {payload}
```

若最终选择其他模板，必须在 tokenization manifest 生成前登记，并在 class/caption arm 中
完全相同。为了与历史 raw-class checkpoint 建立联系，可以增加一个 `raw class name`
bridge run，但它不属于 `2×2` 主矩阵，也不能替代 `C=0,I=0` 的重训。

## 6. 无截断的 segment-aware packing

### 6.1 Hard contract：零截断、零静默丢样本

当前 loader 中 `caption_max_tokens` 和固定 `max_seq_length` 都可能裁掉文本。本 proposal
禁止这两条路径：

- `caption_max_tokens=0/none` 表示保留 tokenizer 产生的全部 token；
- 不调用任何按 `max_seq_length` 裁剪 prefix/suffix 的逻辑；
- 不删除超长 caption，不用 class fallback，不从头尾截取，也不做摘要；
- 训练前对每条 serialized sample 记录 expected token count；collate 后逐样本验证 token
  count、token-id hash 和 image span；
- 如果最长样本超过 backbone context window，preflight 直接失败并要求扩大合法 context，
  绝不能自动截断后继续。

“没有触发 warning”不算无截断证明。正式 artifact 必须包含：

```text
num_source_samples == num_joined_samples == num_serialized_samples
num_truncated_samples == 0
num_dropped_oversize_samples == 0
num_class_fallback_samples == 0
```

### 6.2 Pack unit 与 nominal capacity

每个完整的 text-image 样本是一个不可拆分 segment：

\[
L_i=L_{prefix}+L_{payload,i}+1_{BOI}+256+1_{EOI}+1_{EOS}.
\]

正常 pack 使用 `P=2048` token 的 nominal capacity，并将多个完整 segment 放到一个
physical row。`P=2048` 是吞吐参数，不是单样本长度上限：

- 若 `L_i<=P`，样本由 deterministic best-fit-decreasing 放入容量为 `P` 的 pack；
- 若 `L_i>P`，该样本进入 dedicated overflow row，其长度为
  `ceil_to_multiple(L_i,128)`；
- overflow row 保留完整 token，只改变该 microbatch 的物理长度；
- 若 profiling 证明另一个 `P` 在不改变语义的情况下吞吐更高，可只依据训练前吞吐基准
  从 `{1024,1536,2048,3072,4096}` 选择一次，并把选择写入 manifest；禁止依据 FID 选择。

为了让四个 cell 具有相同图像曝光，global sampler 先产生固定的 `256` 个 `img_id` 组成
一个 optimizer step，再在每个 rank 内根据该 cell 的完整 serialized lengths 打 pack。
packing 只能改变物理布局，不能改变该 step 的 image membership。初始化不同但 text
condition 相同的两个 cell 必须复用逐 step 完全相同的 pack manifest。

建议算法：

1. 按显式 dataloader seed 产生 epoch/step 的 image-id 顺序；
2. 每个 rank 收到相同数量的 image samples；
3. 在当前 step 内按 `(length descending, img_id ascending)` 排序；
4. best-fit-decreasing 将 segment 放入剩余空间最小且仍容纳它的 row；
5. pack row 之间按最小 `img_id` 排序，保证 deterministic replay；
6. normal rows pad 到 `P`，overflow rows按 128-token bucket 分组；
7. loss 按 image sample/target token 归一化，而不是按 pack row 数平均。

### 6.3 Block-diagonal selfless attention

仅把序列拼接在一起是不正确的：不同图像样本绝不能互相 attention。每个 token 都携带
`segment_id`，padding 使用 `-1`。packed selfless mask 定义为：

\[
M_b(q,k)=\mathbf 1[valid_q]\mathbf 1[valid_k]
\mathbf 1[segment_q=segment_k]
\mathbf 1[\sigma_k<\sigma_q].
\]

因此：

- 不同 segment 之间没有任何 Q/K edge；
- 每个 segment 内继续使用原 strict `sigma[k] < sigma[q]`；
- text、BOI、image、EOI、EOS 的 sigma 只在各自 segment 内定义并从零开始；
- `position_ids`、row/column coordinate cursor 和 image-local position 都在 segment
  边界重置；
- 每个 image span 的 flow target、visible clean-latent context 与 K/V cache 只引用同一
  segment；
- 改变同一 pack 中其他 segment 的内容或顺序，不得改变目标 segment 的 logits、hidden、
  flow loss 或 gradients（允许预注册数值 tolerance）。

FlexAttention 应使用 segment-aware block mask 跳过跨 segment block；不能先做 dense
`P×P` attention 再把输出清零。MLP 仍处理物理 token，因此还要记录 pack utilization。

### 6.4 Packed metadata

每个 batch 至少携带：

```text
input_ids, token_types, sigma, position_ids,
segment_ids, image_latents,
image_span_table[row, segment, start, end, img_id],
valid_token_count, padding_token_count, image_count,
pack_capacity, pack_manifest_sha256
```

训练日志报告：有效 tokens/s、image samples/s、padding ratio、segments/row、overflow
frequency、peak memory 和 optimizer-step wall time。不能用“packed length=2048”代替真实
有效 token 数。

## 7. 正式 `2×2` 实验矩阵

| ID | Condition payload | Backbone initialization | 主要作用 |
| --- | --- | --- | --- |
| `CI00` | class name | Qwen pretrained | packed class reference |
| `CI10` | full caption | Qwen pretrained | caption effect with language prior |
| `CI01` | class name | fully scratch | initialization effect under low-entropy text |
| `CI11` | full caption | fully scratch | caption effect without language prior |

四个 cell 都必须重新训练。历史 architecture winner checkpoint 不能直接充当 `CI00`，因为
新的 serialization、packing、ImageNet-1K membership 或 data order 只要有一项不同，就不是
同协议对照。

正式 seeds 预注册为 `{43,44,45}`。四个 cell 使用相同 seed set、相同每-step image-id
membership、相同 augmentation decisions 和相同 evaluator random streams。

## 8. Initialization contract

### 8.1 Qwen-pretrained (`I=0`)

- 从固定 digest 的 Qwen3-0.6B-Base 加载 vocabulary embedding、transformer layers 和
  final norm；
- image token embedder、special image tokens、flow condition projection 和 flow head
  按 architecture manifest 的统一初始化规则重建；
- 不加载任何历史 image-flow adapter 或 flow-head weights；
- 记录成功加载、缺失、unexpected 和重新初始化的 parameter-name manifest。

### 8.2 Fully scratch (`I=1`)

- tokenizer、vocabulary size、special-token ids 和 backbone config 与 `I=0` 相同；
- vocabulary embedding、所有 transformer layers 和 final norm 全部随机初始化；
- 不能保留 Qwen 的任意 learned backbone tensor，包括 embedding/norm；
- positional frequency/buffer 等由 config 确定的非 learned state 可以共享；
- 非 backbone image/flow modules 与 pretrained cell 使用匹配的初始化 seed 和逐模块 hash。

### 8.3 两类公平性问题必须分开回答

主矩阵使用完全相同的 optimizer、module-wise LR、warmup、scheduler、weight decay、image
budget 和 checkpoint schedule。这给出 **protocol-matched initialization effect**。

当前为 pretrained backbone 设计的较小 backbone LR 可能使 scratch under-train。因此主矩阵
不能被表述为 scratch 的 best-achievable 上限。主矩阵结束后，如 scratch loss curve 仍显著
未收敛，可追加一个单独的 `scratch-calibrated` study：

- 在 class/caption 两个 scratch cell 上使用相同、预注册的 LR/warmup grid；
- grid 和 tuning budget 对两个 text conditions 对称；
- tuning 与主 `2×2` 结果分开报告；
- 只有 calibrated study 才能讨论两种训练方式的 best-achievable 差异。

## 9. 固定训练预算与计算报告

主矩阵沿用当前架构实验的 global image batch `256` 和 `35,920` optimizer steps，即每个
cell 精确处理：

\[
256\times 35{,}920=9{,}195{,}520
\]

次 image presentation。packing 不得改变这个数。

class 与 caption 的有效 text-token 数不同，因此相同 image budget 不是相同 FLOPs。主结论
回答“每张训练图像配更丰富文本的 system-level effect”，同时必须报告：

- 累计有效 text/image/padding tokens；
- 估算 attention/MLP FLOPs；
- GPU-hours、images/s、valid tokens/s；
- 每个 checkpoint 已见 image 数，而不只报告 optimizer step；
- FID/图文一致性相对 image count、token count 和 wall time 的 learning curve。

如需要 compute-matched 结论，应另建 secondary analysis：在相同累计有效 tokens 或 GPU-hours
处比较曲线。不能用 compute-matched checkpoint 替换主矩阵的固定-image checkpoint。

## 10. 评测与模型差异分析

### 10.1 两套 prompt suite

所有四个模型都在两套固定 prompt 上评测：

1. **Class suite**：同一 fixed template + canonical class name，保持 ImageNet class
   generation 的可比性；
2. **Caption suite**：同一 fixed template + validation `recaption_short`，使用训练前冻结的
   stratified 10K image-id/prompt manifest。

两套 suite 均使用相同 sampler、CFG、NFE、temperature、seed 和 image count。class/caption
training cell 都必须跑两套 suite，不能各自只在熟悉的 prompt distribution 上评测。

### 10.2 Primary 与 guardrail metrics

- Class suite：FID（primary）、IS、ImageNet classifier accuracy；
- Caption suite：FID、冻结图文 encoder 的 pairwise alignment/retrieval metric；
- 两套 suite：precision/recall 或等价的 fidelity/diversity 分解；
- prompt-length buckets：按完整 caption token count 报告 p0--50、p50--90、p90--99、p99+；
- 所有 metrics 先验证 finite，再写入 immutable result artifact。

caption alignment metric 不能单独决定 winner，因为 caption 由 Qwen3-VL family 生成，可能
偏向同家族语言风格。至少同时报告 class accuracy、FID 和 prompt perturbation diagnostics。

### 10.3 Condition-utilization diagnostics

为了区分“生成质量变好”与“模型真正读取长文本”，固定分析：

- **Prompt shuffle**：在同一 class 内和跨 class 随机错配 caption，测生成/velocity 输出变化；
- **Conditional delta**：记录 conditional 与 unconditional velocity prediction 的 norm/cosine；
- **Caption corruption**：删除属性短语或交换关键名词，测图文 metric 的对应变化；
- **Length sensitivity**：完整长 caption 上的收益是否只来自前几十个 token；
- **Stage buckets**：caption 对 early/middle/late reveal-stage flow loss 的影响；
- **CFG response**：只在主 fixed CFG ranking 后，对四个 cell 使用同一预注册 grid。

### 10.4 Pretrained 与 scratch 表征差异

每个 checkpoint 额外记录：

- train/validation flow loss、FID 与 alignment 随 image presentations 的曲线；
- backbone、image projector 和 flow head 的 gradient norm/update norm；
- pretrained cell 相对初始 Qwen 权重的 per-layer relative drift；
- 固定 held-out class/caption prompts 的 text hidden CKA 或 centered-cosine similarity；
- text-only held-out loss作为语言能力遗忘诊断（不参与生成模型选择）；
- prompt shuffle 前后的 hidden/velocity 差异。

这些分析用于解释 Qwen prior 如何被使用，不可替代生成指标。

## 11. 统计协议与决策规则

- 四个 cell 使用 matched seeds `{43,44,45}`；
- 逐 seed 报告全部 factorial contrasts、mean、sample SD、95% CI 和胜场；
- evaluator prompt/image/noise manifests 固定并共享；
- 不以单次 seed winner 选择 cell；
- 不因某个 scratch run 训练不稳而静默增加其 steps 或更换 LR；任何校准属于独立 study；
- 如果 caption 提高 alignment 但显著恶化 class FID/accuracy，结论应为 trade-off，而不是
  “caption universally better”；
- 如果 pretrained 优势只存在于 caption condition，支持语言先验 interaction；
- 如果 pretrained 和 scratch 在长训练后收敛到同一质量，但 pretrained 更快，只声称
  sample efficiency；
- 如果 CI 无法区分，报告 tie，并优先保留训练/推理更简单的 data recipe。

## 12. 建议配置接口

以下仅描述未来实现接口，本轮不修改代码：

```yaml
experiment:
  architecture_manifest: output/.../final_architecture_manifest.json

model:
  model_path: public/models/Qwen/Qwen3-0.6B-Base

training:
  from_scratch: false                 # factorial factor
  seed: 43
  dataloader_shuffle_seed: 43
  total_batch_size: 256               # image samples, not pack rows
  max_train_steps: 35920

dataset:
  params:
    conditioning_mode: prompt_image
    condition_payload: class          # class | recaption_short
    fixed_t2i_prefix: "Generate an image matching this description:"
    caption_sequence_modes: [t2i]
    caption_missing_policy: error
    caption_max_tokens: null           # null means forbidden to truncate
    max_seq_length: null               # no sample-level truncation cap
    pad_to_length: null
    packing:
      enabled: true
      algorithm: deterministic_best_fit_decreasing
      nominal_capacity: 2048
      overflow_policy: dedicated_round_up_128
      reset_positions_per_segment: true
      block_diagonal_attention: true
      loss_normalization: per_image_token
```

`null` 的语义必须是“完整保留”，不能在 loader 中通过 `or default_value` 重新变成 192 或
320。

## 13. Repo 实施落点（未来工作，不在本轮实现）

### `utils/dataset_imagenet_flow_cache.py`

- 删除 caption 和 `_fit_text_ids` 的静默截断行为，改为完整 serialization；
- 构建 frozen caption join/token-length manifest；
- 返回 per-sample token hash、img id 和完整 image-span metadata；
- 保留 T2I-only 固定模板路径，避免随机 prefix/I2T 混杂。

### 建议新增 `utils/multimodal_segment_packing.py`

- deterministic best-fit-decreasing pack planner；
- normal/overflow bucket builder；
- segment ids、position reset 和 pack provenance；
- 按固定 image membership 构造 optimizer-step pack；
- 不包含模型参数或数据过滤逻辑。

### `utils/utils.py` 与 backbone mask builder

- `get_selfless_mask` 增加 `segment_ids`/valid-token 条件；
- FlexAttention block mask 真正跳过跨 segment blocks；
- unpacked batch 使用单 segment id，数值路径保持兼容。

### `models/modeling_model/modeling_selfless_flow.py`

- 支持一个 physical row 中的多个 image spans；
- text/image coordinates 按 segment 重置；
- flow condition、target、context 和 K/V cache 严格按 image span 隔离；
- loss 以 image target token 为单位聚合。

### `pretrain/train_selfless_flow.py`

- 以 global image count 定义 optimizer step；
- 支持一个 step 内 normal/overflow buckets 的梯度累积；
- 记录 pack utilization、manifest digest 和真实 token/image throughput；
- pretrained/scratch parameter-load manifest 与 module hashes 写入 artifact。

## 14. 必须通过的测试

### No truncation

- 构造超过 192、320、2048 tokens 的 captions，token ids 全部保留；
- normal 与 overflow path 均满足 source token hash == unpacked token hash；
- 任一 caption 缺失、空、join 冲突或超过 model context 时 fail fast；
- `num_truncated/dropped/fallback == 0` 才允许生成 training manifest。

### Packing isolation/equivalence

- eval mode 下，同一样本 packed/unpacked 的 hidden、velocity、loss 在 tolerance 内一致；
- 改变相邻 segment 的 token、latent、sigma 或顺序不影响目标 segment；
- attention mask 不存在跨 segment edge；
- position ids 和 row/column coordinates 在每个 segment 起点重置；
- 多 image spans 的 flow context 不交叉；
- packed/unpacked 的 loss 与 gradients在相同 reduction 下等价；
- BFD plan 对相同 seed/manifest bitwise deterministic；
- DDP 各 rank 的 image count、global reduction 权重和 optimizer-step membership 正确。

### Factorial integrity

- class/caption paired runs 的每-step image-id 与 augmentation hashes 相同；
- pretrained/scratch paired runs 的 architecture digest 相同；
- scratch parameter-load report 不含任意 learned Qwen tensor；
- pretrained report 只对预注册的 image/flow/special-token modules重新初始化；
- 四个 cell 的 evaluator prompt/noise manifests完全相同。

## 15. 结果边界与后续决策

本实验能支持的最强结论是：在冻结的 Selfless-Flow 架构和固定 ImageNet membership 下，
完整 image captions 与 Qwen language pretraining 如何共同影响生成质量、条件利用率和
sample efficiency。

本实验不能单独支持：

- CSFM source-distribution 方法本身有效；
- 任意 caption 数据都优于 class labels；
- protocol-matched scratch 结果代表 scratch 的最终上限；
- Qwen-family recaption 上的收益能无条件泛化到人工 caption；
- packing 改善模型质量。packing 只应被视为语义等价的数据布局优化。

如果 `CI10` 最优且 interaction 稳定为正，后续默认使用 caption + Qwen initialization。
如果 caption 只改善 alignment、却损害 class suite，则保留 class/caption mixed curriculum
作为下一份独立 proposal，不能事后把 mixture 加入本矩阵。如果 scratch 在 calibrated
study 追平 pretrained，则把结论限制为 pretraining 加速收敛。
