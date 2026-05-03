# Unified Multimodal Model via Permutation-Based Selfless Attention

## Project Identity

This is a research project investigating whether **permutation-based training with a two-stream selfless attention mechanism** can serve as a superior unified paradigm for multimodal (text + image) pretraining, understanding, and generation—compared to pure autoregressive (AR) approaches such as Emu3 and Chameleon, and modality-switching approaches such as Show-o.

The core thesis: **AR enforces a single, fixed generation order (left-to-right raster scan) that is unnatural for 2D visual data. A permutation-based approach—training on diverse partial orderings and generating via iterative parallel block-wise decoding—can achieve better image generation quality, stronger cross-modal consistency, and comparable understanding, without architectural separation by modality.**

The key differentiation from prior work: **Unlike Show-o which switches between two attention patterns (causal for text, full bidirectional for image), our model uses a SINGLE attention mechanism (σ_kv > σ_q) parametrized by continuous scalar sigma values. This subsumes AR and bidirectional as special cases at two ends of a continuous spectrum.**

**Precise scope of the "unified" claim:** The attention mechanism is modality-agnostic—one σ_kv > σ_q pattern applies to all tokens. Modality-specific components (dual embedding matrices, dual LM heads) exist because text and image have different vocabulary spaces. The contribution is that the core Transformer layers are fully shared and the attention pattern is unified.

> **Full research plan, literature review, experimental design, and competitive analysis:** see `docs/RESEARCH.md`

---

## Core Technical Concept: Selfless Attention

### What It Is

Selfless Attention is a **two-stream permutation-based masked language modeling architecture** built on Qwen3:

1. Every token is assigned a scalar **sigma (σ) value** ∈ ℝ
2. Attention mask: **σ_kv > σ_q** (token i attends to token j only if σ_j > σ_i)
3. The diagonal is explicitly **excluded** — no token can attend to itself
4. Two streams in every Transformer layer:
   - **X0 (Content) stream**: real token embeddings → produces K and V for both streams
   - **XT (Query) stream**: `[MASK]` token embeddings → produces Q only

| | XLNet | Selfless Attention (Ours) |
|---|---|---|
| Content stream mask | σ_kv ≥ σ_q (includes diagonal) | σ_kv > σ_q (**excludes diagonal**) |
| Query stream mask | σ_kv > σ_q | σ_kv > σ_q |
| Can a token "see itself"? | Yes (content stream) | **No (neither stream)** |

### Sigma as a Universal Ordering Coordinate

```
Descending by position (σ=[L,...,1])  →  Strict AR (left→right causal)
Uniform random [0,1]                  →  Random-order / Partial-order
Prompt σ∈[2,3], Target σ∈[0,1]       →  Conditional generation
Text=AR sigma, Image=random sigma     →  Mixed mode, same forward pass
2D-aware image sigma                  →  Coarse→fine, center→edge
```

### Two-Stream Necessity

XT stream produces **no K/V** — only Q. All K/V come from X0 (real token embeddings). If XT were the only stream, K/V would be projected from [MASK] embeddings ("blank" information). X0's self-attention (X0_Q → X0_K/V) is not wasted — it contextualizes X0 hidden states, which serve as K/V for XT at the next layer. Without X0's self-attention, K/V would be based on isolated token embeddings.

**Training**: loss is computed from **XT stream** output (`last_hidden_state=XT_hidden_states`). XT stream is only active during training.
**Inference**: XT is absent (`XT = None` when `not self.training`). Only X0 flows — identical to single-stream inference.

### Attention Mask

```python
# utils/utils.py line 626
def get_selfless_mask(v_sample, seq_len, device):
    def diffusion_mask(b, h, q_idx, kv_idx):
        return v_sample[b, kv_idx] > v_sample[b, q_idx]  # strict — no diagonal
    return create_block_mask(diffusion_mask, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len)

# utils/utils.py line 651
def get_selfless_ar_mask(seq_len, B=None, device="cuda"):
    def causal_mask(b, h, q_idx, kv_idx):
        return kv_idx < q_idx  # strict — no diagonal
    return create_block_mask(causal_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
```

### Training Procedure

```python
# pretrain/train_selfless.py
for batch in dataloader:
    text_ids = batch["input_ids"]

    # 1. Sample sigma values per token
    v_sample = sample_sigma(text_ids)  # "random" (uniform) or "ar" (descending)

    # 2. Build selfless mask (σ_kv > σ_q)
    attention_mask = get_selfless_mask(v_sample=v_sample, seq_len=L)

    # 3. Forward: X0 (real tokens) → K/V, XT ([MASK]) → Q
    loss = model(X0_input_ids=text_ids, labels=text_ids,
                 attention_mask=attention_mask).loss  # CE on ALL positions

    # 4. Modality-aware loss weighting
    total_loss = text_loss + λ_image * image_loss  # λ ≈ 0.3–0.5

    total_loss.backward()
```

### Generation Procedure

```python
# modeling_selfless.py line 688
def generate(prompt_ids, gen_length, block_size=4, decode_strategy='confidence'):
    sigma = torch.zeros(max_seq_len)
    sigma[:prompt_len] = prompt_len + 1 - arange(0, prompt_len)  # AR sigma for prompt
    prefix = prompt_ids

    while prefix.shape[-1] < prompt_len + gen_length:
        seq = prefix + [MASK] * block_size  # pad with MASK tokens
        while seq has MASK:
            logits = model(seq, attention_mask=get_selfless_mask(sigma))
            fill_mask = topk(confidence[valid_positions], k=parallel_rate)
            seq[fill_mask] = argmax(logits[fill_mask])
            sigma[fill_mask] = 0.1 + 0.8 * (1.0 - step / gen_length)
            step += 1
        prefix = seq

    return seq
```

---

## Codebase Architecture

```
unified-mm/
├── CLAUDE.md                              # This document
├── docs/
│   └── RESEARCH.md                        # Full research plan, experiments, lit review
├── pyproject.toml                         # Dependencies: torch, flash-attn, transformers
├── configs/
│   └── selfless/
│       └── pretraining.yaml               # Training hyperparameters
├── models/
│   ├── logging.py                         # HuggingFace-style logging
│   └── modeling_model/
│       └── modeling_selfless.py           # Main architecture (~1300 lines)
│           ├── Qwen3RMSNorm               #   Flash-Attn Triton RMSNorm
│           ├── Qwen3MLP                   #   SwiGLU MLP (Liger kernel optional)
│           ├── Qwen3RotaryEmbedding       #   RoPE
│           ├── Qwen3Attention             #   ★ Two-stream selfless attention
│           ├── Qwen3DecoderLayer          #   ★ Two-stream decoder layer
│           ├── Qwen3Model                 #   ★ Two-stream Transformer body
│           ├── Qwen3ForCausalLM           #   ★ Full model + generate() + speculative_generate()
│           ├── Qwen3ForSequenceClassification  # Stub
│           ├── Qwen3ForTokenClassification     # Stub
│           └── Qwen3ForQuestionAnswering       # Stub
├── pretrain/
│   └── train_selfless.py                  # Training script (416 lines)
├── generation/
│   └── selfless_gen.py                    # Standalone generation script
├── script/
│   └── selfless/
│       └── pretraining.sh                 # Launch script (8 GPU, DeepSpeed Zero-2)
└── utils/
    ├── utils.py                           # Model loading, attention masks, config
    ├── diffusion_utils.py                 # Sigma sampling, DiffusionLanguage class
    ├── wsd_schedule.py                    # WSD LR scheduler
    └── chat_template/
        └── r1_zero.prompt                 # Jinja2 chat template
```

### Implementation: Where the Code Lives

| Component | File | Key Lines |
|---|---|---|
| **Selfless Attention module** | `models/modeling_model/modeling_selfless.py` | L214-317 (`Qwen3Attention.forward`) |
| **Two-stream Decoder Layer** | `models/modeling_model/modeling_selfless.py` | L320-373 (`Qwen3DecoderLayer`) |
| **Two-stream Transformer** | `models/modeling_model/modeling_selfless.py` | L432-527 (`Qwen3Model`) |
| **Full model with LM head** | `models/modeling_model/modeling_selfless.py` | L530-613 (`Qwen3ForCausalLM`) |
| **Selfless attention mask** | `utils/utils.py` | L626-648 (`get_selfless_mask`) |
| **Selfless AR attention mask** | `utils/utils.py` | L651-660 (`get_selfless_ar_mask`) |
| **XLNet attention mask** | `utils/utils.py` | L663-720 (`get_xlnet_mask`) |
| **Sigma sampling (training)** | `utils/diffusion_utils.py` | L60-105 (`DiffusionLanguage.sample_v`) |
| **Prompt sigma processing** | `utils/diffusion_utils.py` | L107-154 (`DiffusionLanguage.prompt_process`) |
| **Training loop** | `pretrain/train_selfless.py` | L36-416 |
| **Validation (AR + random loss)** | `pretrain/train_selfless.py` | L341-409 (`validate`) |
| **Generation (block-wise)** | `models/modeling_model/modeling_selfless.py` | L688-802 (`generate`) |
| **Speculative generation** | `models/modeling_model/modeling_selfless.py` | L805-960 (`speculative_generate`) |
| **Training config** | `configs/selfless/pretraining.yaml` | — |
| **Model loading dispatch** | `utils/utils.py` | L63-175 (`load_model_tokenizer`) |

### Dispatch Mechanism

`utils/utils.py:load_model_tokenizer()` dispatches to model classes by project name. Currently only `modeling_selfless.py` is implemented; placeholders exist for: `sdar`, `llada`, `dream`, `mad`, `dam`, `pnts`, `xlnet`, `ar`, `causal`. The `xlnet` reference is for the XLNet-style baseline (σ_kv ≥ σ_q for content stream).

### Key Configuration

```yaml
# configs/selfless/pretraining.yaml
model:
    model_path: "public/models/Qwen/Qwen3-1.7B-Base"
    attention_pattern: "random"     # "random" or "ar"
    use_flex_attention: true

training:
    batch_size: 16                  # Per GPU
    total_batch_size: 512           # Across all GPUs
    max_train_steps: 80000
    max_seq_length: 2048
    mixed_precision: "bf16"
    from_scratch: true              # Random init from Qwen3 config
```

### Dependencies

```
torch==2.8.0, flash-attn==2.8.3, transformers>=5.7.0, torchvision>=0.23.0
```

Environment: Python 3.12, uv for package management, `.venv/` for virtual environment.

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Selfless Attention (two-stream) | ✅ Done | `modeling_selfless.py` |
| AR mode (descending sigma) | ✅ Done | `diffusion_utils.py` |
| Random mode (uniform sigma) | ✅ Done | `diffusion_utils.py` |
| Block-wise generation | ✅ Done | `modeling_selfless.py` |
| Speculative decoding | ✅ Deferred | `modeling_selfless.py` |
| Validation (AR + random loss) | ✅ Done | `train_selfless.py` |
| Text-only training pipeline | ✅ Done | `pretrain/train_selfless.py` |
| XLNet baseline mask | ✅ Defined | `utils/utils.py` |
| Qwen3-1.7B-Base model | ✅ Downloaded | `public/models/Qwen/Qwen3-1.7B-Base` |
| XQ-GAN VP2-16384 tokenizer | ✅ Downloaded | `public/models/xqgan_vp2_16384/` |
| Image tokenizer integration | ❌ Not started | — |
| Multimodal data pipeline | ❌ Not started | — |
| Dual LM head (text + image) | ❌ Not started | — |
| Modality-aware loss weighting | ❌ Not started | — |
| 2D sigma schedules | ❌ Not started | — |
| 2D RoPE for image tokens | ❌ Not started | — |
| Gradient conflict measurement | ❌ Not started | — |
| Show-o style baseline | ❌ Not started | — |
| Multimodal evaluation suite | ❌ Not started | — |
| Wall-clock decoding benchmark | ❌ Not started | — |

### Implementation Priorities

1. **Image tokenizer wrapper** — wrap XQ-GAN encoder/decoder, download pretrained weights from HuggingFace
2. **Show-o baseline** — causal mask for text + full bidirectional for image, same tokenizer/data/model size
3. **Dual embedding + dual LM head** — separate embedding matrices and output heads for text and image tokens
4. **Multimodal sequence processing** — data loading, special token registration (`<|boi|>`, `<|eoi|>`), token type tracking
5. **Sigma schedule for multimodal** — three-mode task-dependent sigma (text_to_image / image_to_text / text_only)
6. **Modality-aware loss** — weighted sum with configurable λ_image
7. **2D RoPE** — row+column positional encoding for image tokens (ablation)
8. **Gradient analysis tooling** — per-layer gradient cosine similarity
9. **Wall-clock decoding benchmark** — systematic timing of AR vs. block-wise
10. **Tokenizer quality baseline** — report XQ-GAN rFID/rLPIPS/PSNR as upper bound
11. **Multimodal evaluation** — VQA, captioning, image generation metrics (Phase 2+)

---

## Key Implementation Decisions

### Codebook Size

**Choice:** XQ-GAN VP2 with 16,384 codebook (ICLR 2025, rFID 0.64).
- Identical codebook size to VQGAN f16 but 7.8× better reconstruction
- LM head: 16K × hidden_dim ≈ 25M params (negligible addition)
- Can use sub-codebook decomposition: 16K = 128 × 128 (two 128-class heads)

### Understanding Path

**Option A (Pure Discrete):** Image → XQ-GAN → discrete tokens → joint sequence with text. Everything through shared Transformer + LM head. Requires model to learn semantics from scratch.
**Option B (Hybrid):** Understanding uses continuous SigLIP features (projected via MLP), generation uses discrete XQ-GAN tokens. Manzano-style. Two input paths, shared backbone.

**Recommendation:** Start with Option A. Pivot to B if Phase 1 understanding doesn't emerge.

### Loss Weighting

Per-modality loss weights: `total_loss = text_loss + λ_image × image_loss`, λ_image ≈ 0.3–0.5.
Following Emu3's 0.5× finding. Implementation in `multimodal_loss()`.

### Data Mixture

70% pure text + 30% multimodal during pretraining (VILA CVPR 2024 finding).
Text-only data is essential for preserving language capability.

### Sequence Organization

```
1. [BOS] text [BOI] image [EOI] text ...           (interleaved)
2. [BOS] text_caption [BOI] image [EOI]             (text→image)
3. [BOS] [BOI] image [EOI] text_caption             (image→text)
4. [BOS] text_only                                   (pure text, 70%)
```

### Sigma Schedule for Multimodal Training

Three-mode task-dependent sigma assignment to ensure bidirectional cross-modal attention:

```python
def assign_sigma_multimodal(sequence, token_types, task_mode=None):
    # task_mode randomly selected per sample:
    #   "text_to_image" (30%): text σ∈[2,2+n] (condition), image σ∈[0,1] (target)
    #   "image_to_text" (30%): image σ∈[2,3] (condition), text σ∈[0,n] (target)
    #   "text_only"      (40%): standard AR descending
    # Special tokens (BOS, BOI, EOI) always get max sigma, visible to all.
```

This ensures: (a) image can attend to text during generation, (b) text can attend to image during understanding, (c) within-modality attention is always active, (d) text-only data preserves language capability.

---

## References

- **Emu3** (BAAI, Nature 2026): [arXiv:2409.18869](https://arxiv.org/abs/2409.18869)
- **Chameleon** (Meta, 2024): [arXiv:2405.09818](https://arxiv.org/abs/2405.09818)
- **Show-o** (ICLR 2025): [arXiv:2408.12528](https://arxiv.org/abs/2408.12528)
- **Uni-X** (ICLR 2026): [arXiv:2509.24365](https://arxiv.org/abs/2509.24365)
- **MAR** (NeurIPS 2024): [arXiv:2406.11838](https://arxiv.org/abs/2406.11838)
- **MaskGIT** (CVPR 2022): [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)
- **VILA** (CVPR 2024): [arXiv:2312.07533](https://arxiv.org/abs/2312.07533)
- **XQ-GAN / ImageFolder** (ICLR 2025): [arXiv:2412.01762](https://arxiv.org/abs/2412.01762), [github.com/lxa9867/ImageFolder](https://github.com/lxa9867/ImageFolder)
- **XLNet** (NeurIPS 2019): [arXiv:1906.08237](https://arxiv.org/abs/1906.08237)
