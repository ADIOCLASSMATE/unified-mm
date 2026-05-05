# Unified Multimodal Model via Permutation-Based Selfless Attention — Research Plan

> **Companion to `CLAUDE.md`** — this document contains the full research plan, literature review, experimental design, competitive analysis, and risk assessment. The implementation-focused `CLAUDE.md` covers code structure, key files, and configuration.

---

## Project Identity

This is a research project investigating whether **permutation-based training with a two-stream selfless attention mechanism** can serve as a superior unified paradigm for multimodal (text + image) pretraining, understanding, and generation—compared to pure autoregressive (AR) approaches such as Emu3 and Chameleon, and modality-switching approaches such as Show-o.

The core thesis: **AR enforces a single, fixed generation order (left-to-right raster scan) that is unnatural for 2D visual data and constrains cross-modal interaction. A permutation-based approach that trains on diverse partial orderings—and generates via iterative parallel block-wise decoding—can achieve better image generation quality, stronger cross-modal consistency, and comparable understanding, without architectural separation by modality.**

The key differentiation from prior work: **Unlike Show-o which switches between two different attention patterns (causal for text, full bidirectional for image), our model uses a SINGLE attention mechanism (σ_kv > σ_q) parametrized by continuous scalar sigma values. This subsumes AR and bidirectional as special cases at two ends of a continuous spectrum, enabling modality-appropriate generation orders without modality-specific architectural components.**

**Precise scope of the "unified" claim:** The attention mechanism itself is modality-agnostic—a single σ_kv > σ_q pattern applies to all tokens regardless of modality. The modality-appropriate behavior (AR for text, partial-order for images) emerges from the sigma values, not from switching attention patterns. However, we acknowledge that other components are necessarily modality-aware: dual text/image embedding matrices and output heads exist because text and image have different vocabulary spaces (an unavoidable fact of discrete multimodal modeling). The modality-specificity is redistributed from the attention mechanism (Show-o's approach) to the sigma schedule—a design choice that keeps the core Transformer layers fully shared and the attention pattern unified.

---

## Advisor Feedback Integration (Round 1 + Round 2)

The following concerns were raised across two rounds of advisor review and are explicitly addressed throughout this document:

| # | Advisor Concern | Round | Status | Where Addressed |
|---|----------------|-------|--------|-----------------|
| 1 | Image generation resembles MaskGIT | R1 | Addressed | § Competitive Differentiation |
| 2 | Show-o is the most direct competitor | R1 | Addressed | § Show-o vs. Our Model |
| 3 | 2D-aware sigma schedule is needed | R1 | Redesigned | § 2D-Aware Sigma Schedules |
| 4 | Block-wise decoding may not beat AR (KV-cache) | R1/R2 | Corrected | § KV-Cache Analysis |
| 5 | Training-inference sigma mismatch | R1 | Addressed | § Training-Inference Sigma Mismatch |
| 6 | "O(2^L) training signals" is misleading | R1 | Fixed | § H3 |
| 7 | "Emergence without CLIP" already proven by Emu3 | R1 | Accepted | § Research Questions |
| 8 | Speculative decoding not needed now | R1 | Accepted | § Implementation Roadmap |
| 9 | 2D position encoding consideration | R1 | Accepted | § 2D Position Encoding |
| 10 | Selfless vs XLNet alone not enough | R1 | Accepted | § Contribution Thesis |
| 11 | Frequency-based sigma invalid in VQ token space | R2 | Accepted — removed | § 2D-Aware Sigma Schedules |
| 12 | Two-stream necessity not justified | R2 | Addressed | § Two-Stream Necessity |
| 13 | "No modality-specific components" overclaimed | R2 | Scoped precisely | § Project Identity |
| 14 | Training FLOPs not accounted | R2 | Added | § Training FLOPs Analysis |
| 15 | Tokenizer quality baseline missing | R2 | Added | § Tokenizer Quality Baseline |
| 16 | VQGAN rFID~5.0 too poor | R2 | Upgraded | § The Tokenizer Landscape |
| 17 | Phase 1 metrics (VQA/FID) have no signal | R2 | Redesigned | § Experimental Plan Phase 1 |
| 18 | Single-step partial order vs multi-step implicit — difference? | R2 | Analyzed | § Training vs. Inference Distinction (below) |

### Training vs. Inference Distinction: Where the Real Difference Lies

The advisor raised a fundamental question: MaskGIT's multi-step iterative decoding also creates an implicit partial order across steps (tokens decoded earlier are "more visible"). What is the advantage of encoding the partial order within a SINGLE forward pass (our approach)?

**Training:** This is where the difference is substantial and measurable.
- **MaskGIT training:** A single forward pass uses uniform random binary masking. All unmasked tokens are mutually visible (full bidirectional). The training signal for each token is: "predict yourself given this set of equally-visible context tokens."
- **Our training:** A single forward pass encodes hierarchical visibility through continuous sigma. The training signal for each token is: "predict yourself given these context tokens, where some are more 'confirmed' (higher sigma) than others." This exposes the model to **graded** context quality — tokens learn to weigh evidence from high-sigma (reliable) sources more heavily than low-sigma (uncertain) sources.

**Inference:** The advisor is correct that the **inference-time behavior** of the two approaches is superficially similar. Both fill tokens iteratively based on confidence. Our key advantage at inference is the ability to inject prior knowledge through the sigma schedule (e.g., center-out generation for images), while MaskGIT's generation order is purely confidence-emergent.

**The core contribution is training-side:** continuous sigma ordering provides richer, more structured supervision per training sample than binary masking. Whether this translates to better downstream performance is the empirical question our experiments must answer.

---

## Core Technical Concept: Selfless Attention

### What It Is

Selfless Attention is a **two-stream permutation-based masked language modeling architecture** built on top of Qwen3. It replaces standard causal attention with a **sigma-value-based partial ordering mechanism** where:

1. Every token in a sequence is assigned a scalar **sigma (σ) value** ∈ ℝ
2. The attention mask follows a strict partial order: **σ_kv > σ_q** (a query token at position i can attend to a key/value token at position j only if σ_j > σ_i)
3. The diagonal is explicitly **excluded** — no token can attend to itself (σ_i > σ_i is always false)
4. Two streams flow through every Transformer layer:
   - **X0 (Content) stream**: real token embeddings → produces K and V for both streams
   - **XT (Query) stream**: `[MASK]` token embeddings → produces Q only

### The Key Distinction from XLNet

XLNet also uses two-stream attention, but with a critical difference:

| | XLNet | Selfless Attention (Ours) |
|---|---|---|
| Content stream mask | σ_kv ≥ σ_q (includes diagonal) | σ_kv > σ_q (**excludes diagonal**) |
| Query stream mask | σ_kv > σ_q | σ_kv > σ_q |
| Can a token "see itself"? | Yes (content stream via =) | **No (neither stream)** |
| Shortcut learning risk | Token can peek at own embedding | Token must infer itself from context |

The exclusion of the diagonal is the defining constraint: a token at position i must be predicted entirely from the context of *other* tokens. This forces the model to learn genuine contextual representations rather than relying on identity shortcuts. When extended to multimodal sequences, this means an image token cannot "cheat" by reading its own embedding—it must use surrounding text tokens and other image tokens to infer its value, compelling cross-modal reasoning.

**Note on contribution scope:** The selfless vs. XLNet diagonal difference is a component of the architecture, not a standalone contribution. Its value is demonstrated in the context of the full sigma-based multimodal framework, particularly through ablation studies comparing selfless (σ_kv > σ_q) against XLNet-style (σ_kv ≥ σ_q for content stream) in multimodal settings.

### Sigma as a Universal Ordering Coordinate

The sigma value is the mechanism that unifies different generation paradigms:

```
┌─────────────────────────────────────────────────────────────┐
│  Sigma Schedule           │  Resulting Behavior             │
├───────────────────────────┼────────────────────────────────┤
│  Descending by position   │  Strict Autoregressive (AR)     │
│  σ = [L, L-1, ..., 1]    │  Left→right causal              │
├───────────────────────────┼────────────────────────────────┤
│  Uniform random [0,1]     │  Random-order / Partial-order   │
│                           │  Directed: σ higher → visible   │
│                           │  as K/V to lower-σ queries       │
├───────────────────────────┼────────────────────────────────┤
│  Prompt: σ ∈ [2, 3]       │  Conditional generation         │
│  Target: σ ∈ [0, 1]       │  Prompt always visible to all   │
│                           │  Target tokens fill iteratively │
├───────────────────────────┼────────────────────────────────┤
│  Text segments: AR sigma  │  Mixed mode: text AR + image    │
│  Image segments: random σ │  random-order, same forward pass│
├───────────────────────────┼────────────────────────────────┤
│  2D-aware image sigma     │  Structured generation:         │
│  (center-based)           │  center→edge, etc.              │
└─────────────────────────────────────────────────────────────┘
```

**Critical distinction from binary masking (MaskGIT, Show-o):** Binary masking treats all unmasked tokens as equally visible (full bidirectional attention). Our sigma-based ordering creates a **directed partial order** where visibility is hierarchical: a token with σ=0.8 can be seen by tokens with σ<0.8, but NOT by tokens with σ>0.8. This asymmetry creates a richer dependency structure than flat bidirectional attention and enables coarse-to-fine generation schedules.

### Architecture Diagram

```
Input Sequence: [BOS] text_tokens [BOI] image_tokens [EOI] text_tokens ...

┌─────────────────────────────────────────────────────────────┐
│                    Token Embedding                           │
│  Text tokens → Qwen3 text embedding                         │
│  Image tokens → XQ-GAN codebook embedding (shared or separate)│
│  Special tokens → [BOS], [BOI], [EOI], [MASK] embeddings    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Sigma Assignment                             │
│  High σ (≥ 2.0): condition/prompt tokens (always visible)   │
│  Low σ ([0,1]): target/generation tokens (iterative fill)   │
│  Descending σ: AR behavior for text segments                │
│  Random σ: partial-order behavior for image segments        │
│  2D-structured σ: coarse-to-fine for image generation       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Selfless Attention Transformer (N layers)           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  X0 Stream (Content)          XT Stream (Query)       │   │
│  │  ┌──────────────────┐        ┌──────────────────┐    │   │
│  │  │ Real token emb   │        │ [MASK] token emb │    │   │
│  │  │   ↓              │        │   ↓              │    │   │
│  │  │ QKV projection    │        │ Q projection only  │    │   │
│  │  │   ↓              │        │   ↓              │    │   │
│  │  │ K, V ────────────┼────────│→ attended by Q   │    │   │
│  │  │ Q (self-attend)   │        │                   │    │   │
│  │  └──────────────────┘        └──────────────────┘    │   │
│  │                                                       │   │
│  │  Attention Mask: σ_kv > σ_q (strict, no diagonal)    │   │
│  │  Both streams use the SAME mask                       │   │
│  │                                                       │   │
│  │  Residual + MLP (shared weights for both streams)     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Key properties:                                             │
│  - All layers shared between modalities and streams          │
│  - No modality-specific parameters                           │
│  - The attention mask IS the only difference between         │
│    AR mode and random/bidirectional mode                     │
│  - Single attention pattern, continuous sigma parametrization│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LM Heads                                  │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │ Text head:        │  │ Image head:                       │ │
│  │ Linear(d, V_text) │  │ Linear(d, V_image)  (or sub-codebook heads) │
│  │ V_text ≈ 150K     │  │ V_image ≈ 16K                     │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
│                                                              │
│  Head selection: determined by position (text region vs      │
│  image region), marked by [BOI]/[EOI] special tokens         │
└─────────────────────────────────────────────────────────────┘
```

### Two-Stream Necessity: Why X0 and XT Are Both Required

A natural question: if X0 and XT both use the same attention mask (σ_kv > σ_q, excluding diagonal), and only XT's output contributes to the training loss (line 522 of `Qwen3Model.forward`: `last_hidden_state=XT_hidden_states`), why do we need the X0 stream at all? Could we simplify to a single-stream architecture?

**The answer is no, and the reason is fundamental to how the selfless constraint interacts with masked prediction.**

**Trace the data flow from the code:**

```
Qwen3Attention.forward() (lines 242-317):

  XT_Q = Q_proj(XT_hidden)  ← projects from [MASK] embedding → no token identity
  X0_Q = Q_proj(X0_hidden)  ← projects from real token embedding → carries token identity
  X0_K = K_proj(X0_hidden)  ← from real token embedding
  X0_V = V_proj(X0_hidden)  ← from real token embedding

  XT attention: XT_Q (unbiased)  → attends to X0_K, X0_V  (content-aware)
  X0 attention: X0_Q (informed)  → attends to X0_K, X0_V  (contextualizes K/V)

  Both use the SAME mask: σ_kv > σ_q (strict, no diagonal)
```

**What would happen with only XT stream (no X0)?**

If XT were the only stream, it would need to produce K and V from its own hidden states:
```
XT_K = K_proj(XT_hidden)  ← from [MASK] embedding!
XT_V = V_proj(XT_hidden)  ← from [MASK] embedding!
```

**All K/V at all positions would be projected from the [MASK] token embedding.** The model would see nothing but "blank" information as keys and values. This is fundamentally different from BERT/MaskGIT, where masked positions receive attention but do NOT produce K/V—only unmasked positions (with real token content) provide K/V.

**Why X0's Q matters (even though it doesn't directly contribute to loss):**

The X0 stream's self-attention (X0_Q → X0_K, X0_V) is NOT wasteful computation. Its output goes through `o_proj` and is added to the X0 residual stream (line 361):

```python
X0_hidden_states = X0_residual + X0_hidden_states  # after attention
```

This residual update means X0's hidden states evolve at each layer, becoming progressively more contextualized. These contextualized X0 hidden states then serve as K/V for the next layer's XT attention. **Without X0's self-attention, the K/V at each layer would be based on isolated token embeddings without cross-token context.**

**Analogy to XLNet:** XLNet needs two streams because:
- Content stream: σ_kv ≥ σ_q (includes diagonal) → can see itself, provides K/V
- Query stream: σ_kv > σ_q (excludes diagonal) → cannot see itself, provides Q

XLNet's two streams have DIFFERENT masks. Our two streams have the SAME mask (both exclude diagonal). The distinction is instead in the INPUT: X0 processes real tokens (informed Q, content-aware K/V), XT processes [MASK] tokens (unbiased Q, no K/V). Both are necessary for the same reason: **the model needs content-aware K/V (from X0) and unbiased Q (from XT) simultaneously.** You cannot collapse this into one stream without either (a) leaking token identity through Q, or (b) providing K/V from meaningless [MASK] embeddings.

---

## Competitive Differentiation: Why This Is Not MaskGIT, Not Show-o

### Show-o vs. Our Model (Most Direct Competitor)

Show-o (Xie et al., ICLR 2025) claims "One Single Transformer to Unify Multimodal Understanding and Generation." Superficially similar to our work, but fundamentally different in architecture:

| | Show-o (ICLR 2025) | Our Model |
|---|---|---|
| **Attention mechanism** | TWO patterns: causal mask for text, full bidirectional mask for image | ONE pattern: σ_kv > σ_q for ALL tokens |
| **Modality routing** | Hard switch based on token type | Soft, continuous — sigma value IS the routing |
| **Image masking** | Binary (token is masked or unmasked) | Continuous sigma values → hierarchical visibility |
| **Training objectives** | Two: AR CE (text) + mask-predict CE (image) | One: CE on all positions (unified loss) |
| **Cross-modal attention** | Hard boundary at modality switch | No boundary — sigma values mediate flow |
| **Coarse-to-fine generation** | Not supported (binary mask, equal visibility) | Supported (sigma layers encode generation priority) |
| **Image tokenizer** | MAGVIT-v2 (LFQ, 262K codebook) | XQ-GAN (VQ, 16K codebook) |
| **Parameters** | 1.3B | 0.6B (initial) |

**Concrete code-level difference:**

```python
# Show-o's approach (conceptual):
if token_is_text:
    mask = causal_mask()           # standard AR attention
elif token_is_image:
    mask = full_bidirectional_mask()  # all unmasked tokens mutually visible

# Our approach (actual code in utils/utils.py:get_selfless_mask):
mask = sigma_kv_greater_than_sigma_q(v_sample)  # single pattern
# text with descending sigma → AR behavior emerges
# image with random sigma → partial-order behavior emerges
# NO modality check, NO attention pattern switch
```

**Why this difference matters:**

1. **Show-o cannot do partial visibility.** An image token in Show-o is either fully masked (sees nothing) or fully unmasked (sees all other unmasked tokens). There is no intermediate state. Our model can assign σ=0.3 to a token, making it visible to tokens with σ<0.3 but invisible to tokens with σ>0.3. This enables hierarchical, coarse-to-fine generation.

2. **Show-o's modality boundary is hardcoded.** The attention pattern switch at modality boundaries creates a discontinuity in information flow. Our sigma values create smooth transitions — a text token with σ=1.5 and an image token with σ=1.4 can interact almost symmetrically.

3. **Show-o trains with two objectives.** AR loss for text, mask-predict loss for image. Our model uses ONE cross-entropy loss on ALL positions, simplifying the optimization landscape.

### MaskGIT vs. Our Model

MaskGIT (Chang et al., CVPR 2022) pioneered confidence-based parallel decoding for image generation. Key differences:

| | MaskGIT | Our Model |
|---|---|---|
| **Attention** | Full bidirectional (all unmasked tokens mutually visible) | Directed partial order (σ_kv > σ_q) |
| **Visibility** | Equal — all unmasked tokens see each other | Hierarchical — higher σ tokens dominate as K/V |
| **Generation order** | Confidence-driven (emerges from model) | Confidence-driven AND sigma-scheduled (can be steered) |
| **Modality** | Image only | Text + image, unified |
| **Architecture** | Standard bidirectional Transformer | Two-stream selfless attention |

**Why this difference matters:** In full bidirectional attention, the generation order is purely determined by per-step confidence. There is no mechanism to encode "this region should be generated before that region." Our sigma-based approach allows **top-down scheduling** — we can inject prior knowledge about generation order through sigma schedules (e.g., center-first, coarse-to-fine) while still allowing the model's confidence to guide fine-grained decisions.

### MAR vs. Our Model

MAR (Li et al., NeurIPS 2024 Spotlight) uses random-order training for image generation. Key differences:

| | MAR | Our Model |
|---|---|---|
| **Token type** | Continuous VAE latents (no quantization) | Discrete VQ tokens |
| **Loss** | Diffusion loss (DDPM, per-token MLP denoiser) | Cross-entropy (standard LM loss) |
| **Architecture** | Encoder-decoder with cross-attention | Decoder-only two-stream |
| **Modality** | Image only | Text + image, unified |
| **Training paradigm** | Random masking + bidirectional encoder | Random sigma + directed attention |

### Emu3 / Chameleon vs. Our Model

| | Emu3 / Chameleon | Our Model |
|---|---|---|
| **Training paradigm** | Pure AR (fixed left→right) | Permutation-based (diverse orderings) |
| **Image generation** | Token-by-token raster scan | Block-wise parallel, any order |
| **Generation flexibility** | One order only | AR, random, 2D-structured — all in one model |
| **Cross-modal ordering** | Text always before image (causal) | Text and image can be interleaved with flexible sigma |

---

## KV-Cache Analysis for Block-Wise Decoding

A key concern raised: does block-wise parallel decoding actually beat AR in wall-clock time, considering that AR models CAN use KV-cache? This section provides a corrected, rigorous analysis.

**Correction from prior version:** Standard AR image generation fully supports KV-cache. Each step computes O(1) new KV projections and O(k) attention (for query attending to k cached keys). The claim that "AR has no KV-cache for images" was incorrect.

### Corrected Analysis

**Setup:** Sequence length L containing prompt tokens (P) + image tokens (I=256). We compare only the image token generation portion (text generation is identical in both methods).

**AR with KV-cache (256 steps for 256 image tokens):**

```
Step 1: Query at position P+0 attends to P cached keys → O(P) attention
Step 2: Query at position P+1 attends to P+1 cached keys → O(P+1) attention
...
Step 256: Query at position P+255 attends to P+255 cached keys → O(P+255) attention

Total attention (dot products per head):
  Σ_{i=1}^{256} (P + i) = 256*P + 256*257/2 ≈ 256P + 32,896

New KV computation per step: O(1) → 256 × O(1)
```

**Our block-wise with KV-cache (~26 steps for 256 image tokens, k=10):**

```
Step 1: All P+256 positions attend to all P filled positions → (P+256)×P attention
        Fill k positions → update KV cache for these k
Step 2: All P+256 positions attend to all P+k filled positions → (P+256)×(P+k)
...
Step 26: All P+256 positions attend to all P+256 filled positions → (P+256)²

Total attention (dot products per head):
  Σ_{s=1}^{26} (P+256) × (P + s×k) ≈ 26 × (P+256) × (P + 128)

FLOPs comparison for image token generation only:
```

| Sequence Length L | AR Total Attention | Block-wise Total (k=10) | Ratio |
|-------------------|-------------------|------------------------|-------|
| L=512 (P=256, I=256) | ~98K per head | ~5.1M per head | AR wins (52× less) |
| L=2048 (P=1792, I=256) | ~491K per head | ~20.4M per head | AR wins (42× less) |

**This is a genuine disadvantage for block-wise decoding.** The full-attention per step dominates. However, several mitigating factors must be considered:

1. **Wall-clock measurement, not FLOP counting, is what matters.** Modern hardware (GPUs) is highly optimized for dense matrix multiplications (the block-wise workload) compared to sequential incremental decoding (the AR workload). The FLOP ratio may not translate linearly to wall-clock ratio.

2. **Parallel sampling across batch.** Block-wise decoding can process multiple samples simultaneously with batching. AR decoding with varying sequence lengths per sample is harder to batch efficiently.

3. **The comparison above is for image-only generation.** In interleaved text-image generation, the block-wise method fills multiple image tokens while the text continues AR. The overall efficiency depends on the interleaving pattern.

4. **For very long sequences (L >> 256), block-wise may be competitive** because AR's O(L) per step × 256 steps = O(256L) grows faster than block-wise's fixed-step structure.

**Honest assessment:** Block-wise image decoding may NOT be faster than AR with KV-cache in the typical regime (L ~ 512-2048). We should:
- Run actual wall-clock benchmarks on identical hardware, measuring tokens/second for both methods at various sequence lengths
- If block-wise is slower, remove the efficiency claim and focus the contribution on generation quality and flexibility
- If block-wise is faster (due to hardware batching effects), report the measured speedup, not the theoretical FLOP analysis
- Explore potential optimizations: partial KV-cache reuse across block-wise steps (sigma-updated positions only need recomputation of attention to newly filled tokens)

### Text Generation: AR Mode Preserves Efficiency

For text generation, our model can operate in pure AR sigma mode (descending sigma values). In this mode:
- The attention pattern is effectively causal (higher-sigma tokens attend to lower-sigma ones, which for descending sigma = left→right)
- Standard KV-cache incremental decoding applies
- The overhead: appending a [MASK] token, decoding it, then appending a new [MASK] for the next step. Each step recomputes KV for the just-decoded token (O(1) overhead).
- **Text generation efficiency is within ~2% of standard AR models.**

---

## Training-Inference Sigma Mismatch

### The Problem

During training, image tokens are assigned random sigma values (uniform [0,1]), meaning:
- Some image tokens have high sigma (visible to many other tokens)
- Some image tokens have low sigma (visible to few other tokens)
- The model learns to predict image tokens from varying levels of partial context

During inference for understanding tasks (VQA, captioning):
- ALL image tokens need to be fully visible (high sigma ≥ 2.0) so the model can "see" the entire image
- This is a specific sigma configuration that the model was trained on (since random uniform covers [0,1], values near 1.0 approximate "mostly visible"), but is not the predominant training regime

During inference for generation tasks:
- Image tokens start as [MASK] with low sigma (target), progressively filled
- The sigma schedule during generation may differ from the training distribution

### Mitigation Strategies

**Strategy 1: Explicit sigma distribution coverage in training.**
Ensure the training sigma distribution includes:
- A fraction of samples where ALL image tokens get sigma ∈ [1.5, 2.0] (simulating understanding inference)
- A fraction of samples where image tokens get descending sigma by 2D position (simulating structured generation)
- This is a lightweight data augmentation, not a training pipeline change

**Strategy 2: 2D-aware sigma schedules for generation (see next section).**
By designing generation sigma schedules that mirror training distributions, we reduce the mismatch. If training includes frequency-based or center-based sigma assignments, and generation uses the same schedules, the inference distribution matches training.

**Strategy 3: Adaptive sigma during generation.**
Rather than using a fixed sigma schedule, use the model's own confidence as a signal to update sigma values. This is what our current code does (`sigma = 0.1 + 0.8 * (1.0 - step / total_steps)`), which naturally aligns with the confidence-based filling order. High-confidence tokens (likely "easier" coarse structures) get filled first and receive higher sigma, matching the coarse-to-fine intuition.

---

## 2D-Aware Sigma Schedules for Image Generation

This is identified as a **core novelty opportunity** — prior work (MaskGIT, Show-o, Emu3) does not exploit 2D spatial structure in generation ordering. Binary masking (MaskGIT/Show-o) treats all spatial positions identically. AR (Emu3/Chameleon) uses arbitrary raster scan.

**Important constraint:** VQ tokens (including XQ-GAN's) exist in a learned discrete latent space, NOT in a frequency-decomposed space. Frequency-domain operations (DCT/DWT) are undefined in the VQ token grid — they require continuous pixel or latent values, not discrete codebook indices. Sigma schedules must operate on spatial position information alone.

### Design Space

Given a VQ-encoded H×W image token grid, we design sigma schedules that encode spatial priors:

#### Scheme A: Center-Out (Foveated)

**Intuition:** The center of the image is typically the focal point. Generate the center first, expand outward. This is a simple, spatially-grounded heuristic that works for a broad class of natural images.

**Implementation:**
```python
def center_out_sigma(H=16, W=16):
    """
    Center-out: center patches get high sigma (generate first),
    edge patches get low sigma (generate later).
    """
    y, x = torch.meshgrid(
        torch.arange(H).float(),
        torch.arange(W).float(),
        indexing='ij'
    )
    center_y, center_x = (H-1)/2, (W-1)/2
    dist_from_center = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
    max_dist = dist_from_center.max()
    sigma = 1.0 - dist_from_center / max_dist  # center=1.0, corners=0.0
    sigma += torch.randn_like(sigma) * 0.05  # small noise for stochasticity
    return sigma.clamp(0.0, 1.0).flatten()  # [H*W]
```

#### Scheme B: Saliency-Weighted

**Intuition:** Use a lightweight, off-the-shelf saliency detector (trained on human fixation data) to identify visually important regions, and prioritize their generation.

**Implementation:**
```python
def saliency_weighted_sigma(image_pixels, H=16, W=16):
    """
    Use saliency map to assign generation priority.
    High-saliency regions → high sigma (generate first).
    """
    from torchvision.transforms import functional as F
    saliency_map = lightweight_saliency_detector(image_pixels)  # [H_img, W_img]
    # Downsample to token grid
    saliency_grid = F.resize(saliency_map.unsqueeze(0), (H, W)).squeeze()
    sigma = (saliency_grid - saliency_grid.min()) / (saliency_grid.max() - saliency_grid.min() + 1e-8)
    return sigma.flatten()
```

#### Scheme C: Learnable Sigma Predictor (Strongest Novelty)

**Intuition:** Rather than hand-designing sigma schedules, train a lightweight network to predict the optimal generation order for each image, conditioned on image content or text prompt. This is the most promising direction for genuine novelty.

**Implementation sketch:**
```python
class SigmaPredictor(nn.Module):
    """
    Lightweight network that predicts per-position sigma values
    from image-level features or text embeddings.
    """
    def __init__(self, hidden_dim=256, num_positions=256):
        super().__init__()
        self.position_query = nn.Embedding(num_positions, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, image_feature, text_feature=None):
        # image_feature: pooled embedding from a frozen vision encoder
        # text_feature: optional pooled text embedding
        pos_queries = self.position_query.weight  # [256, d]
        condition = image_feature
        if text_feature is not None:
            condition = condition + text_feature

        pos_features, _ = self.cross_attn(
            pos_queries.unsqueeze(1),
            condition.unsqueeze(0).unsqueeze(0),
            condition.unsqueeze(0).unsqueeze(0)
        )
        sigma = self.sigma_head(pos_features.squeeze(1))  # [256]
        return sigma
```

**Training considerations:**
- The sigma predictor can be trained jointly with the main model
- A spatial smoothness regularization (e.g., total variation on the sigma grid) prevents degenerate sigma assignments
- Text-aligned sigma should reflect semantic importance (text-mentioned objects get higher sigma)
- The predictor is lightweight (<5M params) relative to the main model

### Experimental Plan for 2D Sigma Schedules

| Experiment | Sigma Schedule | Expected Advantage | Status |
|---|---|---|
| Baseline | Random uniform | — | Current default |
| Ablation A | Center-out | Better focal object coherence | Simple to implement |
| Ablation B | Saliency-weighted | Priority to visually salient regions | Moderate implementation effort |
| Ablation C | Learnable predictor (image-only) | Image-adaptive optimal ordering | Higher effort, core novelty |
| Ablation D | Learnable predictor (image + text) | Text-conditioned semantic ordering | Highest novelty, most complex |

**Evaluation metrics (for Phase 2+ at scale):**
- FID / rFID (image generation quality)
- CLIP score / PickScore (text-image alignment)
- Spatial consistency metrics (autocorrelation of generated token patterns)
- User study: which generation order produces most "natural" images?

---

## 2D Position Encoding for Image Tokens

### Motivation

Standard 1D RoPE encodes position along the raster-scan order of the 1D sequence. For 16×16 image token grids flattened to 256 tokens, this means:
- Adjacent tokens in the 1D sequence are 2D neighbors only within the same row
- Cross-row adjacency (e.g., position (i,j) and (i+1,j)) spans 16 positions apart in the 1D sequence
- When using random sigma ordering, the 1D position index provides no meaningful 2D spatial information

### Proposed Solution: 2D RoPE

For image tokens specifically, apply separate RoPE encoding for row and column positions, then combine:

```python
def apply_2d_rope(image_hidden_states, grid_H=16, grid_W=16):
    """
    Apply 2D Rotary Position Embedding to image tokens.
    Each token at grid position (h, w) gets:
      - Row encoding: RoPE with position h
      - Column encoding: RoPE with position w
    """
    B, N, D = image_hidden_states.shape  # N = H*W = 256
    half_D = D // 2

    # Reshape to 2D grid
    x = image_hidden_states.view(B, grid_H, grid_W, D)

    # Row RoPE (first half of dimensions)
    row_pos = torch.arange(grid_H, device=x.device).float()
    row_encoding = rope(x[..., :half_D], row_pos.unsqueeze(-1).unsqueeze(0))

    # Column RoPE (second half of dimensions)
    col_pos = torch.arange(grid_W, device=x.device).float()
    col_encoding = rope(x[..., half_D:], col_pos.unsqueeze(0).unsqueeze(0))

    # Concatenate
    encoded = torch.cat([row_encoding, col_encoding], dim=-1)
    return encoded.view(B, N, D)
```

**Integration:** 2D RoPE is applied only to image tokens (positions between [BOI] and [EOI]), while text tokens continue to use standard 1D RoPE. This is a minimal change — no additional parameters, only a positional encoding variant.

**Experimental plan:** Compare 1D RoPE vs. 2D RoPE for image tokens as an ablation study, measuring:
- rFID (image reconstruction/generation quality)
- Spatial consistency (are generated images spatially coherent?)
- VQA accuracy (does better spatial encoding improve understanding?)

---

## Research Context and Related Work

### Directly Related: Unified Multimodal Models via Next-Token Prediction

#### Emu3 (BAAI, Nature 2026)
- Pure decoder-only Transformer, next-token prediction from scratch
- SBER-MoVQGAN tokenizer (32,768 codebook, reconstruction-only, zero CLIP)
- Matches LLaVA-1.6 on 12 VLM benchmarks **without CLIP**
- Surpasses SDXL on image generation
- Key finding: cross-modal understanding **emerges** from pure NTP without semantic supervision
- **Limitation:** AR only—image generation is raster-scan, token-by-token. Each image token can only see previous tokens.
- **Our positioning:** We accept Emu3's emergence finding as a background assumption (cite, not re-prove). Our contribution is replacing AR with permutation-based training and comparing the two paradigms.

#### Chameleon (Meta FAIR, 2024)
- Early-fusion token-based model, VQ-VAE tokenizer (8,192 codebook)
- 34B parameters, ~10T mixed-modal tokens
- SOTA on captioning; competitive with GPT-4V on mixed-modal generation
- **Limitation:** Same AR raster-scan constraint. Visual token inefficiency (1,024 tokens per 512² image).

#### Show-o (Show Lab, ICLR 2025) — MOST DIRECT COMPETITOR
- "One Single Transformer to Unify Multimodal Understanding and Generation"
- Uses TWO attention patterns: causal (AR) for text, full bidirectional for image
- Discrete diffusion (D3PM simplified to MaskGIT-like mask-predict) for image generation
- MAGVIT-v2 tokenizer (LFQ)
- 1.3B parameters, 3-stage training pipeline
- **Key limitation:** Modality-dependent attention switching — hard boundary between text and image processing
- **Our differentiation:** Single continuous attention pattern parametrized by sigma values, no modality switching, subsumes AR and bidirectional as special cases

#### VILA (NVIDIA/MIT, CVPR 2024)
- Interleaved image-text pretraining is essential for in-context learning
- Unfreezing LLM during pretraining enables multimodal reasoning
- Re-blending text-only data is crucial to prevent text capability degradation
- **Key practical lesson:** 70% text-only + 30% multimodal mixture preserves text performance

#### MAR (Li et al., MIT, NeurIPS 2024 Spotlight)
- Random-order autoregressive training for image generation
- Uses continuous VAE latents (NOT discrete tokens) + diffusion loss per token
- Encoder-decoder architecture with cross-attention
- Image-only (no text, no multimodal)
- **Key contribution:** Random-order training + diffusion loss for continuous tokens
- **Difference from our work:** We use discrete tokens + CE loss + unified text-image training

#### MaskGIT (Chang et al., Google, CVPR 2022)
- Masked token prediction + confidence-based parallel decoding for image generation
- Full bidirectional attention (all unmasked tokens mutually visible)
- Image-only, binary masking
- **Key difference from our work:** Full bidirectional vs. directed partial order; binary mask vs. continuous sigma; image-only vs. multimodal

### The Gradient Conflict Problem

#### Uni-X (ICLR 2026)
- **Formally quantifies gradient conflict** in fully-shared multimodal Transformers
- Metric: c_g = -(cos(g_text, g_img) − baseline)
- Conflict is **depth-dependent**: severe in shallow (layer 0-6) and deep (layer 17-24) layers; naturally mitigated in middle layers (7-16)
- **Root cause:** visual token sequences have far higher N-gram conditional entropy than natural language
- **Solution:** "Two-end separated, middle shared" X-shaped architecture (modality-specific shallow/deep layers)
- **Our positioning:** Sigma-based role separation (high σ = condition/KV, low σ = target/Q) may naturally mitigate gradient conflict WITHOUT architectural changes. We test this with a simple gradient cosine similarity experiment, not a full Uni-X replication.

#### Symbiotic-MoE (April 2026)
- Generation training causes **catastrophic forgetting** of understanding due to gradient conflicts
- Standard MoE leads to routing collapse (generative gradients dominate)
- **Solution:** Modality-aware expert disentanglement + early-stage gradient shielding

### The Tokenizer Landscape

#### Why Not MAGVIT2 (LFQ, 262K codebook)?
- **Designed for generation-only**, no semantic supervision in training
- 262K vocabulary → LM head explosion (262K × hidden_dim dominates parameter count)
- Requires sub-codebook decomposition (asymmetric token factorization) which adds complexity
- Community consensus (2024-2025): pure reconstruction-trained tokenizers have a **representational capacity bottleneck** for understanding tasks
- Show-o uses MAGVIT2 but shows poor understanding performance on fine-grained tasks (text reading, counting)

#### Primary Recommendation: XQ-GAN ImageFolder VP2 (rFID 0.64, ICLR 2025)

**This is the strongest available open-source discrete tokenizer that meets all our constraints: no CLIP distillation, manageable codebook size, pretrained weights available, and competitive reconstruction quality.**

**Code & weights:** `github.com/lxa9867/ImageFolder` → HuggingFace links in the GitHub Model Zoo table.

| Property | VQGAN f16 16384 | XQ-GAN VP2 16384 |
|----------|----------------|------------------|
| **rFID** (ImageNet 256²) | 4.98 | **0.64** (7.8× better) |
| **Codebook size** | 16,384 | 16,384 (identical) |
| **Tokens per 256² image** | 256 (16×16) | 256 (16×16) |
| **LM head params** | 16K × d_model ≈ 25M | 16K × d_model ≈ 25M (identical) |
| **Pretrained weights** | CompVis download | HuggingFace download |
| **Venue** | CVPR 2021 | ICLR 2025 (Adobe Research + MIT) |
| **License** | MIT | MIT |

**Why XQ-GAN:**
- **Identical codebook size and token count** to VQGAN — zero engineering changes to LM head design
- **7.8× better reconstruction quality** (rFID 0.64 vs 4.98) — moves from "barely usable" to "competitive with modern tokenizers"
- **ICLR 2025 accepted** — academically citable as recent work
- **Modular framework** — supports VQ, RQ, MSVQ, and VP variants; easy to swap quantization methods during experimentation

#### Comparison with Other Modern Tokenizers

| Tokenizer | rFID | Codebook | Tokens | CLIP-Free? | Pretrained? | Viability for Us |
|-----------|------|----------|-------|------------|-------------|------------------|
| **XQ-GAN VP2** | **0.64** | 16,384 | 256 | ✅ Yes | ✅ HF | **★ Best choice** |
| WeTok | 0.12 | GQ (multi-LFQ) | 256 | ✅ Yes | ✅ GitHub | ⚠️ Very new (Aug 2025), integration complexity TBD |
| UniTok | 0.38 | 8×4,096 MCQ | 256 | ❌ Uses CLIP distillation | ✅ GitHub | ❌ Conflicts with our "no CLIP" route |
| MGVQ | 0.49 | 8×8,192 subgroups | 256 | ✅ Yes | ✅ GitHub | ⚠️ Multi-codebook adds integration complexity |
| Open-MAGVIT2 | 1.17 | 262,144 LFQ | 256 | ✅ Yes | ✅ HF | ❌ Codebook too large (262K) |
| VQGAN f16 | 4.98 | 16,384 | 256 | ✅ Yes | ✅ CompVis | ❌ Quality too poor |

**Decision:** XQ-GAN ImageFolder VP2 is the primary recommendation. If higher reconstruction quality is needed and integration complexity proves acceptable, MGVQ (rFID 0.49) is the fallback. WeTok (rFID 0.12) should be monitored as its ecosystem matures.

#### Alternative: TiTok-L-32
- 32 tokens per image, 4K codebook
- Extreme compression, smallest possible LM head
- HF: `yucornetto/tokenizer_titok_l32_imagenet`
- Tradeoff: significant detail loss, unsuitable for high-quality generation

---

## Tokenizer Quality Baseline (Critical Infrastructure)

**Any image generation evaluation must report the tokenizer's own reconstruction metrics as an upper bound.** If the tokenizer's rFID is 0.64, no model can achieve FID better than this threshold — the tokenizer is the ceiling.

**Required baseline reporting:**
- rFID (reconstruction FID on ImageNet val)
- rLPIPS (reconstruction perceptual similarity)
- PSNR (reconstruction signal-to-noise ratio)
- Visual examples: original images alongside tokenizer-reconstructed images

**This is not optional.** Reviewers will ask: "Is your FID of X due to your model or your tokenizer?" Without this baseline, the question is unanswerable.

---

## Training FLOPs Analysis

### Two-Stream Training Overhead

Per attention layer, the two-stream design adds computational overhead relative to a standard single-stream Transformer. The dominant cost is the MLP (SwiGLU, intermediate_size ≈ 4× hidden_size):

| Operation | Single-Stream | Two-Stream (Ours) | Overhead Factor |
|-----------|--------------|-------------------|-----------------|
| Q projection | d² | 2d² (X0_Q + XT_Q) | 2× |
| K projection | d² | d² (X0_K only) | 1× |
| V projection | d² | d² (X0_V only) | 1× |
| O projection | d² | 2d² (X0_O + XT_O) | 2× |
| Attention compute | ~L²d | ~2L²d | 2× |
| **MLP (SwiGLU)** | **~8d²** | **~16d²** (X0 + XT, shared weights) | **2×** |

Total per-layer FLOPs breakdown for projection + MLP (excluding attention, which depends on L):
- Single-stream: ~12d² per layer
- Two-stream: ~22d² per layer
- **Projection + MLP overhead: ~22d² / 12d² ≈ 1.8×**

Including attention (2×), the **overall training FLOPs overhead is approximately 1.8–2.0×** relative to a single-stream Transformer of the same parameter count. The MLP dominates the overhead because XT's MLP computation is a full duplicate (different inputs, shared weights).

**Potential optimization — XT skip MLP:** If XT stream skips the MLP (only does attention), the overhead drops to ~14d² per layer (~1.2× projection, ~1.5× total with attention). This should be tested as an efficiency ablation: if image token CE loss doesn't degrade significantly, it's a valuable engineering contribution.

**Inference:** XT stream is not used at inference (`XT = None` when `not self.training`). Inference FLOPs are identical to a standard single-stream Transformer. **The training overhead does not carry to inference.**

### FLOPs-Matched Comparison

For fair comparison against AR baselines, we must compare at equal FLOPs budget, not just equal parameter count:

| Comparison | Parameter Count | Relative Training FLOPs |
|------------|----------------|------------------------|
| Our 0.6B selfless vs. AR 0.6B | Equal params | We use ~1.9× more FLOPs |
| Our 0.6B selfless vs. AR 3.2B | AR has more params | Matched training FLOPs ≈ |
| Our 0.6B selfless vs. Show-o 1.3B | Similar params, Show-o has 2× forward patterns | FLOPs roughly comparable |

**Recommendation:** Report both parameter-matched and FLOPs-matched numbers in all comparison tables. If we cannot afford a FLOPs-matched AR baseline, acknowledge training cost as a limitation and report the FLOPs ratio transparently. Note that the inference cost is unaffected (single-stream).

### Generation Inference Cost

At inference, our model has **no two-stream overhead** (XT is absent). The generation cost is determined by:
- Text: AR mode, identical to standard AR inference (with minor [MASK] KV recomputation overhead)
- Image: block-wise parallel decoding, whose per-step cost is analyzed in the KV-Cache Analysis section
- **No training-specific overhead carries to inference.**

---

## Research Questions and Hypotheses

### Primary Hypotheses

**H1: Sigma-Based Unified Attention > Modality-Switching Attention.**

A single continuous attention mechanism (σ_kv > σ_q) that subsumes both AR and bidirectional as special cases outperforms modality-dependent attention switching (Show-o's approach) for unified multimodal tasks. Specifically:
- The absence of a hard modality boundary enables smoother cross-modal information flow
- Continuous sigma allows hierarchical, partial visibility that binary masking cannot express
- A single training objective (CE on all positions) avoids the optimization complexity of multi-objective training

**H2: Gradient Conflict Mitigation via Sigma Ordering.**

Permutation-based sigma ordering naturally mitigates the gradient conflict between text and visual modalities identified by Uni-X, **without requiring modality-specific architectural separation.** By assigning text tokens to high-sigma (condition/KV) roles and image tokens to low-sigma (target/query) roles, the gradient signals are implicitly decomposed—text gradients flow primarily through X0 stream content prediction, image gradients through XT stream masked prediction. **This is a testable hypothesis measured via per-layer gradient cosine similarity.**

**H3: Permutation-Based Training > Pure AR for Multimodal.**

Training on diverse partial orderings produces better multimodal performance than training on a single fixed order (AR left-to-right):
- Image tokens have no natural 1D ordering; forcing raster scan is arbitrary
- Each training epoch exposes the model to L! possible orderings (one per sample drawn from the sigma distribution), compared to exactly 1 fixed ordering for AR
- Cross-modal attention patterns are more diverse and robust
- The model learns general conditional distributions P(x_i | any subset), not just P(x_i | x_<i)

**Note on the "L! possible orderings" claim:** Each training sample sees ONE permutation (one set of sigma values). Across N samples in an epoch, the model sees up to N different permutations. Over many epochs, the coverage of the L! possible orderings is determined by the sigma sampling distribution. This is characterized more precisely as: "AR = 1 ordering; our method = distribution over L! orderings, sampled per batch." This is a significant increase in training signal diversity, but we avoid misleading "O(2^L)" or "all permutations" language.

### Secondary Research Questions

**RQ1: 2D Sigma Schedule Impact.** Does a 2D-aware sigma schedule (center-out) for image generation improve image quality (FID, CLIP score) over random uniform sigma? Does it improve cross-modal consistency (text→image alignment)?

**RQ2: Text Capability Preservation.** What loss-weighting strategy (λ_image) and data mixture ratio (% pure text) is sufficient to prevent text capability degradation? Does permutation-based training degrade text more or less than AR-based multimodal training?

**RQ3: Block-Wise Decoding Efficiency.** Is block-wise confidence-based parallel decoding significantly faster than token-by-token AR for image generation in wall-clock time, accounting for KV-cache differences?

**RQ4: Training-Inference Sigma Alignment.** Does explicit sigma distribution coverage in training (including "all image tokens high sigma" samples) improve understanding task performance compared to random uniform-only sigma training?

**RQ5: 2D Position Encoding.** Does 2D RoPE for image tokens improve spatial coherence and reconstruction quality compared to standard 1D RoPE?

---

## Revised Experimental Plan

### Phase 0: Infrastructure (Current Status → Ready)

- [ ] **Image Tokenizer Integration**
  - XQ-GAN VP2 16384 checkpoint downloaded to `public/models/xqgan_vp2_16384/`
  - Implement `ImageTokenizer` wrapper class with encode/decode
  - Dual embedding: `nn.Embedding(16384, hidden_dim)` for image token embeddings
  - Dual LM head: `nn.Linear(hidden_dim, 16384)` for image token prediction
  - Report tokenizer reconstruction metrics (rFID, rLPIPS, PSNR) as generation upper bound

- [ ] **Multimodal Data Pipeline**
  - Extend dataloader to handle interleaved text-image sequences
  - Sequence format: `[BOS] text_tokens [BOI] img_tokens [EOI] text_tokens [BOI] img_tokens [EOI] ...`
  - Special tokens: `<|boi|>`, `<|eoi|>` (begin/end of image)
  - Token type tracking: per-position mask indicating text vs. image token

- [ ] **Loss Weighting**
  - Implement modality-aware loss: `loss = text_loss + λ * image_loss`
  - λ = 0.3–0.5 (following Emu3's 0.5× finding)

- [ ] **Sigma Schedules for Multimodal**
  - Text tokens: descending sigma (AR) or high sigma (prompt/condition)
  - Image tokens: uniform random sigma (baseline), center-out
  - Prompt/condition tokens: sigma ∈ [2, 3] (always visible)
  - Three-mode task-dependent assignment: text_to_image / image_to_text / text_only

- [ ] **2D RoPE for Image Tokens** (ablation preparation)
  - Implement 2D RoPE variant alongside existing 1D RoPE
  - Apply only to image token positions

- [ ] **Gradient Analysis Tooling**
  - Per-layer gradient cosine similarity measurement between text-only and image-only batches
  - Simple implementation: compute gradients from two separate forward passes, compare cosine similarity per layer

### Phase 1: Proof of Concept (Small Scale, ~1B tokens, ~5K steps)

**Objective:** Validate core hypotheses at small scale before committing to full training.

**Setup:**
- Model: Qwen3-0.6B-Base with Selfless Attention (using pretrained Qwen3-0.6B weights for text capability)
- Image tokenizer: XQ-GAN VP2 16384 (rFID 0.64)
- Data: ~1B multimodal tokens (mixed text + image interleaved)
- 70% pure text + 30% multimodal mixture

**Critical note on scale:** At 1B tokens / 5K steps, generation metrics (FID, VQA, BLEU, captioning) will NOT produce meaningful signals. The model will not have learned to generate coherent images or answer visual questions. Phase 1 focuses exclusively on **structural metrics** that provide signal even at small scale.

**Variants to test:**

| Variant | Text Sigma | Image Sigma | Image Loss Weight | 2D RoPE | Purpose |
|---------|-----------|-------------|-------------------|---------|---------|
| A (AR baseline) | AR (descending) | AR (descending, raster) | 0.5 | No | Upper bound for AR |
| B (Ours, basic) | AR (descending) | Random uniform | 0.5 | No | Core comparison vs. A |
| C (Ours, prompt style) | High (2-3) | Low (0-1) | 0.5 | No | Conditional generation mode |
| D (Ours, center-out sigma) | AR (descending) | Center-out | 0.5 | No | Test 2D sigma schedule |
| E (Ours + 2D RoPE) | AR (descending) | Random uniform | 0.5 | Yes | 2D position ablation |

**Phase 1 Metrics (Structural Only):**

✅ **Metrics that produce meaningful signal at this scale:**
- Image token cross-entropy loss (perplexity on image token prediction — directly comparable across variants)
- Text token perplexity (monitor degradation — must stay within 5% of pure-text baseline)
- Gradient cosine similarity per layer (text gradients vs. image gradients) — structural metric, independent of model capability
- Training loss convergence speed (how quickly does each variant reduce loss?)
- Tokenizer reconstruction upper bound (report rFID/rLPIPS/PSNR of XQ-GAN itself)

❌ **Metrics deferred to Phase 2 (meaningless at 1B token scale):**
- VQA accuracy — model cannot generate coherent answers yet
- FID — generation quality too poor, variance too high
- Captioning BLEU/ROUGE — generation quality insufficient
- CLIP score — requires coherent image generation

**Success criteria:**
- Variant B shows image token CE loss comparable to or better than Variant A (AR baseline)
- Text perplexity within 5% of pure-text baseline
- Gradient cosine similarity higher (less conflict) than AR-only variant A
- At least one 2D sigma variant (D) shows improved image token CE over random uniform (B)

### Phase 2: Scaling Validation (~10-80B tokens, 80K steps)

**Objective:** Demonstrate scaling behavior and solidify comparisons.

**Setup:**
- Model: Qwen3-0.6B-Base, possibly scale to larger variant if results warrant
- Data: ~10-80B multimodal tokens
- Training: full config as in `configs/selfless/pretraining.yaml`

**Comparisons:**
- Against LLaVA-style (CLIP + AR) on understanding benchmarks — **baseline for "how close can we get without CLIP?"**
- Against Chameleon-style (pure AR, same tokenizer) on generation quality — **core paradigm comparison**
- Against Show-o-style (causal text + full-bidir image, binary masking) — **core architectural comparison**
- Ablation: selfless (σ_kv > σ_q) vs. XLNet-style (σ_kv ≥ σ_q for content stream)

### Phase 3: Contribution Validation

**Objective:** Demonstrate the specific advantages claimed.

**Key experiments:**

1. **Single-pattern vs. dual-pattern attention** — compare our sigma-based unified attention against Show-o-style modality-switching attention, both trained with the same data and tokenizer. This is the single most important experiment for differentiation.

2. **2D sigma schedule ablation** — compare center-out, random uniform, and (if ready) learnable sigma predictor for image generation quality.

3. **Gradient conflict measurement** — simple per-layer gradient cosine similarity comparison between our method and AR baseline. Report as supporting evidence for H2, not as a standalone contribution.

4. **Block-wise decoding efficiency benchmark** — wall-clock time measurement on identical hardware: AR image generation (256 steps) vs. block-wise (varying k), accounting for KV-cache. Report tokens-per-second and total latency.

5. **2D position encoding ablation** — 1D RoPE vs. 2D RoPE for image tokens, measuring reconstruction quality and spatial coherence.

6. **Sigma distribution coverage ablation** — compare training with and without explicit "all image tokens high sigma" samples on understanding task performance (testing training-inference mismatch mitigation).

---

## Revised Contribution Thesis

### What This Work Contributes

**1. Continuous Sigma Ordering as a Unified Attention Paradigm.**
We propose that a single continuous attention mechanism (σ_kv > σ_q) parametrized by scalar sigma values can subsume both AR and bidirectional attention as special cases at two ends of a continuous spectrum. This is architecturally simpler and more flexible than modality-dependent attention switching (Show-o) or modality-specific architectural separation (Uni-X).

**2. Empirical Comparison of Attention Paradigms for Unified Multimodal Models.**
We provide the first direct experimental comparison between:
- Pure AR (Emu3/Chameleon paradigm)
- Modality-switching attention (Show-o paradigm, causal + full bidirectional)
- Continuous sigma-based attention (our paradigm)
All using the same tokenizer, same data, and same training budget.

**3. 2D-Aware Sigma Schedules for Non-Sequential Modalities.**
We introduce structured sigma schedules (center-out) that encode spatial priors into the generation ordering for 2D image data. This is a novel capability enabled by continuous sigma parametrization that binary masking cannot express.

**4. Gradient Conflict Analysis Without Architectural Separation.**
We test whether sigma-based role separation (condition vs. target) naturally mitigates the modality gradient conflict identified by Uni-X, without requiring modality-specific layers.

### What This Work Does NOT Claim

- We do NOT claim to be the first to show emergence of cross-modal understanding without CLIP (Emu3 already demonstrated this). We cite Emu3 and position our work as exploring an alternative paradigm.
- We do NOT claim the selfless diagonal removal alone as a major contribution. It is a component of the architecture, validated through ablation.
- We do NOT claim speculative decoding as a contribution (deferred to future work).
- We do NOT claim "O(2^L) training signals." We claim "distribution over L! orderings, sampled per batch — a significant increase in training signal diversity over AR's single fixed ordering."

### Expected Paper Positioning

> "We present a permutation-based two-stream Transformer with selfless attention for unified multimodal pretraining. Unlike prior work that either forces all modalities into a single autoregressive order (Emu3, Chameleon) or switches between separate attention patterns for different modalities (Show-o), our approach uses a single continuous attention mechanism parametrized by scalar sigma values. This subsumes AR and bidirectional as special cases of a spectrum, and enables modality-appropriate generation orders—including novel 2D-aware schedules for images—without modality-specific architectural components. We show that (1) continuous sigma ordering naturally mitigates gradient conflict between modalities without architectural separation, (2) 2D-aware sigma schedules improve image generation coherence over both raster-scan AR and uniform random ordering, and (3) block-wise parallel decoding with sigma-based KV-cache provides measurable efficiency gains for multimodal generation."

---

## Honest Risks and Unknowns

### Risk 1: Understanding May Not Emerge at Our Scale
Emu3 and Chameleon demonstrated emergence at massive scale (10T+ tokens, 7-34B parameters). Our experiments at 0.6B scale with 80B tokens may not cross the emergence threshold. **Mitigation:** We do not claim "emergence" as our contribution. We cite Emu3 as having established the feasibility, and focus on the paradigm comparison (sigma-based vs. AR).

### Risk 2: Permutation-Based Training May Degrade Text Performance
Text has meaningful sequential structure. Training text tokens with random sigma (rather than AR) may confuse the model's understanding of syntax and discourse. **Mitigation:** Text tokens always use descending (AR) sigma in training. Only image tokens use random or 2D-structured sigma. This mixed schedule preserves text sequence structure.

### Risk 3: The Gradient Conflict Hypothesis May Not Hold
Uni-X found conflict in AR models. Our sigma-based "role separation" (condition vs. target) is theoretically different from modality separation, but there is zero empirical evidence that it actually reduces gradient conflict. **Mitigation:** We frame this as a testable hypothesis with a simple experiment (gradient cosine similarity), not as a guaranteed result. A null result is still informative.

### Risk 4: Competitive Landscape — Show-o Has Priority
Show-o (ICLR 2025) already demonstrated "one transformer for understanding and generation." **Mitigation:** We differentiate on architecture (single continuous attention pattern vs. dual switching), not on the high-level concept. The paper must clearly articulate why continuous sigma > binary mask switching.

### Risk 5: Scale Asymmetry
If permutation-based training only beats AR at small-to-medium scale but AR catches up (or surpasses) at very large scale (10T+ tokens), our contribution is limited to a "data efficiency" argument. **Mitigation:** Acknowledge this limitation explicitly. A data-efficiency win at smaller scale is still a valid contribution if properly scoped.

### Risk 6: Wall-Clock Efficiency May Not Materialize
Block-wise decoding might not be faster than AR in wall-clock time if KV-cache recomputation overhead is larger than anticipated. **Mitigation:** Run the benchmark and report honestly. If block-wise is not faster, drop the efficiency claim and focus on quality and flexibility advantages.

### Risk 7: Training-Inference Sigma Mismatch
Image tokens at inference (all high sigma for understanding) differ from the training distribution (random sigma). **Mitigation:** Explicit sigma distribution coverage in training, tested in Phase 1. If mismatch is significant, add "all image tokens high sigma" samples to training.

---

## Advisor Round 3 & 4: Technical Deep-Dive Responses

This section provides detailed technical analysis addressing the most recent rounds of advisor feedback. These are not changes to the research direction — they are clarifications and precise specifications.

---

### 1. Sigma Range Design: Three-Mode Task-Dependent Assignment (Full Specification)

**Background.** The advisor identified a potential issue: if text tokens use descending sigma [L, L-1, ..., 1] and image tokens use uniform random [0, 1] in all training samples, then text tokens (as query, σ ∈ [1, L]) can never attend to image tokens (as K/V, σ ∈ [0, 1]) because σ_kv > σ_q requires image σ > text σ, which is always false.

**Resolution.** This is by design for certain tasks, but requires explicit per-sample mode switching to ensure all inference configurations are covered during training. The three-mode design below addresses this.

#### 1.1 The σ_kv > σ_q Asymmetry

The attention rule is directional: a query at position i can attend to K/V at position j if and only if σ_j > σ_i. This creates an asymmetry:
- If modality A has uniformly higher sigma than modality B, then A is invisible to B but B is visible to A
- Bidirectional visibility requires overlapping sigma ranges

#### 1.2 Three Training Modes

Each training sample is assigned one of three modes. The modes are sampled per-sample within each batch, so the model sees all three configurations in every training iteration.

```python
def assign_sigma_multimodal(sequence, token_types, task_mode=None):
    """
    Full specification of three-mode sigma assignment.

    token_types[b, i] ∈ {0: text, 1: image, 2: special, 3: padding}

    task_mode determines the sigma range assignment:
      "text_to_image" (30%): text=condition, image=target
      "image_to_text" (30%): image=condition, text=target
      "text_only"      (40%): standard AR for text
    """
    B, L = sequence.shape
    sigma = torch.zeros(B, L)

    for b in range(B):
        if task_mode is None:
            task_mode = random.choices(
                ["text_to_image", "image_to_text", "text_only"],
                weights=[0.30, 0.30, 0.40]
            )[0]

        text_mask = token_types[b] == 0
        image_mask = token_types[b] == 1
        special_mask = token_types[b] == 2
        pad_mask = token_types[b] == 3

        n_text = text_mask.sum().item()
        n_image = image_mask.sum().item()

        if task_mode == "text_to_image":
            # ── Text is condition, Image is target ──
            # ── Text σ ∈ [2, 2+n_text], Image σ ∈ [0, 1] ──
            # Visibility check:
            #   Image Q (σ∈[0,1]) → Text K/V (σ∈[2,2+n])  ✓  (text σ > image σ)
            #   Text Q (σ∈[2,2+n]) → Image K/V (σ∈[0,1])  ✗  (image σ < text σ)
            # This is CORRECT for generation: model predicts images given text prompt.
            # Text does NOT need to see image (text is the prompt, not the target).
            if n_text > 0:
                sigma[b, text_mask] = 2.0 + n_text - torch.arange(
                    n_text, dtype=torch.float32, device=sigma.device
                )  # σ ∈ [2, 2+n_text], AR descending within text
            if n_image > 0:
                sigma[b, image_mask] = torch.rand(n_image, device=sigma.device)  # σ ∈ [0, 1]

        elif task_mode == "image_to_text":
            # ── Image is condition, Text is target ──
            # ── Image σ ∈ [2, 3], Text σ ∈ [0, n_text] ──
            # Visibility check:
            #   Text Q (σ∈[0,n_text]) → Image K/V (σ∈[2,3])  ✓  (image σ > text σ)
            #   Image Q (σ∈[2,3]) → Text K/V (σ∈[0,n_text])  ✗  (text σ < image σ)
            # This is CORRECT for understanding: model generates text given image input.
            # Image does NOT need to see text (image is the input, not the target).
            if n_image > 0:
                sigma[b, image_mask] = 2.0 + torch.rand(
                    n_image, device=sigma.device
                )  # σ ∈ [2, 3]
            if n_text > 0:
                sigma[b, text_mask] = n_text - torch.arange(
                    n_text, dtype=torch.float32, device=sigma.device
                )  # σ ∈ [0, n_text], AR descending

        elif task_mode == "text_only":
            # ── Standard AR for text-only documents ──
            if n_text > 0:
                sigma[b, text_mask] = n_text - torch.arange(
                    n_text, dtype=torch.float32, device=sigma.device
                )
            if n_image > 0:
                sigma[b, image_mask] = -1.0  # no image tokens in text-only mode

        # ── Special tokens: always max sigma, visible to ALL ──
        max_sigma = 0.0
        if text_mask.any():
            max_sigma = max(max_sigma, sigma[b, text_mask].max().item())
        if image_mask.any() and task_mode != "text_only":
            max_sigma = max(max_sigma, sigma[b, image_mask].max().item())
        if special_mask.any():
            sigma[b, special_mask] = max_sigma + 1.0

        # ── Padding: excluded from attention ──
        if pad_mask.any():
            sigma[b, pad_mask] = -1.0

    return sigma
```

#### 1.3 Cross-Modal Visibility Verification

| Mode | Text→Image Visibility | Image→Text Visibility | Is This Correct? |
|------|----------------------|----------------------|-----------------|
| `text_to_image` | Image Q sees Text K/V ✓ | Text Q does NOT see Image K/V | ✅ Yes — text is condition, image is target |
| `image_to_text` | Image Q does NOT see Text K/V | Text Q sees Image K/V ✓ | ✅ Yes — image is condition, text is target |
| `text_only` | N/A | N/A | ✅ Yes — no image tokens present |

**Inference configuration coverage:**

| Inference Task | Sigma Assignment | Covered by Training Mode? |
|---------------|-----------------|---------------------------|
| Text→Image generation | Text σ high, Image σ low, iterative fill | ✅ `text_to_image` |
| Image captioning | Image σ high, Text σ low, AR decode | ✅ `image_to_text` |
| VQA | Image σ high, Question+Answer σ low, AR | ✅ `image_to_text` |
| Interleaved generation | Alternating high/low sigma blocks | ⚠️ Partially — covered by mixture of modes across samples |

**For truly interleaved bidirectional attention** (where text and image tokens need to mutually attend within a single forward pass), we would need overlapping sigma ranges. This can be done by assigning all content tokens sigma in a shared range [0, 2] with modality-appropriate internal ordering, but is deferred to post-training / Phase 2. For pretraining, the three-mode design covers all essential inference configurations.

#### 1.4 Sigma Range Overlap for Interleaved (Future)

If bidirectional cross-modal attention within a single sample is needed (e.g., for chain-of-thought reasoning over images):

```python
# "interleaved" mode (deferred to Phase 2):
# All content tokens σ ∈ [0, 2]
# Text: AR descending within text segments, shifted to [1, 2] for prompt, [0, 1] for response
# Image: random within [0, 2]
# → Ranges overlap → bidirectional cross-modal visibility possible
```

---

### 2. Quality-Heterogeneous Context: Correcting the Narrative

**Issue identified by advisor.** The previous CLAUDE.md narrative described high-σ tokens as "more confirmed" or "more reliable" context sources. This is incorrect given the σ_kv > σ_q attention rule. We correct the analysis below.

#### 2.1 Mathematical Reality

The attention rule σ_kv > σ_q implies:

```
For any token at position i with sigma σ_i:
  As QUERY: can attend to K/V at positions j where σ_j > σ_i
            → fraction of visible tokens ≈ 1 - σ_i (if σ ∼ Uniform[0,1])
  
  As K/V:   can be attended by queries at positions k where σ_k < σ_i
            → fraction of queries that can see this K/V ≈ σ_i
```

**Concrete numbers for σ ∼ Uniform[0,1]:**

| σ_i | As Query: % tokens visible | As K/V: % of queries that see it | X0 hidden state quality |
|-----|---------------------------|----------------------------------|------------------------|
| 0.9 | ~10% | ~90% | **Poor** (sees almost nothing) |
| 0.5 | ~50% | ~50% | Medium |
| 0.1 | ~90% | ~10% | **Rich** (sees almost everything) |

#### 2.2 Implication for X0 Stream K/V Quality

The X0 stream produces K/V for both its own self-attention and XT's cross-stream attention. The quality of X0's K/V at position j depends on how much context that position's hidden state has aggregated through self-attention at previous layers:

- **High-σ X0 positions** (σ ≈ 0.9): As queries, they see ~10% of other tokens. Their self-attention output is poorly contextualized — essentially "raw" token embeddings with minimal cross-token information. However, their K/V is **highly visible** to ~90% of other positions.

- **Low-σ X0 positions** (σ ≈ 0.1): As queries, they see ~90% of other tokens. Their self-attention output is richly contextualized — the hidden state aggregates information from almost the entire sequence. However, their K/V is **barely visible** to only ~10% of other positions.

#### 2.3 Implication for XT Stream Prediction

The XT stream queries (from [MASK] embeddings, typically predicting target tokens with low σ) attend to X0's K/V. The information they receive is **quality-heterogeneous**:

```
XT_Q (σ ≈ 0.0) → attends to almost ALL X0 K/V:
  ├── High-σ K/V (σ ≈ 0.8-1.0): "Raw" signals from isolated tokens
  │                              → local, uncontextualized information
  ├── Mid-σ K/V (σ ≈ 0.4-0.7): Partially contextualized signals
  │                              → moderate cross-token aggregation
  └── Low-σ K/V (σ ≈ 0.0-0.3): Well-contextualized signals
                                 → rich cross-token aggregation
```

#### 2.4 Contrast with MaskGIT

In MaskGIT, all unmasked tokens are mutually visible through full bidirectional attention. Every unmasked token's K/V is **homogeneously** well-contextualized — each unmasked position has aggregated information from all other unmasked positions.

| Property | MaskGIT (Binary Mask, Full Bidir) | Selfless Attention (Continuous σ) |
|----------|-----------------------------------|-----------------------------------|
| K/V quality distribution | Homogeneous (all well-contextualized) | Heterogeneous (raw ↔ refined gradient) |
| Information sources for masked prediction | Uniformly high-quality context | Mixture of raw + refined signals |
| Training signal type | "Predict from uniformly good context" | "Predict from heterogeneous, quality-varying context" |

#### 2.5 Is Heterogeneity an Advantage?

This is an empirical question. Possible arguments:

**Potential advantage:**
- The model learns to weigh evidence based on source quality — distinguishing reliable (well-contextualized) from unreliable (raw) signals
- This may produce more robust representations that don't over-rely on the assumption that all context is equally good
- During generation, as tokens are progressively filled and sigma values updated, the context quality naturally improves, creating an implicit curriculum

**Potential disadvantage:**
- The prediction task is harder because some context sources are low-quality
- Training signal may be noisier than MaskGIT's homogeneous context
- The model may learn to ignore high-σ K/V (since they carry little information), wasting representational capacity

**This should be tested empirically.** The comparison of our random-uniform sigma against a binary-mask baseline (Show-o style variant in Phase 1) will directly measure whether heterogeneous context helps or hurts.

---

### 3. Show-o Baseline Specification for Phase 1

The advisor correctly identified that Phase 1 lacked a Show-o-style attention baseline. This section provides the implementation specification.

#### 3.1 What "Show-o Baseline" Means in Our Context

We are NOT replicating Show-o's full training pipeline (3-stage training, MAGVIT2 tokenizer, etc.). We are implementing Show-o's **attention pattern** within our framework, using our tokenizer (XQ-GAN) and our training data.

| Component | Show-o (Original) | Our Show-o Baseline |
|-----------|-------------------|---------------------|
| Tokenizer | MAGVIT2 (LFQ, 262K) | XQ-GAN VP2 (VQ, 16K) — same as our main model |
| Model backbone | Single-stream Transformer | Single-stream Transformer |
| Text attention | Standard causal | Standard causal |
| Image attention | Full bidirectional (among unmasked) | Full bidirectional (among unmasked) |
| Masking strategy | Cosine schedule, binary mask | Cosine schedule, binary mask |
| Training objective | AR CE + Mask-predict CE | CE on all positions |
| Two-stream? | No | No (single-stream, lower training FLOPs than ours) |

#### 3.2 Implementation

```python
def get_show_o_style_mask(text_mask, image_mask, mask_ratio=0.5, seq_len=None):
    """
    Show-o-style attention mask:
    - Text positions: standard causal (kv_idx <= q_idx)
    - Image positions: binary random mask
      - Masked image tokens: can only see unmasked image tokens + all text tokens
      - Unmasked image tokens: full bidirectional among themselves + see all text tokens

    This is single-stream (no XT), so attention is computed once per layer.
    """
    B = text_mask.shape[0]

    def show_o_mask(b, h, q_idx, kv_idx):
        q_is_text = text_mask[b, q_idx]
        q_is_image = image_mask[b, q_idx]
        kv_is_text = text_mask[b, kv_idx]
        kv_is_image = image_mask[b, kv_idx]
        kv_is_masked = image_masked[b, kv_idx] if kv_is_image else False

        # Text tokens use causal attention
        if q_is_text:
            return kv_idx <= q_idx  # standard causal

        # Image tokens
        if q_is_image:
            if kv_is_text:
                return True  # image can see all text
            if kv_is_image:
                if q_is_masked[b, q_idx]:
                    # Masked image Q: can only see unmasked image K/V
                    return not kv_is_masked
                else:
                    # Unmasked image Q: full bidirectional among all unmasked
                    # plus sees text (already handled above)
                    return not kv_is_masked
            return False

        return False

    return create_block_mask(show_o_mask, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
```

#### 3.3 Phase 1 Variant Table (Updated)

| Variant | Architecture | Text Attention | Image Attention | Image Sigma/Mask | Purpose |
|---------|-------------|---------------|-----------------|------------------|---------|
| A | Two-stream selfless | σ_kv > σ_q, AR σ | σ_kv > σ_q, AR σ | AR descending | AR upper bound |
| B | Two-stream selfless | σ_kv > σ_q, AR σ | σ_kv > σ_q, random σ | Random uniform | Our core method |
| C | Two-stream selfless | σ_kv > σ_q, high σ (2-3) | σ_kv > σ_q, low σ (0-1) | Conditional sigma | text→image generation mode |
| D | Two-stream selfless | σ_kv > σ_q, AR σ | σ_kv > σ_q, center-out σ | Center-out spatial | 2D sigma schedule |
| E | Two-stream selfless | σ_kv > σ_q, AR σ | σ_kv > σ_q, random σ | Random uniform + 2D RoPE | 2D position ablation |
| **F** | **Single-stream** | **Causal** | **Full bidirectional** | **Binary mask (cosine)** | **★ Show-o baseline** |

**Note on FLOPs fairness:** Variant F (single-stream) has ~1× training FLOPs while our variants A-E (two-stream) have ~1.8-2.0×. For fairness:
- Primary comparison: equal training steps (parameter-matched, FLOPs-different). Report FLOPs ratio transparently.
- Secondary comparison: if Variant F is within 10% of our best variant despite using 50% less compute, this weakens our contribution. This should be discussed honestly.

#### 3.4 Why This Baseline Is Critical

Without Variant F, Phase 1 can only tell us which **sigma configuration within selfless attention** works best. It CANNOT tell us whether **selfless attention itself** is better than the Show-o dual-mask approach. Variant F answers the core research question: "Does continuous sigma ordering outperform binary-mask modality-switching for multimodal training?"

---

### 4. Conceptual Correction: σ_kv > σ_q Implies Inverted Quality Hierarchy

**The precise relationship between sigma and representation quality under selfless attention:**

Let L be the number of Transformer layers. Let h^{(ℓ)}_i be the hidden state of token i at layer ℓ.

At layer 0: h^{(0)}_i = Embed(token_i). All tokens have equally "raw" representations.

At layer 1:
- Token i with σ_i = 0.9 attends to ~10% of tokens → h^{(1)}_i aggregates information from ~10% of the sequence
- Token j with σ_j = 0.1 attends to ~90% of tokens → h^{(1)}_j aggregates information from ~90% of the sequence

After L layers, the quality gap compounds. Low-σ tokens have progressively richer representations because at each layer they can aggregate information from a broader set of (increasingly contextualized) sources. High-σ tokens remain relatively isolated — they see few other tokens and those they see are themselves poorly contextualized.

**This is a structural property of the σ_kv > σ_q rule, not a bug.** Whether it's beneficial for learning is an empirical question that Phase 1 will address.

**Corrected phrasing for paper:**

> Under selfless attention (σ_kv > σ_q), token visibility and representation quality exhibit an inverse relationship: tokens with high sigma values are highly visible to others but poorly contextualized themselves, while tokens with low sigma values are less visible but carry richer, more aggregated representations. The query stream (XT) therefore receives a heterogeneous mixture of "raw" (high-σ, isolated) and "refined" (low-σ, connected) key/value signals. This contrasts with MaskGIT-style binary masking, where all unmasked tokens' representations are homogeneously contextualized through mutual bidirectional attention.
- **MPNet** (Microsoft, NeurIPS 2020): "Masked and Permuted Pre-training for Language Understanding" — [arXiv:2004.09297](https://arxiv.org/abs/2004.09297)
