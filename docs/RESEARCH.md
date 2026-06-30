# Unified Multimodal Model via Two-Stream LLM and Contextual Image Flow

This document describes the current research direction of the repository. It
supersedes the older plan centered on discrete image-token cross-entropy and
OmniCorpus-only pretraining.

## Project Thesis

Autoregressive language models are powerful because every token can be predicted
from a causal history. The usual implementation, however, trains hidden state
`h_i` to predict token `x_{i+1}`. That one-position shift becomes awkward for a
unified text-image model: image generation wants a condition at the same spatial
slot as the latent being generated, while image understanding wants text tokens
to naturally attend to image latents as ordinary context.

The current thesis is:

> A two-stream LLM can preserve autoregressive language modeling while removing
> the tensor-level shift. Once every target is predicted at its own position, the
> same hidden state can condition a contextual rectified-flow head for continuous
> image latents. Text-to-image and image-to-text then become two sigma schedules
> over the same shared attention mechanism.

The model is "unified" in the following precise sense:

- the Transformer backbone is shared;
- text and image positions use the same strict attention rule;
- image generation conditions are produced by ordinary self-attention;
- image understanding uses ordinary attention over image latent tokens;
- modality-specific pieces are limited to input projection and output loss
  heads, because text tokens and continuous image latents are different data
  types.

## What Changed From The Old Plan

The old document described a permutation-based model over discrete image tokens,
with an image vocabulary, image LM head, and image-token cross-entropy. That is
not the active path.

The current code uses:

- continuous KL16 VAE latents with shape `[256, 16]`;
- `ImageTokenEmbedder` to project visible image latents into the Qwen hidden
  space;
- `FlowLoss` from `models/modeling_model/image_flow_loss.py` to train a velocity
  field over each image latent token;
- `Qwen3ForCausalLM.forward()` to combine text CE and image flow loss;
- `sample_image_latents_single_stream()` to generate images by repeatedly using
  backbone hidden states as flow conditions.

The strict sigma attention mechanism remains central, but its role changed. It
is no longer mainly a discrete image-token ordering trick. It is the mechanism
that aligns text targets, visible image latents, and generated image slots in a
single sequence.

## Architecture

### Strict Selfless Attention

Every position receives a scalar sigma value. Attention is allowed only when:

```text
sigma[kv] < sigma[q]
```

The diagonal is excluded. A position cannot see its own content token. This is
implemented in `utils/utils.py:get_selfless_mask()`.

This one rule covers:

- left-to-right text modeling: `sigma = [0, 1, 2, ...]`;
- image generation: prompt and already-filled image latents have lower sigma,
  future image slots have higher sigma;
- image understanding: image latents have lower sigma than answer text tokens;
- random or spatial image generation orders: image slot sigma values can be
  permuted.

### Two Streams

The Qwen3 backbone is modified into two streams:

- `X0` content stream receives real content where it is visible.
- `XT` query stream receives mask embeddings and predicts the target at the same
  position.

In each attention layer:

```text
Q_X0 = q_proj(X0)
K_X0 = k_proj(X0)
V_X0 = v_proj(X0)
Q_XT = q_proj(XT)

X0 attends to X0 K/V with sigma[kv] < sigma[q]
XT attends to X0 K/V with sigma[kv] < sigma[q]
```

Only `XT` is used for likelihood training. At inference, when only contextual
hidden states are needed for generation, the code can run the `X0` stream only.

### Why This Removes The Shift

For text, training uses:

```text
input_ids = [x_0, x_1, ..., x_n]
labels    = [x_0, x_1, ..., x_n]
sigma     = [0,   1,   ..., n]
```

The query stream at position `i` contains a mask embedding, cannot attend to its
own content, and can only attend `x_<i`. Therefore it predicts `x_i` from the
same context a causal LM would use. The loss is still autoregressive in
information flow, but the target is no longer stored one slot to the right.

This matters for images because the hidden state at image position `i` is now
the natural condition for latent `z_i`.

## Image Flow Objective

Image latents are continuous. For each image position, the model obtains:

- target latent `z_0` from `image_latents`;
- condition `c` from the `XT` hidden state at the same position;
- optional flow positional embedding based on the local image index.

`FlowLoss.forward()` samples noise and time:

```text
t ~ logit_normal/uniform mixture
eps ~ N(0, I)
z_t = (1 - t) * eps + t * z_0
v_target = z_0 - eps
v_pred = flow_head(z_t, t, condition=c)
loss_image = mean((v_pred - v_target)^2)
```

The total loss is:

```text
loss = lambda_text * loss_text + lambda_image * loss_image
```

where text positions use cross-entropy through the Qwen LM head, and image
positions use only the flow loss.

## Data Representation

The active image dataset is `ImageNetFlowCacheDataset`.

Each sample is represented as:

```text
prefix text [BOI] image slots [EOI] suffix text [EOS]
```

or, without prompts:

```text
[BOI] image slots [EOI] [EOS]
```

Important details:

- image slots use `<|img_mask|>` in `input_ids`;
- real image targets live in `image_latents`;
- image slots have `token_types == 1`;
- image CE labels are `-100`;
- text, BOI, suffix, and EOS can contribute to text CE;
- EOI is kept as context but ignored by CE;
- image slots are exactly 256 KL16 latent tokens;
- local image positions are computed from contiguous image spans.

The collator assigns sigma values so that prompt/special context is visible and
image slots receive a randomized generation order. Suffix text receives later
sigma values, so it can attend to image latent tokens for understanding-style
training.

The mixed dataset is `CombinedImageNetTextFlowDataset`, which alternates image
flow batches with text-only Arrow batches. This protects language ability while
the image flow modules and shared backbone are trained.

## Generation And Understanding

### Text To Image

For image generation, a prompt is placed in the sequence, followed by BOI and
masked image slots. The model repeatedly:

1. builds the strict selfless attention mask from current sigma values;
2. runs the backbone to obtain hidden states at unfilled image slots;
3. normalizes those hidden states into flow conditions;
4. samples image latents with the contextual flow head;
5. writes selected latents back into the sequence as visible `X0` content;
6. updates sigma/order and continues until all image slots are filled.

The implemented order strategies include sigma replay, random order,
hidden-norm confidence, latent-projection cosine, and spatial Halton-style
orders. CFG is implemented by building an unconditional image attention mask
that removes non-image context for selected image rows.

### Image To Text

For understanding, image latents are already visible in the content stream.
Answer text positions are assigned later sigma values and trained with text CE.
Thus text queries naturally attend to image K/V through the same Transformer
layers. No special vision encoder or cross-attention bridge is needed.

## Training Pipeline

The current staged path is:

1. `text_selfless_2048_ft`
   - adapt a Qwen3 checkpoint to the two-stream selfless text objective;
   - text labels are not shifted;
   - unused image modules can be frozen.

2. `imagenet_flow_stage0_10c`
   - small 10-class ImageNet run for debugging the current flow baseline;
   - exercises train, validation, single-stream generation, and image decode.

3. `imagenet_flow_full_from_qwen3base`
   - full ImageNet class-conditioned flow training from Qwen3-Base;
   - trains the image latent embedder, Qwen backbone, condition projection, and
     contextual flow head as the current baseline.

## Implementation Map

- `models/modeling_model/modeling_selfless_flow.py`
  - `ImageTokenEmbedder`
  - two-stream Qwen attention and decoder layers
  - text CE plus image flow loss
  - single-stream image latent sampling

- `models/modeling_model/image_flow_loss.py`
  - rectified-flow objective
  - velocity prediction
  - Euler/Heun sampling

- `utils/utils.py`
  - special token registration
  - model selection
  - strict selfless `BlockMask`
  - image-unconditional CFG mask support

- `utils/dataset_imagenet_flow_cache.py`
  - KL16 latent-cache dataset
  - prompt template rendering
  - sigma and label assignment

- `utils/dataset_combined_flow.py`
  - text Arrow dataset
  - mixed image/text dataloader
  - accumulation-aware batch scheduling

- `pretrain/train_selfless_flow.py`
  - active training loop
  - optimizer LR groups for backbone/projector/flow head/special tokens
  - EMA support
  - flow adapter save/load
  - validation image decoding and flow diagnostics

## Research Questions

### RQ1: Does Same-Position Training Help?

Compare the two-stream same-position objective against a conventional shifted
causal baseline for conditioning image flow. The hypothesis is that same-slot
conditioning produces cleaner image latent conditions because the hidden state
and target latent are aligned by construction.

Useful signals:

- text validation loss and perplexity;
- image flow MSE;
- `x0` estimate MSE at probe times;
- validation decoded images;
- gradient norms for the image projector and flow head.

### RQ2: Is Attention A Sufficient Image Condition?

The model should not need a separate image-conditioning network. Text-to-image
quality should improve when prompt tokens are visible to image slots, and
image-to-text behavior should improve when text suffix/answer tokens can attend
image latents.

Useful ablations:

- full context vs image-unconditional context;
- prompt templates vs no prompt;
- suffix text after image vs image-only batches;
- frozen backbone vs unfrozen backbone.

### RQ3: How Much Text Training Is Needed?

`CombinedImageNetTextFlowDataset` supports a configurable text batch ratio and
accumulation-aware scheduling. The goal is to prevent language drift while
training image flow.

Useful ablations:

- `text_batch_ratio`;
- `lambda_text`;
- backbone LR;
- EMA vs non-EMA validation.

### RQ4: Which Image Generation Order Works?

The sampler supports several order strategies. The important question is not
only final image quality, but whether the strategy uses the model's attention
conditions efficiently.

Useful ablations:

- `sigma`;
- `random`;
- `hidden_norm`;
- `latent_proj_cosine`;
- `spatial_halton`;
- `spatial_uniform`;
- different `parallel_rate` values.

### RQ5: How Valuable Is Legacy Adapter Migration?

The loader can migrate legacy `.safetensors` keys into the image flow head,
image projector, flow positional embeddings, and image mask token. The key
question is whether this provides a better initialization than training the flow
head from scratch under the LLM condition distribution.

## Evaluation Plan

Early-stage validation should focus on structural and latent-space signals:

- `train/loss_text`;
- `train/loss_image_flow`;
- `train/flow/v_mse`;
- `train/flow/v_pred_rms`;
- `val/flow_x0_est_*_latent_mse`;
- `val/flow_full_sample_latent_mse`;
- single-stream strategy MSE to target and teacher;
- decoded validation grids.

FID and large-scale understanding benchmarks should come later. They are not
reliable indicators during the current adapter/warmup phase.

## Known Risks

- ImageNet prompts are a narrow text-image distribution. They test the mechanism
  but are not a full multimodal pretraining corpus.
- Flow sampling quality depends on the hidden-state condition distribution. A
  good flow head under teacher-forced conditions may still fail during iterative
  generation if generated latents drift.
- Single-stream image generation can be slower than AR text decoding because it
  repeatedly reruns sequence attention and flow sampling.
- The two-stream training path increases training compute. Inference for
  condition extraction can omit `XT`, but likelihood training needs both streams.
- Too little text mixing can degrade language ability; too much can starve image
  flow learning.

## Current Contribution Statement

This project studies a unified multimodal objective in which a two-stream LLM
removes the standard autoregressive label shift, aligns every target with its
own query position, and uses those aligned hidden states to condition a
contextual rectified-flow head over continuous image latents. Under this design,
image generation and image understanding are not separate architectural modes:
they are produced by the same strict attention rule under different sigma
schedules.
