# Unified-MM Project Context

This project currently studies a **two-stream Qwen3 LLM with contextual image
flow** for unified text-image modeling. Older documentation about
OmniCorpus-only discrete image-token CE is obsolete for the active path.

## Research Thesis

Standard causal LMs train hidden state `h_i` to predict token `x_{i+1}`. This
one-token shift is inconvenient for a unified multimodal model because image
generation wants a hidden condition aligned with the same spatial latent slot it
will generate.

The active model removes that tensor-level shift with two streams:

- `X0`: content stream with real text tokens and visible image latent embeddings.
- `XT`: query stream with mask embeddings, trained to predict the target at the
  same sequence position.

Both streams use one strict attention rule:

```text
q attends to kv iff sigma[kv] < sigma[q]
```

The diagonal is excluded, so a token cannot see itself. With AR sigma values,
text still gets causal information flow, but `labels == input_ids` at valid text
positions. Image positions use the aligned `XT` hidden state as the condition
for a contextual rectified-flow head.

## Active Architecture

Core model:

- `models/modeling_model/modeling_selfless_flow.py`
- Qwen3-based two-stream Transformer.
- `ImageTokenEmbedder` projects KL16 VAE latents into the hidden space.
- `FlowLoss` predicts rectified-flow velocity for continuous image latents.
- Text uses normal LM-head cross-entropy.
- Image slots use flow MSE, not discrete image-token CE.

Key objective:

```text
loss = lambda_text * text_ce + lambda_image * image_flow_mse
```

Text-to-image generation works by letting image slots attend to prompt/context
tokens, then sampling KL16 latents from the flow head. Image-to-text works
by letting later text positions attend image latent tokens through the same
Transformer attention.

## Current Data Direction

Active image data uses cached KL16 VAE latents:

```text
ImageNet image -> KL16 VAE -> Tensor[256, 16]
```

Each training sample is represented as:

```text
optional text [BOI] 256 image latent slots [EOI] optional text [EOS]
```

Batch fields:

- `input_ids`: text/special ids; image slots use `<|img_mask|>`.
- `token_types`: `0=text`, `1=image`, `2=special`, `3=padding`.
- `sigma`: strict attention order.
- `labels`: text CE labels; image positions are `-100`.
- `image_latents`: aligned continuous image targets.

## Active Configs And Commands

Text selfless adaptation:

```bash
bash script/selfless/pretraining_text_selfless_2048.sh
```

Encode ImageNet into the KL16 latent cache:

```bash
bash script/selfless/encode_imagenet_full_kl16_vae.sh
```

10-class ImageNet flow debug run:

```bash
bash script/selfless/pretraining_imagenet_flow_stage0_10c.sh
```

Full ImageNet flow training:

```bash
bash script/selfless/pretraining_imagenet_flow_full_from_qwen3base.sh
```

Main configs:

```text
configs/selfless/text_selfless_2048_ft.yaml
configs/selfless/imagenet_flow_stage0_10c.yaml
configs/selfless/imagenet_flow_full_from_qwen3base.yaml
```

## Implementation Status

Done:

- Two-stream strict selfless attention for Qwen3.
- Shift-free text objective with same-position labels.
- Continuous image latent embedding path.
- Contextual rectified-flow image loss.
- Legacy adapter migration for image modules.
- ImageNet latent-cache dataset.
- Combined image/text dataloader.
- Single-stream image latent generation with multiple order strategies.
- Validation image decoding and flow diagnostics.

Still research/development work:

- Better multimodal data beyond ImageNet class-prompt conditioning.
- Stronger understanding evaluation.
- Generation-order ablations.
- Text retention vs image learning balance.
- More robust long-context/interleaved multimodal training.

## Engineering Notes

- The active training loop is `pretrain/train_selfless_flow.py`.
- The active model class is loaded when the project name contains `flow`.
- Do not reintroduce shifted labels into the selfless flow path.
- Do not treat image slots as discrete codebook CE targets in the active flow
  configs.
- Keep `image_tokens_per_img=256` and `image_latent_dim=16` aligned with KL16
  latent assumptions unless all datasets, validation, and samplers are updated.
- If changing image generation, check `sample_image_latents_single_stream()` and
  `scripts/generate_flow_validation_images.py`.
- If changing attention visibility, check `utils/utils.py:get_selfless_mask()`.
