# Unified-MM: Two-Stream LLM with Contextual Image Flow

This repository trains a unified text-image model by combining a two-stream
Qwen3 language backbone with a contextual rectified-flow head over continuous
image latents.

The current idea is simple:

- use a two-stream LLM to remove the usual one-token label shift;
- keep text prediction aligned with the same sequence position;
- use the hidden state at an image position as the condition for a flow head;
- train image generation as continuous latent flow rather than discrete image
  token cross-entropy;
- let text-to-image and image-to-text behavior emerge from the same attention
  rule and sigma ordering.

The active path is implemented in `models/modeling_model/modeling_selfless_flow.py`.

## Core Idea

Standard causal LMs train position `i` to predict token `i + 1`. That shift is
awkward for a unified text-image model because the hidden state is not naturally
aligned with the image latent it should generate.

This code uses two streams:

- `X0` is the content stream. It receives real text embeddings and visible image
  latent embeddings, then provides K/V to attention.
- `XT` is the query stream. It receives mask embeddings and predicts the token
  or image latent at the same position.

Both streams use the same strict selfless attention mask:

```text
query position q attends to key/value position k iff sigma[k] < sigma[q]
```

Because the diagonal is excluded, a position cannot see its own content token.
With AR sigma values, text still behaves like left-to-right language modeling,
but the tensors no longer need a shifted label convention: `labels == input_ids`
at valid text positions. For image positions, the `XT` hidden state at that same
position becomes the conditioning vector for a contextual flow head.

## Unified Multimodal Behavior

Image generation uses normal attention as conditioning. Prompt text and already
known image latents are placed at lower sigma values. Image slots to be generated
query the backbone, attend to visible context through the shared Transformer, and
the resulting hidden states condition `FlowLoss.sample()`.

Image understanding is the same mechanism in the other direction. Image latents
are embedded into the `X0` stream. Later text positions attend those image
tokens through the same selfless mask and are trained with standard text
cross-entropy. There is no separate vision cross-attention module.

The total supervised objective is:

```text
loss = lambda_text * CE(text_head(XT_hidden), text_label)
     + lambda_image * flow_mse(image_latent, condition=XT_hidden)
```

## Current Data Format

Training batches return:

```python
{
    "input_ids": LongTensor[B, L],
    "token_types": UInt8Tensor[B, L],
    "sigma": LongTensor[B, L],
    "labels": LongTensor[B, L],
    "image_latents": FloatTensor[B, L, image_latent_dim],
}
```

`token_types` are:

```text
0 = text
1 = image latent slot
2 = special token
3 = padding
```

For image slots, `input_ids` contains the image mask token and the real target
lives in `image_latents`. Image labels are `-100` for text CE; image supervision
comes from the flow loss.

The active ImageNet flow format is:

```text
optional prompt text [BOI] 256 image latent slots [EOI] optional suffix text [EOS]
```

The 256 image slots are KL16 VAE latents with shape `[256, 16]`.

## Active Training Path

1. Adapt the Qwen text backbone to the selfless two-stream objective:

```bash
bash script/selfless/pretraining_text_selfless_2048.sh
```

2. Encode ImageNet images into the KL16 latent cache:

```bash
bash script/selfless/encode_imagenet_full_kl16_vae.sh
```

3. Run the 10-class ImageNet stage0 preflight/debug training:

```bash
bash script/selfless/pretraining_imagenet_flow_stage0_10c.sh
```

4. Train the full ImageNet flow baseline from Qwen3-Base:

```bash
bash script/selfless/pretraining_imagenet_flow_full_from_qwen3base.sh
```

Main configs:

```text
configs/selfless/text_selfless_2048_ft.yaml
configs/selfless/imagenet_flow_stage0_10c.yaml
configs/selfless/imagenet_flow_full_from_qwen3base.yaml
```

## Key Files

- `models/modeling_model/modeling_selfless_flow.py`: two-stream Qwen3 model,
  image latent embedding, text CE, image flow loss, and image latent sampling.
- `models/modeling_model/image_flow_loss.py`: contextual rectified-flow objective.
- `utils/utils.py`: tokenizer/model loading and strict selfless mask creation.
- `utils/dataset_imagenet_flow_cache.py`: ImageNet latent-cache dataset.
- `utils/dataset_combined_flow.py`: mixed ImageNet-flow and text dataloaders.
- `pretrain/train_selfless_flow.py`: active training loop, validation image
  decoding, EMA, adapter save/load, and flow diagnostics.
- `scripts/imagenet_encode_kl16_vae.py`: KL16 VAE latent encoder.
- `scripts/generate_flow_validation_images.py`: manual flow validation and
  strategy comparison.
- `docs/RESEARCH.md`: current research plan and rationale.

## Validation

Run the focused behavior tests:

```bash
uv run pytest tests/test_selfless_flow_behavior.py
```

Run a manual image-flow validation pass:

```bash
uv run python scripts/generate_flow_validation_images.py \
  --config configs/selfless/imagenet_flow_full_from_qwen3base.yaml \
  --single_stream \
  --strategies sigma,spatial_halton,hidden_norm
```

For text regression against a local checkpoint:

```bash
uv run python scripts/check_selfless_flow_text_regression.py
```

## Legacy Paths

Older discrete image-token and OmniCorpus preprocessing code is still present in
the repository, but it is no longer the active research path described here. The
mainline flow configs train on continuous KL16 latents and text Arrow shards.
