# Unified-MM Project Context

This project studies **permutation-based selfless attention** for unified text-image pretraining. The current implementation is centered on **OpenGVLab/OmniCorpus-CC-210M** as the only multimodal pretraining dataset. Older COCO, FineWeb, and synthetic SII data paths are obsolete and have been removed.

## Research Thesis

Autoregressive multimodal models force a single left-to-right order over text and image tokens. That is reasonable for language but a poor inductive bias for 2D visual tokens. This project uses one shared Transformer and one strict sigma-ordered attention rule:

```text
q attends to kv iff sigma[kv] > sigma[q]
```

The contribution is not “no modality-specific components.” Text and image still need different vocabularies, embeddings, and output heads. The contribution is that the **core Transformer layers and attention rule are shared**, while modality-appropriate behavior is expressed through sigma schedules.

## Current Data Direction

The project is now **OmniCorpus-only**:

```text
OpenGVLab/OmniCorpus-CC-210M parquet
  -> interleaved document JSONL + downloaded images
  -> Open-MAGVIT2 image tokens
  -> pre-tokenized Arrow shards
  -> packed training batches
```

Training must not read OmniCorpus parquet, download images, tokenize text, or run MAGVIT2. All expensive work happens offline.

## Sequence Format

Every document is converted to:

```text
text [BOI] image_tokens [EOI] text [BOI] image_tokens [EOI] ... [EOS]
```

Hard invariants:

- `token_types`: `0=text`, `1=image`, `2=special`, `3=padding`.
- Every image span is exactly `BOI + 256 image tokens + EOI`.
- Image tokens are Open-MAGVIT2 codes shifted by `image_offset`.
- Image spans are never sliced or truncated.
- Long documents are shortened by trimming text first, then dropping complete image blocks from the end.
- `EOI` labels are `-100`.
- Padding labels are `-100`.
- `EOS` separates documents when multiple documents are packed together.

## Current Architecture

Core model:

- `models/modeling_model/modeling_selfless.py`
- Qwen3-based two-stream Transformer.
- X0 stream receives real token embeddings and produces K/V.
- XT stream receives mask embeddings and produces Q during training.
- Inference uses X0 only.

Multimodal token handling:

- Text ids use the tokenizer vocabulary.
- Image ids are `image_offset + image_code`.
- Dual-head mode uses a text LM head and image LM head.
- Loss is `text_loss + lambda_image * image_loss`.

Data loading:

- `utils/dataset_omnicorpus.py`
- `OmniCorpusPackedDataset` loads Arrow shards by memory mapping.
- It stores document lengths and pack indices, not all token lists.
- `__getitem__` concatenates the indexed documents, validates image spans, assigns sigma/labels, and returns tensors.
- `set_epoch()` shuffles pack order but keeps dataset length stable.

## Active Configs And Commands

Main config:

```text
configs/selfless/omnicorpus.yaml
```

Launch training:

```bash
bash script/selfless/pretraining_omnicorpus.sh
```

Download OmniCorpus shards:

```bash
uv run python scripts/omnicorpus_download.py \
  --max_shards 4 \
  --local_dir public/datasets/omnicorpus/raw
```

Prepare interleaved documents and images:

```bash
uv run python scripts/omnicorpus_prepare_docs.py \
  --parquet_glob 'public/datasets/omnicorpus/raw/data/**/*.parquet' \
  --output_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_dir public/datasets/omnicorpus/images
```

Encode images:

```bash
uv run python scripts/omnicorpus_encode_images.py \
  --docs_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_dir public/datasets/omnicorpus/images \
  --output_dir public/datasets/omnicorpus/image_tokens_magvit2 \
  --device cuda
```

Build Arrow shards:

```bash
uv run python scripts/omnicorpus_build_arrow.py \
  --config configs/selfless/omnicorpus.yaml \
  --docs_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_token_dir public/datasets/omnicorpus/image_tokens_magvit2 \
  --output_dir public/datasets/omnicorpus/arrow
```

Visualize dataloader:

```bash
uv run python scripts/viz_dataloader.py --config configs/selfless/omnicorpus.yaml
```

## Implementation Status

Done:

- Two-stream selfless attention.
- Strict selfless attention mask.
- Qwen3 model loading.
- Open-MAGVIT2 wrapper.
- Dual text/image embedding and LM head path.
- Modality-aware loss.
- OmniCorpus parquet-to-doc conversion.
- OmniCorpus image download and MAGVIT2 encoding.
- OmniCorpus Arrow builder.
- OmniCorpus packed dataloader.

Still research/development work:

- 2D image sigma schedules.
- 2D RoPE for image tokens.
- Show-o-style baseline.
- Gradient conflict measurement.
- Full-scale distributed preprocessing.
- Multimodal evaluation suite.

## Engineering Notes

- Keep preprocessing offline and resumable.
- Do not reintroduce COCO/FineWeb/SII synthetic data into the main multimodal path.
- Do not truncate image tokens.
- Do not change `image_tokens_per_img=256` without updating all validation and model assumptions.
- Any new dataset transform must preserve `BOI/EOI/EOS` boundaries.
- Before claiming data changes are safe, run a small Arrow build and dataloader smoke test.
