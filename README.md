# Unified-MM: Selfless Attention for Interleaved Multimodal Pretraining

This repository investigates **permutation-based selfless attention** for unified text-image pretraining. The model uses a single sigma-ordered attention rule over all tokens:

```text
token q attends to token kv iff sigma[kv] < sigma[q]
```

The current project direction is **OmniCorpus-only pretraining** using real web documents from `OpenGVLab/OmniCorpus-CC-210M`. Older COCO/FineWeb/SII synthetic data paths have been removed.

## Core Idea

Autoregressive multimodal models impose a fixed left-to-right order on both text and image tokens. That order is natural for text but unnatural for 2D image tokens. This project keeps one shared Transformer and one unified attention mechanism, but changes the token ordering through sigma schedules:

- Text tokens keep AR-style ordering.
- Image tokens can use permutation/random or structured 2D ordering.
- Interleaved documents preserve their natural web order.
- Cross-modal behavior comes from sigma values, not from switching attention implementations by modality.

The training sequence format is:

```text
text [BOI] image_tokens [EOI] text [BOI] image_tokens [EOI] ... [EOS]
```

Hard invariants:

- Each image is exactly 256 Open-MAGVIT2 tokens.
- Arrow stores image tokens as raw Open-MAGVIT2 codebook ids in `[0, image_vocab_size)`.
- `token_types`, not `image_offset`, determine whether an `input_id` is text, image, special, or padding.
- Image spans are never truncated.
- Every image span must be `BOI + 256 image tokens + EOI`.
- Long documents are shortened by trimming text first, then dropping complete image blocks.
- Training consumes pre-tokenized Arrow shards only; it does not download images, read parquet, tokenize text, or run MAGVIT2.

## Data Pipeline

1. Download OmniCorpus parquet shards:

```bash
uv run python scripts/omnicorpus_download.py \
  --max_shards 4 \
  --local_dir public/datasets/omnicorpus/raw
```

2. Convert parquet documents to interleaved JSONL and downloaded images:

```bash
uv run python scripts/omnicorpus_prepare_docs.py \
  --parquet_glob 'public/datasets/omnicorpus/raw/data/**/*.parquet' \
  --output_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_dir public/datasets/omnicorpus/images \
  --download_workers 1024 \
  --batch_size 8192 \
  --candidate_batch_size 4096
```

If a run was killed after writing `train.jsonl` but before a checkpoint was
created, recover from the JSONL tail once:

```bash
uv run python scripts/omnicorpus_prepare_docs.py \
  --parquet_glob 'public/datasets/omnicorpus/raw/data/**/*.parquet' \
  --output_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_dir public/datasets/omnicorpus/images \
  --recover_from_jsonl \
  --download_workers 1024 \
  --batch_size 8192 \
  --candidate_batch_size 4096
```

3. Encode images with Open-MAGVIT2:

```bash
uv run python scripts/omnicorpus_encode_images.py \
  --docs_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_dir public/datasets/omnicorpus/images \
  --output_dir public/datasets/omnicorpus/image_tokens_magvit2 \
  --device cuda
```

4. Build Arrow shards for training:

```bash
uv run python scripts/omnicorpus_build_arrow.py \
  --config configs/selfless/omnicorpus.yaml \
  --docs_jsonl public/datasets/omnicorpus/docs/train.jsonl \
  --image_token_dir public/datasets/omnicorpus/image_tokens_magvit2 \
  --output_dir public/datasets/omnicorpus/arrow
```

## Training

```bash
bash script/selfless/pretraining_omnicorpus.sh
```

Main config:

```text
configs/selfless/omnicorpus.yaml
```

The dataloader is implemented in:

```text
utils/dataset_omnicorpus.py
```

It loads Arrow shards with memory mapping, packs multiple documents up to `max_seq_length`, pads at collate time, and returns:

```python
{
    "input_ids": LongTensor[B, L],
    "token_types": UInt8Tensor[B, L],
    "sigma": LongTensor[B, L],
    "labels": LongTensor[B, L],
}
```

`token_types`: `0=text`, `1=image`, `2=special`, `3=padding`.

## Validation

Visualize a batch:

```bash
uv run python scripts/viz_dataloader.py --config configs/selfless/omnicorpus.yaml
```

Run Python syntax checks:

```bash
uv run python -m py_compile \
  scripts/omnicorpus_*.py \
  utils/dataset_omnicorpus.py \
  pretrain/train_selfless.py
```

## Key Files

- `models/modeling_model/modeling_selfless.py`: two-stream selfless Qwen3 model.
- `utils/utils.py`: tokenizer/model loading and selfless mask creation.
- `utils/dataset_omnicorpus.py`: OmniCorpus packed Arrow dataloader.
- `scripts/omnicorpus_*.py`: offline OmniCorpus preprocessing.
- `scripts/magvit2_wrapper.py`: Open-MAGVIT2 tokenizer wrapper.
- `configs/selfless/omnicorpus.yaml`: active multimodal pretraining config.
- `docs/RESEARCH.md`: research plan and experimental rationale.
