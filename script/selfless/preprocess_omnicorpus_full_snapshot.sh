#!/usr/bin/env bash
# =============================================================================
# Full OmniCorpus snapshot preprocessing for training.
#
# Produces:
#   DOCS_JSONL + JPG images -> Open-MAGVIT2 .pt tokens -> Arrow shards
#
# The Arrow format is shared by unified_head=true and unified_head=false:
#   - image tokens stay raw Open-MAGVIT2 codebook ids
#   - token_types marks text/image/special/padding
#   - image_offset is applied by the model at runtime only for unified_head
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

DOCS_JSONL="${DOCS_JSONL:-public/datasets/omnicorpus/docs/snapshots/train_20260623_064740_2946703docs.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-public/datasets/omnicorpus/images}"
CONFIG="${CONFIG:-configs/selfless/omnicorpus.yaml}"

OUTPUT_ROOT="${OUTPUT_ROOT:-public/datasets/omnicorpus/pretrain_full_snapshot}"
IMAGE_TOKEN_DIR="${IMAGE_TOKEN_DIR:-${OUTPUT_ROOT}/image_tokens_magvit2}"
ARROW_DIR="${ARROW_DIR:-${OUTPUT_ROOT}/arrow}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

SHARD_SIZE="${SHARD_SIZE:-100000}"
MAX_IMAGES="${MAX_IMAGES:--1}"
MAX_DOCS="${MAX_DOCS:--1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ENCODE_WORKERS="${ENCODE_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ARROW_WORKERS="${ARROW_WORKERS:-8}"
VALIDATE_SAMPLE_DOCS="${VALIDATE_SAMPLE_DOCS:-2000}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
SKIP_ENCODE="${SKIP_ENCODE:-0}"
SKIP_ARROW="${SKIP_ARROW:-0}"

if [ -z "${DEVICES:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
    else
        DEVICES="0"
    fi
fi

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
NUM_SHARDS="${#DEVICE_LIST[@]}"

if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "No CUDA devices configured. Set DEVICES=0 or DEVICES=0,1,2,3." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$IMAGE_TOKEN_DIR" "$LOG_DIR"

echo "=== OmniCorpus Full Snapshot Preprocessing ==="
echo "repo:             $SCRIPT_DIR"
echo "docs_jsonl:       $DOCS_JSONL"
echo "image_dir:        $IMAGE_DIR"
echo "config:           $CONFIG"
echo "output_root:      $OUTPUT_ROOT"
echo "image_token_dir:  $IMAGE_TOKEN_DIR"
echo "arrow_dir:        $ARROW_DIR"
echo "log_dir:          $LOG_DIR"
echo "devices:          $DEVICES"
echo "num_shards:       $NUM_SHARDS"
echo "shard_size:       $SHARD_SIZE"
echo "max_images:       $MAX_IMAGES"
echo "max_docs:         $MAX_DOCS"
echo "batch_size:       $BATCH_SIZE"
echo "encode_workers:   $ENCODE_WORKERS"
echo "prefetch_factor:  $PREFETCH_FACTOR"
echo "arrow_workers:    $ARROW_WORKERS"
echo

if [ ! -f "$DOCS_JSONL" ]; then
    echo "Missing DOCS_JSONL: $DOCS_JSONL" >&2
    exit 1
fi
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Missing IMAGE_DIR: $IMAGE_DIR" >&2
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "Missing CONFIG: $CONFIG" >&2
    exit 1
fi

if [ "$SKIP_VALIDATE" != "1" ]; then
    echo "=== Step 1/4: validate JSONL/JPG snapshot sample ==="
    uv run python scripts/omnicorpus_validate_data.py \
        --docs_jsonl "$DOCS_JSONL" \
        --image_dir "$IMAGE_DIR" \
        --sample_docs "$VALIDATE_SAMPLE_DOCS"
fi

if [ "$SKIP_ENCODE" != "1" ]; then
    echo "=== Step 2/4: encode images to Open-MAGVIT2 tokens ==="
    echo "Launching ${NUM_SHARDS} encoder shards. Logs:"
    pids=()
    names=()
    for shard_idx in "${!DEVICE_LIST[@]}"; do
        device="${DEVICE_LIST[$shard_idx]}"
        log_file="${LOG_DIR}/encode_shard_${shard_idx}_gpu_${device}.log"
        echo "  shard ${shard_idx}/${NUM_SHARDS} on GPU ${device}: ${log_file}"
        (
            export CUDA_VISIBLE_DEVICES="$device"
            uv run python scripts/omnicorpus_encode_images.py \
                --docs_jsonl "$DOCS_JSONL" \
                --image_dir "$IMAGE_DIR" \
                --output_dir "$IMAGE_TOKEN_DIR" \
                --device cuda:0 \
                --max_images "$MAX_IMAGES" \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$ENCODE_WORKERS" \
                --prefetch_factor "$PREFETCH_FACTOR" \
                --num_shards "$NUM_SHARDS" \
                --shard_index "$shard_idx"
        ) >"$log_file" 2>&1 &
        pids+=("$!")
        names+=("shard_${shard_idx}_gpu_${device}")
    done

    failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "Encoder ${names[$i]} finished."
        else
            echo "Encoder ${names[$i]} failed. See ${LOG_DIR}/encode_${names[$i]}.log" >&2
            failed=1
        fi
    done
    if [ "$failed" != "0" ]; then
        echo "At least one encoder shard failed; not building Arrow." >&2
        exit 1
    fi
fi

if [ "$SKIP_ARROW" != "1" ]; then
    echo "=== Step 3/4: build training Arrow shards ==="
    mkdir -p "$ARROW_DIR"
    if find "$ARROW_DIR" -maxdepth 1 -type d -name 'shard-*' | grep -q .; then
        echo "Arrow shards already exist in $ARROW_DIR." >&2
        echo "Use a new ARROW_DIR/OUTPUT_ROOT, or remove the old generated Arrow shards yourself before rebuilding." >&2
        exit 1
    fi
    if [ "$ARROW_WORKERS" -le 1 ]; then
        uv run python scripts/omnicorpus_build_arrow.py \
            --config "$CONFIG" \
            --docs_jsonl "$DOCS_JSONL" \
            --image_token_dir "$IMAGE_TOKEN_DIR" \
            --output_dir "$ARROW_DIR" \
            --shard_size "$SHARD_SIZE" \
            --max_docs "$MAX_DOCS" \
            2>&1 | tee "${LOG_DIR}/build_arrow.log"
    else
        echo "Launching ${ARROW_WORKERS} Arrow build workers. Logs:"
        pids=()
        names=()
        for worker_idx in $(seq 0 "$((ARROW_WORKERS - 1))"); do
            log_file="${LOG_DIR}/build_arrow_worker_${worker_idx}.log"
            echo "  worker ${worker_idx}/${ARROW_WORKERS}: ${log_file}"
            (
                uv run python scripts/omnicorpus_build_arrow.py \
                    --config "$CONFIG" \
                    --docs_jsonl "$DOCS_JSONL" \
                    --image_token_dir "$IMAGE_TOKEN_DIR" \
                    --output_dir "$ARROW_DIR" \
                    --shard_size "$SHARD_SIZE" \
                    --max_docs "$MAX_DOCS" \
                    --num_shards "$ARROW_WORKERS" \
                    --shard_index "$worker_idx"
            ) >"$log_file" 2>&1 &
            pids+=("$!")
            names+=("arrow_worker_${worker_idx}")
        done

        failed=0
        for i in "${!pids[@]}"; do
            if wait "${pids[$i]}"; then
                echo "Arrow ${names[$i]} finished."
            else
                echo "Arrow ${names[$i]} failed. See ${LOG_DIR}/build_${names[$i]}.log" >&2
                failed=1
            fi
        done
        if [ "$failed" != "0" ]; then
            echo "At least one Arrow worker failed." >&2
            exit 1
        fi
    fi
fi

echo "=== Step 4/4: quick Arrow sanity check ==="
if [ ! -d "$ARROW_DIR" ] || ! find "$ARROW_DIR" -maxdepth 1 -type d -name 'shard-*' | grep -q .; then
    echo "No Arrow shards found in $ARROW_DIR; skipping quick check because SKIP_ARROW=$SKIP_ARROW."
    exit 0
fi
uv run python - <<PY
from pathlib import Path
from datasets import load_from_disk

arrow_dir = Path("$ARROW_DIR")
shards = sorted(arrow_dir.glob("shard-*"))
if not shards:
    raise SystemExit(f"No Arrow shards found in {arrow_dir}")

ds = load_from_disk(str(shards[0]), keep_in_memory=False)
row = ds[0]
ids = row["input_ids"]
types = row["token_types"]
image_ids = [x for x, t in zip(ids, types) if t == 1]

print(f"arrow_dir: {arrow_dir}")
print(f"first_shard: {shards[0]}")
print(f"first_shard_rows: {len(ds)}")
print(f"row0_len: {len(ids)}")
print(f"row0_type_values: {sorted(set(types))}")
if image_ids:
    print(f"row0_image_tokens: {len(image_ids)}")
    print(f"row0_image_token_range: [{min(image_ids)}, {max(image_ids)}]")
    assert min(image_ids) >= 0 and max(image_ids) < 262144
print("quick_check: OK")
PY

uv run python - <<PY
import json
from pathlib import Path

metadata = {
    "docs_jsonl": "$DOCS_JSONL",
    "image_dir": "$IMAGE_DIR",
    "config": "$CONFIG",
    "output_root": "$OUTPUT_ROOT",
    "image_token_dir": "$IMAGE_TOKEN_DIR",
    "arrow_dir": "$ARROW_DIR",
    "devices": "$DEVICES",
    "num_encoder_shards": $NUM_SHARDS,
    "shard_size": int("$SHARD_SIZE"),
    "max_images": int("$MAX_IMAGES"),
    "max_docs": int("$MAX_DOCS"),
    "batch_size": int("$BATCH_SIZE"),
    "encode_workers": int("$ENCODE_WORKERS"),
    "prefetch_factor": int("$PREFETCH_FACTOR"),
    "arrow_workers": int("$ARROW_WORKERS"),
    "format": {
        "image_token_format": "raw_open_magvit2_codebook_ids_no_offset",
        "token_types": {"0": "text", "1": "image", "2": "special", "3": "padding"},
        "compatible_lm_heads": ["unified_head", "separate_text_image_heads"],
        "unified_head_image_offset_applied_at": "model forward/loss, not preprocessing",
    },
}
path = Path("$OUTPUT_ROOT") / "preprocess_metadata.json"
path.write_text(json.dumps(metadata, indent=2) + "\n")
print(f"metadata: {path}")
PY

echo
echo "Done."
echo "Use this Arrow directory for training:"
echo "  $ARROW_DIR"
