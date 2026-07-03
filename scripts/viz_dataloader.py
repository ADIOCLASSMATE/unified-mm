"""
Visualize OmniCorpus multimodal dataloader samples.

Usage:
    python scripts/viz_dataloader.py
    python scripts/viz_dataloader.py --config configs/selfless/omnicorpus.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from utils.utils import load_model_tokenizer
from utils.dataset_utils import get_dataloaders

# ── terminal colors ──────────────────────────────────────────
C = {
    "text":    "\033[37m",
    "image":   "\033[36m",
    "special": "\033[33m",
    "pad":     "\033[90m",
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "header":  "\033[1;35m",
    "green":   "\033[32m",
    "red":     "\033[31m",
    "warn":    "\033[1;31m",
}
TYPE_CHAR = {0: "T", 1: "I", 2: "S", 3: "·"}


def token_types_line(types, ids, start, end):
    """One line of colored token type chars from start to end."""
    parts = []
    for i in range(start, min(end, types.shape[0])):
        t = types[i].item()
        if t == 3:
            parts.append(f"{C['pad']}·{C['reset']}")
            continue
        if t == 2:
            tid = ids[i].item()
            parts.append(f"{C['special']}{tid}{C['reset']}")
        elif t == 1:
            parts.append(f"{C['image']}I{C['reset']}")
        else:
            parts.append(f"{C['text']}T{C['reset']}")
    return " ".join(parts)


def sigma_line(sigma, types, start, end):
    parts = []
    for i in range(start, min(end, types.shape[0])):
        s = sigma[i].item()
        t = types[i].item()
        if t == 1:
            parts.append(f"{C['image']}{s:>4d}{C['reset']}")
        elif t == 3:
            parts.append(f"{C['pad']}{s:>4d}{C['reset']}")
        else:
            parts.append(f"{s:>4d}")
    return " ".join(parts)


def labels_line(labels, start, end):
    parts = []
    for i in range(start, min(end, labels.shape[0])):
        l = labels[i].item()
        if l == -100:
            parts.append(f"{C['red']}-100{C['reset']}")
        else:
            parts.append(f"{l:>5d}")
    return " ".join(parts)


def detect_mode(ids, types, boi_id, eoi_id):
    """Detect task_mode from token sequence."""
    n_image = (types == 1).sum().item()
    if n_image == 0:
        return "text_only"

    boi_pos = (ids == boi_id).nonzero(as_tuple=True)[0]
    eoi_pos = (ids == eoi_id).nonzero(as_tuple=True)[0]
    if len(boi_pos) == 0 or len(eoi_pos) == 0:
        return "text_only"

    first_boi = boi_pos[0].item()
    first_eoi = eoi_pos[0].item()
    text_before = (types[:first_boi] == 0).sum().item()
    text_after = (types[first_eoi:] == 0).sum().item()

    if len(boi_pos) > 1:
        return "interleaved"
    if text_before > 0 and text_after == 0:
        return "text_to_image"
    if text_after > 0 and text_before == 0:
        return "image_to_text"
    if text_before > 0 and text_after > 0:
        return "interleaved"
    return "text_to_image"


def show_sample(ids, types, sigma, labels, title, eoi_id, seg_len=80):
    """Print a visualized sample with line-wrapped sigma/labels display."""
    L = (types != 3).sum().item()
    pad = (types == 3).sum().item()
    total = types.shape[0]
    n_text = (types == 0).sum().item()
    n_img = (types == 1).sum().item()
    n_spec = (types == 2).sum().item()
    real_sigma = sigma[types != 3]
    s_min = real_sigma.min().item() if real_sigma.numel() > 0 else -1
    s_max = real_sigma.max().item()
    n_ignore = (labels == -100).sum().item()

    # Monotonic checks
    tp = (types == 0).nonzero(as_tuple=True)[0]
    text_mono = True
    if len(tp) > 1:
        d = sigma[tp][1:] - sigma[tp][:-1]
        text_mono = (d > 0).all().item()

    ip = (types == 1).nonzero(as_tuple=True)[0]
    img_rand = True
    if len(ip) > 3:
        d = sigma[ip][1:] - sigma[ip][:-1]
        img_rand = not (d > 0).all().item()

    # EOI label check
    eoi = (ids == eoi_id) & (types != 3)
    eoi_labels_ok = (labels[eoi] == -100).all().item() if eoi.any() else "N/A"

    print(f"\n{C['header']}{'═'*70}{C['reset']}")
    print(f"{C['header']}{C['bold']}  {title}{C['reset']}")
    print(f"  L={L} pad={pad}/{total} | T:{n_text} I:{n_img} S:{n_spec} | "
          f"σ∈[{s_min},{s_max}] | -100:{n_ignore} | text_mono={text_mono} img_rand={img_rand} "
          f"EOI=-100:{eoi_labels_ok}")

    # Print in segments
    pos = 0
    seg_num = 0
    while pos < L:
        end = min(pos + seg_len, L)
        print(f"\n  {C['bold']}[{pos:>4d}-{end:>4d}]{C['reset']} {'T'*3}: {token_types_line(types, ids, pos, end)}")
        print(f"             {'σ'*3}: {sigma_line(sigma, types, pos, end)}")
        print(f"             {'L'*3}: {labels_line(labels, pos, end)}")
        pos = end
        seg_num += 1
        if seg_num >= 4:  # show first ~320 tokens, that's enough to see structure
            remaining = L - pos
            if remaining > 0:
                print(f"\n  {C['pad']}  ... ({remaining} more tokens omitted){C['reset']}")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/selfless/omnicorpus.yaml")
    parser.add_argument("--seg_len", type=int, default=80)
    args = parser.parse_args()

    print(f"{C['header']}{C['bold']}=== Loading config: {args.config} ==={C['reset']}")
    config = OmegaConf.load(args.config)

    print(f"{C['header']}{C['bold']}=== Loading model & tokenizer ==={C['reset']}")
    model, tokenizer = load_model_tokenizer(config=config, logger=None)
    boi_id = config.model.boi_token_id
    eoi_id = config.model.eoi_token_id

    print(f"{C['header']}{C['bold']}=== Building dataloaders ==={C['reset']}")
    train_loader, val_loader = get_dataloaders(config, tokenizer)

    ds = train_loader.dataset
    while hasattr(ds, "dataset") and not hasattr(ds, "_packs"):
        ds = ds.dataset
    raw_docs = len(ds.dataset) if hasattr(ds, "dataset") else len(ds)
    print(f"Raw docs: {raw_docs} -> Packed sequences: {len(ds._packs)}")
    print(f"EOS={tokenizer.eos_token_id}, BOI={boi_id}, EOI={eoi_id}")

    # ── Collect one sample per mode ──────────────────────────
    target_modes = {"interleaved"}
    collected = {}  # mode -> (ids, types, sigma, labels)
    batches = 0

    print(f"\n{C['header']}{C['bold']}=== Scanning for all modes: {target_modes} ==={C['reset']}")
    for batch in train_loader:
        batches += 1
        B = batch["input_ids"].shape[0]
        for s in range(B):
            ids = batch["input_ids"][s]
            types = batch["token_types"][s]
            sigma = batch["sigma"][s]
            labels = batch["labels"][s]
            mode = detect_mode(ids, types, boi_id, eoi_id)
            if mode not in collected:
                collected[mode] = (ids, types, sigma, labels)
                print(f"  batch {batches} sample {s}: collected '{mode}'")

        if target_modes.issubset(collected.keys()):
            print(f"  All {len(target_modes)} modes collected after {batches} batches.\n")
            break

    missing = target_modes - collected.keys()
    if missing:
        print(f"  {C['warn']}Warning: missing modes: {missing}{C['reset']}")

    # ── Display in fixed order ───────────────────────────────
    for mode in ["text_only", "text_to_image", "image_to_text", "interleaved"]:
        if mode in collected:
            ids, types, sigma, labels = collected[mode]
            show_sample(ids, types, sigma, labels, f"[{mode}]", eoi_id=eoi_id, seg_len=args.seg_len)

    # ── Quick stats ──────────────────────────────────────────
    print(f"\n{C['header']}{C['bold']}=== Determinism ==={C['reset']}")
    s1 = ds[0]["sigma"]
    s2 = ds[0]["sigma"]
    print(f"ds[0] same across calls: {(s1 == s2).all().item()}")

    print(f"\n{C['header']}{C['bold']}=== Re-pack ==={C['reset']}")
    p0 = tuple(ds._packs[0][:20])
    ds.set_epoch(1)
    p1 = tuple(ds._packs[0][:20])
    print(f"Pack[0][:20] same after set_epoch(1): {p0 == p1} (should differ)")

    print(f"\n{C['header']}{C['bold']}Done.{C['reset']}")


if __name__ == "__main__":
    main()
