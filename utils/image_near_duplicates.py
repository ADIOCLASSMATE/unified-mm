"""Compact perceptual-hash screening against frozen benchmark images."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


def perceptual_hashes(data):
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image_perceptual_hashes(image)


def image_perceptual_hashes(image):
    """Reuse an already decoded, oriented RGB image during preparation."""
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    views = [image, image.crop((left, top, left + side, top + side))]
    result = set()
    for view in views:
        gray = np.asarray(view.resize((32, 32), Image.Resampling.LANCZOS).convert("L"), dtype=np.float32)
        low = dctn(gray, norm="ortho")[:8, :8].reshape(-1)
        bits = low > np.median(low[1:])
        bits[0] = False
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        result.add(value)
    return sorted(result)


def write_index(root, hashes, source_ids, paths):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    values = np.asarray(hashes, dtype=np.uint64)
    np.save(root / "hashes.npy", values, allow_pickle=False)
    np.save(root / "source_ids.npy", np.asarray(source_ids, dtype=np.uint32), allow_pickle=False)
    orders, offsets = [], []
    for shift in (0, 16, 32, 48):
        keys = ((values >> np.uint64(shift)) & np.uint64(65535)).astype(np.uint16)
        order = np.argsort(keys, kind="stable").astype(np.uint32)
        counts = np.bincount(keys, minlength=65536)
        orders.append(order)
        offsets.append(np.concatenate(([0], counts.cumsum())).astype(np.uint32))
    np.save(root / "orders.npy", np.stack(orders), allow_pickle=False)
    np.save(root / "offsets.npy", np.stack(offsets), allow_pickle=False)
    (root / "sources.json").write_text(json.dumps(paths) + "\n")
    (root / "index.json").write_text(json.dumps({"version": "phash64-full-center-v1", "images": len(paths),
        "hashes": len(values), "maximum_hamming_distance": 3,
        "scope": "Conservative near-duplicate screening, not proof against every semantic derivative."}, indent=2) + "\n")


class NearDuplicateIndex:
    def __init__(self, root):
        root = Path(root)
        self.root = root
        contract = json.loads((root / "index.json").read_text())
        if contract["version"] != "phash64-full-center-v1":
            raise ValueError("unknown perceptual hash index")
        self.hashes = np.load(root / "hashes.npy", mmap_mode="r", allow_pickle=False)
        self.source_ids = np.load(root / "source_ids.npy", mmap_mode="r", allow_pickle=False)
        self.orders = np.load(root / "orders.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(root / "offsets.npy", mmap_mode="r", allow_pickle=False)
        self.sources = json.loads((root / "sources.json").read_text())

    def lookup(self, values):
        # At Hamming distance <= 3, at least one of four 16-bit pieces is exact.
        # Candidate lookup is small; no scan over all benchmark hashes per image.
        for value in values:
            candidates = set()
            for part, shift in enumerate((0, 16, 32, 48)):
                bucket = (int(value) >> shift) & 65535
                start, end = self.offsets[part, bucket:bucket + 2]
                candidates.update(int(x) for x in self.orders[part, int(start):int(end)])
            for candidate in sorted(candidates):
                distance = (int(value) ^ int(self.hashes[candidate])).bit_count()
                if distance <= 3:
                    return {"benchmark_path": self.sources[int(self.source_ids[candidate])],
                            "hamming_distance": distance}
        return None
