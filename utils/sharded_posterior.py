"""Lazy mmap access to KL16 shards without materializing a merged tensor."""

from collections import OrderedDict
import json
from pathlib import Path

import torch


SCHEMA = "kl16_posterior_shard_index_v1"


class ShardedPosterior:
    def __init__(self, paths, shard_rows, token_shape, img_ids, limit=None, storage_img_ids=None):
        self.paths = paths
        self.shard_rows = shard_rows
        self.img_ids = img_ids
        self.storage_img_ids = img_ids if storage_img_ids is None else storage_img_ids
        self.limit = len(shard_rows) if limit is None else limit
        self.shape = (self.limit, *token_shape)
        self.ndim = 3
        self.dtype = torch.float16
        self._opened = OrderedDict()

    def is_floating_point(self):
        return True

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.limit)
            if start != 0 or step != 1:
                raise ValueError("sharded posterior only supports prefix slices")
            return type(self)(self.paths, self.shard_rows, self.shape[1:], self.img_ids, stop,
                              self.storage_img_ids)
        index = int(index)
        if index < 0 or index >= self.limit:
            raise IndexError(index)
        shard, row = map(int, self.shard_rows[index])
        if shard not in self._opened:
            payload = torch.load(self.paths[shard], mmap=True, map_location="cpu", weights_only=True)
            stats = payload["posterior_stats"]
            if tuple(stats.shape[1:]) != self.shape[1:] or stats.dtype != self.dtype:
                raise ValueError(f"shard shape/dtype changed: {self.paths[shard]}")
            self._opened[shard] = (stats, payload["img_ids"])
            if len(self._opened) > 32:
                self._opened.popitem(last=False)
        self._opened.move_to_end(shard)
        stats, ids = self._opened[shard]
        if int(ids[row]) != int(self.storage_img_ids[index]):
            raise ValueError(f"shard image identity changed: {self.paths[shard]} row {row}")
        return stats[row]

    def __getstate__(self):
        return {**self.__dict__, "_opened": OrderedDict()}


def load_sharded_posterior(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"invalid posterior shard index: {path}")
    def resolve(value):
        value = Path(value)
        return str(value if value.is_absolute() else path.parent / value)
    index = torch.load(resolve(manifest["row_index"]), mmap=True, map_location="cpu", weights_only=True)
    ids, rows = index["img_ids"], index["shard_rows"]
    storage_ids = index.get("storage_img_ids", ids)
    if rows.shape != (len(ids), 2) or rows.dtype != torch.int64 or ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError("invalid posterior row index")
    if storage_ids.shape != ids.shape or storage_ids.dtype != torch.int64:
        raise ValueError("invalid posterior storage image IDs")
    paths = [resolve(value) for value in manifest["shards"]]
    if len(ids) and (bool((rows < 0).any()) or int(rows[:, 0].max()) >= len(paths)):
        raise ValueError("posterior row index references an invalid shard/row")
    return {
        "posterior_stats": ShardedPosterior(paths, rows, manifest["token_shape"], ids,
                                            storage_img_ids=storage_ids),
        "img_ids": ids,
        "metadata": manifest["metadata"],
    }
