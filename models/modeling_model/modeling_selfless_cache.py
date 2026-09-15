"""Static backbone K/V cache with explicit physical sequence positions."""
from __future__ import annotations

import types

import torch
from transformers.cache_utils import Cache


class _SelflessStaticCacheLayer:
    def __init__(self, max_cache_len: int):
        self.max_cache_len = int(max_cache_len)
        self.is_initialized = False
        self.keys = None
        self.values = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.keys = torch.zeros(
            key_states.shape[0],
            key_states.shape[1],
            self.max_cache_len,
            key_states.shape[-1],
            device=key_states.device,
            dtype=key_states.dtype,
        )
        self.values = torch.zeros(
            value_states.shape[0],
            value_states.shape[1],
            self.max_cache_len,
            value_states.shape[-1],
            device=value_states.device,
            dtype=value_states.dtype,
        )
        self.is_initialized = True


class SelflessStaticCache(Cache):
    """Static K/V cache that writes batch-specific original sequence slots."""

    def __init__(self, config, max_cache_len: int):
        super().__init__(
            layers=[
                _SelflessStaticCacheLayer(max_cache_len)
                for _ in range(int(config.num_hidden_layers))
            ]
        )
        self._max_cache_len = int(max_cache_len)
        self.visible_length = self._max_cache_len

    def read(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        return (layer.keys[:, :, :self.visible_length],
                layer.values[:, :, :self.visible_length])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return 0

    def get_max_cache_shape(self) -> int:
        return self._max_cache_len

    def fork(self) -> "SelflessStaticCache":
        """Clone an initialized cache for an independent branch."""

        cloned = SelflessStaticCache(
            config=types.SimpleNamespace(num_hidden_layers=len(self.layers)),
            max_cache_len=self._max_cache_len,
        )
        cloned.visible_length = self.visible_length
        for source, target in zip(self.layers, cloned.layers, strict=True):
            if not source.is_initialized:
                continue
            target.keys = source.keys.clone()
            target.values = source.values.clone()
            target.is_initialized = True
        return cloned

    def repeat_batch_(self, repeats: int) -> "SelflessStaticCache":
        """Repeat cache rows in branch-major order without another model pass."""

        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        if repeats == 1:
            return self
        for layer in self.layers:
            if not layer.is_initialized:
                continue
            repeat_shape = (repeats, *([1] * (layer.keys.ndim - 1)))
            layer.keys = layer.keys.repeat(repeat_shape)
            layer.values = layer.values.repeat(repeat_shape)
        return self

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = args[0] if args else kwargs
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise ValueError(
                "SelflessStaticCache requires explicit cache_position."
            )
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)
        cache_position = cache_position.to(
            device=key_states.device,
            dtype=torch.long,
        )
        if cache_position.ndim == 1:
            cache_position = cache_position.unsqueeze(0).expand(
                key_states.shape[0], -1
            )
        expected_shape = (key_states.shape[0], key_states.shape[-2])
        if tuple(cache_position.shape) != expected_shape:
            raise ValueError(
                "cache_position must align with [batch, query_length]: "
                f"{tuple(cache_position.shape)} != {expected_shape}"
            )
        cache_write_prefix = cache_kwargs.get("cache_write_prefix")
        if cache_write_prefix is not None:
            cache_write_prefix = int(cache_write_prefix)
            if not 0 < cache_write_prefix <= key_states.shape[-2]:
                raise ValueError(
                    "cache_write_prefix must be in [1, query_length], got "
                    f"{cache_write_prefix} for {key_states.shape[-2]} queries"
                )
            cache_position = cache_position[:, :cache_write_prefix]
            key_states = key_states[:, :, :cache_write_prefix]
            value_states = value_states[:, :, :cache_write_prefix]
            expected_shape = (
                key_states.shape[0],
                key_states.shape[-2],
            )
        key_indices = cache_position[:, None, :, None].expand(
            -1,
            key_states.shape[1],
            -1,
            key_states.shape[-1],
        )
        value_indices = cache_position[:, None, :, None].expand(
            -1,
            value_states.shape[1],
            -1,
            value_states.shape[-1],
        )
        cache_write_mask = cache_kwargs.get("cache_write_mask")
        if cache_write_mask is not None:
            cache_write_mask = cache_write_mask.to(
                device=key_states.device,
                dtype=torch.bool,
            )
            if cache_write_prefix is not None:
                cache_write_mask = cache_write_mask[:, :cache_write_prefix]
            if tuple(cache_write_mask.shape) != expected_shape:
                raise ValueError(
                    "cache_write_mask must align with cache_position: "
                    f"{tuple(cache_write_mask.shape)} != {expected_shape}"
                )
            key_states = torch.where(
                cache_write_mask[:, None, :, None],
                key_states,
                layer.keys.gather(2, key_indices),
            )
            value_states = torch.where(
                cache_write_mask[:, None, :, None],
                value_states,
                layer.values.gather(2, value_indices),
            )
        layer.keys.scatter_(2, key_indices, key_states)
        layer.values.scatter_(2, value_indices, value_states)
        return self.read(layer_idx)
