"""Unit tests for §47 O(1) keep-accepted KV rollback helpers."""

import torch
from transformers import DynamicCache

from osuT5.osuT5.inference.cache_utils import MapperatorinatorCache
from osuT5.osuT5.inference.turbo.kv_rollback import (
    align_kwargs_after_rewind,
    reset_self_cache,
    rewind_self_cache,
)
from osuT5.osuT5.inference.turbo.speculate import crop_self_cache


def _dynamic_cache_with_len(length: int) -> MapperatorinatorCache:
    cache = MapperatorinatorCache(DynamicCache(), DynamicCache(), 1.0)
    # Simulate written KV via crop-compatible tensors.
    keys = torch.randn(1, 2, length, 4)
    values = torch.randn(1, 2, length, 4)
    cache.self_attention_cache.update(keys, values, 0, {})
    return cache


def test_rewind_dynamic_crops_to_new_len():
    cache = _dynamic_cache_with_len(8)
    rewind_self_cache(cache, 5, occupied_end=8)
    assert cache.self_attention_cache.get_seq_length() == 5


def test_align_kwargs_after_rewind_drops_cache_position():
    mk = {"cache_position": torch.arange(4)}
    align_kwargs_after_rewind(mk, length=6, device=torch.device("cpu"))
    assert mk["decoder_attention_mask"].shape == (1, 6)
    assert "cache_position" not in mk


def test_crop_self_cache_still_full_wipe_from_length():
    cache = _dynamic_cache_with_len(8)
    crop_self_cache(cache, 3)
    assert cache.self_attention_cache.get_seq_length() == 3


def test_reset_self_cache_clears_dynamic():
    cache = _dynamic_cache_with_len(4)
    reset_self_cache(cache)
    assert cache.self_attention_cache.get_seq_length() == 0
