"""Unit tests for turbo speculative helpers (no GPU / model required)."""

import torch
from transformers import DynamicCache

from osuT5.osuT5.inference.cache_utils import MapperatorinatorCache
from osuT5.osuT5.inference.turbo.speculate import _align_decoder_mask, get_turbo_cache


def test_get_turbo_cache_uses_dynamic():
    cache = get_turbo_cache(cfg_scale=1.0)
    assert isinstance(cache, MapperatorinatorCache)
    assert isinstance(cache.self_attention_cache, DynamicCache)
    assert isinstance(cache.cross_attention_cache, DynamicCache)


def test_align_decoder_mask_sets_length():
    mk = {"cache_position": torch.arange(3)}
    _align_decoder_mask(mk, length=7, device=torch.device("cpu"))
    assert mk["decoder_attention_mask"].shape == (1, 7)
    assert int(mk["decoder_attention_mask"].sum().item()) == 7
    assert "cache_position" not in mk
