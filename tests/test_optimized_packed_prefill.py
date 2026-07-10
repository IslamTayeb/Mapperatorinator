from types import SimpleNamespace

import pytest
import torch

from osuT5.osuT5.inference.optimized.batch.packed_prefill import (
    pack_b1_values,
    pack_static_cache_rows,
)


class _Layer:
    def __init__(self, keys: torch.Tensor | None = None, values: torch.Tensor | None = None):
        self.is_initialized = keys is not None
        if keys is not None and values is not None:
            self.keys = keys
            self.values = values

    def lazy_initialization(self, shape_probe: torch.Tensor) -> None:
        batch, heads, _, width = shape_probe.shape
        self.keys = torch.zeros(batch, heads, 4, width, dtype=shape_probe.dtype)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True


def _cache(row_value: float, *, updated: bool = True):
    self_keys = torch.full((1, 2, 4, 3), row_value)
    cross_keys = torch.full((1, 2, 4, 3), row_value + 10)
    return SimpleNamespace(
        cfg_scale=1.0,
        self_attention_cache=SimpleNamespace(
            layers=[_Layer(self_keys, self_keys + 1)],
        ),
        cross_attention_cache=SimpleNamespace(
            layers=[_Layer(cross_keys, cross_keys + 1)],
        ),
        is_updated={0: updated},
    )


def _empty_merged_cache(batch_size: int):
    del batch_size
    return SimpleNamespace(
        cfg_scale=1.0,
        self_attention_cache=SimpleNamespace(layers=[_Layer()]),
        cross_attention_cache=SimpleNamespace(layers=[_Layer()]),
        is_updated={0: False},
    )


def test_cache_rows_and_cross_update_flags_pack_into_stable_slots():
    references = [_cache(float(row)) for row in range(3)]
    merged = _empty_merged_cache(3)

    report = pack_static_cache_rows(references, merged)

    assert report["pass"] is True
    assert report["stable_slot_rows"] == [0, 1, 2]
    assert merged.is_updated == {0: True}
    for row, reference in enumerate(references):
        assert torch.equal(
            merged.self_attention_cache.layers[0].keys[row:row + 1],
            reference.self_attention_cache.layers[0].keys,
        )
        assert torch.equal(
            merged.cross_attention_cache.layers[0].values[row:row + 1],
            reference.cross_attention_cache.layers[0].values,
        )
        assert (
            merged.self_attention_cache.layers[0].keys.untyped_storage().data_ptr()
            != reference.self_attention_cache.layers[0].keys.untyped_storage().data_ptr()
        )


def test_cache_packer_rejects_flags_that_cannot_be_represented_per_row():
    references = [_cache(1.0, updated=True), _cache(2.0, updated=False)]

    with pytest.raises(ValueError, match="is_updated flags differ"):
        pack_static_cache_rows(references, _empty_merged_cache(2))


def test_request_values_pack_tensors_and_reject_mixed_non_tensor_state():
    packed = pack_b1_values(
        [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])],
        name="tokens",
    )

    assert packed.tolist() == [[1, 2], [3, 4]]
    assert pack_b1_values(["map", "map"], name="context") == "map"
    with pytest.raises(ValueError, match="must match across rows"):
        pack_b1_values(["map", "timing"], name="context")
