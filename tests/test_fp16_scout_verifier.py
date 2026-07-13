import unittest
from types import SimpleNamespace

import torch

from osuT5.osuT5.inference.optimized.scout.verifier import (
    snapshot_cache_state,
    tensor_difference,
    verify_candidate_cache_behavior,
)


def _layer(values: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        is_initialized=True,
        keys=values.clone(),
        values=(values + 100).clone(),
    )


def _cache(dtype: torch.dtype = torch.float16) -> SimpleNamespace:
    self_values = torch.arange(1 * 2 * 5 * 3, dtype=dtype).reshape(1, 2, 5, 3)
    cross_values = torch.arange(1 * 2 * 4 * 3, dtype=dtype).reshape(1, 2, 4, 3)
    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=[_layer(self_values)]),
        cross_attention_cache=SimpleNamespace(layers=[_layer(cross_values)]),
    )


def _assert_restored(cache: SimpleNamespace, snapshot) -> None:
    assert torch.equal(
        cache.self_attention_cache.layers[0].keys,
        snapshot.tensors["self_keys"].value,
    )
    assert torch.equal(
        cache.self_attention_cache.layers[0].values,
        snapshot.tensors["self_values"].value,
    )
    assert torch.equal(
        cache.cross_attention_cache.layers[0].keys,
        snapshot.tensors["cross_keys"].value,
    )
    assert torch.equal(
        cache.cross_attention_cache.layers[0].values,
        snapshot.tensors["cross_values"].value,
    )


class ScoutVerifierTest(unittest.TestCase):
    def test_candidate_cache_behavior_passes_and_restores_state(self) -> None:
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                cache = _cache(dtype)
                snapshot = snapshot_cache_state(cache, layer_idx=0, cache_position=2)
                self_layer = cache.self_attention_cache.layers[0]

                def candidate() -> torch.Tensor:
                    self_layer.keys[..., 2:3, :].fill_(7)
                    self_layer.values[..., 2:3, :].fill_(9)
                    return torch.tensor([1.5, 2.5], dtype=dtype)

                report = verify_candidate_cache_behavior(
                    cache,
                    layer_idx=0,
                    cache_position=torch.tensor([2]),
                    candidate=candidate,
                    repeats=3,
                )

                self.assertTrue(report["pass"])
                self.assertTrue(report["candidate_repeat_output_exact"])
                self.assertTrue(report["candidate_repeat_self_key_slot_exact"])
                self.assertTrue(report["candidate_repeat_self_value_slot_exact"])
                self.assertTrue(
                    all(
                        repeat["current_slot_written"]
                        for repeat in report["repeats"]
                    )
                )
                self.assertTrue(
                    all(
                        repeat["self_keys"]["future_unchanged"]
                        for repeat in report["repeats"]
                    )
                )
                self.assertTrue(
                    all(
                        all(repeat["ownership_stable"].values())
                        for repeat in report["repeats"]
                    )
                )
                _assert_restored(cache, snapshot)

    def test_candidate_cache_behavior_rejects_out_of_scope_writes(self) -> None:
        for dtype in (torch.float16, torch.float32):
            for mutation in ("future", "cross"):
                with self.subTest(dtype=dtype, mutation=mutation):
                    cache = _cache(dtype)
                    snapshot = snapshot_cache_state(
                        cache, layer_idx=0, cache_position=2
                    )
                    self_layer = cache.self_attention_cache.layers[0]
                    cross_layer = cache.cross_attention_cache.layers[0]

                    def candidate() -> torch.Tensor:
                        self_layer.keys[..., 2:3, :].fill_(7)
                        self_layer.values[..., 2:3, :].fill_(9)
                        if mutation == "future":
                            self_layer.keys[..., 4:5, :].add_(1)
                        else:
                            cross_layer.values.add_(1)
                        return torch.ones(1, dtype=dtype)

                    report = verify_candidate_cache_behavior(
                        cache,
                        layer_idx=0,
                        cache_position=2,
                        candidate=candidate,
                    )

                    self.assertFalse(report["pass"])
                    if mutation == "future":
                        self.assertFalse(
                            report["repeats"][0]["self_keys"]["future_unchanged"]
                        )
                    else:
                        self.assertFalse(
                            report["repeats"][0]["cross_values_unchanged"]
                        )
                    _assert_restored(cache, snapshot)

    def test_candidate_cache_behavior_rejects_nonfinite_output(self) -> None:
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                cache = _cache(dtype)
                self_layer = cache.self_attention_cache.layers[0]

                def candidate() -> torch.Tensor:
                    self_layer.keys[..., 2:3, :].fill_(7)
                    self_layer.values[..., 2:3, :].fill_(9)
                    return torch.tensor([float("inf")], dtype=dtype)

                report = verify_candidate_cache_behavior(
                    cache,
                    layer_idx=0,
                    cache_position=2,
                    candidate=candidate,
                )

                self.assertFalse(report["pass"])
                self.assertFalse(report["repeats"][0]["finite"]["output"])

    def test_candidate_cache_behavior_rejects_repeat_drift(self) -> None:
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                cache = _cache(dtype)
                self_layer = cache.self_attention_cache.layers[0]
                call_count = 0

                def candidate() -> torch.Tensor:
                    nonlocal call_count
                    call_count += 1
                    self_layer.keys[..., 2:3, :].fill_(call_count)
                    self_layer.values[..., 2:3, :].fill_(9)
                    return torch.tensor([call_count], dtype=dtype)

                report = verify_candidate_cache_behavior(
                    cache,
                    layer_idx=0,
                    cache_position=2,
                    candidate=candidate,
                )

                self.assertFalse(report["pass"])
                self.assertFalse(report["candidate_repeat_output_exact"])
                self.assertFalse(report["candidate_repeat_self_key_slot_exact"])

    def test_candidate_cache_behavior_fails_loudly_on_storage_replacement(self) -> None:
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                cache = _cache(dtype)
                self_layer = cache.self_attention_cache.layers[0]

                def candidate() -> torch.Tensor:
                    self_layer.keys = self_layer.keys.clone()
                    self_layer.keys[..., 2:3, :].fill_(7)
                    self_layer.values[..., 2:3, :].fill_(9)
                    return torch.ones(1, dtype=dtype)

                with self.assertRaisesRegex(
                    RuntimeError, "replaced or resized cache storage"
                ):
                    verify_candidate_cache_behavior(
                        cache,
                        layer_idx=0,
                        cache_position=2,
                        candidate=candidate,
                    )

    def test_tensor_difference_is_dtype_generic(self) -> None:
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                reference = torch.tensor([1.0, 2.0], dtype=dtype)
                candidate = torch.tensor([1.0, 2.5], dtype=dtype)

                difference = tensor_difference(reference, candidate)

                self.assertEqual(
                    difference,
                    {
                        "shape_matches": True,
                        "dtype_matches": True,
                        "reference_finite": True,
                        "candidate_finite": True,
                        "exact": False,
                        "max_abs": 0.5,
                    },
                )


if __name__ == "__main__":
    unittest.main()
