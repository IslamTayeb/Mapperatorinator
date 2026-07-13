from types import SimpleNamespace
import unittest

import torch

from utils.profile_native_prefix_dtype_scout import (
    ALL_BUCKETS,
    BUCKET_COUNTS,
    REQUIRED_CHECKS,
    _bucket_entry,
    _load_args,
    convert_static_inputs_dtype,
    select_buckets,
    validate_accepted_graph_cache,
)


def _graph_cache():
    return {
        (prefix,): {
            "active_prefix_length": prefix,
            "graph": object(),
            "outputs": object(),
            "static_inputs": {"cache_position": torch.tensor([1])},
            "decode_replays": count,
        }
        for prefix, count in BUCKET_COUNTS.items()
    }


def _cache(dtype):
    def layer(length):
        return SimpleNamespace(
            is_initialized=True,
            keys=torch.ones(1, 2, length, 3, dtype=dtype),
            values=torch.ones(1, 2, length, 3, dtype=dtype),
        )

    return SimpleNamespace(
        self_attention_cache=SimpleNamespace(layers=[layer(5)]),
        cross_attention_cache=SimpleNamespace(layers=[layer(4)]),
    )


class ProfileNativePrefixDtypeScoutTest(unittest.TestCase):
    def test_load_args_registers_structured_train_and_diffusion_configs(self):
        args = _load_args(
            "profile_salvalai",
            ["audio_path=/tmp/salvalai.mp3", "inference_engine=optimized"],
        )

        self.assertEqual(args.audio_path, "/tmp/salvalai.mp3")
        self.assertEqual(args.inference_engine, "optimized")
        self.assertEqual(args.train.model.name, "OliBomby/varwhisper-small")

    def test_validate_accepted_graph_cache_requires_exact_counts(self):
        entries = validate_accepted_graph_cache(_graph_cache())
        self.assertEqual(tuple(entries), ALL_BUCKETS)

        bad = _graph_cache()
        bad[(128,)]["decode_replays"] += 1
        with self.assertRaisesRegex(RuntimeError, "replay counts changed"):
            validate_accepted_graph_cache(bad)

    def test_validate_accepted_graph_cache_rejects_duplicate_prefix(self):
        bad = _graph_cache()
        bad[("duplicate",)] = dict(bad[(128,)])
        with self.assertRaisesRegex(RuntimeError, "repeats prefix 128"):
            validate_accepted_graph_cache(bad)

    def test_bucket_selection_is_exact(self):
        entries = validate_accepted_graph_cache(_graph_cache())
        self.assertEqual(tuple(select_buckets(entries, "sentinel")), (128, 576, 640))
        self.assertEqual(tuple(select_buckets(entries, "all")), ALL_BUCKETS)
        with self.assertRaisesRegex(ValueError, "sentinel.*all"):
            select_buckets(entries, "partial")

    def test_dtype_conversion_preserves_integer_and_mask_semantics(self):
        for source, target in (
            (torch.float32, torch.float16),
            (torch.float16, torch.float32),
        ):
            with self.subTest(source=source, target=target):
                cache = _cache(source)
                inputs = {
                    "hidden_states": torch.ones(1, 1, 3, dtype=source),
                    "attention_mask": torch.tensor([[[[0.0, -1e4]]]], dtype=torch.float32),
                    "cache_position": torch.tensor([2], dtype=torch.long),
                    "past_key_values": cache,
                }
                converted = convert_static_inputs_dtype(inputs, target)
                self.assertEqual(converted["hidden_states"].dtype, target)
                self.assertEqual(converted["attention_mask"].dtype, target)
                self.assertTrue(
                    torch.equal(
                        converted["attention_mask"].float(),
                        inputs["attention_mask"].float(),
                    )
                )
                self.assertEqual(converted["cache_position"].dtype, torch.long)
                self.assertEqual(cache.self_attention_cache.layers[0].keys.dtype, target)
                self.assertEqual(cache.cross_attention_cache.layers[0].values.dtype, target)

    def test_bucket_entry_emits_summarizer_schema(self):
        checks = {name: True for name in REQUIRED_CHECKS}
        entry = _bucket_entry(
            full_ms=1.0,
            setup_seconds=0.25,
            prefix_ms=0.1,
            checks=checks,
            drift={
                "layer_output_max_abs": 0.0,
                "cache_key_slot_max_abs": 0.0,
                "cache_value_slot_max_abs": 0.0,
                "logits_max_abs": 0.0,
            },
            details={"raw": True},
        )
        self.assertEqual(entry["checks"], checks)
        self.assertEqual(entry["full_model_replay_ms_per_call"], 1.0)
        self.assertEqual(entry["capture_setup_seconds"], 0.25)
        self.assertEqual(entry["prefix_replay_ms_per_layer"], 0.1)

        with self.assertRaisesRegex(RuntimeError, "missing"):
            _bucket_entry(
                full_ms=1.0,
                setup_seconds=0.0,
                prefix_ms=0.1,
                checks={},
                drift={
                    "layer_output_max_abs": 0.0,
                    "cache_key_slot_max_abs": 0.0,
                    "cache_value_slot_max_abs": 0.0,
                    "logits_max_abs": 0.0,
                },
                details={},
            )


if __name__ == "__main__":
    unittest.main()
