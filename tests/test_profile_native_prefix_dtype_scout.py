from types import SimpleNamespace
import inspect
import unittest
from unittest.mock import call, patch

import torch

from utils.profile_native_prefix_dtype_scout import (
    ALL_BUCKETS,
    BUCKET_COUNTS,
    REQUIRED_CHECKS,
    _accepted_main_session_run,
    _bucket_entry,
    _fp32_parity_gate,
    _full_model_variant_context,
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
    def _accepted_run_args(self):
        return SimpleNamespace(
            seed=12345,
            model_path="model-path",
            train=object(),
            device="cuda",
            max_batch_size=1,
            use_server=False,
            precision="fp32",
            attn_implementation="sdpa",
            lora_path="main-mode-lora",
            gamemode=3,
            auto_select_gamemode_model=True,
            inference_engine="optimized",
        )

    def _run_with_timing_selection(self, *, separate: bool):
        import inference
        from osuT5.osuT5.inference import Processor

        args = self._accepted_run_args()
        main_binding, main_tokenizer = object(), object()
        timing_binding, timing_tokenizer = object(), object()
        main_processor = SimpleNamespace(model=object(), decode_session_state=object())

        def processor_generate(processor, *positional, **kwargs):
            return processor, positional, kwargs

        def generate(*positional, **kwargs):
            self.assertIs(kwargs["model"], main_binding)
            self.assertIs(kwargs["tokenizer"], main_tokenizer)
            self.assertIs(
                kwargs["timing_model"], timing_binding if separate else None
            )
            self.assertIs(
                kwargs["timing_tokenizer"], timing_tokenizer if separate else None
            )
            Processor.generate(main_processor, profile_label="main_generation")
            return "generated", "/tmp/result.osu"

        loader_results = [(main_binding, main_tokenizer)]
        if separate:
            loader_results.append((timing_binding, timing_tokenizer))
        with (
            patch.object(inference, "compile_args"),
            patch.object(inference, "setup_inference_environment"),
            patch.object(
                inference,
                "load_model_with_engine",
                side_effect=loader_results,
            ) as loader,
            patch.object(
                inference,
                "should_load_separate_timing_model",
                return_value=separate,
            ),
            patch.object(inference, "get_config", return_value=("generation", "beatmap")),
            patch.object(inference, "generate", side_effect=generate),
            patch.object(Processor, "generate", processor_generate),
        ):
            result = _accepted_main_session_run(args, output_path=SimpleNamespace())
        return args, loader.call_args_list, result

    def test_accepted_run_loads_distinct_base_timing_binding_when_required(self):
        args, loader_calls, result = self._run_with_timing_selection(separate=True)

        self.assertEqual(len(loader_calls), 2)
        self.assertEqual(
            loader_calls[1],
            call(
                "model-path",
                args.train,
                "cuda",
                max_batch_size=1,
                use_server=False,
                precision="fp32",
                attn_implementation="sdpa",
                gamemode=3,
                auto_select_gamemode_model=False,
                inference_engine="optimized",
            ),
        )
        self.assertIsNotNone(result["timing_model"])
        self.assertIsNotNone(result["timing_tokenizer"])

    def test_accepted_run_leaves_timing_pair_unset_when_shared_model_is_valid(self):
        _, loader_calls, result = self._run_with_timing_selection(separate=False)

        self.assertEqual(len(loader_calls), 1)
        self.assertIsNone(result["timing_model"])
        self.assertIsNone(result["timing_tokenizer"])

    def test_accepted_run_rejects_aliased_separate_timing_binding(self):
        import inference

        args = self._accepted_run_args()
        binding = object()
        with (
            patch.object(inference, "compile_args"),
            patch.object(inference, "setup_inference_environment"),
            patch.object(
                inference,
                "load_model_with_engine",
                side_effect=[(binding, object()), (binding, object())],
            ),
            patch.object(
                inference,
                "should_load_separate_timing_model",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "aliased the main model"):
                _accepted_main_session_run(args, output_path=SimpleNamespace())

    def test_shared_specialized_context_does_not_replace_decoder_forward(self):
        source = inspect.getsource(_full_model_variant_context)

        self.assertNotIn("layer.forward =", source)
        self.assertIn("specialized_prefix_attention_context", source)

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

    def test_fp32_parity_gate_requires_exact_dispatch_and_one_percent(self):
        def entry(ms, *, drift=0.0, kernel_source):
            return {
                "full_model_replay_ms_per_call": ms,
                "checks": {name: True for name in REQUIRED_CHECKS},
                "drift": {
                    "layer_output_max_abs": drift,
                    "cache_key_slot_max_abs": 0.0,
                    "cache_value_slot_max_abs": 0.0,
                    "logits_max_abs": 0.0,
                },
                "details": {
                    "decode_replays": 100,
                    "dispatch": {
                        "original_decoder_layer": True,
                        "q1_bmm_cross_attention": True,
                        "native_q1_self_attention": True,
                        "native_q1_rope_cache_self_attention": True,
                        "kernel_source": kernel_source,
                    },
                },
            }

        variants = {
            "fp32_accepted": {
                "buckets": {
                    "128": entry(1.0, kernel_source="accepted_cached_graph"),
                }
            },
            "fp32_shared_specialized": {
                "buckets": {
                    "128": entry(
                        1.01,
                        kernel_source="recaptured_production_dispatch",
                    ),
                }
            },
        }
        report = _fp32_parity_gate(variants)
        self.assertTrue(report["pass"])
        self.assertAlmostEqual(report["replay_regression_pct"], 1.0)

        variants["fp32_shared_specialized"]["buckets"]["128"]["drift"][
            "layer_output_max_abs"
        ] = 1e-7
        with self.assertRaisesRegex(RuntimeError, "FP32_PARITY_FAILED"):
            _fp32_parity_gate(variants)


if __name__ == "__main__":
    unittest.main()
