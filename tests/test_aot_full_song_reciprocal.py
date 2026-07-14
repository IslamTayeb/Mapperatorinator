from __future__ import annotations

from pathlib import Path
import subprocess
import unittest
from unittest import mock

from utils import run_k4_shared_rope_k1_remainder_int8_mlp_weight_only as retained
from utils.run_aot_full_song_reciprocal import (
    _validate_extension_evidence,
    _validate_retained_initialization,
    evaluate_gate,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "dcc" / "verify_aot_native_extension_full_song_reciprocal.sbatch"
DRIVER = ROOT / "utils" / "run_aot_full_song_reciprocal.py"
BUILDER = ROOT / "utils" / "build_native_extension_manifest.py"
LOADER_BENCHMARK = ROOT / "utils" / "benchmark_native_extension_loading.py"
RUNNER = (
    ROOT
    / "utils"
    / "run_k4_shared_rope_k1_remainder_int8_mlp_weight_only.py"
)


def _analysis(*, warm_delta: float = -0.1) -> dict:
    return {
        "metrics": {
            "complete_request_wall_seconds": {
                "candidate_minus_baseline": warm_delta,
            },
            "cold_process_outer_wall_seconds": {
                "candidate_minus_baseline": -1.8,
                "run_values": {
                    "cached_first": 44.5,
                    "direct_first": 42.7,
                    "direct_second": 42.9,
                    "cached_second": 44.7,
                },
            },
        },
        "parity": {
            "cross_candidate_exact": True,
            "required_exact_labels_pass": True,
            "required_exact_dispatch_labels_pass": True,
            "output_divergence": {"final_map_equal": True},
        },
    }


class AotFullSongReciprocalTest(unittest.TestCase):
    def test_jit_control_allows_rebuilt_binary_hash_but_direct_requires_package_hash(
        self,
    ) -> None:
        manifest = {
            "extension": {
                "source_sha256": "source",
                "library_sha256": "packaged-library",
                "functions": ["forward"],
            }
        }
        jit = {
            "extension": {
                **manifest["extension"],
                "library_sha256": "fresh-jit-library",
                "mode": "load_inline",
                "load_seconds": 1.0,
            }
        }
        direct = {
            "extension": {
                **manifest["extension"],
                "mode": "direct",
                "load_seconds": 0.01,
            }
        }

        self.assertEqual(
            _validate_extension_evidence(jit, manifest, mode="cached"),
            1.0,
        )
        self.assertEqual(
            _validate_extension_evidence(direct, manifest, mode="direct"),
            0.01,
        )
        direct["extension"]["library_sha256"] = "wrong-library"
        with self.assertRaisesRegex(RuntimeError, "differs in library_sha256"):
            _validate_extension_evidence(direct, manifest, mode="direct")

    def test_wrapper_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    def test_wrapper_is_exact_gpu_reciprocal_and_validates_external_artifacts(
        self,
    ) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=gpu-common", source)
        self.assertIn("#SBATCH --gres=gpu:2080:1", source)
        self.assertIn("#SBATCH --time=00:30:00", source)
        self.assertIn("AOT_NATIVE_EXTENSION_MANIFEST", source)
        self.assertIn("AOT_NATIVE_EXTENSION_MANIFEST_SHA256", source)
        self.assertIn("AOT_NATIVE_EXTENSION_CACHE_ROOT", source)
        self.assertIn("AOT_NATIVE_EXTENSION_BUILD_RESULT", source)
        self.assertIn("AOT_NATIVE_EXTENSION_BUILD_RESULT_SHA256", source)
        self.assertIn('"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"', source)
        self.assertIn('rev-parse "$REMOTE_REF"', source)
        self.assertIn("run_aot_full_song_reciprocal.py", source)
        self.assertIn("--minimum-cold-saving-seconds 0.5", source)
        self.assertIn("exact_parity_pass", source)
        self.assertIn("cold_saving_pass", source)
        self.assertIn("warm_no_regression_pass", source)
        self.assertIn("AOT_NATIVE_EXTENSION_FULL_SONG_PASS", source)
        self.assertIn("k1-remainder-int8-mlp-v1", source)
        self.assertIn("fixed_8294_main_seconds", source)
        self.assertIn("complete_request_wall_seconds", source)
        self.assertIn("accepted_fp32_exactness_claim", source)
        self.assertNotIn("build_native_extension_manifest.py", source)

    def test_driver_uses_fresh_reciprocal_processes_and_exact_analysis(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn('"cached_first"', source)
        self.assertIn('"direct_first"', source)
        self.assertIn('"direct_second"', source)
        self.assertIn('"cached_second"', source)
        self.assertIn("subprocess.run(", source)
        self.assertIn('env[MANIFEST_ENV] = str(local_manifest)', source)
        self.assertIn('env.pop(MANIFEST_ENV, None)', source)
        self.assertIn('mode="relaxed"', source)
        self.assertIn(
            'required_exact_labels=("timing_context", "main_generation")', source
        )
        self.assertIn(
            'required_exact_dispatch_labels=("timing_context", "main_generation")',
            source,
        )
        self.assertIn("validate_packaged_manifest", source)
        self.assertIn("validate_weight", source)
        self.assertIn("validate_k4", source)
        self.assertIn("validate_k1", source)
        self.assertIn("validate_int8_mlp", source)
        self.assertIn("_validate_retained_initialization", source)
        self.assertIn("fixed_8294_main_seconds", source)
        self.assertIn("process_wall_reconciliation", source)
        self.assertIn("native_extension_storage_bytes", source)
        self.assertIn("peak_cuda_memory_allocated_mb", source)

    def test_retained_k1_int8_runner_emits_extension_loader_evidence(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--output-extension-json", source)
        self.assertIn("output_extension_json", source)
        self.assertIn("composition_version=COMPOSITION_VERSION", source)

    def test_manifest_builder_and_loader_benchmark_include_int8_extension(self) -> None:
        for path in (BUILDER, LOADER_BENCHMARK):
            source = path.read_text(encoding="utf-8")
            self.assertIn("mapperatorinator_int8_mlp_scout_v1", source)
            self.assertIn("preload_int8_mlp_extension", source)

    def test_retained_runner_threads_extension_evidence_without_changing_topology(
        self,
    ) -> None:
        with mock.patch.object(retained, "run_combined") as run_combined:
            retained.run(
                "profile_salvalai",
                ["seed=12345"],
                Path("init.json"),
                Path("extensions.json"),
            )

        run_combined.assert_called_once_with(
            "profile_salvalai",
            ["seed=12345"],
            Path("init.json"),
            Path("extensions.json"),
            graph_remainders=True,
            weight_runner=retained.run_int8_weight_only,
            composition_version=retained.COMPOSITION_VERSION,
        )

    def test_retained_initialization_requires_k1_int8_identity_and_drift_metadata(
        self,
    ) -> None:
        shared = {
            "version": "shared-decoder-rope-v1",
            "scope": "main-model-only",
            "incremental_exactness_claim": True,
            "original_decoder_forward_required": True,
            "stats": {
                "forwards": 1,
                "computes": 1,
                "reuses": 11,
                "expected_computes": 1,
                "expected_reuses": 11,
            },
        }
        overlay = {
            "version": "per-row-symmetric-int8-mlp-v1",
            "result_class": "documented-drift",
            "exactness_claim": False,
            "scope": "main-model-decoder-mlp-only",
            "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
            "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
            "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
            "fp32_activations_norm_bias_reductions_residual_outputs": True,
            "quantization": "symmetric-per-output-row",
            "dispatch_counter": "int8_weight_mlp_tail",
            "extension_init_seconds": 0.1,
            "weight_pack_seconds": 0.2,
            "packed_weight_bytes": 3,
        }
        payload = {
            "combined_runtime": retained.COMPOSITION_VERSION,
            "result_class": "documented-drift",
            "exactness_claim": False,
            "shared_rope": shared,
            "int8_mlp_overlay": overlay,
        }

        evidence = _validate_retained_initialization(payload, role="direct_first")

        self.assertEqual(evidence["combined_runtime"], retained.COMPOSITION_VERSION)
        self.assertFalse(evidence["int8_mlp_overlay"]["exactness_claim"])

    def test_gate_requires_half_second_cold_saving_exactness_and_no_warm_regression(
        self,
    ) -> None:
        walls = {
            "cached_first": 45.0,
            "direct_first": 43.0,
            "direct_second": 43.2,
            "cached_second": 45.2,
        }
        passed = evaluate_gate(
            _analysis(),
            walls,
            minimum_cold_saving_seconds=0.5,
        )
        self.assertTrue(passed["pass"])
        self.assertAlmostEqual(passed["complete_cold_wall_saving_seconds"], 2.0)
        self.assertAlmostEqual(
            passed["process_wall_reconciliation"]["cached_first"][
                "python_startup_and_unprofiled_seconds"
            ],
            0.5,
        )

        regressed = evaluate_gate(
            _analysis(warm_delta=0.001),
            walls,
            minimum_cold_saving_seconds=0.5,
        )
        self.assertFalse(regressed["warm_no_regression_pass"])
        self.assertFalse(regressed["pass"])

    def test_gate_records_build_load_and_storage_evidence(self) -> None:
        walls = {
            "cached_first": 45.0,
            "direct_first": 43.0,
            "direct_second": 43.2,
            "cached_second": 45.2,
        }
        loads = {
            "cached_first": 2.7,
            "direct_first": 0.04,
            "direct_second": 0.05,
            "cached_second": 2.8,
        }
        result = evaluate_gate(
            _analysis(),
            walls,
            minimum_cold_saving_seconds=0.5,
            build_result={"build_seconds": 267.0},
            extension_load_seconds=loads,
            storage_bytes={"prebuilt_package": 10, "cached_jit_tree": 20},
        )

        self.assertEqual(result["native_extension_build_seconds"], 267.0)
        self.assertAlmostEqual(
            result["native_extension_load_seconds"]["direct_saving"],
            2.705,
        )
        self.assertEqual(
            result["native_extension_storage_bytes"]["prebuilt_package"], 10
        )


if __name__ == "__main__":
    unittest.main()
