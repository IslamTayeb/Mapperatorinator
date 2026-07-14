from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

from utils.run_aot_full_song_reciprocal import evaluate_gate


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "dcc" / "verify_aot_native_extension_full_song_reciprocal.sbatch"
DRIVER = ROOT / "utils" / "run_aot_full_song_reciprocal.py"
RUNNER = ROOT / "utils" / "run_k4_approximate_weight_only.py"


def _analysis(*, warm_delta: float = -0.1) -> dict:
    return {
        "metrics": {
            "complete_request_wall_seconds": {
                "candidate_minus_baseline": warm_delta,
            },
            "cold_process_outer_wall_seconds": {
                "candidate_minus_baseline": -1.8,
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
    def test_wrapper_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)

    def test_wrapper_is_exact_gpu_reciprocal_and_validates_external_artifacts(
        self,
    ) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=gpu-common", source)
        self.assertIn("#SBATCH --gres=gpu:2080:1", source)
        self.assertIn("AOT_NATIVE_EXTENSION_MANIFEST", source)
        self.assertIn("AOT_NATIVE_EXTENSION_MANIFEST_SHA256", source)
        self.assertIn("AOT_NATIVE_EXTENSION_CACHE_ROOT", source)
        self.assertIn('"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"', source)
        self.assertIn('rev-parse "$REMOTE_REF"', source)
        self.assertIn("run_aot_full_song_reciprocal.py", source)
        self.assertIn("--minimum-cold-saving-seconds 0.5", source)
        self.assertIn("exact_parity_pass", source)
        self.assertIn("cold_saving_pass", source)
        self.assertIn("warm_no_regression_pass", source)
        self.assertIn("AOT_NATIVE_EXTENSION_FULL_SONG_PASS", source)

    def test_driver_uses_fresh_reciprocal_processes_and_exact_analysis(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn('"cached_first"', source)
        self.assertIn('"direct_first"', source)
        self.assertIn('"direct_second"', source)
        self.assertIn('"cached_second"', source)
        self.assertIn("subprocess.run(", source)
        self.assertIn('env[MANIFEST_ENV] = str(args.manifest.resolve())', source)
        self.assertIn('env.pop(MANIFEST_ENV, None)', source)
        self.assertIn('mode="exact-fp32"', source)
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

    def test_combined_runner_emits_extension_loader_evidence(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--output-extension-json", source)
        self.assertIn("loaded_extension_records", source)
        self.assertIn("K4 mixed-weight run loaded no native extensions", source)

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

        regressed = evaluate_gate(
            _analysis(warm_delta=0.001),
            walls,
            minimum_cold_saving_seconds=0.5,
        )
        self.assertFalse(regressed["warm_no_regression_pass"])
        self.assertFalse(regressed["pass"])


if __name__ == "__main__":
    unittest.main()
