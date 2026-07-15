from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from osuT5.osuT5.inference.audio_preparation import (
    audio_array_metadata,
    resolve_audio_samples,
    validate_preloaded_audio,
)
from osuT5.osuT5.inference.optimized.audio_model_overlap_scout import (
    AudioPreparationTask,
)
from utils.profile_audio_model_overlap import (
    RUN_ORDER,
    SELECTED_STACK_VERSION,
    analyze_reciprocal,
)


class _Loader:
    def __init__(self, samples):
        self.samples = samples
        self.calls: list[str] = []

    def load(self, path: str):
        self.calls.append(path)
        return self.samples


def _success_manifest(path: Path, *, result_hash: str = "osu") -> None:
    payload = {
        "status": "success",
        "audio_task_closed": True,
        "inner_request_wall_seconds": 8.0,
        "audio": {"audio_array_sha256": "audio"},
        "audio_worker": None,
        "result_sha256": result_hash,
        "profile_contract": {"seed": 12345},
        "selected_topology": {"version": SELECTED_STACK_VERSION},
        "generation_signature": {
            label: {
                "token_stream_sha256": f"{label}-tokens",
                "stopping_sha256": f"{label}-stopping",
            }
            for label in ("timing_context", "main_generation")
        },
        "profile_stage_wall_seconds": {"load_main_model": 2.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reciprocal_records(tmp_path: Path) -> list[dict]:
    walls = (10.0, 9.0, 9.0, 10.0)
    records = []
    for (run_name, mode), wall in zip(RUN_ORDER, walls, strict=True):
        manifest = tmp_path / f"{run_name}.json"
        _success_manifest(manifest)
        records.append(
            {
                "run_name": run_name,
                "mode": mode,
                "exit_code": 0,
                "cold_process_wall_seconds": wall,
                "manifest_path": str(manifest),
            }
        )
    return records


class AudioModelOverlapScoutTest(unittest.TestCase):
    def test_default_audio_resolution_uses_unchanged_loader_result(self) -> None:
        samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        loader = _Loader(samples)

        actual = resolve_audio_samples(loader, "song.mp3")

        self.assertIs(actual, samples)
        self.assertEqual(loader.calls, ["song.mp3"])

    def test_default_audio_resolution_does_not_add_validation(self) -> None:
        sentinel = object()
        loader = _Loader(sentinel)

        actual = resolve_audio_samples(loader, "song.mp3")

        self.assertIs(actual, sentinel)
        self.assertEqual(loader.calls, ["song.mp3"])

    def test_opt_in_audio_resolution_reuses_exact_array_without_loading(self) -> None:
        loaded = np.array([9.0], dtype=np.float32)
        prepared = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        loader = _Loader(loaded)

        actual = resolve_audio_samples(loader, "song.mp3", prepared)

        self.assertIs(actual, prepared)
        self.assertEqual(loader.calls, [])
        self.assertEqual(audio_array_metadata(actual), audio_array_metadata(prepared))

    def test_preloaded_audio_validation_fails_loudly(self) -> None:
        cases = [
            ([0.0], "numpy.ndarray"),
            (np.array([0.0], dtype=np.float64), "float32"),
            (np.zeros((1, 1), dtype=np.float32), "one-dimensional"),
            (np.array([], dtype=np.float32), "at least one"),
            (np.array([np.nan], dtype=np.float32), "non-finite"),
        ]
        for samples, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex((TypeError, ValueError), error):
                    validate_preloaded_audio(samples)

    def test_audio_array_hash_covers_shape_dtype_and_values(self) -> None:
        first = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        same = first.copy()
        changed = first.copy()
        changed[-1] = -0.25

        self.assertEqual(audio_array_metadata(first), audio_array_metadata(same))
        self.assertNotEqual(
            audio_array_metadata(first)["audio_array_sha256"],
            audio_array_metadata(changed)["audio_array_sha256"],
        )

    def test_audio_task_returns_exact_worker_result_and_restores_cleanly(self) -> None:
        samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        task = AudioPreparationTask(lambda path: samples, "song.mp3")

        result = task.result()
        task.close()
        task.close()

        self.assertIs(result.samples, samples)
        self.assertEqual(result.metadata, audio_array_metadata(samples))
        self.assertGreaterEqual(result.worker_wall_seconds, 0)
        self.assertTrue(result.worker_thread_name.startswith("mapperatorinator-audio"))
        self.assertTrue(task.closed)
        self.assertFalse(
            any(
                thread.name.startswith("mapperatorinator-audio")
                for thread in threading.enumerate()
            )
        )
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            task.result()

    def test_audio_task_loader_failure_propagates_and_cleanup_restores(self) -> None:
        def fail(_path: str) -> np.ndarray:
            raise RuntimeError("decoder failed")

        task = AudioPreparationTask(fail, "song.mp3")
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            task.result()
        task.close()

        self.assertTrue(task.closed)
        self.assertFalse(
            any(
                thread.name.startswith("mapperatorinator-audio")
                for thread in threading.enumerate()
            )
        )

    def test_scout_import_keeps_native_extensions_cold_and_quiet(self) -> None:
        command = [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import contextlib, io, sys, warnings",
                    "warnings.filterwarnings('ignore', message='Couldn.t find ffmpeg or avconv.*', category=RuntimeWarning)",
                    "stdout=io.StringIO()",
                    "stderr=io.StringIO()",
                    "with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):",
                    "    import osuT5.osuT5.inference.optimized.audio_model_overlap_scout",
                    "assert stdout.getvalue()==''",
                    "assert stderr.getvalue()==''",
                    "assert 'osuT5.osuT5.inference.optimized.kernels.q1_attention' not in sys.modules",
                    "assert 'osuT5.osuT5.inference.optimized.kernels.decoder_layer' not in sys.modules",
                ]
            ),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_default_inference_import_does_not_import_overlap_scout(self) -> None:
        command = [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import contextlib, io, sys",
                    "stdout=io.StringIO()",
                    "stderr=io.StringIO()",
                    "with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):",
                    "    import inference",
                    "assert 'osuT5.osuT5.inference.optimized.audio_model_overlap_scout' not in sys.modules",
                ]
            ),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_reciprocal_gate_requires_exactness_and_half_second_cold_win(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = analyze_reciprocal(_reciprocal_records(Path(directory)))

        self.assertTrue(report["promotion_pass"])
        self.assertTrue(report["exactness_pass"])
        self.assertTrue(report["exactness"]["selected_topology_pass"])
        self.assertEqual(
            report["performance"]["paired_cold_wall_savings_seconds"],
            [1.0, 1.0],
        )
        self.assertEqual(
            report["performance"]["mean_cold_wall_saving_seconds"],
            1.0,
        )

    def test_reciprocal_gate_stops_on_changed_osu_or_order_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = _reciprocal_records(Path(directory))
            candidate_manifest = Path(records[1]["manifest_path"])
            _success_manifest(candidate_manifest, result_hash="changed")
            records[1]["cold_process_wall_seconds"] = 10.1
            report = analyze_reciprocal(records)

        self.assertFalse(report["promotion_pass"])
        self.assertFalse(report["exactness"]["result_osu_hash_pass"])
        self.assertFalse(
            report["performance"]["no_reciprocal_order_regression"]
        )

    def test_reciprocal_setup_failure_forbids_performance_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = _reciprocal_records(Path(directory))
            records[2]["exit_code"] = 2
            report = analyze_reciprocal(records)

        self.assertEqual(report["status"], "setup_failure")
        self.assertIs(report["performance_conclusion_allowed"], False)
        self.assertIs(report["promotion_pass"], False)

    def test_dcc_wrapper_pins_selected_stack_and_half_second_gate(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "dcc"
            / "profile_audio_model_overlap_reciprocal.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn('refs/remotes/$REMOTE/$BRANCH', source)
        self.assertIn('rev-parse "$REMOTE_REF"', source)
        self.assertIn("preload_weight_only_extension", source)
        self.assertIn("preload_int8_mlp_extension", source)
        self.assertIn("--minimum-saving-seconds 0.5", source)
        self.assertIn("super_timing=false", source)


if __name__ == "__main__":
    unittest.main()
