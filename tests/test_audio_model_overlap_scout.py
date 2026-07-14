from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from osuT5.osuT5.inference.audio_preparation import (
    audio_array_metadata,
    resolve_audio_samples,
    validate_preloaded_audio,
)
from osuT5.osuT5.inference.optimized.scout.audio_model_overlap import (
    AudioPreparationTask,
)
from utils.profile_audio_model_overlap import (
    RUN_ORDER,
    analyze_reciprocal,
)


class _Loader:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.calls: list[str] = []

    def load(self, path: str) -> np.ndarray:
        self.calls.append(path)
        return self.samples


def test_default_audio_resolution_uses_unchanged_loader_result() -> None:
    samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    loader = _Loader(samples)

    actual = resolve_audio_samples(loader, "song.mp3")

    assert actual is samples
    assert loader.calls == ["song.mp3"]


def test_opt_in_audio_resolution_reuses_exact_array_without_loading() -> None:
    loaded = np.array([9.0], dtype=np.float32)
    prepared = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    loader = _Loader(loaded)

    actual = resolve_audio_samples(loader, "song.mp3", prepared)

    assert actual is prepared
    assert loader.calls == []
    assert audio_array_metadata(actual) == audio_array_metadata(prepared)


@pytest.mark.parametrize(
    ("samples", "error"),
    [
        ([0.0], "numpy.ndarray"),
        (np.array([0.0], dtype=np.float64), "float32"),
        (np.zeros((1, 1), dtype=np.float32), "one-dimensional"),
        (np.array([], dtype=np.float32), "at least one"),
        (np.array([np.nan], dtype=np.float32), "non-finite"),
    ],
)
def test_preloaded_audio_validation_fails_loudly(samples, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        validate_preloaded_audio(samples)


def test_audio_array_hash_covers_shape_dtype_and_values() -> None:
    first = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    same = first.copy()
    changed = first.copy()
    changed[-1] = -0.25

    assert audio_array_metadata(first) == audio_array_metadata(same)
    assert (
        audio_array_metadata(first)["audio_array_sha256"]
        != audio_array_metadata(changed)["audio_array_sha256"]
    )


def test_audio_task_returns_exact_worker_result_and_restores_cleanly() -> None:
    samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    task = AudioPreparationTask(lambda path: samples, "song.mp3")

    result = task.result()
    task.close()
    task.close()

    assert result.samples is samples
    assert result.metadata == audio_array_metadata(samples)
    assert result.worker_wall_seconds >= 0
    assert result.worker_thread_name.startswith("mapperatorinator-audio")
    assert task.closed
    assert not any(
        thread.name.startswith("mapperatorinator-audio")
        for thread in threading.enumerate()
    )
    with pytest.raises(RuntimeError, match="already closed"):
        task.result()


def test_audio_task_loader_failure_propagates_and_cleanup_restores() -> None:
    def fail(_path: str) -> np.ndarray:
        raise RuntimeError("decoder failed")

    task = AudioPreparationTask(fail, "song.mp3")
    with pytest.raises(RuntimeError, match="decoder failed"):
        task.result()
    task.close()

    assert task.closed
    assert not any(
        thread.name.startswith("mapperatorinator-audio")
        for thread in threading.enumerate()
    )


def test_scout_import_keeps_native_extensions_cold_and_quiet() -> None:
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import contextlib, io, sys",
                "stdout=io.StringIO()",
                "stderr=io.StringIO()",
                "with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):",
                "    import osuT5.osuT5.inference.optimized.scout.audio_model_overlap",
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

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def _success_manifest(path: Path, *, result_hash: str = "osu") -> None:
    payload = {
        "status": "success",
        "audio_task_closed": True,
        "inner_request_wall_seconds": 8.0,
        "audio": {"audio_array_sha256": "audio"},
        "audio_worker": None,
        "result_sha256": result_hash,
        "profile_contract": {"seed": 12345},
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


def test_reciprocal_gate_requires_exactness_and_half_second_cold_win(
    tmp_path: Path,
) -> None:
    report = analyze_reciprocal(_reciprocal_records(tmp_path))

    assert report["promotion_pass"]
    assert report["exactness_pass"]
    assert report["performance"]["paired_cold_wall_savings_seconds"] == [1.0, 1.0]
    assert report["performance"]["mean_cold_wall_saving_seconds"] == 1.0


def test_reciprocal_gate_stops_on_changed_osu_or_order_regression(
    tmp_path: Path,
) -> None:
    records = _reciprocal_records(tmp_path)
    candidate_manifest = Path(records[1]["manifest_path"])
    _success_manifest(candidate_manifest, result_hash="changed")
    records[1]["cold_process_wall_seconds"] = 10.1

    report = analyze_reciprocal(records)

    assert not report["promotion_pass"]
    assert not report["exactness"]["result_osu_hash_pass"]
    assert not report["performance"]["no_reciprocal_order_regression"]


def test_reciprocal_setup_failure_forbids_performance_conclusion(
    tmp_path: Path,
) -> None:
    records = _reciprocal_records(tmp_path)
    records[2]["exit_code"] = 2

    report = analyze_reciprocal(records)

    assert report["status"] == "setup_failure"
    assert report["performance_conclusion_allowed"] is False
    assert report["promotion_pass"] is False
