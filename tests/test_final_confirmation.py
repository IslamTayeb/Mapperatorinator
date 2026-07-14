import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.analyze_final_confirmation import _memory_stability
from utils.run_final_confirmation import (
    MANIFEST_SCHEMA_VERSION,
    ConfirmationRuntime,
    _load_manifest,
)
from utils.run_serial_final_confirmation import _songs


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_500tps_final_confirmation.sbatch"


class _Runtime:
    def __init__(self):
        self.calls = []

    def profile_metadata(self):
        return {"owner": "fake"}

    def generate_window(self, **kwargs):
        generate_kwargs = kwargs["generate_kwargs"]
        prompt = kwargs["model_kwargs"]["decoder_input_ids"]
        steps = int(generate_kwargs.get("max_length", prompt.shape[1] + 3)) - int(
            prompt.shape[1]
        )
        self.calls.append((dict(generate_kwargs), steps))
        result = torch.zeros((1, prompt.shape[1] + steps), dtype=torch.long)
        return result, {"generated_tokens": steps, "elapsed_seconds": 1.0}


def _manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "labels": {"timing_context": [2], "main_generation": [4]},
    }


def test_fixed_work_runtime_replays_exact_per_window_lengths() -> None:
    runtime = _Runtime()
    wrapped = ConfirmationRuntime(runtime, mode="replay-fixed-work", manifest=_manifest())
    prompt = torch.ones((1, 5), dtype=torch.long)
    tokenizer = SimpleNamespace(eos_id=99, context_eos={}, pad_id=0)

    timing, _ = wrapped.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "timing", "max_length": 100},
    )
    main, _ = wrapped.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "map", "max_length": 100},
    )
    wrapped.validate_complete()

    assert timing.shape[1] == 7
    assert main.shape[1] == 9
    assert [call[1] for call in runtime.calls] == [2, 4]
    assert [record["logical_steps"] for record in wrapped.records] == [2, 4]
    assert all(record["target_steps"] == record["logical_steps"] for record in wrapped.records)


def test_fixed_work_runtime_fails_on_extra_or_missing_windows() -> None:
    prompt = torch.ones((1, 5), dtype=torch.long)
    tokenizer = SimpleNamespace(eos_id=99, context_eos={}, pad_id=0)
    wrapped = ConfirmationRuntime(_Runtime(), mode="replay-fixed-work", manifest=_manifest())
    wrapped.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "timing"},
    )
    with pytest.raises(RuntimeError, match="consumed 0 main_generation"):
        wrapped.validate_complete()
    with pytest.raises(RuntimeError, match="too many timing_context"):
        wrapped.generate_window(
            tokenizer=tokenizer,
            model_kwargs={"decoder_input_ids": prompt},
            generate_kwargs={"context_type": "timing"},
        )


def test_fixed_work_executes_target_but_returns_natural_eos_prefix() -> None:
    class EosRuntime(_Runtime):
        def generate_window(self, **kwargs):
            result, stats = super().generate_window(**kwargs)
            prompt_width = kwargs["model_kwargs"]["decoder_input_ids"].shape[1]
            result[0, prompt_width + 1] = 99
            return result, stats

    wrapped = ConfirmationRuntime(
        EosRuntime(),
        mode="replay-fixed-work",
        manifest=_manifest(),
    )
    prompt = torch.ones((1, 5), dtype=torch.long)
    tokenizer = SimpleNamespace(eos_id=99, context_eos={}, pad_id=0)
    result, stats = wrapped.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "map"},
    )

    assert wrapped.records[0]["logical_steps"] == 4
    assert wrapped.records[0]["consumer_steps"] == 2
    assert result.shape[1] == prompt.shape[1] + 2
    assert stats["fixed_work_logical_steps"] == 4


def test_manifest_and_five_song_inputs_fail_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "fixed.json"
    manifest.write_text(json.dumps(_manifest()))
    assert _load_manifest(manifest)["labels"]["main_generation"] == [4]

    song_rows = []
    for index in range(5):
        audio = tmp_path / f"song-{index}.mp3"
        audio.write_bytes(f"audio-{index}".encode())
        song_rows.append(
            {
                "name": f"song-{index}",
                "audio_path": str(audio),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            }
        )
    songs = tmp_path / "songs.json"
    songs.write_text(json.dumps({"schema_version": 1, "songs": song_rows}))
    assert len(_songs(songs)) == 5
    song_rows[-1]["sha256"] = "0" * 64
    songs.write_text(json.dumps({"schema_version": 1, "songs": song_rows}))
    with pytest.raises(ValueError, match="hash mismatch"):
        _songs(songs)


def test_serial_memory_gate_uses_allocated_not_reserved_growth() -> None:
    mib = 1024**2
    evidence = {
        "runs": [
            {
                "cuda_memory": {
                    "allocated_bytes_before": 100 * mib,
                    "allocated_bytes_after": (100 + index) * mib,
                    "reserved_bytes_before": 100 * mib,
                    "reserved_bytes_after": (500 + 100 * index) * mib,
                }
            }
            for index in range(5)
        ]
    }
    report = _memory_stability(evidence, tolerance_mb=16.0)
    assert report["pass"] is True
    evidence["runs"][-1]["cuda_memory"]["allocated_bytes_after"] = 130 * mib
    assert _memory_stability(evidence, tolerance_mb=16.0)["pass"] is False


def test_dcc_wrapper_is_one_serial_authoritative_json_text_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2080:1" in source
    assert "EXPECTED_MAIN_STEPS=${EXPECTED_MAIN_STEPS:-8294}" in source
    assert "FIXED_REPETITIONS=${FIXED_REPETITIONS:-5}" in source
    assert "utils/run_final_confirmation.py" in source
    assert "utils/run_serial_final_confirmation.py" in source
    assert "utils/analyze_final_confirmation.py" in source
    assert '"$RUN_ROOT/analysis.json"' in source
    assert '"$RUN_ROOT/analysis.txt"' in source
    assert ".html" not in source
    assert ".md" not in source
