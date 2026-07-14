import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.analyze_final_confirmation import (
    CandidateAnalysisError,
    _k4_contract,
    _memory_stability,
    _validate_initialization,
    _verified_artifact,
    _wall_comparison,
    analysis_exit_code,
)
from utils.run_final_confirmation import (
    MANIFEST_SCHEMA_VERSION,
    ConfirmationRuntime,
    ConfirmationState,
    _load_manifest,
)
from utils.run_serial_final_confirmation import _songs


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_500tps_final_confirmation.sbatch"
FRESH_RUNNER = ROOT / "utils/run_final_confirmation.py"
SERIAL_RUNNER = ROOT / "utils/run_serial_final_confirmation.py"


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
    wrapped = ConfirmationRuntime(
        runtime, mode="replay-fixed-work", manifest=_manifest()
    )
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
    assert all(
        record["target_steps"] == record["logical_steps"] for record in wrapped.records
    )


def test_fixed_work_runtime_fails_on_extra_or_missing_windows() -> None:
    prompt = torch.ones((1, 5), dtype=torch.long)
    tokenizer = SimpleNamespace(eos_id=99, context_eos={}, pad_id=0)
    wrapped = ConfirmationRuntime(
        _Runtime(), mode="replay-fixed-work", manifest=_manifest()
    )
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


def test_distinct_timing_and_main_runtimes_share_one_manifest_state() -> None:
    state = ConfirmationState()
    timing = ConfirmationRuntime(
        _Runtime(),
        mode="replay-fixed-work",
        manifest=_manifest(),
        state=state,
    )
    main = ConfirmationRuntime(
        _Runtime(),
        mode="replay-fixed-work",
        manifest=_manifest(),
        state=state,
    )
    prompt = torch.ones((1, 5), dtype=torch.long)
    tokenizer = SimpleNamespace(eos_id=99, context_eos={}, pad_id=0)

    timing.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "timing"},
    )
    main.generate_window(
        tokenizer=tokenizer,
        model_kwargs={"decoder_input_ids": prompt},
        generate_kwargs={"context_type": "map"},
    )
    timing.validate_complete()

    assert timing.records is main.records
    assert [record["profile_label"] for record in state.records] == [
        "timing_context",
        "main_generation",
    ]


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
    assert "MAPPERATORINATOR_REMOTE_BRANCH" in source
    assert '[[ "$BRANCH" == DETACHED ]]' in source
    assert "utils/run_final_confirmation.py" in source
    assert "utils/run_serial_final_confirmation.py" in source
    assert "utils/analyze_final_confirmation.py" in source
    assert '"$RUN_ROOT/analysis.json"' in source
    assert '"$RUN_ROOT/analysis.txt"' in source
    assert ".html" not in source
    assert ".md" not in source


def test_candidate_confirmation_uses_k4_and_supports_separate_timing_model() -> None:
    fresh = FRESH_RUNNER.read_text(encoding="utf-8")
    serial = SERIAL_RUNNER.read_text(encoding="utf-8")

    assert "install_k8_candidate(block_size=4)" in fresh
    assert "state = ConfirmationState()" in fresh
    assert "state = ConfirmationState()" in serial
    assert (
        "separate_timing_model = inference.should_load_separate_timing_model" in serial
    )
    assert "timing_model=timing_binding" in serial
    assert "supports one shared timing/main model only" not in fresh
    assert "songs must share one timing/main model" not in serial


def _candidate_k4_profile(*, main_physical: int = 8, rng_policy: str | None = None):
    def record(label: str, logical: int, physical: int):
        block_steps = logical - (logical % 4)
        wasted = physical - logical
        return {
            "profile_label": label,
            "generated_tokens": logical,
            "optimized_cuda_graphs": {
                "k8_candidate": {
                    "block_size": 4,
                    "rng_policy": (
                        rng_policy
                        if rng_policy is not None
                        else "counter_request_seed_window_prompt_v2"
                    ),
                    "rng_exact": False,
                    "rng_early_eos_isolation": True,
                    "capture_state_restore_synchronized": True,
                    "prefill_steps": 0,
                    "eligible_steps": block_steps,
                    "block_replays": block_steps // 4,
                    "remainder_steps": physical - block_steps,
                    "physical_steps": physical,
                    "logical_steps": logical,
                    "wasted_steps": wasted,
                }
            },
        }

    return {
        "generation": [
            record("timing_context", 4, 4),
            record("main_generation", 8, main_physical),
        ]
    }


def test_fixed_k4_contract_requires_exact_physical_logical_work_and_rng(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_candidate_k4_profile()), encoding="utf-8")
    report = _k4_contract(
        profile,
        candidate=True,
        expected_fixed_main_steps=8,
    )
    assert report["fixed_main_work"] == {
        "physical_steps": 8,
        "logical_steps": 8,
        "wasted_steps": 0,
    }

    profile.write_text(
        json.dumps(_candidate_k4_profile(main_physical=9)),
        encoding="utf-8",
    )
    with pytest.raises(CandidateAnalysisError, match="zero waste"):
        _k4_contract(profile, candidate=True, expected_fixed_main_steps=8)

    profile.write_text(
        json.dumps(_candidate_k4_profile(rng_policy="global-generator")),
        encoding="utf-8",
    )
    with pytest.raises(CandidateAnalysisError, match="wrong RNG policy"):
        _k4_contract(profile, candidate=True, expected_fixed_main_steps=8)


def test_artifact_hashes_are_verified_before_analysis(tmp_path: Path) -> None:
    artifact = tmp_path / "result.osu"
    artifact.write_bytes(b"stable")
    evidence = {
        "result_path": str(artifact),
        "result_sha256": hashlib.sha256(b"stable").hexdigest(),
    }
    assert _verified_artifact(evidence, "result") == artifact.resolve()
    artifact.write_bytes(b"tampered")
    with pytest.raises(CandidateAnalysisError, match="hash mismatch"):
        _verified_artifact(evidence, "result")


def test_candidate_initialization_proves_the_mixed_weight_topology() -> None:
    initialization = {
        "version": "approximate-fp16-weights-fp32-state-v2",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "fp32_activations_caches_reductions_logits": True,
        "fp16_weight_regions": [
            "self_qkv",
            "self_output",
            "mlp_fc1",
            "mlp_fc2",
            "final_logits",
        ],
        "fp32_selected_decode_matrix_regions": ["cross_query", "cross_output"],
        "initialization_wall_seconds": 2.0,
        "extension_init_seconds": 0.5,
        "weight_pack_seconds": 1.0,
        "retained_fp32_source_weight_bytes": 100,
        "packed_weight_bytes": 50,
        "initialization_cuda_memory": {},
    }
    _validate_initialization({"initialization": initialization}, candidate=True)
    _validate_initialization({"initialization": None}, candidate=False)

    initialization["version"] = "wrong"
    with pytest.raises(CandidateAnalysisError, match="wrong version"):
        _validate_initialization({"initialization": initialization}, candidate=True)
    with pytest.raises(CandidateAnalysisError, match="baseline unexpectedly"):
        _validate_initialization({"initialization": {}}, candidate=False)


def test_wall_nonregression_and_failed_acceptance_exit_loudly() -> None:
    assert _wall_comparison([10.0, 12.0], [9.0, 11.0])["pass"] is True
    assert _wall_comparison([10.0, 12.0], [11.0, 13.0])["pass"] is False
    assert analysis_exit_code({"acceptance": {"pass": True}}) == 0
    assert analysis_exit_code({"acceptance": {"pass": False}}) == 3
    with pytest.raises(CandidateAnalysisError, match="boolean acceptance.pass"):
        analysis_exit_code({"acceptance": {"pass": 1}})


def test_analyzer_writes_reports_before_nonzero_failed_gate() -> None:
    source = (ROOT / "utils/analyze_final_confirmation.py").read_text(encoding="utf-8")
    json_write = source.index("args.json_output.write_text")
    text_write = source.index("args.text_output.write_text")
    exit_call = source.index("raise SystemExit(analysis_exit_code(report))")
    assert json_write < text_write < exit_call
