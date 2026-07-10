from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "osuT5/osuT5/inference/optimized/batch/weighted_bucket.py"
)
_SPEC = importlib.util.spec_from_file_location("_mapperatorinator_weighted_bucket", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ACTIVE_PREFIX_LENGTH = _MODULE.ACTIVE_PREFIX_LENGTH
AcceptedProfileContract = _MODULE.AcceptedProfileContract
CAPTURE_PROMPT_LENGTH = _MODULE.CAPTURE_PROMPT_LENGTH
PHASE_A_HORIZON = _MODULE.PHASE_A_HORIZON
SOURCE_PREFIX_REPLAY_TOKENS = _MODULE.SOURCE_PREFIX_REPLAY_TOKENS
TARGET_PREFIX_REPLAY_TOKEN_IDS = _MODULE.TARGET_PREFIX_REPLAY_TOKEN_IDS
canonical_json_sha256 = _MODULE.canonical_json_sha256
load_accepted_profile = _MODULE.load_accepted_profile
load_reviewed_source_capture_report = _MODULE.load_reviewed_source_capture_report
reviewed_context_replay_report = _MODULE.reviewed_context_replay_report
token_ids_sha256 = _MODULE.token_ids_sha256
validate_accepted_profile = _MODULE.validate_accepted_profile
validate_context_replay_report = _MODULE.validate_context_replay_report
validate_model_free_ceiling_report = _MODULE.validate_model_free_ceiling_report
validate_reviewed_source_contract = _MODULE.validate_reviewed_source_contract
validate_reviewed_context_evidence = _MODULE.validate_reviewed_context_evidence
validate_reviewed_context_replay_pin = _MODULE.validate_reviewed_context_replay_pin
validate_reviewed_source_capture_report = _MODULE.validate_reviewed_source_capture_report
validate_source_contract = _MODULE.validate_source_contract
require_reviewed_context_replay = _MODULE.require_reviewed_context_replay


def _sha(value: int) -> str:
    return f"{value:064x}"


def _tensor(value: int, shape: list[int], dtype: str) -> dict:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return {
        "sha256": _sha(value),
        "shape": shape,
        "stride": list(reversed(stride)),
        "dtype": dtype,
    }


def _reviewed_tensor(value: int, shape: list[int], dtype: str, sha256: str) -> dict:
    descriptor = _tensor(value, shape, dtype)
    descriptor["sha256"] = sha256
    return descriptor


def _source_contract() -> dict:
    contract = _MODULE

    steps = []
    for index, token in enumerate(TARGET_PREFIX_REPLAY_TOKEN_IDS):
        steps.append({
            "step": index,
            "accepted_token": token,
            "sampled_token": token,
            "match": True,
            "rng_before_sha256": (
                contract.REVIEWED_SOURCE_PRE_TARGET_RNG_SHA256
                if index == 0
                else _sha(1000 + index)
            ),
            "rng_after_sha256": (
                contract.REVIEWED_SOURCE_POST_REPLAY_RNG_SHA256
                if index == SOURCE_PREFIX_REPLAY_TOKENS - 1
                else _sha(1001 + index)
            ),
        })
    return {
        "schema_version": contract.WEIGHTED_SOURCE_CONTRACT_SCHEMA_VERSION,
        "gate": contract.WEIGHTED_SOURCE_CONTRACT_GATE,
        "source": {
            "job_id": contract.SOURCE_JOB_ID,
            "commit": contract.SOURCE_COMMIT,
            "profile_sha256": contract.SOURCE_PROFILE_SHA256,
            "profile_size_bytes": contract.SOURCE_PROFILE_SIZE_BYTES,
            "audio_sha256": contract.SOURCE_AUDIO_SHA256,
            "audio_size_bytes": contract.SOURCE_AUDIO_SIZE_BYTES,
            "audio_samples": contract.SOURCE_AUDIO_SAMPLES,
            "result_sha256": contract.SOURCE_RESULT_SHA256,
            "profile_label": contract.SOURCE_PROFILE_LABEL,
            "sequence_index": contract.SOURCE_SEQUENCE_INDEX,
            "source_prompt_length": contract.SOURCE_PROMPT_LENGTH,
            "source_generated_tokens": contract.SOURCE_GENERATED_TOKENS,
            "model_snapshot_revision": contract.MODEL_SNAPSHOT_REVISION,
            "model_config_sha256": contract.MODEL_CONFIG_SHA256,
        },
        "transcript": {
            "timing_tokens": contract.TIMING_TRANSCRIPT_TOKENS,
            "timing_sha256": contract.TIMING_TRANSCRIPT_SHA256,
            "pre_target_main_tokens": contract.PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
            "pre_target_main_sha256": contract.PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
            "pre_target_rng_draws": contract.PRE_TARGET_RNG_DRAWS,
            "target_tokens": contract.SOURCE_GENERATED_TOKENS,
            "target_sha256": contract.TARGET_TRANSCRIPT_SHA256,
            "target_prefix_replay_tokens": contract.SOURCE_PREFIX_REPLAY_TOKENS,
            "target_prefix_replay_sha256": contract.TARGET_PREFIX_REPLAY_SHA256,
        },
        "reconstruction": {
            "source_prompt_length": contract.SOURCE_PROMPT_LENGTH,
            "capture_prompt_length": CAPTURE_PROMPT_LENGTH,
            "active_prefix_length": ACTIVE_PREFIX_LENGTH,
            "phase_a_horizon": PHASE_A_HORIZON,
            "prefix_build": "DecodeSession prompt478 plus exact sampled replay42",
            "prior_context_tokens_source": "accepted_profile",
            "target_prefix_forced_tokens": False,
            "dummy_rng_advance_only_before_target": True,
            "timing_tokenizer_scope": contract.BASE_TIMING_TOKENIZER_SCOPE,
            "main_tokenizer_scope": contract.MAIN_TOKENIZER_SCOPE,
            "timing_vocab_size": contract.BASE_TIMING_VOCAB_SIZE,
            "main_vocab_size": contract.MAIN_VOCAB_SIZE,
            "timing_beat_token_id": contract.BASE_TIMING_BEAT_TOKEN_ID,
            "timing_measure_token_id": contract.BASE_TIMING_MEASURE_TOKEN_ID,
            "main_beat_token_id": contract.MAIN_BEAT_TOKEN_ID,
            "main_measure_token_id": contract.MAIN_MEASURE_TOKEN_ID,
            "timing_pretrim_event_count": 122,
            "timing_event_count": 60,
            "timing_time_shift_count": 30,
            "timing_beat_count": 22,
            "timing_measure_count": 8,
            "timing_marker_count": 30,
            "timing_marker_min_time_ms": 71237,
            "timing_marker_max_time_ms": 85875,
            "timing_point_count": 1,
            "model_max_target_positions": contract.MODEL_MAX_TARGET_POSITIONS,
            "self_cache_max_length": contract.MODEL_MAX_TARGET_POSITIONS,
            "rng_seed": 12345,
            "rng_advance_method": (
                "one CUDA multinomial(num_samples=1) per accepted prior generated token"
            ),
            "timing_rng_advance_draw_calls": 132,
            "main_rng_advance_draw_calls": 1042,
            "cache_populated_through_position": 518,
            "prepared_cache_position": 519,
            "prepared_decoder_token_id": 924,
            "next_output_position": 42,
            "processor_calls": 42,
            "model_decode_calls_after_prefill": 41,
            "self_cache_position518_populated": True,
            "self_cache_position519_zero": True,
            "prepared_attention_mask_allowed_positions": 520,
            "prepared_attention_mask_masked_positions": 2040,
            "active_attention_mask_allowed_positions": 520,
            "active_attention_mask_masked_positions": 56,
            "prepared_attention_mask_values_exact": True,
            "active_attention_mask_values_exact": True,
            "source_prompt": _reviewed_tensor(
                1, [1, 478], "torch.int64", contract.REVIEWED_CONTEXT_PROMPT_SHA256
            ),
            "source_prompt_attention_mask": _reviewed_tensor(
                2,
                [1, 478],
                "torch.bool",
                contract.REVIEWED_CONTEXT_PROMPT_MASK_SHA256,
            ),
            "frames": _reviewed_tensor(
                3,
                [1, 262016],
                "torch.float32",
                contract.REVIEWED_CONTEXT_FRAMES_SHA256,
            ),
            "condition_tensors": {},
            "capture_prompt": _reviewed_tensor(
                4,
                [1, 520],
                "torch.int64",
                contract.REVIEWED_SOURCE_CAPTURE_PROMPT_SHA256,
            ),
            "capture_prompt_attention_mask": _reviewed_tensor(
                5,
                [1, 520],
                "torch.bool",
                contract.REVIEWED_SOURCE_CAPTURE_MASK_SHA256,
            ),
            "initial_rng_state": _reviewed_tensor(
                999,
                [16],
                "torch.uint8",
                contract.REVIEWED_SOURCE_INITIAL_RNG_SHA256,
            ),
            "post_timing_rng_state": _reviewed_tensor(
                998,
                [16],
                "torch.uint8",
                contract.REVIEWED_SOURCE_POST_TIMING_RNG_SHA256,
            ),
            "pre_target_rng_state": _reviewed_tensor(
                1000,
                [16],
                "torch.uint8",
                contract.REVIEWED_SOURCE_PRE_TARGET_RNG_SHA256,
            ),
            "post_replay_rng_state": _reviewed_tensor(
                1042,
                [16],
                "torch.uint8",
                contract.REVIEWED_SOURCE_POST_REPLAY_RNG_SHA256,
            ),
            "pre_last_sample_raw_logits": _reviewed_tensor(
                14,
                [1, 4069],
                "torch.float32",
                contract.REVIEWED_SOURCE_RAW_LOGITS_SHA256,
            ),
            "timing_rng_advance_probability": _tensor(15, [1, 4097], "torch.float32"),
            "main_rng_advance_probability": _tensor(16, [1, 4069], "torch.float32"),
            "prepared_static_inputs": {
                "cache_position": _tensor(6, [1], "torch.int64"),
                "decoder_attention_mask": _reviewed_tensor(
                    7,
                    [1, 1, 1, contract.MODEL_MAX_TARGET_POSITIONS],
                    "torch.float32",
                    contract.REVIEWED_SOURCE_PREPARED_MASK_SHA256,
                ),
                "decoder_input_ids": _tensor(8, [1, 1], "torch.int64"),
                "decoder_position_ids": _tensor(9, [1, 1], "torch.int64"),
            },
            "prospective_active_attention_mask": _reviewed_tensor(
                17,
                [1, 1, 1, ACTIVE_PREFIX_LENGTH],
                "torch.float32",
                contract.REVIEWED_SOURCE_ACTIVE_MASK_SHA256,
            ),
            "pre_next_forward_cache": {
                "self_sha256": contract.REVIEWED_SOURCE_SELF_CACHE_SHA256,
                "cross_sha256": contract.REVIEWED_SOURCE_CROSS_CACHE_SHA256,
                "self_tensors": [_tensor(12, [1, 8, 1024, 64], "torch.float32")],
                "cross_tensors": [_tensor(13, [1, 8, 1024, 64], "torch.float32")],
            },
        },
        "accepted_prefix_replay": {
            "committed_tokens": SOURCE_PREFIX_REPLAY_TOKENS,
            "all_tokens_match": True,
            "forced_tokens": False,
            "stopped": False,
            "accepted_token_ids_sha256": contract.TARGET_PREFIX_REPLAY_SHA256,
            "sampled_token_ids_sha256": contract.TARGET_PREFIX_REPLAY_SHA256,
            "steps": steps,
        },
        "maintainer_boundary": {
            "engine": "v32",
            "default_behavior_changed": False,
            "runtime_wiring_added": False,
            "phase_a_authorized": False,
            "authorization_requires_reviewed_contract_sha_commit": True,
        },
    }


def _context_report() -> dict:
    contract = _MODULE
    return {
        "schema_version": contract.WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION,
        "gate": contract.WEIGHTED_CONTEXT_REPLAY_GATE,
        "source": {
            "job_id": contract.SOURCE_JOB_ID,
            "commit": contract.SOURCE_COMMIT,
            "profile_sha256": contract.SOURCE_PROFILE_SHA256,
            "profile_size_bytes": contract.SOURCE_PROFILE_SIZE_BYTES,
            "audio_sha256": contract.SOURCE_AUDIO_SHA256,
            "audio_size_bytes": contract.SOURCE_AUDIO_SIZE_BYTES,
            "audio_samples": contract.SOURCE_AUDIO_SAMPLES,
            "timing_transcript_sha256": contract.TIMING_TRANSCRIPT_SHA256,
            "pre_target_main_transcript_sha256": (
                contract.PRE_TARGET_MAIN_TRANSCRIPT_SHA256
            ),
            "target_transcript_sha256": contract.TARGET_TRANSCRIPT_SHA256,
        },
        "tokenizers": {
            "timing_scope": contract.BASE_TIMING_TOKENIZER_SCOPE,
            "timing_vocab_size": contract.BASE_TIMING_VOCAB_SIZE,
            "timing_beat_token_id": contract.BASE_TIMING_BEAT_TOKEN_ID,
            "timing_measure_token_id": contract.BASE_TIMING_MEASURE_TOKEN_ID,
            "main_scope": contract.MAIN_TOKENIZER_SCOPE,
            "main_vocab_size": contract.MAIN_VOCAB_SIZE,
            "main_beat_token_id": contract.MAIN_BEAT_TOKEN_ID,
            "main_measure_token_id": contract.MAIN_MEASURE_TOKEN_ID,
        },
        "timing": {
            "pretrim_event_count": 122,
            "event_count": 60,
            "time_shift_count": 30,
            "beat_count": 22,
            "measure_count": 8,
            "marker_count": 30,
            "marker_min_time_ms": 71237,
            "marker_max_time_ms": 85875,
            "timing_point_count": 1,
        },
        "main": {
            "prompt_token_counts": list(contract.EXPECTED_MAIN_PROMPT_TOKENS),
            "target_sequence_index": contract.SOURCE_SEQUENCE_INDEX,
            "target_frame_time_ms": 76965.0,
            "target_context_type": "map",
            "target_prompt": {
                "sha256": contract.REVIEWED_CONTEXT_PROMPT_SHA256,
                "shape": [1, 478],
                "stride": [478, 1],
                "dtype": "torch.int64",
            },
            "target_prompt_attention_mask": {
                "sha256": contract.REVIEWED_CONTEXT_PROMPT_MASK_SHA256,
                "shape": [1, 478],
                "stride": [478, 1],
                "dtype": "torch.bool",
            },
            "target_frames": {
                "sha256": contract.REVIEWED_CONTEXT_FRAMES_SHA256,
                "shape": [1, 262016],
                "stride": [262016, 1],
                "dtype": "torch.float32",
            },
            "condition_tensor_keys": [],
        },
        "maintainer_boundary": {
            "engine": "v32",
            "model_loaded": False,
            "cuda_used": False,
            "gpu_capture_authorized": False,
            "authorization_requires_reviewed_context_sha_commit": True,
        },
    }


def _source_capture_report() -> dict:
    contract = _source_contract()
    reconstruction = contract["reconstruction"]
    method = reconstruction["rng_advance_method"]
    report = {
        **copy.deepcopy(contract),
        "rng_advance": {
            "segments": [
                {
                    "scope": _MODULE.BASE_TIMING_TOKENIZER_SCOPE,
                    "method": method,
                    "vocab_size": _MODULE.BASE_TIMING_VOCAB_SIZE,
                    "draw_calls": _MODULE.TIMING_TRANSCRIPT_TOKENS,
                    "probability": reconstruction["timing_rng_advance_probability"],
                    "rng_before": reconstruction["initial_rng_state"],
                    "rng_after": reconstruction["post_timing_rng_state"],
                },
                {
                    "scope": _MODULE.MAIN_TOKENIZER_SCOPE,
                    "method": method,
                    "vocab_size": _MODULE.MAIN_VOCAB_SIZE,
                    "draw_calls": _MODULE.PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
                    "probability": reconstruction["main_rng_advance_probability"],
                    "rng_before": reconstruction["post_timing_rng_state"],
                    "rng_after": reconstruction["pre_target_rng_state"],
                },
            ],
            "initial_rng_state": reconstruction["initial_rng_state"],
            "post_timing_rng_state": reconstruction["post_timing_rng_state"],
            "pre_target_rng_state": reconstruction["pre_target_rng_state"],
        },
        "runtime_metadata": {
            "git_commit": _MODULE.REVIEWED_SOURCE_CAPTURE_COMMIT,
            "config_name": "profile_salvalai_smoke15",
            "torch_version": "2.10.0+cu128",
            "transformers_version": "4.57.3",
            "cuda_version": "12.8",
            "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
            "gpu_capability": [7, 5],
            "probe": {
                "context_type": "map",
                "sequence_index": _MODULE.SOURCE_SEQUENCE_INDEX,
                "frame_time_ms": 76965.0,
                "lookback_time": 0.0,
                "lookahead_time": 0.0,
                "timing_pretrim_event_count": 122,
                "timing_event_count": 60,
                "timing_time_shift_count": 30,
                "timing_beat_count": 22,
                "timing_measure_count": 8,
                "timing_marker_count": 30,
                "timing_marker_min_time_ms": 71237,
                "timing_marker_max_time_ms": 85875,
                "timing_point_count": 1,
                "main_prompt_token_counts": list(_MODULE.EXPECTED_MAIN_PROMPT_TOKENS),
            },
            "torch_extensions_dir": "/work/imt11/Mapperatorinator/torch_extensions",
            "torch_extensions_cache_before": {
                "path": "/work/imt11/Mapperatorinator/torch_extensions",
                "exists": True,
                "entry_count": 10,
                "file_count": 112,
            },
            "torch_extensions_cache_after": {
                "path": "/work/imt11/Mapperatorinator/torch_extensions",
                "exists": True,
                "entry_count": 10,
                "file_count": 112,
            },
            "total_wall_seconds": 8.1,
            "capture_only": True,
            "h8_executed": False,
            "scheduler_or_runtime_wiring_authorized": False,
            "reviewed_context_report_sha256": (
                _MODULE.REVIEWED_CONTEXT_REPLAY_REPORT_SHA256
            ),
            "context_preflight_wall_seconds": 1.2,
            "fresh_context_replay_match": True,
            "live_context_evidence_match": True,
        },
    }
    report["source_contract_sha256"] = canonical_json_sha256(contract)
    return report


def _accepted_profile() -> tuple[dict, AcceptedProfileContract]:
    timing_tokens = [[1000 + 10 * index + offset for offset in range(2)] for index in range(10)]
    main_tokens = [[2000 + 10 * index + offset for offset in range(3)] for index in range(10)]
    timing_flat = [token for record in timing_tokens for token in record]
    pre_main_flat = [token for record in main_tokens[:9] for token in record]
    target = main_tokens[9]
    frame_times = tuple(10000 + index for index in range(10))
    contract = AcceptedProfileContract(
        profile_sha256="pending",
        source_commit="a" * 40,
        audio_sha256="b" * 64,
        result_sha256="c" * 64,
        timing_transcript_sha256=token_ids_sha256(timing_flat),
        pre_target_main_transcript_sha256=token_ids_sha256(pre_main_flat),
        target_transcript_sha256=token_ids_sha256(target),
        target_prefix_replay_sha256=token_ids_sha256(target[:SOURCE_PREFIX_REPLAY_TOKENS]),
        frame_times_ms=frame_times,
        timing_prompt_tokens=(1,) * 10,
        timing_generated_tokens=(2,) * 10,
        main_prompt_tokens=(2,) * 9 + (478,),
        main_generated_tokens=(3,) * 10,
    )
    generation = []
    for label, context_type, prompts, records in (
            ("timing_context", "timing", contract.timing_prompt_tokens, timing_tokens),
            ("main_generation", "map", contract.main_prompt_tokens, main_tokens),
    ):
        for index, token_ids in enumerate(records):
            generation.append({
                "profile_label": label,
                "context_type": context_type,
                "mode": "sequential",
                "sequence_index": index,
                "frame_time_ms": frame_times[index],
                "prompt_tokens": prompts[index],
                "generated_tokens": len(token_ids),
                "trim_lookback": False,
                "trim_lookahead": index != 9,
                "generated_token_ids": token_ids,
            })
    profile = {
        "schema_version": 1,
        "metadata": {
            "git_commit": contract.source_commit,
            "result_file_sha256": contract.result_sha256,
            "model_path": "OliBomby/Mapperatorinator-v32",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "start_time": 71000,
            "end_time": 86000,
            "song_id": "lambada",
            "repeat_index": 1,
            "run_index": 5,
            "seed": 12345,
            "profile_record_token_ids": True,
        },
        "generation": generation,
    }
    return profile, contract


def test_model_free_ceiling_exact_arithmetic() -> None:
    report = json.loads(
        (Path(__file__).resolve().parents[1] / "notes/inference-weighted-bucket-ceiling-report.json")
        .read_text(encoding="utf-8")
    )
    validate_model_free_ceiling_report(report)
    assert report["five_request_ideal_k2"][
        "target_closed_in_accepted_profile_cost_model"
    ] is True
    assert report["five_request_ideal_k2"]["ideal_scheduler_wall_tokens_per_second"] == pytest.approx(
        495.9834100880575
    )
    assert report["five_request_ideal_k3"]["ideal_scheduler_wall_tokens_per_second"] == pytest.approx(
        640.7674443771249
    )
    assert report["duplicated_ten_request_b2"]["required_seconds_per_pair"] == pytest.approx(
        0.0032068926334242313
    )
    assert report["duplicated_ten_request_b2"][
        "required_weighted_complete_tokens_per_second"
    ] == pytest.approx(623.656675984333)


def test_model_free_ceiling_rejects_mutation() -> None:
    report = json.loads(
        (Path(__file__).resolve().parents[1] / "notes/inference-weighted-bucket-ceiling-report.json")
        .read_text(encoding="utf-8")
    )
    report["duplicated_ten_request_b2"]["optimistic_linearized_setup_seconds"] -= 1.0
    with pytest.raises(ValueError, match="source-derived arithmetic"):
        validate_model_free_ceiling_report(report)

    forged = json.loads(
        (Path(__file__).resolve().parents[1] / "notes/inference-weighted-bucket-ceiling-report.json")
        .read_text(encoding="utf-8")
    )
    forged["source"]["profile_inputs"][0]["main_records"][0][
        "model_elapsed_seconds"
    ] += 1.0
    with pytest.raises(ValueError, match="records changed"):
        validate_model_free_ceiling_report(forged)


def test_context_replay_contract_and_full_capture_guard(monkeypatch) -> None:
    report = _context_report()
    validate_context_replay_report(report)
    assert report == reviewed_context_replay_report()
    assert canonical_json_sha256(report) == (
        _MODULE.REVIEWED_CONTEXT_REPLAY_REPORT_SHA256
    )
    assert require_reviewed_context_replay(report) == (
        _MODULE.REVIEWED_CONTEXT_REPLAY_REPORT_SHA256
    )

    wrong_tokenizer = copy.deepcopy(report)
    wrong_tokenizer["tokenizers"]["timing_vocab_size"] = 4069
    with pytest.raises(ValueError, match="tokenizer scopes changed"):
        validate_context_replay_report(wrong_tokenizer)

    wrong_timing = copy.deepcopy(report)
    wrong_timing["timing"]["marker_count"] = 29
    with pytest.raises(ValueError, match="marker_count changed"):
        validate_context_replay_report(wrong_timing)

    wrong_timing_points = copy.deepcopy(report)
    wrong_timing_points["timing"]["timing_point_count"] = 2
    with pytest.raises(ValueError, match="timing_point_count changed"):
        validate_context_replay_report(wrong_timing_points)

    wrong_prompt = copy.deepcopy(report)
    wrong_prompt["main"]["prompt_token_counts"][-1] = 479
    with pytest.raises(ValueError, match="prompt counts changed"):
        validate_context_replay_report(wrong_prompt)

    wrong_prompt_hash = copy.deepcopy(report)
    wrong_prompt_hash["main"]["target_prompt"]["sha256"] = _sha(23)
    with pytest.raises(ValueError, match="prompt descriptor changed"):
        validate_context_replay_report(wrong_prompt_hash)

    wrong_live = {
        field: copy.deepcopy(report[field])
        for field in ("source", "tokenizers", "timing", "main")
    }
    validate_reviewed_context_evidence(wrong_live)
    wrong_live["main"]["target_frames"]["stride"] = [1, 1]
    with pytest.raises(ValueError, match="field main differs"):
        validate_reviewed_context_evidence(wrong_live)

    monkeypatch.setattr(_MODULE, "REVIEWED_CONTEXT_REPLAY_REPORT_SHA256", None)
    with pytest.raises(RuntimeError, match="post-context-report commit pins"):
        validate_reviewed_context_replay_pin()

    monkeypatch.setattr(_MODULE, "REVIEWED_CONTEXT_REPLAY_REPORT_SHA256", "f" * 64)
    with pytest.raises(RuntimeError, match="does not match its pinned"):
        validate_reviewed_context_replay_pin()


def test_full_capture_preflight_precedes_cuda_and_model_load() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "utils/verify_optimized_weighted_prefix_source.py"
    ).read_text(encoding="utf-8")
    capture = source.split("def run_source_capture(", 1)[1].split(
        "\ndef build_parser()", 1
    )[0]
    replay_offset = capture.index("fresh_context_report = run_context_replay(")
    match_offset = capture.index(
        "reviewed_context_report_sha256 = require_reviewed_context_replay("
    )
    compile_offset = capture.index("compile_args(args, verbose=False)")
    cuda_offset = capture.index("setup_inference_environment(12345)")
    model_offset = capture.index("model, tokenizer = load_model_with_server(")
    assert replay_offset < match_offset < compile_offset < cuda_offset < model_offset
    assert "reviewed-context" not in source
    assert "reviewed_context" not in source.split("def build_parser()", 1)[1]


def test_source_contract_requires_exact_replay_and_reviewed_hash() -> None:
    contract = _source_contract()
    validate_source_contract(contract)
    digest = canonical_json_sha256(contract)
    validate_reviewed_source_contract(contract, expected_contract_sha256=digest)

    bad = copy.deepcopy(contract)
    bad["accepted_prefix_replay"]["steps"][17]["sampled_token"] += 1
    bad["accepted_prefix_replay"]["steps"][17]["match"] = False
    with pytest.raises(ValueError, match="step 17 differs"):
        validate_source_contract(bad)

    bad_rng = copy.deepcopy(contract)
    bad_rng["accepted_prefix_replay"]["steps"][18]["rng_before_sha256"] = _sha(9999)
    with pytest.raises(ValueError, match="RNG chain breaks"):
        validate_source_contract(bad_rng)

    unknown = copy.deepcopy(contract)
    unknown["accepted_prefix_replay"]["unknown"] = True
    with pytest.raises(ValueError, match="replay fields changed"):
        validate_source_contract(unknown)

    string_token = copy.deepcopy(contract)
    string_token["accepted_prefix_replay"]["steps"][0]["accepted_token"] = "848"
    string_token["accepted_prefix_replay"]["steps"][0]["sampled_token"] = "848"
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        validate_source_contract(string_token)

    wrong_timing_scope = copy.deepcopy(contract)
    wrong_timing_scope["reconstruction"]["timing_vocab_size"] = 4069
    with pytest.raises(ValueError, match="timing_vocab_size changed"):
        validate_source_contract(wrong_timing_scope)

    wrong_cache_max = copy.deepcopy(contract)
    wrong_cache_max["reconstruction"]["self_cache_max_length"] = 1024
    with pytest.raises(ValueError, match="self_cache_max_length changed"):
        validate_source_contract(wrong_cache_max)

    wrong_static_mask = copy.deepcopy(contract)
    wrong_static_mask["reconstruction"]["prepared_static_inputs"][
        "decoder_attention_mask"
    ]["shape"][-1] = 1024
    with pytest.raises(ValueError, match="decoder_attention_mask tensor shape changed"):
        validate_source_contract(wrong_static_mask)

    with pytest.raises(ValueError, match="separately reviewed"):
        validate_reviewed_source_contract(contract, expected_contract_sha256="f" * 64)
    with pytest.raises(ValueError, match="64 hex"):
        validate_reviewed_source_contract(contract, expected_contract_sha256="g" * 64)


def test_reviewed_source_capture_loader_is_pin_only(tmp_path, monkeypatch) -> None:
    report = _source_capture_report()
    contract_digest = report["source_contract_sha256"]
    monkeypatch.setattr(
        _MODULE, "REVIEWED_SOURCE_CONTRACT_SHA256", contract_digest
    )
    validate_reviewed_source_capture_report(report)

    path = tmp_path / "weighted-prefix-source.json"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    file_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    monkeypatch.setattr(
        _MODULE, "REVIEWED_SOURCE_REPORT_FILE_SHA256", file_digest
    )
    assert load_reviewed_source_capture_report(path) == report

    wrong_h8 = copy.deepcopy(report)
    wrong_h8["runtime_metadata"]["h8_executed"] = True
    with pytest.raises(ValueError, match="runtime field h8_executed changed"):
        validate_reviewed_source_capture_report(wrong_h8)

    wrong_context = copy.deepcopy(report)
    wrong_context["runtime_metadata"]["live_context_evidence_match"] = False
    with pytest.raises(ValueError, match="live_context_evidence_match changed"):
        validate_reviewed_source_capture_report(wrong_context)

    wrong_rng = copy.deepcopy(report)
    wrong_rng["rng_advance"]["segments"][1]["draw_calls"] -= 1
    with pytest.raises(ValueError, match="segment 1 field draw_calls changed"):
        validate_reviewed_source_capture_report(wrong_rng)

    path.write_text(payload + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="report file SHA-256 changed"):
        load_reviewed_source_capture_report(path)


def test_accepted_profile_loader_pins_file_shape_and_transcript(tmp_path) -> None:
    profile, contract = _accepted_profile()
    evidence = validate_accepted_profile(profile, contract=contract)
    assert len(evidence["timing_token_ids"]) == 20
    assert len(evidence["pre_target_main_token_ids"]) == 27
    assert evidence["target_token_ids"] == [2090, 2091, 2092]

    path = tmp_path / "accepted.profile.json"
    rendered = json.dumps(profile, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")
    file_contract = AcceptedProfileContract(
        **{
            **contract.__dict__,
            "profile_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }
    )
    loaded, loaded_evidence = load_accepted_profile(path, contract=file_contract)
    assert loaded == profile
    assert loaded_evidence["transcript_hashes"] == evidence["transcript_hashes"]

    changed = copy.deepcopy(profile)
    changed["generation"][-1]["prompt_tokens"] = 479
    with pytest.raises(ValueError, match="prompt_tokens changed"):
        validate_accepted_profile(changed, contract=contract)
