"""Capture a reviewed real Lambada prefix contract for the weighted B2 scout.

This is a capture-only gate.  It reconstructs the accepted repeat01 context
from recorded token IDs, advances a private CUDA generator across every prior
sample draw, and then samples the first 42 seq9 tokens through normal FP32
DecodeSession math.  Any token mismatch fails before an H8 timing gate exists.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers

from inference import (
    compile_args,
    get_config,
    load_model_with_server,
    setup_inference_environment,
)
from osuT5.osuT5.dataset.data_utils import (
    BEAT_TYPES,
    TIMING_TYPES,
    events_of_type,
    update_event_times,
)
from osuT5.osuT5.inference import Postprocessor, Preprocessor, Processor
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.direct_decode import DecodeSession
from osuT5.osuT5.inference.optimized.batch.lane_one_token import tensor_sha256
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    ACTIVE_PREFIX_LENGTH,
    BASE_TIMING_BEAT_TOKEN_ID,
    BASE_TIMING_MEASURE_TOKEN_ID,
    BASE_TIMING_TOKENIZER_SCOPE,
    BASE_TIMING_VOCAB_SIZE,
    CAPTURE_PROMPT_LENGTH,
    EXPECTED_FRAME_TIMES_MS,
    EXPECTED_MAIN_PROMPT_TOKENS,
    MAIN_BEAT_TOKEN_ID,
    MAIN_MEASURE_TOKEN_ID,
    MAIN_TOKENIZER_SCOPE,
    MAIN_VOCAB_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_MAX_TARGET_POSITIONS,
    MODEL_SNAPSHOT_REVISION,
    PHASE_A_HORIZON,
    PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
    PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
    PRE_TARGET_RNG_DRAWS,
    SOURCE_AUDIO_SHA256,
    SOURCE_AUDIO_SAMPLES,
    SOURCE_AUDIO_SIZE_BYTES,
    SOURCE_COMMIT,
    SOURCE_GENERATED_TOKENS,
    SOURCE_JOB_ID,
    SOURCE_PREFIX_REPLAY_TOKENS,
    SOURCE_PROFILE_LABEL,
    SOURCE_PROFILE_SHA256,
    SOURCE_PROFILE_SIZE_BYTES,
    SOURCE_PROMPT_LENGTH,
    SOURCE_RESULT_SHA256,
    SOURCE_SEQUENCE_INDEX,
    TARGET_PREFIX_REPLAY_SHA256,
    TARGET_TRANSCRIPT_SHA256,
    TIMING_TRANSCRIPT_SHA256,
    TIMING_TRANSCRIPT_TOKENS,
    WEIGHTED_CONTEXT_REPLAY_GATE,
    WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION,
    WEIGHTED_SOURCE_CONTRACT_GATE,
    WEIGHTED_SOURCE_CONTRACT_SCHEMA_VERSION,
    canonical_json_sha256,
    load_accepted_profile,
    require_reviewed_context_replay,
    validate_context_replay_report,
    validate_reviewed_context_evidence,
    validate_source_contract,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.event import EventType
from osuT5.osuT5.tokenizer import ContextType, Tokenizer
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)
from utils.verify_optimized_b1_lane_capture import (
    _cache_tensor_values,
    _new_generator,
    _sample_next_token,
)
from utils.verify_optimized_hybrid_changing_prefix import _cache_hash
from utils.verify_optimized_hybrid_l2 import (
    _make_factories,
    _validate_fixed_runtime_contract,
)
from osuT5.osuT5.inference.server import get_eos_token_id


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    return {
        "sha256": tensor_sha256(value),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
    }


def _cache_tensor_descriptors(cache_part: Any) -> list[dict[str, Any]]:
    return [
        _tensor_descriptor(value)
        for value in _cache_tensor_values(cache_part)
    ]


def _validate_tokenizer_scopes(
        timing_tokenizer: Tokenizer,
        main_tokenizer: Tokenizer,
) -> dict[str, Any]:
    expected = {
        "timing_scope": BASE_TIMING_TOKENIZER_SCOPE,
        "timing_vocab_size": BASE_TIMING_VOCAB_SIZE,
        "timing_beat_token_id": BASE_TIMING_BEAT_TOKEN_ID,
        "timing_measure_token_id": BASE_TIMING_MEASURE_TOKEN_ID,
        "main_scope": MAIN_TOKENIZER_SCOPE,
        "main_vocab_size": MAIN_VOCAB_SIZE,
        "main_beat_token_id": MAIN_BEAT_TOKEN_ID,
        "main_measure_token_id": MAIN_MEASURE_TOKEN_ID,
    }
    observed = {
        "timing_scope": BASE_TIMING_TOKENIZER_SCOPE,
        "timing_vocab_size": int(timing_tokenizer.vocab_size_out),
        "timing_beat_token_id": int(timing_tokenizer.event_start[EventType.BEAT]),
        "timing_measure_token_id": int(timing_tokenizer.event_start[EventType.MEASURE]),
        "main_scope": MAIN_TOKENIZER_SCOPE,
        "main_vocab_size": int(main_tokenizer.vocab_size_out),
        "main_beat_token_id": int(main_tokenizer.event_start[EventType.BEAT]),
        "main_measure_token_id": int(main_tokenizer.event_start[EventType.MEASURE]),
    }
    if observed != expected:
        raise RuntimeError(
            f"weighted tokenizer scopes changed: observed={observed}, expected={expected}."
        )
    for tokenizer, token_id, event_type, label in (
            (timing_tokenizer, BASE_TIMING_BEAT_TOKEN_ID, EventType.BEAT, "timing beat"),
            (
                timing_tokenizer,
                BASE_TIMING_MEASURE_TOKEN_ID,
                EventType.MEASURE,
                "timing measure",
            ),
            (main_tokenizer, MAIN_BEAT_TOKEN_ID, EventType.BEAT, "main beat"),
            (main_tokenizer, MAIN_MEASURE_TOKEN_ID, EventType.MEASURE, "main measure"),
    ):
        if tokenizer.decode(token_id).type != event_type:
            raise RuntimeError(f"weighted {label} token no longer decodes to {event_type.value}.")
    return observed


def _validate_context_runtime_contract(args: Any, *, config_name: str) -> None:
    """Keep the context prerequisite model-free, CPU-only, and reproducible."""

    if config_name != "profile_salvalai_smoke15":
        raise ValueError(
            "weighted context-only replay requires "
            "config_name=profile_salvalai_smoke15."
        )
    expected = {
        "model_path": "OliBomby/Mapperatorinator-v32",
        "device": "cpu",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "inference_engine": "v32",
        "optimized_inference_mode": "single",
        "gamemode": 0,
        "seed": 12345,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "use_server": False,
        "parallel": False,
        "auto_select_gamemode_model": True,
        "inference_generation_compile": False,
        "inference_active_prefix_decode_loop": False,
        "inference_active_prefix_decode_cuda_graph": False,
        "inference_decode_session_runtime": False,
        "inference_decode_session_cuda_graph": False,
        "inference_native_decode_kernels": False,
        "inference_native_q1_self_attention": False,
        "inference_native_q1_rope_cache_self_attention": False,
    }
    for field, required in expected.items():
        if getattr(args, field, None) != required:
            raise ValueError(
                f"weighted context-only replay requires {field}={required!r}."
            )


def _load_source_evidence(
        source_profile_path: Path,
        source_audio_path: Path,
) -> dict[str, Any]:
    if (
            _file_sha256(source_audio_path) != SOURCE_AUDIO_SHA256
            or source_audio_path.stat().st_size != SOURCE_AUDIO_SIZE_BYTES
    ):
        raise ValueError("accepted Lambada audio SHA-256 changed.")
    if source_profile_path.stat().st_size != SOURCE_PROFILE_SIZE_BYTES:
        raise ValueError("accepted Lambada profile size changed.")
    _, source_evidence = load_accepted_profile(source_profile_path)
    return source_evidence


def _extension_cache_state() -> dict[str, Any]:
    raw_path = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not raw_path:
        return {"path": None, "exists": False, "entry_count": 0, "file_count": 0}
    path = Path(raw_path)
    if not path.exists():
        return {"path": str(path), "exists": False, "entry_count": 0, "file_count": 0}
    return {
        "path": str(path),
        "exists": True,
        "entry_count": sum(1 for _ in path.iterdir()),
        "file_count": sum(1 for value in path.rglob("*") if value.is_file()),
    }


def _find_unfinished_context(out_context: Sequence[Mapping[str, Any]]) -> tuple[int, Any]:
    for index, context in enumerate(out_context):
        if not context.get("finished"):
            return index, context
    raise RuntimeError("accepted source reconstruction found no unfinished output context.")


def _prompt_for_sequence(
        processor: Processor,
        *,
        sequences: Any,
        sequence_index: int,
        in_context: Sequence[Mapping[str, Any]],
        out_context: Sequence[Mapping[str, Any]],
        req_special_tokens: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    frames = processor.prepare_frames(sequences[0][sequence_index])
    frame_time = float(sequences[1][sequence_index].item())
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(
            list(in_context), frame_time, False, list(req_special_tokens)
        ),
        processor.prepare_context_sequences(
            list(out_context), frame_time, True, list(req_special_tokens)
        ),
    )
    if uncond_prompt is not None:
        raise RuntimeError("weighted source reconstruction requires cfg_scale=1.0.")
    [prompt, _], _ = processor.pad_prompts([cond_prompt, uncond_prompt])
    return prompt, frames, frame_time


def _add_profile_tokens(
        processor: Processor,
        context: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        frame_time: float,
) -> None:
    predicted = torch.tensor(record["generated_token_ids"], dtype=torch.long)
    processor.add_predicted_tokens_to_context(
        context,
        predicted,
        frame_time,
        bool(record["trim_lookback"]),
        bool(record["trim_lookahead"]),
    )


def _reconstruct_target_inputs(
        args: Any,
        model: Any | None,
        tokenizer: Tokenizer,
        *,
        timing_tokenizer: Tokenizer,
        source_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_config, _ = get_config(args)
    preprocessor = Preprocessor(args, parallel=False)
    audio = preprocessor.load(args.audio_path)
    if len(audio) != SOURCE_AUDIO_SAMPLES:
        raise ValueError("accepted Lambada decoded audio sample count changed.")
    sequences = preprocessor.segment(audio)
    if len(sequences[0]) != 10:
        raise ValueError("accepted Lambada smoke15 source must contain ten sequences.")
    observed_frame_times = tuple(int(value.item()) for value in sequences[1])
    if observed_frame_times != EXPECTED_FRAME_TIMES_MS:
        raise ValueError("accepted Lambada segmented frame times changed.")

    timing_records = source_evidence["timing_records"]
    timing_processor = Processor(args, None, timing_tokenizer)
    timing_in_types, timing_out_types, timing_special = timing_processor._get_viable_template(
        in_context=[ContextType.NONE],
        out_context=[ContextType.TIMING],
        extra_in_context={},
        gamemode=generation_config.gamemode,
    )
    timing_in = timing_processor.get_in_context(
        in_context=timing_in_types,
        beatmap_path=args.beatmap_path,
        extra_in_context={},
        song_length=sequences[2],
    )
    timing_out = timing_processor.get_out_context(
        out_context=timing_out_types,
        generation_config=generation_config,
        given_context=[ContextType.NONE],
        beatmap_path=args.beatmap_path,
        extra_in_context={},
        song_length=sequences[2],
        verbose=False,
    )
    timing_context_index, timing_context = _find_unfinished_context(timing_out)
    for index, record in enumerate(timing_records):
        prompt, _, frame_time = _prompt_for_sequence(
            timing_processor,
            sequences=sequences,
            sequence_index=index,
            in_context=timing_in,
            out_context=timing_out[:timing_context_index + 1],
            req_special_tokens=timing_special,
        )
        if int(prompt.ne(timing_tokenizer.pad_id).sum().item()) != int(record["prompt_tokens"]):
            raise ValueError(f"reconstructed timing sequence {index} prompt count changed.")
        _add_profile_tokens(
            timing_processor,
            timing_context,
            record,
            frame_time=frame_time,
        )
    timing_pretrim_event_count = len(timing_context["events"])
    if timing_pretrim_event_count != 122:
        raise RuntimeError(
            "base-tokenizer timing replay produced "
            f"{timing_pretrim_event_count} pretrim events; expected 122."
        )
    # Match Processor.generate() post-loop finalization exactly before the
    # timing context is converted into main-generation conditioning.
    timing_context["event_times"] = []
    update_event_times(
        timing_context["events"],
        timing_context["event_times"],
        sequences[2],
        timing_processor.types_first,
    )
    if timing_processor.start_time is not None:
        timing_processor._trim_events_before_time(
            timing_context["events"],
            timing_context["event_times"],
            timing_processor.start_time - 10,
        )
    if timing_processor.end_time is not None:
        timing_processor._trim_events_after_time(
            timing_context["events"],
            timing_context["event_times"],
            timing_processor.end_time + 10,
        )
    timing_events, timing_times = events_of_type(
        timing_context["events"], timing_context["event_times"], TIMING_TYPES
    )
    # Keep both lists materialized: events_of_type checks the accepted replay's
    # time ledger even though generate_timing consumes only events.
    if len(timing_events) != len(timing_times):
        raise RuntimeError("reconstructed timing event/time ledger diverged.")
    type_counts = {
        EventType.TIME_SHIFT: sum(
            event.type == EventType.TIME_SHIFT for event in timing_events
        ),
        EventType.BEAT: sum(event.type == EventType.BEAT for event in timing_events),
        EventType.MEASURE: sum(event.type == EventType.MEASURE for event in timing_events),
    }
    marker_times = [
        int(timing_events[index - 1].value)
        for index, event in enumerate(timing_events)
        if (
            event.type in BEAT_TYPES
            and index > 0
            and timing_events[index - 1].type == EventType.TIME_SHIFT
        )
    ]
    expected_timing_counts = {
        "event_count": 60,
        "time_shift_count": 30,
        "beat_count": 22,
        "measure_count": 8,
        "marker_count": 30,
        "marker_min_time_ms": 71237,
        "marker_max_time_ms": 85875,
    }
    observed_timing_counts = {
        "event_count": len(timing_events),
        "time_shift_count": type_counts[EventType.TIME_SHIFT],
        "beat_count": type_counts[EventType.BEAT],
        "measure_count": type_counts[EventType.MEASURE],
        "marker_count": len(marker_times),
        "marker_min_time_ms": min(marker_times) if marker_times else None,
        "marker_max_time_ms": max(marker_times) if marker_times else None,
    }
    if observed_timing_counts != expected_timing_counts:
        raise RuntimeError(
            "base-tokenizer timing replay ledger changed: "
            f"observed={observed_timing_counts}, expected={expected_timing_counts}."
        )
    timing = Postprocessor(args).generate_timing(timing_events)
    if not timing:
        raise RuntimeError("base-tokenizer timing replay produced no timing points.")

    output_type = args.output_type.copy()
    if ContextType.TIMING in output_type:
        output_type.remove(ContextType.TIMING)
    extra_in_context = {ContextType.TIMING: timing}
    processor = Processor(args, None, tokenizer)
    gen_in_types, gen_out_types, req_special = processor._get_viable_template(
        in_context=args.in_context,
        out_context=output_type,
        extra_in_context=extra_in_context,
        gamemode=generation_config.gamemode,
    )
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
    in_context = processor.get_in_context(
        in_context=gen_in_types,
        beatmap_path=args.beatmap_path,
        extra_in_context=extra_in_context,
        song_length=sequences[2],
    )
    out_context = processor.get_out_context(
        out_context=gen_out_types,
        generation_config=generation_config,
        given_context=args.in_context,
        beatmap_path=args.beatmap_path,
        extra_in_context=extra_in_context,
        song_length=sequences[2],
        verbose=False,
    )
    context_index, context = _find_unfinished_context(out_context)
    main_records = source_evidence["main_records"]
    prompt_token_counts: list[int] = []
    for index, record in enumerate(main_records[:SOURCE_SEQUENCE_INDEX]):
        prompt, _, frame_time = _prompt_for_sequence(
            processor,
            sequences=sequences,
            sequence_index=index,
            in_context=in_context,
            out_context=out_context[:context_index + 1],
            req_special_tokens=req_special,
        )
        prompt_count = int(prompt.ne(tokenizer.pad_id).sum().item())
        prompt_token_counts.append(prompt_count)
        if prompt_count != int(record["prompt_tokens"]):
            raise ValueError(f"reconstructed main sequence {index} prompt count changed.")
        _add_profile_tokens(processor, context, record, frame_time=frame_time)

    prompt, frames, frame_time = _prompt_for_sequence(
        processor,
        sequences=sequences,
        sequence_index=SOURCE_SEQUENCE_INDEX,
        in_context=in_context,
        out_context=out_context[:context_index + 1],
        req_special_tokens=req_special,
    )
    target_record = main_records[SOURCE_SEQUENCE_INDEX]
    nonpad = int(prompt.ne(tokenizer.pad_id).sum().item())
    prompt_token_counts.append(nonpad)
    if nonpad != SOURCE_PROMPT_LENGTH or int(prompt.shape[-1]) != SOURCE_PROMPT_LENGTH:
        raise ValueError(
            f"reconstructed Lambada seq9 prompt must be {SOURCE_PROMPT_LENGTH}; "
            f"got shape={int(prompt.shape[-1])}, nonpad={nonpad}."
        )
    if frame_time != float(target_record["frame_time_ms"]):
        raise ValueError("reconstructed Lambada seq9 frame time changed.")
    if tuple(prompt_token_counts) != EXPECTED_MAIN_PROMPT_TOKENS:
        raise ValueError(
            "reconstructed main prompt ledger changed: "
            f"observed={tuple(prompt_token_counts)}, "
            f"expected={EXPECTED_MAIN_PROMPT_TOKENS}."
        )
    if processor.do_song_position_embed:
        model_kwargs["song_position"] = torch.tensor(
            [
                frame_time / sequences[2],
                (frame_time + processor.miliseconds_per_sequence) / sequences[2],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
    model_inputs = {
        "frames": frames,
        "decoder_input_ids": prompt,
        "decoder_attention_mask": prompt.ne(tokenizer.pad_id),
        **model_kwargs,
    }
    if model is not None:
        model_inputs = _move_kwargs_for_model(model, model_inputs)
    probe = {
        "context_type": context["context_type"].value,
        "sequence_index": SOURCE_SEQUENCE_INDEX,
        "frame_time_ms": frame_time,
        "lookback_time": 0.0,
        "lookahead_time": 0.0,
        "timing_pretrim_event_count": timing_pretrim_event_count,
        "timing_event_count": len(timing_events),
        "timing_time_shift_count": type_counts[EventType.TIME_SHIFT],
        "timing_beat_count": type_counts[EventType.BEAT],
        "timing_measure_count": type_counts[EventType.MEASURE],
        "timing_marker_count": len(marker_times),
        "timing_marker_min_time_ms": min(marker_times),
        "timing_marker_max_time_ms": max(marker_times),
        "timing_point_count": len(timing),
        "main_prompt_token_counts": prompt_token_counts,
    }
    return model_inputs, probe


def _advance_generator_segment(
        generator: torch.Generator,
        *,
        device: torch.device,
        scope: str,
        vocab_size: int,
        draw_calls: int,
) -> dict[str, Any]:
    before = generator.get_state().clone()
    probabilities = torch.full(
        (1, vocab_size),
        1.0 / vocab_size,
        dtype=torch.float32,
        device=device,
    )
    for _ in range(draw_calls):
        torch.multinomial(probabilities, 1, generator=generator)
    torch.cuda.synchronize(device)
    after = generator.get_state().clone()
    return {
        "scope": scope,
        "method": "one CUDA multinomial(num_samples=1) per accepted prior generated token",
        "vocab_size": vocab_size,
        "draw_calls": draw_calls,
        "probability": _tensor_descriptor(probabilities),
        "rng_before": _tensor_descriptor(before),
        "rng_after": _tensor_descriptor(after),
    }


def _build_context_evidence(
        *,
        tokenizer_contract: Mapping[str, Any],
        model_inputs: Mapping[str, Any],
        probe: Mapping[str, Any],
) -> dict[str, Any]:
    condition_tensor_keys = sorted(
        key
        for key, value in _condition_kwargs(model_inputs).items()
        if isinstance(value, torch.Tensor)
    )
    return {
        "source": {
            "job_id": SOURCE_JOB_ID,
            "commit": SOURCE_COMMIT,
            "profile_sha256": SOURCE_PROFILE_SHA256,
            "profile_size_bytes": SOURCE_PROFILE_SIZE_BYTES,
            "audio_sha256": SOURCE_AUDIO_SHA256,
            "audio_size_bytes": SOURCE_AUDIO_SIZE_BYTES,
            "audio_samples": SOURCE_AUDIO_SAMPLES,
            "timing_transcript_sha256": TIMING_TRANSCRIPT_SHA256,
            "pre_target_main_transcript_sha256": PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
            "target_transcript_sha256": TARGET_TRANSCRIPT_SHA256,
        },
        "tokenizers": dict(tokenizer_contract),
        "timing": {
            "pretrim_event_count": probe["timing_pretrim_event_count"],
            "event_count": probe["timing_event_count"],
            "time_shift_count": probe["timing_time_shift_count"],
            "beat_count": probe["timing_beat_count"],
            "measure_count": probe["timing_measure_count"],
            "marker_count": probe["timing_marker_count"],
            "marker_min_time_ms": probe["timing_marker_min_time_ms"],
            "marker_max_time_ms": probe["timing_marker_max_time_ms"],
            "timing_point_count": probe["timing_point_count"],
        },
        "main": {
            "prompt_token_counts": probe["main_prompt_token_counts"],
            "target_sequence_index": probe["sequence_index"],
            "target_frame_time_ms": probe["frame_time_ms"],
            "target_context_type": probe["context_type"],
            "target_prompt": _tensor_descriptor(model_inputs["decoder_input_ids"]),
            "target_prompt_attention_mask": _tensor_descriptor(
                model_inputs["decoder_attention_mask"]
            ),
            "target_frames": _tensor_descriptor(model_inputs["frames"]),
            "condition_tensor_keys": condition_tensor_keys,
        },
    }


def _context_args_for_capture_preflight(args: Any) -> Any:
    """Create the reviewed CPU contract runtime without mutating CUDA args."""

    context_args = copy.deepcopy(args)
    required = {
        "device": "cpu",
        "inference_generation_compile": False,
        "inference_active_prefix_decode_loop": False,
        "inference_active_prefix_decode_cuda_graph": False,
        "inference_stateful_monotonic_logits_processor": False,
        "inference_q1_bmm_cross_attention": False,
        "inference_decode_session_runtime": False,
        "inference_decode_session_cuda_graph": False,
        "inference_native_decode_kernels": False,
        "inference_native_q1_self_attention": False,
        "inference_native_q1_rope_cache_self_attention": False,
    }
    for field, value in required.items():
        setattr(context_args, field, value)
    return context_args


def run_context_replay(
        args: Any,
        *,
        config_name: str,
        source_profile_path: Path,
        source_audio_path: Path,
) -> dict[str, Any]:
    """Reconstruct the accepted CPU context without loading a model or CUDA."""

    source_evidence = _load_source_evidence(source_profile_path, source_audio_path)
    args.audio_path = str(source_audio_path)
    compile_args(args, verbose=False)
    _validate_context_runtime_contract(args, config_name=config_name)
    _assert_supported_probe(args)

    timing_tokenizer = Tokenizer.from_pretrained(args.model_path)
    main_tokenizer = Tokenizer.from_pretrained(
        args.model_path,
        subfolder=f"gamemode={args.gamemode}",
    )
    tokenizer_contract = _validate_tokenizer_scopes(timing_tokenizer, main_tokenizer)
    model_inputs, probe = _reconstruct_target_inputs(
        args,
        None,
        main_tokenizer,
        timing_tokenizer=timing_tokenizer,
        source_evidence=source_evidence,
    )
    tensor_inputs = {
        key: value
        for key, value in model_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    if any(value.device.type != "cpu" for value in tensor_inputs.values()):
        raise RuntimeError("weighted context-only replay created a non-CPU tensor.")
    evidence = _build_context_evidence(
        tokenizer_contract=tokenizer_contract,
        model_inputs=model_inputs,
        probe=probe,
    )
    report = {
        "schema_version": WEIGHTED_CONTEXT_REPLAY_SCHEMA_VERSION,
        "gate": WEIGHTED_CONTEXT_REPLAY_GATE,
        **evidence,
        "maintainer_boundary": {
            "engine": "v32",
            "model_loaded": False,
            "cuda_used": False,
            "gpu_capture_authorized": False,
            "authorization_requires_reviewed_context_sha_commit": True,
        },
    }
    validate_context_replay_report(report)
    return report


@torch.no_grad()
def run_source_capture(
        args: Any,
        *,
        config_name: str,
        source_profile_path: Path,
        source_audio_path: Path,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    # Rebuild the CPU-only prerequisite in this process and compare its complete
    # canonical report before compiling CUDA arguments, seeding CUDA, loading
    # native extensions, or loading the model.  No CLI digest/report override
    # exists.
    started = time.perf_counter()
    fresh_context_report = run_context_replay(
        _context_args_for_capture_preflight(args),
        config_name=config_name,
        source_profile_path=source_profile_path,
        source_audio_path=source_audio_path,
    )
    reviewed_context_report_sha256 = require_reviewed_context_replay(
        fresh_context_report
    )
    context_preflight_wall_seconds = time.perf_counter() - started
    source_evidence = _load_source_evidence(source_profile_path, source_audio_path)
    args.audio_path = str(source_audio_path)
    _validate_fixed_runtime_contract(
        args,
        config_name=config_name,
        q1_bmm_cross_attention=q1_bmm_cross_attention,
        native_q1_self_attention=native_q1_self_attention,
        native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    )
    _assert_supported_probe(args)
    compile_args(args, verbose=False)
    setup_inference_environment(12345)
    extension_before = _extension_cache_state()
    timing_tokenizer = Tokenizer.from_pretrained(args.model_path)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=False,
    )
    model.eval()
    if model.device.type != "cuda" or model.dtype != torch.float32:
        raise RuntimeError("weighted source capture requires a CUDA FP32 model.")
    model_max_target_positions = int(model.config.max_target_positions)
    if model_max_target_positions != MODEL_MAX_TARGET_POSITIONS:
        raise RuntimeError(
            "weighted source model max_target_positions changed: "
            f"observed={model_max_target_positions}, "
            f"expected={MODEL_MAX_TARGET_POSITIONS}."
        )
    tokenizer_contract = _validate_tokenizer_scopes(timing_tokenizer, tokenizer)
    model_inputs, probe = _reconstruct_target_inputs(
        args,
        model,
        tokenizer,
        timing_tokenizer=timing_tokenizer,
        source_evidence=source_evidence,
    )
    live_context_evidence = _build_context_evidence(
        tokenizer_contract=tokenizer_contract,
        model_inputs=model_inputs,
        probe=probe,
    )
    validate_reviewed_context_evidence(live_context_evidence)
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    condition_descriptors = {
        key: _tensor_descriptor(value)
        for key, value in sorted(condition_kwargs.items())
        if isinstance(value, torch.Tensor)
    }
    if condition_descriptors:
        raise RuntimeError(
            "accepted Lambada source unexpectedly produced condition tensor keys: "
            f"{sorted(condition_descriptors)}"
        )
    target_ids = source_evidence["target_token_ids"]
    target_prefix_ids = target_ids[:SOURCE_PREFIX_REPLAY_TOKENS]
    base_factory, _, combined_factory = _make_factories(
        args,
        tokenizer,
        torch.device(model.device),
        lookback_time=0.0,
    )
    del base_factory
    context_type = ContextType(probe["context_type"])
    eos_ids = {
        int(value)
        for value in get_eos_token_id(
            tokenizer,
            lookback_time=float(probe["lookback_time"]),
            lookahead_time=float(probe["lookahead_time"]),
            context_type=context_type,
        )
    }
    if not eos_ids:
        raise RuntimeError("weighted source replay received no EOS token IDs.")

    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        session = DecodeSession.prefill(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_mask,
            frames=frames,
            condition_kwargs=dict(condition_kwargs),
            active_prefix_self_attention=False,
        )
        state = session.one_token_state()
        raw_logits = state.prefill_logits
        if raw_logits is None:
            raise RuntimeError("weighted source replay lost prefill logits.")
        if int(raw_logits.shape[-1]) != MAIN_VOCAB_SIZE:
            raise RuntimeError(
                f"weighted main logits vocab is {int(raw_logits.shape[-1])}; "
                f"expected {MAIN_VOCAB_SIZE}."
            )
        generator = _new_generator(torch.device(model.device), 12345)
        initial_rng_state = generator.get_state().clone()
        timing_rng_advance = _advance_generator_segment(
            generator,
            device=torch.device(model.device),
            scope=BASE_TIMING_TOKENIZER_SCOPE,
            vocab_size=BASE_TIMING_VOCAB_SIZE,
            draw_calls=TIMING_TRANSCRIPT_TOKENS,
        )
        post_timing_rng_state = generator.get_state().clone()
        main_rng_advance = _advance_generator_segment(
            generator,
            device=torch.device(model.device),
            scope=MAIN_TOKENIZER_SCOPE,
            vocab_size=MAIN_VOCAB_SIZE,
            draw_calls=PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
        )
        pre_target_rng_state = generator.get_state().clone()
        processor = combined_factory()
        prefix = prompt.clone(memory_format=torch.contiguous_format)
        mask = prompt_mask.clone(memory_format=torch.contiguous_format)
        replay_steps: list[dict[str, Any]] = []
        stopped = False
        model_decode_calls = 0
        for step, accepted_token in enumerate(target_prefix_ids):
            rng_before = generator.get_state().clone()
            sampled = _sample_next_token(
                input_ids=prefix,
                raw_logits=raw_logits,
                logits_processor=processor,
                generator=generator,
                do_sample=True,
            )
            sampled_token = int(sampled.item())
            stopped = sampled_token in eos_ids
            rng_after = generator.get_state().clone()
            match = sampled_token == int(accepted_token)
            replay_steps.append({
                "step": step,
                "accepted_token": int(accepted_token),
                "sampled_token": sampled_token,
                "match": match,
                "rng_before_sha256": tensor_sha256(rng_before),
                "rng_after_sha256": tensor_sha256(rng_after),
            })
            if not match:
                raise RuntimeError(
                    "accepted Lambada seq9 replay diverged at step "
                    f"{step}: sampled={sampled_token}, accepted={int(accepted_token)}; "
                    "dummy RNG advancement is not sufficient, so H8 is blocked."
                )
            if stopped:
                raise RuntimeError(
                    f"accepted Lambada seq9 replay reached EOS at step {step}; H8 is blocked."
                )
            prefix = torch.cat([prefix, sampled[:, None]], dim=-1)
            mask = torch.cat(
                [mask, torch.ones_like(sampled[:, None], dtype=mask.dtype)], dim=-1
            )
            # Leave the last accepted token unprocessed. H8 begins by running
            # t41 at cache position 519, then processes/samples t42.
            if step < SOURCE_PREFIX_REPLAY_TOKENS - 1:
                cache_position = SOURCE_PROMPT_LENGTH + step
                bucket = _bucketed_prefix_length(
                    cache_position + 1,
                    64,
                    int(model.config.max_target_positions),
                )
                decoded = session.decode_one_token_raw_logits(
                    full_prefix=prefix,
                    full_attention_mask=mask,
                    cache_position=torch.arange(
                        cache_position,
                        cache_position + 1,
                        device=prefix.device,
                    ),
                    active_prefix_self_attention=True,
                    active_prefix_self_attention_length=bucket,
                )
                raw_logits = decoded.logits
                model_decode_calls += 1
        torch.cuda.synchronize(model.device)

    if int(prefix.shape[-1]) != CAPTURE_PROMPT_LENGTH:
        raise RuntimeError("accepted replay did not produce the bucket-576 capture prefix.")
    if _bucketed_prefix_length(
            CAPTURE_PROMPT_LENGTH + 1,
            64,
            int(model.config.max_target_positions),
    ) != ACTIVE_PREFIX_LENGTH:
        raise RuntimeError("accepted replay prefix does not enter active-prefix bucket 576.")
    if CAPTURE_PROMPT_LENGTH + 1 + PHASE_A_HORIZON > ACTIVE_PREFIX_LENGTH:
        raise RuntimeError("weighted H8 would cross the bucket-576 boundary.")
    if SOURCE_PREFIX_REPLAY_TOKENS + PHASE_A_HORIZON > len(target_ids):
        raise RuntimeError("weighted H8 would exceed the accepted seq9 transcript.")
    if CAPTURE_PROMPT_LENGTH + PHASE_A_HORIZON > int(model.config.max_target_positions):
        raise RuntimeError("weighted H8 would exceed model max target positions.")
    cache = session.cache_state.cache
    if cache is None:
        raise RuntimeError("weighted source replay lost its cache.")
    encoder_outputs = session.encoder_state.encoder_outputs
    if encoder_outputs is None:
        raise RuntimeError("weighted source replay lost encoder outputs.")
    if model_decode_calls != SOURCE_PREFIX_REPLAY_TOKENS - 1:
        raise RuntimeError(
            "weighted source replay executed the wrong number of cached decoder calls."
        )
    self_cache_tensors = _cache_tensor_values(cache.self_attention_cache)
    self_cache_max_length = int(
        cache.self_attention_cache.get_max_cache_shape()
    )
    if self_cache_max_length != model_max_target_positions:
        raise RuntimeError(
            "weighted decoder self-cache maximum differs from the model target "
            f"capacity: cache={self_cache_max_length}, "
            f"model={model_max_target_positions}."
        )
    if any(int(value.shape[-2]) != self_cache_max_length for value in self_cache_tensors):
        raise RuntimeError(
            "weighted decoder self-cache tensor sequence dimensions differ from "
            f"the reviewed maximum {self_cache_max_length}."
        )
    position518_populated = all(
        bool(torch.count_nonzero(value[..., 518, :]).item())
        for value in self_cache_tensors
    )
    position519_zero = all(
        not bool(torch.count_nonzero(value[..., 519, :]).item())
        for value in self_cache_tensors
    )
    if not position518_populated or not position519_zero:
        raise RuntimeError(
            "weighted source replay cache is not positioned before accepted t41."
        )
    prepared_cache_position = SOURCE_PROMPT_LENGTH + SOURCE_PREFIX_REPLAY_TOKENS - 1
    prepared = model.prepare_inputs_for_generation(
        prefix,
        past_key_values=cache,
        use_cache=True,
        encoder_outputs=encoder_outputs,
        decoder_attention_mask=mask,
        cache_position=torch.arange(
            prepared_cache_position,
            prepared_cache_position + 1,
            device=prefix.device,
        ),
        **session.condition_kwargs,
    )
    required_static_keys = {
        "cache_position",
        "decoder_attention_mask",
        "decoder_input_ids",
        "decoder_position_ids",
    }
    static_inputs = {
        key: prepared.get(key)
        for key in required_static_keys
    }
    if any(not isinstance(value, torch.Tensor) for value in static_inputs.values()):
        raise RuntimeError("weighted source replay did not prepare every H8 static tensor.")
    if int(static_inputs["cache_position"].item()) != prepared_cache_position:
        raise RuntimeError("weighted source replay prepared the wrong cache position.")
    if int(static_inputs["decoder_input_ids"].item()) != int(target_prefix_ids[-1]):
        raise RuntimeError("weighted source replay did not prepare accepted t41.")
    if int(static_inputs["decoder_position_ids"].item()) != prepared_cache_position:
        raise RuntimeError("weighted source replay prepared the wrong decoder position ID.")
    prepared_attention_mask = static_inputs["decoder_attention_mask"]
    expected_mask_shape = [1, 1, 1, self_cache_max_length]
    if (
            list(prepared_attention_mask.shape) != expected_mask_shape
            or prepared_attention_mask.dtype != torch.float32
    ):
        raise RuntimeError(
            "weighted source replay prepared an unexpected static mask: "
            f"shape={list(prepared_attention_mask.shape)}, "
            f"stride={list(prepared_attention_mask.stride())}, "
            f"dtype={prepared_attention_mask.dtype}, "
            f"sha256={tensor_sha256(prepared_attention_mask)}, "
            f"model_max_target_positions={model_max_target_positions}, "
            f"self_cache_max_length={self_cache_max_length}, "
            f"cache_position={prepared_cache_position}, "
            f"active_prefix_length={ACTIVE_PREFIX_LENGTH}; "
            f"expected_shape={expected_mask_shape}, expected_dtype=torch.float32."
        )
    minimum = torch.finfo(torch.float32).min
    prepared_allowed = prepared_attention_mask[..., :CAPTURE_PROMPT_LENGTH]
    prepared_masked = prepared_attention_mask[..., CAPTURE_PROMPT_LENGTH:]
    prepared_attention_mask_values_exact = bool(
        torch.all(prepared_allowed == 0).item()
        and torch.all(prepared_masked == minimum).item()
    )
    prospective_active_attention_mask = prepared_attention_mask[
        ..., :ACTIVE_PREFIX_LENGTH
    ]
    active_allowed = prospective_active_attention_mask[
        ..., :CAPTURE_PROMPT_LENGTH
    ]
    active_masked = prospective_active_attention_mask[
        ..., CAPTURE_PROMPT_LENGTH:
    ]
    active_attention_mask_values_exact = bool(
        torch.all(active_allowed == 0).item()
        and torch.all(active_masked == minimum).item()
    )
    if (
            list(prospective_active_attention_mask.shape)
            != [1, 1, 1, ACTIVE_PREFIX_LENGTH]
            or not prepared_attention_mask_values_exact
            or not active_attention_mask_values_exact
    ):
        raise RuntimeError(
            "weighted source replay prepared static-mask values that differ "
            "from the reviewed cached q_len=1 semantics: "
            f"prepared_shape={list(prepared_attention_mask.shape)}, "
            f"active_shape={list(prospective_active_attention_mask.shape)}, "
            f"prepared_values_exact={prepared_attention_mask_values_exact}, "
            f"active_values_exact={active_attention_mask_values_exact}."
        )
    post_replay_rng = generator.get_state().clone()
    reconstruction = {
        "source_prompt_length": SOURCE_PROMPT_LENGTH,
        "capture_prompt_length": CAPTURE_PROMPT_LENGTH,
        "active_prefix_length": ACTIVE_PREFIX_LENGTH,
        "phase_a_horizon": PHASE_A_HORIZON,
        "prefix_build": "DecodeSession prompt478 plus exact sampled replay42",
        "prior_context_tokens_source": "accepted_profile",
        "target_prefix_forced_tokens": False,
        "dummy_rng_advance_only_before_target": True,
        "timing_tokenizer_scope": tokenizer_contract["timing_scope"],
        "main_tokenizer_scope": tokenizer_contract["main_scope"],
        "timing_vocab_size": tokenizer_contract["timing_vocab_size"],
        "main_vocab_size": tokenizer_contract["main_vocab_size"],
        "timing_beat_token_id": tokenizer_contract["timing_beat_token_id"],
        "timing_measure_token_id": tokenizer_contract["timing_measure_token_id"],
        "main_beat_token_id": tokenizer_contract["main_beat_token_id"],
        "main_measure_token_id": tokenizer_contract["main_measure_token_id"],
        "timing_pretrim_event_count": probe["timing_pretrim_event_count"],
        "timing_event_count": probe["timing_event_count"],
        "timing_time_shift_count": probe["timing_time_shift_count"],
        "timing_beat_count": probe["timing_beat_count"],
        "timing_measure_count": probe["timing_measure_count"],
        "timing_marker_count": probe["timing_marker_count"],
        "timing_marker_min_time_ms": probe["timing_marker_min_time_ms"],
        "timing_marker_max_time_ms": probe["timing_marker_max_time_ms"],
        "timing_point_count": probe["timing_point_count"],
        "model_max_target_positions": model_max_target_positions,
        "self_cache_max_length": self_cache_max_length,
        "rng_seed": 12345,
        "rng_advance_method": timing_rng_advance["method"],
        "timing_rng_advance_draw_calls": timing_rng_advance["draw_calls"],
        "main_rng_advance_draw_calls": main_rng_advance["draw_calls"],
        "cache_populated_through_position": 518,
        "prepared_cache_position": prepared_cache_position,
        "prepared_decoder_token_id": int(target_prefix_ids[-1]),
        "next_output_position": SOURCE_PREFIX_REPLAY_TOKENS,
        "processor_calls": SOURCE_PREFIX_REPLAY_TOKENS,
        "model_decode_calls_after_prefill": model_decode_calls,
        "self_cache_position518_populated": position518_populated,
        "self_cache_position519_zero": position519_zero,
        "prepared_attention_mask_allowed_positions": CAPTURE_PROMPT_LENGTH,
        "prepared_attention_mask_masked_positions": (
            self_cache_max_length - CAPTURE_PROMPT_LENGTH
        ),
        "active_attention_mask_allowed_positions": CAPTURE_PROMPT_LENGTH,
        "active_attention_mask_masked_positions": (
            ACTIVE_PREFIX_LENGTH - CAPTURE_PROMPT_LENGTH
        ),
        "prepared_attention_mask_values_exact": (
            prepared_attention_mask_values_exact
        ),
        "active_attention_mask_values_exact": active_attention_mask_values_exact,
        "source_prompt": _tensor_descriptor(prompt),
        "source_prompt_attention_mask": _tensor_descriptor(prompt_mask),
        "frames": _tensor_descriptor(frames),
        "condition_tensors": condition_descriptors,
        "capture_prompt": _tensor_descriptor(prefix),
        "capture_prompt_attention_mask": _tensor_descriptor(mask),
        "initial_rng_state": _tensor_descriptor(initial_rng_state),
        "post_timing_rng_state": _tensor_descriptor(post_timing_rng_state),
        "pre_target_rng_state": _tensor_descriptor(pre_target_rng_state),
        "post_replay_rng_state": _tensor_descriptor(post_replay_rng),
        "pre_last_sample_raw_logits": _tensor_descriptor(raw_logits),
        "timing_rng_advance_probability": timing_rng_advance["probability"],
        "main_rng_advance_probability": main_rng_advance["probability"],
        "prepared_static_inputs": {
            key: _tensor_descriptor(value)
            for key, value in sorted(static_inputs.items())
        },
        "prospective_active_attention_mask": _tensor_descriptor(
            prospective_active_attention_mask
        ),
        "pre_next_forward_cache": {
            "self_sha256": _cache_hash(cache.self_attention_cache),
            "cross_sha256": _cache_hash(cache.cross_attention_cache),
            "self_tensors": _cache_tensor_descriptors(cache.self_attention_cache),
            "cross_tensors": _cache_tensor_descriptors(cache.cross_attention_cache),
        },
    }
    report = {
        "schema_version": WEIGHTED_SOURCE_CONTRACT_SCHEMA_VERSION,
        "gate": WEIGHTED_SOURCE_CONTRACT_GATE,
        "source": {
            "job_id": SOURCE_JOB_ID,
            "commit": SOURCE_COMMIT,
            "profile_sha256": SOURCE_PROFILE_SHA256,
            "profile_size_bytes": SOURCE_PROFILE_SIZE_BYTES,
            "audio_sha256": SOURCE_AUDIO_SHA256,
            "audio_size_bytes": SOURCE_AUDIO_SIZE_BYTES,
            "audio_samples": SOURCE_AUDIO_SAMPLES,
            "result_sha256": SOURCE_RESULT_SHA256,
            "profile_label": SOURCE_PROFILE_LABEL,
            "sequence_index": SOURCE_SEQUENCE_INDEX,
            "source_prompt_length": SOURCE_PROMPT_LENGTH,
            "source_generated_tokens": SOURCE_GENERATED_TOKENS,
            "model_snapshot_revision": MODEL_SNAPSHOT_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
        },
        "transcript": {
            "timing_tokens": TIMING_TRANSCRIPT_TOKENS,
            "timing_sha256": TIMING_TRANSCRIPT_SHA256,
            "pre_target_main_tokens": PRE_TARGET_MAIN_TRANSCRIPT_TOKENS,
            "pre_target_main_sha256": PRE_TARGET_MAIN_TRANSCRIPT_SHA256,
            "pre_target_rng_draws": PRE_TARGET_RNG_DRAWS,
            "target_tokens": SOURCE_GENERATED_TOKENS,
            "target_sha256": TARGET_TRANSCRIPT_SHA256,
            "target_prefix_replay_tokens": SOURCE_PREFIX_REPLAY_TOKENS,
            "target_prefix_replay_sha256": TARGET_PREFIX_REPLAY_SHA256,
        },
        "reconstruction": reconstruction,
        "accepted_prefix_replay": {
            "committed_tokens": SOURCE_PREFIX_REPLAY_TOKENS,
            "all_tokens_match": True,
            "forced_tokens": False,
            "stopped": stopped,
            "accepted_token_ids_sha256": TARGET_PREFIX_REPLAY_SHA256,
            "sampled_token_ids_sha256": TARGET_PREFIX_REPLAY_SHA256,
            "steps": replay_steps,
        },
        "maintainer_boundary": {
            "engine": "v32",
            "default_behavior_changed": False,
            "runtime_wiring_added": False,
            "phase_a_authorized": False,
            "authorization_requires_reviewed_contract_sha_commit": True,
        },
        "rng_advance": {
            "segments": [timing_rng_advance, main_rng_advance],
            "initial_rng_state": _tensor_descriptor(initial_rng_state),
            "post_timing_rng_state": _tensor_descriptor(post_timing_rng_state),
            "pre_target_rng_state": _tensor_descriptor(pre_target_rng_state),
        },
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "config_name": config_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(model.device),
            "gpu_capability": list(torch.cuda.get_device_capability(model.device)),
            "probe": probe,
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "torch_extensions_cache_before": extension_before,
            "torch_extensions_cache_after": _extension_cache_state(),
            "total_wall_seconds": time.perf_counter() - started,
            "capture_only": True,
            "h8_executed": False,
            "scheduler_or_runtime_wiring_authorized": False,
            "reviewed_context_report_sha256": reviewed_context_report_sha256,
            "context_preflight_wall_seconds": context_preflight_wall_seconds,
            "fresh_context_replay_match": True,
            "live_context_evidence_match": True,
        },
    }
    # The strict source contract intentionally validates only immutable source,
    # transcript, tensor, RNG, and replay evidence. Runtime metadata is removed
    # from the separately reviewed contract so reruns can compare it exactly.
    contract = {
        key: value
        for key, value in report.items()
        if key != "runtime_metadata" and key != "rng_advance"
    }
    validate_source_contract(contract)
    report["source_contract_sha256"] = canonical_json_sha256(contract)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="reconstruct tokenizer/context state on CPU without loading a model or CUDA",
    )
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    args = _load_args(cli.config_name, cli.overrides)
    if cli.context_only:
        if any((
                cli.q1_bmm_cross_attention,
                cli.native_q1_self_attention,
                cli.native_q1_rope_cache_self_attention,
        )):
            raise ValueError("weighted context-only replay rejects GPU-kernel CLI flags.")
        report = run_context_replay(
            args,
            config_name=cli.config_name,
            source_profile_path=cli.source_profile,
            source_audio_path=cli.source_audio,
        )
    else:
        report = run_source_capture(
            args,
            config_name=cli.config_name,
            source_profile_path=cli.source_profile,
            source_audio_path=cli.source_audio,
            q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
            native_q1_self_attention=cli.native_q1_self_attention,
            native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
        )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
