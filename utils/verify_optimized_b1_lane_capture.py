"""Run the first isolated B1 CUDA-graph lane parity/capture gate.

Only one lane is implemented.  L=2-4 must remain fail-loud until this report
passes on the target RTX 2080/2080 Ti.  The utility does not wire a scheduler,
server, or production generation path.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
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
from transformers import TopKLogitsWarper, TopPLogitsWarper

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.direct_decode import DecodeSession
from osuT5.osuT5.inference.optimized.batch.lane_one_token import (
    DEFAULT_SEPARATE_GRAPH_POOL,
    PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
    LaneResourceEvidence,
    OneLaneCaptureConfig,
    capture_prepared_b1_lane,
    compare_lane_logits,
    normalized_one_token_request_ids,
    normalized_one_token_workload_contract,
    tensor_sha256,
    time_b1_lane_replay,
    validate_lane_gate_order,
    validate_lane_resource_ownership,
)
from osuT5.osuT5.inference.optimized.benchmark import (
    BatchPhysicsExecutionFamily,
    BatchPhysicsObservation,
    LANE_STATE_OWNERSHIP_CONTRACT,
)
from osuT5.osuT5.inference.optimized.exactness import ExactnessResultClass
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _json_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _new_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _sample_next_token(
        *,
        input_ids: torch.LongTensor,
        raw_logits: torch.Tensor,
        logits_processor: Any,
        generator: torch.Generator,
        do_sample: bool,
) -> torch.LongTensor:
    scores = logits_processor(
        input_ids,
        raw_logits.clone(memory_format=torch.contiguous_format),
    )
    if do_sample:
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(1)
    return torch.argmax(scores, dim=-1)


def _tensor_storage_ptr(value: torch.Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


def _top_level_tensor_ptrs(values: Mapping[str, Any]) -> dict[str, int]:
    return {
        key: _tensor_storage_ptr(value)
        for key, value in sorted(values.items())
        if isinstance(value, torch.Tensor)
    }


def _cache_tensor_values(cache_part: Any) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            pointer = _tensor_storage_ptr(value)
            if pointer not in seen:
                seen.add(pointer)
                values.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    add(getattr(cache_part, "key_cache", None))
    add(getattr(cache_part, "value_cache", None))
    for layer in getattr(cache_part, "layers", ()) or ():
        for name in ("keys", "values", "key_cache", "value_cache"):
            add(getattr(layer, name, None))
    if not values:
        raise RuntimeError(
            f"could not locate cache tensors in {type(cache_part).__name__}; "
            "do not claim lane-private cache ownership."
        )
    return values


def _cache_tensor_ptrs(cache_part: Any) -> tuple[int, ...]:
    return tuple(sorted(_tensor_storage_ptr(value) for value in _cache_tensor_values(cache_part)))


def _cache_position_views(cache_part: Any, position: int) -> list[torch.Tensor]:
    if position < 0:
        raise ValueError("cache position must be non-negative.")
    views: list[torch.Tensor] = []
    for tensor in _cache_tensor_values(cache_part):
        if tensor.ndim < 2 or position >= tensor.shape[-2]:
            raise ValueError(
                f"cache tensor shape {list(tensor.shape)} does not contain position {position}."
            )
        views.append(tensor[..., position, :])
    return views


def _zero_cache_position(cache_part: Any, position: int) -> None:
    for view in _cache_position_views(cache_part, position):
        view.zero_()


def _cache_position_is_zero(cache_part: Any, position: int) -> bool:
    return all(
        bool(torch.count_nonzero(view).detach().cpu().item() == 0)
        for view in _cache_position_views(cache_part, position)
    )


def _cache_position_is_overwritten(cache_part: Any, position: int) -> bool:
    return all(
        bool(torch.count_nonzero(view).detach().cpu().item() > 0)
        for view in _cache_position_views(cache_part, position)
    )


def _cache_part_contract(cache_part: Any) -> dict[str, Any]:
    max_cache_shape = None
    if hasattr(cache_part, "get_max_cache_shape"):
        max_cache_shape = int(cache_part.get_max_cache_shape())
    return {
        "type": type(cache_part).__name__,
        "max_cache_shape": max_cache_shape,
        "tensor_shapes": [
            list(tensor.shape)
            for tensor in _cache_tensor_values(cache_part)
        ],
    }


def _compare_cache_parts(
        reference: Any,
        candidate: Any,
        *,
        atol: float,
        rtol: float,
) -> dict[str, Any]:
    reference_tensors = _cache_tensor_values(reference)
    candidate_tensors = _cache_tensor_values(candidate)
    if len(reference_tensors) != len(candidate_tensors):
        return {
            "pass": False,
            "tensor_count_match": False,
            "reference_tensor_count": len(reference_tensors),
            "candidate_tensor_count": len(candidate_tensors),
        }
    tensors = []
    for index, (reference_tensor, candidate_tensor) in enumerate(
            zip(reference_tensors, candidate_tensors)
    ):
        shape_match = reference_tensor.shape == candidate_tensor.shape
        allclose = bool(
            shape_match
            and torch.allclose(reference_tensor, candidate_tensor, atol=atol, rtol=rtol)
        )
        bitwise = bool(shape_match and torch.equal(reference_tensor, candidate_tensor))
        difference = (
            torch.abs(reference_tensor - candidate_tensor)
            if shape_match
            else None
        )
        tensors.append({
            "index": index,
            "shape_match": shape_match,
            "reference_shape": list(reference_tensor.shape),
            "candidate_shape": list(candidate_tensor.shape),
            "allclose": allclose,
            "bitwise": bitwise,
            "max_abs": (
                float(difference.max().item())
                if difference is not None and difference.numel()
                else 0.0
            ),
            "reference_sha256": tensor_sha256(reference_tensor),
            "candidate_sha256": tensor_sha256(candidate_tensor),
        })
    return {
        "pass": all(item["allclose"] for item in tensors),
        "bitwise": all(item["bitwise"] for item in tensors),
        "tensor_count_match": True,
        "reference_tensor_count": len(reference_tensors),
        "candidate_tensor_count": len(candidate_tensors),
        "tensors": tensors,
    }


def _load_previous_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("previous lane report must contain a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--lane-count", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--previous-gate-report", type=Path)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--capture-warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=50)
    parser.add_argument("--active-prefix-bucket-size", type=int, default=64)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as audio_path=/path/song.mp3")
    return parser


@torch.no_grad()
def run_one_lane_gate(
        args: Any,
        *,
        config_name: str,
        sequence_index: int,
        seed: int,
        atol: float,
        rtol: float,
        comparison_top_k: int,
        capture_warmup_repeats: int,
        timing_repeats: int,
        active_prefix_bucket_size: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    """Build an eager reference and one separately owned captured B1 lane."""

    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("the B1 lane gate is FP32-only.")
    if args.inference_generation_compile:
        raise ValueError(
            "the manual B1 lane graph gate requires inference_generation_compile=false; "
            "do not stack two graph owners in the first parity gate."
        )
    if native_q1_rope_cache_self_attention and not native_q1_self_attention:
        raise ValueError("RoPE/cache native self-attention requires native q1 self-attention.")
    compile_args(args, verbose=False)
    setup_inference_environment(seed)
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
    if model.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the B1 lane gate requires a CUDA GPU allocation.")
    if model.dtype != torch.float32:
        raise ValueError(f"the B1 lane gate requires FP32 model weights; got {model.dtype}.")

    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    probe_metadata = {
        key: model_inputs.pop(key)
        for key in ("sequence_index", "frame_time_ms", "context_type", "lookback_time", "lookahead_time")
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_attention_mask = model_inputs["decoder_attention_mask"]
    frames = model_inputs["frames"]
    condition_kwargs = _condition_kwargs(model_inputs)
    active_prefix_length = _bucketed_prefix_length(
        int(prompt.shape[-1]) + 1,
        int(active_prefix_bucket_size),
        int(model.config.max_target_positions),
    )
    gate_config = OneLaneCaptureConfig(
        seed=seed,
        active_prefix_length=active_prefix_length,
        atol=atol,
        rtol=rtol,
        top_k=comparison_top_k,
        capture_warmup_repeats=capture_warmup_repeats,
        timing_repeats=timing_repeats,
    )

    context_type = ContextType(probe_metadata["context_type"])
    eos_token_ids = [
        int(token_id)
        for token_id in get_eos_token_id(
            tokenizer,
            lookback_time=float(probe_metadata["lookback_time"]),
            lookahead_time=float(probe_metadata["lookahead_time"]),
            context_type=context_type,
        )
    ]

    def logits_processor_factory():
        processors = _build_generation_logits_processors(
            args,
            tokenizer,
            model.device,
            lookback_time=float(probe_metadata["lookback_time"]),
        )
        if args.do_sample:
            if args.top_k > 0:
                processors.append(TopKLogitsWarper(top_k=int(args.top_k)))
            if 0.0 < args.top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p=float(args.top_p)))
        return processors

    device = torch.device(model.device)
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=q1_bmm_cross_attention,
            native_q1_self_attention=native_q1_self_attention,
            native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
    ):
        reference_processor = logits_processor_factory()
        reference_generator = _new_generator(device, seed)
        reference_session = DecodeSession.prefill(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=False,
        )
        reference_prefill_logits = reference_session.one_token_state().prefill_logits
        if reference_prefill_logits is None:
            raise RuntimeError("reference DecodeSession did not retain prefill logits.")
        reference_anchor = _sample_next_token(
            input_ids=prompt,
            raw_logits=reference_prefill_logits,
            logits_processor=reference_processor,
            generator=reference_generator,
            do_sample=bool(args.do_sample),
        )
        reference_full_prefix = torch.cat([prompt, reference_anchor[:, None]], dim=-1)
        reference_full_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones_like(reference_anchor[:, None], dtype=prompt_attention_mask.dtype),
            ],
            dim=-1,
        )
        reference_decode = reference_session.decode_one_token_raw_logits(
            full_prefix=reference_full_prefix,
            full_attention_mask=reference_full_mask,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        reference_next = _sample_next_token(
            input_ids=reference_full_prefix,
            raw_logits=reference_decode.logits,
            logits_processor=reference_processor,
            generator=reference_generator,
            do_sample=bool(args.do_sample),
        )
        reference_rng_hash = tensor_sha256(reference_generator.get_state())

        lane_processor = logits_processor_factory()
        lane_generator = _new_generator(device, seed)
        lane_session = DecodeSession.prefill(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=frames,
            condition_kwargs=condition_kwargs,
            active_prefix_self_attention=False,
        )
        lane_prefill_logits = lane_session.one_token_state().prefill_logits
        if lane_prefill_logits is None:
            raise RuntimeError("lane DecodeSession did not retain prefill logits.")
        lane_anchor = _sample_next_token(
            input_ids=prompt,
            raw_logits=lane_prefill_logits,
            logits_processor=lane_processor,
            generator=lane_generator,
            do_sample=bool(args.do_sample),
        )
        lane_full_prefix = torch.cat([prompt, lane_anchor[:, None]], dim=-1)
        lane_full_mask = torch.cat(
            [
                prompt_attention_mask,
                torch.ones_like(lane_anchor[:, None], dtype=prompt_attention_mask.dtype),
            ],
            dim=-1,
        )
        lane_setup_decode = lane_session.decode_one_token_raw_logits(
            full_prefix=lane_full_prefix,
            full_attention_mask=lane_full_mask,
            active_prefix_self_attention=True,
            active_prefix_self_attention_length=active_prefix_length,
        )
        torch.cuda.reset_peak_memory_stats(device)
        lane = capture_prepared_b1_lane(
            model,
            lane_setup_decode.prepared_inputs,
            active_prefix_length=active_prefix_length,
            capture_warmup_repeats=gate_config.capture_warmup_repeats,
        )
        if lane_session.cache_state.cache is None:
            raise RuntimeError("lane session lost cache ownership before graph replay.")
        replay_cache_position = int(lane_setup_decode.cache_position[-1].detach().cpu().item())
        with torch.cuda.stream(lane.stream):
            _zero_cache_position(
                lane_session.cache_state.cache.self_attention_cache,
                replay_cache_position,
            )
        lane.stream.synchronize()
        cache_sentinel_zeroed = _cache_position_is_zero(
            lane_session.cache_state.cache.self_attention_cache,
            replay_cache_position,
        )
        lane.replay()
        lane.stream.synchronize()
        cache_sentinel_overwritten = _cache_position_is_overwritten(
            lane_session.cache_state.cache.self_attention_cache,
            replay_cache_position,
        )
        lane_logits = lane.logits
        with torch.cuda.stream(lane.stream):
            lane_next = _sample_next_token(
                input_ids=lane_full_prefix,
                raw_logits=lane_logits,
                logits_processor=lane_processor,
                generator=lane_generator,
                do_sample=bool(args.do_sample),
            )
        lane.stream.synchronize()
        lane_rng_hash = tensor_sha256(lane_generator.get_state())

        prefill_comparison = compare_lane_logits(
            reference_prefill_logits,
            lane_prefill_logits,
            atol=atol,
            rtol=rtol,
            top_k=comparison_top_k,
        )
        setup_comparison = compare_lane_logits(
            reference_decode.logits,
            lane_setup_decode.logits,
            atol=atol,
            rtol=rtol,
            top_k=comparison_top_k,
        )
        replay_comparison = compare_lane_logits(
            reference_decode.logits,
            lane_logits,
            atol=atol,
            rtol=rtol,
            top_k=comparison_top_k,
        )
        reference_cache = reference_session.cache_state.cache
        lane_cache = lane_session.cache_state.cache
        encoder_outputs = lane_session.encoder_state.encoder_outputs
        if reference_cache is None or lane_cache is None or encoder_outputs is None:
            raise RuntimeError("reference/lane session lost cache or encoder ownership.")
        self_cache_comparison = _compare_cache_parts(
            reference_cache.self_attention_cache,
            lane_cache.self_attention_cache,
            atol=atol,
            rtol=rtol,
        )
        cross_cache_comparison = _compare_cache_parts(
            reference_cache.cross_attention_cache,
            lane_cache.cross_attention_cache,
            atol=atol,
            rtol=rtol,
        )
        cache_position_match = bool(torch.equal(
            reference_decode.cache_position,
            lane_setup_decode.cache_position,
        ))
        reference_cache_contract = {
            "self_attention": _cache_part_contract(
                reference_cache.self_attention_cache
            ),
            "cross_attention": _cache_part_contract(
                reference_cache.cross_attention_cache
            ),
        }
        lane_cache_contract = {
            "self_attention": _cache_part_contract(
                lane_cache.self_attention_cache
            ),
            "cross_attention": _cache_part_contract(
                lane_cache.cross_attention_cache
            ),
        }
        cache_contract_match = reference_cache_contract == lane_cache_contract

        graph_pool_id = repr(lane.graph.pool())
        stream_id = int(lane.stream.cuda_stream)
        ownership = LaneResourceEvidence(
            lane_id=0,
            shared_model_id=id(model),
            stream_id=stream_id,
            graph_id=id(lane.graph),
            graph_pool_id=graph_pool_id,
            graph_pool_policy=DEFAULT_SEPARATE_GRAPH_POOL,
            graph_pool_sharing_requested=False,
            session_id=id(lane_session),
            cache_id=id(lane_cache),
            encoder_output_storage_ptr=_tensor_storage_ptr(encoder_outputs.last_hidden_state),
            generator_id=id(lane_generator),
            logits_processor_id=id(lane_processor),
            static_input_storage_ptrs=_top_level_tensor_ptrs(lane.static_inputs),
            self_cache_storage_ptrs=_cache_tensor_ptrs(lane_cache.self_attention_cache),
            cross_cache_storage_ptrs=_cache_tensor_ptrs(lane_cache.cross_attention_cache),
            cublas_workspace_policy=PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
            cublas_workspace_owner_stream_id=stream_id,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
        validate_lane_resource_ownership([ownership], expected_lane_count=1)

        reference_cache_pointers = set(
            _cache_tensor_ptrs(reference_cache.self_attention_cache)
        ) | set(_cache_tensor_ptrs(reference_cache.cross_attention_cache))
        lane_cache_pointers = set(ownership.self_cache_storage_ptrs) | set(
            ownership.cross_cache_storage_ptrs
        )
        reference_lane_cache_storage_disjoint = not bool(
            reference_cache_pointers & lane_cache_pointers
        )
        exactness_pass = bool(
            prefill_comparison["allclose"]
            and prefill_comparison["topk_match"]
            and setup_comparison["allclose"]
            and setup_comparison["topk_match"]
            and replay_comparison["allclose"]
            and replay_comparison["topk_match"]
            and torch.equal(reference_anchor, lane_anchor)
            and torch.equal(reference_next, lane_next)
            and reference_rng_hash == lane_rng_hash
            and cache_position_match
            and cache_contract_match
            and cache_sentinel_zeroed
            and cache_sentinel_overwritten
            and self_cache_comparison["pass"]
            and cross_cache_comparison["pass"]
        )
        resource_ownership_pass = reference_lane_cache_storage_disjoint
        capture_pass = bool(
            lane.replay_count == 1
            and graph_pool_id
            and cache_sentinel_zeroed
            and cache_sentinel_overwritten
        )
        timing = (
            time_b1_lane_replay(lane, timing_repeats=gate_config.timing_repeats)
            if exactness_pass and resource_ownership_pass and capture_pass
            else None
        )

    one_step_exactness_pass = exactness_pass
    timing_exactness_pass: bool | None = None
    reference_complete_tokens: list[int] = []
    candidate_complete_tokens: list[int] = []
    reference_complete_rng_hash: str | None = None
    candidate_complete_rng_hash: str | None = None
    workload_contract: dict[str, Any] | None = None
    observation = None
    if timing is not None:
        reference_complete_processor = logits_processor_factory()
        reference_complete_generator = _new_generator(device, seed)
        reference_complete_token_tensors: list[torch.Tensor] = []
        with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=native_q1_self_attention,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
        ):
            for _ in range(gate_config.timing_repeats):
                reference_complete_result = reference_session.decode_one_token_raw_logits(
                    full_prefix=reference_full_prefix,
                    full_attention_mask=reference_full_mask,
                    active_prefix_self_attention=True,
                    active_prefix_self_attention_length=active_prefix_length,
                )
                reference_complete_token_tensors.append(_sample_next_token(
                    input_ids=reference_full_prefix,
                    raw_logits=reference_complete_result.logits,
                    logits_processor=reference_complete_processor,
                    generator=reference_complete_generator,
                    do_sample=bool(args.do_sample),
                ))
        torch.cuda.synchronize(device)
        reference_complete_tokens = [
            int(token.detach().cpu().item())
            for token in reference_complete_token_tensors
        ]
        reference_complete_rng_hash = tensor_sha256(
            reference_complete_generator.get_state()
        )

        complete_processor = logits_processor_factory()
        complete_generator = _new_generator(device, seed)
        candidate_complete_token_tensors: list[torch.Tensor] = []
        complete_start = torch.cuda.Event(enable_timing=True)
        complete_end = torch.cuda.Event(enable_timing=True)
        complete_wall_started = time.perf_counter()
        with torch.cuda.stream(lane.stream):
            complete_start.record(lane.stream)
            for _ in range(gate_config.timing_repeats):
                lane.graph.replay()
                candidate_complete_token_tensors.append(_sample_next_token(
                    input_ids=lane_full_prefix,
                    raw_logits=lane.logits,
                    logits_processor=complete_processor,
                    generator=complete_generator,
                    do_sample=bool(args.do_sample),
                ))
            complete_end.record(lane.stream)
        lane.replay_count += gate_config.timing_repeats
        complete_end.synchronize()
        complete_wall_seconds = time.perf_counter() - complete_wall_started
        complete_cuda_seconds = complete_start.elapsed_time(complete_end) / 1000.0
        candidate_complete_tokens = [
            int(token.detach().cpu().item())
            for token in candidate_complete_token_tensors
        ]
        candidate_complete_rng_hash = tensor_sha256(complete_generator.get_state())
        timing_exactness_pass = bool(
            len(candidate_complete_tokens) == gate_config.timing_repeats
            and reference_complete_tokens == candidate_complete_tokens
            and reference_complete_rng_hash == candidate_complete_rng_hash
        )
        timing.update({
            "complete_sampled_step_wall_seconds": complete_wall_seconds,
            "complete_sampled_step_cuda_seconds": complete_cuda_seconds,
            "complete_wall_tokens_per_second": (
                gate_config.timing_repeats / complete_wall_seconds
            ),
            "complete_cuda_tokens_per_second": (
                gate_config.timing_repeats / complete_cuda_seconds
            ),
        })
        workload_contract = normalized_one_token_workload_contract(
            batch_size=1,
            seeds=[seed],
            do_sample=bool(args.do_sample),
            prompt_sha256=tensor_sha256(prompt),
            prompt_attention_mask_sha256=tensor_sha256(prompt_attention_mask),
            frames_sha256=tensor_sha256(frames),
            condition_tensor_hashes={
                key: tensor_sha256(value)
                for key, value in sorted(condition_kwargs.items())
                if isinstance(value, torch.Tensor)
            },
            active_prefix_prefill=False,
            active_prefix_decode=True,
            active_prefix_decode_length=active_prefix_length,
            runtime_contract={
                "config_name": config_name,
                "model_path": args.model_path,
                "precision": args.precision,
                "attn_implementation": args.attn_implementation,
                "global_seed": seed,
                "row_seeds": [seed],
                "do_sample": bool(args.do_sample),
                "top_p": float(args.top_p),
                "top_k_sampling": int(args.top_k),
                "temperature": float(args.temperature),
                "stateful_monotonic_logits_processor": bool(
                    args.inference_stateful_monotonic_logits_processor
                ),
                "eos_token_ids": eos_token_ids,
                "probe": probe_metadata,
            },
        )
        if timing_exactness_pass:
            request_id = normalized_one_token_request_ids(1)[0]
            observation = BatchPhysicsObservation(
                execution_family=BatchPhysicsExecutionFamily.B1_LANE_POOL,
                parallelism=1,
                state_ownership_contract=LANE_STATE_OWNERSHIP_CONTRACT,
                workload_contract_hash=_json_sha256(workload_contract),
                result_class=ExactnessResultClass.EXACT_OUTPUT,
                seeds={request_id: seed},
                generated_tokens=gate_config.timing_repeats,
                scheduler_wall_seconds=complete_wall_seconds,
                model_seconds=float(timing["model_only_cuda_seconds"]),
                cuda_seconds=complete_cuda_seconds,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                graph_capture_count=1,
                graph_replay_count=gate_config.timing_repeats,
                active_batch_size_histogram={1: gate_config.timing_repeats},
                token_hashes={request_id: _json_sha256(candidate_complete_tokens)},
                final_rng_state_hashes={request_id: candidate_complete_rng_hash},
                stop_reasons={request_id: "fixed_shape_repeats_complete"},
            ).as_dict()

    exactness_pass = bool(
        one_step_exactness_pass
        and timing_exactness_pass is True
    )

    device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    report = {
        "pass": bool(exactness_pass and resource_ownership_pass and capture_pass),
        "lane_count": 1,
        "gate": "one_lane_identical_prompt_one_token_capture",
        "claim_scope": "parity_capture_and_fixed_shape_scout_not_runtime_throughput",
        "gate_config": {
            "seed": gate_config.seed,
            "active_prefix_length": gate_config.active_prefix_length,
            "atol": gate_config.atol,
            "rtol": gate_config.rtol,
            "top_k": gate_config.top_k,
            "capture_warmup_repeats": gate_config.capture_warmup_repeats,
            "timing_repeats": gate_config.timing_repeats,
        },
        "exactness_pass": exactness_pass,
        "one_step_exactness_pass": one_step_exactness_pass,
        "timing_exactness_pass": timing_exactness_pass,
        "resource_ownership_pass": resource_ownership_pass,
        "capture_pass": capture_pass,
        "performance_gate_pass": None,
        "anchor_token_match": bool(torch.equal(reference_anchor, lane_anchor)),
        "sampled_token_match": bool(torch.equal(reference_next, lane_next)),
        "rng_state_match": reference_rng_hash == lane_rng_hash,
        "reference_final_rng_state_hash": reference_rng_hash,
        "lane_final_rng_state_hash": lane_rng_hash,
        "prefill_logits": prefill_comparison,
        "setup_eager_logits": setup_comparison,
        "graph_replay_logits": replay_comparison,
        "self_cache": self_cache_comparison,
        "cross_cache": cross_cache_comparison,
        "cache_replay_sentinel": {
            "position": replay_cache_position,
            "zeroed_before_replay": cache_sentinel_zeroed,
            "overwritten_by_replay": cache_sentinel_overwritten,
            "cache_position_match": cache_position_match,
            "cache_contract_match": cache_contract_match,
            "reference_contract": reference_cache_contract,
            "lane_contract": lane_cache_contract,
        },
        "complete_timing_exactness": {
            "reference_token_ids": reference_complete_tokens,
            "candidate_token_ids": candidate_complete_tokens,
            "token_hash_match": (
                (
                    _json_sha256(reference_complete_tokens)
                    == _json_sha256(candidate_complete_tokens)
                )
                if timing_exactness_pass is not None
                else None
            ),
            "reference_final_rng_state_hash": reference_complete_rng_hash,
            "candidate_final_rng_state_hash": candidate_complete_rng_hash,
            "rng_state_match": (
                reference_complete_rng_hash == candidate_complete_rng_hash
                if timing_exactness_pass is not None
                else None
            ),
        },
        "reference_lane_cache_storage_disjoint": reference_lane_cache_storage_disjoint,
        "resource_ownership": ownership.as_dict(),
        "graph": {
            "capture_count": 1,
            "total_replay_count": lane.replay_count,
            "parity_replay_count": 1,
            "model_only_timing_replay_count": (
                gate_config.timing_repeats if timing is not None else 0
            ),
            "complete_interval_replay_count": (
                gate_config.timing_repeats if timing is not None else 0
            ),
            "capture_seconds": lane.capture_seconds,
            "private_pool_id": graph_pool_id,
            "pool_sharing_requested": False,
            "capture_memory_allocated_delta_bytes": (
                lane.capture_memory_allocated_delta_bytes
            ),
            "capture_memory_reserved_delta_bytes": lane.capture_memory_reserved_delta_bytes,
        },
        "timing": timing,
        "workload_contract": workload_contract,
        "observation": observation,
        "runtime_metadata": {
            "git_commit": _git_commit(),
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "profile_sdpa_backend": args.profile_sdpa_backend,
            "generation_compile_requested": bool(args.inference_generation_compile),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": device_properties.name,
            "cuda_compute_capability": [device_properties.major, device_properties.minor],
            "cuda_total_memory_bytes": device_properties.total_memory,
            "seed": seed,
            "do_sample": bool(args.do_sample),
            "top_p": float(args.top_p),
            "top_k_sampling": int(args.top_k),
            "active_prefix_bucket_size": active_prefix_bucket_size,
            "active_prefix_length": active_prefix_length,
            "q1_bmm_cross_attention": q1_bmm_cross_attention,
            "native_q1_self_attention": native_q1_self_attention,
            "native_q1_rope_cache_self_attention": native_q1_rope_cache_self_attention,
            "probe": probe_metadata,
            "eos_token_ids": eos_token_ids,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cublas_workspace_policy": PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
            "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    previous_report = _load_previous_report(cli.previous_gate_report)
    validate_lane_gate_order(cli.lane_count, previous_report)
    if cli.lane_count != 1:
        raise RuntimeError(
            "concurrent B1 lane replay is intentionally not implemented. Run and review the "
            "L=1 parity/capture gate on RTX 2080/2080 Ti first."
        )
    total_started = time.perf_counter()
    args = _load_args(cli.config_name, cli.overrides)
    seed = int(cli.seed if cli.seed is not None else args.seed)
    report = run_one_lane_gate(
        args,
        config_name=cli.config_name,
        sequence_index=cli.sequence_index,
        seed=seed,
        atol=cli.atol,
        rtol=cli.rtol,
        comparison_top_k=cli.comparison_top_k,
        capture_warmup_repeats=cli.capture_warmup_repeats,
        timing_repeats=cli.timing_repeats,
        active_prefix_bucket_size=cli.active_prefix_bucket_size,
        q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
        native_q1_self_attention=cli.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
    )
    report["runtime_metadata"]["config_name"] = cli.config_name
    report["runtime_metadata"]["hydra_overrides"] = list(cli.overrides)
    report["total_wall_seconds"] = time.perf_counter() - total_started
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
