"""Isolated B1 CUDA-graph lane capture helpers.

This module is verifier infrastructure only.  It does not implement a lane
pool, scheduler, or server path.  The first GPU gate captures one prepared B1
one-token model call on a persistent private stream, using the default separate
CUDA-graph memory pool, then validates the replay against an independent B1
reference.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from ...direct_decode import last_token_logits
from ....runtime_profiling import active_prefix_self_attention_context


PYTORCH_PER_STREAM_CUBLAS_WORKSPACE = "pytorch_per_handle_and_stream_workspace"
DEFAULT_SEPARATE_GRAPH_POOL = "default_separate_capture_pool"


def normalized_one_token_request_ids(request_count: int) -> tuple[str, ...]:
    """Use the same row IDs as the merged one-token reports."""

    if request_count <= 0:
        raise ValueError("request_count must be positive.")
    return tuple(f"row-{row}" for row in range(request_count))


def normalized_one_token_workload_contract(
        *,
        batch_size: int,
        seeds: Sequence[int],
        do_sample: bool,
        prompt_sha256: str,
        prompt_attention_mask_sha256: str,
        frames_sha256: str,
        condition_tensor_hashes: Mapping[str, str],
        active_prefix_prefill: bool,
        active_prefix_decode: bool,
        active_prefix_decode_length: int | None,
        runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the logical workload schema shared with merged one-token reports."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(seeds) != batch_size:
        raise ValueError("seeds must contain one value per logical request.")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise TypeError("seeds must contain integers.")
    for name, value in (
            ("prompt_sha256", prompt_sha256),
            ("prompt_attention_mask_sha256", prompt_attention_mask_sha256),
            ("frames_sha256", frames_sha256),
    ):
        if not value:
            raise ValueError(f"{name} must be non-empty.")
    if not runtime_contract:
        raise ValueError("runtime_contract must be non-empty.")
    return {
        "batch_size": batch_size,
        "seeds": list(seeds),
        "do_sample": bool(do_sample),
        "prompt_sha256": prompt_sha256,
        "prompt_attention_mask_sha256": prompt_attention_mask_sha256,
        "frames_sha256": frames_sha256,
        "condition_tensor_hashes": dict(sorted(condition_tensor_hashes.items())),
        "active_prefix_prefill": bool(active_prefix_prefill),
        "active_prefix_decode": bool(active_prefix_decode),
        "active_prefix_decode_length": active_prefix_decode_length,
        "runtime_contract": dict(runtime_contract),
    }


@dataclass(frozen=True)
class OneLaneCaptureConfig:
    """Settings for the smallest lane-physics gate."""

    seed: int
    active_prefix_length: int
    atol: float = 1e-4
    rtol: float = 1e-4
    top_k: int = 20
    capture_warmup_repeats: int = 1
    timing_repeats: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")
        if self.active_prefix_length <= 0:
            raise ValueError("active_prefix_length must be positive.")
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("atol and rtol must be non-negative.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.capture_warmup_repeats < 1:
            raise ValueError(
                "capture_warmup_repeats must be positive so the lane stream's "
                "library handles and workspace are initialized before capture."
            )
        if self.timing_repeats <= 0:
            raise ValueError("timing_repeats must be positive.")


@dataclass(frozen=True)
class LaneResourceEvidence:
    """Auditable ownership evidence for one captured lane.

    PyTorch does not expose the cuBLAS workspace pointer.  Its documented
    allocation unit is a cuBLAS-handle/CUDA-stream combination, so the verifier
    records the unique persistent stream as the workspace owner and records the
    active ``CUBLAS_WORKSPACE_CONFIG`` value instead of inventing a pointer.
    """

    lane_id: int
    shared_model_id: int
    stream_id: int
    graph_id: int
    graph_pool_id: str
    graph_pool_policy: str
    graph_pool_sharing_requested: bool
    session_id: int
    cache_id: int
    encoder_output_storage_ptr: int
    generator_id: int
    logits_processor_id: int
    static_input_storage_ptrs: Mapping[str, int]
    self_cache_storage_ptrs: tuple[int, ...]
    cross_cache_storage_ptrs: tuple[int, ...]
    cublas_workspace_policy: str
    cublas_workspace_owner_stream_id: int
    cublas_workspace_config: str | None
    cublas_workspace_pointer_exposed: bool = False

    def __post_init__(self) -> None:
        if self.lane_id < 0:
            raise ValueError("lane_id must be non-negative.")
        for field_name in (
                "shared_model_id",
                "stream_id",
                "graph_id",
                "session_id",
                "cache_id",
                "encoder_output_storage_ptr",
                "generator_id",
                "logits_processor_id",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not self.graph_pool_id:
            raise ValueError("graph_pool_id must be non-empty.")
        if self.graph_pool_policy != DEFAULT_SEPARATE_GRAPH_POOL:
            raise ValueError(
                f"lane capture requires graph_pool_policy={DEFAULT_SEPARATE_GRAPH_POOL!r}."
            )
        if self.graph_pool_sharing_requested:
            raise ValueError("concurrently replayable lanes must not share CUDA graph pools.")
        if not self.static_input_storage_ptrs:
            raise ValueError("static_input_storage_ptrs must be non-empty.")
        if any(not name or pointer <= 0 for name, pointer in self.static_input_storage_ptrs.items()):
            raise ValueError("static input names and storage pointers must be non-empty and positive.")
        if not self.self_cache_storage_ptrs or not self.cross_cache_storage_ptrs:
            raise ValueError("self and cross cache storage pointers must be recorded.")
        if any(pointer <= 0 for pointer in self.self_cache_storage_ptrs + self.cross_cache_storage_ptrs):
            raise ValueError("cache storage pointers must be positive.")
        if self.cublas_workspace_policy != PYTORCH_PER_STREAM_CUBLAS_WORKSPACE:
            raise ValueError(
                "lane capture must use the documented PyTorch per-handle/stream "
                "cuBLAS workspace policy."
            )
        if self.cublas_workspace_owner_stream_id != self.stream_id:
            raise ValueError("cuBLAS workspace ownership must be attributed to the lane stream.")
        if self.cublas_workspace_pointer_exposed:
            raise ValueError("PyTorch does not expose a cuBLAS workspace pointer for this gate.")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["static_input_storage_ptrs"] = dict(sorted(self.static_input_storage_ptrs.items()))
        payload["self_cache_storage_ptrs"] = list(self.self_cache_storage_ptrs)
        payload["cross_cache_storage_ptrs"] = list(self.cross_cache_storage_ptrs)
        return payload


@dataclass
class CapturedB1Lane:
    """Live resources retained for a single verifier-only B1 graph lane."""

    stream: torch.cuda.Stream
    graph: torch.cuda.CUDAGraph
    graph_outputs: Any
    static_inputs: dict[str, Any]
    warmup_seconds: float
    capture_seconds: float
    capture_memory_allocated_delta_bytes: int
    capture_memory_reserved_delta_bytes: int
    replay_count: int = 0

    @property
    def logits(self) -> torch.Tensor:
        return last_token_logits(self.graph_outputs.logits)

    def replay(self) -> None:
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        self.replay_count += 1


def compare_lane_logits(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
    """Compare FP32 logits without hiding non-finite layout differences."""

    reference = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float32)
    if reference.shape != candidate.shape:
        return {
            "allclose": False,
            "topk_match": False,
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference_nonfinite = ~torch.isfinite(reference)
    candidate_nonfinite = ~torch.isfinite(candidate)
    finite_mask = ~reference_nonfinite & ~candidate_nonfinite
    finite_reference = reference[finite_mask]
    finite_candidate = candidate[finite_mask]
    finite_allclose = bool(
        torch.allclose(finite_reference, finite_candidate, atol=atol, rtol=rtol)
    )
    nonfinite_match = bool(torch.equal(reference_nonfinite, candidate_nonfinite))
    positive_inf_match = bool(torch.equal(torch.isposinf(reference), torch.isposinf(candidate)))
    negative_inf_match = bool(torch.equal(torch.isneginf(reference), torch.isneginf(candidate)))
    difference = torch.abs(finite_reference - finite_candidate)
    k = min(top_k, int(reference.shape[-1]))
    reference_topk = torch.topk(reference, k=k, dim=-1).indices
    candidate_topk = torch.topk(candidate, k=k, dim=-1).indices
    topk_match = bool(torch.equal(reference_topk, candidate_topk))
    return {
        "allclose": (
            finite_allclose
            and nonfinite_match
            and positive_inf_match
            and negative_inf_match
        ),
        "finite_allclose": finite_allclose,
        "nonfinite_match": nonfinite_match,
        "positive_inf_match": positive_inf_match,
        "negative_inf_match": negative_inf_match,
        "topk_match": topk_match,
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "reference_topk": reference_topk[0].tolist(),
        "candidate_topk": candidate_topk[0].tolist(),
    }


def validate_lane_resource_ownership(
        evidence: Sequence[LaneResourceEvidence],
        *,
        expected_lane_count: int,
) -> None:
    """Fail if resources that must be private alias across lanes."""

    if expected_lane_count <= 0:
        raise ValueError("expected_lane_count must be positive.")
    if len(evidence) != expected_lane_count:
        raise ValueError(
            f"expected {expected_lane_count} lane ownership records; got {len(evidence)}."
        )
    if sorted(item.lane_id for item in evidence) != list(range(expected_lane_count)):
        raise ValueError("lane IDs must be contiguous from zero.")
    if len({item.shared_model_id for item in evidence}) != 1:
        raise ValueError("all lanes must share exactly one immutable model/weight owner.")

    unique_fields = (
        "stream_id",
        "graph_id",
        "graph_pool_id",
        "session_id",
        "cache_id",
        "encoder_output_storage_ptr",
        "generator_id",
        "logits_processor_id",
    )
    for field_name in unique_fields:
        values = [getattr(item, field_name) for item in evidence]
        if len(set(values)) != len(values):
            raise ValueError(f"lane-private {field_name} values alias across lanes.")

    pointer_sets: list[set[int]] = []
    for item in evidence:
        pointers = set(item.static_input_storage_ptrs.values())
        pointers.update(item.self_cache_storage_ptrs)
        pointers.update(item.cross_cache_storage_ptrs)
        pointers.add(item.encoder_output_storage_ptr)
        pointer_sets.append(pointers)
    for lane, pointers in enumerate(pointer_sets):
        for other_lane in range(lane):
            overlap = pointers & pointer_sets[other_lane]
            if overlap:
                raise ValueError(
                    f"lane-private tensor storage aliases between lanes {other_lane} and "
                    f"{lane}: {sorted(overlap)}."
                )


def validate_lane_gate_order(
        lane_count: int,
        previous_report: Mapping[str, Any] | None,
) -> None:
    """Enforce B1 parity before any concurrent lane measurement."""

    if lane_count not in {1, 2, 3, 4}:
        raise ValueError("lane_count must be one of 1, 2, 3, or 4.")
    if lane_count == 1:
        if previous_report is not None:
            raise ValueError("the B1 parity/capture gate must not receive a previous report.")
        return
    if previous_report is None:
        raise ValueError(f"L={lane_count} requires a passed L={lane_count - 1} report.")
    if int(previous_report.get("lane_count", -1)) != lane_count - 1:
        raise ValueError(f"L={lane_count} requires an L={lane_count - 1} previous report.")
    for field_name in ("pass", "exactness_pass", "resource_ownership_pass", "capture_pass"):
        if not bool(previous_report.get(field_name)):
            raise ValueError(f"previous lane report did not pass {field_name}.")
    if lane_count >= 3 and not bool(previous_report.get("performance_gate_pass")):
        raise ValueError(
            "do not graduate beyond L=2 unless concurrent lanes clear the 5% B1 keep bar."
        )


def summarize_lane_scaling(
        *,
        b1_tokens_per_second: float,
        candidate_lane_count: int,
        candidate_tokens_per_second: float,
) -> dict[str, Any]:
    """Apply the campaign keep bar against the one-lane replay denominator."""

    if not math.isfinite(b1_tokens_per_second) or b1_tokens_per_second <= 0:
        raise ValueError("b1_tokens_per_second must be finite and positive.")
    if candidate_lane_count <= 1:
        raise ValueError("candidate_lane_count must exceed one.")
    if not math.isfinite(candidate_tokens_per_second) or candidate_tokens_per_second <= 0:
        raise ValueError("candidate_tokens_per_second must be finite and positive.")
    relative_gain = candidate_tokens_per_second / b1_tokens_per_second - 1.0
    ideal_tokens_per_second = b1_tokens_per_second * candidate_lane_count
    return {
        "b1_tokens_per_second": b1_tokens_per_second,
        "candidate_lane_count": candidate_lane_count,
        "candidate_tokens_per_second": candidate_tokens_per_second,
        "relative_throughput_gain_vs_b1": relative_gain,
        "clears_five_percent_b1_keep_bar": relative_gain >= 0.05,
        "ideal_linear_tokens_per_second": ideal_tokens_per_second,
        "fraction_of_ideal_linear_capacity": (
            candidate_tokens_per_second / ideal_tokens_per_second
        ),
    }


def _clone_graph_inputs(model_inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.clone(memory_format=torch.contiguous_format)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in model_inputs.items()
    }


@torch.no_grad()
def capture_prepared_b1_lane(
        model: Any,
        prepared_inputs: Mapping[str, Any],
        *,
        active_prefix_length: int,
        capture_warmup_repeats: int,
) -> CapturedB1Lane:
    """Capture one prepared B1 model call on one persistent private stream."""

    if not torch.cuda.is_available():
        raise RuntimeError("the B1 lane capture gate requires a CUDA GPU allocation.")
    if active_prefix_length <= 0:
        raise ValueError("active_prefix_length must be positive.")
    if capture_warmup_repeats < 1:
        raise ValueError("capture_warmup_repeats must be positive.")
    decoder_input_ids = prepared_inputs.get("decoder_input_ids")
    if not isinstance(decoder_input_ids, torch.Tensor) or decoder_input_ids.shape[0] != 1:
        raise ValueError("prepared_inputs must contain B1 decoder_input_ids.")

    device = decoder_input_ids.device
    current_stream = torch.cuda.current_stream(device)
    lane_stream = torch.cuda.Stream(device=device)
    static_inputs = _clone_graph_inputs(prepared_inputs)
    warmup_started = time.perf_counter()
    lane_stream.wait_stream(current_stream)
    with torch.cuda.stream(lane_stream):
        for _ in range(capture_warmup_repeats):
            with active_prefix_self_attention_context(active_prefix_length):
                model(**static_inputs, return_dict=True)
    current_stream.wait_stream(lane_stream)
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph, stream=lane_stream):
        with active_prefix_self_attention_context(active_prefix_length):
            graph_outputs = model(**static_inputs, return_dict=True)
    capture_seconds = time.perf_counter() - capture_started
    return CapturedB1Lane(
        stream=lane_stream,
        graph=graph,
        graph_outputs=graph_outputs,
        static_inputs=static_inputs,
        warmup_seconds=warmup_seconds,
        capture_seconds=capture_seconds,
        capture_memory_allocated_delta_bytes=max(
            0,
            torch.cuda.memory_allocated(device) - allocated_before,
        ),
        capture_memory_reserved_delta_bytes=max(
            0,
            torch.cuda.memory_reserved(device) - reserved_before,
        ),
    )


def time_b1_lane_replay(
        lane: CapturedB1Lane,
        *,
        timing_repeats: int,
) -> dict[str, Any]:
    """Measure warmed model-only replay on the lane's persistent stream."""

    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive.")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    with torch.cuda.stream(lane.stream):
        start.record(lane.stream)
        for _ in range(timing_repeats):
            lane.graph.replay()
        end.record(lane.stream)
    lane.replay_count += timing_repeats
    end.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    cuda_seconds = start.elapsed_time(end) / 1000.0
    return {
        "timing_repeats": timing_repeats,
        "model_only_cuda_seconds": cuda_seconds,
        "model_only_wall_seconds": wall_seconds,
        "model_only_cuda_tokens_per_second": timing_repeats / cuda_seconds,
        "model_only_wall_tokens_per_second": timing_repeats / wall_seconds,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def cublas_workspace_metadata(stream_id: int) -> dict[str, Any]:
    """Describe the workspace ownership basis without claiming a pointer."""

    return {
        "policy": PYTORCH_PER_STREAM_CUBLAS_WORKSPACE,
        "owner_stream_id": stream_id,
        "workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "workspace_pointer_exposed": False,
        "ownership_basis": (
            "PyTorch allocates a cuBLAS workspace for each cuBLAS-handle/CUDA-stream "
            "combination; the lane stream is warmed before graph capture."
        ),
    }
