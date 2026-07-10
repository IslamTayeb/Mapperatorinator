"""Verifier-only contracts for the fixed-shape hybrid two-lane scout.

The hybrid scout is deliberately not a scheduler or runtime.  It measures one
fixed SALVALAI shape where two private B1 CUDA graphs feed one shared B2
full-scan logits-processor graph, followed by private sampling graphs.  This
module keeps the report schema and promotion arithmetic dependency-light so
CPU tests can reject malformed evidence before a GPU job is considered.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .lane_one_token import LaneResourceEvidence, validate_lane_resource_ownership


HYBRID_L2_SCHEMA_VERSION = 1
HYBRID_L2_GATE = "fixed_shape_identical_prompt_hybrid_two_lane"

REVIEWED_L1_COMPLETE_TPS = 414.9046973558729
FIVE_PERCENT_KEEP_TPS = REVIEWED_L1_COMPLETE_TPS * 1.05
TARGET_TPS = 500.0

REVIEWED_L2_COMMIT = "aa662c90c92b9a16afa626fa7e47c4309c87a429"
REVIEWED_L2_WORST_ORDER_TPS = 397.9033780517879
REVIEWED_L2_ORDER_TPS = {
    "0,1": 397.9033780517879,
    "1,0": 398.2907334765823,
}
REVIEWED_L2_REPORT_SHA256 = (
    "2533d7be39a065f62fd0fa5af6cf8fbe8173c01bbfc2eb1dc43bb4ccb296332e"
)
REVIEWED_MODEL_PATH = "OliBomby/Mapperatorinator-v32"
REVIEWED_PROMPT_SHA256 = "128f6bc10fbae29eb858c024fa975ea4face725182793b6e90c5140a2555ffd5"
REVIEWED_PROMPT_ATTENTION_MASK_SHA256 = (
    "b1e3b556e3e2b822e0097f9862808bca60367e3ba3dbc6b93dc0e56d5361e015"
)
REVIEWED_FRAMES_SHA256 = "101b40dc26d082850634ec51a10d63d8597cb51dc432ec9e85182841e69bc4eb"
REVIEWED_WORKLOAD_CONTRACT_SHA256 = (
    "d83d1dc26ac73f84c4c024b1293b045a54d6edfc23dadebe832ad5b24d606875"
)
REVIEWED_WORKLOAD_CONTRACT: dict[str, Any] = {
    "batch_size": 2,
    "seeds": [12345, 23456],
    "do_sample": True,
    "prompt_sha256": REVIEWED_PROMPT_SHA256,
    "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
    "frames_sha256": REVIEWED_FRAMES_SHA256,
    "condition_tensor_hashes": {},
    "active_prefix_prefill": False,
    "active_prefix_decode": True,
    "active_prefix_decode_length": 128,
    "runtime_contract": {
        "config_name": "profile_salvalai_smoke15",
        "model_path": REVIEWED_MODEL_PATH,
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "global_seed": 12345,
        "row_seeds": [12345, 23456],
        "do_sample": True,
        "top_p": 0.9,
        "top_k_sampling": 0,
        "temperature": 0.9,
        "stateful_monotonic_logits_processor": True,
        "eos_token_ids": [2, 6],
        "probe": {
            "sequence_index": 9,
            "frame_time_ms": 76965.0,
            "context_type": "map",
            "lookback_time": 0,
            "lookahead_time": 0,
        },
    },
}

FIXED_SEQUENCE_INDEX = 9
FIXED_ROW_SEEDS = (12345, 23456)
FIXED_TIMING_REPEATS = 50
FIXED_ACTIVE_PREFIX_LENGTH = 128
FIXED_ACTIVE_PREFIX_BUCKET_SIZE = 64

REQUIRED_BASE_PROCESSOR_CLASSES = (
    "MonotonicTimeShiftLogitsProcessor",
    "TemperatureLogitsWarper",
)
REQUIRED_PRIVATE_WARPER_CLASSES = ("TopPLogitsWarper",)

HYBRID_EVENT_ORDER = (
    "lane_model_graph_replay",
    "lane_output_ready_record",
    "control_wait_lane_output_ready",
    "copy_lane_logits_to_preallocated_B2_bridge",
    "source_logits_released_record",
    "shared_B2_full_scan_processor_graph_replay",
    "copy_processed_rows_to_private_sampling_inputs",
    "processor_done_record",
    "lane_wait_processor_done",
    "private_sampling_graph_replay",
    "sample_done_record",
    "control_wait_sample_done_at_interval_end",
)


def canonical_workload_contract_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HybridSamplingResourceEvidence:
    """Private resources owned by one lane's captured sampling tail."""

    lane_id: int
    sampling_graph_id: int
    sampling_graph_pool_id: str
    processed_scores_storage_ptr: int
    probabilities_storage_ptr: int
    sampled_token_storage_ptr: int
    generator_id: int
    logits_warper_id: int

    def __post_init__(self) -> None:
        if self.lane_id < 0:
            raise ValueError("lane_id must be non-negative.")
        for name in (
                "sampling_graph_id",
                "processed_scores_storage_ptr",
                "probabilities_storage_ptr",
                "sampled_token_storage_ptr",
                "generator_id",
                "logits_warper_id",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.sampling_graph_pool_id:
            raise ValueError("sampling_graph_pool_id must be non-empty.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HybridBridgeResourceEvidence:
    """Shared bridge resources plus the two private sampling tails."""

    control_stream_id: int
    processor_graph_id: int
    processor_graph_pool_id: str
    prefix_batch_storage_ptr: int
    raw_logits_bridge_storage_ptr: int
    processed_scores_storage_ptr: int
    source_release_event_ids: tuple[int, ...]
    lane_ready_event_ids: tuple[int, ...]
    sample_done_event_ids: tuple[int, ...]
    processor_done_event_id: int
    sampling_lanes: tuple[HybridSamplingResourceEvidence, ...]

    def __post_init__(self) -> None:
        for name in (
                "control_stream_id",
                "processor_graph_id",
                "prefix_batch_storage_ptr",
                "raw_logits_bridge_storage_ptr",
                "processed_scores_storage_ptr",
                "processor_done_event_id",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.processor_graph_pool_id:
            raise ValueError("processor_graph_pool_id must be non-empty.")
        if len(self.sampling_lanes) != 2:
            raise ValueError("the fixed hybrid gate requires exactly two sampling lanes.")
        for name in (
                "source_release_event_ids",
                "lane_ready_event_ids",
                "sample_done_event_ids",
        ):
            values = getattr(self, name)
            if len(values) != 2 or any(value <= 0 for value in values):
                raise ValueError(f"{name} must contain two positive event IDs.")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_release_event_ids"] = list(self.source_release_event_ids)
        payload["lane_ready_event_ids"] = list(self.lane_ready_event_ids)
        payload["sample_done_event_ids"] = list(self.sample_done_event_ids)
        payload["sampling_lanes"] = [lane.as_dict() for lane in self.sampling_lanes]
        return payload


def processor_class_names(processors: Sequence[Any]) -> tuple[str, ...]:
    return tuple(type(processor).__name__ for processor in processors)


def validate_hybrid_processor_family(
        *,
        base_processor_classes: Sequence[str],
        private_warper_classes: Sequence[str],
) -> None:
    """Pin the scout to the reviewed SALVALAI seq9 processor contract."""

    if tuple(base_processor_classes) != REQUIRED_BASE_PROCESSOR_CLASSES:
        raise ValueError(
            "hybrid L2 requires the reviewed base processor family "
            f"{REQUIRED_BASE_PROCESSOR_CLASSES}; got {tuple(base_processor_classes)}."
        )
    if tuple(private_warper_classes) != REQUIRED_PRIVATE_WARPER_CLASSES:
        raise ValueError(
            "hybrid L2 requires the reviewed private warper family "
            f"{REQUIRED_PRIVATE_WARPER_CLASSES}; got {tuple(private_warper_classes)}."
        )


def validate_reviewed_l2_report(report: Mapping[str, Any]) -> None:
    """Require the exact rejected L2 report that established the control gap."""

    if int(report.get("lane_count", -1)) != 2:
        raise ValueError("reviewed hybrid source report must be the L2 lane gate.")
    for field in ("exactness_pass", "resource_ownership_pass", "capture_pass"):
        if not bool(report.get(field)):
            raise ValueError(f"reviewed L2 report did not pass {field}.")
    runtime = report.get("runtime_metadata")
    if not isinstance(runtime, Mapping):
        raise ValueError("reviewed L2 report is missing runtime metadata.")
    if runtime.get("git_commit") != REVIEWED_L2_COMMIT:
        raise ValueError("reviewed L2 report commit does not match the audited source.")
    if tuple(runtime.get("seeds", ())) != FIXED_ROW_SEEDS:
        raise ValueError("reviewed L2 report seeds do not match the fixed gate.")
    if int(runtime.get("active_prefix_length", -1)) != FIXED_ACTIVE_PREFIX_LENGTH:
        raise ValueError("reviewed L2 report active-prefix length is not 128.")
    probe = runtime.get("probe")
    if not isinstance(probe, Mapping) or int(probe.get("sequence_index", -1)) != FIXED_SEQUENCE_INDEX:
        raise ValueError("reviewed L2 report is not SALVALAI sequence 9.")
    performance = report.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("reviewed L2 report is missing performance evidence.")
    observed_worst = float(performance.get("worst_order_tokens_per_second", math.nan))
    if not math.isclose(
            observed_worst,
            REVIEWED_L2_WORST_ORDER_TPS,
            rel_tol=0.0,
            abs_tol=1e-9,
    ):
        raise ValueError("reviewed L2 worst-order throughput does not match the audited report.")
    orders = report.get("orders")
    if not isinstance(orders, Mapping) or set(orders) != set(REVIEWED_L2_ORDER_TPS):
        raise ValueError("reviewed L2 report must contain both reciprocal orders.")
    for order, expected_tps in REVIEWED_L2_ORDER_TPS.items():
        timing = orders[order].get("complete_timing")
        if not isinstance(timing, Mapping):
            raise ValueError(f"reviewed L2 order {order} is missing complete timing.")
        observed_tps = float(timing.get("wall_tokens_per_second", math.nan))
        if not math.isclose(observed_tps, expected_tps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"reviewed L2 order {order} throughput changed.")
    workload_contract = report.get("workload_contract")
    if workload_contract != REVIEWED_WORKLOAD_CONTRACT:
        raise ValueError("reviewed L2 workload contract differs from the pinned seq9 source.")
    if canonical_workload_contract_sha256(workload_contract) != REVIEWED_WORKLOAD_CONTRACT_SHA256:
        raise ValueError("reviewed L2 workload contract hash does not match the pinned source.")


def validate_hybrid_resource_ownership(
        lane_evidence: Sequence[LaneResourceEvidence],
        bridge_evidence: HybridBridgeResourceEvidence,
) -> None:
    """Allow immutable weights and one bridge while rejecting mutable aliasing."""

    validate_lane_resource_ownership(lane_evidence, expected_lane_count=2)
    if tuple(lane.lane_id for lane in bridge_evidence.sampling_lanes) != (0, 1):
        raise ValueError("hybrid sampling lane IDs must be exactly 0 and 1.")

    event_ids = (
        *bridge_evidence.lane_ready_event_ids,
        *bridge_evidence.source_release_event_ids,
        *bridge_evidence.sample_done_event_ids,
        bridge_evidence.processor_done_event_id,
    )
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(
            "lane-ready, source-release, sample-done, and processor-done "
            "event IDs must be pairwise distinct."
        )

    lane_stream_ids = {lane.stream_id for lane in lane_evidence}
    if bridge_evidence.control_stream_id in lane_stream_ids:
        raise ValueError("the bridge control stream must be distinct from lane streams.")
    graph_ids = [lane.graph_id for lane in lane_evidence]
    graph_ids.append(bridge_evidence.processor_graph_id)
    graph_ids.extend(lane.sampling_graph_id for lane in bridge_evidence.sampling_lanes)
    if len(graph_ids) != len(set(graph_ids)):
        raise ValueError("model, processor, and sampling graph objects must be private.")
    graph_pools = [lane.graph_pool_id for lane in lane_evidence]
    graph_pools.append(bridge_evidence.processor_graph_pool_id)
    graph_pools.extend(
        lane.sampling_graph_pool_id for lane in bridge_evidence.sampling_lanes
    )
    if len(graph_pools) != len(set(graph_pools)):
        raise ValueError("concurrently replayable hybrid graphs must use separate pools.")

    lane_tensor_ptrs: set[int] = set()
    for lane in lane_evidence:
        lane_tensor_ptrs.update(lane.static_input_storage_ptrs.values())
        lane_tensor_ptrs.update(lane.self_cache_storage_ptrs)
        lane_tensor_ptrs.update(lane.cross_cache_storage_ptrs)
        lane_tensor_ptrs.add(lane.encoder_output_storage_ptr)
    bridge_ptrs = {
        bridge_evidence.prefix_batch_storage_ptr,
        bridge_evidence.raw_logits_bridge_storage_ptr,
        bridge_evidence.processed_scores_storage_ptr,
    }
    if len(bridge_ptrs) != 3:
        raise ValueError("the three shared bridge tensor pointers must be pairwise distinct.")
    sampling_ptrs: list[set[int]] = []
    for lane in bridge_evidence.sampling_lanes:
        pointers = {
            lane.processed_scores_storage_ptr,
            lane.probabilities_storage_ptr,
            lane.sampled_token_storage_ptr,
        }
        if len(pointers) != 3:
            raise ValueError(
                f"sampling lane {lane.lane_id} tensor pointers must be pairwise distinct."
            )
        sampling_ptrs.append(pointers)
    if lane_tensor_ptrs & bridge_ptrs:
        raise ValueError("bridge tensors must not alias lane model/cache tensors.")
    if any(lane_tensor_ptrs & pointers for pointers in sampling_ptrs):
        raise ValueError("sampling tensors must not alias lane model/cache tensors.")
    if bridge_ptrs & set.union(*sampling_ptrs):
        raise ValueError("shared bridge and private sampling tensors must not alias.")
    if sampling_ptrs[0] & sampling_ptrs[1]:
        raise ValueError("private sampling tensors must not alias across lanes.")
    for name, values in (
            ("sampling generator", [lane.generator_id for lane in bridge_evidence.sampling_lanes]),
            ("sampling warper", [lane.logits_warper_id for lane in bridge_evidence.sampling_lanes]),
    ):
        if len(set(values)) != 2:
            raise ValueError(f"private {name} objects must not alias across lanes.")


def _lane_evidence_from_mapping(value: Mapping[str, Any]) -> LaneResourceEvidence:
    payload = dict(value)
    payload["self_cache_storage_ptrs"] = tuple(payload["self_cache_storage_ptrs"])
    payload["cross_cache_storage_ptrs"] = tuple(payload["cross_cache_storage_ptrs"])
    return LaneResourceEvidence(**payload)


def _bridge_evidence_from_mapping(value: Mapping[str, Any]) -> HybridBridgeResourceEvidence:
    payload = dict(value)
    payload["source_release_event_ids"] = tuple(payload["source_release_event_ids"])
    payload["lane_ready_event_ids"] = tuple(payload["lane_ready_event_ids"])
    payload["sample_done_event_ids"] = tuple(payload["sample_done_event_ids"])
    payload["sampling_lanes"] = tuple(
        HybridSamplingResourceEvidence(**lane)
        for lane in payload["sampling_lanes"]
    )
    return HybridBridgeResourceEvidence(**payload)


def summarize_hybrid_l2_performance(
        order_tokens_per_second: Mapping[str, float],
) -> dict[str, Any]:
    """Gate on the slower reciprocal order and expose both campaign bars."""

    if set(order_tokens_per_second) != {"0,1", "1,0"}:
        raise ValueError("hybrid L2 timing requires reciprocal orders 0,1 and 1,0.")
    if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in order_tokens_per_second.values()
    ):
        raise ValueError("hybrid reciprocal throughput values must be finite and positive.")
    worst = min(float(value) for value in order_tokens_per_second.values())
    best = max(float(value) for value in order_tokens_per_second.values())
    return {
        "order_tokens_per_second": dict(sorted(order_tokens_per_second.items())),
        "worst_order_tokens_per_second": worst,
        "best_order_tokens_per_second": best,
        "reciprocal_order_spread_fraction": best / worst - 1.0,
        "reviewed_l1_complete_tokens_per_second": REVIEWED_L1_COMPLETE_TPS,
        "required_five_percent_tokens_per_second": FIVE_PERCENT_KEEP_TPS,
        "target_tokens_per_second": TARGET_TPS,
        "relative_gain_vs_reviewed_l1": worst / REVIEWED_L1_COMPLETE_TPS - 1.0,
        "relative_gain_vs_rejected_l2": worst / REVIEWED_L2_WORST_ORDER_TPS - 1.0,
        "five_percent_keep_bar_pass": worst > FIVE_PERCENT_KEEP_TPS,
        "target_500_bar_pass": worst > TARGET_TPS,
        "performance_gate_uses": "worst_reciprocal_complete_output_discard_wall_order",
    }


def validate_hybrid_l2_report(report: Mapping[str, Any]) -> None:
    """Self-validate exactness, ownership, event, and performance evidence."""

    if int(report.get("schema_version", -1)) != HYBRID_L2_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid L2 report schema.")
    if report.get("gate") != HYBRID_L2_GATE or int(report.get("lane_count", -1)) != 2:
        raise ValueError("report is not the fixed two-lane hybrid gate.")
    contract = report.get("fixed_workload_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("hybrid report is missing its fixed workload contract.")
    expected_contract = {
        "config_name": "profile_salvalai_smoke15",
        "sequence_index": FIXED_SEQUENCE_INDEX,
        "row_seeds": list(FIXED_ROW_SEEDS),
        "timing_repeats": FIXED_TIMING_REPEATS,
        "active_prefix_length": FIXED_ACTIVE_PREFIX_LENGTH,
        "active_prefix_bucket_size": FIXED_ACTIVE_PREFIX_BUCKET_SIZE,
        "model_path": REVIEWED_MODEL_PATH,
        "device": "cuda",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "profile_sdpa_backend": None,
        "inference_engine": "v32",
        "optimized_inference_mode": "single",
        "do_sample": True,
        "temperature": 0.9,
        "timeshift_bias": 0.0,
        "top_p": 0.9,
        "top_k": 0,
        "cfg_scale": 1.0,
        "num_beams": 1,
        "use_server": False,
        "parallel": False,
        "generation_compile": False,
        "active_prefix_decode_loop": True,
        "active_prefix_decode_cuda_graph": True,
        "active_prefix_decode_cuda_graph_warmup": 0,
        "active_prefix_decode_cuda_graph_min_decode_steps": 1,
        "stateful_monotonic_logits_processor": True,
        "q1_bmm_cross_attention": True,
        "native_q1_self_attention": True,
        "native_q1_rope_cache_self_attention": True,
        "decode_session_runtime": True,
        "decode_session_cuda_graph": True,
        "decode_session_chunk_size": 1,
        "native_decode_kernels": True,
        "prompt_sha256": REVIEWED_PROMPT_SHA256,
        "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
        "frames_sha256": REVIEWED_FRAMES_SHA256,
        "condition_tensor_hashes": {},
        "probe": {
            "context_type": "map",
            "frame_time_ms": 76965.0,
            "lookahead_time": 0,
            "lookback_time": 0,
            "sequence_index": FIXED_SEQUENCE_INDEX,
        },
        "identical_base_prompt": True,
        "row_private_anchor_and_sampling_state": True,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"fixed workload field {field} differs from {expected!r}.")
    if contract.get("do_sample") is not True:
        raise ValueError("fixed workload do_sample must be the boolean true.")
    if (
            not isinstance(contract.get("top_p"), (int, float))
            or isinstance(contract.get("top_p"), bool)
            or not math.isfinite(float(contract["top_p"]))
            or float(contract["top_p"]) != 0.9
    ):
        raise ValueError("fixed workload top_p must be numeric 0.9.")
    if (
            not isinstance(contract.get("top_k"), int)
            or isinstance(contract.get("top_k"), bool)
            or contract["top_k"] != 0
    ):
        raise ValueError("fixed workload top_k must be integer 0.")
    for field, expected in (
            ("temperature", 0.9),
            ("timeshift_bias", 0.0),
            ("cfg_scale", 1.0),
    ):
        value = contract.get(field)
        if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) != expected
        ):
            raise ValueError(f"fixed workload {field} must be numeric {expected}.")
    for field, expected in (
            ("num_beams", 1),
            ("active_prefix_decode_cuda_graph_warmup", 0),
            ("active_prefix_decode_cuda_graph_min_decode_steps", 1),
            ("decode_session_chunk_size", 1),
    ):
        value = contract.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected:
            raise ValueError(f"fixed workload {field} must be integer {expected}.")
    for field in (
            "do_sample",
            "active_prefix_decode_loop",
            "active_prefix_decode_cuda_graph",
            "stateful_monotonic_logits_processor",
            "q1_bmm_cross_attention",
            "native_q1_self_attention",
            "native_q1_rope_cache_self_attention",
            "decode_session_runtime",
            "decode_session_cuda_graph",
            "native_decode_kernels",
            "identical_base_prompt",
            "row_private_anchor_and_sampling_state",
    ):
        if contract.get(field) is not True:
            raise ValueError(f"fixed workload {field} must be the boolean true.")
    for field in ("use_server", "parallel", "generation_compile"):
        if contract.get(field) is not False:
            raise ValueError(f"fixed workload {field} must be the boolean false.")
    validate_hybrid_processor_family(
        base_processor_classes=report.get("base_processor_classes", ()),
        private_warper_classes=report.get("private_warper_classes", ()),
    )
    if tuple(report.get("event_order", ())) != HYBRID_EVENT_ORDER:
        raise ValueError("hybrid report event order does not match the audited safe sequence.")
    if report.get("safe_event_order_pass") is not True:
        raise ValueError("hybrid report did not pass the safe event-order gate.")

    resources = report.get("resource_ownership")
    if not isinstance(resources, Mapping):
        raise ValueError("hybrid report is missing resource ownership evidence.")
    model_lanes = resources.get("model_lanes")
    bridge_resource = resources.get("hybrid_bridge")
    if (
            not isinstance(model_lanes, Sequence)
            or isinstance(model_lanes, (str, bytes))
            or len(model_lanes) != 2
            or not all(isinstance(lane, Mapping) for lane in model_lanes)
            or not isinstance(bridge_resource, Mapping)
    ):
        raise ValueError("hybrid resource evidence must contain two model lanes and one bridge.")
    validate_hybrid_resource_ownership(
        [_lane_evidence_from_mapping(lane) for lane in model_lanes],
        _bridge_evidence_from_mapping(bridge_resource),
    )
    if report.get("resource_ownership_pass") is not True:
        raise ValueError("hybrid report did not pass resource ownership.")

    capture = report.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("hybrid report is missing capture evidence.")
    expected_capture_counts = {
        "model_graph_count": 2,
        "processor_graph_count": 1,
        "private_sampling_graph_count": 2,
        "total_graph_count": 5,
    }
    for field, expected in expected_capture_counts.items():
        if int(capture.get(field, -1)) != expected:
            raise ValueError(f"hybrid capture field {field} must be {expected}.")
    if capture.get("explicit_generator_registration") is not True:
        raise ValueError("private sampling generators must be registered with their graphs.")
    if report.get("capture_pass") is not True:
        raise ValueError("hybrid report did not pass graph capture.")

    reviewed_source = report.get("reviewed_l2_source")
    if not isinstance(reviewed_source, Mapping):
        raise ValueError("hybrid report is missing reviewed L2 source evidence.")
    reviewed_source_contract = {
        "job_id": "49548733",
        "commit": REVIEWED_L2_COMMIT,
        "report_sha256": REVIEWED_L2_REPORT_SHA256,
        "workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
    }
    for field, expected in reviewed_source_contract.items():
        if reviewed_source.get(field) != expected:
            raise ValueError(f"reviewed L2 source field {field} differs from {expected!r}.")
    source_worst = reviewed_source.get("worst_order_tokens_per_second")
    if (
            not isinstance(source_worst, (int, float))
            or isinstance(source_worst, bool)
            or not math.isfinite(float(source_worst))
            or not math.isclose(
                float(source_worst),
                REVIEWED_L2_WORST_ORDER_TPS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
    ):
        raise ValueError("reviewed L2 source worst-order throughput does not match.")

    runtime = report.get("runtime_metadata")
    if not isinstance(runtime, Mapping):
        raise ValueError("hybrid report is missing runtime metadata.")
    runtime_commit = runtime.get("git_commit")
    if not isinstance(runtime_commit, str) or not runtime_commit.strip():
        raise ValueError("hybrid runtime git commit must be non-empty.")

    workload_contract = report.get("workload_contract")
    if workload_contract != REVIEWED_WORKLOAD_CONTRACT:
        raise ValueError("hybrid workload contract differs from the pinned reviewed L2 source.")
    recomputed_workload_hash = canonical_workload_contract_sha256(workload_contract)
    if recomputed_workload_hash != REVIEWED_WORKLOAD_CONTRACT_SHA256:
        raise ValueError("hybrid workload contract recomputed hash differs from the source.")
    if report.get("workload_contract_sha256") != recomputed_workload_hash:
        raise ValueError("hybrid workload contract reported hash does not self-validate.")

    reference_token_hashes = report.get("reference_token_hashes")
    reference_rng_hashes = report.get("reference_final_rng_state_hashes")
    required_request_ids = {"row-0", "row-1"}
    if (
            not isinstance(reference_token_hashes, Mapping)
            or set(reference_token_hashes) != required_request_ids
            or not isinstance(reference_rng_hashes, Mapping)
            or set(reference_rng_hashes) != required_request_ids
    ):
        raise ValueError("hybrid report must contain two reference token and RNG hashes.")

    orders = report.get("orders")
    if not isinstance(orders, Mapping) or set(orders) != {"0,1", "1,0"}:
        raise ValueError("hybrid report must contain exactly both reciprocal orders.")
    order_tps: dict[str, float] = {}
    for order, values in orders.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"hybrid order {order} must be an object.")
        for field in (
                "processed_rows_bitwise_match",
                "processed_rows_topk_match",
                "transcript_match",
                "final_rng_state_match",
                "timed_final_rng_state_match",
                "exactness_pass",
        ):
            if not isinstance(values.get(field), bool):
                raise ValueError(f"hybrid order {order} field {field} must be boolean.")
        expected_order = [int(item) for item in order.split(",")]
        if values.get("graph_and_sampling_order") != expected_order:
            raise ValueError(f"hybrid order {order} did not preserve reciprocal execution order.")
        processed_rows = values.get("processed_rows")
        if (
                not isinstance(processed_rows, Sequence)
                or isinstance(processed_rows, (str, bytes))
                or len(processed_rows) != 2
        ):
            raise ValueError(f"hybrid order {order} must contain two processed-row reports.")
        row_bitwise_matches: list[bool] = []
        row_topk_matches: list[bool] = []
        for row, processed in enumerate(processed_rows):
            if (
                    not isinstance(processed, Mapping)
                    or int(processed.get("row", -1)) != row
            ):
                raise ValueError(f"hybrid order {order} processed row {row} is malformed.")
            for field in ("bitwise_match", "topk_match", "pass"):
                if not isinstance(processed.get(field), bool):
                    raise ValueError(
                        f"hybrid order {order} processed row {row} field {field} must be boolean."
                    )
            private_sha256 = processed.get("private_sha256")
            shared_sha256 = processed.get("shared_sha256")
            if (
                    not isinstance(private_sha256, str)
                    or not private_sha256
                    or not isinstance(shared_sha256, str)
                    or not shared_sha256
            ):
                raise ValueError(
                    f"hybrid order {order} processed row {row} requires nonempty score hashes."
                )
            recomputed_bitwise_match = private_sha256 == shared_sha256
            if processed["bitwise_match"] != recomputed_bitwise_match:
                raise ValueError(
                    f"hybrid order {order} processed row {row} bitwise_match "
                    "does not match its score hashes."
                )
            comparison = processed.get("comparison")
            if not isinstance(comparison, Mapping):
                raise ValueError(
                    f"hybrid order {order} processed row {row} comparison is missing."
                )
            for field in (
                    "shape_match",
                    "allclose",
                    "finite_allclose",
                    "nonfinite_match",
                    "positive_inf_match",
                    "negative_inf_match",
                    "topk_match",
            ):
                if not isinstance(comparison.get(field), bool):
                    raise ValueError(
                        f"hybrid order {order} processed row {row} comparison field "
                        f"{field} must be boolean."
                    )
            reference_shape = comparison.get("reference_shape")
            candidate_shape = comparison.get("candidate_shape")
            if (
                    not isinstance(reference_shape, list)
                    or not isinstance(candidate_shape, list)
                    or not all(
                        isinstance(size, int) and not isinstance(size, bool) and size >= 0
                        for size in reference_shape + candidate_shape
                    )
            ):
                raise ValueError(
                    f"hybrid order {order} processed row {row} comparison shapes are malformed."
                )
            if comparison["shape_match"] != (reference_shape == candidate_shape):
                raise ValueError(
                    f"hybrid order {order} processed row {row} shape_match is inconsistent."
                )
            reference_topk = comparison.get("reference_topk")
            candidate_topk = comparison.get("candidate_topk")
            if not isinstance(reference_topk, list) or not isinstance(candidate_topk, list):
                raise ValueError(
                    f"hybrid order {order} processed row {row} top-k evidence is malformed."
                )
            recomputed_topk_match = reference_topk == candidate_topk
            if comparison["topk_match"] != recomputed_topk_match:
                raise ValueError(
                    f"hybrid order {order} processed row {row} comparison top-k is inconsistent."
                )
            if processed["topk_match"] != comparison["topk_match"]:
                raise ValueError(
                    f"hybrid order {order} processed row {row} topk_match "
                    "differs from nested comparison evidence."
                )
            if processed["bitwise_match"] and not all(
                    comparison[field]
                    for field in (
                        "shape_match",
                        "allclose",
                        "finite_allclose",
                        "nonfinite_match",
                        "positive_inf_match",
                        "negative_inf_match",
                    )
            ):
                raise ValueError(
                    f"hybrid order {order} processed row {row} bitwise evidence "
                    "conflicts with comparison fields."
                )
            expected_row_pass = bool(
                processed["bitwise_match"] and processed["topk_match"]
            )
            if processed["pass"] != expected_row_pass:
                raise ValueError(
                    f"hybrid order {order} processed row {row} pass does not self-validate."
                )
            row_bitwise_matches.append(processed["bitwise_match"])
            row_topk_matches.append(processed["topk_match"])
        processed_bitwise_match = all(row_bitwise_matches)
        processed_topk_match = all(row_topk_matches)
        if values["processed_rows_bitwise_match"] != processed_bitwise_match:
            raise ValueError(f"hybrid order {order} processed bitwise aggregate is inconsistent.")
        if values["processed_rows_topk_match"] != processed_topk_match:
            raise ValueError(f"hybrid order {order} processed top-k aggregate is inconsistent.")

        candidate_token_hashes = values.get("candidate_token_hashes")
        candidate_rng_hashes = values.get("candidate_final_rng_state_hashes")
        timed_rng_hashes = values.get("timed_final_rng_state_hashes")
        for field, candidate in (
                ("candidate_token_hashes", candidate_token_hashes),
                ("candidate_final_rng_state_hashes", candidate_rng_hashes),
                ("timed_final_rng_state_hashes", timed_rng_hashes),
        ):
            if not isinstance(candidate, Mapping) or set(candidate) != required_request_ids:
                raise ValueError(f"hybrid order {order} field {field} is malformed.")
        transcript_match = candidate_token_hashes == reference_token_hashes
        final_rng_match = candidate_rng_hashes == reference_rng_hashes
        timed_rng_match = timed_rng_hashes == reference_rng_hashes
        for field, recomputed_match in (
                ("transcript_match", transcript_match),
                ("final_rng_state_match", final_rng_match),
                ("timed_final_rng_state_match", timed_rng_match),
        ):
            if values[field] != recomputed_match:
                raise ValueError(f"hybrid order {order} field {field} does not self-validate.")
        expected_order_exactness = bool(
            processed_bitwise_match
            and processed_topk_match
            and transcript_match
            and final_rng_match
            and timed_rng_match
        )
        if values["exactness_pass"] != expected_order_exactness:
            raise ValueError(f"hybrid order {order} exactness_pass does not self-validate.")
        timing = values.get("complete_output_discard_timing")
        if not isinstance(timing, Mapping):
            raise ValueError(f"hybrid order {order} is missing complete timing.")
        if timing.get("measurement") != "complete_output_discard":
            raise ValueError("hybrid performance must use complete output-discard timing.")
        if timing.get("outputs_discarded") is not True:
            raise ValueError("timed hybrid outputs must be discarded.")
        if bool(timing.get("per_step_global_synchronize")):
            raise ValueError("timed hybrid loop must not globally synchronize per step.")
        if bool(timing.get("per_step_stack_or_cat")):
            raise ValueError("timed hybrid loop must not stack or concatenate tensors per step.")
        if bool(timing.get("per_step_python_tensor_allocation")):
            raise ValueError("timed hybrid loop must reuse captured/static tensor storage.")
        timing_repeats = timing.get("timing_repeats")
        if (
                not isinstance(timing_repeats, int)
                or isinstance(timing_repeats, bool)
                or timing_repeats != FIXED_TIMING_REPEATS
        ):
            raise ValueError("hybrid timing must use exactly 50 repeats.")
        reported_generated_tokens = timing.get("generated_tokens")
        if (
                not isinstance(reported_generated_tokens, int)
                or isinstance(reported_generated_tokens, bool)
                or reported_generated_tokens != 2 * FIXED_TIMING_REPEATS
        ):
            raise ValueError("hybrid timing must account for exactly two tokens per repeat.")
        generated_tokens = 2 * FIXED_TIMING_REPEATS
        wall_seconds = timing.get("wall_seconds")
        cuda_seconds = timing.get("cuda_seconds")
        for field, value in (
                ("wall_seconds", wall_seconds),
                ("cuda_seconds", cuda_seconds),
        ):
            if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0
            ):
                raise ValueError(f"hybrid timing {field} must be finite and positive.")
        recomputed_wall_tps = generated_tokens / float(wall_seconds)
        recomputed_cuda_tps = generated_tokens / float(cuda_seconds)
        for field, recomputed_tps in (
                ("wall_tokens_per_second", recomputed_wall_tps),
                ("cuda_tokens_per_second", recomputed_cuda_tps),
        ):
            reported_tps = timing.get(field)
            if (
                    not isinstance(reported_tps, (int, float))
                    or isinstance(reported_tps, bool)
                    or not math.isfinite(float(reported_tps))
                    or float(reported_tps) <= 0
                    or not math.isclose(
                        float(reported_tps),
                        recomputed_tps,
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
            ):
                raise ValueError(f"hybrid timing {field} does not self-validate.")

        memory_fields: dict[str, int] = {}
        for field in (
                "memory_before_allocated_bytes",
                "memory_before_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "peak_allocated_delta_bytes",
                "peak_reserved_delta_bytes",
        ):
            value = timing.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"hybrid timing {field} must be a non-negative integer.")
            memory_fields[field] = value
        if memory_fields["memory_before_allocated_bytes"] > memory_fields["memory_before_reserved_bytes"]:
            raise ValueError("hybrid timing allocated memory cannot exceed reserved memory.")
        if memory_fields["peak_allocated_bytes"] < memory_fields["memory_before_allocated_bytes"]:
            raise ValueError("hybrid peak allocated memory cannot precede the interval baseline.")
        if memory_fields["peak_reserved_bytes"] < memory_fields["memory_before_reserved_bytes"]:
            raise ValueError("hybrid peak reserved memory cannot precede the interval baseline.")
        if memory_fields["peak_allocated_bytes"] > memory_fields["peak_reserved_bytes"]:
            raise ValueError("hybrid peak allocated memory cannot exceed peak reserved memory.")
        expected_allocated_delta = (
            memory_fields["peak_allocated_bytes"]
            - memory_fields["memory_before_allocated_bytes"]
        )
        expected_reserved_delta = (
            memory_fields["peak_reserved_bytes"]
            - memory_fields["memory_before_reserved_bytes"]
        )
        if memory_fields["peak_allocated_delta_bytes"] != expected_allocated_delta:
            raise ValueError("hybrid peak allocated memory delta does not self-validate.")
        if memory_fields["peak_reserved_delta_bytes"] != expected_reserved_delta:
            raise ValueError("hybrid peak reserved memory delta does not self-validate.")
        for field in (
                "static_tensor_storage",
                "sampling_tail_in_cuda_graph",
                "explicit_generators_registered_with_sampling_graphs",
        ):
            if timing.get(field) is not True:
                raise ValueError(f"hybrid timing requires {field}=true.")

        # Promotion arithmetic uses only generated_tokens / measured wall_seconds.
        # Never trust the report's precomputed TPS fields as denominators.
        order_tps[order] = recomputed_wall_tps

    recomputed = summarize_hybrid_l2_performance(order_tps)
    performance = report.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("hybrid report is missing performance summary.")
    for field in (
            "worst_order_tokens_per_second",
            "best_order_tokens_per_second",
            "reciprocal_order_spread_fraction",
            "reviewed_l1_complete_tokens_per_second",
            "required_five_percent_tokens_per_second",
            "target_tokens_per_second",
            "relative_gain_vs_reviewed_l1",
            "relative_gain_vs_rejected_l2",
    ):
        if not math.isclose(
                float(performance.get(field, math.nan)),
                float(recomputed[field]),
                rel_tol=0.0,
                abs_tol=1e-9,
        ):
            raise ValueError(f"hybrid performance field {field} does not self-validate.")
    reported_order_tps = performance.get("order_tokens_per_second")
    if not isinstance(reported_order_tps, Mapping) or set(reported_order_tps) != set(order_tps):
        raise ValueError("hybrid performance reciprocal-order mapping is malformed.")
    for order, expected_tps in recomputed["order_tokens_per_second"].items():
        if not math.isclose(
                float(reported_order_tps.get(order, math.nan)),
                float(expected_tps),
                rel_tol=1e-12,
                abs_tol=1e-9,
        ):
            raise ValueError(f"hybrid performance order {order} does not self-validate.")
    for field in ("five_percent_keep_bar_pass", "target_500_bar_pass"):
        if not isinstance(performance.get(field), bool) or performance[field] != recomputed[field]:
            raise ValueError(f"hybrid performance field {field} does not self-validate.")
    if performance.get("performance_gate_uses") != recomputed["performance_gate_uses"]:
        raise ValueError("hybrid performance gate scope does not self-validate.")

    exactness_pass = all(values["exactness_pass"] is True for values in orders.values())
    promotion_pass = bool(
        exactness_pass
        and report.get("resource_ownership_pass")
        and report.get("capture_pass")
        and report.get("safe_event_order_pass")
        and recomputed["five_percent_keep_bar_pass"]
    )
    graduation_pass = bool(promotion_pass and recomputed["target_500_bar_pass"])
    for field, expected in (
            ("exactness_pass", exactness_pass),
            ("promotion_gate_pass", promotion_pass),
            ("graduation_to_next_shape_pass", graduation_pass),
            ("pass", graduation_pass),
    ):
        if not isinstance(report.get(field), bool) or report[field] != expected:
            raise ValueError(f"top-level {field} does not self-validate.")
