"""Strict contract for the real-prefix bucket-576 Hybrid-B2 H8 scout.

This module contains report validation and source-derived arithmetic only.  It
does not expose a scheduler or runtime implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .weighted_bucket import (
    REVIEWED_SOURCE_CAPTURE_COMMIT,
    REVIEWED_SOURCE_CAPTURE_JOB_ID,
    REVIEWED_SOURCE_CONTRACT_SHA256,
    REVIEWED_SOURCE_REPORT_FILE_SHA256,
)


WEIGHTED_H8_SCHEMA_VERSION = 1
WEIGHTED_H8_GATE = "weighted_real_prefix_bucket576_h8_hybrid_b2"
ACTIVE_PREFIX_LENGTH = 576
HORIZON = 8
TIMING_TRIALS_PER_ORDER = 5
NEUTRAL_PAD_TOKEN_ID = 0
ROW_SEED = 12345
ROW_SOURCE_OFFSETS = (0, 1)
ROW_INITIAL_PREFIX_LENGTHS = (520, 521)
ROW_INITIAL_CACHE_POSITIONS = (519, 520)
ROW_FIRST_OUTPUT_OFFSETS = (42, 43)
REQUIRED_WEIGHTED_COMPLETE_TPS = 623.656675984333
WEIGHTED_CEILING_REPORT_SHA256 = (
    "44a680ab29867e3aea8dde713127bdb154ef42a316fd2834bc75afbbc0927fc9"
)

EVENT_ORDER = (
    "copy_initial_prefix_static_and_stop_state",
    "record_state_ready",
    "private_model_graphs_wait_and_replay",
    "record_lane_ready",
    "shared_b2_processor_wait_copy_and_replay",
    "copy_processed_rows_to_private_sampling_inputs",
    "private_sampling_and_stop_graphs_replay",
    "copy_sampled_tokens_and_next_static_inputs",
)

HASH_PAIR_FIELDS = {"reference_sha256", "candidate_sha256", "bitwise_match"}
COMPARISON_FIELDS = {
    "shape_match", "allclose", "topk_match", "reference_shape",
    "candidate_shape", "reference_topk", "candidate_topk", "max_abs",
    "reference_sha256", "candidate_sha256",
}


def weighted_h8_steps_signature(steps: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical digest used to bind a report to its step ledger."""

    rendered = json.dumps(
        steps, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_sha256(value: Any, label: str) -> None:
    if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")


def _validate_hash_pair(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != HASH_PAIR_FIELDS:
        raise ValueError(f"{label} hash-pair fields changed.")
    _validate_sha256(value["reference_sha256"], f"{label} reference")
    _validate_sha256(value["candidate_sha256"], f"{label} candidate")
    expected = value["reference_sha256"] == value["candidate_sha256"]
    if value["bitwise_match"] is not expected:
        raise ValueError(f"{label} bitwise result is inconsistent.")


def _validate_comparison(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != COMPARISON_FIELDS:
        raise ValueError(f"{label} comparison fields changed.")
    for field in ("shape_match", "allclose", "topk_match"):
        if not isinstance(value[field], bool):
            raise ValueError(f"{label} {field} must be boolean.")
    for field in (
            "reference_shape", "candidate_shape", "reference_topk",
            "candidate_topk",
    ):
        if not isinstance(value[field], list):
            raise ValueError(f"{label} {field} must be a list.")
    if (
            not isinstance(value["max_abs"], (int, float))
            or isinstance(value["max_abs"], bool)
            or not math.isfinite(float(value["max_abs"]))
            or float(value["max_abs"]) < 0.0
    ):
        raise ValueError(f"{label} max_abs changed.")
    _validate_sha256(value["reference_sha256"], f"{label} reference")
    _validate_sha256(value["candidate_sha256"], f"{label} candidate")


def _finite_positive(value: Any, label: str) -> float:
    if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive.")
    return float(value)


def _validate_timing(
        timing: Mapping[str, Any],
        *,
        order: str,
        measurement: str,
) -> None:
    expected_tokens = 2 * HORIZON * TIMING_TRIALS_PER_ORDER
    expected_fields = {
        "measurement",
        "horizon",
        "trial_count",
        "generated_tokens",
        "wall_seconds",
        "cuda_seconds",
        "wall_tokens_per_second",
        "cuda_tokens_per_second",
        "trials",
        "no_per_step_synchronize",
        "no_per_step_allocation",
        "no_graph_recapture",
        "reset_outside_timing",
        "initial_state_copy_inside_timing",
        "final_state_matches_exact",
    }
    if set(timing) != expected_fields:
        raise ValueError(f"order {order} {measurement} timing fields changed.")
    if (
            timing["measurement"] != measurement
            or timing["horizon"] != HORIZON
            or timing["trial_count"] != TIMING_TRIALS_PER_ORDER
            or timing["generated_tokens"] != expected_tokens
    ):
        raise ValueError(f"order {order} {measurement} timing shape changed.")
    for field in (
            "no_per_step_synchronize",
            "no_per_step_allocation",
            "no_graph_recapture",
            "reset_outside_timing",
            "initial_state_copy_inside_timing",
            "final_state_matches_exact",
    ):
        if timing[field] is not True:
            raise ValueError(f"order {order} {measurement} requires {field}=true.")
    wall = _finite_positive(timing["wall_seconds"], f"order {order} {measurement} wall")
    cuda = _finite_positive(timing["cuda_seconds"], f"order {order} {measurement} CUDA")
    if not math.isclose(
            float(timing["wall_tokens_per_second"]),
            expected_tokens / wall,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ) or not math.isclose(
            float(timing["cuda_tokens_per_second"]),
            expected_tokens / cuda,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"order {order} {measurement} throughput is inconsistent.")
    trials = timing["trials"]
    if (
            not isinstance(trials, Sequence)
            or isinstance(trials, (str, bytes))
            or len(trials) != TIMING_TRIALS_PER_ORDER
    ):
        raise ValueError(f"order {order} {measurement} trials changed.")
    trial_tokens = 2 * HORIZON
    for index, trial in enumerate(trials):
        if set(trial) != {
            "trial", "generated_tokens", "wall_seconds", "cuda_seconds",
            "wall_tokens_per_second", "cuda_tokens_per_second",
            "final_state_matches_exact", "memory_before_allocated_bytes",
            "memory_before_reserved_bytes", "peak_allocated_bytes",
            "peak_reserved_bytes", "peak_allocated_delta_bytes",
            "peak_reserved_delta_bytes",
        }:
            raise ValueError(f"order {order} {measurement} trial {index} fields changed.")
        if (
                trial["trial"] != index
                or trial["generated_tokens"] != trial_tokens
                or trial["final_state_matches_exact"] is not True
                or trial["peak_allocated_delta_bytes"] != 0
                or trial["peak_reserved_delta_bytes"] != 0
                or trial["peak_allocated_bytes"]
                != trial["memory_before_allocated_bytes"]
                or trial["peak_reserved_bytes"]
                != trial["memory_before_reserved_bytes"]
        ):
            raise ValueError(f"order {order} {measurement} trial {index} changed.")
        trial_wall = _finite_positive(
            trial["wall_seconds"], f"order {order} {measurement} trial {index} wall"
        )
        trial_cuda = _finite_positive(
            trial["cuda_seconds"], f"order {order} {measurement} trial {index} CUDA"
        )
        if not math.isclose(
                float(trial["wall_tokens_per_second"]), trial_tokens / trial_wall,
                rel_tol=1e-12, abs_tol=1e-9,
        ) or not math.isclose(
                float(trial["cuda_tokens_per_second"]), trial_tokens / trial_cuda,
                rel_tol=1e-12, abs_tol=1e-9,
        ):
            raise ValueError(
                f"order {order} {measurement} trial {index} throughput changed."
            )
    if not math.isclose(
            sum(float(trial["wall_seconds"]) for trial in trials), wall,
            rel_tol=1e-12, abs_tol=1e-9,
    ) or not math.isclose(
            sum(float(trial["cuda_seconds"]) for trial in trials), cuda,
            rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise ValueError(f"order {order} {measurement} trial totals changed.")


def summarize_weighted_h8_performance(
        candidate_timings: Mapping[str, Mapping[str, Any]],
        control_timings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidate_timings) != {"0,1", "1,0"} or set(control_timings) != {
        "0,1", "1,0"
    }:
        raise ValueError("weighted H8 requires reciprocal orders 0,1 and 1,0.")
    candidate_tps: dict[str, float] = {}
    control_tps: dict[str, float] = {}
    gains: dict[str, float] = {}
    for order in ("0,1", "1,0"):
        candidate = candidate_timings[order]
        control = control_timings[order]
        _validate_timing(candidate, order=order, measurement="complete_hybrid_b2_h8")
        _validate_timing(control, order=order, measurement="serial_private_b1_graph_h8")
        candidate_tps[order] = float(candidate["wall_tokens_per_second"])
        control_tps[order] = float(control["wall_tokens_per_second"])
        gains[order] = candidate_tps[order] / control_tps[order] - 1.0
    worst = min(candidate_tps.values())
    minimum_gain = min(gains.values())
    absolute_pass = bool(
        worst > REQUIRED_WEIGHTED_COMPLETE_TPS
        and not math.isclose(
            worst,
            REQUIRED_WEIGHTED_COMPLETE_TPS,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    relative_pass = bool(
        minimum_gain > 0.05
        and not math.isclose(minimum_gain, 0.05, rel_tol=0.0, abs_tol=1e-12)
    )
    return {
        "candidate_order_wall_tokens_per_second": candidate_tps,
        "control_order_wall_tokens_per_second": control_tps,
        "same_order_gain_fraction": gains,
        "worst_candidate_wall_tokens_per_second": worst,
        "minimum_same_order_gain_fraction": minimum_gain,
        "required_weighted_complete_tokens_per_second": (
            REQUIRED_WEIGHTED_COMPLETE_TPS
        ),
        "absolute_weighted_bar_pass": absolute_pass,
        "five_percent_same_order_bar_pass": relative_pass,
        "performance_pass": bool(absolute_pass and relative_pass),
        "performance_gate_uses": (
            "worst_reciprocal_complete_wall_and_minimum_same_order_gain"
        ),
    }


def validate_weighted_h8_report(report: Mapping[str, Any]) -> None:
    """Validate exactness, timing arithmetic, and the verifier-only boundary."""

    if set(report) != {
        "schema_version", "gate", "claim_scope", "pass", "exactness_pass",
        "resource_ownership_pass", "result", "source_contract", "workload", "event_order",
        "resources", "reset_contract", "orders", "performance", "setup",
        "decision", "runtime_metadata",
    }:
        raise ValueError("weighted H8 top-level report fields changed.")
    if (
            report["schema_version"] != WEIGHTED_H8_SCHEMA_VERSION
            or report["gate"] != WEIGHTED_H8_GATE
            or report["claim_scope"]
            != "verifier_only_real_prefix_bucket576_phase_a_not_runtime"
    ):
        raise ValueError("weighted H8 report identity changed.")
    expected_source = {
        "job_id": REVIEWED_SOURCE_CAPTURE_JOB_ID,
        "commit": REVIEWED_SOURCE_CAPTURE_COMMIT,
        "contract_sha256": REVIEWED_SOURCE_CONTRACT_SHA256,
        "report_file_sha256": REVIEWED_SOURCE_REPORT_FILE_SHA256,
        "digest_override_allowed": False,
    }
    if report["source_contract"] != expected_source:
        raise ValueError("weighted H8 reviewed source changed.")
    expected_workload = {
        "active_prefix_length": ACTIVE_PREFIX_LENGTH,
        "horizon": HORIZON,
        "timing_trials_per_order": TIMING_TRIALS_PER_ORDER,
        "row_seeds": [ROW_SEED, ROW_SEED],
        "row_source_offsets": list(ROW_SOURCE_OFFSETS),
        "row_initial_prefix_lengths": list(ROW_INITIAL_PREFIX_LENGTHS),
        "row_initial_cache_positions": list(ROW_INITIAL_CACHE_POSITIONS),
        "row_first_output_offsets": list(ROW_FIRST_OUTPUT_OFFSETS),
        "processor_prefix_shape": [2, ACTIVE_PREFIX_LENGTH],
        "neutral_pad_token_id": NEUTRAL_PAD_TOKEN_ID,
        "request_shape": "same_seed_staggered_reachable_production_states",
        "distinct_seed_rejected_reason": (
            "source contract authorizes only seed12345; resetting a future row to "
            "seed23456 would fabricate an unreachable request state"
        ),
        "weighted_ceiling_report_sha256": WEIGHTED_CEILING_REPORT_SHA256,
        "required_weighted_complete_tokens_per_second": (
            REQUIRED_WEIGHTED_COMPLETE_TPS
        ),
    }
    if report["workload"] != expected_workload:
        raise ValueError("weighted H8 workload changed.")
    if tuple(report["event_order"]) != EVENT_ORDER:
        raise ValueError("weighted H8 event order changed.")
    if report["resource_ownership_pass"] is not True:
        raise ValueError("weighted H8 resources are not private.")
    resources = report["resources"]
    if not isinstance(resources, Mapping) or resources.get("total_graph_count") != 7:
        raise ValueError("weighted H8 resource evidence changed.")
    if resources.get("model_graph_count") != 2 or resources.get(
            "shared_b2_processor_graph_count"
    ) != 1 or resources.get("private_b1_processor_graph_count") != 2 or resources.get(
            "private_sampling_stop_graph_count"
    ) != 2:
        raise ValueError("weighted H8 graph ownership changed.")
    private_ownership = resources.get("private_b1_processors")
    if (
            not isinstance(private_ownership, Sequence)
            or isinstance(private_ownership, (str, bytes))
            or len(private_ownership) != 2
            or resources.get("private_b1_ownership_pass") is not True
            or resources.get("private_b1_shared_bridge_non_alias") is not True
    ):
        raise ValueError("weighted H8 private-B1 ownership evidence changed.")
    pointer_fields = (
        "processor_input_storage_ptr", "output_prefix_storage_ptr",
        "raw_logits_storage_ptr", "processed_scores_storage_ptr",
        "monotonic_input_has_time_shift_storage_ptr",
        "monotonic_input_last_time_shift_storage_ptr",
        "monotonic_output_has_time_shift_storage_ptr",
        "monotonic_output_last_time_shift_storage_ptr",
    )
    for row, evidence in enumerate(private_ownership):
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "row", "graph_id", "graph_pool_id", "processor_id", *pointer_fields,
        }:
            raise ValueError(f"weighted H8 private-B1 row {row} fields changed.")
        if evidence["row"] != row or not isinstance(evidence["graph_pool_id"], str):
            raise ValueError(f"weighted H8 private-B1 row {row} identity changed.")
        for field in ("graph_id", "processor_id", *pointer_fields):
            if (
                    not isinstance(evidence[field], int)
                    or isinstance(evidence[field], bool)
                    or evidence[field] <= 0
            ):
                raise ValueError(
                    f"weighted H8 private-B1 row {row} field {field} is invalid."
                )
        if (
                evidence["monotonic_input_has_time_shift_storage_ptr"]
                == evidence["monotonic_output_has_time_shift_storage_ptr"]
                or evidence["monotonic_input_last_time_shift_storage_ptr"]
                == evidence["monotonic_output_last_time_shift_storage_ptr"]
        ):
            raise ValueError(
                f"weighted H8 private-B1 row {row} state input/output aliases."
            )
    for field in ("graph_id", "graph_pool_id", "processor_id", *pointer_fields):
        if len({evidence[field] for evidence in private_ownership}) != 2:
            raise ValueError(f"weighted H8 private-B1 field {field} aliases rows.")
    if (
            resources.get("private_b1_state_feedback_pass") is not True
            or resources.get("private_b1_state_feedback_inside_timed_interval")
            is not True
            or resources.get("private_b1_state_feedback_policy")
            != "captured_output_copied_to_captured_input_after_every_private_processor_replay"
    ):
        raise ValueError("weighted H8 private-B1 state feedback changed.")
    reset = report["reset_contract"]
    if not isinstance(reset, Mapping) or any(
            reset.get(field) is not True
            for field in (
                "in_place_cache_restore", "exact_generator_state_restore",
                "static_pointer_stability", "restore_verified_after_capture",
            )
    ):
        raise ValueError("weighted H8 reset contract changed.")
    orders = report["orders"]
    if not isinstance(orders, Mapping) or set(orders) != {"0,1", "1,0"}:
        raise ValueError("weighted H8 reciprocal orders changed.")
    candidate_timings: dict[str, Mapping[str, Any]] = {}
    control_timings: dict[str, Mapping[str, Any]] = {}
    exact_signatures: list[Any] = []
    private_signatures: list[Any] = []
    for order, values in orders.items():
        if not isinstance(values, Mapping) or set(values) != {
            "execution_order", "exactness", "private_control_exactness",
            "candidate_timing", "control_timing", "same_order_gain_fraction",
        }:
            raise ValueError(f"weighted H8 order {order} fields changed.")
        if values["execution_order"] != [int(value) for value in order.split(",")]:
            raise ValueError(f"weighted H8 order {order} execution order changed.")
        exactness = values["exactness"]
        if not isinstance(exactness, Mapping) or set(exactness) != {
            "steps", "accepted_tokens_match", "candidate_tokens_match",
            "stop_behavior_match", "per_step_rng_match", "final_rng_match",
            "raw_logits_allclose", "raw_logits_topk_match",
            "raw_logits_bitwise_match",
            "processed_scores_bitwise_match", "processed_scores_topk_match",
            "sampling_probabilities_bitwise_match", "static_inputs_bitwise_match",
            "padded_prefixes_bitwise_match", "active_cache_allclose",
            "active_cache_bitwise_match",
            "current_cache_written", "future_cache_unchanged",
            "cross_cache_unchanged", "cross_cache_bitwise_match",
            "source_handoff_match", "exactness_pass", "signature",
        }:
            raise ValueError(f"weighted H8 order {order} exactness fields changed.")
        calculation_fields = {
            "raw_logits_bitwise_match", "active_cache_bitwise_match",
            "cross_cache_bitwise_match",
        }
        for field, value in exactness.items():
            if field in {"steps", "signature"}:
                continue
            if field in calculation_fields:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"weighted H8 order {order} exactness {field} must be boolean."
                    )
                continue
            if value is not True:
                raise ValueError(f"weighted H8 order {order} exactness {field} failed.")
        steps = exactness["steps"]
        if not isinstance(steps, Sequence) or len(steps) != HORIZON:
            raise ValueError(f"weighted H8 order {order} step ledger changed.")
        observed_raw_bitwise = True
        observed_cache_bitwise = True
        observed_cross_bitwise = True
        for step_index, step in enumerate(steps):
            if (
                    not isinstance(step, Mapping)
                    or set(step) != {"step", "rows"}
                    or step.get("step") != step_index
            ):
                raise ValueError(f"weighted H8 order {order} step {step_index} changed.")
            rows = step.get("rows")
            if not isinstance(rows, Sequence) or len(rows) != 2:
                raise ValueError(f"weighted H8 order {order} step rows changed.")
            for row, row_value in enumerate(rows):
                expected_row_fields = {
                    "row", "accepted_output_offset", "accepted_token_id",
                    "reference_token_id", "candidate_token_id",
                    "reference_stop", "candidate_stop",
                    "accepted_token_match", "candidate_token_match", "stop_match",
                    "rng_match", "raw_allclose", "raw_topk_match",
                    "raw_bitwise_match", "processed_bitwise_match",
                    "processed_topk_match", "probabilities_bitwise_match",
                    "static_inputs_bitwise_match", "padded_prefix_bitwise_match",
                    "active_cache_allclose", "active_cache_bitwise_match",
                    "current_cache_written", "future_cache_unchanged",
                    "cross_cache_unchanged", "cross_cache_bitwise_match",
                    "raw_logits", "processed_scores", "sampling_probabilities",
                    "rng_state", "prefix", "static_inputs", "cache",
                }
                if (
                        not isinstance(row_value, Mapping)
                        or set(row_value) != expected_row_fields
                        or row_value.get("row") != row
                        or row_value.get("accepted_output_offset")
                        != ROW_FIRST_OUTPUT_OFFSETS[row] + step_index
                ):
                    raise ValueError(
                        f"weighted H8 order {order} step {step_index} row {row} changed."
                    )
                for token_field in (
                        "accepted_token_id", "reference_token_id",
                        "candidate_token_id",
                ):
                    if (
                            not isinstance(row_value[token_field], int)
                            or isinstance(row_value[token_field], bool)
                    ):
                        raise ValueError(
                            f"weighted H8 order {order} row {row} token IDs changed."
                        )
                for stop_field in ("reference_stop", "candidate_stop"):
                    if not isinstance(row_value[stop_field], bool):
                        raise ValueError(
                            f"weighted H8 order {order} row {row} stop state changed."
                        )
                for field in (
                        "raw_bitwise_match", "active_cache_bitwise_match",
                        "cross_cache_bitwise_match",
                ):
                    if not isinstance(row_value.get(field), bool):
                        raise ValueError(
                            f"weighted H8 order {order} step {step_index} row {row} "
                            f"field {field} must be boolean."
                        )
                observed_raw_bitwise = bool(
                    observed_raw_bitwise and row_value["raw_bitwise_match"]
                )
                observed_cache_bitwise = bool(
                    observed_cache_bitwise
                    and row_value["active_cache_bitwise_match"]
                )
                observed_cross_bitwise = bool(
                    observed_cross_bitwise
                    and row_value["cross_cache_bitwise_match"]
                )
                if any(
                        row_value.get(field) is not True
                        for field in (
                            "accepted_token_match", "candidate_token_match",
                            "stop_match", "rng_match", "raw_allclose", "raw_topk_match",
                            "raw_bitwise_match",
                            "processed_bitwise_match", "processed_topk_match",
                            "probabilities_bitwise_match", "static_inputs_bitwise_match",
                            "padded_prefix_bitwise_match", "active_cache_allclose",
                            "active_cache_bitwise_match",
                            "current_cache_written", "future_cache_unchanged",
                            "cross_cache_unchanged", "cross_cache_bitwise_match",
                        )
                        if field not in {
                            "raw_bitwise_match", "active_cache_bitwise_match",
                            "cross_cache_bitwise_match",
                        }
                ):
                    raise ValueError(
                        f"weighted H8 order {order} step {step_index} row {row} failed."
                    )
                _validate_comparison(
                    row_value["raw_logits"],
                    f"weighted H8 order {order} step {step_index} row {row} raw",
                )
                processed_evidence = row_value["processed_scores"]
                if (
                        not isinstance(processed_evidence, Mapping)
                        or set(processed_evidence) != HASH_PAIR_FIELDS | {"topk_match"}
                ):
                    raise ValueError("weighted H8 processed-score evidence changed.")
                _validate_hash_pair(
                    {key: processed_evidence[key] for key in HASH_PAIR_FIELDS},
                    "weighted H8 processed scores",
                )
                if not isinstance(processed_evidence["topk_match"], bool):
                    raise ValueError("weighted H8 processed top-k evidence changed.")
                for evidence_field in (
                        "sampling_probabilities", "rng_state", "prefix",
                ):
                    _validate_hash_pair(
                        row_value[evidence_field],
                        f"weighted H8 {evidence_field}",
                    )
                static = row_value["static_inputs"]
                if not isinstance(static, Mapping) or set(static) != {"before", "after"}:
                    raise ValueError("weighted H8 static-input evidence changed.")
                for phase in ("before", "after"):
                    if not isinstance(static[phase], Mapping) or not static[phase]:
                        raise ValueError("weighted H8 static-input map changed.")
                    for key, evidence in static[phase].items():
                        if not isinstance(key, str):
                            raise ValueError("weighted H8 static-input key changed.")
                        _validate_hash_pair(evidence, f"weighted H8 static {phase} {key}")
                cache = row_value["cache"]
                if not isinstance(cache, Mapping) or set(cache) != {
                    "active", "current", "future_sha256",
                    "future_expected_sha256", "current_nonzero", "cross",
                    "cross_expected_sha256",
                }:
                    raise ValueError("weighted H8 cache evidence changed.")
                _validate_comparison(cache["active"], "weighted H8 active cache")
                _validate_comparison(cache["current"], "weighted H8 current cache")
                _validate_comparison(cache["cross"], "weighted H8 cross cache")
                _validate_sha256(cache["future_sha256"], "weighted H8 future cache")
                _validate_sha256(
                    cache["future_expected_sha256"],
                    "weighted H8 expected future cache",
                )
                if not isinstance(cache["current_nonzero"], bool):
                    raise ValueError("weighted H8 current-cache evidence changed.")
                _validate_sha256(
                    cache["cross_expected_sha256"],
                    "weighted H8 expected cross cache",
                )
                expected_flags = {
                    "accepted_token_match": (
                        row_value["candidate_token_id"]
                        == row_value["accepted_token_id"]
                    ),
                    "candidate_token_match": (
                        row_value["candidate_token_id"]
                        == row_value["reference_token_id"]
                    ),
                    "stop_match": (
                        row_value["candidate_stop"] == row_value["reference_stop"]
                        and not row_value["reference_stop"]
                    ),
                    "rng_match": row_value["rng_state"]["bitwise_match"],
                    "raw_allclose": row_value["raw_logits"]["allclose"],
                    "raw_topk_match": row_value["raw_logits"]["topk_match"],
                    "raw_bitwise_match": (
                        row_value["raw_logits"]["reference_sha256"]
                        == row_value["raw_logits"]["candidate_sha256"]
                    ),
                    "processed_bitwise_match": processed_evidence["bitwise_match"],
                    "processed_topk_match": processed_evidence["topk_match"],
                    "probabilities_bitwise_match": row_value[
                        "sampling_probabilities"
                    ]["bitwise_match"],
                    "static_inputs_bitwise_match": all(
                        evidence["bitwise_match"]
                        for phase in ("before", "after")
                        for evidence in static[phase].values()
                    ),
                    "padded_prefix_bitwise_match": row_value["prefix"][
                        "bitwise_match"
                    ],
                    "active_cache_allclose": cache["active"]["allclose"],
                    "active_cache_bitwise_match": (
                        cache["active"]["reference_sha256"]
                        == cache["active"]["candidate_sha256"]
                    ),
                    "current_cache_written": bool(
                        cache["current"]["allclose"] and cache["current_nonzero"]
                    ),
                    "future_cache_unchanged": (
                        cache["future_sha256"] == cache["future_expected_sha256"]
                    ),
                    "cross_cache_bitwise_match": (
                        cache["cross"]["reference_sha256"]
                        == cache["cross"]["candidate_sha256"]
                    ),
                    "cross_cache_unchanged": bool(
                        cache["cross"]["allclose"]
                        and cache["cross"]["candidate_sha256"]
                        == cache["cross_expected_sha256"]
                    ),
                }
                for field, expected in expected_flags.items():
                    if row_value[field] is not expected:
                        raise ValueError(
                            f"weighted H8 order {order} row {row} {field} "
                            "is inconsistent with its evidence."
                        )
        if (
                exactness["raw_logits_bitwise_match"] != observed_raw_bitwise
                or exactness["active_cache_bitwise_match"]
                != observed_cache_bitwise
                or exactness["cross_cache_bitwise_match"]
                != observed_cross_bitwise
        ):
            raise ValueError(
                f"weighted H8 order {order} calculation parity is inconsistent."
            )
        observed_signature = weighted_h8_steps_signature(steps)
        if exactness["signature"] != observed_signature:
            raise ValueError(
                f"weighted H8 order {order} exact signature is inconsistent."
            )
        exact_signatures.append(observed_signature)
        private_exact = values["private_control_exactness"]
        private_fields = {
            "starting_states", "steps", "starting_monotonic_state_match",
            "offset41_time_shift_retention_match",
            "processed_scores_bitwise_match", "processed_scores_topk_match",
            "sampling_probabilities_bitwise_match", "candidate_tokens_match",
            "per_step_rng_match", "monotonic_state_match",
            "state_feedback_match", "final_state_matches_exact",
            "exactness_pass", "signature",
        }
        if not isinstance(private_exact, Mapping) or set(private_exact) != private_fields:
            raise ValueError(
                f"weighted H8 order {order} private-control fields changed."
            )
        for field in private_fields - {"starting_states", "steps", "signature"}:
            if private_exact[field] is not True:
                raise ValueError(
                    f"weighted H8 order {order} private-control {field} failed."
                )
        starting_states = private_exact["starting_states"]
        if not isinstance(starting_states, Sequence) or len(starting_states) != 2:
            raise ValueError("weighted H8 private-control starting states changed.")
        for row, evidence in enumerate(starting_states):
            if not isinstance(evidence, Mapping) or set(evidence) != {
                "row", "current_output_offset", "current_token_id",
                "has_time_shift", "last_time_shift_value",
                "state_match",
            }:
                raise ValueError("weighted H8 private-control starting evidence changed.")
            if (
                    evidence["row"] != row
                    or evidence["current_output_offset"]
                    != ROW_FIRST_OUTPUT_OFFSETS[row] - 1
                    or not isinstance(evidence["current_token_id"], int)
                    or isinstance(evidence["current_token_id"], bool)
                    or evidence["state_match"] is not True
            ):
                raise ValueError("weighted H8 private-control starting identity changed.")
            _validate_hash_pair(
                evidence["has_time_shift"], "weighted H8 private starting has"
            )
            _validate_hash_pair(
                evidence["last_time_shift_value"],
                "weighted H8 private starting last",
            )
        private_steps = private_exact["steps"]
        if not isinstance(private_steps, Sequence) or len(private_steps) != HORIZON:
            raise ValueError("weighted H8 private-control step ledger changed.")
        for step_index, step in enumerate(private_steps):
            if (
                    not isinstance(step, Mapping)
                    or set(step) != {"step", "rows"}
                    or step["step"] != step_index
                    or not isinstance(step["rows"], Sequence)
                    or len(step["rows"]) != 2
            ):
                raise ValueError("weighted H8 private-control step shape changed.")
            for row, evidence in enumerate(step["rows"]):
                if not isinstance(evidence, Mapping) or set(evidence) != {
                    "row", "accepted_output_offset", "accepted_token_id",
                    "candidate_token_id", "processed_scores",
                    "sampling_probabilities", "rng_state", "has_time_shift",
                    "last_time_shift_value", "feedback_has_time_shift",
                    "feedback_last_time_shift_value", "processed_bitwise_match",
                    "processed_topk_match", "probabilities_bitwise_match",
                    "candidate_token_match", "rng_match", "monotonic_state_match",
                    "state_feedback_match",
                }:
                    raise ValueError("weighted H8 private-control row evidence changed.")
                if (
                        evidence["row"] != row
                        or evidence["accepted_output_offset"]
                        != ROW_FIRST_OUTPUT_OFFSETS[row] + step_index
                        or not isinstance(evidence["accepted_token_id"], int)
                        or isinstance(evidence["accepted_token_id"], bool)
                        or not isinstance(evidence["candidate_token_id"], int)
                        or isinstance(evidence["candidate_token_id"], bool)
                ):
                    raise ValueError("weighted H8 private-control row identity changed.")
                for field in (
                        "processed_bitwise_match", "processed_topk_match",
                        "probabilities_bitwise_match", "candidate_token_match",
                        "rng_match", "monotonic_state_match",
                        "state_feedback_match",
                ):
                    if evidence[field] is not True:
                        raise ValueError(
                            f"weighted H8 private-control step {step_index} row {row} failed."
                        )
                processed = evidence["processed_scores"]
                if not isinstance(processed, Mapping) or set(processed) != (
                        HASH_PAIR_FIELDS | {"topk_match"}
                ):
                    raise ValueError("weighted H8 private processed evidence changed.")
                _validate_hash_pair(
                    {key: processed[key] for key in HASH_PAIR_FIELDS},
                    "weighted H8 private processed",
                )
                if processed["topk_match"] is not True:
                    raise ValueError("weighted H8 private processed top-k failed.")
                for field in (
                        "sampling_probabilities", "rng_state", "has_time_shift",
                        "last_time_shift_value", "feedback_has_time_shift",
                        "feedback_last_time_shift_value",
                ):
                    _validate_hash_pair(
                        evidence[field], f"weighted H8 private {field}"
                    )
                expected_private_flags = {
                    "processed_bitwise_match": processed["bitwise_match"],
                    "processed_topk_match": processed["topk_match"],
                    "probabilities_bitwise_match": evidence[
                        "sampling_probabilities"
                    ]["bitwise_match"],
                    "candidate_token_match": (
                        evidence["candidate_token_id"]
                        == evidence["accepted_token_id"]
                    ),
                    "rng_match": evidence["rng_state"]["bitwise_match"],
                    "monotonic_state_match": bool(
                        evidence["has_time_shift"]["bitwise_match"]
                        and evidence["last_time_shift_value"]["bitwise_match"]
                    ),
                    "state_feedback_match": bool(
                        evidence["feedback_has_time_shift"]["bitwise_match"]
                        and evidence["feedback_last_time_shift_value"][
                            "bitwise_match"
                        ]
                    ),
                }
                for field, expected in expected_private_flags.items():
                    if evidence[field] is not expected:
                        raise ValueError(
                            f"weighted H8 private-control step {step_index} row "
                            f"{row} {field} is inconsistent with its evidence."
                        )
        private_signature = weighted_h8_steps_signature(
            [*starting_states, *private_steps]
        )
        if private_exact["signature"] != private_signature:
            raise ValueError("weighted H8 private-control signature is inconsistent.")
        private_signatures.append(private_signature)
        candidate_timings[order] = values["candidate_timing"]
        control_timings[order] = values["control_timing"]
    if exact_signatures[0] != exact_signatures[1]:
        raise ValueError("weighted H8 exact evidence is reciprocal-order dependent.")
    if private_signatures[0] != private_signatures[1]:
        raise ValueError(
            "weighted H8 private-control evidence is reciprocal-order dependent."
        )
    raw_bitwise = all(
        bool(values["exactness"]["raw_logits_bitwise_match"])
        for values in orders.values()
    )
    cache_bitwise = all(
        bool(values["exactness"]["active_cache_bitwise_match"])
        for values in orders.values()
    )
    cross_bitwise = all(
        bool(values["exactness"]["cross_cache_bitwise_match"])
        for values in orders.values()
    )
    expected_classification = (
        "bitwise-calculation-exact"
        if raw_bitwise and cache_bitwise and cross_bitwise
        else "exact-sampled-prefix"
    )
    if report["result"] != {
        "scope": "h8_sampled_prefix_scout_not_full_exact_output_or_osu",
        "classification": expected_classification,
        "raw_logits_bitwise_match": raw_bitwise,
        "active_cache_bitwise_match": cache_bitwise,
        "cross_cache_bitwise_match": cross_bitwise,
        "full_exact_output_claimed": False,
    }:
        raise ValueError("weighted H8 result classification changed.")
    performance = summarize_weighted_h8_performance(
        candidate_timings, control_timings
    )
    if report["performance"] != performance:
        raise ValueError("weighted H8 performance arithmetic changed.")
    for order, values in orders.items():
        if not math.isclose(
                float(values["same_order_gain_fraction"]),
                float(performance["same_order_gain_fraction"][order]),
                rel_tol=1e-12, abs_tol=1e-9,
        ):
            raise ValueError(f"weighted H8 order {order} gain changed.")
    expected_pass = bool(report["exactness_pass"] and performance["performance_pass"])
    if report["exactness_pass"] is not True or report["pass"] != expected_pass:
        raise ValueError("weighted H8 final pass changed.")
    if report["decision"] != {
        "phase_a_only": True,
        "phase_b_authorized": False,
        "weighted_bucket_sweep_authorized": expected_pass,
        "scheduler_or_runtime_wiring_authorized": False,
        "server_authorized": False,
    }:
        raise ValueError("weighted H8 graduation boundary changed.")
    setup = report["setup"]
    if not isinstance(setup, Mapping) or set(setup) != {
        "excluded_from_performance", "excluded_components", "total_gate_wall_seconds",
        "graph_capture_count", "model_graphs", "shared_b2_processor_graph",
        "private_b1_processor_graphs", "private_sampling_graphs",
        "extension_cache_before", "extension_cache_after",
        "memory_current_allocated_bytes", "memory_current_reserved_bytes",
        "memory_peak_allocated_bytes", "memory_peak_reserved_bytes",
    } or setup.get("excluded_from_performance") is not True:
        raise ValueError("weighted H8 setup accounting changed.")
    if setup["excluded_components"] != [
        "source_reconstruction", "prefill", "graph_warmup", "graph_capture"
    ] or setup["graph_capture_count"] != 7:
        raise ValueError("weighted H8 excluded setup components changed.")
    _finite_positive(setup["total_gate_wall_seconds"], "weighted H8 setup wall")
    for field, count in (
            ("model_graphs", 2), ("private_b1_processor_graphs", 2),
            ("private_sampling_graphs", 2),
    ):
        values = setup[field]
        if not isinstance(values, Sequence) or len(values) != count:
            raise ValueError(f"weighted H8 setup field {field} changed.")
    if not isinstance(setup["shared_b2_processor_graph"], Mapping):
        raise ValueError("weighted H8 shared processor setup changed.")
    for values in (
            setup["model_graphs"], setup["private_b1_processor_graphs"],
            setup["private_sampling_graphs"],
            [setup["shared_b2_processor_graph"]],
    ):
        for value in values:
            if (
                    not isinstance(value, Mapping)
                    or not isinstance(value.get("warmup_seconds"), (int, float))
                    or not isinstance(value.get("capture_seconds"), (int, float))
                    or not math.isfinite(float(value["warmup_seconds"]))
                    or not math.isfinite(float(value["capture_seconds"]))
                    or float(value["warmup_seconds"]) < 0.0
                    or float(value["capture_seconds"]) < 0.0
            ):
                raise ValueError("weighted H8 graph setup timing changed.")
    extension_fields = {"path", "exists", "entry_count", "file_count"}
    for field in ("extension_cache_before", "extension_cache_after"):
        cache_state = setup[field]
        if (
                not isinstance(cache_state, Mapping)
                or set(cache_state) != extension_fields
                or not isinstance(cache_state["exists"], bool)
                or not isinstance(cache_state["entry_count"], int)
                or isinstance(cache_state["entry_count"], bool)
                or cache_state["entry_count"] < 0
                or not isinstance(cache_state["file_count"], int)
                or isinstance(cache_state["file_count"], bool)
                or cache_state["file_count"] < 0
                or (cache_state["path"] is not None
                    and not isinstance(cache_state["path"], str))
        ):
            raise ValueError("weighted H8 extension-cache evidence changed.")
    for field in (
            "memory_current_allocated_bytes", "memory_current_reserved_bytes",
            "memory_peak_allocated_bytes", "memory_peak_reserved_bytes",
    ):
        if not isinstance(setup[field], int) or setup[field] < 0:
            raise ValueError(f"weighted H8 setup memory field {field} changed.")
    if (
            setup["memory_peak_allocated_bytes"]
            < setup["memory_current_allocated_bytes"]
            or setup["memory_peak_reserved_bytes"]
            < setup["memory_current_reserved_bytes"]
    ):
        raise ValueError("weighted H8 setup peak memory is below current memory.")
    runtime = report["runtime_metadata"]
    if not isinstance(runtime, Mapping) or runtime.get("h8_executed") is not True:
        raise ValueError("weighted H8 runtime metadata changed.")
    if runtime.get("scheduler_or_runtime_wiring_authorized") is not False:
        raise ValueError("weighted H8 runtime crossed its verifier boundary.")
    if (
            runtime.get("device") != "cuda"
            or runtime.get("precision") != "fp32"
            or runtime.get("gpu_capability") != [7, 5]
            or "RTX 2080" not in str(runtime.get("gpu_name"))
            or runtime.get("model_max_target_positions") != 2560
            or not isinstance(runtime.get("torch_version"), str)
            or not isinstance(runtime.get("cuda_version"), str)
            or not isinstance(runtime.get("git_commit"), str)
            or len(runtime["git_commit"]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in runtime["git_commit"]
            )
            or not isinstance(runtime.get("slurm_job_id"), str)
            or not runtime["slurm_job_id"]
    ):
        raise ValueError("weighted H8 hardware/runtime contract changed.")
