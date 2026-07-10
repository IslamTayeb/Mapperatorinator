"""Contracts for the bounded changing-prefix hybrid B2 verifier.

This remains verifier infrastructure.  A pass proves only that the fixed
SALVALAI sequence-9 bucket-128 shape can advance real token/cache state.  It
does not authorize scheduler or production runtime wiring.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .hybrid_l2 import (
    REVIEWED_FRAMES_SHA256,
    REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
    REVIEWED_PROMPT_SHA256,
    REVIEWED_WORKLOAD_CONTRACT_SHA256,
    HybridBridgeResourceEvidence,
    HybridSamplingResourceEvidence,
    validate_hybrid_resource_ownership,
)
from .lane_one_token import LaneResourceEvidence


CHANGING_PREFIX_SCHEMA_VERSION = 1
CHANGING_PREFIX_GATE = "changing_prefix_bucket128_hybrid_two_lane"
PHASE_A_HORIZON = 8
PHASE_B_HORIZON = 16
PHASE_B_TIMING_TRIALS = 5
ACTIVE_PREFIX_LENGTH = 128
NEUTRAL_PAD_TOKEN_ID = 0
TARGET_TPS = 500.0
FIVE_PERCENT_KEEP_TPS = 435.64993222366656

FIXED_HYBRID_SOURCE_JOB = "49552768"
FIXED_HYBRID_SOURCE_COMMIT = "682abdc0c42743c31940e38ecf0c40494f44dac0"
FIXED_HYBRID_SOURCE_REPORT_SHA256 = (
    "d7f47e5c174a8f3504401ec0d428b0070f4724e95971efc150c3d13007f343ef"
)
FIXED_HYBRID_SOURCE_WORST_TPS = 670.0264805398129

# Accepted exact production traces, not synthetic probes.  This deliberately
# prevents a fixed bucket-128 result from being presented as a queue result.
PRODUCTION_BUCKET128_DECODE_TOKENS = 305
PRODUCTION_TOTAL_DECODE_TOKENS = 5905
FIXED_PROMPT_LENGTH = 84

CHANGING_PREFIX_EVENT_ORDER = (
    "control_copy_initial_prefix_and_static_inputs",
    "state_ready_record",
    "lane_wait_state_ready",
    "lane_model_graph_replay",
    "lane_output_ready_record",
    "control_wait_lane_output_ready",
    "copy_lane_logits_to_preallocated_B2_bridge",
    "source_logits_released_record",
    "shared_B2_padded_processor_graph_replay",
    "copy_processed_rows_to_private_sampling_inputs",
    "processor_done_record",
    "lane_wait_processor_done",
    "private_sampling_and_stop_graph_replay",
    "sample_done_record",
    "control_wait_sample_done",
    "control_copy_sampled_token_and_next_static_inputs",
)


def _finite_positive(value: Any, name: str) -> float:
    if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def summarize_changing_prefix_performance(
        order_timings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute reciprocal complete-wall throughput from tokens and seconds."""

    if set(order_timings) != {"0,1", "1,0"}:
        raise ValueError("changing-prefix timing requires reciprocal orders 0,1 and 1,0.")
    order_tps: dict[str, float] = {}
    for order, timing in order_timings.items():
        tokens = timing.get("generated_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError(f"order {order} generated_tokens must be a positive integer.")
        wall = _finite_positive(timing.get("wall_seconds"), f"order {order} wall_seconds")
        order_tps[order] = tokens / wall
    worst = min(order_tps.values())
    best = max(order_tps.values())
    return {
        "order_tokens_per_second": dict(sorted(order_tps.items())),
        "worst_order_tokens_per_second": worst,
        "best_order_tokens_per_second": best,
        "reciprocal_order_spread_fraction": best / worst - 1.0,
        "required_five_percent_tokens_per_second": FIVE_PERCENT_KEEP_TPS,
        "target_tokens_per_second": TARGET_TPS,
        "five_percent_keep_bar_pass": worst > FIVE_PERCENT_KEEP_TPS,
        "target_500_bar_pass": worst > TARGET_TPS,
        "performance_gate_uses": "worst_reciprocal_complete_changing_prefix_wall",
    }


def _validate_timing(
        timing: Mapping[str, Any],
        *,
        horizon: int,
        trial_count: int,
        order: str,
        reset_contract: Mapping[str, Any],
        final_state_fingerprints: Mapping[str, Any],
) -> float:
    expected_tokens = 2 * horizon * trial_count
    if timing.get("generated_tokens") != expected_tokens:
        raise ValueError(
            f"phase order {order} must report exactly {expected_tokens} paired tokens."
        )
    if timing.get("horizon") != horizon or timing.get("trial_count") != trial_count:
        raise ValueError(f"phase order {order} timing shape is inconsistent.")
    wall = _finite_positive(timing.get("wall_seconds"), f"phase order {order} wall_seconds")
    cuda = _finite_positive(timing.get("cuda_seconds"), f"phase order {order} cuda_seconds")
    expected_wall_tps = expected_tokens / wall
    expected_cuda_tps = expected_tokens / cuda
    if not math.isclose(
            float(timing.get("wall_tokens_per_second", math.nan)),
            expected_wall_tps,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} wall throughput does not self-validate.")
    if not math.isclose(
            float(timing.get("cuda_tokens_per_second", math.nan)),
            expected_cuda_tps,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} CUDA throughput does not self-validate.")
    for field in (
            "reset_outside_timing",
            "initial_state_copies_inside_timing",
            "per_step_updates_inside_timing",
            "device_stop_flags",
            "static_tensor_storage",
            "sampling_tail_in_cuda_graph",
            "explicit_generators_registered_with_sampling_graphs",
            "no_eos_in_timed_horizon",
    ):
        if timing.get(field) is not True:
            raise ValueError(f"phase order {order} timing requires {field}=true.")
    for field in (
            "per_step_global_synchronize",
            "per_step_stack_or_cat",
            "per_step_python_tensor_allocation",
            "graph_recapture_between_trials",
            "dummy_or_stopped_row_sampled",
    ):
        if timing.get(field) is not False:
            raise ValueError(f"phase order {order} timing requires {field}=false.")
    before_allocated = int(timing.get("memory_before_allocated_bytes", -1))
    before_reserved = int(timing.get("memory_before_reserved_bytes", -1))
    peak_allocated = int(timing.get("peak_allocated_bytes", -1))
    peak_reserved = int(timing.get("peak_reserved_bytes", -1))
    if timing.get("peak_allocated_delta_bytes") != peak_allocated - before_allocated:
        raise ValueError(f"phase order {order} allocated memory delta is inconsistent.")
    if timing.get("peak_reserved_delta_bytes") != peak_reserved - before_reserved:
        raise ValueError(f"phase order {order} reserved memory delta is inconsistent.")
    if peak_allocated != before_allocated or peak_reserved != before_reserved:
        raise ValueError(f"phase order {order} allocated during the timed loop.")
    trials = timing.get("trials")
    if (
            not isinstance(trials, Sequence)
            or isinstance(trials, (str, bytes))
            or len(trials) != trial_count
    ):
        raise ValueError(f"phase order {order} must retain every reset-separated trial.")
    expected_trial_tokens = 2 * horizon
    if any(item.get("generated_tokens") != expected_trial_tokens for item in trials):
        raise ValueError(f"phase order {order} each trial must contain two tokens per step.")
    for trial_index, trial in enumerate(trials):
        trial_wall = _finite_positive(
            trial.get("wall_seconds"),
            f"phase order {order} trial {trial_index} wall_seconds",
        )
        trial_cuda = _finite_positive(
            trial.get("cuda_seconds"),
            f"phase order {order} trial {trial_index} cuda_seconds",
        )
        if not math.isclose(
                float(trial.get("wall_tokens_per_second", math.nan)),
                expected_trial_tokens / trial_wall,
                rel_tol=1e-12,
                abs_tol=1e-9,
        ):
            raise ValueError(f"phase order {order} trial {trial_index} wall TPS is inconsistent.")
        if not math.isclose(
                float(trial.get("cuda_tokens_per_second", math.nan)),
                expected_trial_tokens / trial_cuda,
                rel_tol=1e-12,
                abs_tol=1e-9,
        ):
            raise ValueError(f"phase order {order} trial {trial_index} CUDA TPS is inconsistent.")
        if trial.get("reset_evidence") != reset_contract:
            raise ValueError(
                f"phase order {order} trial {trial_index} reset evidence changed."
            )
        if trial.get("initial_rng_state_hashes") != reset_contract.get(
                "post_anchor_rng_state_hashes"
        ):
            raise ValueError(
                f"phase order {order} trial {trial_index} did not start post-anchor RNG."
            )
        if trial.get("final_stop_flags") != {"row-0": False, "row-1": False}:
            raise ValueError(f"phase order {order} trial {trial_index} observed EOS.")
        if trial.get("final_state_fingerprints") != final_state_fingerprints:
            raise ValueError(
                f"phase order {order} trial {trial_index} final state differs from exact decode."
            )
    if not math.isclose(
            sum(float(item.get("wall_seconds", math.nan)) for item in trials),
            wall,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} trial wall seconds are inconsistent.")
    if not math.isclose(
            sum(float(item.get("cuda_seconds", math.nan)) for item in trials),
            cuda,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} trial CUDA seconds are inconsistent.")
    return expected_wall_tps


def _validate_control_timing(
        timing: Mapping[str, Any],
        *,
        horizon: int,
        trial_count: int,
        order: str,
        reset_contract: Mapping[str, Any],
        final_state_fingerprints: Mapping[str, Any],
) -> float:
    """Validate the ordinary-private-B1 diagnostic without candidate claims."""

    expected_trial_tokens = 2 * horizon
    expected_tokens = expected_trial_tokens * trial_count
    if (
            timing.get("measurement") != "ordinary_private_B1_changing_prefix_control"
            or timing.get("generated_tokens") != expected_tokens
            or timing.get("horizon") != horizon
            or timing.get("trial_count") != trial_count
    ):
        raise ValueError(f"phase order {order} same-job control shape is inconsistent.")
    wall = _finite_positive(timing.get("wall_seconds"), f"order {order} control wall")
    cuda = _finite_positive(timing.get("cuda_seconds"), f"order {order} control CUDA")
    if not math.isclose(
            float(timing.get("wall_tokens_per_second", math.nan)),
            expected_tokens / wall,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ) or not math.isclose(
            float(timing.get("cuda_tokens_per_second", math.nan)),
            expected_tokens / cuda,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} control throughput is inconsistent.")
    trials = timing.get("trials")
    if (
            not isinstance(trials, Sequence)
            or isinstance(trials, (str, bytes))
            or len(trials) != trial_count
    ):
        raise ValueError(f"phase order {order} control trials are malformed.")
    for trial_index, trial in enumerate(trials):
        if (
                trial.get("generated_tokens") != expected_trial_tokens
                or trial.get("reset_evidence") != reset_contract
                or trial.get("initial_rng_state_hashes")
                != reset_contract.get("post_anchor_rng_state_hashes")
                or trial.get("final_state_fingerprints") != final_state_fingerprints
        ):
            raise ValueError(f"phase order {order} control trial {trial_index} reset is invalid.")
        trial_wall = _finite_positive(
            trial.get("wall_seconds"), f"order {order} control trial {trial_index} wall"
        )
        trial_cuda = _finite_positive(
            trial.get("cuda_seconds"), f"order {order} control trial {trial_index} CUDA"
        )
        if not math.isclose(
                float(trial.get("wall_tokens_per_second", math.nan)),
                expected_trial_tokens / trial_wall,
                rel_tol=1e-12,
                abs_tol=1e-9,
        ) or not math.isclose(
                float(trial.get("cuda_tokens_per_second", math.nan)),
                expected_trial_tokens / trial_cuda,
                rel_tol=1e-12,
                abs_tol=1e-9,
        ):
            raise ValueError(f"phase order {order} control trial {trial_index} TPS is invalid.")
    if not math.isclose(
            sum(float(trial["wall_seconds"]) for trial in trials),
            wall,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ) or not math.isclose(
            sum(float(trial["cuda_seconds"]) for trial in trials),
            cuda,
            rel_tol=1e-12,
            abs_tol=1e-9,
    ):
        raise ValueError(f"phase order {order} control trial totals are inconsistent.")
    if timing.get("promotion_gate_uses_control") is not False:
        raise ValueError("same-job B1 control is diagnostic only, not a promotion denominator.")
    return expected_tokens / wall


def _hash_pair_match(value: Mapping[str, Any], *, label: str) -> bool:
    reference = value.get("reference_sha256")
    candidate = value.get("candidate_sha256")
    if not isinstance(reference, str) or not reference or not isinstance(candidate, str) or not candidate:
        raise ValueError(f"{label} requires nonempty reference/candidate hashes.")
    if not isinstance(value.get("bitwise_match"), bool):
        raise ValueError(f"{label} bitwise_match must be boolean.")
    recomputed = reference == candidate
    if value.get("bitwise_match") != recomputed:
        raise ValueError(f"{label} bitwise_match does not match its hashes.")
    return recomputed


def _validate_numeric_comparison(value: Mapping[str, Any], *, label: str) -> tuple[bool, bool]:
    for field in ("shape_match", "allclose", "topk_match"):
        if not isinstance(value.get(field), bool):
            raise ValueError(f"{label} field {field} must be boolean.")
    reference_shape = value.get("reference_shape")
    candidate_shape = value.get("candidate_shape")
    if not isinstance(reference_shape, list) or not isinstance(candidate_shape, list):
        raise ValueError(f"{label} shapes must be lists.")
    if value["shape_match"] != (reference_shape == candidate_shape):
        raise ValueError(f"{label} shape_match is inconsistent.")
    if (value["allclose"] or value["topk_match"]) and not value["shape_match"]:
        raise ValueError(f"{label} cannot match values when tensor shapes differ.")
    reference_topk = value.get("reference_topk")
    candidate_topk = value.get("candidate_topk")
    if not isinstance(reference_topk, list) or not isinstance(candidate_topk, list):
        raise ValueError(f"{label} top-k values must be lists.")
    if value["topk_match"] != (reference_topk == candidate_topk):
        raise ValueError(f"{label} topk_match is inconsistent.")
    for field in ("reference_sha256", "candidate_sha256"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{label} requires a nonempty {field}.")
    max_abs = value.get("max_abs")
    if (
            not isinstance(max_abs, (int, float))
            or isinstance(max_abs, bool)
            or not math.isfinite(float(max_abs))
            or float(max_abs) < 0.0
    ):
        raise ValueError(f"{label} max_abs must be finite and non-negative.")
    return bool(value["allclose"]), bool(value["topk_match"])


def _validate_row_step(
        row_value: Mapping[str, Any],
        *,
        row: int,
        step: int,
        previous_raw_hash: str | None,
) -> tuple[dict[str, bool], str, str, str, bool, dict[str, str]]:
    label = f"step {step} row {row}"
    if row_value.get("row") != row:
        raise ValueError(f"{label} row identifier changed.")
    raw = row_value.get("raw_logits")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} is missing raw logits evidence.")
    raw_hash = raw.get("candidate_sha256")
    if not isinstance(raw.get("reference_sha256"), str) or not isinstance(raw_hash, str):
        raise ValueError(f"{label} raw logits hashes are malformed.")
    raw_allclose, raw_topk = _validate_numeric_comparison(raw, label=f"{label} raw logits")
    expected_changed = previous_raw_hash is not None and raw_hash != previous_raw_hash
    if raw.get("changed_from_previous") != (None if previous_raw_hash is None else expected_changed):
        raise ValueError(f"{label} raw-logit change evidence is inconsistent.")

    processed = row_value.get("processed_scores")
    if not isinstance(processed, Mapping):
        raise ValueError(f"{label} is missing processed-score evidence.")
    processed_bitwise = _hash_pair_match(processed, label=f"{label} processed scores")
    processed_topk = _validate_numeric_comparison(
        processed,
        label=f"{label} processed scores",
    )[1]

    token = row_value.get("sampled_token")
    if not isinstance(token, Mapping):
        raise ValueError(f"{label} is missing sampled-token evidence.")
    if any(
            not isinstance(token.get(field), int)
            or isinstance(token.get(field), bool)
            or token[field] < 0
            for field in ("reference", "candidate")
    ):
        raise ValueError(f"{label} sampled token IDs must be non-negative integers.")
    token_match = token.get("reference") == token.get("candidate")
    if not isinstance(token.get("match"), bool):
        raise ValueError(f"{label} sampled-token match must be boolean.")
    if token.get("match") != token_match:
        raise ValueError(f"{label} sampled-token match is inconsistent.")

    rng = row_value.get("rng_state")
    if not isinstance(rng, Mapping):
        raise ValueError(f"{label} is missing RNG evidence.")
    rng_match = _hash_pair_match(rng, label=f"{label} RNG state")
    rng_hash = rng.get("candidate_sha256")

    stop = row_value.get("stop")
    if not isinstance(stop, Mapping):
        raise ValueError(f"{label} is missing stop evidence.")
    for field in ("reference", "candidate", "match", "sampled_after_stop"):
        if not isinstance(stop.get(field), bool):
            raise ValueError(f"{label} stop field {field} must be boolean.")
    stop_match = stop.get("reference") == stop.get("candidate")
    if stop.get("match") != stop_match or stop.get("sampled_after_stop") is not False:
        raise ValueError(f"{label} stop behavior is inconsistent.")

    padded = row_value.get("padded_prefix")
    if not isinstance(padded, Mapping):
        raise ValueError(f"{label} is missing padded-prefix evidence.")
    expected_real_length = FIXED_PROMPT_LENGTH + 1 + step
    if (
            padded.get("real_length") != expected_real_length
            or padded.get("full_length") != ACTIVE_PREFIX_LENGTH
            or padded.get("model_cache_position") != FIXED_PROMPT_LENGTH + step
            or padded.get("pad_token_id") != NEUTRAL_PAD_TOKEN_ID
            or padded.get("suffix_all_pad") is not True
            or padded.get("pad_outside_sos_ids") is not True
            or padded.get("pad_outside_time_shift_range") is not True
    ):
        raise ValueError(f"{label} padded-prefix layout is inconsistent.")
    padded_match = _hash_pair_match(padded, label=f"{label} padded prefix")
    suffix = padded.get("suffix")
    if not isinstance(suffix, Mapping):
        raise ValueError(f"{label} padded suffix evidence is missing.")
    suffix_match = _hash_pair_match(suffix, label=f"{label} padded suffix")

    static = row_value.get("static_inputs")
    if not isinstance(static, Mapping) or set(static) != {
            "decoder_input_ids",
            "cache_position",
            "decoder_position_ids",
            "decoder_attention_mask",
    }:
        raise ValueError(f"{label} static-input evidence is incomplete.")
    static_match = all(
        _hash_pair_match(value, label=f"{label} static input {name}")
        for name, value in static.items()
    )

    cache = row_value.get("cache")
    if not isinstance(cache, Mapping):
        raise ValueError(f"{label} is missing cache evidence.")
    active = cache.get("active_prefix")
    current = cache.get("current_slot")
    future = cache.get("future_suffix")
    cross = cache.get("cross")
    if not all(isinstance(item, Mapping) for item in (active, current, future, cross)):
        raise ValueError(f"{label} cache evidence is incomplete.")
    active_allclose, _ = _validate_numeric_comparison(active, label=f"{label} active cache")
    current_allclose, _ = _validate_numeric_comparison(current, label=f"{label} current cache slot")
    current_written = current.get("candidate_nonzero") is True and current.get("changed_from_reset") is True
    future_unchanged = _hash_pair_match(future, label=f"{label} future cache suffix")
    cross_unchanged = _hash_pair_match(cross, label=f"{label} candidate cross cache")
    cross_reference = cross.get("reference_comparison")
    if not isinstance(cross_reference, Mapping):
        raise ValueError(f"{label} is missing reference cross-cache comparison.")
    cross_allclose, _ = _validate_numeric_comparison(
        cross_reference,
        label=f"{label} reference cross cache",
    )
    post_state = row_value.get("post_step_state")
    if not isinstance(post_state, Mapping):
        raise ValueError(f"{label} is missing post-step state fingerprints.")
    post_rng = post_state.get("rng_state")
    post_prefix = post_state.get("prefix")
    post_static = post_state.get("next_static_inputs")
    post_self = post_state.get("self_cache")
    post_cross = post_state.get("cross_cache")
    if not all(
            isinstance(item, Mapping)
            for item in (post_rng, post_prefix, post_static, post_self, post_cross)
    ):
        raise ValueError(f"{label} post-step reference/candidate evidence is incomplete.")
    if not _hash_pair_match(post_rng, label=f"{label} post-step RNG state"):
        raise ValueError(f"{label} post-step RNG state differs from reference.")
    if post_rng["candidate_sha256"] != rng_hash:
        raise ValueError(f"{label} post-step RNG fingerprint differs from sampled RNG.")
    if not _hash_pair_match(post_prefix, label=f"{label} post-step prefix"):
        raise ValueError(f"{label} post-step prefix differs from reference.")
    if set(post_static) != {
            "decoder_input_ids",
            "cache_position",
            "decoder_position_ids",
            "decoder_attention_mask",
    } or not all(
            _hash_pair_match(value, label=f"{label} post-step static input {name}")
            for name, value in post_static.items()
    ):
        raise ValueError(f"{label} post-step static inputs differ from normal preparation.")
    post_self_allclose, _ = _validate_numeric_comparison(
        post_self,
        label=f"{label} post-step self cache",
    )
    post_cross_allclose, _ = _validate_numeric_comparison(
        post_cross,
        label=f"{label} post-step cross cache",
    )
    if not post_self_allclose or not post_cross_allclose:
        raise ValueError(f"{label} post-step cache differs from independent reference.")
    if post_cross["candidate_sha256"] != cross.get("candidate_sha256"):
        raise ValueError(f"{label} post-step cross-cache fingerprint is inconsistent.")
    if (
            post_state.get("executed_cache_position") != FIXED_PROMPT_LENGTH + step
            or post_state.get("next_static_cache_position") != FIXED_PROMPT_LENGTH + step + 1
            or post_state.get("prefix_real_length") != FIXED_PROMPT_LENGTH + 2 + step
    ):
        raise ValueError(f"{label} post-step position/length is inconsistent.")
    return ({
        "raw_allclose": raw_allclose,
        "raw_topk": raw_topk,
        "raw_changed": previous_raw_hash is None or expected_changed,
        "processed_bitwise": processed_bitwise,
        "processed_topk": processed_topk,
        "token": token_match,
        "rng": rng_match,
        "stop": stop_match and stop.get("sampled_after_stop") is False,
        "static": static_match,
        "padding": padded_match and suffix_match,
        "active_cache": active_allclose,
        "current_cache": current_allclose and current_written,
        "future_cache": future_unchanged,
        "cross_unchanged": cross_unchanged,
        "cross_reference": cross_allclose,
    }, raw_hash, str(rng_hash), str(token.get("candidate")), bool(stop.get("candidate")), {
        "post_step_state": repr(sorted(post_state.items())),
        "static_decoder_input_sha256": str(
            static["decoder_input_ids"]["candidate_sha256"]
        ),
        "static_cache_position_sha256": str(
            static["cache_position"]["candidate_sha256"]
        ),
        "padded_prefix_sha256": str(padded["candidate_sha256"]),
        "future_cache_sha256": str(future["candidate_sha256"]),
    })


def _validate_exact_order(
        values: Mapping[str, Any],
        *,
        horizon: int,
        order: str,
        reset_contract: Mapping[str, Any],
) -> tuple[bool, dict[str, list[str]]]:
    if values.get("graph_and_sampling_order") != [int(item) for item in order.split(",")]:
        raise ValueError(f"phase order {order} reciprocal execution order changed.")
    if values.get("committed_steps") != horizon:
        raise ValueError(f"phase order {order} must commit {horizon} exact steps.")
    if values.get("reset_evidence") != reset_contract:
        raise ValueError(f"phase order {order} reset evidence changed.")
    if values.get("initial_rng_state_hashes") != reset_contract.get(
            "post_anchor_rng_state_hashes"
    ):
        raise ValueError(f"phase order {order} did not start from post-anchor RNG state.")
    required_exact_fields = (
        "raw_logits_allclose",
        "raw_logits_topk_match",
        "raw_logits_changed_each_step",
        "processed_rows_bitwise_match",
        "processed_rows_topk_match",
        "sampled_tokens_match",
        "per_step_rng_state_match",
        "final_rng_state_match",
        "stop_behavior_match",
        "static_inputs_bitwise_match",
        "cache_active_prefix_allclose",
        "cache_current_slot_written",
        "cache_future_suffix_unchanged",
        "cross_cache_unchanged_bitwise",
        "cross_cache_reference_allclose",
        "neutral_padding_proof_pass",
        "no_dummy_or_stopped_row_sampled",
        "no_eos_in_horizon",
        "exactness_pass",
    )
    for field in required_exact_fields:
        if not isinstance(values.get(field), bool):
            raise ValueError(f"phase order {order} exact field {field} must be boolean.")
    recomputed = all(bool(values[field]) for field in required_exact_fields[:-1])
    if values["exactness_pass"] != recomputed:
        raise ValueError(f"phase order {order} exactness aggregate is inconsistent.")
    steps = values.get("steps")
    if (
            not isinstance(steps, Sequence)
            or isinstance(steps, (str, bytes))
            or len(steps) != horizon
    ):
        raise ValueError(f"phase order {order} must retain {horizon} per-step exact records.")
    if [int(step.get("step", -1)) for step in steps] != list(range(horizon)):
        raise ValueError(f"phase order {order} per-step records are not contiguous.")
    observed: dict[str, list[str]] = {
        "row-0-token": [], "row-1-token": [],
        "row-0-rng": [], "row-1-rng": [],
        "row-0-raw": [], "row-1-raw": [],
        "row-0-stop": [], "row-1-stop": [],
        "row-0-state": [], "row-1-state": [],
    }
    aggregate = {field: True for field in required_exact_fields[:-1]}
    previous_raw: list[str | None] = [None, None]
    for step_index, step_value in enumerate(steps):
        rows = step_value.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 2:
            raise ValueError(f"phase order {order} step {step_index} requires two rows.")
        for row in range(2):
            checks, raw_hash, rng_hash, token, stopped, state = _validate_row_step(
                rows[row],
                row=row,
                step=step_index,
                previous_raw_hash=previous_raw[row],
            )
            previous_raw[row] = raw_hash
            observed[f"row-{row}-raw"].append(raw_hash)
            observed[f"row-{row}-rng"].append(rng_hash)
            observed[f"row-{row}-token"].append(token)
            observed[f"row-{row}-stop"].append(str(stopped))
            observed[f"row-{row}-state"].append(repr(sorted(state.items())))
            aggregate["raw_logits_allclose"] &= checks["raw_allclose"]
            aggregate["raw_logits_topk_match"] &= checks["raw_topk"]
            aggregate["raw_logits_changed_each_step"] &= checks["raw_changed"]
            aggregate["processed_rows_bitwise_match"] &= checks["processed_bitwise"]
            aggregate["processed_rows_topk_match"] &= checks["processed_topk"]
            aggregate["sampled_tokens_match"] &= checks["token"]
            aggregate["per_step_rng_state_match"] &= checks["rng"]
            aggregate["stop_behavior_match"] &= checks["stop"]
            aggregate["static_inputs_bitwise_match"] &= checks["static"]
            aggregate["cache_active_prefix_allclose"] &= checks["active_cache"]
            aggregate["cache_current_slot_written"] &= checks["current_cache"]
            aggregate["cache_future_suffix_unchanged"] &= checks["future_cache"]
            aggregate["cross_cache_unchanged_bitwise"] &= checks["cross_unchanged"]
            aggregate["cross_cache_reference_allclose"] &= checks["cross_reference"]
            aggregate["neutral_padding_proof_pass"] &= (
                checks["processed_bitwise"] and checks["padding"]
            )
            aggregate["no_dummy_or_stopped_row_sampled"] &= not stopped or step_index == horizon - 1
            aggregate["no_eos_in_horizon"] &= not stopped
    final_rng = values.get("final_rng_state_hashes")
    if not isinstance(final_rng, Mapping) or set(final_rng) != {"row-0", "row-1"}:
        raise ValueError(f"phase order {order} final RNG hashes are malformed.")
    aggregate["final_rng_state_match"] = all(
        final_rng[f"row-{row}"] == observed[f"row-{row}-rng"][-1]
        for row in range(2)
    )
    final_states = values.get("final_state_fingerprints")
    if not isinstance(final_states, Mapping) or set(final_states) != {"row-0", "row-1"}:
        raise ValueError(f"phase order {order} final state fingerprints are malformed.")
    for row in range(2):
        final_observed = steps[-1]["rows"][row]["post_step_state"]
        if final_states[f"row-{row}"] != final_observed:
            raise ValueError(f"phase order {order} final state fingerprint differs from exact step.")
    for field, recomputed_field in aggregate.items():
        if values.get(field) != recomputed_field:
            raise ValueError(f"phase order {order} aggregate {field} is inconsistent.")
    recomputed = all(aggregate.values())
    if values["exactness_pass"] != recomputed:
        raise ValueError(f"phase order {order} exactness aggregate is inconsistent.")
    return recomputed, observed


def _validate_phase(
        phase: Mapping[str, Any],
        *,
        horizon: int,
        trial_count: int,
        reset_contract: Mapping[str, Any],
) -> tuple[bool, bool]:
    if phase.get("horizon") != horizon or phase.get("timing_trials_per_order") != trial_count:
        raise ValueError("changing-prefix phase shape differs from its fixed gate.")
    orders = phase.get("orders")
    if not isinstance(orders, Mapping) or set(orders) != {"0,1", "1,0"}:
        raise ValueError("changing-prefix phase requires both reciprocal orders.")
    exact_values: list[bool] = []
    order_observed: dict[str, dict[str, list[str]]] = {}
    for order, values in orders.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"phase order {order} must be an object.")
        exact, observed = _validate_exact_order(
            values,
            horizon=horizon,
            order=order,
            reset_contract=reset_contract,
        )
        exact_values.append(exact)
        order_observed[order] = observed
    if order_observed["0,1"] != order_observed["1,0"]:
        raise ValueError("changing-prefix exact transcript/RNG/raw evidence is order-dependent.")
    exact_pass = all(exact_values)
    if not exact_pass:
        for order, values in orders.items():
            if values.get("complete_timing") is not None or values.get(
                    "same_job_private_b1_control"
            ) is not None:
                raise ValueError(
                    f"phase order {order} must stop before timing after exactness/EOS failure."
                )
        if phase.get("performance") is not None:
            raise ValueError("a failed exact phase must not report performance evidence.")
        if phase.get("exactness_pass") is not False or phase.get("target_500_gate_pass") is not False:
            raise ValueError("failed changing-prefix phase aggregates are inconsistent.")
        return False, False

    timing_input: dict[str, Mapping[str, Any]] = {}
    for order, values in orders.items():
        timing = values.get("complete_timing")
        if not isinstance(timing, Mapping):
            raise ValueError(f"phase order {order} is missing complete timing.")
        _validate_timing(
            timing,
            horizon=horizon,
            trial_count=trial_count,
            order=order,
            reset_contract=reset_contract,
            final_state_fingerprints=values["final_state_fingerprints"],
        )
        timing_input[order] = timing
        control = values.get("same_job_private_b1_control")
        if not isinstance(control, Mapping):
            raise ValueError(f"phase order {order} is missing same-job B1 control.")
        control_tps = _validate_control_timing(
            control,
            horizon=horizon,
            trial_count=trial_count,
            order=order,
            reset_contract=reset_contract,
            final_state_fingerprints=values["final_state_fingerprints"],
        )
        candidate_tps = timing["generated_tokens"] / timing["wall_seconds"]
        relative = candidate_tps / control_tps - 1.0
        if not math.isclose(
                float(values.get("relative_gain_vs_same_job_control", math.nan)),
                relative,
                rel_tol=1e-12,
                abs_tol=1e-9,
        ):
            raise ValueError(f"phase order {order} control comparison is inconsistent.")
        if values.get("five_percent_vs_same_job_control") != (relative > 0.05):
            raise ValueError(f"phase order {order} five-percent control diagnostic is inconsistent.")
    performance = summarize_changing_prefix_performance(timing_input)
    if phase.get("performance") != performance:
        raise ValueError("changing-prefix phase performance summary does not self-validate.")
    if phase.get("exactness_pass") != exact_pass:
        raise ValueError("changing-prefix phase exactness aggregate is inconsistent.")
    target_pass = bool(exact_pass and performance["target_500_bar_pass"])
    if phase.get("target_500_gate_pass") != target_pass:
        raise ValueError("changing-prefix phase 500 TPS aggregate is inconsistent.")
    return exact_pass, target_pass


def _lane_from_mapping(value: Mapping[str, Any]) -> LaneResourceEvidence:
    payload = dict(value)
    payload["self_cache_storage_ptrs"] = tuple(payload["self_cache_storage_ptrs"])
    payload["cross_cache_storage_ptrs"] = tuple(payload["cross_cache_storage_ptrs"])
    return LaneResourceEvidence(**payload)


def _bridge_from_mapping(value: Mapping[str, Any]) -> HybridBridgeResourceEvidence:
    payload = dict(value)
    payload["source_release_event_ids"] = tuple(payload["source_release_event_ids"])
    payload["lane_ready_event_ids"] = tuple(payload["lane_ready_event_ids"])
    payload["sample_done_event_ids"] = tuple(payload["sample_done_event_ids"])
    payload["sampling_lanes"] = tuple(
        HybridSamplingResourceEvidence(**lane)
        for lane in payload["sampling_lanes"]
    )
    return HybridBridgeResourceEvidence(**payload)


def _validate_resources(resources: Mapping[str, Any]) -> tuple[list[LaneResourceEvidence], HybridBridgeResourceEvidence]:
    lanes_raw = resources.get("model_lanes")
    bridge_raw = resources.get("hybrid_bridge")
    if (
            not isinstance(lanes_raw, Sequence)
            or isinstance(lanes_raw, (str, bytes))
            or len(lanes_raw) != 2
            or not all(isinstance(item, Mapping) for item in lanes_raw)
            or not isinstance(bridge_raw, Mapping)
    ):
        raise ValueError("changing-prefix resources require two model lanes and one bridge.")
    lanes = [_lane_from_mapping(item) for item in lanes_raw]
    bridge = _bridge_from_mapping(bridge_raw)
    validate_hybrid_resource_ownership(lanes, bridge)
    source_views = resources.get("source_logit_views")
    if (
            not isinstance(source_views, Sequence)
            or isinstance(source_views, (str, bytes))
            or len(source_views) != 2
    ):
        raise ValueError("changing-prefix resources require two source-logit views.")
    for row, source in enumerate(source_views):
        if not isinstance(source, Mapping) or source.get("row") != row:
            raise ValueError("changing-prefix source-logit row evidence is malformed.")
        graph_ptr = source.get("graph_output_storage_ptr")
        view_ptr = source.get("view_storage_ptr")
        if not isinstance(graph_ptr, int) or graph_ptr <= 0 or not isinstance(view_ptr, int) or view_ptr <= 0:
            raise ValueError("changing-prefix source-logit pointers must be positive integers.")
        if source.get("zero_copy_alias") != (graph_ptr == view_ptr) or graph_ptr != view_ptr:
            raise ValueError("changing-prefix source logits must alias graph output storage.")
        if source.get("captured_lane_logits_property_used") is not False:
            raise ValueError("CapturedB1Lane.logits is a stale clone and must not feed the bridge.")
    state_ready = resources.get("state_ready_event_ids")
    if (
            not isinstance(state_ready, list)
            or len(state_ready) != 2
            or not all(isinstance(value, int) and value > 0 for value in state_ready)
    ):
        raise ValueError("changing-prefix resources require two state-ready event IDs.")
    all_events = [
        *state_ready,
        *bridge.source_release_event_ids,
        *bridge.lane_ready_event_ids,
        *bridge.sample_done_event_ids,
        bridge.processor_done_event_id,
    ]
    if len(set(all_events)) != len(all_events):
        raise ValueError("changing-prefix state and pipeline event IDs must be pairwise distinct.")
    stop_ptrs = resources.get("stop_flag_storage_ptrs")
    if (
            not isinstance(stop_ptrs, list)
            or len(stop_ptrs) != 2
            or not all(isinstance(value, int) and value > 0 for value in stop_ptrs)
            or len(set(stop_ptrs)) != 2
    ):
        raise ValueError("changing-prefix stop flags must own two private storage pointers.")
    eos_ptrs = resources.get("sampling_eos_token_storage_ptrs")
    if (
            not isinstance(eos_ptrs, list)
            or len(eos_ptrs) != 2
            or not all(isinstance(value, int) and value > 0 for value in eos_ptrs)
            or len(set(eos_ptrs)) != 2
    ):
        raise ValueError("changing-prefix sampling graphs must retain private EOS tensors.")
    stop_output_ptrs = resources.get("sampling_stop_output_storage_ptrs")
    if (
            not isinstance(stop_output_ptrs, list)
            or len(stop_output_ptrs) != 2
            or not all(isinstance(value, int) and value > 0 for value in stop_output_ptrs)
            or len(set(stop_output_ptrs)) != 2
    ):
        raise ValueError("changing-prefix sampling graphs must retain private stop outputs.")
    model_ptrs: set[int] = set()
    for lane in lanes:
        model_ptrs.update(lane.static_input_storage_ptrs.values())
        model_ptrs.update(lane.self_cache_storage_ptrs)
        model_ptrs.update(lane.cross_cache_storage_ptrs)
        model_ptrs.add(lane.encoder_output_storage_ptr)
    sampling_ptrs = {
        pointer
        for lane in bridge.sampling_lanes
        for pointer in (
            lane.processed_scores_storage_ptr,
            lane.probabilities_storage_ptr,
            lane.sampled_token_storage_ptr,
        )
    }
    eos_forbidden = {
        bridge.prefix_batch_storage_ptr,
        bridge.raw_logits_bridge_storage_ptr,
        bridge.processed_scores_storage_ptr,
        *model_ptrs,
        *sampling_ptrs,
        *stop_ptrs,
        *stop_output_ptrs,
    }
    if set(eos_ptrs) & eos_forbidden:
        raise ValueError("changing-prefix retained EOS tensors alias mutable pipeline storage.")
    stop_output_forbidden = {
        bridge.prefix_batch_storage_ptr,
        bridge.raw_logits_bridge_storage_ptr,
        bridge.processed_scores_storage_ptr,
        *model_ptrs,
        *sampling_ptrs,
        *stop_ptrs,
        *eos_ptrs,
    }
    if set(stop_output_ptrs) & stop_output_forbidden:
        raise ValueError("changing-prefix retained stop outputs alias mutable pipeline storage.")
    used_ptrs = {
        bridge.prefix_batch_storage_ptr,
        bridge.raw_logits_bridge_storage_ptr,
        bridge.processed_scores_storage_ptr,
        *model_ptrs,
        *sampling_ptrs,
        *eos_ptrs,
        *stop_output_ptrs,
    }
    if set(stop_ptrs) & used_ptrs:
        raise ValueError("changing-prefix stop flags must not alias mutable pipeline storage.")
    source_ptrs = {int(source["view_storage_ptr"]) for source in source_views}
    if len(source_ptrs) != 2:
        raise ValueError("changing-prefix lanes must own distinct graph-output views.")
    forbidden_source_aliases = {
        bridge.prefix_batch_storage_ptr,
        bridge.raw_logits_bridge_storage_ptr,
        bridge.processed_scores_storage_ptr,
        *model_ptrs,
        *sampling_ptrs,
        *stop_ptrs,
        *eos_ptrs,
        *stop_output_ptrs,
    }
    if source_ptrs & forbidden_source_aliases:
        raise ValueError("changing-prefix graph-output views alias mutable pipeline storage.")
    capture = resources.get("capture")
    expected_capture = {
        "model_graph_count": 2,
        "processor_graph_count": 1,
        "private_sampling_and_stop_graph_count": 2,
        "total_graph_count": 5,
        "explicit_generator_registration": True,
        "graph_recapture_between_orders_or_trials": False,
    }
    if capture != expected_capture:
        raise ValueError("changing-prefix capture evidence changed.")
    return lanes, bridge


def _validate_reset_contract(
        reset: Mapping[str, Any],
        *,
        lanes: Sequence[LaneResourceEvidence],
) -> None:
    request_ids = {"row-0", "row-1"}
    for field in (
            "self_cache_snapshot_hashes",
            "post_restore_self_cache_hashes",
            "cross_cache_snapshot_hashes",
            "post_restore_cross_cache_hashes",
            "initial_seed_rng_state_hashes",
            "post_anchor_rng_state_hashes",
            "anchor_tokens",
            "post_anchor_rng_states",
            "static_input_storage_ptrs",
            "self_cache_storage_ptrs",
            "cross_cache_storage_ptrs",
    ):
        value = reset.get(field)
        if not isinstance(value, Mapping) or set(value) != request_ids:
            raise ValueError(f"changing-prefix reset field {field} requires both rows.")
    for field in (
            "self_cache_snapshot_hashes",
            "post_restore_self_cache_hashes",
            "cross_cache_snapshot_hashes",
            "post_restore_cross_cache_hashes",
            "initial_seed_rng_state_hashes",
            "post_anchor_rng_state_hashes",
    ):
        if any(
                not isinstance(value, str) or not value
                for value in reset[field].values()
        ):
            raise ValueError(f"changing-prefix reset field {field} requires nonempty hashes.")
    if reset["self_cache_snapshot_hashes"] != reset["post_restore_self_cache_hashes"]:
        raise ValueError("changing-prefix self-cache restore does not match its snapshot.")
    if reset["cross_cache_snapshot_hashes"] != reset["post_restore_cross_cache_hashes"]:
        raise ValueError("changing-prefix cross-cache restore does not match its snapshot.")
    if any(
            reset["initial_seed_rng_state_hashes"][request_id]
            == reset["post_anchor_rng_state_hashes"][request_id]
            for request_id in request_ids
    ):
        raise ValueError("changing-prefix sampling must begin after the anchor consumed RNG.")
    for request_id in request_ids:
        anchor = reset["anchor_tokens"][request_id]
        anchor_rng = reset["post_anchor_rng_states"][request_id]
        if not isinstance(anchor, Mapping) or any(
                not isinstance(anchor.get(field), int)
                or isinstance(anchor.get(field), bool)
                or anchor[field] < 0
                for field in ("reference", "candidate")
        ):
            raise ValueError("changing-prefix anchor tokens must be non-negative integers.")
        anchor_match = anchor["reference"] == anchor["candidate"]
        if not isinstance(anchor.get("match"), bool):
            raise ValueError("changing-prefix anchor token match must be boolean.")
        if anchor.get("match") != anchor_match or not anchor_match:
            raise ValueError("changing-prefix candidate anchor differs from independent reference.")
        if not isinstance(anchor_rng, Mapping):
            raise ValueError("changing-prefix post-anchor RNG evidence is malformed.")
        anchor_rng_match = _hash_pair_match(
            anchor_rng,
            label=f"changing-prefix {request_id} post-anchor RNG",
        )
        if not anchor_rng_match or anchor_rng["candidate_sha256"] != reset[
                "post_anchor_rng_state_hashes"
        ][request_id]:
            raise ValueError("changing-prefix post-anchor RNG differs from independent reference.")
    for row, lane in enumerate(lanes):
        request_id = f"row-{row}"
        if reset["static_input_storage_ptrs"][request_id] != lane.static_input_storage_ptrs:
            raise ValueError("changing-prefix static-input pointers changed during reset.")
        if reset["self_cache_storage_ptrs"][request_id] != list(lane.self_cache_storage_ptrs):
            raise ValueError("changing-prefix self-cache pointers changed during reset.")
        if reset["cross_cache_storage_ptrs"][request_id] != list(lane.cross_cache_storage_ptrs):
            raise ValueError("changing-prefix cross-cache pointers changed during reset.")
    if reset.get("in_place_restore") is not True or reset.get("captured_pointer_swap") is not False:
        raise ValueError("changing-prefix reset must restore in place without pointer swaps.")
    if (
            reset.get("cache_snapshot_stage") != "post_prefill_before_anchor_or_capture"
            or reset.get("restore_verified_after_graph_capture") is not True
    ):
        raise ValueError("changing-prefix cache snapshots must pin the post-prefill stage.")


def validate_changing_prefix_report(report: Mapping[str, Any]) -> None:
    """Self-validate staged exactness, resources, timing, and graduation scope."""

    if int(report.get("schema_version", -1)) != CHANGING_PREFIX_SCHEMA_VERSION:
        raise ValueError("unsupported changing-prefix report schema.")
    if report.get("gate") != CHANGING_PREFIX_GATE or report.get("lane_count") != 2:
        raise ValueError("report is not the changing-prefix B2 gate.")
    source = report.get("fixed_hybrid_source")
    expected_source = {
        "job_id": FIXED_HYBRID_SOURCE_JOB,
        "commit": FIXED_HYBRID_SOURCE_COMMIT,
        "report_sha256": FIXED_HYBRID_SOURCE_REPORT_SHA256,
        "worst_order_tokens_per_second": FIXED_HYBRID_SOURCE_WORST_TPS,
    }
    if source != expected_source:
        raise ValueError("changing-prefix fixed-hybrid source differs from audited evidence.")
    contract = report.get("fixed_workload_contract")
    expected_contract = {
        "phase_a_horizon": PHASE_A_HORIZON,
        "phase_b_horizon": PHASE_B_HORIZON,
        "phase_b_timing_trials": PHASE_B_TIMING_TRIALS,
        "active_prefix_length": ACTIVE_PREFIX_LENGTH,
        "prompt_length": FIXED_PROMPT_LENGTH,
        "neutral_pad_token_id": NEUTRAL_PAD_TOKEN_ID,
        "model_path": "OliBomby/Mapperatorinator-v32",
        "precision": "fp32",
        "attn_implementation": "sdpa",
        "sequence_index": 9,
        "row_seeds": [12345, 23456],
        "processor_prefix_shape": [2, ACTIVE_PREFIX_LENGTH],
        "reference_prefix_policy": "private_unpadded_B1",
        "candidate_prefix_policy": "shared_neutral_padded_B2",
        "prompt_sha256": REVIEWED_PROMPT_SHA256,
        "prompt_attention_mask_sha256": REVIEWED_PROMPT_ATTENTION_MASK_SHA256,
        "frames_sha256": REVIEWED_FRAMES_SHA256,
        "condition_tensor_hashes": {},
        "source_workload_contract_sha256": REVIEWED_WORKLOAD_CONTRACT_SHA256,
    }
    if contract != expected_contract:
        raise ValueError("changing-prefix fixed workload contract changed.")
    if tuple(report.get("event_order", ())) != CHANGING_PREFIX_EVENT_ORDER:
        raise ValueError("changing-prefix event order differs from the reviewed sequence.")
    resources = report.get("resource_ownership")
    if not isinstance(resources, Mapping):
        raise ValueError("changing-prefix report is missing resource ownership evidence.")
    lanes, _ = _validate_resources(resources)
    reset_contract = report.get("reset_contract")
    if not isinstance(reset_contract, Mapping):
        raise ValueError("changing-prefix report is missing reset evidence.")
    _validate_reset_contract(reset_contract, lanes=lanes)

    processor = report.get("processor_contract")
    expected_processor = {
        "shared_base_processor_classes": [
            "MonotonicTimeShiftLogitsProcessor",
            "TemperatureLogitsWarper",
        ],
        "private_warper_classes": ["TopPLogitsWarper"],
        "neutral_pad_token_id": NEUTRAL_PAD_TOKEN_ID,
    }
    if not isinstance(processor, Mapping):
        raise ValueError("changing-prefix report is missing processor contract.")
    for field, expected in expected_processor.items():
        if processor.get(field) != expected:
            raise ValueError(f"changing-prefix processor field {field} changed.")
    sos_ids = processor.get("sos_token_ids")
    time_shift_start = processor.get("time_shift_start")
    time_shift_end = processor.get("time_shift_end")
    if (
            not isinstance(sos_ids, list)
            or NEUTRAL_PAD_TOKEN_ID in sos_ids
            or not isinstance(time_shift_start, int)
            or not isinstance(time_shift_end, int)
            or time_shift_start >= time_shift_end
            or time_shift_start <= NEUTRAL_PAD_TOKEN_ID < time_shift_end
    ):
        raise ValueError("neutral pad token is not outside SOS/time-shift semantics.")
    for field, recomputed in (
            ("resource_ownership_pass", True),
            ("capture_pass", True),
            ("pointer_stability_pass", True),
            ("reset_in_place_pass", True),
            ("processor_family_pass", True),
    ):
        if report.get(field) != recomputed:
            raise ValueError(f"changing-prefix report aggregate {field} is inconsistent.")

    phase_a = report.get("phase_a")
    if not isinstance(phase_a, Mapping):
        raise ValueError("changing-prefix report is missing Phase A.")
    phase_a_exact, phase_a_target = _validate_phase(
        phase_a,
        horizon=PHASE_A_HORIZON,
        trial_count=1,
        reset_contract=reset_contract,
    )
    phase_b = report.get("phase_b")
    if phase_a_target:
        if not isinstance(phase_b, Mapping):
            raise ValueError("Phase B is required after exact Phase A exceeds 500 TPS.")
        phase_b_exact, phase_b_target = _validate_phase(
            phase_b,
            horizon=PHASE_B_HORIZON,
            trial_count=PHASE_B_TIMING_TRIALS,
            reset_contract=reset_contract,
        )
    else:
        if phase_b is not None:
            raise ValueError("Phase B must not run when Phase A fails exactness or 500 TPS.")
        phase_b_exact = False
        phase_b_target = False
    expected_pass = bool(phase_a_exact and phase_a_target and phase_b_exact and phase_b_target)
    if report.get("pass") != expected_pass:
        raise ValueError("changing-prefix final pass does not self-validate.")
    scope = report.get("graduation_scope")
    expected_scope = {
        "production_bucket128_decode_tokens": PRODUCTION_BUCKET128_DECODE_TOKENS,
        "production_total_decode_tokens": PRODUCTION_TOTAL_DECODE_TOKENS,
        "production_bucket128_fraction": (
            PRODUCTION_BUCKET128_DECODE_TOKENS / PRODUCTION_TOTAL_DECODE_TOKENS
        ),
        "authorized_next_step": "weighted_active_prefix_bucket_sweep",
        "scheduler_or_runtime_wiring_authorized": False,
    }
    if scope != expected_scope:
        raise ValueError("changing-prefix graduation scope changed.")
