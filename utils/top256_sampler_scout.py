"""Real-logit correctness and timing gate for the bounded top-256 sampler."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from utils.vocab_sampling_scout import (
    VocabSample,
    _capture_graph,
    _representatives_by_vocab_size,
    _sampling_with_threshold,
    _top_p_kwargs,
)


SCHEMA_VERSION = 1
FIXED_MAIN_STEPS = 8294
MAX_CANDIDATE_MS_PER_STEP = 0.035
MIN_FIXED_MAIN_SAVING_SECONDS = 0.3


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _sample_top_p(samples: list[VocabSample]) -> float:
    top_ps: set[float] = set()
    for sample in samples:
        rows = [
            dict(attributes)
            for name, attributes in sample.processor_descriptor
            if name == "TopPLogitsWarper"
        ]
        if len(rows) != 1:
            raise RuntimeError("each captured sample must contain one TopP warper")
        kwargs = _top_p_kwargs(rows[0])
        if kwargs["filter_value"] != -math.inf:
            raise RuntimeError("top-256 scout requires TopP filter_value=-inf")
        if kwargs["min_tokens_to_keep"] != 1:
            raise RuntimeError("top-256 scout requires min_tokens_to_keep=1")
        top_ps.add(float(kwargs["top_p"]))
    if len(top_ps) != 1:
        raise RuntimeError(f"captured samples use multiple top_p values: {sorted(top_ps)}")
    return next(iter(top_ps))


def bounded_reference(
    scores: Any,
    threshold: float,
    *,
    top_p: float,
    top_k: int = 256,
) -> dict[str, Any]:
    """CPU reference for boundedness and original-vocabulary-order sampling."""

    import torch
    from transformers import TopPLogitsWarper

    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a tensor")
    if scores.device.type != "cpu" or scores.dtype != torch.float32:
        raise ValueError("reference scores must use CPU FP32 storage")
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("reference scores must have shape [1, vocab]")
    _positive_int("top_k", top_k)
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be inside (0, 1)")
    if not math.isfinite(float(top_p)) or not 0.0 < float(top_p) < 1.0:
        raise ValueError("top_p must be finite and inside (0, 1)")
    if not bool(torch.isfinite(scores).any()) or bool(torch.isnan(scores).any()) or bool(
        torch.isposinf(scores).any()
    ):
        return {"token": -1, "kept_count": 0, "overflow": True}

    warper = TopPLogitsWarper(
        top_p=float(top_p), filter_value=-math.inf, min_tokens_to_keep=1
    )
    filtered = warper(torch.zeros((1, 1), dtype=torch.long), scores)
    kept = torch.isfinite(filtered)
    kept_count = int(kept.sum().item())
    sorted_scores, sorted_ids = torch.sort(
        scores[0], descending=True, stable=True
    )
    tie_overflow = (
        scores.shape[1] > top_k
        and torch.isfinite(sorted_scores[top_k - 1])
        and sorted_scores[top_k - 1] == sorted_scores[top_k]
    )
    overflow = kept_count > top_k or bool(tie_overflow)
    if overflow:
        token = -1
    else:
        threshold_tensor = torch.tensor(float(threshold), dtype=torch.float32)
        token = int(_sampling_with_threshold(filtered, threshold_tensor).item())
        retained_ids = sorted_ids[:top_k]
        if not bool(kept[0, retained_ids].all()):
            # Some selected top-k elements may be outside the nucleus, which is
            # fine.  Every retained nucleus token must still occur in top-k.
            nucleus_ids = torch.nonzero(kept[0], as_tuple=False).reshape(-1)
            if not set(nucleus_ids.tolist()).issubset(set(retained_ids.tolist())):
                raise RuntimeError("reference nucleus is not contained in top-k")
    return {"token": token, "kept_count": kept_count, "overflow": overflow}


def _time_graph_once(graph: Any, iterations: int) -> float:
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _reciprocal_rounds(
    baseline_graph: Any,
    candidate_graph: Any,
    *,
    iterations: int,
    rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(rounds):
        order = (
            ("baseline", baseline_graph, "candidate", candidate_graph)
            if index % 2 == 0
            else ("candidate", candidate_graph, "baseline", baseline_graph)
        )
        first_name, first_graph, second_name, second_graph = order
        values = {
            first_name: _time_graph_once(first_graph, iterations),
            second_name: _time_graph_once(second_graph, iterations),
        }
        rows.append(
            {
                "order": [first_name, second_name],
                "baseline_ms_per_step": values["baseline"],
                "candidate_ms_per_step": values["candidate"],
                "saving_ms_per_step": values["baseline"] - values["candidate"],
            }
        )
    return rows


def _validate_all_samples(
    samples: list[VocabSample],
    *,
    top_p: float,
) -> dict[str, Any]:
    import torch

    from osuT5.osuT5.inference.optimized.scout.top256_sampler import top256_sample

    failures: list[dict[str, Any]] = []
    overflow_count = 0
    kept_count_mismatches = 0
    threshold_mutation_count = 0
    for index, sample in enumerate(samples):
        scores = sample.pre_top_p.to(device="cuda")
        threshold = torch.tensor(
            sample.threshold, dtype=torch.float32, device=scores.device
        )
        token, kept_count, overflow = top256_sample(
            scores, threshold, top_p=top_p
        )
        torch.cuda.synchronize()
        actual_overflow = bool(overflow.cpu().item())
        actual_count = int(kept_count.cpu().item())
        actual_token = int(token.cpu().item())
        actual_threshold = float(threshold.cpu().item())
        threshold_changed = actual_threshold != sample.threshold
        overflow_count += int(actual_overflow)
        kept_count_mismatches += int(actual_count != sample.nucleus_size)
        threshold_mutation_count += int(threshold_changed)
        if (
            actual_overflow
            or actual_count != sample.nucleus_size
            or actual_token != sample.selected_token
            or threshold_changed
        ):
            failures.append(
                {
                    "sample_index": index,
                    "vocab_size": sample.vocab_size,
                    "source": sample.source,
                    "expected_token": sample.selected_token,
                    "actual_token": actual_token,
                    "expected_kept_count": sample.nucleus_size,
                    "actual_kept_count": actual_count,
                    "overflow": actual_overflow,
                    "threshold": sample.threshold,
                    "threshold_after": actual_threshold,
                }
            )
    return {
        "samples": len(samples),
        "overflow_count": overflow_count,
        "kept_count_mismatches": kept_count_mismatches,
        "threshold_mutation_count": threshold_mutation_count,
        "token_or_overflow_failures": failures,
        "selected_token_exact": not failures,
        "counter_threshold_consumed_unchanged": threshold_mutation_count == 0,
    }


def benchmark_candidate(
    samples: list[VocabSample],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, Any]:
    """Benchmark existing and bounded tails in reciprocal graph order."""

    import torch
    from transformers import TopPLogitsWarper

    from osuT5.osuT5.inference.optimized.scout.top256_sampler import (
        preload_top256_sampler_extension,
        top256_sample,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("top-256 component timing requires CUDA")
    _positive_int("warmup", warmup)
    _positive_int("iterations", iterations)
    _positive_int("rounds", rounds)
    if not samples:
        raise ValueError("top-256 component timing requires real-logit samples")
    for sample in samples:
        sample.validate()
    top_p = _sample_top_p(samples)

    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    extension_started = time.perf_counter()
    preload_top256_sampler_extension()
    torch.cuda.synchronize()
    extension_seconds = time.perf_counter() - extension_started
    extension_allocated_delta = torch.cuda.memory_allocated() - allocated_before
    extension_reserved_delta = torch.cuda.memory_reserved() - reserved_before

    correctness = _validate_all_samples(samples, top_p=top_p)
    representatives = _representatives_by_vocab_size(samples)
    rows: list[dict[str, Any]] = []
    for vocab_size, selected in representatives.items():
        for label, sample in selected.items():
            descriptor = dict(
                next(
                    attributes
                    for name, attributes in sample.processor_descriptor
                    if name == "TopPLogitsWarper"
                )
            )
            kwargs = _top_p_kwargs(descriptor)
            scores = sample.pre_top_p.to(device="cuda")
            threshold = torch.tensor(
                sample.threshold, dtype=torch.float32, device=scores.device
            )
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=scores.device)
            warper = TopPLogitsWarper(**kwargs)

            torch.cuda.synchronize()
            baseline_capture_started = time.perf_counter()
            baseline_graph, baseline_token = _capture_graph(
                lambda: _sampling_with_threshold(
                    warper(input_ids, scores), threshold
                ),
                warmup=warmup,
            )
            baseline_capture_seconds = time.perf_counter() - baseline_capture_started
            torch.cuda.synchronize()
            candidate_capture_started = time.perf_counter()
            candidate_graph, candidate_outputs = _capture_graph(
                lambda: top256_sample(scores, threshold, top_p=top_p),
                warmup=warmup,
            )
            candidate_capture_seconds = time.perf_counter() - candidate_capture_started
            candidate_token, candidate_count, candidate_overflow = candidate_outputs
            torch.cuda.synchronize()
            baseline_value = int(baseline_token.cpu().item())
            candidate_value = int(candidate_token.cpu().item())
            overflow_value = bool(candidate_overflow.cpu().item())
            kept_value = int(candidate_count.cpu().item())
            if baseline_value != sample.selected_token:
                raise RuntimeError("recaptured baseline token differs from observation")

            reciprocal = _reciprocal_rounds(
                baseline_graph,
                candidate_graph,
                iterations=iterations,
                rounds=rounds,
            )
            rows.append(
                {
                    "vocab_size": vocab_size,
                    "score_shape": [1, vocab_size],
                    "representative": label,
                    "nucleus_size": sample.nucleus_size,
                    "threshold": sample.threshold,
                    "baseline_token": baseline_value,
                    "candidate_token": candidate_value,
                    "candidate_kept_count": kept_value,
                    "candidate_overflow": overflow_value,
                    "selected_token_exact": (
                        not overflow_value
                        and candidate_value == sample.selected_token
                        and kept_value == sample.nucleus_size
                    ),
                    "baseline_capture_seconds": baseline_capture_seconds,
                    "candidate_capture_seconds": candidate_capture_seconds,
                    "rounds": reciprocal,
                    "baseline_ms_per_step_worst": max(
                        row["baseline_ms_per_step"] for row in reciprocal
                    ),
                    "candidate_ms_per_step_worst": max(
                        row["candidate_ms_per_step"] for row in reciprocal
                    ),
                    "saving_ms_per_step_worst_pair": min(
                        row["saving_ms_per_step"] for row in reciprocal
                    ),
                }
            )
    return {
        "top_p": top_p,
        "warmup": warmup,
        "iterations": iterations,
        "rounds_per_representative": rounds,
        "reciprocal_order": "alternating baseline/candidate by round",
        "extension_setup": {
            "seconds": extension_seconds,
            "allocated_bytes_delta": extension_allocated_delta,
            "reserved_bytes_delta": extension_reserved_delta,
            "charged_to_replay": False,
        },
        "correctness": correctness,
        "representatives": rows,
    }


def fixed_physical_work(
    output_dir: Path,
    *,
    fixed_main_steps: int = FIXED_MAIN_STEPS,
) -> dict[str, Any]:
    """Extract the selected K4/K1 physical-step multiplier from one profile."""

    import json

    fixed_main_steps = _positive_int("fixed_main_steps", fixed_main_steps)
    profiles = list(output_dir.glob("*.profile.json"))
    if len(profiles) != 1:
        raise RuntimeError(
            f"expected one inference profile under {output_dir}, got {len(profiles)}"
        )
    payload = json.loads(profiles[0].read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("generation", [])
        if row.get("profile_label") == "main_generation"
    ]
    if not rows:
        raise RuntimeError("inference profile has no main-generation records")
    logical = 0
    physical = 0
    wasted = 0
    for row in rows:
        graphs = row.get("optimized_cuda_graphs")
        candidate = graphs.get("k8_candidate") if isinstance(graphs, dict) else None
        if not isinstance(candidate, dict):
            raise RuntimeError("main record lacks K4/K1 physical-work evidence")
        logical += int(candidate.get("logical_steps", 0))
        physical += int(candidate.get("physical_steps", 0))
        wasted += int(candidate.get("wasted_steps", 0))
    if logical <= 0 or physical < logical or physical - logical != wasted:
        raise RuntimeError("K4/K1 physical-work evidence is inconsistent")
    return {
        "observed_logical_steps": logical,
        "observed_physical_steps": physical,
        "observed_wasted_steps": wasted,
        "physical_per_logical": physical / logical,
        "fixed_logical_steps": fixed_main_steps,
        "fixed_physical_steps": fixed_main_steps * physical / logical,
        "profile_path": str(profiles[0]),
    }


def summarize_candidate(
    component: dict[str, Any],
    physical_work: dict[str, Any],
    *,
    max_candidate_ms_per_step: float = MAX_CANDIDATE_MS_PER_STEP,
    minimum_saving_seconds: float = MIN_FIXED_MAIN_SAVING_SECONDS,
) -> dict[str, Any]:
    max_candidate_ms_per_step = _finite_nonnegative(
        "max_candidate_ms_per_step", max_candidate_ms_per_step
    )
    minimum_saving_seconds = _finite_nonnegative(
        "minimum_saving_seconds", minimum_saving_seconds
    )
    rows = component.get("representatives")
    if not isinstance(rows, list) or not rows:
        raise ValueError("component report has no representative rows")
    correctness = component.get("correctness")
    if not isinstance(correctness, dict):
        raise TypeError("component correctness must be an object")
    fixed_physical_steps = _finite_nonnegative(
        "fixed_physical_steps", physical_work.get("fixed_physical_steps")
    )
    if fixed_physical_steps <= 0.0:
        raise ValueError("fixed physical work must be positive")

    baseline_ms = max(
        _finite_nonnegative(
            "baseline_ms_per_step_worst", row.get("baseline_ms_per_step_worst")
        )
        for row in rows
    )
    candidate_ms = max(
        _finite_nonnegative(
            "candidate_ms_per_step_worst", row.get("candidate_ms_per_step_worst")
        )
        for row in rows
    )
    overflow_fraction = (
        int(correctness.get("overflow_count", 0))
        / _positive_int("correctness.samples", correctness.get("samples"))
    )
    fallback_charge_ms = overflow_fraction * baseline_ms
    charged_candidate_ms = candidate_ms + fallback_charge_ms
    baseline_capture = sum(float(row["baseline_capture_seconds"]) for row in rows)
    candidate_capture = sum(float(row["candidate_capture_seconds"]) for row in rows)
    capture_delta = max(0.0, candidate_capture - baseline_capture)
    replay_saving_seconds = (
        (baseline_ms - charged_candidate_ms) * fixed_physical_steps / 1000.0
    )
    fixed_saving_seconds = replay_saving_seconds - capture_delta
    correctness_pass = (
        bool(correctness.get("selected_token_exact"))
        and bool(correctness.get("counter_threshold_consumed_unchanged"))
        and all(bool(row.get("selected_token_exact")) for row in rows)
    )
    overflow_pass = int(correctness.get("overflow_count", -1)) == 0
    speed_pass = charged_candidate_ms <= max_candidate_ms_per_step
    saving_pass = fixed_saving_seconds >= minimum_saving_seconds
    promotion_pass = correctness_pass and overflow_pass and speed_pass and saving_pass
    return {
        "schema_version": SCHEMA_VERSION,
        "result_class": "top256-sampler-component-scout",
        "production_wiring_changed": False,
        "component": component,
        "physical_work": physical_work,
        "gate": {
            "max_candidate_ms_per_step": max_candidate_ms_per_step,
            "minimum_fixed_main_saving_seconds": minimum_saving_seconds,
            "baseline_ms_per_step_worst": baseline_ms,
            "candidate_ms_per_step_worst": candidate_ms,
            "observed_overflow_fraction": overflow_fraction,
            "fallback_charge_ms_per_step": fallback_charge_ms,
            "charged_candidate_ms_per_step": charged_candidate_ms,
            "capture_setup_delta_seconds": capture_delta,
            "replay_saving_seconds": replay_saving_seconds,
            "fixed_main_saving_seconds": fixed_saving_seconds,
            "correctness_pass": correctness_pass,
            "overflow_pass": overflow_pass,
            "speed_pass": speed_pass,
            "saving_pass": saving_pass,
            "promotion_pass": promotion_pass,
        },
        "coverage_limitations": [
            "captured correctness samples observe K4 final child steps and eager K1 remainders",
            "unobserved K4 child steps require a full runtime gate before integration",
            "top-256 overflow must retain an exact unbounded fallback in any runtime candidate",
        ],
        "decision": (
            "retain_for_full_runtime_scout"
            if promotion_pass
            else "stop_component_scout"
        ),
    }


def render_text(report: dict[str, Any]) -> str:
    gate = report["gate"]
    correctness = report["component"]["correctness"]
    return "\n".join(
        (
            f"decision={report['decision']}",
            f"promotion_pass={gate['promotion_pass']}",
            f"baseline_ms_per_step_worst={gate['baseline_ms_per_step_worst']:.9f}",
            f"candidate_ms_per_step_worst={gate['candidate_ms_per_step_worst']:.9f}",
            f"charged_candidate_ms_per_step={gate['charged_candidate_ms_per_step']:.9f}",
            f"fixed_main_saving_seconds={gate['fixed_main_saving_seconds']:.9f}",
            f"captured_samples={correctness['samples']}",
            f"overflow_count={correctness['overflow_count']}",
            f"selected_token_exact={correctness['selected_token_exact']}",
            "production_wiring_changed=False",
            "",
        )
    )


__all__ = [
    "FIXED_MAIN_STEPS",
    "MAX_CANDIDATE_MS_PER_STEP",
    "MIN_FIXED_MAIN_SAVING_SECONDS",
    "benchmark_candidate",
    "bounded_reference",
    "fixed_physical_work",
    "render_text",
    "summarize_candidate",
]
