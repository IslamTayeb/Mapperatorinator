from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import torch

from osuT5.osuT5.runtime_profiling import generation_profile_context
from utils.profile_native_prefix_dtype_scout import (
    CapturedGraph,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _capture_cuda_graph,
    _load_args,
    _max_abs,
    _observe_full_graph,
    _reciprocal_graph_rounds,
    _restore_all_cache,
    _verify_full_graph,
    select_buckets,
    validate_accepted_graph_cache,
)


SAVING_TARGET_SECONDS = 1.412
MAX_LOGIT_DRIFT = 1e-3


def _candidate_context(prefix: int):
    return generation_profile_context(
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        native_cross_mlp_tail=True,
    )


def summarize_component(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets:
        raise ValueError("component report requires at least one bucket")
    weighted_accepted = 0.0
    weighted_candidate = 0.0
    measured_replays = 0
    correctness_failures: dict[str, list[str]] = {}
    for bucket, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(entry["decode_replays"])
        measured_replays += count
        weighted_accepted += count * float(entry["accepted_ms_per_call"])
        weighted_candidate += count * float(entry["candidate_ms_per_call"])
        failures = [
            name
            for name, passed in entry["candidate_checks"].items()
            if not passed
        ]
        drift = entry["drift"]
        if float(drift["cache_key_slot_max_abs"]) != 0.0:
            failures.append("cache_key_slot_not_exact")
        if float(drift["cache_value_slot_max_abs"]) != 0.0:
            failures.append("cache_value_slot_not_exact")
        if float(drift["logits_max_abs"]) > MAX_LOGIT_DRIFT:
            failures.append("logits_drift_above_limit")
        if failures:
            correctness_failures[bucket] = failures
    accepted_seconds = weighted_accepted / 1000.0
    candidate_seconds = weighted_candidate / 1000.0
    saving_seconds = accepted_seconds - candidate_seconds
    return {
        "measured_replays": measured_replays,
        "total_replays": int(total_replays),
        "coverage_fraction": measured_replays / total_replays,
        "weighted_accepted_seconds": accepted_seconds,
        "weighted_candidate_seconds": candidate_seconds,
        "measured_saving_seconds": saving_seconds,
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "sizing_pass": saving_seconds >= SAVING_TARGET_SECONDS,
        "correctness_failures": correctness_failures,
        "correctness_pass": not correctness_failures,
    }


@torch.no_grad()
def profile_native_cross_mlp_component(
    args,
    *,
    output_path: Path,
    bucket_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("native cross+MLP component profiler requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected = select_buckets(accepted_entries, bucket_mode)
    total_replays = sum(
        int(entry["decode_replays"])
        for entry in accepted_entries.values()
    )

    from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
        preload_native_decoder_layer,
    )

    torch.cuda.synchronize()
    preload_started = time.perf_counter()
    preload_native_decoder_layer()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - preload_started

    buckets: dict[str, dict[str, Any]] = {}
    for prefix, accepted in selected.items():
        inputs = accepted["static_inputs"]
        cache = _cache_from_static_inputs(inputs)
        cache_position = inputs.get("cache_position")
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
            raise RuntimeError(f"prefix {prefix} has invalid cache_position")
        snapshots = _all_cache_snapshots(cache, cache_position)
        _restore_all_cache(cache, snapshots)
        candidate = _capture_cuda_graph(
            lambda inputs=inputs: model(**inputs, return_dict=True),
            context=lambda prefix=prefix: _candidate_context(prefix),
            warmup=0,
        )
        accepted_graph = CapturedGraph(
            graph=accepted["graph"],
            outputs=accepted["outputs"],
            setup_seconds=float(accepted.get("capture_seconds", 0.0)),
            peak_vram_bytes=0,
        )
        graphs = {
            "accepted": accepted_graph,
            "candidate": candidate,
        }
        checks: dict[str, dict[str, bool]] = {}
        details: dict[str, Any] = {}
        for name, graph in graphs.items():
            checks[name], details[name] = _verify_full_graph(
                graph,
                cache=cache,
                snapshots=snapshots,
            )
        timings, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: _restore_all_cache(cache, snapshots),
            warmup=warmup,
            iters=iters,
        )
        observations = {
            name: _observe_full_graph(graph, cache=cache, snapshots=snapshots)
            for name, graph in graphs.items()
        }
        reference = observations["accepted"]
        observed = observations["candidate"]
        key_drift = max(
            _max_abs(expected, actual)
            for expected, actual in zip(
                reference.key_slots,
                observed.key_slots,
                strict=True,
            )
        )
        value_drift = max(
            _max_abs(expected, actual)
            for expected, actual in zip(
                reference.value_slots,
                observed.value_slots,
                strict=True,
            )
        )
        drift = {
            "cache_key_slot_max_abs": key_drift,
            "cache_value_slot_max_abs": value_drift,
            "logits_max_abs": _max_abs(reference.logits, observed.logits),
        }
        candidate_checks = dict(checks["candidate"])
        candidate_checks["memory_stable"] &= bool(memory["candidate"])
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "accepted_ms_per_call": timings["accepted"],
            "candidate_ms_per_call": timings["candidate"],
            "candidate_capture_seconds": candidate.setup_seconds,
            "candidate_peak_vram_bytes": candidate.peak_vram_bytes,
            "candidate_checks": candidate_checks,
            "drift": drift,
            "rounds": rounds,
            "details": details,
        }

    summary = summarize_component(buckets, total_replays=total_replays)
    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "native_cross_mlp_tail",
            "precision": "fp32",
            "attn_implementation": "sdpa",
            "bucket_mode": bucket_mode,
            "extension_preload_seconds": preload_seconds,
            "warmup": warmup,
            "iters": iters,
        },
        "summary": summary,
        "buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bucket-mode", choices=("sentinel", "all"), default="sentinel")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    result = profile_native_cross_mlp_component(
        args,
        output_path=cli.output_path,
        bucket_mode=cli.bucket_mode,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    summary = result["summary"]
    if not summary["correctness_pass"]:
        raise SystemExit(1)
    if not summary["sizing_pass"]:
        raise SystemExit(3)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
