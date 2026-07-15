#!/usr/bin/env python3
"""Measure torch.compile q1 cross BMM regions inside an outer CUDA graph."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    SENTINEL_BUCKETS,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _load_args,
    _restore_all_cache,
    _time_graph,
    validate_accepted_graph_cache,
)


SCHEMA_VERSION = 1
MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")
WARMUP = 50
ITERATIONS = 1_000
DECODER_LAYERS = 12


def _max_abs(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        return math.inf
    return float((reference.float() - candidate.float()).abs().max().item())


def _capture_outer_graph(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("compile graph scout requires CUDA")
    started = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    output = None
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            output = fn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    if not isinstance(output, torch.Tensor):
        raise TypeError("compiled cross BMM region must return a tensor")
    return graph, output, time.perf_counter() - started


def _capture_real_cross_inputs(
    model: torch.nn.Module,
    static_inputs: dict[str, Any],
    *,
    prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from osuT5.osuT5.inference.optimized.kernels import cross_mlp
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    original = cross_mlp._q1_bmm_cross_attention
    found: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def recorder(query, key, value, *, expected_dtype):
        if not found:
            if expected_dtype != torch.float32:
                raise TypeError("compile graph scout requires accepted FP32 tensors")
            found["inputs"] = (
                query.detach().clone(memory_format=torch.contiguous_format),
                key.detach().clone(memory_format=torch.contiguous_format),
                value.detach().clone(memory_format=torch.contiguous_format),
            )
        return original(
            query,
            key,
            value,
            expected_dtype=expected_dtype,
        )

    cache = _cache_from_static_inputs(static_inputs)
    cache_position = static_inputs.get("cache_position")
    if not isinstance(cache_position, torch.Tensor):
        raise TypeError("accepted graph inputs expose no cache_position tensor")
    snapshots = _all_cache_snapshots(cache, cache_position)
    dispatch_counts = {
        "native_q1_rope_cache_self_attention": 0,
        "native_q1_self_attention": 0,
        "q1_bmm_cross_attention": 0,
        "native_cross_mlp_tail": 0,
    }
    cross_mlp._q1_bmm_cross_attention = recorder
    try:
        _restore_all_cache(cache, snapshots)
        with generation_profile_context(
            active_prefix_self_attention_length=prefix,
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=True,
            native_cross_mlp_tail=True,
            optimized_expected_dtype=torch.float32,
            optimized_dispatch_counts=dispatch_counts,
        ):
            model(**static_inputs, return_dict=True)
        torch.cuda.synchronize()
    finally:
        cross_mlp._q1_bmm_cross_attention = original
        _restore_all_cache(cache, snapshots)
    if "inputs" not in found:
        raise RuntimeError(f"accepted prefix {prefix} never dispatched cross BMM")
    if dispatch_counts["native_cross_mlp_tail"] <= 0:
        raise RuntimeError("accepted native cross+MLP dispatch did not execute")
    return found["inputs"]


def _compile_region(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    *,
    mode: str,
) -> tuple[Callable[..., torch.Tensor], float, torch.Tensor]:
    kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": False}
    if mode != "default":
        kwargs["mode"] = mode
    compiled = torch.compile(fn, **kwargs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = compiled(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not isinstance(output, torch.Tensor):
        raise TypeError("torch.compile q1 BMM returned a non-tensor")
    return compiled, elapsed, output


def summarize(
    buckets: dict[str, Any],
    *,
    live_counts: dict[int, int],
) -> dict[str, Any]:
    measured = tuple(sorted(int(prefix) for prefix in buckets))
    if measured != tuple(SENTINEL_BUCKETS):
        raise ValueError(
            f"compile scout requires sentinel buckets {SENTINEL_BUCKETS}, got {measured}"
        )
    if any(live_counts.get(prefix, 0) <= 0 for prefix in measured):
        raise ValueError("live replay counts are missing a sentinel bucket")
    modes: dict[str, Any] = {}
    for mode in MODES:
        entries = [buckets[str(prefix)]["modes"].get(mode) for prefix in measured]
        if not all(isinstance(entry, dict) and entry.get("pass") for entry in entries):
            modes[mode] = {
                "pass": False,
                "failure_buckets": [
                    prefix
                    for prefix, entry in zip(measured, entries, strict=True)
                    if not isinstance(entry, dict) or not entry.get("pass")
                ],
            }
            continue
        weighted_eager = DECODER_LAYERS * sum(
            live_counts[prefix]
            * float(buckets[str(prefix)]["eager_outer_graph_ms_per_call"])
            / 1_000.0
            for prefix in measured
        )
        weighted_compiled = DECODER_LAYERS * sum(
            live_counts[prefix]
            * float(buckets[str(prefix)]["modes"][mode]["outer_graph_ms_per_call"])
            / 1_000.0
            for prefix in measured
        )
        setup = sum(
            float(buckets[str(prefix)]["modes"][mode]["compile_seconds"])
            + float(buckets[str(prefix)]["modes"][mode]["outer_capture_seconds"])
            for prefix in measured
        )
        modes[mode] = {
            "pass": True,
            "all_exact": all(
                bool(buckets[str(prefix)]["modes"][mode]["exact"])
                for prefix in measured
            ),
            "maximum_max_abs": max(
                float(buckets[str(prefix)]["modes"][mode]["max_abs"])
                for prefix in measured
            ),
            "weighted_eager_seconds_12_layers": weighted_eager,
            "weighted_compiled_seconds_12_layers": weighted_compiled,
            "projected_main_saving_seconds": weighted_eager - weighted_compiled,
            "compile_plus_capture_setup_seconds": setup,
        }
    passing = {
        mode: row
        for mode, row in modes.items()
        if row.get("pass") and row["projected_main_saving_seconds"] > 0
    }
    best = max(
        passing,
        key=lambda mode: passing[mode]["projected_main_saving_seconds"],
        default=None,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "measured_buckets": list(measured),
        "measured_replays": sum(live_counts[prefix] for prefix in measured),
        "total_live_replays": sum(live_counts.values()),
        "modes": modes,
        "best_positive_mode": best,
        "retain_for_full_graph_gate": best is not None,
    }


@torch.no_grad()
def run(args, *, output_path: Path, warmup: int, iterations: int) -> dict[str, Any]:
    _assert_scout_args(args)
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    accepted = _accepted_main_session_run(args, output_path=output_path)
    model = accepted["model"]
    entries = validate_accepted_graph_cache(accepted["session"].graph_cache)
    live_counts = {
        prefix: int(entry["decode_replays"])
        for prefix, entry in entries.items()
    }
    buckets: dict[str, Any] = {}

    from osuT5.osuT5.inference.optimized.kernels.dispatch import (
        _q1_bmm_cross_attention,
    )

    def region(query, key, value):
        return _q1_bmm_cross_attention(
            query,
            key,
            value,
            expected_dtype=torch.float32,
        )

    for prefix in SENTINEL_BUCKETS:
        entry = entries[prefix]
        static_inputs = entry.get("static_inputs")
        if not isinstance(static_inputs, dict):
            raise TypeError(f"accepted prefix {prefix} has no static input mapping")
        real_args = _capture_real_cross_inputs(model, static_inputs, prefix=prefix)
        eager_reference = region(*real_args)
        eager_graph, eager_output, eager_capture = _capture_outer_graph(
            region,
            real_args,
            warmup=warmup,
        )
        eager_timing = _time_graph(eager_graph, warmup=warmup, iters=iterations)
        if not eager_timing.get("memory_stable"):
            raise RuntimeError("eager outer graph changed allocation")
        bucket = {
            "decode_replays": live_counts[prefix],
            "q_shape": list(real_args[0].shape),
            "kv_shape": list(real_args[1].shape),
            "eager_outer_capture_seconds": eager_capture,
            "eager_outer_graph_ms_per_call": float(eager_timing["ms_per_call"]),
            "eager_repeat_exact": bool(torch.equal(eager_reference, eager_output)),
            "modes": {},
        }
        for mode in MODES:
            try:
                torch._dynamo.reset()
                compiled, compile_seconds, compile_output = _compile_region(
                    region,
                    real_args,
                    mode=mode,
                )
                graph, graph_output, capture_seconds = _capture_outer_graph(
                    compiled,
                    real_args,
                    warmup=warmup,
                )
                timing = _time_graph(graph, warmup=warmup, iters=iterations)
                if not timing.get("memory_stable"):
                    raise RuntimeError("compiled outer graph changed allocation")
                bucket["modes"][mode] = {
                    "pass": True,
                    "compile_seconds": compile_seconds,
                    "outer_capture_seconds": capture_seconds,
                    "outer_graph_ms_per_call": float(timing["ms_per_call"]),
                    "exact": bool(
                        torch.equal(eager_reference, compile_output)
                        and torch.equal(eager_reference, graph_output)
                    ),
                    "max_abs": max(
                        _max_abs(eager_reference, compile_output),
                        _max_abs(eager_reference, graph_output),
                    ),
                }
            except Exception as exc:
                bucket["modes"][mode] = {
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        buckets[str(prefix)] = bucket
    report = {
        "schema_version": SCHEMA_VERSION,
        "question": "Can torch.compile accelerate accepted q1 cross BMM inside outer CUDA graphs?",
        "buckets": buckets,
        "live_counts": {str(key): value for key, value in sorted(live_counts.items())},
        "summary": summarize(buckets, live_counts=live_counts),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parsed, overrides = parser.parse_known_args()
    args = _load_args(parsed.config_name, overrides)
    report = run(
        args,
        output_path=parsed.output_path,
        warmup=parsed.warmup,
        iterations=parsed.iterations,
    )
    parsed.json_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
