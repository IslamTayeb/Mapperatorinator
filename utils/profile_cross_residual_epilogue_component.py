"""Size cross out_proj layout+residual epilogue fusion vs selected production bridge.

Baseline (production): transpose+contiguous+view + fused Wo residual.
Candidate (layout-fused): reshape view + fused Wo residual.
Diagnostic (unfused residual): contiguous flatten + Wo + separate add.

Does not compose with compiled-cross BMM. Full-song only if projected
main saving >= SAVING_TARGET_SECONDS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    CapturedGraph,
    _accepted_main_session_run,
    _assert_scout_args,
    _capture_cuda_graph,
    _capture_representative_layer,
    _load_args,
    _max_abs,
    _reciprocal_graph_rounds,
)


DECODER_LAYERS = 12
MAX_ABS_DRIFT = 1e-3
SAVING_TARGET_SECONDS = 0.08
FIXED_MAIN_GENERATED_TOKENS = 8_294
FIXED_MAIN_DECODE_REPLAYS = 8_207
FIXED_MAIN_WINDOWS = 87
BASE_TIP_CHOICE = {
    "selected": "codex/500tps-arena-compiled-cross-last-mile@0dbab9e5",
    "reason": (
        "cross Wo residual already fused in selected stack; this scout measures "
        "remaining BMM→Wo contiguous-copy elimination only, without compiled-cross"
    ),
}


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_live_graph_cache(
    graph_cache: dict[Any, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError("live graph cache must be a non-empty mapping")
    entries: dict[int, dict[str, Any]] = {}
    for entry in graph_cache.values():
        if not isinstance(entry, dict):
            raise TypeError("live graph cache entries must be mappings")
        prefix = entry.get("active_prefix_length")
        count = entry.get("decode_replays")
        if (
            isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or prefix <= 0
            or prefix in entries
        ):
            raise ValueError(f"invalid or duplicate live prefix {prefix!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"live prefix {prefix} has invalid replay count {count!r}")
        missing = [
            name for name in ("graph", "outputs", "static_inputs") if name not in entry
        ]
        if missing:
            raise ValueError(f"live prefix {prefix} is missing {missing}")
        entries[prefix] = entry
    return dict(sorted(entries.items()))


def validate_fixed_main_work(
    run: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    processor = run.get("processor")
    stats = getattr(processor, "last_generation_stats", None)
    if not isinstance(stats, dict):
        raise RuntimeError("main Processor did not expose aggregate generation stats")
    generated_tokens = stats.get("generated_tokens")
    if isinstance(generated_tokens, bool) or not isinstance(generated_tokens, int):
        raise RuntimeError("main generated-token count is missing or invalid")
    decode_replays = sum(int(entry["decode_replays"]) for entry in entries.values())
    windows = generated_tokens - decode_replays
    expected = (
        FIXED_MAIN_GENERATED_TOKENS,
        FIXED_MAIN_DECODE_REPLAYS,
        FIXED_MAIN_WINDOWS,
    )
    actual = (generated_tokens, decode_replays, windows)
    if actual != expected:
        raise RuntimeError(
            "fixed SALVALAI work changed: "
            f"expected tokens/replays/windows={expected}, got {actual}"
        )
    return {
        "generated_tokens": generated_tokens,
        "decode_replays": decode_replays,
        "prefill_windows": windows,
        "bucket_counts": {
            str(prefix): int(entry["decode_replays"])
            for prefix, entry in entries.items()
        },
    }


def summarize_component(
    buckets: dict[str, dict[str, Any]],
    *,
    total_replays: int,
) -> dict[str, Any]:
    if not buckets or total_replays <= 0:
        raise ValueError("component summary requires buckets and positive total replays")
    measured = 0
    baseline_weighted_ms = 0.0
    candidate_weighted_ms = 0.0
    unfused_weighted_ms = 0.0
    failures: dict[str, list[str]] = {}
    for prefix, entry in sorted(buckets.items(), key=lambda item: int(item[0])):
        count = int(entry["decode_replays"])
        baseline_ms = float(entry["baseline_ms_per_call"])
        candidate_ms = float(entry["candidate_ms_per_call"])
        unfused_ms = float(entry["unfused_residual_ms_per_call"])
        if count <= 0 or min(baseline_ms, candidate_ms, unfused_ms) <= 0:
            raise ValueError(f"prefix {prefix} has invalid count or timing")
        if not all(
            math.isfinite(value) for value in (baseline_ms, candidate_ms, unfused_ms)
        ):
            raise ValueError(f"prefix {prefix} has non-finite timing")
        measured += count
        baseline_weighted_ms += count * baseline_ms
        candidate_weighted_ms += count * candidate_ms
        unfused_weighted_ms += count * unfused_ms
        bucket_failures: list[str] = []
        if not bool(entry.get("baseline_memory_stable")):
            bucket_failures.append("baseline_memory_stable")
        if not bool(entry.get("candidate_memory_stable")):
            bucket_failures.append("candidate_memory_stable")
        if not bool(entry.get("unfused_memory_stable")):
            bucket_failures.append("unfused_memory_stable")
        for name, value in entry["drift"].items():
            if not math.isfinite(float(value)) or float(value) < 0:
                bucket_failures.append(f"{name}_invalid")
            elif float(value) > MAX_ABS_DRIFT:
                bucket_failures.append(f"{name}_above_{MAX_ABS_DRIFT}")
        if bucket_failures:
            failures[prefix] = bucket_failures
    if measured > total_replays:
        raise ValueError("measured replay count exceeds live total")
    baseline_seconds = DECODER_LAYERS * baseline_weighted_ms / 1000.0
    candidate_seconds = DECODER_LAYERS * candidate_weighted_ms / 1000.0
    unfused_seconds = DECODER_LAYERS * unfused_weighted_ms / 1000.0
    saving = baseline_seconds - candidate_seconds
    landed_fusion_saving = unfused_seconds - baseline_seconds
    coverage_pass = measured == total_replays
    correctness_pass = not failures
    sizing_pass = saving >= SAVING_TARGET_SECONDS
    return {
        "measured_replays": measured,
        "total_replays": total_replays,
        "coverage_fraction": measured / total_replays,
        "coverage_pass": coverage_pass,
        "weighted_baseline_seconds_12_layers": baseline_seconds,
        "weighted_candidate_seconds_12_layers": candidate_seconds,
        "weighted_unfused_residual_seconds_12_layers": unfused_seconds,
        "projected_main_saving_seconds": saving,
        "already_landed_residual_fusion_saving_seconds": landed_fusion_saving,
        "saving_target_seconds": SAVING_TARGET_SECONDS,
        "weighted_local_speedup": (
            baseline_weighted_ms / candidate_weighted_ms
        ),
        "correctness_failures": failures,
        "correctness_pass": correctness_pass,
        "sizing_pass": sizing_pass,
        "promotion_pass": coverage_pass and correctness_pass and sizing_pass,
    }


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _cross_out_inputs(capture):
    from osuT5.osuT5.inference.optimized.kernels.dispatch import (
        _q1_bmm_cross_attention,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import PackedLinear
    from osuT5.osuT5.inference.optimized.scout.native_prefix import _norm

    module = capture.module
    cross_attn = module.cross_attn
    residual = capture.hidden_states
    # Build a post-self residual proxy: use captured hidden as the residual that
    # cross out_proj would add into (selected stack uses the post-self state).
    normalized = _norm(module.cross_attn_layer_norm, residual)
    query = torch.nn.functional.linear(
        normalized, cross_attn.Wq.weight, cross_attn.Wq.bias
    )
    query = query.view(1, 1, cross_attn.num_heads, cross_attn.head_dim).transpose(1, 2)
    cache_layer = capture.past_key_value.cross_attention_cache.layers[
        int(cross_attn.layer_idx)
    ]
    keys = getattr(cache_layer, "keys")
    values = getattr(cache_layer, "values")
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise RuntimeError("cross cache keys/values missing for representative layer")
    cross_output = _q1_bmm_cross_attention(
        query.contiguous(),
        keys.contiguous(),
        values.contiguous(),
        expected_dtype=torch.float32,
    )
    if not cross_output.is_contiguous():
        cross_output = cross_output.contiguous()
    packed = PackedLinear.from_module(cross_attn.Wo)
    return cross_output.detach(), residual.detach(), packed


@torch.no_grad()
def profile_component(
    args,
    *,
    output_path: Path,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cross residual epilogue profiler requires CUDA")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("cross residual epilogue profiler requires SM75")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    entries = validate_live_graph_cache(run["session"].graph_cache)
    work = validate_fixed_main_work(run, entries)

    from osuT5.osuT5.inference.optimized.kernels.cross_out_epilogue import (
        cross_out_proj_residual,
    )
    from osuT5.osuT5.inference.optimized.kernels.weight_only import (
        preload_weight_only_extension,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    preload_weight_only_extension()
    torch.cuda.synchronize()
    preload_seconds = time.perf_counter() - started

    buckets: dict[str, Any] = {}
    for prefix, accepted in entries.items():
        capture = _capture_representative_layer(
            model, accepted["static_inputs"], prefix=prefix
        )
        cross_output, residual, packed = _cross_out_inputs(capture)

        def baseline_call() -> torch.Tensor:
            return cross_out_proj_residual(
                cross_output,
                residual,
                packed,
                layout_fused=False,
                fuse_residual=True,
            )

        def candidate_call() -> torch.Tensor:
            return cross_out_proj_residual(
                cross_output,
                residual,
                packed,
                layout_fused=True,
                fuse_residual=True,
            )

        def unfused_call() -> torch.Tensor:
            return cross_out_proj_residual(
                cross_output,
                residual,
                packed,
                layout_fused=False,
                fuse_residual=False,
            )

        calls = {
            "baseline": baseline_call,
            "layout_fused": candidate_call,
            "unfused_residual": unfused_call,
        }
        graphs: dict[str, CapturedGraph] = {}
        snapshots: dict[str, torch.Tensor] = {}
        for name, call in calls.items():
            graphs[name] = _capture_cuda_graph(
                call, context=lambda: _null_context(), warmup=0
            )
            snapshots[name] = graphs[name].outputs.detach().float().cpu().clone()
        timings, rounds, memory = _reciprocal_graph_rounds(
            {name: graph.graph for name, graph in graphs.items()},
            restore=lambda: None,
            warmup=warmup,
            iters=iters,
        )
        buckets[str(prefix)] = {
            "decode_replays": int(accepted["decode_replays"]),
            "baseline_ms_per_call": timings["baseline"],
            "candidate_ms_per_call": timings["layout_fused"],
            "unfused_residual_ms_per_call": timings["unfused_residual"],
            "baseline_memory_stable": bool(memory["baseline"]),
            "candidate_memory_stable": bool(memory["layout_fused"]),
            "unfused_memory_stable": bool(memory["unfused_residual"]),
            "drift": {
                "layout_fused_vs_baseline_max_abs": _max_abs(
                    snapshots["baseline"], snapshots["layout_fused"]
                ),
                "unfused_vs_baseline_max_abs": _max_abs(
                    snapshots["baseline"], snapshots["unfused_residual"]
                ),
            },
            "rounds": rounds,
        }

    return {
        "schema_version": 1,
        "metadata": {
            "candidate": "cross_out_layout_fused_residual_epilogue",
            "hypothesis": (
                "eliminate transpose+contiguous before fused Wo+residual on the "
                "cross path; residual fusion itself is already production"
            ),
            "exactness_claim": False,
            "base_tip_choice": BASE_TIP_CHOICE,
            "commit": _git_head(),
            "precision": "fp32_acts_fp16_wo_weights",
            "decoder_layers": DECODER_LAYERS,
            "warmup": warmup,
            "iters": iters,
            "extension_preload_seconds": preload_seconds,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "compiled_cross_composed": False,
        },
        "fixed_main_work": work,
        "summary": summarize_component(
            buckets, total_replays=int(work["decode_replays"])
        ),
        "buckets": buckets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_component(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if not report["summary"]["promotion_pass"]:
        raise SystemExit("STOP_CROSS_RESIDUAL_EPILOGUE_COMPONENT")


if __name__ == "__main__":
    main()
