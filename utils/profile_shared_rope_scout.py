"""Real-prefix exact shared-RoPE CUDA-graph component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.inference.optimized.scout.shared_rope import (  # noqa: E402
    SharedRopeStats,
    build_shared_rope_plan,
    shared_decoder_rope_context,
)
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    ALL_BUCKETS,
    CapturedGraph,
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _load_args,
    _model_logits,
    _observe_full_graph,
    _reciprocal_graph_rounds,
    _restore_all_cache,
    _verify_full_graph,
    _capture_cuda_graph,
    validate_accepted_graph_cache,
)


SCHEMA_VERSION = "mapperatorinator.shared-rope-scout.v1"
REQUIRED_MAIN_SAVING_SECONDS = 1.503
SHORT_LOOP_STEPS = 4


def _git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    try:
        payload = value.numpy().tobytes()
    except TypeError:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _specialized_context(prefix: int, dispatch_counts: dict[str, int]):
    from osuT5.osuT5.runtime_profiling import generation_profile_context

    return generation_profile_context(
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        native_cross_mlp_tail=True,
        optimized_expected_dtype=torch.float32,
        optimized_dispatch_counts=dispatch_counts,
    )


def _exact_observation(reference, candidate) -> dict[str, Any]:
    logits_exact = bool(torch.equal(reference.logits, candidate.logits))
    key_exact = [
        bool(torch.equal(left, right))
        for left, right in zip(reference.key_slots, candidate.key_slots, strict=True)
    ]
    value_exact = [
        bool(torch.equal(left, right))
        for left, right in zip(reference.value_slots, candidate.value_slots, strict=True)
    ]
    token_probe = torch.topk(reference.logits, k=min(8, reference.logits.shape[-1])).indices
    candidate_token_probe = torch.topk(
        candidate.logits,
        k=min(8, candidate.logits.shape[-1]),
    ).indices
    return {
        "logits_exact": logits_exact,
        "cache_key_slots_exact": all(key_exact),
        "cache_value_slots_exact": all(value_exact),
        "per_layer_key_exact": key_exact,
        "per_layer_value_exact": value_exact,
        "reference_logits_sha256": _sha256_tensor(reference.logits),
        "candidate_logits_sha256": _sha256_tensor(candidate.logits),
        "reference_top_token_ids": token_probe.flatten().tolist(),
        "candidate_top_token_ids": candidate_token_probe.flatten().tolist(),
        "top_token_ids_exact": bool(torch.equal(token_probe, candidate_token_probe)),
    }


def _snapshot_static_tensors(static_inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    snapshots = {
        name: value.detach().clone()
        for name, value in static_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    required = {
        "decoder_input_ids": ((1, 1), torch.long),
        "cache_position": ((1,), torch.long),
        "decoder_position_ids": ((1, 1), torch.long),
    }
    failures = {
        name: (
            None if not isinstance(static_inputs.get(name), torch.Tensor) else (
                tuple(static_inputs[name].shape),
                static_inputs[name].dtype,
            )
        )
        for name, (shape, dtype) in required.items()
        if not isinstance(static_inputs.get(name), torch.Tensor)
        or tuple(static_inputs[name].shape) != shape
        or static_inputs[name].dtype != dtype
    }
    if failures:
        raise RuntimeError(f"shared RoPE short loop has invalid static inputs: {failures}")
    return snapshots


def _advance_static_inputs(static_inputs: dict[str, Any], token: torch.Tensor) -> None:
    decoder_ids = static_inputs["decoder_input_ids"]
    cache_position = static_inputs["cache_position"]
    position_ids = static_inputs["decoder_position_ids"]
    decoder_ids.copy_(token.view(1, 1))
    cache_position.add_(1)
    position_ids.add_(1)
    mask = static_inputs.get("decoder_attention_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 4:
        if int(cache_position.item()) >= mask.shape[-1]:
            raise RuntimeError("shared RoPE short loop exceeded the decoder mask")
        mask.scatter_(
            -1,
            cache_position.view(1, 1, 1, 1),
            torch.zeros((1, 1, 1, 1), dtype=mask.dtype, device=mask.device),
        )


def _short_loop_argmax_tokens(
    captured: CapturedGraph,
    *,
    static_inputs: dict[str, Any],
    cache: Any,
    cache_snapshots: list[Any],
    steps: int = SHORT_LOOP_STEPS,
) -> list[int]:
    if steps <= 0:
        raise ValueError("short-loop steps must be positive")
    tensor_snapshots = _snapshot_static_tensors(static_inputs)
    tokens: list[int] = []
    try:
        _restore_all_cache(cache, cache_snapshots)
        for _ in range(steps):
            captured.graph.replay()
            torch.cuda.synchronize()
            logits = _model_logits(captured.outputs)
            token = torch.argmax(logits, dim=-1).to(dtype=torch.long)
            if token.shape != (1,):
                raise RuntimeError("short-loop argmax token must have shape [1]")
            tokens.append(int(token.item()))
            _advance_static_inputs(static_inputs, token)
    finally:
        _restore_all_cache(cache, cache_snapshots)
        for name, snapshot in tensor_snapshots.items():
            static_inputs[name].copy_(snapshot)
        torch.cuda.synchronize()
    return tokens


def summarize_shared_rope(
    buckets: dict[str, Any],
    *,
    live_counts: dict[int, int],
    install_setup_seconds: float,
) -> dict[str, Any]:
    measured = tuple(sorted(int(prefix) for prefix in buckets))
    if measured != tuple(ALL_BUCKETS):
        raise ValueError(
            f"shared RoPE requires every live bucket {ALL_BUCKETS}, got {measured}"
        )
    if any(prefix not in live_counts or live_counts[prefix] <= 0 for prefix in measured):
        raise ValueError("live counts must contain positive sentinel replay counts")
    if not math.isfinite(install_setup_seconds) or install_setup_seconds < 0:
        raise ValueError("install setup seconds must be finite and non-negative")
    accepted_seconds = sum(
        live_counts[prefix]
        * float(buckets[str(prefix)]["timing"]["accepted_ms_per_call"])
        / 1_000.0
        for prefix in measured
    )
    candidate_seconds = sum(
        live_counts[prefix]
        * float(buckets[str(prefix)]["timing"]["shared_rope_ms_per_call"])
        / 1_000.0
        for prefix in measured
    )
    replay_saving = accepted_seconds - candidate_seconds
    projected_saving = replay_saving - install_setup_seconds
    exact = all(bool(buckets[str(prefix)]["exact_pass"]) for prefix in measured)
    call_accounting = all(
        bool(buckets[str(prefix)]["rope_call_accounting_pass"])
        for prefix in measured
    )
    projected_pass = projected_saving >= REQUIRED_MAIN_SAVING_SECONDS
    return {
        "accepted_measured_seconds": accepted_seconds,
        "shared_rope_measured_seconds": candidate_seconds,
        "replay_saving_seconds": replay_saving,
        "install_setup_seconds_charged": install_setup_seconds,
        "projected_main_saving_seconds": projected_saving,
        "required_main_saving_seconds": REQUIRED_MAIN_SAVING_SECONDS,
        "unmeasured_buckets_assumed_saving_seconds": 0.0,
        "exact_pass": exact,
        "rope_call_accounting_pass": call_accounting,
        "performance_pass": projected_pass,
        "promotion_pass": exact and call_accounting and projected_pass,
        "follow_up_if_below_threshold": (
            "precomputed FP32 RoPE table row loaded and applied inside the accepted "
            "split-q1 kernel; remove angle-generation nodes without replacing the decoder"
        ),
    }


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected shared-RoPE schema version")
    metadata = report.get("metadata")
    buckets = report.get("buckets")
    summary = report.get("summary")
    if not all(isinstance(value, dict) for value in (metadata, buckets, summary)):
        raise TypeError("shared-RoPE metadata, buckets, and summary must be objects")
    if tuple(metadata.get("measured_buckets", ())) != ALL_BUCKETS:
        raise ValueError("shared-RoPE report must cover every live bucket")
    if set(buckets) != {str(prefix) for prefix in ALL_BUCKETS}:
        raise ValueError("shared-RoPE bucket payload does not match metadata")
    for prefix in ALL_BUCKETS:
        bucket = buckets[str(prefix)]
        timing = bucket.get("timing")
        if not isinstance(timing, dict):
            raise TypeError(f"prefix {prefix} timing must be an object")
        values = [
            float(timing["accepted_ms_per_call"]),
            float(timing["shared_rope_ms_per_call"]),
        ]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"prefix {prefix} contains invalid graph timing")
        if not isinstance(bucket.get("short_loop_argmax_tokens"), dict):
            raise TypeError(f"prefix {prefix} short-loop tokens must be an object")
    expected = (
        bool(summary["exact_pass"])
        and bool(summary["rope_call_accounting_pass"])
        and float(summary["projected_main_saving_seconds"])
        >= float(summary["required_main_saving_seconds"])
    )
    if bool(summary["promotion_pass"]) is not expected:
        raise ValueError("shared-RoPE promotion decision is inconsistent")


@torch.no_grad()
def profile_shared_rope_scout(
    args,
    *,
    output_path: Path,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("shared-RoPE component gate requires CUDA")
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be positive")
    _assert_scout_args(args)
    output_path.mkdir(parents=True, exist_ok=True)
    run = _accepted_main_session_run(args, output_path=output_path)
    model = run["model"]
    model.eval()
    plan = build_shared_rope_plan(model)
    accepted_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    selected = {prefix: accepted_entries[prefix] for prefix in ALL_BUCKETS}
    live_counts = {
        prefix: int(entry["decode_replays"])
        for prefix, entry in accepted_entries.items()
    }
    dispatch_by_prefix = {
        prefix: {
            "native_q1_rope_cache_self_attention": 0,
            "native_q1_self_attention": 0,
            "q1_bmm_cross_attention": 0,
            "native_cross_mlp_tail": 0,
        }
        for prefix in ALL_BUCKETS
    }
    captured_candidates: dict[int, CapturedGraph] = {}
    capture_stats_by_prefix: dict[int, dict[str, Any]] = {}
    stats = SharedRopeStats()
    install_started = time.perf_counter()
    with shared_decoder_rope_context(model, stats=stats):
        install_setup_seconds = time.perf_counter() - install_started
        for prefix, entry in selected.items():
            before = stats.as_dict()
            inputs = entry["static_inputs"]
            cache = _cache_from_static_inputs(inputs)
            cache_position = inputs.get("cache_position")
            if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
                raise RuntimeError(f"prefix {prefix} has invalid cache_position")
            snapshots = _all_cache_snapshots(cache, cache_position)
            _restore_all_cache(cache, snapshots)
            captured_candidates[prefix] = _capture_cuda_graph(
                lambda inputs=inputs: model(**inputs, return_dict=True),
                context=lambda prefix=prefix: _specialized_context(
                    prefix,
                    dispatch_by_prefix[prefix],
                ),
                warmup=0,
            )
            after = stats.as_dict()
            capture_stats_by_prefix[prefix] = {
                "forwards": after["forwards"] - before["forwards"],
                "computes": after["computes"] - before["computes"],
                "reuses": after["reuses"] - before["reuses"],
                "module_count": after["module_count"],
                "group_count": after["group_count"],
                "eliminated_per_forward": after["eliminated_per_forward"],
            }

    buckets: dict[str, Any] = {}
    for prefix, entry in selected.items():
        inputs = entry["static_inputs"]
        cache = _cache_from_static_inputs(inputs)
        cache_position = inputs["cache_position"]
        snapshots = _all_cache_snapshots(cache, cache_position)
        accepted = CapturedGraph(
            graph=entry["graph"],
            outputs=entry["outputs"],
            setup_seconds=float(entry.get("capture_seconds", 0.0)),
            peak_vram_bytes=0,
        )
        candidate = captured_candidates[prefix]
        checks: dict[str, Any] = {}
        details: dict[str, Any] = {}
        for name, graph in (("accepted", accepted), ("shared_rope", candidate)):
            checks[name], details[name] = _verify_full_graph(
                graph,
                cache=cache,
                snapshots=snapshots,
            )
        medians, rounds, memory = _reciprocal_graph_rounds(
            {"accepted": accepted.graph, "shared_rope": candidate.graph},
            restore=lambda: _restore_all_cache(cache, snapshots),
            warmup=warmup,
            iters=iters,
        )
        reference = _observe_full_graph(accepted, cache=cache, snapshots=snapshots)
        observation = _observe_full_graph(candidate, cache=cache, snapshots=snapshots)
        exact = _exact_observation(reference, observation)
        accepted_tokens = _short_loop_argmax_tokens(
            accepted,
            static_inputs=inputs,
            cache=cache,
            cache_snapshots=snapshots,
        )
        candidate_tokens = _short_loop_argmax_tokens(
            candidate,
            static_inputs=inputs,
            cache=cache,
            cache_snapshots=snapshots,
        )
        call_stats = capture_stats_by_prefix[prefix]
        call_accounting = (
            call_stats["forwards"] == 1
            and call_stats["computes"] == call_stats["group_count"]
            and call_stats["reuses"]
            == call_stats["module_count"] - call_stats["group_count"]
        )
        dispatch = dispatch_by_prefix[prefix]
        dispatch_pass = (
            dispatch["native_q1_rope_cache_self_attention"] == plan.module_count
            and dispatch["q1_bmm_cross_attention"] == plan.module_count
            and dispatch["native_cross_mlp_tail"] == plan.module_count
        )
        exact_pass = (
            all(checks["accepted"].values())
            and all(checks["shared_rope"].values())
            and bool(memory["accepted"])
            and bool(memory["shared_rope"])
            and exact["logits_exact"]
            and exact["cache_key_slots_exact"]
            and exact["cache_value_slots_exact"]
            and exact["top_token_ids_exact"]
            and accepted_tokens == candidate_tokens
            and dispatch_pass
        )
        buckets[str(prefix)] = {
            "decode_replays": live_counts[prefix],
            "cache_position": int(cache_position.item()),
            "timing": {
                "accepted_ms_per_call": medians["accepted"],
                "shared_rope_ms_per_call": medians["shared_rope"],
                "saving_ms_per_call": medians["accepted"] - medians["shared_rope"],
                "reciprocal_rounds": rounds,
                "memory_stable": memory,
            },
            "exact": exact,
            "short_loop_argmax_tokens": {
                "accepted": accepted_tokens,
                "shared_rope": candidate_tokens,
                "exact": accepted_tokens == candidate_tokens,
                "steps": SHORT_LOOP_STEPS,
            },
            "graph_checks": checks,
            "graph_details": details,
            "dispatch_counts": dispatch,
            "dispatch_pass": dispatch_pass,
            "rope_capture_stats": call_stats,
            "rope_call_accounting_pass": call_accounting,
            "exact_pass": exact_pass,
            "candidate_capture_seconds": candidate.setup_seconds,
            "candidate_capture_peak_vram_bytes": candidate.peak_vram_bytes,
        }
    summary = summarize_shared_rope(
        buckets,
        live_counts=live_counts,
        install_setup_seconds=install_setup_seconds,
    )
    total_replays = sum(live_counts.values())
    measured_replays = sum(live_counts[prefix] for prefix in ALL_BUCKETS)
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "exact-component-scout",
            "production_wiring": False,
            "original_decoder_layer": True,
            "original_specialized_dispatch": True,
            "precision": "fp32",
            "measured_buckets": list(ALL_BUCKETS),
            "accepted_bucket_replay_counts": {
                str(prefix): count for prefix, count in live_counts.items()
            },
            "measured_replays": measured_replays,
            "total_replays": total_replays,
            "coverage_fraction": measured_replays / total_replays,
            "unmeasured_buckets_assumed_saving_seconds": 0.0,
            "rope_plan": {
                "module_count": plan.module_count,
                "group_count": plan.group_count,
                "members": [member.name for member in plan.members],
                "groups": stats.as_dict()["group_members"],
            },
            "rope_capture_totals": stats.as_dict(),
            "warmup": warmup,
            "iterations": iters,
            "commit": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "result_path": run["result_path"],
        },
        "buckets": buckets,
        "summary": summary,
    }
    _validate_report(report)
    return report


def _text_report(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    summary = report["summary"]
    lines = [
        f"schema={report['schema_version']}",
        f"commit={metadata['commit']}",
        f"gpu={metadata['cuda_device']}",
        f"rope_modules={metadata['rope_plan']['module_count']}",
        f"rope_groups={metadata['rope_plan']['group_count']}",
        f"projected_main_saving_seconds={summary['projected_main_saving_seconds']:.9f}",
        f"required_main_saving_seconds={summary['required_main_saving_seconds']:.9f}",
        f"exact_pass={str(summary['exact_pass']).lower()}",
        f"call_accounting_pass={str(summary['rope_call_accounting_pass']).lower()}",
        f"promotion_pass={str(summary['promotion_pass']).lower()}",
        f"follow_up={summary['follow_up_if_below_threshold']}",
    ]
    for prefix in ALL_BUCKETS:
        bucket = report["buckets"][str(prefix)]
        lines.append(
            f"prefix_{prefix}=accepted_ms:{bucket['timing']['accepted_ms_per_call']:.9f},"
            f"shared_ms:{bucket['timing']['shared_rope_ms_per_call']:.9f},"
            f"exact:{str(bucket['exact_pass']).lower()},"
            f"eliminated:{bucket['rope_capture_stats']['eliminated_per_forward']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1_000)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_shared_rope_scout(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iters=cli.iters,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cli.text_path.write_text(_text_report(report), encoding="utf-8")
    print(_text_report(report), end="")
    if not report["summary"]["promotion_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
