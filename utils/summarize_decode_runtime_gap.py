from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.summarize_active_prefix_diagnostics import summarize_active_prefix_diagnostics


AGGREGATE_EVENT_NAMES = {
    "loop_total",
    "decode_forward",
    "decode_forward.cuda_graph",
}

CONTROL_EVENT_NAMES = {
    "prepare_inputs",
    "stopping_criteria",
    "prefill_forward",
}


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _records_for_label(profile: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [record for record in profile.get("generation", []) if record.get("profile_label") == label]


def _record_sum(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record.get(key, 0.0) or 0.0) for record in records)


def _record_tokens(records: list[dict[str, Any]]) -> int:
    return sum(int(record.get("generated_tokens", 0) or 0) for record in records)


def _projection_seconds(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    for projection_key in ("cuda_graph_replay_projection", "projection"):
        projection = result.get(projection_key)
        if isinstance(projection, dict):
            seconds = projection.get("full_song_seconds_at_decode_steps")
            if isinstance(seconds, (int, float)):
                return float(seconds)
    return None


def _result_projection(report: dict[str, Any] | None, *names: str) -> float | None:
    if not isinstance(report, dict):
        return None
    results = report.get("results")
    if not isinstance(results, dict):
        return None
    for name in names:
        seconds = _projection_seconds(results.get(name))
        if seconds is not None:
            return seconds
    return None


def _decoder_layer_projection(report: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(report, dict):
        return None
    projected = report.get("projected_full_song")
    if not isinstance(projected, dict):
        return None
    total = 0.0
    found = False
    for value in projected.values():
        if not isinstance(value, dict):
            continue
        seconds = value.get(key)
        if isinstance(seconds, (int, float)):
            total += float(seconds)
            found = True
    return total if found else None


def _decoder_layer_segment_projection(report: dict[str, Any] | None, segment_name: str) -> float | None:
    if not isinstance(report, dict):
        return None
    projected = report.get("projected_full_song")
    if not isinstance(projected, dict):
        return None
    total = 0.0
    found = False
    for value in projected.values():
        if not isinstance(value, dict):
            continue
        segments = value.get("cuda_graph_segment_seconds")
        if not isinstance(segments, dict):
            continue
        seconds = segments.get(segment_name)
        if isinstance(seconds, (int, float)):
            total += float(seconds)
            found = True
    return total if found else None


def _tps(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if tokens > 0 and seconds > 0 else None


def _if_removed_tps(tokens: int, baseline_model_seconds: float, removed_seconds: float | None) -> float | None:
    if removed_seconds is None:
        return None
    remaining = baseline_model_seconds - removed_seconds
    return _tps(tokens, remaining)


def _active_event_projections(
        active_summary: dict[str, Any] | None,
        *,
        baseline_model_seconds: float,
        full_song_main_tokens: int,
) -> list[dict[str, Any]]:
    if not isinstance(active_summary, dict):
        return []
    generated_tokens = int(active_summary.get("generated_tokens", 0) or 0)
    if generated_tokens <= 0:
        return []
    scale = full_song_main_tokens / generated_tokens
    rows = []
    cuda_events = active_summary.get("cuda_event_ms") or {}
    calls = active_summary.get("cuda_event_calls") or {}
    for name, milliseconds in cuda_events.items():
        if not isinstance(milliseconds, (int, float)):
            continue
        event_name = str(name)
        event_seconds = float(milliseconds) / 1000.0
        projected_seconds = event_seconds * scale
        is_aggregate = event_name in AGGREGATE_EVENT_NAMES
        rows.append({
            "name": event_name,
            "is_aggregate_or_composite": is_aggregate,
            "is_control_or_setup": event_name in CONTROL_EVENT_NAMES,
            "standalone_target_candidate": not is_aggregate,
            "diagnostic_event_seconds": event_seconds,
            "projected_full_song_seconds": projected_seconds,
            "calls": int(calls.get(name, 0) or 0),
            "projected_fraction_of_baseline_model_time": (
                projected_seconds / baseline_model_seconds
                if baseline_model_seconds > 0
                else None
            ),
            "projected_tps_if_removed_from_baseline": _if_removed_tps(
                full_song_main_tokens,
                baseline_model_seconds,
                projected_seconds,
            ),
        })
    return sorted(rows, key=lambda item: item["projected_full_song_seconds"], reverse=True)


def _recommended_next_steps(report: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    ledger = report.get("model_generate_ledger") or {}
    host_gap = ledger.get("host_gap_seconds")
    model_seconds = ledger.get("model_seconds")
    if isinstance(host_gap, (int, float)) and isinstance(model_seconds, (int, float)) and model_seconds > 0:
        if host_gap / model_seconds < 0.01:
            recommendations.append(
                "Do not prioritize Python outside model.generate; host gap is below 1% of diagnostic model time."
            )
    events = report.get("active_event_projections") or []
    targetable_events = [
        event for event in events
        if isinstance(event, dict) and event.get("standalone_target_candidate")
    ]
    top_event = targetable_events[0] if targetable_events else None
    if isinstance(top_event, dict):
        name = top_event.get("name")
        if name in {"graph.replay"}:
            recommendations.append(
                "The largest measured bucket is graph-replayed decoder work; target broad decoder-layer/runtime compute."
            )
    keep_bars = report.get("keep_bars") or {}
    five_pct_model = keep_bars.get("five_pct_model_time_seconds")
    for event in events:
        if event.get("name") in {"graph.input_copy", "logits_processor", "sampling.multinomial"}:
            seconds = event.get("projected_full_song_seconds")
            if isinstance(seconds, (int, float)) and isinstance(five_pct_model, (int, float)) and seconds < five_pct_model:
                recommendations.append(
                    f"Do not pursue standalone {event['name']} cleanup from this evidence; projected ceiling is below the 5% model-time bar."
                )
    control_events_above_bar = [
        event for event in targetable_events
        if (
            event.get("is_control_or_setup")
            and isinstance(event.get("projected_full_song_seconds"), (int, float))
            and isinstance(five_pct_model, (int, float))
            and event["projected_full_song_seconds"] >= five_pct_model
        )
    ]
    if control_events_above_bar:
        recommendations.append(
            "Control/setup ranges above the keep bar need a low-perturb exclusive proof before coding; prior verifier ceilings have overstated production wins."
        )
    if report.get("production_minus_full_forward_replay_seconds") is not None:
        recommendations.append(
            "Explain any remaining production-vs-replay gap with a more exclusive auditor before production runtime code."
        )
    return recommendations


def summarize_runtime_gap(
        profile: dict[str, Any],
        *,
        label: str,
        full_song_main_tokens: int,
        full_song_model_time_seconds: float,
        target_tps: float,
        active_summary: dict[str, Any] | None = None,
        full_forward_report: dict[str, Any] | None = None,
        decoder_stack_report: dict[str, Any] | None = None,
        decoder_layer_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = _records_for_label(profile, label)
    active_summary = active_summary or summarize_active_prefix_diagnostics(profile, label=label)
    tokens = _record_tokens(records)
    model_seconds = _record_sum(records, "model_elapsed_seconds")
    wall_seconds = _record_sum(records, "wall_seconds")
    generate_cuda_seconds = _record_sum(records, "model_generate_cuda_event_seconds")
    host_gap_seconds = _record_sum(records, "model_generate_host_gap_seconds")
    target_model_seconds = full_song_main_tokens / target_tps

    full_forward_s = _result_projection(
        full_forward_report,
        "full_model_forward",
        "full_model_forward_logits",
    )
    stack_full_forward_s = _result_projection(decoder_stack_report, "full_model_forward_logits")
    if full_forward_s is None:
        full_forward_s = stack_full_forward_s
    decoder_stack_s = _result_projection(decoder_stack_report, "decoder_stack_hidden")
    decoder_stack_projection_s = _result_projection(decoder_stack_report, "decoder_stack_plus_projection_logits")
    decoder_layer_s = _decoder_layer_projection(decoder_layer_report, "cuda_graph_repo_decoder_layer_s")
    decoder_layer_segments = {
        "self_attn_residual_segment": _decoder_layer_segment_projection(
            decoder_layer_report,
            "self_attn_residual_segment",
        ),
        "cross_attn_residual_segment": _decoder_layer_segment_projection(
            decoder_layer_report,
            "cross_attn_residual_segment",
        ),
        "mlp_residual_segment": _decoder_layer_segment_projection(
            decoder_layer_report,
            "mlp_residual_segment",
        ),
    }

    result: dict[str, Any] = {
        "label": label,
        "diagnostic_profile": {
            "records": len(records),
            "generated_tokens": tokens,
            "model_elapsed_seconds": model_seconds,
            "wall_seconds": wall_seconds,
            "tokens_per_second": _tps(tokens, model_seconds),
        },
        "accepted_baseline": {
            "full_song_main_tokens": full_song_main_tokens,
            "full_song_model_time_seconds": full_song_model_time_seconds,
            "tokens_per_second": _tps(full_song_main_tokens, full_song_model_time_seconds),
        },
        "target": {
            "tokens_per_second": target_tps,
            "required_model_time_seconds": target_model_seconds,
            "required_saving_seconds": full_song_model_time_seconds - target_model_seconds,
        },
        "keep_bars": {
            "five_pct_model_time_seconds": full_song_model_time_seconds * 0.05,
            "ten_pct_model_time_seconds": full_song_model_time_seconds * 0.10,
            "five_pct_throughput_required_saving_seconds": (
                full_song_model_time_seconds - (full_song_main_tokens / (_tps(full_song_main_tokens, full_song_model_time_seconds) * 1.05))
                if _tps(full_song_main_tokens, full_song_model_time_seconds)
                else None
            ),
            "ten_pct_throughput_required_saving_seconds": (
                full_song_model_time_seconds - (full_song_main_tokens / (_tps(full_song_main_tokens, full_song_model_time_seconds) * 1.10))
                if _tps(full_song_main_tokens, full_song_model_time_seconds)
                else None
            ),
        },
        "model_generate_ledger": {
            "cuda_event_seconds": generate_cuda_seconds,
            "host_gap_seconds": host_gap_seconds,
            "model_seconds": model_seconds,
            "host_gap_fraction_of_model_time": (
                host_gap_seconds / model_seconds
                if model_seconds > 0
                else None
            ),
        },
        "active_event_projections": _active_event_projections(
            active_summary,
            baseline_model_seconds=full_song_model_time_seconds,
            full_song_main_tokens=full_song_main_tokens,
        ),
        "island_replay_seconds": {
            "full_forward": full_forward_s,
            "decoder_stack_hidden": decoder_stack_s,
            "decoder_stack_plus_projection": decoder_stack_projection_s,
            "decoder_layers": decoder_layer_s,
            "decoder_layer_segments": decoder_layer_segments,
        },
        "production_minus_full_forward_replay_seconds": (
            full_song_model_time_seconds - full_forward_s
            if full_forward_s is not None
            else None
        ),
        "production_minus_decoder_stack_projection_seconds": (
            full_song_model_time_seconds - decoder_stack_projection_s
            if decoder_stack_projection_s is not None
            else None
        ),
        "notes": [
            "Diagnostic profile timings are not throughput claims.",
            "Active-prefix CUDA-event ranges are nested/non-exclusive and should not be added together.",
            "Graph-replay island timings are target sizing, not end-to-end generation timings.",
        ],
    }
    result["recommendations"] = _recommended_next_steps(result)
    return result


def print_summary(report: dict[str, Any], *, event_limit: int) -> None:
    baseline = report["accepted_baseline"]
    target = report["target"]
    ledger = report["model_generate_ledger"]
    print("Decode Runtime Gap Summary")
    print(f"  label: {report['label']}")
    print(
        "  accepted baseline: "
        f"{baseline['full_song_main_tokens']} tokens, "
        f"{baseline['full_song_model_time_seconds']:.3f}s, "
        f"{baseline['tokens_per_second']:.3f} tok/s"
    )
    print(
        "  target: "
        f"{target['tokens_per_second']:.1f} tok/s requires "
        f"{target['required_model_time_seconds']:.3f}s "
        f"({target['required_saving_seconds']:.3f}s saved)"
    )
    print(
        "  model.generate ledger: "
        f"cuda={ledger['cuda_event_seconds']:.6f}s, "
        f"host_gap={ledger['host_gap_seconds']:.6f}s"
    )
    print()
    print("Projected Active-Loop Event Ceilings")
    for event in report["active_event_projections"][:event_limit]:
        tps = event["projected_tps_if_removed_from_baseline"]
        tps_text = "n/a" if tps is None else f"{tps:.3f}"
        tag = " aggregate" if event.get("is_aggregate_or_composite") else ""
        print(
            f"  {event['projected_full_song_seconds']:.3f}s  "
            f"if_removed={tps_text} tok/s  "
            f"{event['name']}{tag}"
        )
    print()
    print("Graph-Replay Islands")
    islands = report["island_replay_seconds"]
    for key in ("full_forward", "decoder_stack_hidden", "decoder_stack_plus_projection", "decoder_layers"):
        value = islands.get(key)
        value_text = "n/a" if value is None else f"{value:.3f}s"
        print(f"  {key}: {value_text}")
    print()
    print("Recommendations")
    for item in report["recommendations"]:
        print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine production profile diagnostics and graph-replay island reports into a "
            "bottleneck/gap summary. Diagnostic only; not a throughput claim."
        )
    )
    parser.add_argument("profile", type=Path, help="profile_inference JSON with generation records")
    parser.add_argument("--label", default="main_generation")
    parser.add_argument("--active-summary", type=Path, default=None)
    parser.add_argument("--full-forward-report", type=Path, default=None)
    parser.add_argument("--decoder-stack-report", type=Path, default=None)
    parser.add_argument("--decoder-layer-report", type=Path, default=None)
    parser.add_argument("--full-song-main-tokens", type=int, default=7639)
    parser.add_argument("--full-song-model-time-seconds", type=float, default=28.243)
    parser.add_argument("--target-tps", type=float, default=500.0)
    parser.add_argument("--event-limit", type=int, default=12)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    report = summarize_runtime_gap(
        _load_json(args.profile) or {},
        label=args.label,
        full_song_main_tokens=args.full_song_main_tokens,
        full_song_model_time_seconds=args.full_song_model_time_seconds,
        target_tps=args.target_tps,
        active_summary=_load_json(args.active_summary),
        full_forward_report=_load_json(args.full_forward_report),
        decoder_stack_report=_load_json(args.decoder_stack_report),
        decoder_layer_report=_load_json(args.decoder_layer_report),
    )
    print_summary(report, event_limit=args.event_limit)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
