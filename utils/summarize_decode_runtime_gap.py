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
    "finished_check",
    "graph.capture",
    "graph.input_copy",
    "graph.signature_lookup",
    "logits_processor",
    "prepare_inputs",
    "stopping_criteria",
    "prefill_forward",
    "sampling.multinomial",
}

GRAPH_REPLAY_EVENT_NAMES = {
    "graph.replay",
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


def _event_target_class(event_name: str) -> str:
    if event_name in AGGREGATE_EVENT_NAMES:
        return "aggregate_or_composite"
    if event_name in GRAPH_REPLAY_EVENT_NAMES:
        return "graph_replayed_decoder_compute"
    if event_name in CONTROL_EVENT_NAMES:
        return "control_or_setup"
    return "unknown"


def _event_action_note(event_name: str, target_class: str, projected_seconds: float, five_pct_bar: float) -> str:
    if target_class == "aggregate_or_composite":
        return "Do not target directly; split into exclusive child ranges first."
    if target_class == "graph_replayed_decoder_compute":
        return "Target only with broad decoder-layer/runtime compute work; this contains necessary model math."
    if projected_seconds < five_pct_bar:
        return "Below the 5% full-song model-time bar as a standalone target."
    if event_name in {"prepare_inputs", "stopping_criteria", "prefill_forward"}:
        return "Requires low-perturb exclusive proof; prior standalone control-path attempts overstated production gains."
    return "Potentially target-sized, but needs exclusive current-stack proof before production code."


def _active_event_projections(
        active_summary: dict[str, Any] | None,
        *,
        baseline_model_seconds: float,
        full_song_main_tokens: int,
        target_model_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(active_summary, dict):
        return []
    generated_tokens = int(active_summary.get("generated_tokens", 0) or 0)
    if generated_tokens <= 0:
        return []
    scale = full_song_main_tokens / generated_tokens
    baseline_tps = _tps(full_song_main_tokens, baseline_model_seconds)
    five_pct_throughput_saving = (
        baseline_model_seconds - (full_song_main_tokens / (baseline_tps * 1.05))
        if baseline_tps
        else None
    )
    ten_pct_throughput_saving = (
        baseline_model_seconds - (full_song_main_tokens / (baseline_tps * 1.10))
        if baseline_tps
        else None
    )
    rows = []
    cuda_events = active_summary.get("cuda_event_ms") or {}
    calls = active_summary.get("cuda_event_calls") or {}
    for name, milliseconds in cuda_events.items():
        if not isinstance(milliseconds, (int, float)):
            continue
        event_name = str(name)
        event_seconds = float(milliseconds) / 1000.0
        projected_seconds = event_seconds * scale
        five_pct_bar = baseline_model_seconds * 0.05
        ten_pct_bar = baseline_model_seconds * 0.10
        required_saving_for_target = baseline_model_seconds - target_model_seconds
        is_aggregate = event_name in AGGREGATE_EVENT_NAMES
        target_class = _event_target_class(event_name)
        if_removed_tps = _if_removed_tps(
            full_song_main_tokens,
            baseline_model_seconds,
            projected_seconds,
        )
        remaining_seconds_after_removed = baseline_model_seconds - projected_seconds
        rows.append({
            "name": event_name,
            "target_class": target_class,
            "is_aggregate_or_composite": is_aggregate,
            "is_control_or_setup": event_name in CONTROL_EVENT_NAMES,
            "contains_required_model_math": event_name in GRAPH_REPLAY_EVENT_NAMES,
            "exclusive_proof_required": target_class != "unknown",
            "standalone_target_candidate": target_class == "unknown",
            "diagnostic_event_seconds": event_seconds,
            "projected_full_song_seconds": projected_seconds,
            "calls": int(calls.get(name, 0) or 0),
            "projected_fraction_of_baseline_model_time": (
                projected_seconds / baseline_model_seconds
                if baseline_model_seconds > 0
                else None
            ),
            "projected_tps_if_removed_from_baseline": if_removed_tps,
            "throughput_speedup_pct_if_removed": (
                ((if_removed_tps / baseline_tps) - 1.0) * 100.0
                if if_removed_tps is not None and baseline_tps
                else None
            ),
            "remaining_seconds_after_removed": remaining_seconds_after_removed,
            "missing_seconds_to_target_after_removed": max(
                remaining_seconds_after_removed - target_model_seconds,
                0.0,
            ),
            "meets_5pct_model_time_bar": projected_seconds >= five_pct_bar,
            "meets_10pct_model_time_bar": projected_seconds >= ten_pct_bar,
            "meets_5pct_throughput_bar": (
                projected_seconds >= five_pct_throughput_saving
                if five_pct_throughput_saving is not None
                else False
            ),
            "meets_10pct_throughput_bar": (
                projected_seconds >= ten_pct_throughput_saving
                if ten_pct_throughput_saving is not None
                else False
            ),
            "closes_target_tps_gap_if_fully_removed": projected_seconds >= required_saving_for_target,
            "reaches_target_tps_if_fully_removed": (
                if_removed_tps is not None and if_removed_tps >= (full_song_main_tokens / target_model_seconds)
            ),
            "action_note": _event_action_note(event_name, target_class, projected_seconds, five_pct_bar),
        })
    return sorted(rows, key=lambda item: item["projected_full_song_seconds"], reverse=True)


def _island_gap_analysis(
        *,
        full_song_main_tokens: int,
        baseline_model_seconds: float,
        target_model_seconds: float,
        islands: dict[str, Any],
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for name, seconds in islands.items():
        if name == "decoder_layer_segments" or not isinstance(seconds, (int, float)):
            continue
        saved = baseline_model_seconds - float(seconds)
        analysis[name] = {
            "replay_seconds": float(seconds),
            "tps_if_production_reached_replay": _tps(full_song_main_tokens, float(seconds)),
            "saving_vs_accepted_baseline_seconds": saved,
            "saving_fraction_of_baseline_model_time": (
                saved / baseline_model_seconds
                if baseline_model_seconds > 0
                else None
            ),
            "would_reach_target_tps_if_production_reached_replay": float(seconds) <= target_model_seconds,
            "note": (
                "Ceiling only: graph replay removes surrounding generation/runtime overhead and may omit required production work."
            ),
        }
    return analysis


def _candidate_ledger(
        *,
        active_events: list[dict[str, Any]],
        island_analysis: dict[str, Any],
        full_song_main_tokens: int,
        baseline_model_seconds: float,
        target_model_seconds: float,
) -> list[dict[str, Any]]:
    baseline_tps = _tps(full_song_main_tokens, baseline_model_seconds)
    rows: list[dict[str, Any]] = []
    for event in active_events:
        saving = event.get("projected_full_song_seconds")
        remaining = event.get("remaining_seconds_after_removed")
        tps = event.get("projected_tps_if_removed_from_baseline")
        if not isinstance(saving, (int, float)) or not isinstance(remaining, (int, float)):
            continue
        rows.append({
            "source": "active_event",
            "name": event.get("name"),
            "target_class": event.get("target_class"),
            "idealized_saving_seconds": float(saving),
            "idealized_remaining_seconds": float(remaining),
            "idealized_tps": tps,
            "throughput_speedup_pct": event.get("throughput_speedup_pct_if_removed"),
            "meets_5pct_model_time_bar": event.get("meets_5pct_model_time_bar"),
            "meets_10pct_model_time_bar": event.get("meets_10pct_model_time_bar"),
            "meets_5pct_throughput_bar": event.get("meets_5pct_throughput_bar"),
            "meets_10pct_throughput_bar": event.get("meets_10pct_throughput_bar"),
            "reaches_target_tps": event.get("reaches_target_tps_if_fully_removed"),
            "missing_seconds_to_target": event.get("missing_seconds_to_target_after_removed"),
            "exclusive_proof_required": event.get("exclusive_proof_required"),
            "standalone_target_candidate": event.get("standalone_target_candidate"),
            "caveat": event.get("action_note"),
        })
    for name, analysis in island_analysis.items():
        if not isinstance(analysis, dict):
            continue
        saving = analysis.get("saving_vs_accepted_baseline_seconds")
        remaining = analysis.get("replay_seconds")
        tps = analysis.get("tps_if_production_reached_replay")
        if not isinstance(saving, (int, float)) or not isinstance(remaining, (int, float)):
            continue
        rows.append({
            "source": "island_replay",
            "name": name,
            "target_class": "idealized_graph_replay_boundary",
            "idealized_saving_seconds": float(saving),
            "idealized_remaining_seconds": float(remaining),
            "idealized_tps": tps,
            "throughput_speedup_pct": (
                ((float(tps) / baseline_tps) - 1.0) * 100.0
                if isinstance(tps, (int, float)) and baseline_tps
                else None
            ),
            "meets_5pct_model_time_bar": float(saving) >= baseline_model_seconds * 0.05,
            "meets_10pct_model_time_bar": float(saving) >= baseline_model_seconds * 0.10,
            "meets_5pct_throughput_bar": (
                isinstance(tps, (int, float)) and baseline_tps is not None and float(tps) >= baseline_tps * 1.05
            ),
            "meets_10pct_throughput_bar": (
                isinstance(tps, (int, float)) and baseline_tps is not None and float(tps) >= baseline_tps * 1.10
            ),
            "reaches_target_tps": float(remaining) <= target_model_seconds,
            "missing_seconds_to_target": max(float(remaining) - target_model_seconds, 0.0),
            "exclusive_proof_required": True,
            "standalone_target_candidate": False,
            "caveat": analysis.get("note"),
        })
    return sorted(rows, key=lambda item: item["idealized_saving_seconds"], reverse=True)


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
    top_event = next(
        (
            event for event in events
            if isinstance(event, dict) and not event.get("is_aggregate_or_composite")
        ),
        None,
    )
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
        event for event in events
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
    event_actions = [
        event for event in events
        if event.get("standalone_target_candidate")
        and event.get("meets_5pct_model_time_bar")
    ]
    if not event_actions:
        recommendations.append(
            "No standalone active-loop event is both exclusive-looking and above the 5% bar; avoid micro-optimizing tail/setup paths."
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
    island_replay_seconds = {
        "full_forward": full_forward_s,
        "decoder_stack_hidden": decoder_stack_s,
        "decoder_stack_plus_projection": decoder_stack_projection_s,
        "decoder_layers": decoder_layer_s,
        "decoder_layer_segments": decoder_layer_segments,
    }

    active_event_projections = _active_event_projections(
        active_summary,
        baseline_model_seconds=full_song_model_time_seconds,
        full_song_main_tokens=full_song_main_tokens,
        target_model_seconds=target_model_seconds,
    )
    island_gap_analysis = _island_gap_analysis(
        full_song_main_tokens=full_song_main_tokens,
        baseline_model_seconds=full_song_model_time_seconds,
        target_model_seconds=target_model_seconds,
        islands=island_replay_seconds,
    )

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
        "active_event_projections": active_event_projections,
        "island_replay_seconds": island_replay_seconds,
        "island_gap_analysis": island_gap_analysis,
        "candidate_ledger": _candidate_ledger(
            active_events=active_event_projections,
            island_analysis=island_gap_analysis,
            full_song_main_tokens=full_song_main_tokens,
            baseline_model_seconds=full_song_model_time_seconds,
            target_model_seconds=target_model_seconds,
        ),
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
        bar_tags = []
        if event.get("meets_10pct_model_time_bar"):
            bar_tags.append(">=10%")
        elif event.get("meets_5pct_model_time_bar"):
            bar_tags.append(">=5%")
        if event.get("closes_target_tps_gap_if_fully_removed"):
            bar_tags.append("closes_500_gap")
        class_text = event.get("target_class", "unknown")
        bar_text = "" if not bar_tags else " " + ",".join(bar_tags)
        print(
            f"  {event['projected_full_song_seconds']:.3f}s  "
            f"if_removed={tps_text} tok/s  "
            f"{event['name']}{tag}  "
            f"class={class_text}{bar_text}"
        )
    print()
    print("Graph-Replay Islands")
    islands = report["island_replay_seconds"]
    for key in ("full_forward", "decoder_stack_hidden", "decoder_stack_plus_projection", "decoder_layers"):
        value = islands.get(key)
        value_text = "n/a" if value is None else f"{value:.3f}s"
        analysis = (report.get("island_gap_analysis") or {}).get(key) or {}
        reach = analysis.get("would_reach_target_tps_if_production_reached_replay")
        reach_text = "" if reach is None else f" reaches_target={bool(reach)}"
        tps = analysis.get("tps_if_production_reached_replay")
        tps_text = "" if tps is None else f" ({tps:.3f} tok/s ceiling)"
        print(f"  {key}: {value_text}{tps_text}{reach_text}")
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
