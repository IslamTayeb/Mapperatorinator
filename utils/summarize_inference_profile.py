from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt_seconds(value: Any) -> str:
    try:
        return f"{float(value):8.3f}s"
    except (TypeError, ValueError):
        return "     n/a"


def _print_table(title: str, rows: list[tuple[str, Any]], limit: int | None = None) -> None:
    print(title)
    for name, value in rows[:limit]:
        print(f"  {_fmt_seconds(value)}  {name}")
    print()


def _generation_name(record: dict[str, Any]) -> str:
    context = record.get("context_type", "unknown")
    label = record.get("profile_label", "unknown")
    mode = record.get("mode", "unknown")
    if "sequence_index" in record:
        unit = f"seq={record['sequence_index']}"
    else:
        unit = f"batch_start={record.get('batch_start_index', 'n/a')}"
    tokens = record.get("generated_tokens", "n/a")
    tok_s = record.get("tokens_per_second")
    tok_s_text = f", {tok_s:.1f} tok/s" if isinstance(tok_s, (int, float)) else ""
    return f"{label}/{context}/{mode}/{unit}, generated={tokens}{tok_s_text}"


def _load_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path, *, limit: int) -> None:
    profile = _load_profile(path)
    print(f"Profile: {path}")

    metadata = profile.get("metadata", {})
    if metadata:
        print("Run")
        for key in [
            "audio_path",
            "result_path",
            "device",
            "precision",
            "attn_implementation",
            "use_server",
            "parallel",
            "sequence_count",
            "song_length_ms",
        ]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")
        print()

    stages = profile.get("stages", [])
    stage_rows = sorted(
        ((stage.get("name", "unknown"), stage.get("wall_seconds")) for stage in stages),
        key=lambda row: float(row[1] or 0.0),
        reverse=True,
    )
    _print_table("Stages by wall time", stage_rows, limit)

    generation = profile.get("generation", [])
    generation_rows = sorted(
        ((_generation_name(record), record.get("wall_seconds")) for record in generation),
        key=lambda row: float(row[1] or 0.0),
        reverse=True,
    )
    _print_table("Slowest generation records by outer wall time", generation_rows, limit)

    summary = profile.get("summary", {})
    by_context = summary.get("generation_by_context", {})
    if by_context:
        print("Generation by context")
        for context, values in sorted(
                by_context.items(),
                key=lambda item: float(item[1].get("model_elapsed_seconds", 0.0) or 0.0),
                reverse=True,
        ):
            elapsed = values.get("model_elapsed_seconds", 0.0)
            wall = values.get("wall_seconds", 0.0)
            tokens = values.get("generated_tokens", 0)
            tok_s = values.get("tokens_per_second", 0.0)
            records = values.get("records", 0)
            print(
                f"  {context}: model={_fmt_seconds(elapsed)}, wall={_fmt_seconds(wall)}, "
                f"tokens={tokens}, tok/s={tok_s:.1f}, records={records}"
            )


def _summary_for_label(profile: dict[str, Any], label: str) -> dict[str, Any]:
    by_label = profile.get("summary", {}).get("generation_by_label", {})
    if label in by_label:
        return by_label[label]
    return {}


def _flatten_token_ids(profile: dict[str, Any], label: str) -> list[int] | None:
    tokens: list[int] = []
    saw_tokens = False
    for record in profile.get("generation", []):
        if record.get("profile_label") != label:
            continue
        if "generated_token_ids" in record:
            value = record.get("generated_token_ids")
            if value is None:
                continue
            saw_tokens = True
            tokens.extend(int(token) for token in value)
        elif "generated_token_ids_per_sample" in record:
            value = record.get("generated_token_ids_per_sample")
            if value is None:
                continue
            saw_tokens = True
            for sample in value:
                tokens.extend(int(token) for token in sample)
    return tokens if saw_tokens else None


def _compare_number(name: str, baseline: float, candidate: float, *, higher_is_better: bool) -> None:
    delta = candidate - baseline
    pct = (delta / baseline * 100.0) if baseline else 0.0
    direction = "better" if (delta >= 0) == higher_is_better else "worse"
    print(f"  {name}: baseline={baseline:.3f}, candidate={candidate:.3f}, delta={delta:+.3f} ({pct:+.1f}%, {direction})")


def compare_profiles(baseline_path: Path, candidate_path: Path, *, label: str) -> None:
    baseline = _load_profile(baseline_path)
    candidate = _load_profile(candidate_path)
    baseline_summary = _summary_for_label(baseline, label)
    candidate_summary = _summary_for_label(candidate, label)

    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    print(f"Label:     {label}")
    print()

    if not baseline_summary or not candidate_summary:
        print("Missing generation summary for requested label.")
        return

    _compare_number(
        "tokens_per_second",
        float(baseline_summary.get("tokens_per_second", 0.0) or 0.0),
        float(candidate_summary.get("tokens_per_second", 0.0) or 0.0),
        higher_is_better=True,
    )
    _compare_number(
        "model_elapsed_seconds",
        float(baseline_summary.get("model_elapsed_seconds", 0.0) or 0.0),
        float(candidate_summary.get("model_elapsed_seconds", 0.0) or 0.0),
        higher_is_better=False,
    )
    _compare_number(
        "outer_wall_seconds",
        float(baseline_summary.get("wall_seconds", 0.0) or 0.0),
        float(candidate_summary.get("wall_seconds", 0.0) or 0.0),
        higher_is_better=False,
    )
    print(f"  generated_tokens: baseline={baseline_summary.get('generated_tokens')}, candidate={candidate_summary.get('generated_tokens')}")
    print(f"  records: baseline={baseline_summary.get('records')}, candidate={candidate_summary.get('records')}")
    print()

    baseline_tokens = _flatten_token_ids(baseline, label)
    candidate_tokens = _flatten_token_ids(candidate, label)
    if baseline_tokens is None or candidate_tokens is None:
        print("Token equivalence: not checked; rerun with profile_record_token_ids=true.")
        return
    if baseline_tokens == candidate_tokens:
        print(f"Token equivalence: PASS ({len(baseline_tokens)} generated token IDs match).")
        return

    mismatch = next(
        (idx for idx, pair in enumerate(zip(baseline_tokens, candidate_tokens)) if pair[0] != pair[1]),
        min(len(baseline_tokens), len(candidate_tokens)),
    )
    print(
        "Token equivalence: FAIL "
        f"(baseline_len={len(baseline_tokens)}, candidate_len={len(candidate_tokens)}, first_mismatch={mismatch})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Mapperatorinator inference profile JSON.")
    parser.add_argument("profile", type=Path, nargs="?", help="Path to a .profile.json file.")
    parser.add_argument("--limit", type=int, default=12, help="Number of stage/window rows to print.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        type=Path,
        help="Compare two profile JSON files.",
    )
    parser.add_argument("--label", default="main_generation", help="Generation label to compare.")
    args = parser.parse_args()
    if args.compare:
        compare_profiles(args.compare[0], args.compare[1], label=args.label)
    elif args.profile:
        summarize(args.profile, limit=args.limit)
    else:
        parser.error("provide a profile path or --compare BASELINE CANDIDATE")


if __name__ == "__main__":
    main()
