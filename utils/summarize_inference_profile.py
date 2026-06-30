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


def summarize(path: Path, *, limit: int) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Mapperatorinator inference profile JSON.")
    parser.add_argument("profile", type=Path, help="Path to a .profile.json file.")
    parser.add_argument("--limit", type=int, default=12, help="Number of stage/window rows to print.")
    args = parser.parse_args()
    summarize(args.profile, limit=args.limit)


if __name__ == "__main__":
    main()
