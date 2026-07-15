from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_COMPARISON_GATES = (
    "same_calculation_pass",
    "token_equivalence_pass",
    "output_artifact_pass",
    "performance_pass",
)
DISPATCH_PARITY_FIELDS = (
    "optimized_effective_config_version",
    "optimized_dispatch_mode",
    "optimized_dispatch_policy",
    "optimized_dispatch_capture_hits",
)
PROFILE_LABELS = frozenset(("main_generation", "timing_context"))


def _records(profile: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in profile.get("generation", [])
        if isinstance(row, Mapping) and row.get("profile_label") in PROFILE_LABELS
    ]


def validate(
    comparison: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    precision: str,
) -> dict[str, Any]:
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    for key in REQUIRED_COMPARISON_GATES:
        if comparison.get(key) is not True:
            raise ValueError(f"reciprocal comparison failed {key}")

    expected_labels = {
        "baseline_first",
        "candidate_second",
        "candidate_first",
        "baseline_second",
    }
    if set(profiles) != expected_labels:
        raise ValueError(
            f"profiles must contain reciprocal labels {sorted(expected_labels)}"
        )
    for label, profile in profiles.items():
        if profile.get("metadata", {}).get("precision") != precision:
            raise ValueError(f"{label}: precision metadata mismatch")

    for label in ("baseline_first", "baseline_second"):
        for row in _records(profiles[label]):
            if "shared_static_input_arena" in row.get(
                "optimized_cuda_graphs", {}
            ):
                raise ValueError(f"{label}: baseline unexpectedly used shared arena")

    candidate_summary: dict[str, dict[str, int]] = {}
    for label in ("candidate_second", "candidate_first"):
        rows = _records(profiles[label])
        if not rows:
            raise ValueError(f"{label}: missing generation rows")
        arena_rows = 0
        total_refreshes = 0
        total_graph_entries = 0
        total_owned_bytes = 0
        total_avoided_bytes = 0
        for row in rows:
            graphs = row.get("optimized_cuda_graphs", {})
            arena = graphs.get("shared_static_input_arena")
            if not isinstance(arena, Mapping):
                if int(graphs.get("graph_count", 0)) != 0:
                    raise ValueError(
                        f"{label}: graphed row lacks shared arena evidence"
                    )
                continue
            if arena.get("enabled") is not True:
                raise ValueError(f"{label}: shared arena is not enabled")
            arena_rows += 1
            for key in (
                "arena_count",
                "graph_entries",
                "owned_tensor_bytes",
            ):
                if int(arena.get(key, 0)) <= 0:
                    raise ValueError(f"{label}: invalid arena counter {key}")
            for key in (
                "refreshes",
                "refresh_copy_calls",
                "refresh_copy_bytes",
                "avoided_duplicate_graph_input_bytes",
            ):
                if int(arena.get(key, -1)) < 0:
                    raise ValueError(f"{label}: invalid arena counter {key}")
            total_refreshes += int(arena["refreshes"])
            total_graph_entries += int(arena["graph_entries"])
            total_owned_bytes += int(arena["owned_tensor_bytes"])
            total_avoided_bytes += int(
                arena["avoided_duplicate_graph_input_bytes"]
            )
        if arena_rows == 0 or total_refreshes == 0:
            raise ValueError(f"{label}: full song did not exercise arena refresh")
        candidate_summary[label] = {
            "arena_rows": arena_rows,
            "refreshes": total_refreshes,
            "graph_entries": total_graph_entries,
            "owned_tensor_bytes": total_owned_bytes,
            "avoided_duplicate_graph_input_bytes": total_avoided_bytes,
        }

    for baseline_label, candidate_label in (
        ("baseline_first", "candidate_second"),
        ("baseline_second", "candidate_first"),
    ):
        baseline_rows = _records(profiles[baseline_label])
        candidate_rows = _records(profiles[candidate_label])
        if len(baseline_rows) != len(candidate_rows):
            raise ValueError("dispatch evidence row count changed")
        for baseline, candidate in zip(
            baseline_rows,
            candidate_rows,
            strict=True,
        ):
            for key in DISPATCH_PARITY_FIELDS:
                if baseline.get(key) != candidate.get(key):
                    raise ValueError(f"dispatch parity failed for {key}")

    return {
        "schema_version": 1,
        "status": "PASS",
        "precision": precision,
        "comparison_gates": {
            key: True for key in REQUIRED_COMPARISON_GATES
        },
        "dispatch_parity": True,
        "candidate_evidence": candidate_summary,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reciprocal", type=Path, required=True)
    parser.add_argument("--baseline-first", type=Path, required=True)
    parser.add_argument("--candidate-second", type=Path, required=True)
    parser.add_argument("--candidate-first", type=Path, required=True)
    parser.add_argument("--baseline-second", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()

    result = validate(
        _load(args.reciprocal),
        {
            "baseline_first": _load(args.baseline_first),
            "candidate_second": _load(args.candidate_second),
            "candidate_first": _load(args.candidate_first),
            "baseline_second": _load(args.baseline_second),
        },
        precision=args.precision,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence = result["candidate_evidence"]
    args.text_output.write_text(
        "k1_shared_static_input_arena_reciprocal=PASS\n"
        f"precision={args.precision}\n"
        f"candidate_second_refreshes={evidence['candidate_second']['refreshes']}\n"
        f"candidate_first_refreshes={evidence['candidate_first']['refreshes']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
