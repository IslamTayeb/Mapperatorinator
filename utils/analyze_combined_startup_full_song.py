from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .analyze_lazy_startup_full_song import (
        LABELS,
        _aggregate,
        _comparison,
        _load_run,
        _same,
    )
except ImportError:
    from analyze_lazy_startup_full_song import (
        LABELS,
        _aggregate,
        _comparison,
        _load_run,
        _same,
    )


PRECISIONS = ("fp32", "fp16")
RUN_NAMES = ("run-01", "run-02", "run-03")
PERFORMANCE_VARIANTS = ("lazy_parent", "aot_parent", "combined")
AUDIT_VARIANTS = PERFORMANCE_VARIANTS + ("combined_fallback",)


class CombinedFullSongGateError(RuntimeError):
    pass


def _variant_contract(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    return {
        "lazy_parent": (args.lazy_commit, args.lazy_branch),
        "aot_parent": (args.aot_commit, args.aot_branch),
        "combined": (args.combined_commit, args.combined_branch),
        "combined_fallback": (args.combined_commit, args.combined_branch),
    }


def _analyze_precision(
    root: Path,
    *,
    precision: str,
    expected: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for variant in PERFORMANCE_VARIANTS:
        commit, branch = expected[variant]
        runs[variant] = [
            _load_run(
                root / precision / variant / run_name,
                precision=precision,
                commit=commit,
                branch=branch,
                audit=False,
            )
            for run_name in RUN_NAMES
        ]
    audits = {}
    strict_audit = precision == "fp32"
    for variant in AUDIT_VARIANTS:
        commit, branch = expected[variant]
        audits[variant] = _load_run(
            root / precision / variant / "exactness-audit",
            precision=precision,
            commit=commit,
            branch=branch,
            audit=strict_audit,
        )

    all_rows = [
        row for variant in PERFORMANCE_VARIANTS for row in runs[variant]
    ] + list(audits.values())
    checks = {
        "output_bytes": _same([row["output_sha256"] for row in all_rows]),
        "output_structure": _same([row["output_structure"] for row in all_rows]),
        "workload": _same([row["workload"] for row in all_rows]),
        "preset": _same([row["preset"] for row in all_rows]),
        "dispatch_and_graph": _same([row["records"] for row in all_rows]),
        "graph_signature": _same([row["graph_sha256"] for row in all_rows]),
        "token_and_stopping": _same([row["signatures"] for row in all_rows]),
        "rng_and_cache": (
            _same(
                [audits[variant]["strict_exactness"] for variant in AUDIT_VARIANTS]
            )
            if strict_audit
            else None
        ),
        "same_precision_fp16_evidence": (
            _same([audits[variant]["signatures"] for variant in AUDIT_VARIANTS])
            if not strict_audit
            else None
        ),
    }
    checks = {key: value for key, value in checks.items() if value is not None}

    aggregates: dict[str, Any] = {}
    for variant in PERFORMANCE_VARIANTS:
        aggregates[variant] = {
            "process_wall_seconds": _aggregate(
                [row["process_wall_seconds"] for row in runs[variant]]
            ),
            "request_wall_seconds": _aggregate(
                [row["request_wall_seconds"] for row in runs[variant]]
            ),
            "main_model_seconds": _aggregate(
                [
                    row["generation"]["main_generation"][
                        "synchronized_model_seconds"
                    ]
                    for row in runs[variant]
                ]
            ),
            "timing_model_seconds": _aggregate(
                [
                    row["generation"]["timing_context"][
                        "synchronized_model_seconds"
                    ]
                    for row in runs[variant]
                ]
            ),
            "peak_cuda_memory_mb": max(
                row["peak_cuda_memory_mb"] for row in runs[variant]
            ),
        }

    comparisons = {
        parent: {
            metric: _comparison(
                aggregates[parent][metric]["median"],
                aggregates["combined"][metric]["median"],
            )
            for metric in (
                "process_wall_seconds",
                "request_wall_seconds",
                "main_model_seconds",
                "timing_model_seconds",
            )
        }
        for parent in ("lazy_parent", "aot_parent")
    }
    for parent in ("lazy_parent", "aot_parent"):
        parent_comparisons = comparisons[parent]
        checks[f"combined_process_saves_one_second_vs_{parent}"] = (
            parent_comparisons["process_wall_seconds"]["saved_seconds"] >= 1.0
        )
        for metric in (
            "request_wall_seconds",
            "main_model_seconds",
            "timing_model_seconds",
        ):
            checks[f"combined_{metric}_no_more_than_one_percent_slower_vs_{parent}"] = (
                parent_comparisons[metric]["candidate_delta_pct"] <= 1.0
            )

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "runs": runs,
        "exactness_audits": audits,
        "audit_kind": "strict_fp32" if strict_audit else "same_precision_fp16",
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    expected = _variant_contract(args)
    results = {
        precision: _analyze_precision(
            args.run_root,
            precision=precision,
            expected=expected,
        )
        for precision in PRECISIONS
    }
    return {
        "schema_version": 1,
        "status": "PASS"
        if all(result["status"] == "PASS" for result in results.values())
        else "FAIL",
        "precision_results": results,
    }


def _text(report: dict[str, Any]) -> str:
    rows = [f"status={report['status']}"]
    for precision in PRECISIONS:
        result = report["precision_results"][precision]
        rows.append(f"{precision}.status={result['status']}")
        for parent, comparisons in result["comparisons"].items():
            for metric, comparison in comparisons.items():
                prefix = f"{precision}.combined_vs_{parent}.{metric}"
                rows.append(
                    f"{prefix}.saved_seconds={comparison['saved_seconds']:.6f}"
                )
                rows.append(
                    f"{prefix}.candidate_delta_pct="
                    f"{comparison['candidate_delta_pct']:.3f}"
                )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lazy-commit", required=True)
    parser.add_argument("--lazy-branch", required=True)
    parser.add_argument("--aot-commit", required=True)
    parser.add_argument("--aot-branch", required=True)
    parser.add_argument("--combined-commit", required=True)
    parser.add_argument("--combined-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(_text(report))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
