from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tps(tokens: int, seconds: float) -> float | None:
    return tokens / seconds if tokens > 0 and seconds > 0 else None


def _first_signature(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    signatures = report.get("signature_reports")
    if not isinstance(signatures, dict) or not signatures:
        raise ValueError("decoder-layer report has no signature_reports")
    key = sorted(signatures)[0]
    value = signatures[key]
    if not isinstance(value, dict):
        raise ValueError(f"decoder-layer signature {key!r} is malformed")
    return key, value


def _result_ms(signature_report: dict[str, Any]) -> dict[str, float]:
    results = signature_report.get("results")
    if not isinstance(results, dict):
        raise ValueError("signature report has no results")
    values: dict[str, float] = {}
    for name, result in results.items():
        if not isinstance(result, dict):
            continue
        graph_ms = result.get("cuda_graph_replay_ms_per_call")
        if isinstance(graph_ms, (int, float)) and result.get("cuda_graph_replay_allclose", True):
            values[name] = float(graph_ms)
    return values


def _row_prefix(row: dict[str, Any]) -> int:
    return int(row.get("active_prefix_length") or row.get("prefix") or 0)


def _row_replays(row: dict[str, Any]) -> int:
    return int(row.get("decode_replays") or 0)


def summarize_weighted_decoder_layer_island(
        weighted_stack_summary: dict[str, Any],
        *,
        report_dir: Path,
        report_template: str,
        full_song_model_time_s: float | None,
        full_song_main_tokens: int | None,
        require_variants: list[str],
        require_candidate_cache_write_checks: bool,
        target_tps: float,
) -> dict[str, Any]:
    rows = weighted_stack_summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("weighted stack summary has no rows")
    if full_song_model_time_s is None:
        full_song_model_time_s = float(weighted_stack_summary.get("full_song_model_time_s") or 0.0)
    if full_song_main_tokens is None:
        full_song_main_tokens = int(weighted_stack_summary.get("full_song_main_tokens") or 0)
    if full_song_model_time_s <= 0 or full_song_main_tokens <= 0:
        raise ValueError("missing full-song model time or token count")

    target_model_s = full_song_main_tokens / float(target_tps)
    five_pct_model_s = full_song_model_time_s * 0.05
    ten_pct_model_s = full_song_model_time_s * 0.10

    weighted_seconds: dict[str, float] = {}
    bucket_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    total_decode_replays = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        prefix = _row_prefix(row)
        decode_replays = _row_replays(row)
        if prefix <= 0 or decode_replays <= 0:
            continue
        total_decode_replays += decode_replays
        path = report_dir / report_template.format(prefix=prefix)
        if not path.exists():
            failures.append(f"prefix {prefix}: missing report {path}")
            continue
        report = _load_json(path)
        if report.get("pass") is not True:
            failures.append(f"prefix {prefix}: report pass={report.get('pass')!r}")
        active_prefix = int(report.get("active_prefix_length") or 0)
        if active_prefix != prefix:
            failures.append(f"prefix {prefix}: report active_prefix_length={active_prefix}")
        if require_candidate_cache_write_checks and report.get("candidate_cache_write_checks_pass") is not True:
            failures.append(
                f"prefix {prefix}: candidate_cache_write_checks_pass={report.get('candidate_cache_write_checks_pass')!r}"
            )
        signature, signature_report = _first_signature(report)
        member_count = int(signature_report.get("member_count") or 0)
        if member_count <= 0:
            failures.append(f"prefix {prefix}: missing member_count")
            continue
        ms_by_variant = _result_ms(signature_report)
        for variant in require_variants:
            if variant not in ms_by_variant:
                failures.append(f"prefix {prefix}: missing variant {variant!r}")
        bucket_variant_seconds: dict[str, float] = {}
        for variant, ms in ms_by_variant.items():
            seconds = ms * member_count * decode_replays / 1000.0
            weighted_seconds[variant] = weighted_seconds.get(variant, 0.0) + seconds
            bucket_variant_seconds[variant] = seconds
        bucket_rows.append({
            "prefix": prefix,
            "decode_replays": decode_replays,
            "report": str(path),
            "signature": signature,
            "member_count": member_count,
            "variant_ms_per_call": dict(sorted(ms_by_variant.items())),
            "variant_weighted_seconds": dict(sorted(bucket_variant_seconds.items())),
            "candidate_cache_write_checks_pass": report.get("candidate_cache_write_checks_pass"),
            "logits_replay_allclose": report.get("logits_replay_allclose"),
            "logits_replay_max_abs": report.get("logits_replay_max_abs"),
        })

    repo_seconds = weighted_seconds.get("repo_decoder_layer")
    if repo_seconds is None:
        failures.append("missing weighted repo_decoder_layer seconds")

    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant, seconds in sorted(weighted_seconds.items()):
        saved_vs_repo = repo_seconds - seconds if repo_seconds is not None else None
        projected_model_time = (
            full_song_model_time_s - saved_vs_repo
            if saved_vs_repo is not None
            else None
        )
        variant_summaries[variant] = {
            "weighted_seconds": seconds,
            "fraction_of_model_time": seconds / full_song_model_time_s,
            "saved_vs_repo_s": saved_vs_repo,
            "speedup_vs_repo": (
                repo_seconds / seconds
                if repo_seconds is not None and seconds > 0
                else None
            ),
            "projected_model_time_s": projected_model_time,
            "projected_tokens_per_second": (
                _tps(full_song_main_tokens, projected_model_time)
                if projected_model_time is not None
                else None
            ),
            "clears_5pct_model_bar": (
                saved_vs_repo is not None and saved_vs_repo >= five_pct_model_s
            ),
            "clears_10pct_model_bar": (
                saved_vs_repo is not None and saved_vs_repo >= ten_pct_model_s
            ),
            "reaches_target_tps": (
                projected_model_time is not None and projected_model_time <= target_model_s
            ),
        }

    return {
        "pass": not failures,
        "failures": failures,
        "source_weighted_stack_pass": weighted_stack_summary.get("pass"),
        "source_weighted_stack_job_id": weighted_stack_summary.get("dcc_job_id"),
        "source_weighted_stack_commit": weighted_stack_summary.get("commit"),
        "report_dir": str(report_dir),
        "report_template": report_template,
        "assumptions": {
            "full_song_model_time_s": full_song_model_time_s,
            "full_song_main_tokens": full_song_main_tokens,
            "baseline_tokens_per_second": _tps(full_song_main_tokens, full_song_model_time_s),
            "target_tokens_per_second": float(target_tps),
            "target_model_time_s": target_model_s,
            "five_pct_model_time_s": five_pct_model_s,
            "ten_pct_model_time_s": ten_pct_model_s,
            "weighted_decode_replays": total_decode_replays,
            "bucket_count": len(bucket_rows),
            "require_candidate_cache_write_checks": bool(require_candidate_cache_write_checks),
            "require_variants": require_variants,
        },
        "weighted_seconds": dict(sorted(weighted_seconds.items())),
        "variants": variant_summaries,
        "buckets": sorted(bucket_rows, key=lambda item: item["prefix"]),
        "decision": {
            "manual_decoder_runtime_island_target_sized": (
                variant_summaries.get("manual_decoder_runtime_island", {})
                .get("clears_5pct_model_bar", False)
            ),
            "manual_decoder_runtime_island_strong": (
                variant_summaries.get("manual_decoder_runtime_island", {})
                .get("clears_10pct_model_bar", False)
            ),
            "throughput_claim": False,
            "note": (
                "This is weighted CUDA graph replay target sizing only. "
                "A production speed claim still requires normal untraced profile_inference "
                "with token/output equivalence and no-regression gates."
            ),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Weighted Decoder Layer Island Summary")
    print(f"  pass: {report['pass']}")
    print(f"  buckets: {report['assumptions']['bucket_count']}")
    print(f"  weighted decode replays: {report['assumptions']['weighted_decode_replays']}")
    print(
        "  accepted baseline: "
        f"{report['assumptions']['full_song_main_tokens']} tokens, "
        f"{report['assumptions']['full_song_model_time_s']:.3f}s, "
        f"{report['assumptions']['baseline_tokens_per_second']:.3f} tok/s"
    )
    for variant, summary in report["variants"].items():
        saved = summary.get("saved_vs_repo_s")
        if saved is None:
            continue
        print(
            f"  {variant}: weighted={summary['weighted_seconds']:.3f}s "
            f"saved_vs_repo={saved:.3f}s "
            f"projected_tps={summary['projected_tokens_per_second']:.3f}"
        )
    if report["failures"]:
        print("Failures:")
        for failure in report["failures"]:
            print(f"  - {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-prefix decoder-layer island reports using a full-song "
            "active-prefix replay distribution. Diagnostic only; not a throughput claim."
        )
    )
    parser.add_argument("weighted_stack_summary", type=Path)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--report-template", default="decoder_layer_prefix{prefix}.json")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--full-song-model-time-s", type=float, default=None)
    parser.add_argument("--full-song-main-tokens", type=int, default=None)
    parser.add_argument("--target-tps", type=float, default=500.0)
    parser.add_argument(
        "--require-variant",
        action="append",
        dest="require_variants",
        default=None,
    )
    parser.add_argument("--require-candidate-cache-write-checks", action="store_true")
    args = parser.parse_args()

    report = summarize_weighted_decoder_layer_island(
        _load_json(args.weighted_stack_summary),
        report_dir=args.report_dir,
        report_template=args.report_template,
        full_song_model_time_s=args.full_song_model_time_s,
        full_song_main_tokens=args.full_song_main_tokens,
        require_variants=args.require_variants or ["repo_decoder_layer", "manual_decoder_runtime_island"],
        require_candidate_cache_write_checks=args.require_candidate_cache_write_checks,
        target_tps=args.target_tps,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
