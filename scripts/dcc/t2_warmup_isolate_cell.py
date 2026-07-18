#!/usr/bin/env python3
"""T2 warmup-hoist isolate: measure + T5 greedy seal on one GPU.

Runs sequentially:
  1) baseline measure (temp 0.9, like-with-like vs 50194534)
  2) t2_warmup measure (session_warmup_captures only)
  3) baseline greedy (do_sample=false)
  4) t2_warmup greedy

Compares greedy .osu bytes; writes summary.json + T5_GATES.json.
Timing-stride stays OFF. Tip 55949274 frozen. No PR #120 push.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cell(
    *,
    repo: Path,
    python: Path,
    audio: Path,
    run_root: Path,
    variant: str,
    precision: str,
    seed: int,
    expected_gpu: str,
    greedy: bool,
) -> dict[str, Any]:
    label = f"{variant}-{'greedy' if greedy else 'measure'}"
    cell_root = run_root / label
    cell_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        "scripts/dcc/t2_fullmap_cell.py",
        "--repo",
        str(repo),
        "--python",
        str(python),
        "--audio",
        str(audio),
        "--run-root",
        str(cell_root),
        "--precision",
        precision,
        "--seed",
        str(seed),
        "--variant",
        variant,
        "--expected-gpu-substr",
        expected_gpu,
    ]
    if greedy:
        cmd.append("--greedy")
    with (cell_root / "orchestrator_stdout.txt").open("w") as so, (
        cell_root / "orchestrator_stderr.txt"
    ).open("w") as se:
        rc = subprocess.run(cmd, cwd=str(repo), check=False, stdout=so, stderr=se).returncode
    (cell_root / "orchestrator_exit_code.txt").write_text(str(rc) + "\n")
    if rc != 0:
        raise SystemExit(f"{label} failed rc={rc} under {cell_root}")
    summary_path = cell_root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"{label} missing summary.json")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _collect_osu(root: Path) -> list[tuple[str, bytes]]:
    files = sorted((root / "output").rglob("*.osu"))
    return [(str(p.relative_to(root)), p.read_bytes()) for p in files]


def _build_t5_report(
    *,
    repo: Path,
    commit: str,
    equal: bool,
    greedy_detail: dict[str, Any],
    measure_delta: dict[str, Any],
) -> dict[str, Any]:
    # Prefer live T5 harness if present (t5-quality-gates WT or vendored utils).
    candidates = [
        repo / "utils",
        Path("/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t5-quality-gates/utils"),
        Path("/work/projects/Mapperatorinator-worktrees/t5-quality-gates/utils"),
    ]
    for util_dir in candidates:
        scout = util_dir / "t5_scout_gates.py"
        if not scout.is_file():
            continue
        sys.path.insert(0, str(util_dir.parent))
        from utils.t5_scout_gates import (  # type: ignore
            build_report,
            from_greedy_compare,
            skip_gate,
        )

        greedy_gate = from_greedy_compare(
            {"pass": equal, **greedy_detail},
            artifact="greedy .osu byte compare (do_sample=false)",
        )
        # Warmup-only seal: KS not required (token-preserving lever).
        ks_gate = skip_gate(
            "warmup-hoist isolate: KS N/A for greedy seal; timing_stride held opt-in"
        )
        return build_report(
            track="T2",
            greedy_token_match=greedy_gate,
            ks_parity=ks_gate,
            scout="t2_warmup_isolate_cell",
            commit=commit,
            force_greedy_required=True,
            force_ks_required=False,
            t2_levers_enabled=["session_warmup_captures"],
            extra={"measure_delta": measure_delta, "method": "osu_bytes"},
        )

    # Minimal fallback schema if T5 utils are unavailable on the node.
    status = "PASS" if equal else "FAIL"
    return {
        "schema_version": 1,
        "pack": "T5",
        "track": "T2",
        "scout": "t2_warmup_isolate_cell",
        "commit": commit,
        "campaign_tip_frozen": "55949274",
        "gates": {
            "greedy_token_match": {
                "status": status,
                "pass": equal,
                "detail": greedy_detail,
                "artifact": "greedy .osu byte compare (do_sample=false)",
            },
            "ks_parity": {
                "status": "SKIP",
                "pass": None,
                "reason": "warmup-hoist isolate: KS N/A; timing_stride held opt-in",
            },
        },
        "required_pass": ["greedy_token_match"],
        "overall": status,
        "t2_levers_enabled": ["session_warmup_captures"],
        "claims": {
            "five_hundred_tps": False,
            "tip_graduate": False,
            "relaxed_turbo_acceptance": False,
        },
        "extra": {"measure_delta": measure_delta, "method": "osu_bytes", "fallback_schema": True},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--python", required=True, type=Path)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--expected-gpu-substr", default="A5000")
    args = p.parse_args()

    run_root: Path = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    base_m = _run_cell(
        repo=args.repo,
        python=args.python,
        audio=args.audio,
        run_root=run_root,
        variant="baseline",
        precision=args.precision,
        seed=args.seed,
        expected_gpu=args.expected_gpu_substr,
        greedy=False,
    )
    warm_m = _run_cell(
        repo=args.repo,
        python=args.python,
        audio=args.audio,
        run_root=run_root,
        variant="t2_warmup",
        precision=args.precision,
        seed=args.seed,
        expected_gpu=args.expected_gpu_substr,
        greedy=False,
    )
    base_g = _run_cell(
        repo=args.repo,
        python=args.python,
        audio=args.audio,
        run_root=run_root,
        variant="baseline",
        precision=args.precision,
        seed=args.seed,
        expected_gpu=args.expected_gpu_substr,
        greedy=True,
    )
    warm_g = _run_cell(
        repo=args.repo,
        python=args.python,
        audio=args.audio,
        run_root=run_root,
        variant="t2_warmup",
        precision=args.precision,
        seed=args.seed,
        expected_gpu=args.expected_gpu_substr,
        greedy=True,
    )

    base_osu = _collect_osu(run_root / "baseline-greedy")
    warm_osu = _collect_osu(run_root / "t2_warmup-greedy")
    equal = (
        len(base_osu) > 0
        and len(base_osu) == len(warm_osu)
        and all(a[1] == b[1] for a, b in zip(base_osu, warm_osu))
    )
    base_ms = base_m.get("ms_per_map_token")
    warm_ms = warm_m.get("ms_per_map_token")
    delta_pct = None
    if isinstance(base_ms, (int, float)) and isinstance(warm_ms, (int, float)) and base_ms > 0:
        delta_pct = 100.0 * (warm_ms - base_ms) / base_ms

    measure_delta = {
        "baseline_ms_per_map_token": base_ms,
        "t2_warmup_ms_per_map_token": warm_ms,
        "delta_pct": delta_pct,
        "baseline_map_tokens": base_m.get("map_tokens"),
        "t2_warmup_map_tokens": warm_m.get("map_tokens"),
        "baseline_exclude_s": base_m.get("e2e_wall_s_exclude_load_jit"),
        "t2_warmup_exclude_s": warm_m.get("e2e_wall_s_exclude_load_jit"),
        "baseline_main_tps": base_m.get("main_gen_model_tps"),
        "t2_warmup_main_tps": warm_m.get("main_gen_model_tps"),
        "baseline_cold_start_s": base_m.get("cold_start_seconds"),
        "t2_warmup_cold_start_s": warm_m.get("cold_start_seconds"),
    }
    greedy_detail = {
        "pass": equal,
        "equal": equal,
        "n_osu_baseline": len(base_osu),
        "n_osu_warmup": len(warm_osu),
        "baseline_bytes": [len(b) for _, b in base_osu],
        "warmup_bytes": [len(b) for _, b in warm_osu],
        "baseline_map_tokens": base_g.get("map_tokens"),
        "warmup_map_tokens": warm_g.get("map_tokens"),
        "files_baseline": [n for n, _ in base_osu],
        "files_warmup": [n for n, _ in warm_osu],
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(args.repo), text=True
    ).strip()
    t5 = _build_t5_report(
        repo=args.repo,
        commit=commit,
        equal=equal,
        greedy_detail=greedy_detail,
        measure_delta=measure_delta,
    )
    promote_clean = bool(equal) and t5.get("overall") == "PASS"

    summary = {
        "status": "OK",
        "track": "T2",
        "isolate": "session_warmup_captures",
        "timing_stride": "held_opt_in",
        "precision": args.precision,
        "seed": args.seed,
        "repo": str(args.repo),
        "repo_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "campaign_tip_frozen": "55949274",
        "elapsed_s": time.perf_counter() - t0,
        "measure": {
            "baseline": base_m,
            "t2_warmup": warm_m,
            "delta": measure_delta,
        },
        "greedy": {
            "baseline": base_g,
            "t2_warmup": warm_g,
            "match": greedy_detail,
        },
        "t5_quality_gates": t5,
        "t5_quality_gates_overall": t5.get("overall"),
        "greedy_pass": equal,
        "promote_clean": promote_clean,
    }
    _write_json(run_root / "summary.json", summary)
    _write_json(run_root / "T5_GATES.json", t5)
    _write_json(run_root / "greedy_match.json", greedy_detail)
    _write_json(run_root / "measure_delta.json", measure_delta)

    print(
        json.dumps(
            {
                "ms_baseline": base_ms,
                "ms_warmup": warm_ms,
                "delta_pct": delta_pct,
                "map_tok_baseline": base_m.get("map_tokens"),
                "map_tok_warmup": warm_m.get("map_tokens"),
                "greedy_pass": equal,
                "t5_overall": t5.get("overall"),
                "promote_clean": promote_clean,
                "artifact": str(run_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not equal:
        raise SystemExit("T5 greedy seal FAILED — warmup-hoist not promote-clean")


if __name__ == "__main__":
    main()
