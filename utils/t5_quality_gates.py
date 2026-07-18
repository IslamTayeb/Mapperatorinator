#!/usr/bin/env python3
"""T5 quality-gates orchestrator (CPU-first; GPU canary opt-in).

Phases:
  1) CPU unit tests (metrics / greedy / scout schema)
  2) Greedy token-match (from provided dumps, or plan-only)
  3) KS parity (from .osu pack, or awaiting stub)
  4) Scout report emission (T1/T2/T3 schema)
  5) Optional GPU greedy canary seal (--execute-canary)

No 500 claim. Tip 55949274 frozen. T4 PARKED. §34 standing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
UTILS = REPO / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from t5_scout_gates import (  # noqa: E402
    CAMPAIGN_TIP_FROZEN,
    build_report,
    from_greedy_compare,
    from_ks_parity,
    merge_into_summary,
    na_gate,
    skip_gate,
    write_report,
)


def _run(cmd: list[str], *, dry: bool = False) -> int:
    print("+", " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def _git_meta() -> dict[str, str]:
    def g(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
        except Exception:
            return ""

    return {"commit": g("rev-parse", "HEAD"), "branch": g("branch", "--show-current")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "t5_quality_gates.yaml",
    )
    ap.add_argument("--pack-root", type=Path, default=None)
    ap.add_argument(
        "--track",
        choices=["T1", "T2", "T3"],
        default="T1",
        help="Scout track for gate rules (default T1 for CPU dry-run; T3 scouts must pass --track T3).",
    )
    ap.add_argument("--scout", default="t5-quality-gates")
    ap.add_argument("--baseline-tokens", type=Path, default=None)
    ap.add_argument("--candidate-tokens", type=Path, default=None)
    ap.add_argument("--baseline-engine", default=None)
    ap.add_argument("--candidate-engine", default=None)
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--skip-ks", action="store_true")
    ap.add_argument(
        "--t2-lever",
        action="append",
        default=[],
        help=(
            "T2 lever under test (repeatable): session_warmup_captures | "
            "encoder_precompute_dedupe | timing_stride. timing_stride forces KS."
        ),
    )
    ap.add_argument(
        "--execute-canary",
        action="store_true",
        help="Opt-in GPU canary via configured command (sealing only).",
    )
    ap.add_argument("--require-pass", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pack_root = (args.pack_root or (REPO / "runs" / f"t5-quality-gates-{ts}")).resolve()
    pack_root.mkdir(parents=True, exist_ok=True)

    py = os.environ.get("PYTHON", sys.executable)
    meta = _git_meta()
    phases: dict[str, Any] = {}

    # 1) CPU unit tests
    if not args.skip_pytest:
        cmd = [
            py,
            "-m",
            "pytest",
            "-q",
            str(REPO / "tests" / "test_t5_quality_gates.py"),
        ]
        rc = _run(cmd)
        phases["pytest"] = {"rc": rc, "pass": rc == 0, "automated": True}
    else:
        phases["pytest"] = {"skipped": True}

    # 2) Greedy token-match
    greedy_out = pack_root / "greedy_token_match.json"
    greedy_gate: dict[str, Any]
    if args.skip_greedy:
        greedy_gate = skip_gate("skipped by --skip-greedy")
        phases["greedy_token_match"] = {"skipped": True}
    elif args.baseline_tokens and args.candidate_tokens:
        cmd = [
            py,
            str(UTILS / "t5_greedy_token_match.py"),
            "--baseline",
            str(args.baseline_tokens),
            "--candidate",
            str(args.candidate_tokens),
            "--label",
            f"{args.track}-greedy",
            "--out",
            str(greedy_out),
        ]
        if args.require_pass and args.track == "T3":
            cmd.append("--require-pass")
        rc = _run(cmd)
        detail = json.loads(greedy_out.read_text()) if greedy_out.exists() else {"pass": False}
        greedy_gate = from_greedy_compare(detail, artifact=str(greedy_out))
        phases["greedy_token_match"] = {
            "rc": rc,
            "pass": detail.get("pass"),
            "out": str(greedy_out),
            "automated": True,
        }
    else:
        # Plan-only: document expected comparison without failing CPU dry-run.
        plan = {
            "status": "plan_only",
            "baseline_hint": cfg.get("greedy", {}).get("baseline_hint"),
            "candidate_hint": cfg.get("greedy", {}).get("candidate_hint"),
            "note": (
                "Provide --baseline-tokens and --candidate-tokens dumps to execute. "
                "T3 must seal greedy vs uncompiled before promote."
            ),
        }
        greedy_out.write_text(json.dumps(plan, indent=2) + "\n")
        if args.track == "T3":
            # Dry-run: mark SKIP so harness can still complete CPU path;
            # --require-pass will fail T3 overall (correct — not sealed).
            greedy_gate = skip_gate(
                "T3 greedy not sealed — pass --baseline-tokens/--candidate-tokens "
                "or --execute-canary"
            )
        elif args.track == "T1":
            greedy_gate = na_gate("T1 structural scout — no decode stream in default harness")
        else:
            greedy_gate = skip_gate("no token dumps provided")
        phases["greedy_token_match"] = {
            "rc": 0,
            "pass": None,
            "out": str(greedy_out),
            "automated": True,
            "stubbed": True,
        }

    # 3) KS parity
    ks_out = pack_root / "ks_parity.json"
    baseline_engine = args.baseline_engine or cfg.get("engines", {}).get("baseline", "optimized")
    candidate_engine = args.candidate_engine or cfg.get("engines", {}).get("candidate", "candidate")
    if args.skip_ks:
        ks_gate = skip_gate("skipped by --skip-ks")
        phases["ks_parity"] = {"skipped": True}
    else:
        osu_root = pack_root / "osu"
        has_osu = osu_root.exists() and any(osu_root.rglob("*.osu"))
        if has_osu:
            cmd = [
                py,
                str(UTILS / "t5_ks_parity.py"),
                "--pack-root",
                str(pack_root),
                "--baseline-engine",
                str(baseline_engine),
                "--candidate-engine",
                str(candidate_engine),
                "--alpha",
                str(cfg.get("ks", {}).get("alpha", 0.01)),
                "--density-bin-ms",
                str(cfg.get("ks", {}).get("density_bin_ms", 1000)),
                "--out",
                str(ks_out),
            ]
            rc = _run(cmd)
            detail = json.loads(ks_out.read_text()) if ks_out.exists() else {"pass": False}
            ks_gate = from_ks_parity(detail, artifact=str(ks_out))
            phases["ks_parity"] = {
                "rc": rc,
                "pass": detail.get("pass"),
                "out": str(ks_out),
                "automated": True,
            }
        else:
            stub = {
                "pass": None,
                "status": "AWAITING",
                "note": (
                    "KS harness ready; no .osu under pack_root/osu. "
                    "Layout: osu/{baseline,candidate}/{song_id}/seed_*/**/*.osu"
                ),
                "metrics": cfg.get("ks", {}).get("metrics"),
                "alpha": cfg.get("ks", {}).get("alpha", 0.01),
            }
            ks_out.write_text(json.dumps(stub, indent=2) + "\n")
            if args.track == "T1":
                ks_gate = na_gate("T1 structural scout — sampled KS not applicable by default")
            else:
                ks_gate = from_ks_parity(stub, artifact=str(ks_out))
            phases["ks_parity"] = {
                "rc": 0,
                "pass": None,
                "out": str(ks_out),
                "automated": True,
                "stubbed": True,
            }

    # 4) Optional GPU canary
    canary_cfg = cfg.get("canary") or {}
    if args.execute_canary:
        canary_cmd = canary_cfg.get("command")
        if not canary_cmd:
            phases["gpu_canary"] = {
                "rc": 1,
                "pass": False,
                "error": "no canary.command in config",
            }
        else:
            out_json = pack_root / "canary" / "canary_result.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                *(canary_cmd if isinstance(canary_cmd, list) else [str(x) for x in [canary_cmd]]),
            ]
            # Prefer list form from yaml.
            if isinstance(canary_cmd, list):
                cmd = [str(x) for x in canary_cmd]
                # Expand placeholders
                cmd = [
                    c.replace("{pack_root}", str(pack_root))
                    .replace("{out}", str(out_json))
                    .replace("{repo}", str(REPO))
                    for c in cmd
                ]
            else:
                cmd = [
                    py,
                    "-c",
                    f"raise SystemExit('canary.command must be a YAML list, got: {canary_cmd!r}')",
                ]
            rc = _run(cmd)
            phases["gpu_canary"] = {
                "rc": rc,
                "executed": True,
                "out": str(out_json),
            }
            if out_json.exists() and args.baseline_tokens is None:
                # If canary wrote comparable dumps, prefer them when present.
                payload = json.loads(out_json.read_text())
                if "pass" in payload:
                    greedy_gate = from_greedy_compare(payload, artifact=str(out_json))
                    phases["greedy_token_match"] = {
                        "rc": rc,
                        "pass": payload.get("pass"),
                        "out": str(out_json),
                        "from": "gpu_canary",
                    }
    else:
        phases["gpu_canary"] = {"skipped": True, "note": "pass --execute-canary to seal on GPU"}

    # 5) Scout report
    report = build_report(
        track=args.track,  # type: ignore[arg-type]
        greedy_token_match=greedy_gate,
        ks_parity=ks_gate,
        scout=args.scout,
        commit=meta.get("commit"),
        t2_levers_enabled=list(args.t2_lever or []),
        extra={
            "pack_root": str(pack_root),
            "phases": phases,
            "git": meta,
            "campaign_tip_frozen": CAMPAIGN_TIP_FROZEN,
            "docs": "docs/inference_evidence_packs.md",
            "handoff": "notes/500tps-t5-quality-gates-handoff.md",
            "t2_partial_seal": {
                "jobs": {"baseline": "50194534", "t2": "50194535"},
                "map_tok_drift": "7526→8151 under timing_stride",
                "clean_promote": "session_warmup_captures",
                "hold_opt_in": "timing_stride until greedy documented + KS PASS",
            },
        },
    )
    gates_path = write_report(pack_root / "T5_GATES.json", report)
    summary_path = pack_root / "summary.json"
    merge_into_summary(summary_path, report)

    pack_report = {
        "section": "T5",
        "created_utc": ts,
        "pack_root": str(pack_root),
        "track": args.track,
        "git": meta,
        "phases": phases,
        "t5_quality_gates": report,
        "claims": report["claims"],
    }
    (pack_root / "T5_QUALITY_GATES_PACK.json").write_text(json.dumps(pack_report, indent=2) + "\n")
    (pack_root / "T5_QUALITY_GATES_PACK.md").write_text(
        "\n".join(
            [
                "# T5 quality gates pack",
                "",
                f"- Created: `{ts}`",
                f"- Track: `{args.track}`",
                f"- Branch/commit: `{meta.get('branch')}` / `{meta.get('commit')}`",
                f"- Tip frozen: `{CAMPAIGN_TIP_FROZEN}`",
                f"- Overall: **{report['overall']}**",
                f"- Gates file: `{gates_path}`",
                "",
                "## Phases",
                "",
                "```json",
                json.dumps(phases, indent=2),
                "```",
                "",
                "**No 500 claim. No merge. No PR #120 push. T4 PARKED. §34 standing.**",
                "",
            ]
        )
    )
    print(f"wrote {gates_path} overall={report['overall']}", flush=True)

    if args.require_pass:
        bad = []
        if phases.get("pytest", {}).get("pass") is False:
            bad.append("pytest")
        if report["overall"] != "PASS":
            bad.append(f"t5_overall:{report['overall']}")
        if bad:
            raise SystemExit(f"T5 require-pass failed: {bad}")


if __name__ == "__main__":
    main()
