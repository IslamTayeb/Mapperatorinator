#!/usr/bin/env python3
"""T5 scout gate reporter — every T1/T2/T3 scout must emit pass/fail.

Schema (schema_version=1) is written into scout summary.json under
`t5_quality_gates` and/or as a sibling `T5_GATES.json`.

Track rules (§34 standing; no relaxed turbo acceptance):
  T1  — must REPORT both gates (SKIP/N/A allowed with reason)
  T2  — greedy_token_match REQUIRED when decode path exercised;
        ks_parity REQUIRED when sampled path claimed
  T3  — exactness RELAXED (2026-07-18): ks_parity REQUIRED;
        greedy_token_match reported (FAIL = documented Inductor fp16 drift OK);
        coherent / mostly-good maps — NOT bit-identical .osu

T4 turbo is PARKED — do not invent gates that fold drift into turbo.
T3 relaxation does NOT fold into §34 turbo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Status = Literal["PASS", "FAIL", "SKIP", "N/A", "AWAITING"]
Track = Literal["T1", "T2", "T3"]

SCHEMA_VERSION = 1
CAMPAIGN_TIP_FROZEN = "55949274"
SECTION34 = (
    "Never fold relaxed acceptance into turbo. "
    "Distribution claims use TIER1/TIER3 packs as defined in "
    "docs/inference_evidence_packs.md — no invented thresholds."
)

# What each track must present as PASS (or SKIP with explicit reason only where allowed).
TRACK_RULES: dict[str, dict[str, Any]] = {
    "T1": {
        "must_report": ("greedy_token_match", "ks_parity"),
        "required_pass": (),  # structural rails may SKIP both
        "notes": "Report-only; SKIP/N/A OK with reason for unit-test scouts.",
    },
    "T2": {
        "must_report": ("greedy_token_match", "ks_parity"),
        # Promote requires greedy PASS once decode runs; use --require-greedy on seal.
        "required_pass": (),
        "notes": (
            "Must report both gates. Promote seal: greedy vs tiger baseline PASS; "
            "KS PASS if sampled. Use --require-greedy / --require-ks when sealing. "
            "T2 levers: warmup-hoist/dedupe expect greedy PASS; timing-stride is "
            "OPT-IN until greedy+KS seal (known map_tok drift 7526→8151 @ 50194535)."
        ),
    },
    "T3": {
        "must_report": ("greedy_token_match", "ks_parity"),
        # Exactness relaxation (2026-07-18): KS / coherent is the bar — not greedy.
        "required_pass": ("ks_parity",),
        "notes": (
            "T3 full-step compile-then-capture: promote on coherent maps + KS PASS "
            "(or soft-fail with numbers). Greedy FAIL is documented Inductor fp16 "
            "near-tie drift — report it, do not require PASS. Scope: T3 only."
        ),
    },
}

# T2 certain-full-map levers (jobs/50194534 vs 50194535). Timing-stride changes
# timing→map conditioning and is NOT token-preserving; keep opt-in until T5 seals it.
T2_LEVERS: dict[str, dict[str, Any]] = {
    "session_warmup_captures": {
        "expect_greedy_match": True,
        "default_on_ok_without_ks": True,
        "promote": "clean",
        "note": "Warmup/capture hoist only — clean promote path when greedy PASS.",
    },
    "encoder_precompute_dedupe": {
        "expect_greedy_match": True,
        "default_on_ok_without_ks": True,
        "promote": "clean_if_hit",
        "note": "Same-model same-stride reuse; dual-model W1 path may miss.",
    },
    "timing_stride": {
        "expect_greedy_match": False,  # may intentionally diverge
        "default_on_ok_without_ks": False,
        "promote": "opt_in_until_t5",
        "flags": ("timing_lookback", "timing_lookahead"),
        "evidence": {
            "baseline_job": "50194534",
            "candidate_job": "50194535",
            "baseline_map_tok": 7526,
            "candidate_map_tok": 8151,
            "delta_map_tok": 625,
        },
        "note": (
            "Timing-stride tune drifts map tokens (7526→8151). OPT-IN only until "
            "greedy token-match (or documented intentional mismatch) + KS/metric "
            "parity PASS under §34. Do not default-on without T5 seal."
        ),
    },
}


def t2_lever_policy(lever: str) -> dict[str, Any]:
    if lever not in T2_LEVERS:
        raise KeyError(f"unknown T2 lever: {lever}; known={sorted(T2_LEVERS)}")
    return dict(T2_LEVERS[lever])


def recommend_t2_promote(*, levers_enabled: list[str], greedy_pass: bool | None, ks_pass: bool | None) -> dict[str, Any]:
    """Advise which T2 levers may default-on given gate outcomes."""
    enabled = list(levers_enabled)
    clean = []
    hold = []
    for lever in enabled:
        pol = t2_lever_policy(lever)
        if pol["promote"] == "opt_in_until_t5":
            # timing-stride: need KS PASS to default-on; greedy may FAIL by design
            if ks_pass is True:
                clean.append(lever)
            else:
                hold.append(
                    {
                        "lever": lever,
                        "reason": pol["note"],
                        "need": "ks_parity PASS (greedy may FAIL; map_tok drift known)",
                    }
                )
        else:
            # Token-preserving levers need an isolated greedy PASS (do not
            # certify warmup from a timing_stride bundle that already drifted).
            if greedy_pass is True:
                clean.append(lever)
            else:
                hold.append(
                    {
                        "lever": lever,
                        "reason": (
                            "expected greedy match; isolate warmup-only / dedupe-only "
                            "cell when timing_stride is also on"
                            if greedy_pass is False
                            else pol["note"]
                        ),
                        "need": "greedy_token_match PASS on lever-isolated cell",
                    }
                )
    preferred = [x for x in clean if x == "session_warmup_captures"] or [
        x for x in clean if x != "timing_stride"
    ][:1]
    return {
        "levers_enabled": enabled,
        "default_on_ok": clean,
        "hold_opt_in": hold,
        "preferred_clean_promote": preferred,
        "note": (
            "Warmup-hoist alone is the clean promote path when timing-stride is held. "
            "Do not fold timing-stride into default without T5 KS seal. "
            "Certify warmup/dedupe on lever-isolated greedy PASS cells."
        ),
    }


def gate_entry(
    *,
    status: Status,
    detail: dict[str, Any] | None = None,
    reason: str | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    passed: bool | None
    if status == "PASS":
        passed = True
    elif status == "FAIL":
        passed = False
    else:
        passed = None
    row: dict[str, Any] = {
        "status": status,
        "pass": passed,
    }
    if reason:
        row["reason"] = reason
    if artifact:
        row["artifact"] = artifact
    if detail:
        row["detail"] = detail
    return row


def from_greedy_compare(payload: dict[str, Any], *, artifact: str | None = None) -> dict[str, Any]:
    status: Status = "PASS" if payload.get("pass") else "FAIL"
    return gate_entry(status=status, detail=payload, artifact=artifact)


def from_ks_parity(payload: dict[str, Any], *, artifact: str | None = None) -> dict[str, Any]:
    if payload.get("status") == "AWAITING" or payload.get("pass") is None:
        return gate_entry(
            status="AWAITING",
            detail=payload,
            artifact=artifact,
            reason=payload.get("note") or payload.get("reason") or "awaiting .osu artifacts",
        )
    status: Status = "PASS" if payload.get("pass") else "FAIL"
    return gate_entry(status=status, detail=payload, artifact=artifact)


def skip_gate(reason: str) -> dict[str, Any]:
    return gate_entry(status="SKIP", reason=reason)


def na_gate(reason: str) -> dict[str, Any]:
    return gate_entry(status="N/A", reason=reason)


def build_report(
    *,
    track: Track,
    greedy_token_match: dict[str, Any],
    ks_parity: dict[str, Any],
    scout: str | None = None,
    commit: str | None = None,
    extra: dict[str, Any] | None = None,
    force_greedy_required: bool | None = None,
    force_ks_required: bool | None = None,
    t2_levers_enabled: list[str] | None = None,
) -> dict[str, Any]:
    rules = TRACK_RULES[track]
    gates = {
        "greedy_token_match": greedy_token_match,
        "ks_parity": ks_parity,
    }
    for name in rules["must_report"]:
        if name not in gates or not isinstance(gates[name], dict) or "status" not in gates[name]:
            raise ValueError(f"{track} scout missing gate report: {name}")

    required = list(rules["required_pass"])
    if force_greedy_required is True and "greedy_token_match" not in required:
        required.append("greedy_token_match")
    if force_greedy_required is False:
        required = [g for g in required if g != "greedy_token_match"]
    if force_ks_required is True and "ks_parity" not in required:
        required.append("ks_parity")
    if force_ks_required is False:
        required = [g for g in required if g != "ks_parity"]

    # Timing-stride opt-in on T2 forces KS into required_pass when that lever is on.
    levers = list(t2_levers_enabled or [])
    if track == "T2" and "timing_stride" in levers and "ks_parity" not in required:
        required.append("ks_parity")

    # T3: greedy may be SKIP/N/A/AWAITING only if explicitly deferred; prefer
    # PASS/FAIL report. Do NOT auto-fail overall on greedy under relaxation.
    if track == "T3" and gates["greedy_token_match"]["status"] in {"SKIP", "N/A", "AWAITING"}:
        # Keep status; note that a real PASS/FAIL report is preferred for audit.
        gates["greedy_token_match"] = gate_entry(
            status=gates["greedy_token_match"]["status"],
            reason=(
                gates["greedy_token_match"].get("reason")
                or "T3 relaxation: greedy optional; KS/coherent is binding"
            ),
            detail=greedy_token_match,
            artifact=greedy_token_match.get("artifact")
            if isinstance(greedy_token_match, dict)
            else None,
        )

    missing_or_fail = []
    for name in required:
        st = gates[name]["status"]
        if st != "PASS":
            missing_or_fail.append(f"{name}:{st}")

    # Incomplete if a must-report gate is somehow empty (defensive).
    incomplete = []
    for name in rules["must_report"]:
        if gates[name].get("status") is None:
            incomplete.append(name)

    if incomplete:
        overall = "INCOMPLETE"
    elif missing_or_fail:
        overall = "FAIL"
    else:
        # Soft fails: reported FAIL on optional gates still fail overall —
        # except T2 timing_stride and T3 (relaxed), where greedy FAIL is
        # expected/documentable when KS is the binding gate.
        soft_fail = []
        for n in rules["must_report"]:
            if gates[n]["status"] != "FAIL":
                continue
            if (
                track == "T2"
                and n == "greedy_token_match"
                and "timing_stride" in levers
                and gates["ks_parity"]["status"] in {"PASS", "AWAITING", "SKIP"}
            ):
                # Documented drift path: greedy may FAIL; KS must still be reported.
                continue
            if track == "T3" and n == "greedy_token_match":
                # Documented Inductor fp16 near-tie drift under T3 relaxation.
                continue
            soft_fail.append(f"{n}:{gates[n]['status']}")
        overall = "FAIL" if soft_fail else "PASS"
        missing_or_fail = soft_fail

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pack": "T5",
        "track": track,
        "scout": scout,
        "commit": commit,
        "campaign_tip_frozen": CAMPAIGN_TIP_FROZEN,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "section34_standing": SECTION34,
        "track_notes": rules["notes"],
        "gates": gates,
        "required_pass": required,
        "overall": overall,
        "fail_list": missing_or_fail,
        "claims": {
            "five_hundred_tps": False,
            "tip_graduate": False,
            "relaxed_turbo_acceptance": False,
        },
    }
    if levers:
        report["t2_levers_enabled"] = levers
        report["t2_lever_policies"] = {k: t2_lever_policy(k) for k in levers}
        report["t2_promote_advice"] = recommend_t2_promote(
            levers_enabled=levers,
            greedy_pass=gates["greedy_token_match"].get("pass"),
            ks_pass=gates["ks_parity"].get("pass"),
        )
    if extra:
        report["extra"] = extra
    return report


def write_report(path: Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def merge_into_summary(summary_path: Path, report: dict[str, Any]) -> None:
    summary_path = Path(summary_path)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["t5_quality_gates"] = report
    # Mirror overall for quick grepping by CI / collate scripts.
    summary["t5_quality_gates_overall"] = report["overall"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def validate_scout_summary(summary: dict[str, Any], *, track: Track | None = None) -> dict[str, Any]:
    block = summary.get("t5_quality_gates")
    if not isinstance(block, dict):
        return {
            "ok": False,
            "error": "missing t5_quality_gates block — scout did not report T5 gates",
        }
    tr = track or block.get("track")
    if tr not in TRACK_RULES:
        return {"ok": False, "error": f"unknown track: {tr}"}
    for name in TRACK_RULES[tr]["must_report"]:
        gate = (block.get("gates") or {}).get(name)
        if not isinstance(gate, dict) or "status" not in gate:
            return {"ok": False, "error": f"missing gate status: {name}"}
    overall = block.get("overall")
    ok = overall == "PASS"
    return {"ok": ok, "overall": overall, "fail_list": block.get("fail_list") or []}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build a T5_GATES.json from gate statuses")
    p_build.add_argument("--track", choices=["T1", "T2", "T3"], required=True)
    p_build.add_argument("--scout", default=None)
    p_build.add_argument("--commit", default=None)
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--summary", type=Path, default=None, help="Also merge into summary.json")
    p_build.add_argument("--greedy-status", choices=["PASS", "FAIL", "SKIP", "N/A", "AWAITING"], required=True)
    p_build.add_argument("--greedy-reason", default=None)
    p_build.add_argument("--greedy-json", type=Path, default=None, help="Detail JSON from t5_greedy_token_match")
    p_build.add_argument("--ks-status", choices=["PASS", "FAIL", "SKIP", "N/A", "AWAITING"], required=True)
    p_build.add_argument("--ks-reason", default=None)
    p_build.add_argument("--ks-json", type=Path, default=None)
    p_build.add_argument("--require-greedy", action="store_true")
    p_build.add_argument("--require-ks", action="store_true")
    p_build.add_argument("--require-pass", action="store_true")
    p_build.add_argument(
        "--t2-lever",
        action="append",
        default=[],
        choices=sorted(T2_LEVERS),
        help="T2 lever under test (repeatable). timing_stride forces KS into required_pass.",
    )

    p_val = sub.add_parser("validate", help="Validate a scout summary has T5 gates")
    p_val.add_argument("--summary", type=Path, required=True)
    p_val.add_argument("--track", choices=["T1", "T2", "T3"], default=None)
    p_val.add_argument("--require-pass", action="store_true")

    args = ap.parse_args()

    if args.cmd == "validate":
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        result = validate_scout_summary(summary, track=args.track)
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            raise SystemExit(2 if args.require_pass else 1)
        return

    greedy_detail = None
    if args.greedy_json:
        greedy_detail = json.loads(args.greedy_json.read_text(encoding="utf-8"))
        greedy = from_greedy_compare(greedy_detail, artifact=str(args.greedy_json))
    else:
        greedy = gate_entry(status=args.greedy_status, reason=args.greedy_reason)

    ks_detail = None
    if args.ks_json:
        ks_detail = json.loads(args.ks_json.read_text(encoding="utf-8"))
        ks = from_ks_parity(ks_detail, artifact=str(args.ks_json))
    else:
        ks = gate_entry(status=args.ks_status, reason=args.ks_reason)

    report = build_report(
        track=args.track,
        greedy_token_match=greedy,
        ks_parity=ks,
        scout=args.scout,
        commit=args.commit,
        force_greedy_required=True if args.require_greedy else None,
        force_ks_required=True if args.require_ks else None,
        t2_levers_enabled=list(args.t2_lever or []),
    )
    write_report(args.out, report)
    if args.summary:
        merge_into_summary(args.summary, report)
    print(json.dumps({"overall": report["overall"], "out": str(args.out)}, indent=2))
    if args.require_pass and report["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
