"""CPU unit tests for T5 quality gates (greedy + KS + scout schema + T2 levers)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
UTILS = REPO / "utils"
sys.path.insert(0, str(UTILS))

from t5_beatmap_metrics import extract_metrics  # noqa: E402
from t5_greedy_token_match import compare_token_ids, load_token_ids  # noqa: E402
from t5_ks_parity import compare_sets  # noqa: E402
from t5_scout_gates import (  # noqa: E402
    build_report,
    gate_entry,
    recommend_t2_promote,
    skip_gate,
    t2_lever_policy,
    validate_scout_summary,
)


OSU_A = """\
osu file format v14

[General]
AudioFilename: a.mp3

[HitObjects]
100,100,0,1,0,0:0:0:0:
120,120,100,1,0,0:0:0:0:
140,140,200,2,0,B|160:160,1,80.0,0:0:0:0:
160,160,500,1,0,0:0:0:0:
"""

OSU_B = """\
osu file format v14

[General]
AudioFilename: b.mp3

[HitObjects]
100,100,0,1,0,0:0:0:0:
120,120,110,1,0,0:0:0:0:
140,140,220,2,0,B|160:160,1,90.0,0:0:0:0:
160,160,510,1,0,0:0:0:0:
"""


def _write_osu(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_extract_metrics_ho_count_and_types(tmp_path: Path):
    p = _write_osu(tmp_path, "a.osu", OSU_A)
    m = extract_metrics(p, density_bin_ms=1000)
    assert m.ho_count == 4
    assert m.type_counts["circle"] == 3
    assert m.type_counts["slider"] == 1


def test_greedy_token_match_pass_and_fail():
    ok = compare_token_ids([1, 2, 3], [1, 2, 3])
    assert ok["pass"] is True
    assert ok["first_mismatch"] is None
    bad = compare_token_ids([1, 2, 3], [1, 9, 3])
    assert bad["pass"] is False
    assert bad["first_mismatch"] == 1


def test_load_token_ids_from_json(tmp_path: Path):
    p = tmp_path / "tok.json"
    p.write_text(json.dumps({"result_tokens": [10, 11, 12]}))
    assert load_token_ids(p) == [10, 11, 12]


def test_ks_identical_sets_pass(tmp_path: Path):
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    for i in range(5):
        _write_osu(base / f"seed_{i}", "map.osu", OSU_A)
        _write_osu(cand / f"seed_{i}", "map.osu", OSU_A)
    row = compare_sets(base, cand, alpha=0.01)
    assert row["pass"] is True


def test_t1_report_allows_na():
    report = build_report(
        track="T1",
        greedy_token_match=gate_entry(status="N/A", reason="structural"),
        ks_parity=gate_entry(status="N/A", reason="structural"),
        scout="unit",
    )
    assert report["overall"] == "PASS"


def test_t3_relaxation_ks_binding_greedy_optional():
    # Under T3 exactness relaxation, KS is required; greedy SKIP stays reportable.
    report = build_report(
        track="T3",
        greedy_token_match=skip_gate("deferred audit"),
        ks_parity=gate_entry(status="PASS"),
        scout="unit",
    )
    assert report["gates"]["greedy_token_match"]["status"] == "SKIP"
    assert report["required_pass"] == ["ks_parity"]
    assert report["overall"] == "PASS"

    fail_ks = build_report(
        track="T3",
        greedy_token_match=gate_entry(status="FAIL", reason="inductor fp16 drift"),
        ks_parity=gate_entry(status="FAIL", reason="ks miss"),
        scout="unit",
    )
    assert fail_ks["overall"] == "FAIL"
    assert "ks_parity:FAIL" in fail_ks["fail_list"]
    # Greedy FAIL is documented drift — not in fail_list when KS is binding.
    assert not any(x.startswith("greedy_token_match:") for x in fail_ks["fail_list"])


def test_t2_timing_stride_forces_ks_and_allows_greedy_fail():
    pol = t2_lever_policy("timing_stride")
    assert pol["expect_greedy_match"] is False
    assert pol["evidence"]["baseline_map_tok"] == 7526
    assert pol["evidence"]["candidate_map_tok"] == 8151

    # Greedy FAIL + KS AWAITING → overall FAIL (KS required, not yet sealed)
    awaiting = build_report(
        track="T2",
        greedy_token_match=gate_entry(status="FAIL", reason="map_tok drift"),
        ks_parity=gate_entry(status="AWAITING", reason="need sampled pack"),
        t2_levers_enabled=["timing_stride"],
    )
    assert awaiting["overall"] == "FAIL"
    assert "ks_parity" in awaiting["required_pass"]

    # Greedy FAIL + KS PASS → overall PASS (documented drift path for stride)
    sealed = build_report(
        track="T2",
        greedy_token_match=gate_entry(status="FAIL", reason="map_tok drift 7526→8151"),
        ks_parity=gate_entry(status="PASS"),
        t2_levers_enabled=["timing_stride", "session_warmup_captures"],
    )
    assert sealed["overall"] == "PASS"
    advice = sealed["t2_promote_advice"]
    assert "timing_stride" in advice["default_on_ok"]
    # Warmup still held here: bundled greedy FAIL cannot certify token-preserving hoist.
    assert any(h["lever"] == "session_warmup_captures" for h in advice["hold_opt_in"])


def test_t2_warmup_only_clean_when_greedy_pass():
    advice = recommend_t2_promote(
        levers_enabled=["session_warmup_captures"],
        greedy_pass=True,
        ks_pass=None,
    )
    assert advice["default_on_ok"] == ["session_warmup_captures"]
    assert advice["preferred_clean_promote"] == ["session_warmup_captures"]


def test_t2_warmup_clean_promote_with_stride_held():
    advice = recommend_t2_promote(
        levers_enabled=["session_warmup_captures", "timing_stride"],
        greedy_pass=True,
        ks_pass=None,
    )
    assert advice["default_on_ok"] == ["session_warmup_captures"]
    assert any(h["lever"] == "timing_stride" for h in advice["hold_opt_in"])
    assert advice["preferred_clean_promote"] == ["session_warmup_captures"]


def test_validate_requires_block():
    assert validate_scout_summary({}, track="T1")["ok"] is False
    report = build_report(
        track="T1",
        greedy_token_match=gate_entry(status="N/A", reason="x"),
        ks_parity=gate_entry(status="N/A", reason="y"),
    )
    assert validate_scout_summary({"t5_quality_gates": report}, track="T1")["ok"] is True


def test_orchestrator_t1_dry_run(tmp_path: Path):
    pack = tmp_path / "pack"
    proc = subprocess.run(
        [
            sys.executable,
            str(UTILS / "t5_quality_gates.py"),
            "--track",
            "T1",
            "--pack-root",
            str(pack),
            "--skip-pytest",
            "--require-pass",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((pack / "T5_GATES.json").read_text())
    assert report["overall"] == "PASS"
    assert report["claims"]["five_hundred_tps"] is False
    summary = json.loads((pack / "summary.json").read_text())
    assert "t5_quality_gates" in summary
