from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from utils.audit_candidate_composition import (
    BASE_COMMIT,
    FRONTIER_CANDIDATES,
    audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frontier_tips_resolve_from_current_main() -> None:
    report = audit(REPO_ROOT)

    assert report["base_commit"] == BASE_COMMIT
    assert set(report["candidate_reports"]) == {
        candidate.name for candidate in FRONTIER_CANDIDATES
    }
    assert all(
        candidate["surfaces"]["runtime"]
        for candidate in report["candidate_reports"].values()
    )
    assert all(report["guardrails"].values())


def test_tip_merge_conflicts_and_semantic_only_conflicts_are_distinguished() -> None:
    pairs = audit(REPO_ROOT)["pair_reports"]

    assert not pairs["conditional_temperature+split_kv"]["tip_merge"]["clean"]
    assert "osuT5/osuT5/inference/optimized/single/engine.py" in pairs[
        "conditional_temperature+split_kv"
    ]["tip_merge"]["conflict_paths"]
    assert not pairs["conditional_temperature+shared_rope"]["tip_merge"]["clean"]
    assert not pairs["split_kv+shared_rope"]["tip_merge"]["clean"]

    split_aot = pairs["split_kv+lazy_aot_startup"]
    assert split_aot["tip_merge"]["clean"]
    assert "rebuild_manifest_because_q1_source_hash_and_exports_change" in split_aot[
        "semantic_constraints"
    ]


def test_device_state_is_textually_independent_but_not_semantically_unchecked() -> None:
    pairs = audit(REPO_ROOT)["pair_reports"]

    for partner in (
        "conditional_temperature",
        "split_kv",
        "shared_rope",
        "lazy_aot_startup",
    ):
        key = (
            f"{partner}+device_sequence_state"
            if f"{partner}+device_sequence_state" in pairs
            else f"device_sequence_state+{partner}"
        )
        assert pairs[key]["tip_merge"]["clean"]
        assert not pairs[key]["runtime_overlap"]

    conditional_device = pairs[
        "conditional_temperature+device_sequence_state"
    ]
    assert conditional_device["semantic_constraints"]


def test_integration_order_keeps_aot_build_last() -> None:
    plan = audit(REPO_ROOT)["integration_plan"]
    names = [unit["name"] for unit in plan]

    assert names[-1] == "aot_native_loader"
    assert names.index("lazy_import_runtime") < names.index("split_kv")
    assert names.index("split_kv") < names.index("shared_rope")
    assert plan[-1]["strategy"] == (
        "apply_loader_last_then_rebuild_and_revalidate_manifest"
    )


def test_cli_emits_nonempty_json_without_mutating_worktree(tmp_path) -> None:
    before = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    output = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "utils" / "audit_candidate_composition.py"),
            "--repo",
            str(REPO_ROOT),
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.stat().st_size > 0
    after = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after


def test_audit_branch_contains_no_candidate_runtime_changes() -> None:
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--name-only",
            BASE_COMMIT,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    changed.extend(
        subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-files",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )

    assert changed
    assert all(path.startswith(("tests/", "utils/")) for path in changed)
