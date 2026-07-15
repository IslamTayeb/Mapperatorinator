from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import subprocess
from typing import Any


BASE_COMMIT = "367a9563a4d03dc15215c740fbd56cd9290d6d8b"


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    commit: str
    evidence_state: str


FRONTIER_CANDIDATES = (
    Candidate(
        "conditional_temperature",
        "f6c74b844b43ddfcfa0b71a9b99042dab7bdf6ed",
        "requires_independent_gpu_decision",
    ),
    Candidate(
        "split_kv",
        "13d1882cc8cf9b56de07406a1a49522a1cbedf76",
        "requires_independent_gpu_decision",
    ),
    Candidate(
        "device_sequence_state",
        "1d5d4c33cdb2ea45887ffa113e7aa8f7a997c74b",
        "requires_independent_gpu_decision",
    ),
    Candidate(
        "shared_rope",
        "c5ca01223993a43f79ca01ed5a62657d0205b699",
        "requires_independent_gpu_decision",
    ),
    Candidate(
        "lazy_aot_startup",
        "55b655cdb9128abb8d5229c53efe540dab51e7bc",
        "cpu_reciprocal_pass_full_song_pending",
    ),
)


INTEGRATION_UNITS = (
    {
        "name": "canonical_exactness_and_profile_substrate",
        "strategy": "select_one_copy_only",
        "commits": (
            "65cc855",
            "044e49e",
            "b56dcb4",
        ),
        "requires": (),
        "after_if_selected": (),
    },
    {
        "name": "lazy_import_runtime",
        "strategy": "cherry_pick_in_order",
        "commits": (
            "5c3e4d4",
            "e532efe",
            "d9ddd95",
            "7b6e70e",
            "cf8208a",
            "3223c05",
        ),
        "requires": ("canonical_exactness_and_profile_substrate",),
        "after_if_selected": (),
    },
    {
        "name": "device_sequence_state",
        "strategy": "cherry_pick_only_after_independent_promotion",
        "commits": ("8be8d3b",),
        "requires": ("canonical_exactness_and_profile_substrate",),
        "after_if_selected": ("lazy_import_runtime",),
    },
    {
        "name": "conditional_temperature",
        "strategy": "manually_compose_final_runtime_hunks_after_independent_promotion",
        "commits": (
            "3735292",
            "71a0c9d",
            "e4ce12a",
            "8cde938",
            "f8110d0",
            "f6c74b8",
        ),
        "requires": ("canonical_exactness_and_profile_substrate",),
        "after_if_selected": ("lazy_import_runtime", "device_sequence_state"),
    },
    {
        "name": "split_kv",
        "strategy": "manually_compose_final_kernel_and_dispatch_after_independent_promotion",
        "commits": ("1a350e2", "a207fb8", "063c6df"),
        "requires": ("canonical_exactness_and_profile_substrate",),
        "after_if_selected": (
            "lazy_import_runtime",
            "device_sequence_state",
            "conditional_temperature",
        ),
    },
    {
        "name": "shared_rope",
        "strategy": "transplant_final_opt_in_module_and_tests_after_independent_promotion",
        "commits": ("c5ca012",),
        "requires": ("canonical_exactness_and_profile_substrate",),
        "after_if_selected": (
            "lazy_import_runtime",
            "device_sequence_state",
            "conditional_temperature",
            "split_kv",
        ),
    },
    {
        "name": "aot_native_loader",
        "strategy": "apply_loader_last_then_rebuild_and_revalidate_manifest",
        "commits": ("8c6d516",),
        "requires": ("lazy_import_runtime",),
        "after_if_selected": (
            "device_sequence_state",
            "conditional_temperature",
            "split_kv",
            "shared_rope",
        ),
    },
)


SEMANTIC_CONSTRAINTS = {
    frozenset(("conditional_temperature", "device_sequence_state")): (
        "verify_preallocated_input_ids_view_preserves_temperature_token_history",
        "verify_each_temporary_context_restores_its_binding",
    ),
    frozenset(("conditional_temperature", "split_kv")): (
        "manually_compose_engine_policy_and_dispatch_metadata",
        "retain_only_one_copy_of_shared_exactness_and_profile_harnesses",
    ),
    frozenset(("conditional_temperature", "shared_rope")): (
        "manually_compose_engine_policy_without_promoting_either_default",
    ),
    frozenset(("split_kv", "shared_rope")): (
        "revalidate_shared_cos_sin_storage_under_split_kv_graph_capture",
        "manually_compose_engine_policy_without_promoting_either_default",
    ),
    frozenset(("split_kv", "lazy_aot_startup")): (
        "apply_aot_q1_loader_after_final_q1_source_and_export_list",
        "rebuild_manifest_because_q1_source_hash_and_exports_change",
    ),
    frozenset(("device_sequence_state", "shared_rope")): (
        "verify_nested_opt_in_context_lifetimes_and_reverse_restoration",
    ),
}


class CompositionAuditError(RuntimeError):
    pass


def _git(repo: Path, *args: str, allow_conflicts: bool = False) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode and not allow_conflicts:
        raise CompositionAuditError(
            completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        )
    return completed


def _full_commit(repo: Path, value: str) -> str:
    commit = _git(repo, "rev-parse", f"{value}^{{commit}}").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CompositionAuditError(f"invalid commit resolution for {value!r}")
    return commit


def _changed_paths(repo: Path, base: str, commit: str) -> tuple[str, ...]:
    output = _git(repo, "diff", "--name-only", f"{base}..{commit}").stdout
    return tuple(sorted(path for path in output.splitlines() if path))


def _surface(path: str) -> str:
    if path.startswith("tests/"):
        return "test"
    if path.startswith("scripts/") or path.startswith("utils/"):
        return "evidence_harness"
    if (
        path == "inference.py"
        or path.startswith("osuT5/osuT5/inference/")
        or path.startswith("osuT5/osuT5/model/custom_transformers/")
        or path == "osuT5/osuT5/utils/__init__.py"
        or path == "osu_diffusion/__init__.py"
    ):
        return "runtime"
    return "other"


def _merge_tree_conflicts(repo: Path, left: str, right: str) -> dict[str, Any]:
    completed = _git(
        repo,
        "merge-tree",
        "--write-tree",
        left,
        right,
        allow_conflicts=True,
    )
    conflict_paths = sorted(
        {
            match.group(1)
            for match in re.finditer(
                r"^\d{6} [0-9a-f]+ [123]\t(.+)$",
                completed.stdout,
                flags=re.MULTILINE,
            )
        }
    )
    return {
        "clean": completed.returncode == 0,
        "conflict_paths": conflict_paths,
    }


def audit(
    repo: Path,
    *,
    base: str = BASE_COMMIT,
    candidates: tuple[Candidate, ...] = FRONTIER_CANDIDATES,
) -> dict[str, Any]:
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise CompositionAuditError(f"not a Git worktree: {repo}")
    base_commit = _full_commit(repo, base)
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise CompositionAuditError("candidate names must be unique")

    candidate_reports = {}
    for candidate in candidates:
        commit = _full_commit(repo, candidate.commit)
        ancestor = _git(
            repo,
            "merge-base",
            "--is-ancestor",
            base_commit,
            commit,
            allow_conflicts=True,
        ).returncode == 0
        if not ancestor:
            raise CompositionAuditError(
                f"base {base_commit} is not an ancestor of {candidate.name} {commit}"
            )
        paths = _changed_paths(repo, base_commit, commit)
        surfaces = {
            surface: [path for path in paths if _surface(path) == surface]
            for surface in ("runtime", "evidence_harness", "test", "other")
        }
        candidate_reports[candidate.name] = {
            "commit": commit,
            "evidence_state": candidate.evidence_state,
            "changed_paths": list(paths),
            "surfaces": surfaces,
        }

    pairs = {}
    for left, right in itertools.combinations(candidates, 2):
        left_report = candidate_reports[left.name]
        right_report = candidate_reports[right.name]
        all_overlap = sorted(
            set(left_report["changed_paths"]) & set(right_report["changed_paths"])
        )
        runtime_overlap = sorted(
            set(left_report["surfaces"]["runtime"])
            & set(right_report["surfaces"]["runtime"])
        )
        key = f"{left.name}+{right.name}"
        pairs[key] = {
            "changed_path_overlap": all_overlap,
            "runtime_overlap": runtime_overlap,
            "tip_merge": _merge_tree_conflicts(
                repo,
                left_report["commit"],
                right_report["commit"],
            ),
            "semantic_constraints": list(
                SEMANTIC_CONSTRAINTS.get(
                    frozenset((left.name, right.name)),
                    (),
                )
            ),
        }

    plan = []
    seen = set()
    for unit in INTEGRATION_UNITS:
        missing = sorted(set(unit["requires"]) - seen)
        if missing:
            raise CompositionAuditError(
                f"integration unit {unit['name']} precedes dependencies {missing}"
            )
        for commit in unit["commits"]:
            _full_commit(repo, commit)
        plan.append(
            {
                "name": unit["name"],
                "strategy": unit["strategy"],
                "commits": list(unit["commits"]),
                "requires": list(unit["requires"]),
                "after_if_selected": list(unit["after_if_selected"]),
            }
        )
        seen.add(unit["name"])

    return {
        "schema_version": 1,
        "base_commit": base_commit,
        "candidate_reports": candidate_reports,
        "pair_reports": pairs,
        "integration_plan": plan,
        "guardrails": {
            "branch_tips_are_not_merge_units": True,
            "unconfirmed_gpu_candidates_remain_opt_in": True,
            "aot_manifest_is_built_after_final_native_sources": True,
            "each_winner_is_reprofiled_independently_before_composition": True,
            "combined_stack_requires_fresh_exactness_and_wall_gates": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default=BASE_COMMIT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.repo, base=args.base)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
