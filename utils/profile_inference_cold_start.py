from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


OPTIONAL_MODULES = (
    "diffusion_pipeline",
    "osu_diffusion",
    "osuT5.osuT5.inference.super_timing_generator",
    "osuT5.osuT5.utils.train_utils",
    "datasets",
    "wandb",
)


class ColdStartError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ColdStartError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _verify_repo(repo: Path, expected_commit: str, expected_branch: str) -> None:
    if not (repo / "inference.py").is_file():
        raise ColdStartError(f"missing inference.py under {repo}")
    if _git(repo, "status", "--porcelain"):
        raise ColdStartError(f"worktree is dirty: {repo}")
    if _git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ColdStartError(f"worktree commit mismatch: {repo}")
    if _git(repo, "branch", "--show-current") != expected_branch:
        raise ColdStartError(f"worktree branch mismatch: {repo}")


def _parse_import_time(stderr: str) -> dict[str, int]:
    cumulative: dict[str, int] = {}
    for raw_line in stderr.splitlines():
        if not raw_line.startswith("import time:"):
            continue
        fields = raw_line.removeprefix("import time:").split("|")
        if len(fields) != 3:
            continue
        try:
            value = int(fields[1].strip())
        except ValueError:
            continue
        module = fields[2].strip()
        cumulative[module] = max(value, cumulative.get(module, 0))
    if "inference" not in cumulative:
        raise ColdStartError("import-time output did not contain inference")
    return cumulative


def _base_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(repo),
            "PYTHONPROFILEIMPORTTIME": "1",
        }
    )
    return env


def _run_import(python: Path, repo: Path) -> dict[str, Any]:
    source = (
        "import json,sys; import inference; "
        f"names={OPTIONAL_MODULES!r}; "
        "print(json.dumps({'optional_modules_loaded':[n for n in names if n in sys.modules]}))"
    )
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [str(python), "-X", "importtime", "-c", source],
        cwd=repo,
        env=_base_env(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_ns = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise ColdStartError(
            f"import failed in {repo}: {completed.stderr[-4000:]}"
        )
    try:
        loaded = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise ColdStartError(f"import emitted invalid JSON in {repo}") from exc
    cumulative = _parse_import_time(completed.stderr)
    top = sorted(cumulative.items(), key=lambda item: item[1], reverse=True)[:25]
    return {
        "process_wall_seconds": elapsed_ns / 1_000_000_000,
        "inference_import_seconds": cumulative["inference"] / 1_000_000,
        "optional_modules_loaded": loaded["optional_modules_loaded"],
        "top_cumulative_imports": [
            {"module": module, "seconds": micros / 1_000_000}
            for module, micros in top
        ],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "process_wall_seconds_median": statistics.median(
            row["process_wall_seconds"] for row in rows
        ),
        "inference_import_seconds_median": statistics.median(
            row["inference_import_seconds"] for row in rows
        ),
        "runs": rows,
    }


def _improvement(baseline: float, candidate: float) -> dict[str, float]:
    saved = baseline - candidate
    return {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "saved_seconds": saved,
        "improvement_pct": saved / baseline * 100.0,
    }


def profile(args: argparse.Namespace) -> dict[str, Any]:
    python = args.python.resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ColdStartError(f"Python is not executable: {python}")
    _verify_repo(args.baseline_repo, args.baseline_commit, args.baseline_branch)
    _verify_repo(args.candidate_repo, args.candidate_commit, args.candidate_branch)

    rows = {"baseline": [], "candidate": []}
    for index in range(args.repetitions):
        order = ("baseline", "candidate") if index % 2 == 0 else (
            "candidate",
            "baseline",
        )
        for name in order:
            repo = args.baseline_repo if name == "baseline" else args.candidate_repo
            rows[name].append(_run_import(python, repo))

    baseline = _aggregate(rows["baseline"])
    candidate = _aggregate(rows["candidate"])
    process = _improvement(
        baseline["process_wall_seconds_median"],
        candidate["process_wall_seconds_median"],
    )
    import_time = _improvement(
        baseline["inference_import_seconds_median"],
        candidate["inference_import_seconds_median"],
    )
    loaded = sorted(
        {
            module
            for row in candidate["runs"]
            for module in row["optional_modules_loaded"]
        }
    )
    checks = {
        "candidate_optional_modules_cold": not loaded,
        "process_wall_improvement": process["improvement_pct"]
        >= args.minimum_improvement_pct,
        "import_time_improvement": import_time["improvement_pct"]
        >= args.minimum_improvement_pct,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "precision_scope": "precision_agnostic_fp32_and_fp16",
        "repetitions": args.repetitions,
        "minimum_improvement_pct": args.minimum_improvement_pct,
        "baseline": baseline,
        "candidate": candidate,
        "process_wall": process,
        "inference_import": import_time,
        "candidate_optional_modules_loaded": loaded,
        "checks": checks,
    }


def _text(report: dict[str, Any]) -> str:
    process = report["process_wall"]
    imported = report["inference_import"]
    return "\n".join(
        (
            f"status={report['status']}",
            f"precision_scope={report['precision_scope']}",
            f"repetitions={report['repetitions']}",
            f"process_wall_baseline_seconds={process['baseline_seconds']:.6f}",
            f"process_wall_candidate_seconds={process['candidate_seconds']:.6f}",
            f"process_wall_saved_seconds={process['saved_seconds']:.6f}",
            f"process_wall_improvement_pct={process['improvement_pct']:.3f}",
            f"import_baseline_seconds={imported['baseline_seconds']:.6f}",
            f"import_candidate_seconds={imported['candidate_seconds']:.6f}",
            f"import_saved_seconds={imported['saved_seconds']:.6f}",
            f"import_improvement_pct={imported['improvement_pct']:.3f}",
            "optional_modules_loaded="
            + ",".join(report["candidate_optional_modules_loaded"]),
        )
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--baseline-branch", required=True)
    parser.add_argument("--candidate-repo", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-branch", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--minimum-improvement-pct", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    report = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(_text(report))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
