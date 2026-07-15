from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


VARIANTS = ("lazy_parent", "aot_parent", "combined")
OPTIONAL_MODULES = (
    "diffusion_pipeline",
    "osu_diffusion.utils.models",
    "osu_diffusion.utils.diffusion",
    "osuT5.osuT5.inference.super_timing_generator",
    "osuT5.osuT5.utils.train_utils",
    "datasets",
    "wandb",
)
EXTENSION_FUNCTIONS = {
    "mapperatorinator_q1_attention": (
        "q1_attention",
        "q1_rope_cache_attention",
    ),
    "mapperatorinator_native_decoder_layer": (
        "one_token_mlp_residual",
        "one_token_rmsnorm_linear",
        "one_token_linear_residual",
    ),
}


class CombinedStartupError(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CombinedStartupError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout.strip()


def _verify_repo(repo: Path, *, commit: str, branch: str) -> None:
    if not (repo / "inference.py").is_file():
        raise CombinedStartupError(f"missing inference.py under {repo}")
    if _git(repo, "status", "--porcelain"):
        raise CombinedStartupError(f"worktree is dirty: {repo}")
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise CombinedStartupError(f"worktree commit mismatch: {repo}")
    if _git(repo, "branch", "--show-current") != branch:
        raise CombinedStartupError(f"worktree branch mismatch: {repo}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe(function: Any) -> dict[str, str]:
    try:
        function()
    except BaseException as exc:
        return {
            "kind": "exception",
            "type": type(exc).__name__,
            "message": str(exc).splitlines()[0],
        }
    return {"kind": "return", "type": "None", "message": ""}


def worker(repo: Path, *, mode: str, manifest: Path) -> dict[str, Any]:
    repo = repo.resolve()
    if mode not in {"cached", "direct"}:
        raise CombinedStartupError("worker mode must be cached or direct")
    manifest_env = "MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST"
    if mode == "direct":
        if not manifest.is_file():
            raise CombinedStartupError(f"missing direct-load manifest: {manifest}")
        os.environ[manifest_env] = str(manifest.resolve())
    else:
        os.environ.pop(manifest_env, None)
    sys.path.insert(0, str(repo))

    total_started = time.perf_counter()
    import_started = time.perf_counter()
    import inference  # noqa: F401

    import_seconds = time.perf_counter() - import_started
    optional_loaded = [name for name in OPTIONAL_MODULES if name in sys.modules]
    import_modules = sorted(
        name
        for name in sys.modules
        if name == "inference"
        or name == "config"
        or name.startswith(("config.", "osuT5.", "osu_diffusion."))
    )

    native_started = time.perf_counter()
    from osuT5.osuT5.inference.optimized.kernels import decoder_layer, q1_attention

    modules = (
        q1_attention.preload_native_q1_attention(),
        decoder_layer.preload_native_decoder_layer(),
    )
    native_seconds = time.perf_counter() - native_started
    total_seconds = time.perf_counter() - total_started

    extension_evidence = {}
    for expected_name, module in zip(EXTENSION_FUNCTIONS, modules, strict=True):
        if module.__name__ != expected_name:
            raise CombinedStartupError(
                f"native extension name changed: {module.__name__} != {expected_name}"
            )
        library = Path(module.__file__).resolve()
        functions = EXTENSION_FUNCTIONS[expected_name]
        extension_evidence[expected_name] = {
            "library_sha256": _sha256_file(library),
            "functions": list(functions),
            "probes": {
                name: _probe(getattr(module, name)) for name in functions
            },
        }
    return {
        "mode": mode,
        "repo": str(repo),
        "import_seconds": import_seconds,
        "native_load_seconds": native_seconds,
        "ready_seconds": total_seconds,
        "optional_modules_loaded": optional_loaded,
        "import_modules": import_modules,
        "extensions": extension_evidence,
    }


def _run_worker(
    *,
    python: Path,
    script: Path,
    repo: Path,
    mode: str,
    manifest: Path,
) -> dict[str, Any]:
    command = [
        str(python),
        str(script),
        "--worker",
        "--repo",
        str(repo),
        "--mode",
        mode,
        "--manifest",
        str(manifest),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise CombinedStartupError(
            f"startup worker failed for {repo} ({mode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CombinedStartupError(
            f"startup worker emitted invalid JSON: {completed.stdout!r}"
        ) from exc


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: float(statistics.median(float(row[name]) for row in rows))
        for name in ("import_seconds", "native_load_seconds", "ready_seconds")
    } | {"runs": rows}


def _comparison(parent: float, combined: float) -> dict[str, float]:
    saved = parent - combined
    return {
        "parent_seconds": parent,
        "combined_seconds": combined,
        "saved_seconds": saved,
        "improvement_pct": saved / parent * 100.0,
    }


def summarize(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(rows) != set(VARIANTS) or any(not rows[name] for name in VARIANTS):
        raise CombinedStartupError("combined startup requires every parent and candidate")
    aggregates = {name: _aggregate(rows[name]) for name in VARIANTS}
    reference_extensions = rows["combined"][0]["extensions"]
    parity_failures = [
        {"variant": name, "index": index}
        for name in VARIANTS
        for index, row in enumerate(rows[name])
        if row["extensions"] != reference_extensions
    ]
    combined_optional = sorted(
        {
            module
            for row in rows["combined"]
            for module in row["optional_modules_loaded"]
        }
    )
    lazy_import_topologies = {
        tuple(row["import_modules"]) for row in rows["lazy_parent"]
    }
    combined_import_topologies = {
        tuple(row["import_modules"]) for row in rows["combined"]
    }
    comparisons = {
        parent: {
            metric: _comparison(
                aggregates[parent][metric],
                aggregates["combined"][metric],
            )
            for metric in ("import_seconds", "native_load_seconds", "ready_seconds")
        }
        for parent in ("lazy_parent", "aot_parent")
    }
    predicted_independent = (
        aggregates["lazy_parent"]["import_seconds"]
        + aggregates["aot_parent"]["native_load_seconds"]
    )
    interaction = aggregates["combined"]["ready_seconds"] - predicted_independent
    checks = {
        "extension_parity": not parity_failures,
        "combined_optional_modules_cold": not combined_optional,
        "lazy_import_topology_retained": (
            len(lazy_import_topologies) == 1
            and lazy_import_topologies == combined_import_topologies
        ),
        "lazy_import_no_more_than_five_percent_slower": comparisons[
            "lazy_parent"
        ]["import_seconds"]["improvement_pct"]
        >= -5.0,
        "aot_direct_load_retained": comparisons["aot_parent"][
            "native_load_seconds"
        ]["improvement_pct"]
        >= -10.0,
        "combined_ready_beats_lazy_parent": comparisons["lazy_parent"][
            "ready_seconds"
        ]["saved_seconds"]
        > 0.0,
        "combined_ready_beats_aot_parent": comparisons["aot_parent"][
            "ready_seconds"
        ]["saved_seconds"]
        > 0.0,
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "precision_scope": "precision_agnostic_fp32_and_fp16",
        "aggregates": aggregates,
        "comparisons": comparisons,
        "independent_combination": {
            "predicted_ready_seconds": predicted_independent,
            "observed_ready_seconds": aggregates["combined"]["ready_seconds"],
            "interaction_seconds": interaction,
        },
        "extension_parity_failures": parity_failures,
        "combined_optional_modules_loaded": combined_optional,
        "lazy_parent_import_modules": list(next(iter(lazy_import_topologies))),
        "combined_import_modules": list(next(iter(combined_import_topologies))),
        "checks": checks,
    }


def profile(args: argparse.Namespace) -> dict[str, Any]:
    repos = {
        "lazy_parent": args.lazy_repo.resolve(),
        "aot_parent": args.aot_repo.resolve(),
        "combined": args.combined_repo.resolve(),
    }
    expected = {
        "lazy_parent": (args.lazy_commit, args.lazy_branch),
        "aot_parent": (args.aot_commit, args.aot_branch),
        "combined": (args.combined_commit, args.combined_branch),
    }
    for name in VARIANTS:
        commit, branch = expected[name]
        _verify_repo(repos[name], commit=commit, branch=branch)
    if args.rounds < 6:
        raise CombinedStartupError("combined startup requires all six reciprocal orders")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise CombinedStartupError(f"Python is not executable: {args.python}")
    if not args.manifest.is_file():
        raise CombinedStartupError(f"manifest does not exist: {args.manifest}")

    modes = {
        "lazy_parent": "cached",
        "aot_parent": "direct",
        "combined": "direct",
    }
    orders = (
        VARIANTS,
        ("lazy_parent", "combined", "aot_parent"),
        ("aot_parent", "lazy_parent", "combined"),
        ("aot_parent", "combined", "lazy_parent"),
        ("combined", "lazy_parent", "aot_parent"),
        ("combined", "aot_parent", "lazy_parent"),
    )
    rows = {name: [] for name in VARIANTS}
    script = Path(__file__).resolve()
    for index in range(args.rounds):
        for name in orders[index % len(orders)]:
            rows[name].append(
                _run_worker(
                    python=args.python,
                    script=script,
                    repo=repos[name],
                    mode=modes[name],
                    manifest=args.manifest,
                )
            )
    report = summarize(rows)
    report["rounds"] = args.rounds
    report["repos"] = {
        name: {"path": str(repos[name]), "commit": expected[name][0], "branch": expected[name][1]}
        for name in VARIANTS
    }
    report["manifest"] = str(args.manifest.resolve())
    return report


def _text(report: dict[str, Any]) -> str:
    rows = [f"status={report['status']}"]
    for name in VARIANTS:
        aggregate = report["aggregates"][name]
        for metric in ("import_seconds", "native_load_seconds", "ready_seconds"):
            rows.append(f"{name}.{metric}={aggregate[metric]:.6f}")
    for parent in ("lazy_parent", "aot_parent"):
        comparison = report["comparisons"][parent]["ready_seconds"]
        rows.append(f"combined_vs_{parent}.saved_seconds={comparison['saved_seconds']:.6f}")
        rows.append(f"combined_vs_{parent}.improvement_pct={comparison['improvement_pct']:.3f}")
    interaction = report["independent_combination"]
    rows.append(
        "independent_combination.interaction_seconds="
        f"{interaction['interaction_seconds']:.6f}"
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--mode", choices=("cached", "direct"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--lazy-repo", type=Path)
    parser.add_argument("--lazy-commit")
    parser.add_argument("--lazy-branch")
    parser.add_argument("--aot-repo", type=Path)
    parser.add_argument("--aot-commit")
    parser.add_argument("--aot-branch")
    parser.add_argument("--combined-repo", type=Path)
    parser.add_argument("--combined-commit")
    parser.add_argument("--combined-branch")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.repo is None or args.mode is None:
            parser.error("--worker requires --repo and --mode")
        print(json.dumps(worker(args.repo, mode=args.mode, manifest=args.manifest), sort_keys=True))
        return
    required = (
        "python",
        "lazy_repo",
        "lazy_commit",
        "lazy_branch",
        "aot_repo",
        "aot_commit",
        "aot_branch",
        "combined_repo",
        "combined_commit",
        "combined_branch",
        "output",
        "text_output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"benchmark mode is missing: {', '.join(missing)}")
    report = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.text_output.write_text(_text(report))
    if report["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
