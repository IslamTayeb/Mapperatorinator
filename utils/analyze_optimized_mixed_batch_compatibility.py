"""Analyze accepted five-song profiles for exact fixed-slot compatibility."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_model_free_analysis() -> ModuleType:
    """Load without importing the heavyweight ``osuT5.inference`` package."""

    module_path = (
        REPO_ROOT
        / "osuT5/osuT5/inference/optimized/batch/mixed_profile_compatibility.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mapperatorinator_mixed_profile_compatibility",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model-free analyzer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        type=Path,
        required=True,
        help="Accepted repeat01 profile JSON; provide exactly five.",
    )
    parser.add_argument("--report-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.profile) != 5:
        raise ValueError("provide exactly five accepted repeat01 profiles.")
    analysis = _load_model_free_analysis()
    songs = []
    artifacts = []
    for path in args.profile:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"profile does not exist: {resolved}")
        sha256 = _file_sha256(resolved)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"profile must contain a JSON object: {resolved}")
        song = analysis.accepted_song_from_profile(
            payload,
            profile_artifact_sha256=sha256,
        )
        songs.append(song)
        artifacts.append({
            "song_id": song.song_id,
            "path": str(resolved),
            "sha256": sha256,
            "size_bytes": resolved.stat().st_size,
        })
    report = analysis.analyze_five_song_profiles(songs)
    report["analysis_git_commit"] = _git_commit()
    report["profile_artifacts"] = artifacts
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
