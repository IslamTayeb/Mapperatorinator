from pathlib import Path

import pytest

from utils.run_external_inference_with_rng import _parse_args, _validate_repo


def test_external_runner_changes_to_target_worktree_before_runpy() -> None:
    source = Path("utils/run_external_inference_with_rng.py").read_text(
        encoding="utf-8"
    )

    assert source.index("os.chdir(repo)") < source.index("runpy.run_path(")


def test_external_runner_preserves_target_and_hydra_arguments() -> None:
    repo, commit, role, evidence, remaining = _parse_args(
        [
            "--repo",
            "/tmp/baseline",
            "--expected-commit",
            "a" * 40,
            "--role",
            "baseline_first",
            "--evidence-manifest",
            "/tmp/evidence.json",
            "--",
            "--config-name",
            "profile_salvalai",
            "precision=fp16",
        ]
    )
    assert repo == Path("/tmp/baseline")
    assert commit == "a" * 40
    assert role == "baseline_first"
    assert evidence == Path("/tmp/evidence.json")
    assert remaining == ["--config-name", "profile_salvalai", "precision=fp16"]


def test_external_runner_fails_before_git_for_missing_entrypoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="inference.py"):
        _validate_repo(tmp_path, "a" * 40)
