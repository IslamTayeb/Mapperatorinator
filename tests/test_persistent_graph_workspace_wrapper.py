from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dcc_wrapper_is_fail_loud_and_runs_one_same_process_pair() -> None:
    source = (
        ROOT / "scripts/dcc/profile_persistent_graph_workspace.sbatch"
    ).read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert "NVIDIA GeForce RTX 2080 Ti" in source
    assert "git -C \"$REPO\" status --porcelain" in source
    assert "refs/remotes/$REMOTE/$BRANCH" in source
    assert "run_persistent_graph_workspace_scout.py" in source
    assert source.count("run_persistent_graph_workspace_scout.py") == 1
    assert "analyze_persistent_graph_workspace.py" in source
    assert "profile_pass_kind=untraced_control" in source
    assert "sbatch " not in source
