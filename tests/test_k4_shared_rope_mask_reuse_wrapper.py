import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "scripts/dcc/profile_k4_shared_rope_mask_reuse_full_song_reciprocal.sbatch"
)


def test_wrapper_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_wrapper_runs_exact_reciprocal_control_and_candidate_order():
    source = WRAPPER.read_text(encoding="utf-8")
    expected = (
        "run_profile baseline_first",
        "run_profile candidate_first",
        "run_profile candidate_second",
        "run_profile baseline_second",
    )
    positions = [source.index(value) for value in expected]
    assert positions == sorted(positions)
    assert source.count(
        "utils/run_k4_shared_rope_approximate_weight_only.py control"
    ) == 2
    assert source.count(
        "utils/run_k4_shared_rope_mask_reuse_approximate_weight_only.py candidate"
    ) == 2
    assert source.count("utils/validate_k4_mask_reuse_profile.py") == 1
    assert "utils/validate_k4_shared_rope_mask_reuse_initialization.py" in source
    assert "shared-rope-initialization-validation.json" in source
    assert "utils/summarize_k4_mask_reuse_reciprocal.py" in source


def test_wrapper_pins_clean_pushed_commit_and_real_2080ti():
    source = WRAPPER.read_text(encoding="utf-8")
    assert '[[ -n "$(git -C "$REPO" status --porcelain)" ]]' in source
    assert '"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"' in source
    assert '"$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT"' in source
    assert "MAPPERATORINATOR_REMOTE_BRANCH" in source
    assert 'REMOTE_REF="refs/remotes/$REMOTE/$REMOTE_BRANCH"' in source
    assert '"$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti"' in source
    assert "profile_pass_kind=untraced_control" in source
    assert "analysis.exit-code.txt" in source
    assert '[[ "$STATUS" != 0 && "$STATUS" != 3 ]]' in source
