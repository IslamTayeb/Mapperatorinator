import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/verify_inference_smoke.sbatch"


def test_inference_smoke_wrapper_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_reciprocal_native_extension_caches_are_commit_keyed_and_reused() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'EXTENSION_JOB_KEY=${SLURM_JOB_ID:-manual}' in source
    assert (
        'EXTENSION_JOB_ROOT="$WORK/torch_extensions/reciprocal-$EXTENSION_JOB_KEY"'
        in source
    )
    assert (
        'BASELINE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$BASELINE_EXTENSION_KEY"'
        in source
    )
    assert (
        'CANDIDATE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$CANDIDATE_EXTENSION_KEY"'
        in source
    )
    assert '[[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]]' in source
    assert '[[ "$BASELINE_EXTENSION_KEY" == "$CANDIDATE_EXTENSION_KEY" ]]' in source
    assert 'export TORCH_EXTENSIONS_DIR="$extension_dir"' in source
    assert 'export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"' not in source

    assert source.count('"$BASELINE_TORCH_EXTENSIONS_DIR"') >= 5
    assert source.count('"$CANDIDATE_TORCH_EXTENSIONS_DIR"') >= 5


def test_existing_extension_cache_override_is_commit_exact() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "EXTENSION_CACHE_ROOT_OVERRIDE=${EXTENSION_CACHE_ROOT_OVERRIDE:-}" in source
    assert "BASELINE_EXTENSION_KEY_OVERRIDE=${BASELINE_EXTENSION_KEY_OVERRIDE:-}" in source
    assert "CANDIDATE_EXTENSION_KEY_OVERRIDE=${CANDIDATE_EXTENSION_KEY_OVERRIDE:-}" in source
    assert 'EXTENSION_CACHE_MODE=existing_override' in source
    assert 'EXTENSION_JOB_ROOT=$(realpath -e "$EXTENSION_CACHE_ROOT_OVERRIDE")' in source
    assert (
        '[[ "$BASELINE_EXTENSION_KEY_OVERRIDE" != "$BASELINE_EXTENSION_KEY" ]]'
        in source
    )
    assert (
        '[[ "$CANDIDATE_EXTENSION_KEY_OVERRIDE" != "$CANDIDATE_EXTENSION_KEY" ]]'
        in source
    )
    assert 'native extension cache path/commit pairing is not exact' in source


def test_native_extension_preload_wall_is_a_separate_json_artifact() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "preload_native_q1_attention()" in source
    assert "preload_native_decoder_layer()" in source
    assert '"$RUN_ROOT/native-extension-setup/$label.json"' in source
    assert '"commit": commit' in source
    assert '"torch_extensions_dir": extension_dir' in source
    assert '"wall_seconds": wall_seconds' in source
    assert 'local compiler_cache="$RUN_ROOT/compiler-cache/$label"' in source
    assert 'export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"' in source
    assert 'export TRITON_CACHE_DIR="$compiler_cache/triton"' in source


def test_all_reciprocal_profiles_are_authoritative_untraced_controls() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "profile_inference=true" in source
    assert "profile_detail_ranges=false" in source
    assert "profile_cuda_capture=false" in source
    assert "profile_pass_kind=untraced_control" in source
    assert 'metadata.get("profile_pass_kind") != "untraced_control"' in source
    assert 'metadata.get("authoritative_performance") is not True' in source
    assert 'metadata.get("profile_detail_ranges") is not False' in source
    assert 'metadata.get("profile_cuda_capture") is not False' in source
