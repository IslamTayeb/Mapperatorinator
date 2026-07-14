from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dcc" / "profile_aot_native_extension_loading_cpu.sbatch"


def test_cpu_wrapper_is_exact_cpu_only_and_enforces_direct_load_gate() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --partition=common" in source
    assert "#SBATCH --account=romerolab" in source
    assert "#SBATCH --gres" not in source
    assert 'test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"' not in source
    assert '"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"' in source
    assert '"$(git -C "$REPO" branch --show-current)" != "$BRANCH"' in source
    assert 'status --porcelain' in source
    assert "CPU native-extension job received CUDA_VISIBLE_DEVICES" in source
    assert "export TORCH_CUDA_ARCH_LIST=7.5" in source
    assert 'export TORCH_EXTENSIONS_DIR="$RUN_ROOT/torch_extensions"' in source
    assert "build_native_extension_manifest.py" in source
    assert "benchmark_native_extension_loading.py" in source
    assert "--minimum-saving-seconds 0.5" in source
    assert "--rounds 3" in source
    assert "AOT_NATIVE_EXTENSION_LOAD_PASS" in source
