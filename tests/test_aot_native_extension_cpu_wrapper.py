from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dcc" / "profile_aot_native_extension_loading_cpu.sbatch"


class AotNativeExtensionCpuWrapperTest(unittest.TestCase):
    def test_wrapper_is_exact_cpu_only_and_enforces_direct_load_gate(self) -> None:
        source = SCRIPT.read_text()
        self.assertIn("#SBATCH --partition=common", source)
        self.assertIn("#SBATCH --account=romerolab", source)
        self.assertNotIn("#SBATCH --gres", source)
        self.assertIn('"$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT"', source)
        self.assertIn(
            '"$(git -C "$REPO" branch --show-current)" != "$BRANCH"', source
        )
        self.assertIn("status --porcelain", source)
        self.assertIn("CPU native-extension job received CUDA_VISIBLE_DEVICES", source)
        self.assertIn("export TORCH_CUDA_ARCH_LIST=7.5", source)
        self.assertIn(
            'export TORCH_EXTENSIONS_DIR="$RUN_ROOT/torch_extensions"', source
        )
        self.assertIn("build_native_extension_manifest.py", source)
        self.assertIn("benchmark_native_extension_loading.py", source)
        self.assertIn("--minimum-saving-seconds 0.5", source)
        self.assertIn("--rounds 3", source)
        self.assertIn("AOT_NATIVE_EXTENSION_LOAD_PASS", source)


if __name__ == "__main__":
    unittest.main()
