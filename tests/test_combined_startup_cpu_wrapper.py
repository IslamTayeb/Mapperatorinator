from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "dcc" / "profile_combined_startup_cpu.sbatch"


class CombinedStartupCpuWrapperTest(unittest.TestCase):
    def test_wrapper_is_cpu_only_exact_and_reciprocal(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        source = SCRIPT.read_text()
        self.assertIn("#SBATCH --partition=common", source)
        self.assertIn("#SBATCH --account=romerolab", source)
        self.assertNotIn("#SBATCH --gres", source)
        self.assertIn("CPU combined-startup job received CUDA_VISIBLE_DEVICES", source)
        self.assertIn("status --porcelain", source)
        self.assertIn("COMBINED_REMOTE_REF", source)
        self.assertIn('export TORCH_EXTENSIONS_DIR="$NATIVE_CACHE"', source)
        self.assertIn("export TORCH_CUDA_ARCH_LIST=7.5", source)
        self.assertIn("profile_combined_startup.py", source)
        self.assertIn("--rounds 5", source)
        self.assertIn("COMBINED_STARTUP_PASS", source)


if __name__ == "__main__":
    unittest.main()
