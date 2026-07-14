from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dcc/profile_500tps_kernel_component_ladder.sbatch"


class KernelComponentLadderTest(unittest.TestCase):
    def test_ladder_is_one_serial_allocation_with_exact_checkpoints(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2080:1", source)
        self.assertIn("#SBATCH --time=02:00:00", source)
        self.assertNotIn("sbatch --parsable", source)
        for commit in (
            "1c8417d313ce9237f7648aca434f42b6796f2a7d",
            "5fafbaf59b0a0e64bc12da83d5f340dd32aa7b79",
            "e3279d3836d1ff6cc4f5b32f67f45afec8755902",
            "907652706d402a90723724cd52f2d2ca721b635f",
            "f13e55e1bf1ad32629f2981738acb65aab09a45f",
        ):
            self.assertIn(commit, source)

    def test_ladder_requires_each_independent_report(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        for artifact in (
            "q1-cross-component-${SLURM_JOB_ID:-manual}/component.json",
            "conditional-while-probe-${SLURM_JOB_ID:-manual}/decision.txt",
            "fp16-split-kv-q1-component-${SLURM_JOB_ID:-manual}/component.json",
            "int8-mlp-component-${SLURM_JOB_ID:-manual}/component.json",
            "k4-vocab-sampling-${SLURM_JOB_ID:-manual}/scout.json",
        ):
            self.assertIn(artifact, source)
        self.assertIn('require_artifacts "$name" "$report" "$@"', source)
        self.assertIn('classify_gate "$name" "$status" "$report"', source)
        self.assertIn('status == 3 and not promoted', source)
        self.assertIn('status == 3 and sizing is False', source)
        self.assertIn('#SBATCH --time=02:00:00', source)
        self.assertIn('LADDER_REMOTE_BRANCH=codex/500tps-kernel-component-ladder', source)


if __name__ == "__main__":
    unittest.main()
