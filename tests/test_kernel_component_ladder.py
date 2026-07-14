from pathlib import Path
import unittest

from utils.classify_500tps_kernel_component_gate import (
    GateClassificationError,
    classify_gate,
)


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
            "d0de363f6df44195f2b3b73c0797dd57bcab9440",
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
            "shared-rope-component-${SLURM_JOB_ID:-manual}/shared-rope.json",
        ):
            self.assertIn(artifact, source)
        self.assertIn('require_artifacts "$name" "$report" "$@"', source)
        self.assertIn('sha256sum "$artifact" >> "$SUITE_ROOT/child-artifacts.sha256"', source)
        self.assertIn('classify_gate "$name" "$status" "$report"', source)
        self.assertIn('classify_500tps_kernel_component_gate.py', source)
        self.assertIn('MAPPERATORINATOR_REMOTE_REF="$remote_ref"', source)
        self.assertIn('#SBATCH --time=02:00:00', source)
        self.assertIn('LADDER_REMOTE_BRANCH=codex/500tps-kernel-component-ladder', source)
        self.assertIn('MAPPERATORINATOR_LADDER_REPO:?', source)
        self.assertIn('MAPPERATORINATOR_LADDER_COMMIT:?', source)
        self.assertNotIn('BASH_SOURCE[0]', source)

    def test_classifier_accepts_promoted_and_valid_negative_results(self) -> None:
        classify_gate(
            "cross",
            0,
            {
                "variants": {
                    "accepted": {
                        "kv_storage_dtype": "torch.float32",
                        "checks_pass": True,
                    }
                },
                "summary": {"any_fp32_promotion_pass": True},
            },
        )
        classify_gate(
            "fp16_split",
            3,
            {"summary": {"invariants_pass": True, "sizing_pass": False}},
        )
        classify_gate(
            "shared_rope",
            3,
            {
                "summary": {
                    "exact_pass": True,
                    "rope_call_accounting_pass": True,
                    "promotion_pass": False,
                }
            },
        )

    def test_classifier_rejects_string_booleans_and_exit_mismatches(self) -> None:
        with self.assertRaisesRegex(GateClassificationError, "JSON boolean"):
            classify_gate(
                "cross",
                0,
                {
                    "variants": {
                        "bad": {
                            "kv_storage_dtype": "torch.float32",
                            "checks_pass": "false",
                        }
                    },
                    "summary": {"any_fp32_promotion_pass": True},
                },
            )
        with self.assertRaisesRegex(GateClassificationError, "unexpected exit/sizing"):
            classify_gate(
                "int8_mlp",
                0,
                {"summary": {"invariants_pass": True, "sizing_pass": False}},
            )
        with self.assertRaisesRegex(GateClassificationError, "inconsistent"):
            classify_gate(
                "vocab",
                0,
                {
                    "fixed_work_ceiling": {"main_ceiling_clears_threshold": False},
                    "decision": "retain_for_candidate_kernel",
                },
            )


if __name__ == "__main__":
    unittest.main()
