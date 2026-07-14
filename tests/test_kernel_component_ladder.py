from pathlib import Path
import unittest

from utils.classify_500tps_kernel_component_gate import (
    PROMOTION,
    RETAIN_FOR_IMPLEMENTATION,
    VALID_NEGATIVE,
    GateClassificationError,
    classify_gate,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dcc/profile_500tps_kernel_component_ladder.sbatch"


def _cross(promoted: bool) -> dict:
    return {
        "schema_version": 1,
        "variants": {
            "accepted_q1_bmm": {
                "kv_storage_dtype": "torch.float32",
                "checks_pass": True,
            },
            "split4_fp32": {
                "kv_storage_dtype": "torch.float32",
                "checks_pass": True,
            },
        },
        "summary": {
            "candidates": {
                "split4_fp32": {
                    "promotion_eligible": True,
                    "correctness_pass": True,
                    "sizing_pass": promoted,
                    "promotion_pass": promoted,
                }
            },
            "any_fp32_promotion_pass": promoted,
        },
    }


def _conditional_case(*, while_wins: bool) -> dict:
    timings = {"k1": 8.0, "k4": 5.0, "k8": 4.5, "while": 4.0 if while_wins else 5.1}
    return {
        "pass": True,
        "repeatable": {name: True for name in ("k1", "k4", "k8", "while")},
        "visible_tokens_exact": True,
        "logical_cache_exact": True,
        "logical_steps_exact": True,
        "physical_steps_exact": True,
        "while_no_post_stop_waste": True,
        "k1_no_post_stop_waste": True,
        "while_k1_full_cache_exact": True,
        "memory_stable": True,
        "reciprocal_cuda_ms": timings,
    }


def _conditional(*, while_wins: bool) -> dict:
    return {
        "schema_version": 1,
        "metadata": {
            "result_class": "real-prefix-conditional-while-component-gate",
            "production_wiring": False,
            "accepted_dispatch": {
                "precision": "fp32",
                "q1_bmm_cross_attention": True,
                "native_q1_self_attention": True,
                "native_q1_rope_cache_self_attention": True,
            },
        },
        "fixed_work": _conditional_case(while_wins=while_wins),
        "forced_stop": {
            stop: _conditional_case(while_wins=while_wins)
            for stop in ("1", "3", "7")
        },
        "decision": {"pass": True},
    }


def _fp16(promoted: bool) -> dict:
    return {
        "schema_version": 3,
        "summary": {
            "invariants_pass": True,
            "correctness_pass": True,
            "sizing_pass": promoted,
            "promotion_pass": promoted,
        },
    }


def _int8(*, child_sizing: bool, saving: float) -> dict:
    return {
        "schema_version": 1,
        "summary": {
            "invariants_pass": True,
            "sizing_pass": child_sizing,
            "promotion_pass": child_sizing,
            "fixed_work_main_saving_seconds": saving,
        },
    }


def _vocab(clears: bool) -> dict:
    threshold = 1.503
    ideal = 1.6 if clears else 1.4
    return {
        "schema_version": 1,
        "result_class": "verifier-only-component-sizing",
        "production_wiring_changed": False,
        "authoritative_generation_timing": False,
        "rng_exact_with_k4_control": True,
        "component_timing": {
            "representatives": [
                {"selected_token_exact": True, "top_p_mask_exact": True}
            ]
        },
        "fixed_work_ceiling": {
            "ideal_main_total_seconds": ideal,
            "promotion_threshold_seconds": threshold,
            "main_ceiling_clears_threshold": clears,
        },
        "decision": (
            "retain_for_candidate_kernel"
            if clears
            else "stop_below_main_component_gate"
        ),
    }


def _shared_rope(promoted: bool) -> dict:
    return {
        "schema_version": "mapperatorinator.shared-rope-scout.v1",
        "summary": {
            "exact_pass": True,
            "rope_call_accounting_pass": True,
            "performance_pass": promoted,
            "projected_main_saving_seconds": 1.6 if promoted else 1.4,
            "required_main_saving_seconds": 1.503,
            "promotion_pass": promoted,
        },
    }


def _prefill(promoted: bool) -> dict:
    variants = {}
    for index, name in enumerate(
        (
            "exact_fp16_graph",
            "exact_fp32_graph",
            "bucket64_fp16_graph",
            "bucket64_fp32_graph",
        )
    ):
        candidate = promoted and index == 0
        variants[name] = {
            "correctness_pass": True,
            "performance_pass": candidate,
            "promotion_pass": candidate,
        }
    return {
        "schema_version": 1,
        "decision": "PROMOTE_PREFILL_SCOUT" if promoted else "STOP_PREFILL_SCOUT",
        "promotion_pass": promoted,
        "variants": variants,
    }


class KernelComponentLadderTest(unittest.TestCase):
    def test_ladder_is_one_serial_allocation_with_exact_checkpoints(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("#SBATCH --gres=gpu:2080:1", source)
        self.assertIn("#SBATCH --time=02:00:00", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertNotIn("sbatch --parsable", source)
        for commit in (
            "3e6a4dfb934aa03f5fcec639a0b4b8559f8e143c",
            "5fafbaf59b0a0e64bc12da83d5f340dd32aa7b79",
            "5f384760dff8520bc1ad0e11414f994d3024578b",
            "907652706d402a90723724cd52f2d2ca721b635f",
            "f13e55e1bf1ad32629f2981738acb65aab09a45f",
            "d0de363f6df44195f2b3b73c0797dd57bcab9440",
            "c67211b010183101de048853433e8380b02f180d",
        ):
            self.assertIn(commit, source)

    def test_ladder_requires_reports_decisions_and_bounded_child_time(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        for artifact in (
            "q1-cross-component-${SLURM_JOB_ID:-manual}/component.json",
            "conditional-while-probe-${SLURM_JOB_ID:-manual}/real-prefix/real-prefix.json",
            "fp16-split-kv-q1-component-${SLURM_JOB_ID:-manual}/component.json",
            "int8-mlp-component-${SLURM_JOB_ID:-manual}/component.json",
            "k4-vocab-sampling-${SLURM_JOB_ID:-manual}/scout.json",
            "shared-rope-component-${SLURM_JOB_ID:-manual}/shared-rope.json",
            "prefill-graph-scout-${SLURM_JOB_ID:-manual}/analysis.json",
        ):
            self.assertIn(artifact, source)
        self.assertIn('require_artifacts "$name" "$report" "$@"', source)
        self.assertIn('sha256sum "$artifact" >> "$SUITE_ROOT/child-artifacts.sha256"', source)
        self.assertIn('local decision="$SUITE_ROOT/$name-decision.json"', source)
        self.assertIn('gate.$name.outcome=$outcome', source)
        self.assertIn('timeout --foreground --kill-after=30s "$timeout_limit"', source)
        self.assertIn('exceeded its bounded wall time', source)
        self.assertIn('MAPPERATORINATOR_REMOTE_REF="$remote_ref"', source)
        self.assertIn('LADDER_REMOTE_BRANCH=codex/500tps-kernel-component-ladder', source)
        self.assertIn('MAPPERATORINATOR_LADDER_REPO:?', source)
        self.assertIn('MAPPERATORINATOR_LADDER_COMMIT:?', source)
        self.assertNotIn('BASH_SOURCE[0]', source)

    def test_classifier_accepts_true_promotions(self) -> None:
        self.assertEqual(classify_gate("cross", 0, _cross(True)), PROMOTION)
        self.assertEqual(
            classify_gate("conditional", 0, _conditional(while_wins=True)),
            PROMOTION,
        )
        self.assertEqual(classify_gate("fp16_split", 0, _fp16(True)), PROMOTION)
        self.assertEqual(
            classify_gate("int8_mlp", 0, _int8(child_sizing=True, saving=1.6)),
            PROMOTION,
        )
        self.assertEqual(
            classify_gate("vocab", 0, _vocab(True)), RETAIN_FOR_IMPLEMENTATION
        )
        self.assertEqual(
            classify_gate("shared_rope", 0, _shared_rope(True)), PROMOTION
        )
        self.assertEqual(classify_gate("prefill", 0, _prefill(True)), PROMOTION)

    def test_classifier_accepts_valid_negative_results(self) -> None:
        self.assertEqual(classify_gate("cross", 3, _cross(False)), VALID_NEGATIVE)
        self.assertEqual(
            classify_gate("conditional", 0, _conditional(while_wins=False)),
            VALID_NEGATIVE,
        )
        self.assertEqual(
            classify_gate("fp16_split", 3, _fp16(False)), VALID_NEGATIVE
        )
        self.assertEqual(
            classify_gate("int8_mlp", 3, _int8(child_sizing=False, saving=0.2)),
            VALID_NEGATIVE,
        )
        # Passing the child's weaker 1.4x local gate is still negative when the
        # realistic whole-main saving misses 1.503 seconds.
        self.assertEqual(
            classify_gate("int8_mlp", 0, _int8(child_sizing=True, saving=1.4)),
            VALID_NEGATIVE,
        )
        self.assertEqual(classify_gate("vocab", 0, _vocab(False)), VALID_NEGATIVE)
        self.assertEqual(
            classify_gate("shared_rope", 3, _shared_rope(False)), VALID_NEGATIVE
        )
        self.assertEqual(classify_gate("prefill", 3, _prefill(False)), VALID_NEGATIVE)

    def test_every_gate_rejects_setup_or_unexpected_child_failure(self) -> None:
        reports = {
            "cross": _cross(True),
            "conditional": _conditional(while_wins=True),
            "fp16_split": _fp16(True),
            "int8_mlp": _int8(child_sizing=True, saving=1.6),
            "vocab": _vocab(True),
            "shared_rope": _shared_rope(True),
            "prefill": _prefill(True),
        }
        for name, report in reports.items():
            with self.subTest(name=name):
                with self.assertRaises(GateClassificationError):
                    classify_gate(name, 2, report)

    def test_classifier_rejects_capability_only_conditional_and_bad_decisions(self) -> None:
        capability = {
            "schema_version": 1,
            "metadata": {
                "result_class": "cuda-conditional-while-capability-probe",
                "production_wiring": False,
            },
            "decision": {"pass": True},
        }
        with self.assertRaisesRegex(GateClassificationError, "real-prefix"):
            classify_gate("conditional", 0, capability)

        bad = _cross(True)
        bad["summary"]["any_fp32_promotion_pass"] = "true"
        with self.assertRaisesRegex(GateClassificationError, "JSON boolean"):
            classify_gate("cross", 0, bad)

        bad_vocab = _vocab(False)
        bad_vocab["decision"] = "retain_for_candidate_kernel"
        with self.assertRaisesRegex(GateClassificationError, "inconsistent"):
            classify_gate("vocab", 0, bad_vocab)

        bad_prefill = _prefill(True)
        bad_prefill["promotion_pass"] = False
        with self.assertRaisesRegex(GateClassificationError, "aggregate"):
            classify_gate("prefill", 0, bad_prefill)


if __name__ == "__main__":
    unittest.main()
