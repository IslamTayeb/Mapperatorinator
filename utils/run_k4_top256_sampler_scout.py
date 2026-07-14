"""Capture real selected-runtime logits and gate the bounded top-256 sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
)
from utils.run_k4_shared_rope_cross_candidate import run as run_cross  # noqa: E402
from utils.top256_sampler_scout import (  # noqa: E402
    FIXED_MAIN_STEPS,
    MAX_CANDIDATE_MS_PER_STEP,
    MIN_FIXED_MAIN_SAVING_SECONDS,
    benchmark_candidate,
    fixed_physical_work,
    render_text,
    summarize_candidate,
)
from utils.vocab_sampling_scout import (  # noqa: E402
    VocabSamplingObserver,
    install_vocab_sampling_observer,
)


SELECTED_BASE_COMMIT = "7569d50b26bada95c4dafcadbdb68b15f628fc16"


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def run(
    config_name: str,
    overrides: list[str],
    *,
    output_init_json: Path,
    output_report_json: Path,
    output_report_text: Path,
    inference_output_dir: Path,
    max_samples: int,
    warmup: int,
    iterations: int,
    rounds: int,
    fixed_main_steps: int,
    max_candidate_ms_per_step: float,
    minimum_saving_seconds: float,
) -> dict:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    observer = VocabSamplingObserver(max_samples=max_samples)
    with install_vocab_sampling_observer(observer):
        run_cross(
            config_name,
            overrides,
            output_init_json,
            mode=CROSS_FP16_PACKED,
        )
    if not observer.samples:
        raise RuntimeError("selected runtime produced no vocabulary samples")

    component = benchmark_candidate(
        observer.samples,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
    )
    physical = fixed_physical_work(
        inference_output_dir,
        fixed_main_steps=fixed_main_steps,
    )
    report = summarize_candidate(
        component,
        physical,
        max_candidate_ms_per_step=max_candidate_ms_per_step,
        minimum_saving_seconds=minimum_saving_seconds,
    )
    report["metadata"] = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "selected_base_commit": SELECTED_BASE_COMMIT,
        "selected_runtime": (
            "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-"
            "fp16-packed-projections"
        ),
        "sampler_runtime_installed": False,
        "counter_threshold_reused": True,
        "unbounded_fallback_required": True,
    }
    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    output_report_text.parent.mkdir(parents=True, exist_ok=True)
    output_report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_report_text.write_text(render_text(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--output-report-json", type=Path, required=True)
    parser.add_argument("--output-report-text", type=Path, required=True)
    parser.add_argument("--inference-output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fixed-main-steps", type=int, default=FIXED_MAIN_STEPS)
    parser.add_argument(
        "--max-candidate-ms-per-step",
        type=float,
        default=MAX_CANDIDATE_MS_PER_STEP,
    )
    parser.add_argument(
        "--minimum-saving-seconds",
        type=float,
        default=MIN_FIXED_MAIN_SAVING_SECONDS,
    )
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    report = run(
        parsed.config_name,
        parsed.overrides,
        output_init_json=parsed.output_init_json,
        output_report_json=parsed.output_report_json,
        output_report_text=parsed.output_report_text,
        inference_output_dir=parsed.inference_output_dir,
        max_samples=parsed.max_samples,
        warmup=parsed.warmup,
        iterations=parsed.iterations,
        rounds=parsed.rounds,
        fixed_main_steps=parsed.fixed_main_steps,
        max_candidate_ms_per_step=parsed.max_candidate_ms_per_step,
        minimum_saving_seconds=parsed.minimum_saving_seconds,
    )
    print(render_text(report), end="")
    if not report["gate"]["promotion_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
