"""Run combined K4+split+mixed inference and size its vocabulary tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from utils.run_approximate_weight_only import run as run_weight_only  # noqa: E402
from utils.vocab_sampling_scout import (  # noqa: E402
    VocabSamplingObserver,
    benchmark_existing_tail,
    install_vocab_sampling_observer,
    summarize,
)


def run(
    config_name: str,
    overrides: list[str],
    *,
    output_init_json: Path,
    output_scout_json: Path,
    max_samples: int,
    warmup: int,
    iterations: int,
    rounds: int,
    fixed_main_steps: int,
    fixed_timing_steps: int,
    mixed_projection_ms_per_step: float,
    promotion_threshold_seconds: float,
) -> dict:
    observer = VocabSamplingObserver(max_samples=max_samples)
    with install_vocab_sampling_observer(observer), install_k8_candidate(block_size=4):
        run_weight_only(config_name, overrides, output_init_json)
    component = benchmark_existing_tail(
        observer.samples,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
    )
    report = summarize(
        observer,
        component,
        fixed_main_steps=fixed_main_steps,
        fixed_timing_steps=fixed_timing_steps,
        mixed_projection_ms_per_step=mixed_projection_ms_per_step,
        promotion_threshold_seconds=promotion_threshold_seconds,
    )
    report["metadata"] = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
        ).strip(),
        "combined_base_commit": "0e6e07bba981b064da62009b5383ba60da514fb6",
        "mixed_projection_evidence": (
            "weight-only-component-49799185/component.json; "
            "selected mixed final projection weighted over 8207 replays"
        ),
        "fixed_work_evidence": (
            "current-main fixed workload: 8294 main and 821 timing steps"
        ),
    }
    output_scout_json.parent.mkdir(parents=True, exist_ok=True)
    output_scout_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--output-scout-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fixed-main-steps", type=int, default=8294)
    parser.add_argument("--fixed-timing-steps", type=int, default=821)
    parser.add_argument("--mixed-projection-ms-per-step", type=float, required=True)
    parser.add_argument("--promotion-threshold-seconds", type=float, default=1.503)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    report = run(
        parsed.config_name,
        parsed.overrides,
        output_init_json=parsed.output_init_json,
        output_scout_json=parsed.output_scout_json,
        max_samples=parsed.max_samples,
        warmup=parsed.warmup,
        iterations=parsed.iterations,
        rounds=parsed.rounds,
        fixed_main_steps=parsed.fixed_main_steps,
        fixed_timing_steps=parsed.fixed_timing_steps,
        mixed_projection_ms_per_step=parsed.mixed_projection_ms_per_step,
        promotion_threshold_seconds=parsed.promotion_threshold_seconds,
    )
    print(json.dumps({
        "decision": report["decision"],
        "fixed_work_ceiling": report["fixed_work_ceiling"],
        "distribution": report["distribution"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
