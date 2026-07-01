from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import (  # noqa: E402
    compile_args,
    generate,
    get_default_logger,
    get_config,
    load_diff_model,
    load_model_with_server,
    setup_inference_environment,
    should_load_separate_timing_model,
)
from osuT5.osuT5.inference.profiler import InferenceProfiler  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a same-process inference profiling suite with one model load."
    )
    parser.add_argument("--config-name", default="profile_salvalai", help="Hydra inference config name.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs.")
    parser.add_argument(
        "--run-kind",
        default="warm_repeat",
        choices=["warm_repeat", "serial_multi_song"],
        help="Result class recorded in suite/profile metadata.",
    )
    parser.add_argument("--suite-id", default=None, help="Stable suite id. Defaults to a random short id.")
    parser.add_argument("--output-root", default=None, help="Directory for per-run outputs and suite_manifest.json.")
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help="Increment seed by this amount per run. Default 0 resets every run to the same seed.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides for the inference config, e.g. inference_active_prefix_decode_loop=true.",
    )
    return parser.parse_args()


def _compose_config(config_name: str, overrides: list[str]) -> Any:
    config_dir = REPO_ROOT / "configs" / "inference"
    with hydra.initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _load_models(args: Any) -> dict[str, Any]:
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )

    timing_model, timing_tokenizer = None, None
    if should_load_separate_timing_model(args):
        print("Using base model for timing generation.")
        timing_model, timing_tokenizer = load_model_with_server(
            args.model_path,
            args.train,
            args.device,
            max_batch_size=args.max_batch_size,
            use_server=args.use_server,
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            gamemode=args.gamemode,
            auto_select_gamemode_model=False,
            generation_compile=args.inference_generation_compile,
        )

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)
        if os.path.exists(args.diff_refine_ckpt):
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion, args.device)[0]
        if args.compile:
            diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "timing_model": timing_model,
        "timing_tokenizer": timing_tokenizer,
        "diff_model": diff_model,
        "diff_tokenizer": diff_tokenizer,
        "refine_model": refine_model,
    }


def _summary_for_label(profile: dict[str, Any], label: str) -> dict[str, Any]:
    return profile.get("summary", {}).get("generation_by_label", {}).get(label, {})


def _flatten_token_ids(profile: dict[str, Any], label: str) -> list[int] | None:
    tokens: list[int] = []
    saw_tokens = False
    for record in profile.get("generation", []):
        if record.get("profile_label") != label:
            continue
        if "generated_token_ids" in record:
            value = record.get("generated_token_ids")
            if value is None:
                continue
            saw_tokens = True
            tokens.extend(int(token) for token in value)
        elif "generated_token_ids_per_sample" in record:
            value = record.get("generated_token_ids_per_sample")
            if value is None:
                continue
            saw_tokens = True
            for sample in value:
                tokens.extend(int(token) for token in sample)
    return tokens if saw_tokens else None


def _aggregate(runs: list[dict[str, Any]], *, start_index: int) -> dict[str, Any]:
    selected = runs[start_index:]
    generated_tokens = sum(int(run.get("main_generated_tokens") or 0) for run in selected)
    model_elapsed_seconds = sum(float(run.get("main_model_elapsed_seconds") or 0.0) for run in selected)
    return {
        "runs": len(selected),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "tokens_per_second": generated_tokens / model_elapsed_seconds if model_elapsed_seconds > 0 else 0.0,
    }


def _write_manifest(
        suite_dir: Path,
        *,
        suite_id: str,
        config_name: str,
        overrides: list[str],
        run_kind: str,
        seed_step: int,
        runs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "run_kind": run_kind,
        "config_name": config_name,
        "overrides": overrides,
        "rng_reset_policy": "accelerate.set_seed(seed + run_index * seed_step) before each generate()",
        "seed_step": seed_step,
        "warmup_excluded": [0] if len(runs) > 1 else [],
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "runs": runs,
        "aggregate": {
            "all_runs": _aggregate(runs, start_index=0),
            "warmed_runs": _aggregate(runs, start_index=1) if len(runs) > 1 else None,
        },
    }
    manifest_path = suite_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    cli_args = _parse_args()
    if cli_args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    cfg = _compose_config(cli_args.config_name, cli_args.overrides)
    base_args = OmegaConf.to_object(cfg)
    compile_args(base_args)

    if cli_args.run_kind == "serial_multi_song":
        raise ValueError("serial_multi_song requires an explicit multi-song config/list; use warm_repeat for one song.")
    if not base_args.profile_inference:
        raise ValueError("profile_inference must be true for suite profiling.")
    if not base_args.profile_record_token_ids:
        raise ValueError("profile_record_token_ids must be true for suite token-equivalence checks.")
    if base_args.use_server:
        raise ValueError("profile_inference_suite requires use_server=false until server reseeding is explicit.")

    suite_id = cli_args.suite_id or uuid.uuid4().hex[:12]
    suite_dir = Path(cli_args.output_root or base_args.output_path) / f"{cli_args.run_kind}-{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    setup_inference_environment(base_args.seed)
    assets = _load_models(base_args)

    generation_config, beatmap_config = get_config(base_args)
    logger = get_default_logger()
    baseline_tokens: list[int] | None = None
    runs: list[dict[str, Any]] = []

    for run_index in range(cli_args.repeats):
        run_args = copy.deepcopy(base_args)
        if run_args.seed is None:
            raise ValueError("seed must be set before suite runs.")
        run_args.seed = int(run_args.seed) + run_index * cli_args.seed_step

        run_dir = suite_dir / f"run{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_args.output_path = str(run_dir)
        run_args.profile_output_path = str(run_dir / f"run{run_index:02d}.profile.json")

        set_seed(run_args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        profiler = InferenceProfiler.from_args(run_args)
        profiler.set_metadata(
            suite_id=suite_id,
            run_kind=cli_args.run_kind,
            suite_run_index=run_index,
            run_index=run_index,
            song_index=0,
            suite_repeat_count=cli_args.repeats,
            rng_reset_policy="accelerate.set_seed before generate",
            seed_step=cli_args.seed_step,
            warmup_excluded=run_index == 0 and cli_args.repeats > 1,
        )

        print(f"[suite] run {run_index + 1}/{cli_args.repeats}: seed={run_args.seed}")
        _, result_path = generate(
            run_args,
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=assets["model"],
            tokenizer=assets["tokenizer"],
            timing_model=assets["timing_model"],
            timing_tokenizer=assets["timing_tokenizer"],
            diff_model=assets["diff_model"],
            diff_tokenizer=assets["diff_tokenizer"],
            refine_model=assets["refine_model"],
            profiler=profiler,
            logger=logger,
        )

        profile_path = Path(run_args.profile_output_path)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        main_summary = _summary_for_label(profile, "main_generation")
        tokens = _flatten_token_ids(profile, "main_generation")
        if run_index == 0:
            baseline_tokens = tokens
            token_equivalence = "baseline"
        elif cli_args.seed_step != 0:
            token_equivalence = "not_checked_seed_changed"
        elif baseline_tokens is None or tokens is None:
            token_equivalence = "not_checked"
        else:
            token_equivalence = "PASS" if tokens == baseline_tokens else "FAIL"

        run_record = {
            "run_index": run_index,
            "seed": run_args.seed,
            "result_path": str(result_path),
            "profile_path": str(profile_path),
            "main_generated_tokens": int(main_summary.get("generated_tokens", 0) or 0),
            "main_model_elapsed_seconds": float(main_summary.get("model_elapsed_seconds", 0.0) or 0.0),
            "main_tokens_per_second": float(main_summary.get("tokens_per_second", 0.0) or 0.0),
            "token_equivalence_to_run0": token_equivalence,
        }
        print(
            "[suite] run {run_index}: main={tokens} tokens, model={seconds:.3f}s, "
            "tok/s={tok_s:.3f}, equivalence={equivalence}".format(
                run_index=run_index,
                tokens=run_record["main_generated_tokens"],
                seconds=run_record["main_model_elapsed_seconds"],
                tok_s=run_record["main_tokens_per_second"],
                equivalence=token_equivalence,
            )
        )
        runs.append(run_record)

    manifest_path = _write_manifest(
        suite_dir,
        suite_id=suite_id,
        config_name=cli_args.config_name,
        overrides=cli_args.overrides,
        run_kind=cli_args.run_kind,
        seed_step=cli_args.seed_step,
        runs=runs,
    )
    print(f"[suite] manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
