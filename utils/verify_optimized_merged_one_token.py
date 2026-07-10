"""Run the staged real-GPU merged one-token batch-physics gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.decode_loop import _bucketed_prefix_length
from osuT5.osuT5.inference.optimized.batch.merged_one_token import (
    MergedOneTokenConfig,
    resolve_row_seeds,
    run_merged_one_token_gate,
    summarize_previous_gate_scaling,
    validate_previous_gate,
)
from osuT5.osuT5.inference.optimized.benchmark import summarize_sampling_gap_target
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _cache_entry_count(path_value: str | None) -> int | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return 0
    return sum(1 for _ in path.iterdir())


def _load_previous_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("previous gate report must contain a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 5, 8), required=True)
    parser.add_argument("--previous-gate-report", type=Path)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--row-seed", action="append", type=int, default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--warmup-repeats", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=50)
    parser.add_argument("--active-prefix-prefill", action="store_true")
    parser.add_argument("--active-prefix-decode", action="store_true")
    parser.add_argument("--active-prefix-decode-length", type=int)
    parser.add_argument("--active-prefix-bucket-size", type=int)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("--profile-sampling-components", action="store_true")
    parser.add_argument("--sampling-profile-repeats", type=int, default=200)
    parser.add_argument("--sampling-baseline-report", type=Path)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides such as audio_path=/path/song.mp3")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    previous_report = _load_previous_report(cli.previous_gate_report)
    validate_previous_gate(cli.batch_size, previous_report)
    sampling_baseline_report = _load_previous_report(cli.sampling_baseline_report)
    if cli.profile_sampling_components:
        if cli.batch_size != 8:
            raise ValueError("sampling component profiling is a bounded B=8-only gate.")
        if sampling_baseline_report is None:
            raise ValueError("--profile-sampling-components requires --sampling-baseline-report.")
        if int(sampling_baseline_report.get("batch_size", -1)) != 8:
            raise ValueError("sampling baseline must be a B=8 report.")
        if not bool(sampling_baseline_report.get("exactness_pass")):
            raise ValueError("sampling baseline must have exactness_pass=true.")
    elif sampling_baseline_report is not None:
        raise ValueError("--sampling-baseline-report requires --profile-sampling-components.")
    if cli.native_q1_rope_cache_self_attention and not cli.native_q1_self_attention:
        raise ValueError("RoPE/cache native self-attention requires --native-q1-self-attention.")
    if cli.batch_size > 1 and (
            cli.native_q1_self_attention
            or cli.native_q1_rope_cache_self_attention
    ):
        raise ValueError(
            "native q1 self-attention is a validated B1-only path and cannot be used to claim "
            "merged B>1 physics. Run the batch-compatible SDPA path first."
        )

    total_started = time.perf_counter()
    args = _load_args(cli.config_name, cli.overrides)
    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("merged one-token batch physics is FP32-only.")
    if args.seed is None:
        raise ValueError("merged one-token gate requires an explicit seed.")
    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=max(args.max_batch_size, cli.batch_size),
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    if model.device.type != "cuda":
        raise RuntimeError(f"merged one-token gate requires CUDA; model is on {model.device}.")
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=cli.sequence_index),
    )
    probe_metadata = {
        key: model_inputs.pop(key)
        for key in ("sequence_index", "frame_time_ms", "context_type", "lookback_time", "lookahead_time")
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_attention_mask = model_inputs["decoder_attention_mask"]
    condition_kwargs = _condition_kwargs(model_inputs)
    active_prefix_decode = bool(
        cli.active_prefix_decode
        or cli.active_prefix_decode_length is not None
        or cli.active_prefix_bucket_size is not None
    )
    active_prefix_decode_length = cli.active_prefix_decode_length
    if active_prefix_decode_length is None and cli.active_prefix_bucket_size is not None:
        active_prefix_decode_length = _bucketed_prefix_length(
            int(prompt.shape[-1]) + 1,
            int(cli.active_prefix_bucket_size),
            int(model.config.max_target_positions),
        )

    context_type = ContextType(probe_metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(probe_metadata["lookback_time"]),
        lookahead_time=float(probe_metadata["lookahead_time"]),
        context_type=context_type,
    )
    row_seeds = resolve_row_seeds(
        cli.row_seed,
        batch_size=cli.batch_size,
        default_seed=int(args.seed),
    )

    def base_logits_processor_factory():
        return _build_generation_logits_processors(
            args,
            tokenizer,
            model.device,
            lookback_time=float(probe_metadata["lookback_time"]),
        )

    def logits_warper_factory():
        warpers = LogitsProcessorList()
        if args.do_sample:
            if args.top_k > 0:
                warpers.append(TopKLogitsWarper(top_k=int(args.top_k)))
            if 0.0 < args.top_p < 1.0:
                warpers.append(TopPLogitsWarper(top_p=float(args.top_p)))
        return warpers

    def logits_processor_factory():
        processors = base_logits_processor_factory()
        processors.extend(logits_warper_factory())
        return processors

    sampling_gap_target = None
    if sampling_baseline_report is not None:
        baseline_timing = sampling_baseline_report["timing"]
        sampling_gap_target = summarize_sampling_gap_target(
            batch_size=8,
            target_tokens_per_second=500.0,
            complete_wall_seconds_per_step=float(
                baseline_timing["complete_sampled_step_wall_seconds_per_step"]
            ),
            model_seconds_per_step=float(
                baseline_timing["model_only_cuda_seconds_per_step"]
            ),
        )

    device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    runtime_metadata = {
        "git_commit": _git_commit(),
        "config_name": cli.config_name,
        "hydra_overrides": list(cli.overrides),
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "generation_compile_requested": bool(args.inference_generation_compile),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": device_properties.name,
        "cuda_compute_capability": [device_properties.major, device_properties.minor],
        "cuda_total_memory_bytes": device_properties.total_memory,
        "global_seed": int(args.seed),
        "row_seeds": list(row_seeds),
        "do_sample": bool(args.do_sample),
        "top_p": float(args.top_p),
        "top_k_sampling": int(args.top_k),
        "temperature": float(args.temperature),
        "stateful_monotonic_logits_processor": bool(
            args.inference_stateful_monotonic_logits_processor
        ),
        "active_prefix_prefill": bool(cli.active_prefix_prefill),
        "active_prefix_decode": active_prefix_decode,
        "active_prefix_decode_length": active_prefix_decode_length,
        "active_prefix_bucket_size": cli.active_prefix_bucket_size,
        "q1_bmm_cross_attention": bool(cli.q1_bmm_cross_attention),
        "native_q1_self_attention": bool(cli.native_q1_self_attention),
        "native_q1_rope_cache_self_attention": bool(cli.native_q1_rope_cache_self_attention),
        "profile_sampling_components": bool(cli.profile_sampling_components),
        "sampling_profile_repeats": int(cli.sampling_profile_repeats),
        "sampling_baseline_report": (
            str(cli.sampling_baseline_report.resolve())
            if cli.sampling_baseline_report is not None
            else None
        ),
        "eos_token_ids": [int(token_id) for token_id in eos_token_ids],
        "probe": probe_metadata,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "torch_extensions_entries": _cache_entry_count(os.environ.get("TORCH_EXTENSIONS_DIR")),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "torchinductor_cache_entries": _cache_entry_count(os.environ.get("TORCHINDUCTOR_CACHE_DIR")),
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        "cuda_cache_entries": _cache_entry_count(os.environ.get("CUDA_CACHE_PATH")),
    }
    gate_config = MergedOneTokenConfig(
        batch_size=cli.batch_size,
        seeds=row_seeds,
        do_sample=bool(args.do_sample),
        atol=cli.atol,
        rtol=cli.rtol,
        top_k=cli.comparison_top_k,
        warmup_repeats=cli.warmup_repeats,
        timing_repeats=cli.timing_repeats,
        active_prefix_prefill=bool(cli.active_prefix_prefill),
        active_prefix_decode=active_prefix_decode,
        active_prefix_decode_length=active_prefix_decode_length,
        profile_sampling_components=bool(cli.profile_sampling_components),
        sampling_profile_repeats=cli.sampling_profile_repeats,
    )
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
            sdpa_backend=args.profile_sdpa_backend,
            q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
            native_q1_self_attention=cli.native_q1_self_attention,
            native_q1_rope_cache_self_attention=cli.native_q1_rope_cache_self_attention,
    ):
        report = run_merged_one_token_gate(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=model_inputs["frames"],
            condition_kwargs=condition_kwargs,
            logits_processor_factory=logits_processor_factory,
            base_logits_processor_factory=base_logits_processor_factory,
            logits_warper_factory=logits_warper_factory,
            eos_token_ids=eos_token_ids,
            config=gate_config,
            runtime_metadata=runtime_metadata,
            sampling_gap_target=sampling_gap_target,
        )
    report["exactness_pass"] = bool(report["pass"])
    if previous_report is not None:
        previous_batch_size = int(previous_report["batch_size"])
        previous_tps = float(
            previous_report["timing"]["complete_wall_tokens_per_second"]
        )
        candidate_tps = float(report["timing"]["complete_wall_tokens_per_second"])
        report["previous_gate_scaling"] = summarize_previous_gate_scaling(
            previous_batch_size=previous_batch_size,
            previous_tokens_per_second=previous_tps,
            candidate_batch_size=cli.batch_size,
            candidate_tokens_per_second=candidate_tps,
        )
        report["performance_gate_pass"] = bool(
            report["previous_gate_scaling"]["clears_five_percent_gain_gate"]
        )
        report["pass"] = bool(report["exactness_pass"] and report["performance_gate_pass"])
    else:
        report["previous_gate_scaling"] = None
        report["performance_gate_pass"] = True
    report["total_wall_seconds"] = time.perf_counter() - total_started
    report["previous_gate_report"] = (
        str(cli.previous_gate_report.resolve())
        if cli.previous_gate_report is not None
        else None
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
