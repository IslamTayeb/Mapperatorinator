"""Run the bounded B8/16-step changing-prefix exactness gate."""

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
from osuT5.osuT5.inference.optimized.batch.merged_loop import (
    MergedLoopConfig,
    run_merged_loop_gate,
)
from osuT5.osuT5.inference.optimized.batch.merged_one_token import resolve_row_seeds
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
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _cache_entry_count(path_value: str | None) -> int | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return 0
    return sum(1 for _ in path.iterdir())


def _load_previous_control(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("previous control report must be a JSON object.")
    if int(payload.get("batch_size", -1)) != 8 or not bool(payload.get("exactness_pass")):
        raise ValueError("previous control must be an exactness-passing B8 report.")
    candidate = payload.get("batched_processor_candidate")
    if not isinstance(candidate, dict) or not bool(candidate.get("pass")):
        raise ValueError("previous control must contain a passing batched processor candidate.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--loop-steps", type=int, choices=(16,), default=16)
    parser.add_argument(
        "--merged-prefill-mode",
        choices=("batched", "packed_b1"),
        default="batched",
    )
    parser.add_argument("--previous-control-report", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--row-seed", action="append", type=int, default=[])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--comparison-top-k", type=int, default=20)
    parser.add_argument("--active-prefix-prefill", action="store_true")
    parser.add_argument("--active-prefix-decode", action="store_true")
    parser.add_argument("--active-prefix-decode-length", type=int)
    parser.add_argument("--active-prefix-bucket-size", type=int)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    _load_previous_control(cli.previous_control_report)
    total_started = time.perf_counter()
    args = _load_args(cli.config_name, cli.overrides)
    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("merged loop gate is FP32-only.")
    if args.seed is None:
        raise ValueError("merged loop gate requires an explicit seed.")
    if args.inference_generation_compile:
        raise ValueError("the first merged loop gate requires generation compile disabled.")
    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=max(args.max_batch_size, 8),
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    if model.device.type != "cuda":
        raise RuntimeError(f"merged loop gate requires CUDA; model is on {model.device}.")
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
    active_prefix_decode = bool(
        cli.active_prefix_decode
        or cli.active_prefix_decode_length is not None
        or cli.active_prefix_bucket_size is not None
    )
    active_prefix_decode_length = cli.active_prefix_decode_length
    if active_prefix_decode_length is None and cli.active_prefix_bucket_size is not None:
        active_prefix_decode_length = _bucketed_prefix_length(
            int(prompt.shape[-1]) + cli.loop_steps,
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
    row_seeds = resolve_row_seeds(cli.row_seed, batch_size=8, default_seed=int(args.seed))

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

    device_index = model.device.index if model.device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
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
        "cuda_device_name": properties.name,
        "cuda_compute_capability": [properties.major, properties.minor],
        "cuda_total_memory_bytes": properties.total_memory,
        "global_seed": int(args.seed),
        "row_seeds": list(row_seeds),
        "do_sample": bool(args.do_sample),
        "top_p": float(args.top_p),
        "top_k_sampling": int(args.top_k),
        "temperature": float(args.temperature),
        "stateful_monotonic_logits_processor": bool(args.inference_stateful_monotonic_logits_processor),
        "active_prefix_prefill": bool(cli.active_prefix_prefill),
        "active_prefix_decode": active_prefix_decode,
        "active_prefix_decode_length": active_prefix_decode_length,
        "active_prefix_bucket_size": cli.active_prefix_bucket_size,
        "q1_bmm_cross_attention": bool(cli.q1_bmm_cross_attention),
        "merged_prefill_mode": cli.merged_prefill_mode,
        "eos_token_ids": [int(token_id) for token_id in eos_token_ids],
        "probe": probe_metadata,
        "previous_control_report": str(cli.previous_control_report.resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_extensions_entries": _cache_entry_count(os.environ.get("TORCH_EXTENSIONS_DIR")),
        "torchinductor_cache_entries": _cache_entry_count(os.environ.get("TORCHINDUCTOR_CACHE_DIR")),
        "cuda_cache_entries": _cache_entry_count(os.environ.get("CUDA_CACHE_PATH")),
    }
    config = MergedLoopConfig(
        seeds=row_seeds,
        max_new_tokens=cli.loop_steps,
        do_sample=bool(args.do_sample),
        pad_token_id=int(tokenizer.pad_id),
        atol=cli.atol,
        rtol=cli.rtol,
        top_k=cli.comparison_top_k,
        active_prefix_prefill=bool(cli.active_prefix_prefill),
        active_prefix_decode=active_prefix_decode,
        active_prefix_decode_length=active_prefix_decode_length,
        merged_prefill_mode=cli.merged_prefill_mode,
    )
    with torch.autocast(device_type="cuda", enabled=False), generation_profile_context(
        sdpa_backend=args.profile_sdpa_backend,
        q1_bmm_cross_attention=cli.q1_bmm_cross_attention,
        native_q1_self_attention=False,
        native_q1_rope_cache_self_attention=False,
    ):
        report = run_merged_loop_gate(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_attention_mask,
            frames=model_inputs["frames"],
            condition_kwargs=_condition_kwargs(model_inputs),
            base_logits_processor_factory=base_logits_processor_factory,
            logits_warper_factory=logits_warper_factory,
            eos_token_ids=eos_token_ids,
            config=config,
            runtime_metadata=runtime_metadata,
        )
    report["total_wall_seconds"] = time.perf_counter() - total_started
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
