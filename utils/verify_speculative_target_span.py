"""Thin DCC CLI for the isolated q_len=K target-span numerical gate."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from inference import compile_args, load_model_with_server, setup_inference_environment  # noqa: E402
from osuT5.osuT5.inference.optimized.speculative.target_span_gpu import (  # noqa: E402
    TargetSpanGateConfig,
    load_and_validate_prior_gate,
    run_target_span_gpu_gate,
)
from utils.verify_one_token_decode import (  # noqa: E402
    _assert_supported_probe,
    _build_probe_inputs,
    _compare_logits,
    _load_args,
    _move_kwargs_for_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one FP32 q_len=K target forward with K sequential q_len=1 forwards. "
            "This is a verifier-only Slurm GPU gate and has no inference/server integration."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--speculation-k", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--prior-gate-report", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--timing-iters", type=int, default=25)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    return parser.parse_args()


def main() -> int:
    cli = _parse_args()
    prior = load_and_validate_prior_gate(cli.speculation_k, cli.prior_gate_report)
    gate_config = TargetSpanGateConfig(
        speculation_k=cli.speculation_k,
        atol=cli.atol,
        rtol=cli.rtol,
        top_k=cli.top_k,
        timing_warmup=cli.timing_warmup,
        timing_iters=cli.timing_iters,
    )
    args = _load_args(cli.config_name, cli.overrides)
    _assert_supported_probe(args)
    if args.precision != "fp32":
        raise ValueError("verify_speculative_target_span.py requires precision=fp32.")
    if prior is not None:
        expected_contract = {
            "config_name": cli.config_name,
            "sequence_index": cli.sequence_index,
            "model_path": args.model_path,
            "precision": args.precision,
            "attn_implementation": args.attn_implementation,
            "seed": args.seed,
        }
        prior_metadata = prior.get("metadata", {})
        mismatches = {
            key: {"required": value, "prior": prior_metadata.get(key)}
            for key, value in expected_contract.items()
            if prior_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Prior K gate used a different request/runtime contract: {mismatches}")
    compile_args(args, verbose=False)
    setup_inference_environment(args.seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=cli.sequence_index),
    )
    started = time.perf_counter()
    metadata = {
        "config_name": cli.config_name,
        "sequence_index": cli.sequence_index,
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(model.device),
        "cuda_device_capability": list(torch.cuda.get_device_capability(model.device)),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "generation_compile_requested": args.inference_generation_compile,
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "prior_gate_report": str(cli.prior_gate_report) if cli.prior_gate_report else None,
        "prior_gate_pass": prior.get("pass") if prior else None,
    }
    try:
        result = run_target_span_gpu_gate(
            args,
            model,
            tokenizer,
            model_inputs,
            gate_config,
            compare_logits=_compare_logits,
        )
        exit_code = 0 if result["pass"] else 1
    except Exception as exc:
        traceback.print_exc()
        result = {
            "gate": "target_span_q_len_numerics",
            "pass": False,
            "result_class": "fp32_target_span_numerical_scout_blocked",
            "speculation_k": cli.speculation_k,
            "blocker": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 1
    metadata["wall_seconds"] = time.perf_counter() - started
    result["metadata"] = metadata
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
