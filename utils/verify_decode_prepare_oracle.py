from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from config import InferenceConfig
from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import prepare_one_token_decode_inputs_fast
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_direct_decode_loop import _rng_state, _set_rng_state
from utils.verify_one_token_decode import (
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "is_contiguous": bool(value.is_contiguous()),
    }


def _compare_values(reference: Any, candidate: Any) -> dict[str, Any]:
    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            return {
                "match": False,
                "reason": f"type mismatch {type(reference).__name__} vs {type(candidate).__name__}",
            }
        shape_match = tuple(reference.shape) == tuple(candidate.shape)
        dtype_match = reference.dtype == candidate.dtype
        device_match = reference.device == candidate.device
        stride_match = tuple(reference.stride()) == tuple(candidate.stride())
        equal = bool(
            shape_match
            and dtype_match
            and device_match
            and torch.equal(reference, candidate)
        )
        result: dict[str, Any] = {
            "match": equal,
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "device_match": device_match,
            "stride_match": stride_match,
            "reference": _tensor_summary(reference),
            "candidate": _tensor_summary(candidate),
        }
        if shape_match and dtype_match and device_match and reference.numel() and reference.is_floating_point():
            result["max_abs"] = float((reference - candidate).abs().max().detach().cpu().item())
        elif shape_match and dtype_match and device_match and reference.numel() and not equal:
            result["mismatch_count"] = int((reference != candidate).sum().detach().cpu().item())
        return result

    if reference is None or candidate is None:
        return {"match": reference is None and candidate is None}
    if isinstance(reference, (bool, int, float, str)):
        return {"match": reference == candidate, "reference": reference, "candidate": candidate}
    return {
        "match": reference is candidate,
        "identity_match": reference is candidate,
        "reference_type": type(reference).__name__,
        "candidate_type": type(candidate).__name__,
    }


def _compare_prepared_inputs(
        reference: dict[str, Any],
        candidate: dict[str, Any],
) -> dict[str, Any]:
    keys = sorted(set(reference) | set(candidate))
    comparisons = {
        key: _compare_values(reference.get(key), candidate.get(key))
        for key in keys
    }
    return {
        "match": all(item.get("match", False) for item in comparisons.values()),
        "keys": keys,
        "comparisons": comparisons,
    }


def _eos_token_ids(tokenizer, metadata: dict[str, Any]) -> list[int]:
    context_type = ContextType(metadata["context_type"])
    return [
        int(item)
        for item in get_eos_token_id(
            tokenizer,
            lookback_time=float(metadata["lookback_time"]),
            lookahead_time=float(metadata["lookahead_time"]),
            context_type=context_type,
        )
    ]


@torch.no_grad()
def verify_decode_prepare_oracle(
        args: InferenceConfig,
        *,
        sequence_index: int,
        max_new_tokens: int,
        q1_bmm_cross_attention: bool,
        native_q1_self_attention: bool,
        native_q1_rope_cache_self_attention: bool,
) -> dict[str, Any]:
    _assert_supported_probe(args)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

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
    model.eval()
    model_inputs = _move_kwargs_for_model(
        model,
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
        "config_name": None,
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
        "torch_version": torch.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k_sampling": args.top_k,
        "max_new_tokens": int(max_new_tokens),
        "q1_bmm_cross_attention": bool(q1_bmm_cross_attention),
        "native_q1_self_attention": bool(native_q1_self_attention),
        "native_q1_rope_cache_self_attention": bool(native_q1_rope_cache_self_attention),
    }

    input_ids = model_inputs["decoder_input_ids"]
    prompt_len = int(input_ids.shape[-1])
    prompt_mask = model_inputs["decoder_attention_mask"]
    condition_kwargs = _condition_kwargs(model_inputs)
    eos_token_ids = _eos_token_ids(tokenizer, metadata)
    metadata["eos_token_ids"] = eos_token_ids

    logits_processor = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
    model_kwargs: dict[str, Any] = {
        **condition_kwargs,
        "decoder_attention_mask": prompt_mask,
        "past_key_values": cache,
        "use_cache": True,
    }
    cur_len = int(input_ids.shape[-1])
    model_kwargs = model._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    initial_rng_state = _rng_state()
    step_reports: list[dict[str, Any]] = []
    generated_token_ids: list[int] = []
    do_sample = bool(args.do_sample)
    scores = None

    prepare_reference_wall_s = 0.0
    prepare_candidate_wall_s = 0.0
    _set_rng_state(initial_rng_state)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=q1_bmm_cross_attention,
                native_q1_self_attention=native_q1_self_attention,
                native_q1_rope_cache_self_attention=native_q1_rope_cache_self_attention,
            ):
        for step_index in range(max_new_tokens):
            start = time.perf_counter()
            reference_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            prepare_reference_wall_s += time.perf_counter() - start
            if step_index == 0 and reference_inputs.get("encoder_outputs") is None:
                reference_inputs["input_features"] = model_inputs["frames"]

            comparison = None
            candidate_error = None
            if step_index > 0:
                start = time.perf_counter()
                try:
                    candidate_inputs = prepare_one_token_decode_inputs_fast(model, input_ids, model_kwargs)
                    prepare_candidate_wall_s += time.perf_counter() - start
                    comparison = _compare_prepared_inputs(reference_inputs, candidate_inputs)
                except Exception as exc:
                    prepare_candidate_wall_s += time.perf_counter() - start
                    candidate_error = f"{type(exc).__name__}: {exc}"

            outputs = model(**reference_inputs, return_dict=True)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if model_kwargs.get("encoder_outputs") is None and outputs.encoder_last_hidden_state is not None:
                model_kwargs["encoder_outputs"] = BaseModelOutput(
                    last_hidden_state=outputs.encoder_last_hidden_state,
                )

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            generated_token_ids.extend(int(item) for item in next_tokens.tolist())

            step_reports.append({
                "step_index": int(step_index),
                "cur_len_before": int(cur_len),
                "checked_fast_prepare": step_index > 0,
                "candidate_error": candidate_error,
                "comparison": comparison,
                "generated_token_id": int(next_tokens[0].detach().cpu().item()),
            })
            cur_len += 1
            del outputs

    checked_reports = [item for item in step_reports if item["checked_fast_prepare"]]
    match = bool(checked_reports) and all(
        item["candidate_error"] is None
        and item["comparison"] is not None
        and item["comparison"]["match"]
        for item in checked_reports
    )
    return {
        "pass": match,
        "checked_decode_steps": len(checked_reports),
        "prompt_tokens": prompt_len,
        "generated_token_ids": generated_token_ids,
        "prepare_reference_wall_s": float(prepare_reference_wall_s),
        "prepare_candidate_wall_s": float(prepare_candidate_wall_s),
        "prepare_wall_note": "Diagnostic CPU wall only; not a throughput claim.",
        "step_reports": step_reports,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verifier-only oracle for fast one-token decode input preparation."
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--q1-bmm-cross-attention", action="store_true")
    parser.add_argument("--native-q1-self-attention", action="store_true")
    parser.add_argument("--native-q1-rope-cache-self-attention", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. inference_generation_compile=true")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = verify_decode_prepare_oracle(
        args,
        sequence_index=cli_args.sequence_index,
        max_new_tokens=cli_args.max_new_tokens,
        q1_bmm_cross_attention=cli_args.q1_bmm_cross_attention,
        native_q1_self_attention=cli_args.native_q1_self_attention,
        native_q1_rope_cache_self_attention=cli_args.native_q1_rope_cache_self_attention,
    )
    result["metadata"]["config_name"] = cli_args.config_name
    result["wall_seconds"] = time.perf_counter() - start

    if cli_args.report_path is not None:
        cli_args.report_path.parent.mkdir(parents=True, exist_ok=True)
        cli_args.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
