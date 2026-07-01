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

from inference import compile_args, load_model_with_server, setup_inference_environment
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.inference.direct_decode import (
    decode_one_token_raw_logits,
    last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import get_eos_token_id
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType
from utils.verify_one_token_decode import (
    _CaptureRawLogitsProcessor,
    _assert_supported_probe,
    _build_generation_logits_processors,
    _build_probe_inputs,
    _compare_logits,
    _condition_kwargs,
    _load_args,
    _move_kwargs_for_model,
)


def _cuda_event_time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


@torch.no_grad()
def profile_direct_step(
        args,
        *,
        sequence_index: int,
        warmup: int,
        iters: int,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Direct decode step profiling requires CUDA")
    _assert_supported_probe(args)
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
        _build_probe_inputs(args, model, tokenizer, sequence_index=sequence_index),
    )
    metadata = {
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
        "warmup": warmup,
        "iters": iters,
    }
    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])
    condition_kwargs = _condition_kwargs(model_inputs)
    logits_processors = _build_generation_logits_processors(
        args,
        tokenizer,
        model.device,
        lookback_time=float(metadata["lookback_time"]),
    )
    hf_raw_logit_captures: list[torch.Tensor] = []
    logits_processors.insert(0, _CaptureRawLogitsProcessor(hf_raw_logit_captures))
    context_type = ContextType(metadata["context_type"])
    eos_token_ids = get_eos_token_id(
        tokenizer,
        lookback_time=float(metadata["lookback_time"]),
        lookahead_time=float(metadata["lookahead_time"]),
        context_type=context_type,
    )

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        hf_cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        hf_generate_outputs = model.generate(
            inputs=model_inputs["frames"],
            decoder_input_ids=prompt,
            decoder_attention_mask=prompt_mask,
            **condition_kwargs,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=2,
            use_cache=True,
            past_key_values=hf_cache,
            logits_processor=logits_processors,
            eos_token_id=eos_token_ids,
            return_dict_in_generate=True,
            output_logits=True,
        )
        if len(hf_raw_logit_captures) < 2:
            raise RuntimeError(f"HF generate captured {len(hf_raw_logit_captures)} raw-logit steps; expected at least 2")
        probe_token = hf_generate_outputs.sequences[:, prompt_len:prompt_len + 1].to(torch.long)
        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)

        state = prefill_static_cache(
            model,
            prompt=prompt,
            prompt_attention_mask=prompt_mask,
            frames=model_inputs["frames"],
            condition_kwargs=condition_kwargs,
        )
        direct_result = decode_one_token_raw_logits(
            model,
            state,
            full_prefix=full_prefix,
            full_attention_mask=full_mask,
            condition_kwargs=condition_kwargs,
        )
        prepared_inputs = direct_result.prepared_inputs
        reference_logits = direct_result.logits.detach().clone()

        def eager_step() -> torch.Tensor:
            return last_token_logits(model(**prepared_inputs).logits)

        eager_logits = eager_step()
        eager_comparison = _compare_logits(reference_logits, eager_logits, atol=atol, rtol=rtol, top_k=top_k)
        eager_ms = _cuda_event_time_ms(eager_step, warmup=warmup, iters=iters)

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(warmup):
                graph_outputs = model(**prepared_inputs)
        torch.cuda.current_stream().wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = model(**prepared_inputs)
        graph.replay()
        graph_logits = last_token_logits(graph_outputs.logits)
        graph_comparison = _compare_logits(reference_logits, graph_logits, atol=atol, rtol=rtol, top_k=top_k)

        def graph_step() -> None:
            graph.replay()

        graph_ms = _cuda_event_time_ms(graph_step, warmup=warmup, iters=iters)

    return {
        "pass": eager_comparison["allclose"] and eager_comparison["topk_match"]
        and graph_comparison["allclose"] and graph_comparison["topk_match"],
        "prompt_tokens": prompt_len,
        "probe_token_id": int(probe_token.item()),
        "prepared_candidate_shape": list(prepared_inputs["decoder_input_ids"].shape),
        "prefill_cache_position": [int(item) for item in state.prefill_cache_position.tolist()],
        "decode_cache_position": [int(item) for item in direct_result.cache_position.tolist()],
        "eager_ms_per_step": eager_ms,
        "graph_ms_per_step": graph_ms,
        "graph_speedup": eager_ms / graph_ms if graph_ms > 0 else None,
        "eager_comparison": eager_comparison,
        "graph_comparison": graph_comparison,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a fixed one-token direct decode step with eager CUDA launches and manual CUDA graph replay."
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = profile_direct_step(
        args,
        sequence_index=cli_args.sequence_index,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
        atol=cli_args.atol,
        rtol=cli_args.rtol,
        top_k=cli_args.top_k,
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
