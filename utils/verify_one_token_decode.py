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

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers.modeling_outputs import BaseModelOutput

from config import InferenceConfig
from inference import (
    compile_args,
    get_config,
    load_model_with_server,
    setup_inference_environment,
    should_generate_timing_context,
)
from osuT5.osuT5.inference import Postprocessor, Preprocessor, Processor
from osuT5.osuT5.inference.cache_utils import get_cache
from osuT5.osuT5.runtime_profiling import generation_profile_context
from osuT5.osuT5.tokenizer import ContextType


def _load_args(config_name: str, overrides: list[str]) -> InferenceConfig:
    config_dir = REPO_ROOT / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _assert_supported_probe(args: InferenceConfig) -> None:
    if args.use_server:
        raise ValueError("one-token decode gate requires use_server=false")
    if args.parallel:
        raise ValueError("one-token decode gate requires parallel=false")
    if args.num_beams != 1:
        raise ValueError("one-token decode gate requires num_beams=1")
    if args.cfg_scale != 1.0:
        raise ValueError("one-token decode gate requires cfg_scale=1.0")


def _move_tensor_for_model(model, key: str, value: torch.Tensor) -> torch.Tensor:
    value = value.to(model.device)
    if key != "frames" and value.dtype == torch.float32:
        value = value.to(model.dtype)
    return value


def _move_kwargs_for_model(model, kwargs: dict[str, Any]) -> dict[str, Any]:
    moved = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = _move_tensor_for_model(model, key, value)
        else:
            moved[key] = value
    return moved


def _build_timing_context_if_needed(
        args: InferenceConfig,
        model,
        tokenizer,
        sequences,
        generation_config,
) -> tuple[list[ContextType], dict[ContextType, Any]]:
    output_type = args.output_type.copy()
    extra_in_context: dict[ContextType, Any] = {}
    if not should_generate_timing_context(args, output_type):
        return output_type, extra_in_context

    timing_processor = Processor(args, model, tokenizer)
    timing_events, _ = timing_processor.generate(
        sequences=sequences,
        generation_config=generation_config,
        in_context=[ContextType.NONE],
        out_context=[ContextType.TIMING],
        beatmap_path=args.beatmap_path,
        verbose=False,
        profile_label="timing_context_probe",
    )[0]
    timing = Postprocessor(args).generate_timing(timing_events)
    extra_in_context[ContextType.TIMING] = timing
    if ContextType.TIMING in output_type:
        output_type.remove(ContextType.TIMING)
    return output_type, extra_in_context


def _build_probe_inputs(
        args: InferenceConfig,
        model,
        tokenizer,
        *,
        sequence_index: int,
) -> dict[str, Any]:
    generation_config, _ = get_config(args)
    preprocessor = Preprocessor(args, parallel=False)
    audio = preprocessor.load(args.audio_path)
    sequences = preprocessor.segment(audio)
    if sequence_index < 0 or sequence_index >= len(sequences[0]):
        raise IndexError(f"sequence_index={sequence_index} out of range for {len(sequences[0])} sequences")

    output_type, extra_in_context = _build_timing_context_if_needed(
        args,
        model,
        tokenizer,
        sequences,
        generation_config,
    )

    processor = Processor(args, model, tokenizer)
    gen_in_context, gen_out_context, req_special_tokens = processor._get_viable_template(
        in_context=args.in_context,
        out_context=output_type,
        extra_in_context=extra_in_context,
        gamemode=generation_config.gamemode,
    )
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
    song_length = sequences[2]
    in_context_data = processor.get_in_context(
        in_context=gen_in_context,
        beatmap_path=args.beatmap_path,
        extra_in_context=extra_in_context,
        song_length=song_length,
    )
    out_context_data = processor.get_out_context(
        out_context=gen_out_context,
        generation_config=generation_config,
        given_context=args.in_context,
        beatmap_path=args.beatmap_path,
        extra_in_context=extra_in_context,
        song_length=song_length,
        verbose=False,
    )
    context_index, context = next(
        ((idx, item) for idx, item in enumerate(out_context_data) if not item["finished"]),
        (None, None),
    )
    if context is None or context_index is None:
        raise RuntimeError("No unfinished output context available for one-token probe")
    out_context_prefix = out_context_data[:context_index + 1]

    frames = processor.prepare_frames(sequences[0][sequence_index])
    frame_time = sequences[1][sequence_index].item()
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(in_context_data, frame_time, False, req_special_tokens),
        processor.prepare_context_sequences(out_context_prefix, frame_time, True, req_special_tokens),
    )
    if uncond_prompt is not None:
        raise RuntimeError("Unexpected unconditional prompt with cfg_scale=1.0")
    [prompt, _], _ = processor.pad_prompts([cond_prompt, uncond_prompt])

    if processor.do_song_position_embed:
        model_kwargs["song_position"] = torch.tensor(
            [frame_time / song_length, (frame_time + processor.miliseconds_per_sequence) / song_length],
            dtype=torch.float32,
        ).unsqueeze(0)

    return {
        "context_type": context["context_type"].value,
        "sequence_index": sequence_index,
        "frame_time_ms": float(frame_time),
        "frames": frames,
        "decoder_input_ids": prompt,
        "decoder_attention_mask": prompt.ne(tokenizer.pad_id),
        **model_kwargs,
    }


def _last_token_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, :].detach().to(torch.float32)


@torch.no_grad()
def run_one_token_gate(
        args: InferenceConfig,
        *,
        sequence_index: int,
        probe_token_id: int | None,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
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
        "config_name": None,
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "atol": atol,
        "rtol": rtol,
        "top_k": top_k,
    }

    prompt = model_inputs["decoder_input_ids"]
    prompt_mask = model_inputs["decoder_attention_mask"]
    prompt_len = int(prompt.shape[-1])

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "amp"), \
            generation_profile_context(sdpa_backend=args.profile_sdpa_backend):
        prompt_outputs = model(
            **model_inputs,
            use_cache=False,
        )
        next_token_logits = _last_token_logits(prompt_outputs.logits)
        if probe_token_id is None:
            probe_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).to(torch.long)
        else:
            probe_token = torch.full((prompt.shape[0], 1), int(probe_token_id), dtype=torch.long, device=prompt.device)

        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)
        reference_outputs = model(
            **{**model_inputs, "decoder_input_ids": full_prefix, "decoder_attention_mask": full_mask},
            use_cache=False,
        )
        reference_logits = _last_token_logits(reference_outputs.logits)

        cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
        prefill_outputs = model(
            **model_inputs,
            use_cache=True,
            past_key_values=cache,
            cache_position=torch.arange(prompt_len, device=prompt.device),
        )
        encoder_outputs = BaseModelOutput(last_hidden_state=prefill_outputs.encoder_last_hidden_state)
        candidate_inputs = model.prepare_inputs_for_generation(
            full_prefix,
            past_key_values=cache,
            use_cache=True,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=full_mask,
            cache_position=torch.arange(prompt_len, prompt_len + 1, device=prompt.device),
            **{
                key: value
                for key, value in model_inputs.items()
                if key not in {"frames", "decoder_input_ids", "decoder_attention_mask"}
            },
        )
        candidate_outputs = model(**candidate_inputs)
        candidate_logits = _last_token_logits(candidate_outputs.logits)

    abs_diff = torch.abs(reference_logits - candidate_logits)
    rel_diff = abs_diff / torch.clamp(torch.abs(reference_logits), min=1e-12)
    reference_topk = torch.topk(reference_logits, k=min(top_k, reference_logits.shape[-1]), dim=-1).indices
    candidate_topk = torch.topk(candidate_logits, k=min(top_k, candidate_logits.shape[-1]), dim=-1).indices
    allclose = bool(torch.allclose(reference_logits, candidate_logits, atol=atol, rtol=rtol))
    topk_match = bool(torch.equal(reference_topk, candidate_topk))

    return {
        "pass": allclose and topk_match,
        "allclose": allclose,
        "topk_match": topk_match,
        "probe_token_id": int(probe_token.item()),
        "prompt_tokens": prompt_len,
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float(rel_diff.max().item()),
        "reference_topk": [int(item) for item in reference_topk[0].tolist()],
        "candidate_topk": [int(item) for item in candidate_topk[0].tolist()],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a real static-cache q_len=1 decode step matches full-prefix raw logits."
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--probe-token-id", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model_path=/path/to/model")
    cli_args = parser.parse_args()

    start = time.perf_counter()
    args = _load_args(cli_args.config_name, cli_args.overrides)
    result = run_one_token_gate(
        args,
        sequence_index=cli_args.sequence_index,
        probe_token_id=cli_args.probe_token_id,
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
