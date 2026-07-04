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
import transformers
from omegaconf import DictConfig, OmegaConf
from transformers import LogitsProcessor
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
from osuT5.osuT5.inference.direct_decode import (
    DecodeSession,
    decode_one_token_raw_logits,
    last_token_logits as _last_token_logits,
    prefill_static_cache,
)
from osuT5.osuT5.inference.server import build_logits_processor_list, get_eos_token_id
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
        "lookback_time": processor.lookback_time if (
            sequence_index != 0 and processor.types_first and processor.lookback_time > 0
        ) else 0,
        "lookahead_time": processor.lookahead_time if sequence_index != len(sequences[0]) - 1 else 0,
        "frames": frames,
        "decoder_input_ids": prompt,
        "decoder_attention_mask": prompt.ne(tokenizer.pad_id),
        **model_kwargs,
    }


def _compare_logits(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        atol: float,
        rtol: float,
        top_k: int,
) -> dict[str, Any]:
    reference = reference.detach().to(torch.float32)
    candidate = candidate.detach().to(torch.float32)
    finite_mask = torch.isfinite(reference) & torch.isfinite(candidate)
    reference_nonfinite = ~torch.isfinite(reference)
    candidate_nonfinite = ~torch.isfinite(candidate)
    nonfinite_match = bool(torch.equal(reference_nonfinite, candidate_nonfinite))
    positive_inf_match = bool(torch.equal(torch.isposinf(reference), torch.isposinf(candidate)))
    negative_inf_match = bool(torch.equal(torch.isneginf(reference), torch.isneginf(candidate)))
    finite_reference = reference[finite_mask]
    finite_candidate = candidate[finite_mask]
    finite_allclose = bool(torch.allclose(finite_reference, finite_candidate, atol=atol, rtol=rtol))
    abs_diff = torch.abs(finite_reference - finite_candidate)
    rel_diff = abs_diff / torch.clamp(torch.abs(finite_reference), min=1e-12)
    reference_topk = torch.topk(reference, k=min(top_k, reference.shape[-1]), dim=-1).indices
    candidate_topk = torch.topk(candidate, k=min(top_k, candidate.shape[-1]), dim=-1).indices
    topk_match = bool(torch.equal(reference_topk, candidate_topk))
    return {
        "allclose": finite_allclose and nonfinite_match and positive_inf_match and negative_inf_match,
        "finite_allclose": finite_allclose,
        "nonfinite_match": nonfinite_match,
        "positive_inf_match": positive_inf_match,
        "negative_inf_match": negative_inf_match,
        "topk_match": topk_match,
        "finite_count": int(finite_mask.sum().item()),
        "nonfinite_mismatch_count": int((reference_nonfinite != candidate_nonfinite).sum().item()),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() > 0 else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() > 0 else 0.0,
        "max_rel": float(rel_diff.max().item()) if rel_diff.numel() > 0 else 0.0,
        "reference_topk": [int(item) for item in reference_topk[0].tolist()],
        "candidate_topk": [int(item) for item in candidate_topk[0].tolist()],
    }


def _condition_kwargs(model_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_inputs.items()
        if key not in {"frames", "decoder_input_ids", "decoder_attention_mask"}
    }


def _build_generation_logits_processors(args: InferenceConfig, tokenizer, device, *, lookback_time: float):
    return build_logits_processor_list(
        tokenizer,
        cfg_scale=args.cfg_scale,
        timeshift_bias=args.timeshift_bias,
        types_first=args.train.data.types_first,
        temperature=args.temperature,
        timing_temperature=args.timing_temperature,
        mania_column_temperature=args.mania_column_temperature,
        taiko_hit_temperature=args.taiko_hit_temperature,
        lookback_time=lookback_time,
        device=device,
        stateful_monotonic=bool(getattr(args, "inference_stateful_monotonic_logits_processor", False)),
    )


class _CaptureRawLogitsProcessor(LogitsProcessor):
    def __init__(self, captures: list[torch.Tensor]):
        self.captures = captures

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        self.captures.append(scores.detach().clone().to(torch.float32))
        return scores


@torch.no_grad()
def run_one_token_gate(
        args: InferenceConfig,
        *,
        sequence_index: int,
        probe_token_id: int | None,
        atol: float,
        rtol: float,
        top_k: int,
        include_no_cache_diagnostics: bool = False,
        candidate_active_prefix_self_attention: bool = False,
        candidate_active_prefix_prefill: bool | None = None,
        candidate_active_prefix_decode: bool | None = None,
        candidate_active_prefix_decode_length: int | None = None,
        candidate_q1_bmm_cross_attention: bool = False,
        candidate_native_q1_self_attention: bool = False,
        candidate_native_q1_rope_cache_self_attention: bool = False,
        candidate_native_decoder_layer_mlp_tail: bool = False,
        candidate_native_decoder_layer_mlp_tail_outputs_per_block: int = 4,
        candidate_decode_session: bool = False,
) -> dict[str, Any]:
    _assert_supported_probe(args)
    if probe_token_id is not None:
        raise ValueError("one-token HF-generate logits gate does not support --probe-token-id")
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
    active_prefix_prefill = (
        bool(candidate_active_prefix_self_attention)
        if candidate_active_prefix_prefill is None
        else bool(candidate_active_prefix_prefill)
    )
    active_prefix_decode = (
        bool(candidate_active_prefix_self_attention)
        if candidate_active_prefix_decode is None
        else bool(candidate_active_prefix_decode)
    )
    if candidate_active_prefix_decode_length is not None and not active_prefix_decode:
        raise ValueError("--candidate-active-prefix-decode-length requires active-prefix decode to be enabled")
    if candidate_native_decoder_layer_mlp_tail:
        if not candidate_decode_session:
            raise ValueError("--candidate-native-decoder-layer-mlp-tail requires --candidate-decode-session")
        if not active_prefix_decode:
            raise ValueError("--candidate-native-decoder-layer-mlp-tail requires active-prefix candidate decode")
        if not candidate_q1_bmm_cross_attention:
            raise ValueError("--candidate-native-decoder-layer-mlp-tail requires --candidate-q1-bmm-cross-attention")
        if not candidate_native_q1_rope_cache_self_attention:
            raise ValueError(
                "--candidate-native-decoder-layer-mlp-tail requires "
                "--candidate-native-q1-rope-cache-self-attention"
            )

    metadata = {
        "config_name": None,
        "model_path": args.model_path,
        "precision": args.precision,
        "attn_implementation": args.attn_implementation,
        "profile_sdpa_backend": args.profile_sdpa_backend,
        "generation_compile_enabled": not bool(getattr(getattr(model, "generation_config", None), "disable_compile", True)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sequence_index": model_inputs.pop("sequence_index"),
        "frame_time_ms": model_inputs.pop("frame_time_ms"),
        "context_type": model_inputs.pop("context_type"),
        "lookback_time": model_inputs.pop("lookback_time"),
        "lookahead_time": model_inputs.pop("lookahead_time"),
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k_sampling": args.top_k,
        "temperature": args.temperature,
        "timing_temperature": args.timing_temperature,
        "mania_column_temperature": args.mania_column_temperature,
        "taiko_hit_temperature": args.taiko_hit_temperature,
        "timeshift_bias": args.timeshift_bias,
        "cfg_scale": args.cfg_scale,
        "num_beams": args.num_beams,
        "atol": atol,
        "rtol": rtol,
        "top_k": top_k,
        "include_no_cache_diagnostics": bool(include_no_cache_diagnostics),
        "candidate_active_prefix_self_attention": bool(candidate_active_prefix_self_attention),
        "candidate_active_prefix_prefill": active_prefix_prefill,
        "candidate_active_prefix_decode": active_prefix_decode,
        "candidate_active_prefix_decode_length": candidate_active_prefix_decode_length,
        "candidate_q1_bmm_cross_attention": bool(candidate_q1_bmm_cross_attention),
        "candidate_native_q1_self_attention": bool(candidate_native_q1_self_attention),
        "candidate_native_q1_rope_cache_self_attention": bool(candidate_native_q1_rope_cache_self_attention),
        "candidate_native_decoder_layer_mlp_tail": bool(candidate_native_decoder_layer_mlp_tail),
        "candidate_native_decoder_layer_mlp_tail_outputs_per_block": (
            int(candidate_native_decoder_layer_mlp_tail_outputs_per_block)
        ),
        "candidate_decode_session": bool(candidate_decode_session),
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
    metadata["eos_token_ids"] = [int(item) for item in eos_token_ids]

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
            raise RuntimeError(
                "HF generate did not execute at least two captured raw-logit steps; "
                f"got {len(hf_raw_logit_captures)}"
            )
        hf_logits = getattr(hf_generate_outputs, "logits", None)
        if hf_logits is None or len(hf_logits) < 2:
            raise RuntimeError(
                "HF generate did not return at least two raw-logit steps; "
                f"got {0 if hf_logits is None else len(hf_logits)}"
            )
        probe_token = hf_generate_outputs.sequences[:, prompt_len:prompt_len + 1].to(torch.long)
        if probe_token.shape[-1] != 1:
            raise RuntimeError(
                f"HF generate did not produce a first generated token at prompt_len={prompt_len}; "
                f"sequence shape={list(hf_generate_outputs.sequences.shape)}"
            )
        if not torch.equal(hf_generate_outputs.sequences[:, :prompt_len], prompt):
            raise RuntimeError("HF generate output prefix does not match the probe prompt")
        hf_reference_logits = hf_raw_logit_captures[1].detach().to(torch.float32)
        hf_output_logits_reference = hf_logits[1].detach().to(torch.float32)

        full_prefix = torch.cat([prompt, probe_token], dim=-1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(probe_token, dtype=prompt_mask.dtype)], dim=-1)
        prompt_prepared = None
        reference_prepared = None
        no_cache_reference_logits = None
        no_cache_first_step_logits = None
        if include_no_cache_diagnostics:
            prompt_prepared = model.prepare_inputs_for_generation(
                prompt,
                use_cache=False,
                decoder_attention_mask=prompt_mask,
                frames=model_inputs["frames"],
                **condition_kwargs,
            )
            prompt_outputs = model(
                **prompt_prepared,
            )
            no_cache_first_step_logits = _last_token_logits(prompt_outputs.logits)

            encoder_outputs = BaseModelOutput(last_hidden_state=prompt_outputs.encoder_last_hidden_state)
            reference_prepared = model.prepare_inputs_for_generation(
                full_prefix,
                use_cache=False,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=full_mask,
                **condition_kwargs,
            )
            reference_outputs = model(
                **reference_prepared,
            )
            no_cache_reference_logits = _last_token_logits(reference_outputs.logits)

        with generation_profile_context(
                sdpa_backend=args.profile_sdpa_backend,
                q1_bmm_cross_attention=candidate_q1_bmm_cross_attention,
                native_q1_self_attention=candidate_native_q1_self_attention,
                native_q1_rope_cache_self_attention=candidate_native_q1_rope_cache_self_attention,
                native_decoder_layer_mlp_tail=candidate_native_decoder_layer_mlp_tail,
                native_decoder_layer_mlp_tail_outputs_per_block=(
                    candidate_native_decoder_layer_mlp_tail_outputs_per_block
                ),
        ):
            decode_session_metadata = None
            if candidate_decode_session:
                decode_session = DecodeSession.prefill(
                    model,
                    prompt=prompt,
                    prompt_attention_mask=prompt_mask,
                    frames=model_inputs["frames"],
                    condition_kwargs=condition_kwargs,
                    active_prefix_self_attention=active_prefix_prefill,
                )
                candidate_result = decode_session.decode_one_token_raw_logits(
                    full_prefix=full_prefix,
                    full_attention_mask=full_mask,
                    active_prefix_self_attention=active_prefix_decode,
                    active_prefix_self_attention_length=candidate_active_prefix_decode_length,
                )
                decode_state = decode_session.one_token_state()
                decode_session_metadata = decode_session.metadata()
            else:
                decode_state = prefill_static_cache(
                    model,
                    prompt=prompt,
                    prompt_attention_mask=prompt_mask,
                    frames=model_inputs["frames"],
                    condition_kwargs=condition_kwargs,
                    active_prefix_self_attention=active_prefix_prefill,
                )
                candidate_result = decode_one_token_raw_logits(
                    model,
                    decode_state,
                    full_prefix=full_prefix,
                    full_attention_mask=full_mask,
                    condition_kwargs=condition_kwargs,
                    active_prefix_self_attention=active_prefix_decode,
                    active_prefix_self_attention_length=candidate_active_prefix_decode_length,
                )
        candidate_inputs = candidate_result.prepared_inputs
        candidate_logits = candidate_result.logits

    comparison = _compare_logits(hf_reference_logits, candidate_logits, atol=atol, rtol=rtol, top_k=top_k)
    output_logits_comparison = _compare_logits(
        hf_output_logits_reference,
        candidate_logits,
        atol=atol,
        rtol=rtol,
        top_k=top_k,
    )
    no_cache_abs_diff = (
        torch.abs(no_cache_reference_logits - candidate_logits)
        if no_cache_reference_logits is not None
        else None
    )
    no_cache_first_step_abs_diff = (
        torch.abs(no_cache_first_step_logits - hf_raw_logit_captures[0])
        if no_cache_first_step_logits is not None
        else None
    )

    return {
        "pass": comparison["allclose"] and comparison["topk_match"],
        "allclose": comparison["allclose"],
        "topk_match": comparison["topk_match"],
        "probe_token_id": int(probe_token.item()),
        "prompt_tokens": prompt_len,
        "prompt_nonpad_tokens": int(prompt_mask.to(torch.long).sum().item()),
        "prefill_cache_position": [int(item) for item in decode_state.prefill_cache_position.tolist()],
        "decode_cache_position": [int(item) for item in candidate_result.cache_position.tolist()],
        "hf_generated_sequence_shape": list(hf_generate_outputs.sequences.shape),
        "hf_generate_logits_steps": len(hf_logits),
        "hf_captured_raw_logit_steps": len(hf_raw_logit_captures),
        "hf_logit_shape_step0": list(hf_raw_logit_captures[0].shape),
        "hf_logit_shape_step1": list(hf_raw_logit_captures[1].shape),
        "prepared_prompt_shape": (
            list(prompt_prepared["decoder_input_ids"].shape)
            if prompt_prepared is not None
            else None
        ),
        "prepared_reference_shape": (
            list(reference_prepared["decoder_input_ids"].shape)
            if reference_prepared is not None
            else None
        ),
        "prepared_prefill_shape": list(decode_state.prefill_prepared_inputs["decoder_input_ids"].shape),
        "prepared_candidate_shape": list(candidate_inputs["decoder_input_ids"].shape),
        "decode_session": decode_session_metadata,
        "finite_allclose": comparison["finite_allclose"],
        "nonfinite_match": comparison["nonfinite_match"],
        "positive_inf_match": comparison["positive_inf_match"],
        "negative_inf_match": comparison["negative_inf_match"],
        "finite_count": comparison["finite_count"],
        "nonfinite_mismatch_count": comparison["nonfinite_mismatch_count"],
        "max_abs": comparison["max_abs"],
        "mean_abs": comparison["mean_abs"],
        "max_rel": comparison["max_rel"],
        "hf_output_logits_vs_candidate": {
            "allclose": output_logits_comparison["allclose"],
            "topk_match": output_logits_comparison["topk_match"],
            "finite_allclose": output_logits_comparison["finite_allclose"],
            "nonfinite_match": output_logits_comparison["nonfinite_match"],
            "nonfinite_mismatch_count": output_logits_comparison["nonfinite_mismatch_count"],
            "max_abs": output_logits_comparison["max_abs"],
            "mean_abs": output_logits_comparison["mean_abs"],
            "max_rel": output_logits_comparison["max_rel"],
        },
        "no_cache_reference_max_abs": (
            float(no_cache_abs_diff.max().item())
            if no_cache_abs_diff is not None
            else None
        ),
        "no_cache_reference_mean_abs": (
            float(no_cache_abs_diff.mean().item())
            if no_cache_abs_diff is not None
            else None
        ),
        "no_cache_first_step_vs_hf_max_abs": (
            float(no_cache_first_step_abs_diff.max().item())
            if no_cache_first_step_abs_diff is not None
            else None
        ),
        "no_cache_first_step_vs_hf_mean_abs": (
            float(no_cache_first_step_abs_diff.mean().item())
            if no_cache_first_step_abs_diff is not None
            else None
        ),
        "reference_topk": comparison["reference_topk"],
        "candidate_topk": comparison["candidate_topk"],
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a real static-cache q_len=1 decode step matches HF generate cached raw logits."
    )
    parser.add_argument("--config-name", default="profile_salvalai_smoke15")
    parser.add_argument("--sequence-index", type=int, default=9)
    parser.add_argument("--probe-token-id", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--include-no-cache-diagnostics",
        action="store_true",
        help="Also run slower no-cache full-prefix comparisons for debugging. Not required for the gate pass.",
    )
    parser.add_argument(
        "--candidate-active-prefix-self-attention",
        action="store_true",
        help="Enable active-prefix SDPA self-attention for both direct candidate prefill and decode.",
    )
    parser.add_argument(
        "--candidate-active-prefix-prefill",
        action="store_true",
        help="Enable active-prefix SDPA self-attention only while prefilling the direct candidate cache.",
    )
    parser.add_argument(
        "--candidate-active-prefix-decode",
        action="store_true",
        help="Enable active-prefix SDPA self-attention only for the direct candidate one-token decode step.",
    )
    parser.add_argument(
        "--candidate-active-prefix-decode-length",
        type=int,
        default=None,
        help="Use this fixed active-prefix K/V length during candidate decode, e.g. a graph-reusable bucket.",
    )
    parser.add_argument(
        "--candidate-q1-bmm-cross-attention",
        action="store_true",
        help="Enable the experimental fp32 q_len=1 BMM cross-attention candidate for direct decode.",
    )
    parser.add_argument(
        "--candidate-native-q1-self-attention",
        action="store_true",
        help="Enable the experimental native fp32 q_len=1 self-attention candidate for direct decode.",
    )
    parser.add_argument(
        "--candidate-native-q1-rope-cache-self-attention",
        action="store_true",
        help="Enable the experimental fused RoPE/cache native q_len=1 self-attention candidate.",
    )
    parser.add_argument(
        "--candidate-native-decoder-layer-mlp-tail",
        action="store_true",
        help="Enable the experimental native fp32 decoder-layer MLP tail candidate.",
    )
    parser.add_argument(
        "--candidate-native-decoder-layer-mlp-tail-outputs-per-block",
        type=int,
        default=4,
        choices=(2, 4, 8),
        help="Warp-group width for the experimental native decoder-layer MLP tail.",
    )
    parser.add_argument(
        "--candidate-decode-session",
        action="store_true",
        help="Route the direct candidate through the verifier-first DecodeSession ABI.",
    )
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
        include_no_cache_diagnostics=cli_args.include_no_cache_diagnostics,
        candidate_active_prefix_self_attention=cli_args.candidate_active_prefix_self_attention,
        candidate_active_prefix_prefill=(
            cli_args.candidate_active_prefix_prefill
            or cli_args.candidate_active_prefix_self_attention
        ),
        candidate_active_prefix_decode=(
            cli_args.candidate_active_prefix_decode
            or cli_args.candidate_active_prefix_self_attention
            or cli_args.candidate_active_prefix_decode_length is not None
        ),
        candidate_active_prefix_decode_length=cli_args.candidate_active_prefix_decode_length,
        candidate_q1_bmm_cross_attention=cli_args.candidate_q1_bmm_cross_attention,
        candidate_native_q1_self_attention=cli_args.candidate_native_q1_self_attention,
        candidate_native_q1_rope_cache_self_attention=cli_args.candidate_native_q1_rope_cache_self_attention,
        candidate_native_decoder_layer_mlp_tail=cli_args.candidate_native_decoder_layer_mlp_tail,
        candidate_native_decoder_layer_mlp_tail_outputs_per_block=(
            cli_args.candidate_native_decoder_layer_mlp_tail_outputs_per_block
        ),
        candidate_decode_session=cli_args.candidate_decode_session,
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
