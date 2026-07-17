#!/usr/bin/env python3
"""§42 diagnostic: isolate draft fastpath CUDA failures on 2080 Ti."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

REPO = Path(os.environ.get("PROBE_REPO", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(REPO))
__import__("config")

from utils.s42_draft_fastpath_bench import (  # noqa: E402
    _ensure_encoder_outputs,
    build_first_map_window_kwargs,
    find_osu,
    load_args,
)
from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.inference.turbo.draft_fastpath import DraftFastpathRunner  # noqa: E402
from osuT5.osuT5.runtime_profiling import generation_profile_context  # noqa: E402
from osuT5.osuT5.inference.optimized.single.runtime_context import (  # noqa: E402
    active_prefix_self_attention_context,
)
from osuT5.osuT5.inference.optimized.single.decode_loop import (  # noqa: E402
    _bucketed_prefix_length,
)


def _cache_info(cache) -> dict[str, Any]:
    self_c = cache.self_attention_cache
    n = len(getattr(self_c, "layers", []) or [])
    if n == 0 and hasattr(self_c, "key_cache"):
        n = len(self_c.key_cache)
    return {
        "self_layers": n,
        "cross_layers": len(getattr(cache.cross_attention_cache, "layers", []) or [])
        or len(getattr(cache.cross_attention_cache, "key_cache", []) or []),
        "is_updated": dict(getattr(cache, "is_updated", {})),
    }


def try_case(name: str, fn) -> dict[str, Any]:
    row: dict[str, Any] = {"case": name, "ok": False}
    try:
        torch.cuda.synchronize()
        out = fn()
        torch.cuda.synchronize()
        row["ok"] = True
        if isinstance(out, dict):
            row.update(out)
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)
        row["traceback"] = traceback.format_exc()[-2000:]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--draft-ckpt", required=True)
    ap.add_argument("--osu-hint", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--precision", default="fp16")
    args = ap.parse_args()

    os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    overrides = [
        f"precision={args.precision}",
        "device=cuda",
        "attn_implementation=sdpa",
        "inference_engine=turbo",
        "use_server=false",
        "parallel=false",
        f"audio_path={args.audio}",
    ]
    args_inf = load_args("profile_salvalai", overrides)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

    bound, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision=args.precision,
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="turbo",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    teacher, runtime = unwrap_engine_binding(bound)
    draft = runtime.draft
    teacher.eval()
    draft.eval()

    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, teacher, tokenizer)
    audio = preprocessor.load(str(args.audio))
    sequences = preprocessor.segment(audio)
    song_length = sequences[2]
    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    osu_path = args.osu_hint or find_osu(
        Path("/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133")
    )
    model_kwargs, meta = build_first_map_window_kwargs(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )
    prompt_ids = model_kwargs["decoder_input_ids"].to(teacher.device)
    teacher_mk = _ensure_encoder_outputs(teacher, model_kwargs)
    enc = teacher_mk["encoder_outputs"]
    bind_kwargs = {
        k: v
        for k, v in teacher_mk.items()
        if k
        not in {
            "past_key_values",
            "cache_position",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
    }
    bind_kwargs["encoder_outputs"] = enc

    report: dict[str, Any] = {
        "prompt_len": int(prompt_ids.shape[1]),
        "meta": meta,
        "draft_layers": len(draft.transformer.model.decoder.layers),
        "teacher_layers": len(teacher.transformer.model.decoder.layers),
        "cases": [],
    }

    # Case A: draft eager SDPA only
    def case_sdpa():
        r = DraftFastpathRunner(
            draft, dtype=dtype, enable_native_kernels=False, cuda_graph=False
        )
        r.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
        logits = r.prefill()
        logits2 = r.decode_token(1)
        return {
            "cache": _cache_info(r._cache),
            "logits_finite": bool(torch.isfinite(logits).all() and torch.isfinite(logits2).all()),
        }

    report["cases"].append(try_case("draft_eager_sdpa", case_sdpa))

    # Case B: incremental native flag sets
    flag_sets = [
        ("q1_bmm_only", dict(q1_bmm_cross_attention=True)),
        ("q1_self", dict(q1_bmm_cross_attention=True, native_q1_self_attention=True)),
        (
            "q1_rope",
            dict(
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=True,
            ),
        ),
        (
            "all_native",
            dict(
                q1_bmm_cross_attention=True,
                native_q1_self_attention=True,
                native_q1_rope_cache_self_attention=True,
                native_cross_mlp_tail=True,
            ),
        ),
    ]

    for name, flags in flag_sets:
        def case(flags=flags):
            r = DraftFastpathRunner(
                draft, dtype=dtype, enable_native_kernels=False, cuda_graph=False
            )
            r.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
            r.prefill()
            # Override profile context for one decode with selected flags.
            tok = torch.tensor([[1]], device=prompt_ids.device, dtype=prompt_ids.dtype)
            ids = torch.cat([r._input_ids, tok], dim=-1)
            r._model_kwargs.pop("cache_position", None)
            model_inputs = r.model.prepare_inputs_for_generation(ids, **r._model_kwargs)
            shapes = {
                k: list(v.shape)
                for k, v in model_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            bucket = _bucketed_prefix_length(
                int(ids.shape[1]), r.bucket_size, r._max_cache_len
            )
            with generation_profile_context(
                **flags,
                optimized_expected_dtype=dtype,
            ), active_prefix_self_attention_context(bucket):
                out = r.model(**model_inputs, return_dict=True)
            return {
                "shapes": shapes,
                "bucket": bucket,
                "cache": _cache_info(r._cache),
                "logits_finite": bool(torch.isfinite(out.logits).all()),
                "flags": flags,
            }

        report["cases"].append(try_case(f"draft_flags_{name}", case))
        if not report["cases"][-1]["ok"]:
            # Device may be poisoned; stop further GPU cases.
            break

    # Case C: teacher all-native (control)
    if report["cases"][-1]["ok"]:
        def case_teacher():
            r = DraftFastpathRunner(
                teacher, dtype=dtype, enable_native_kernels=True, cuda_graph=False
            )
            r.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
            r.prefill()
            r.decode_token(1)
            return {"cache": _cache_info(r._cache), "layers": r.layer_count}

        report["cases"].append(try_case("teacher_eager_all_native", case_teacher))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["cases"] and report["cases"][0]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
