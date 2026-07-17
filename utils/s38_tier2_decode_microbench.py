#!/usr/bin/env python3
"""§38 decode microbench: tip optimized vs turbo 7-stage fused step.

Measures synchronized GPU ms/token on short generate windows (not a song-wall
TPS claim). Budget gate: turbo savings ≥0.15 ms/token vs tip before reciprocal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402

BUDGET_MS = 0.15


def load_engine(args_inf, engine: str, precision: str):
    model, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision=precision,
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine=engine,
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    raw, _ = unwrap_engine_binding(model)
    raw.eval()
    return model, tokenizer


@torch.no_grad()
def time_forward_decode_proxy(
    model,
    *,
    frames: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    model_kwargs: dict,
    precision: str,
    warmup: int,
    iters: int,
) -> float:
    """Proxy microbench: timed full teacher-forced forwards / token count.

    Not identical to one-token decode, but exercises the fused layer path under
    turbo and provides a first-order ms/token delta vs tip.
    """
    raw, _ = unwrap_engine_binding(model)
    kw = {
        k: (v.to(raw.device) if isinstance(v, torch.Tensor) else v)
        for k, v in dict(model_kwargs).items()
    }
    for k, v in list(kw.items()):
        if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            kw[k] = v.to(raw.dtype)
    frames = frames.to(raw.device)
    dec = decoder_input_ids.to(raw.device)
    attn = decoder_attention_mask.to(raw.device)
    n_tok = int(dec.shape[1])

    def _once():
        with torch.autocast(
            device_type=raw.device.type,
            dtype=torch.bfloat16,
            enabled=(precision == "amp"),
        ):
            return raw.forward(
                frames=frames,
                decoder_input_ids=dec,
                decoder_attention_mask=attn,
                use_cache=False,
                **kw,
            )

    for _ in range(warmup):
        _ = _once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = _once()
    end.record()
    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    return total_ms / (iters * max(n_tok, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        ),
    )
    ap.add_argument(
        "--audio",
        type=Path,
        default=Path("/work/imt11/Mapperatorinator/data/salvalai.mp3"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=256, help="cap decoder length")
    args = ap.parse_args()

    os.chdir(args.repo)
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))

    from utils.s38_tier2_logit_agreement_probe import (
        find_osu,
        find_profile,
        load_dump_tokens,
        resolve_inference_config,
        teacher_force_logits,
    )
    from osuT5.osuT5.inference import Preprocessor, Processor
    from osuT5.osuT5.tokenizer import ContextType
    from inference import GenerationConfig
    from slider import Beatmap

    profile_path = find_profile(args.dump_run)
    dumps = load_dump_tokens(profile_path)
    config_dir = str(args.repo / "configs" / "inference")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"audio_path={args.audio}",
                "device=cuda",
                f"precision={args.precision}",
                "attn_implementation=sdpa",
                "inference_engine=optimized",
                "use_server=false",
                "parallel=false",
                "cfg_scale=1.0",
                "num_beams=1",
                "seed=12345",
                "profile_inference=false",
            ],
        )
    args_inf = resolve_inference_config(cfg)

    t0 = time.perf_counter()
    teacher, tokenizer = load_engine(args_inf, "optimized", args.precision)
    os.environ["MAPPERATORINATOR_TURBO_TIER2_FUSED"] = "1"
    candidate, _ = load_engine(args_inf, "turbo", args.precision)

    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, teacher, tokenizer)
    audio = preprocessor.load(str(args.audio))
    sequences = preprocessor.segment(audio)
    song_length = sequences[2]
    frames0, frame_time0 = sequences[0][0], sequences[1][0]
    dump_tokens = dumps[("map", 0)][: max(args.max_seq - 32, 8)]

    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        beatmap_id=getattr(args_inf, "beatmap_id", -1),
        difficulty=args_inf.difficulty,
        mapper_id=getattr(args_inf, "mapper_id", -1),
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        keycount=getattr(args_inf, "keycount", 4),
        hold_note_ratio=getattr(args_inf, "hold_note_ratio", -1),
        scroll_speed_ratio=getattr(args_inf, "scroll_speed_ratio", -1),
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    osu_path = find_osu(args.dump_run)
    tip_beatmap = Beatmap.from_path(osu_path)
    timing = list(tip_beatmap.timing_points)
    extra_in_context = {ContextType.TIMING: timing}
    output_type = [t for t in list(args_inf.output_type) if t != ContextType.TIMING]
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
    gen_in, gen_out, req_special = processor._get_viable_template(
        in_context=list(args_inf.in_context) if args_inf.in_context else [],
        out_context=output_type,
        extra_in_context=extra_in_context,
        gamemode=generation_config.gamemode,
    )
    in_ctx = processor.get_in_context(
        in_context=gen_in,
        beatmap_path=None,
        extra_in_context=extra_in_context,
        song_length=song_length,
    )
    out_ctx = processor.get_out_context(
        out_context=gen_out,
        generation_config=generation_config,
        given_context=list(args_inf.in_context) if args_inf.in_context else [],
        beatmap_path=None,
        extra_in_context=extra_in_context,
        song_length=song_length,
        verbose=False,
    )
    frames = processor.prepare_frames(frames0)
    frame_time_v = frame_time0.item()
    map_i = next(i for i, c in enumerate(out_ctx) if c["context_type"] == ContextType.MAP)
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
        processor.prepare_context_sequences(out_ctx[: map_i + 1], frame_time_v, True, req_special),
    )
    [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
    prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())]
    gen = torch.tensor(dump_tokens, dtype=torch.long)
    full = torch.cat([prompt_ids.cpu(), gen], dim=0).unsqueeze(0)
    dec_in = full[:, :-1]
    if dec_in.shape[1] > args.max_seq:
        dec_in = dec_in[:, : args.max_seq]
    dec_attn = dec_in.ne(processor.tokenizer.pad_id)
    mk = dict(model_kwargs)
    if processor.do_song_position_embed:
        global_pos_start = frame_time_v / song_length
        global_pos_end = (frame_time_v + processor.miliseconds_per_sequence) / song_length
        mk["song_position"] = torch.tensor(
            [global_pos_start, global_pos_end], dtype=torch.float32
        ).unsqueeze(0)

    tip_ms = time_forward_decode_proxy(
        teacher,
        frames=frames,
        decoder_input_ids=dec_in,
        decoder_attention_mask=dec_attn,
        model_kwargs=mk,
        precision=args.precision,
        warmup=args.warmup,
        iters=args.iters,
    )
    turbo_ms = time_forward_decode_proxy(
        candidate,
        frames=frames,
        decoder_input_ids=dec_in,
        decoder_attention_mask=dec_attn,
        model_kwargs=mk,
        precision=args.precision,
        warmup=args.warmup,
        iters=args.iters,
    )
    saved = tip_ms - turbo_ms
    report = {
        "section": 38,
        "kind": "decode_microbench_teacher_forced_proxy",
        "campaign_tip": "55949274",
        "auth_fp16_main_tps": 366.11,
        "precision": args.precision,
        "seq_len": int(dec_in.shape[1]),
        "warmup": args.warmup,
        "iters": args.iters,
        "tip_ms_per_token": tip_ms,
        "turbo_ms_per_token": turbo_ms,
        "saved_ms_per_token": saved,
        "budget_ms_per_token": BUDGET_MS,
        "clears_token_budget": bool(saved >= BUDGET_MS),
        "note": (
            "Teacher-forced full-seq proxy (use_cache=False). "
            "One-token decode microbench preferred before reciprocal; "
            "not a song-wall TPS claim."
        ),
        "no_500_claim": True,
        "elapsed_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    sys.exit(0 if report["clears_token_budget"] else 4)


if __name__ == "__main__":
    main()
