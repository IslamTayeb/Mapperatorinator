#!/usr/bin/env python3
"""§36/§37 turbo speculative generate_window smoke (Track C).

1) End-to-end short-window speculative decode with tiny draft.
2) Optional TIER1a greedy canary: turbo vs optimized bit-exact on same window.

Not a production TPS claim. TIER1 pack required before turbo/500 ship.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__import__("config")

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.tokenizer import ContextType  # noqa: E402


def load_args(config_name: str, overrides: list[str]):
    config_dir = REPO / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            config_name=config_name.removeprefix("inference/"), overrides=overrides
        )
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def find_osu(run_root: Path) -> Path:
    cands = [p for p in sorted(run_root.glob("**/beatmap*.osu")) if p.suffix == ".osu"]
    if not cands:
        raise SystemExit(f"no .osu under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def build_first_map_window_kwargs(processor, args_inf, generation_config, song_length, osu_path, sequences):
    sys.path.insert(0, str(REPO / "utils"))
    import s37_tiny_draft_train_smoke as s37  # type: ignore

    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = s37.prepare_contexts(
        processor, args_inf, generation_config, song_length, osu_path
    )
    sequence_index = 0
    frames, frame_time = sequences[0][sequence_index], sequences[1][sequence_index]
    frames_p = processor.prepare_frames(frames)
    frame_time_v = frame_time.item()
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
        processor.prepare_context_sequences(out_ctx[: map_i + 1], frame_time_v, True, req_special),
    )
    [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
    prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())].cpu()
    # prepare_frames already ensures batch dim [B, T]; do not unsqueeze again.
    if frames_p.ndim == 1:
        frames_p = frames_p.unsqueeze(0)
    model_kwargs: dict[str, Any] = {
        "inputs": frames_p,
        "decoder_input_ids": prompt_ids.unsqueeze(0),
        "decoder_attention_mask": torch.ones_like(prompt_ids.unsqueeze(0)),
        **cond_kwargs,
    }
    if processor.do_song_position_embed:
        model_kwargs["song_position"] = torch.tensor(
            [
                [
                    frame_time_v / song_length,
                    (frame_time_v + processor.miliseconds_per_sequence) / song_length,
                ]
            ],
            dtype=torch.float32,
        )
    meta = {
        "prompt_len": int(prompt_ids.numel()),
        "timing_points": n_timing,
        "sequence_index": sequence_index,
        "context_type": "map",
    }
    return model_kwargs, meta


def run_engine_window(
    *,
    engine_name: str,
    args_inf,
    draft_ckpt: Path | None,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
):
    if engine_name == "turbo":
        if draft_ckpt is None:
            raise SystemExit("turbo requires --draft-ckpt")
        os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(draft_ckpt)

    bound, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision=args_inf.precision,
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine=engine_name,
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    raw, runtime = unwrap_engine_binding(bound)
    if runtime is None:
        raise SystemExit(f"{engine_name} produced no runtime binding")
    raw.eval()
    session = runtime.new_context_state()
    t0 = time.perf_counter()
    result, stats = runtime.generate_window(
        model=raw,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        generate_kwargs=dict(generate_kwargs),
        context_state=session,
    )
    wall = time.perf_counter() - t0
    return {
        "engine": engine_name,
        "result_tokens": result[0].tolist(),
        "stats": stats,
        "wall_s": wall,
        "session": {
            "accepted_tokens_total": getattr(session, "accepted_tokens_total", None),
            "verify_steps": getattr(session, "verify_steps", None),
            "draft_calls": getattr(session, "draft_calls", None),
        },
        "runtime_meta": (
            runtime.profile_metadata() if hasattr(runtime, "profile_metadata") else {}
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        ),
    )
    ap.add_argument(
        "--audio", type=Path, default=Path("/work/imt11/Mapperatorinator/data/salvalai.mp3")
    )
    ap.add_argument(
        "--draft-ckpt",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt"
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--greedy-canary", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args_inf = load_args(
        args.config_name,
        [
            f"audio_path={args.audio}",
            "device=cuda",
            f"precision={args.precision}",
            "attn_implementation=sdpa",
            "inference_engine=v32",
            "use_server=false",
            "parallel=false",
            "cfg_scale=1.0",
            "num_beams=1",
            f"seed={args.seed}",
            "profile_inference=false",
        ],
    )

    helper_model, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision=args.precision,
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="v32",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    if args.precision == "fp16":
        helper_model = helper_model.half()
    helper_model.eval()

    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, helper_model, tokenizer)
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
    osu_path = find_osu(args.dump_run)
    model_kwargs, meta = build_first_map_window_kwargs(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )
    prompt_len = int(meta["prompt_len"])
    max_length = prompt_len + int(args.max_new_tokens)
    generate_kwargs = {
        "precision": args.precision,
        "do_sample": (not args.greedy_canary),
        "num_beams": 1,
        "top_p": 0.9,
        "max_length": max_length,
        "cfg_scale": 1.0,
        "timeshift_bias": 0,
        # Structural processors in turbo v1 omit conditional temperature warper.
        "types_first": False,
        # Greedy canary: do_sample=False (argmax). Keep temperature=1.0 so
        # optimized TemperatureLogitsWarper stays well-defined.
        "temperature": 1.0 if args.greedy_canary else 0.9,
        "lookback_time": 0.0,
        "lookahead_time": 0.0,
        "context_type": "map",
        "sync_model_timing": True,
        "pad_token_id": getattr(tokenizer, "pad_id", None),
    }
    print(
        f"prompt_len={prompt_len} max_length={max_length} greedy_canary={args.greedy_canary}",
        flush=True,
    )

    del helper_model, processor
    torch.cuda.empty_cache()

    turbo_run = run_engine_window(
        engine_name="turbo",
        args_inf=args_inf,
        draft_ckpt=args.draft_ckpt,
        model_kwargs=model_kwargs,
        generate_kwargs=generate_kwargs,
    )
    print(
        f"turbo_done wall_s={turbo_run['wall_s']:.3f} "
        f"gen_tokens={turbo_run['stats'].get('generated_tokens')} "
        f"accepted_per_verify={turbo_run['stats'].get('turbo_accepted_per_verify')} "
        f"speculative={turbo_run['stats'].get('turbo_speculative')}",
        flush=True,
    )

    canary = None
    if args.greedy_canary:
        torch.cuda.empty_cache()
        opt_run = run_engine_window(
            engine_name="optimized",
            args_inf=args_inf,
            draft_ckpt=None,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
        )
        turbo_tok = turbo_run["result_tokens"]
        opt_tok = opt_run["result_tokens"]
        match = turbo_tok == opt_tok
        first_mismatch = None
        if not match:
            n = min(len(turbo_tok), len(opt_tok))
            first_mismatch = next(
                (i for i in range(n) if turbo_tok[i] != opt_tok[i]),
                n,
            )
        canary = {
            "match": match,
            "turbo_len": len(turbo_tok),
            "optimized_len": len(opt_tok),
            "first_mismatch": first_mismatch,
            "optimized_wall_s": opt_run["wall_s"],
            "optimized_stats": {
                k: opt_run["stats"].get(k)
                for k in (
                    "generated_tokens",
                    "tokens_per_second",
                    "elapsed_seconds",
                    "decoder_loop_backend",
                )
            },
        }
        print(f"tier1a_greedy_canary_match={match}", flush=True)

    payload = {
        "section": "36",
        "track": "C",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} branch --show-current").read().strip(),
        "draft_ckpt": str(args.draft_ckpt),
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "greedy_canary": bool(args.greedy_canary),
        "meta": meta,
        "turbo": {
            "wall_s": turbo_run["wall_s"],
            "stats": turbo_run["stats"],
            "session": turbo_run["session"],
            "runtime_meta": turbo_run["runtime_meta"],
            "result_len": len(turbo_run["result_tokens"]),
        },
        "tier1a_greedy_canary": canary,
        "gate": {
            "smoke_ok": bool(turbo_run["stats"].get("turbo_speculative")),
            "tier1a_ok": (None if canary is None else bool(canary["match"])),
            "note": "Not a 500 claim. Full TIER1 pack still required.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)
    if not payload["gate"]["smoke_ok"]:
        raise SystemExit("turbo speculative smoke failed")
    if args.greedy_canary and not payload["gate"]["tier1a_ok"]:
        raise SystemExit("TIER1a greedy canary mismatch")


if __name__ == "__main__":
    main()
