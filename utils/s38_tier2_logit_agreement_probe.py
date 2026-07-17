#!/usr/bin/env python3
"""§38 rung-1: teacher-forced TIER2 logit agreement on real tensors.

Compares tip ``optimized`` (teacher) vs ``turbo`` TIER2 fused fp32-accumulate
decoder numerics (candidate) over dumped SALVALAI map token sequences.

Gates (docs/inference_evidence_packs.md TIER2):
  - max relative logit delta ≤ 1e-2
  - top-1 agreement ≥ 99.5%
  - over ≥ 100k positions (report reachability if fewer)

Not a production TPS claim. Does not touch optimized bit-exact default.
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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import InferenceConfig  # noqa: E402
from inference import GenerationConfig, load_model_with_engine  # noqa: E402
from osuT5.osuT5.dataset.data_utils import (  # noqa: E402
    TIMING_TYPES,
    events_of_type,
    update_event_times,
)
from osuT5.osuT5.inference import Postprocessor, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.inference.turbo.fused_step import (  # noqa: E402
    relative_logit_delta,
    top1_agreement,
)
from osuT5.osuT5.tokenizer import ContextType  # noqa: E402

MAX_REL_GATE = 1e-2
TOP1_GATE = 0.995
MIN_POSITIONS = 100_000


def load_dump_tokens(profile_path: Path) -> dict[tuple[str, int], list[int]]:
    payload = json.loads(profile_path.read_text())
    out: dict[tuple[str, int], list[int]] = {}
    for rec in payload.get("generation", []):
        ids = rec.get("generated_token_ids")
        if not (isinstance(ids, list) and ids and isinstance(ids[0], int)):
            continue
        ctx = str(rec.get("context_type"))
        idx = int(rec.get("sequence_index"))
        out[(ctx, idx)] = [int(x) for x in ids]
    if not out:
        raise SystemExit(f"no generated_token_ids in {profile_path}")
    return out


def find_profile(run_root: Path) -> Path:
    cands = sorted(run_root.glob("**/beatmap*.osu.profile.json"))
    if not cands:
        raise SystemExit(f"no profile json under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def find_osu(run_root: Path) -> Path:
    cands = sorted(run_root.glob("**/beatmap*.osu"))
    # prefer non-profile .osu beside candidate_first
    plain = [p for p in cands if not str(p).endswith(".profile.json")]
    for p in plain:
        if "candidate_first" in str(p):
            return p
    if plain:
        return plain[0]
    raise SystemExit(f"no .osu under {run_root}")


def resolve_inference_config(cfg) -> InferenceConfig:
    """Build nested InferenceConfig (incl. train.data) despite hydra MISSING."""
    merged = OmegaConf.merge(OmegaConf.structured(InferenceConfig), cfg)
    # Structured InferenceConfig marks hydra as MISSING; probes do not need it.
    OmegaConf.update(merged, "hydra", {}, merge=False)
    args_inf = OmegaConf.to_object(merged)
    if not isinstance(args_inf, InferenceConfig):
        raise TypeError(f"expected InferenceConfig, got {type(args_inf)}")
    return args_inf


def raw_model(model):
    raw, _ = unwrap_engine_binding(model)
    return raw


@torch.no_grad()
def teacher_force_logits(
    model,
    *,
    frames: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    model_kwargs: dict[str, Any],
    precision: str,
) -> torch.Tensor:
    m = raw_model(model)
    kw = {
        k: (v.to(m.device) if isinstance(v, torch.Tensor) else v)
        for k, v in dict(model_kwargs).items()
    }
    for k, v in list(kw.items()):
        if k != "inputs" and isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            kw[k] = v.to(m.dtype)
    frames = frames.to(m.device)
    dec = decoder_input_ids.to(m.device)
    attn = decoder_attention_mask.to(m.device)
    with torch.autocast(
        device_type=m.device.type,
        dtype=torch.bfloat16,
        enabled=(precision == "amp"),
    ):
        out = m.forward(
            frames=frames,
            decoder_input_ids=dec,
            decoder_attention_mask=attn,
            use_cache=False,
            **kw,
        )
    return out.logits.float()


def inject_dump_context(
    processor: Processor,
    *,
    sequences,
    out_context: list[dict],
    dumps: dict[tuple[str, int], list[int]],
    context_type: ContextType,
) -> None:
    ctx = next(c for c in out_context if c["context_type"] == context_type)
    if ctx.get("finished"):
        return
    for sequence_index, (_frames, frame_time) in enumerate(zip(*sequences[:2])):
        trim_lookback = (
            sequence_index != 0 and processor.types_first and processor.lookback_time > 0
        )
        trim_lookahead = sequence_index != len(sequences[0]) - 1
        frame_time_v = frame_time.item()
        key = (context_type.value, sequence_index)
        if key not in dumps:
            raise SystemExit(f"missing dump tokens for {key}")
        tokens = torch.tensor(dumps[key], dtype=torch.long)
        processor.add_predicted_tokens_to_context(
            ctx, tokens, frame_time_v, trim_lookback, trim_lookahead
        )


def load_engine(args_inf: InferenceConfig, engine: str, precision: str):
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
    raw_model(model).eval()
    return model, tokenizer


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
    ap.add_argument("--max-map-windows", type=int, default=0, help="0 = all")
    ap.add_argument("--max-positions", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    os.chdir(args.repo)
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))

    profile_path = find_profile(args.dump_run)
    dumps = load_dump_tokens(profile_path)
    map_keys = sorted(k for k in dumps if k[0] == "map")
    print(f"dump={profile_path}", flush=True)
    print(f"map_windows={len(map_keys)}", flush=True)

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
    print("loading teacher optimized…", flush=True)
    teacher, tokenizer = load_engine(args_inf, "optimized", args.precision)
    print("loading candidate turbo…", flush=True)
    os.environ["MAPPERATORINATOR_TURBO_TIER2_FUSED"] = "1"
    candidate, _ = load_engine(args_inf, "turbo", args.precision)
    print(f"model_loaded_s={time.perf_counter() - t0:.1f}", flush=True)

    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, teacher, tokenizer)
    postprocessor = Postprocessor(args_inf)

    audio = preprocessor.load(str(args.audio))
    sequences = preprocessor.segment(audio)
    song_length = sequences[2]
    print(f"sequences={len(sequences[0])} song_ms={song_length}", flush=True)

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

    # Use tip-auth .osu TimingPoints (same FIX as §35 50145885).
    from slider import Beatmap

    osu_path = find_osu(args.dump_run)
    print(f"timing_source_osu={osu_path}", flush=True)
    tip_beatmap = Beatmap.from_path(osu_path)
    timing = list(tip_beatmap.timing_points)
    if len(timing) == 0:
        raise SystemExit(f"tip .osu has no timing points: {osu_path}")
    extra_in_context = {ContextType.TIMING: timing}
    output_type = [t for t in list(args_inf.output_type) if t != ContextType.TIMING]
    print(f"timing_points={len(timing)}", flush=True)
    model_kwargs = processor._get_model_cond_kwargs(generation_config)

    gen_in, gen_out, req_special = processor._get_viable_template(
        in_context=list(args_inf.in_context) if args_inf.in_context else [],
        out_context=output_type,
        extra_in_context=extra_in_context,
        gamemode=generation_config.gamemode,
    )
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
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
    map_ctx = next(c for c in out_ctx if c["context_type"] == ContextType.MAP)

    max_rel_all = 0.0
    agree_n = 0
    pos_n = 0
    windows_done = 0
    per_window: list[dict[str, Any]] = []
    max_windows = args.max_map_windows if args.max_map_windows > 0 else len(map_keys)

    t_fwd = time.perf_counter()
    for sequence_index, (frames, frame_time) in enumerate(zip(*sequences[:2])):
        if windows_done >= max_windows:
            break
        key = ("map", sequence_index)
        if key not in dumps:
            raise SystemExit(f"missing map dump {key}")
        dump_tokens = dumps[key]
        if len(dump_tokens) < 1:
            continue

        trim_lookback = (
            sequence_index != 0 and processor.types_first and processor.lookback_time > 0
        )
        trim_lookahead = sequence_index != len(sequences[0]) - 1
        frames = processor.prepare_frames(frames)
        frame_time_v = frame_time.item()

        map_i = next(i for i, c in enumerate(out_ctx) if c["context_type"] == ContextType.MAP)
        cond_prompt, uncond_prompt = processor.get_prompts(
            processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
            processor.prepare_context_sequences(out_ctx[: map_i + 1], frame_time_v, True, req_special),
        )
        [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
        prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())]

        gen = torch.tensor(dump_tokens, dtype=torch.long)
        full = torch.cat([prompt_ids.cpu(), gen], dim=0).unsqueeze(0)
        if full.shape[1] < 2:
            processor.add_predicted_tokens_to_context(
                map_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
            )
            continue

        dec_in = full[:, :-1]
        dec_attn = dec_in.ne(processor.tokenizer.pad_id)
        mk = dict(model_kwargs)
        if processor.do_song_position_embed:
            global_pos_start = frame_time_v / song_length
            global_pos_end = (
                frame_time_v + processor.miliseconds_per_sequence
            ) / song_length
            mk["song_position"] = torch.tensor(
                [global_pos_start, global_pos_end], dtype=torch.float32
            ).unsqueeze(0)

        logits_t = teacher_force_logits(
            teacher,
            frames=frames,
            decoder_input_ids=dec_in,
            decoder_attention_mask=dec_attn,
            model_kwargs=mk,
            precision=args.precision,
        )
        logits_c = teacher_force_logits(
            candidate,
            frames=frames,
            decoder_input_ids=dec_in,
            decoder_attention_mask=dec_attn,
            model_kwargs=mk,
            precision=args.precision,
        )

        prompt_len = int(prompt_ids.numel())
        start = max(prompt_len - 1, 0)
        end = start + len(dump_tokens)
        lt = logits_t[0, start:end]
        lc = logits_c[0, start:end]
        L = min(lt.shape[0], lc.shape[0])
        lt = lt[:L]
        lc = lc[:L]
        if L == 0:
            processor.add_predicted_tokens_to_context(
                map_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
            )
            continue

        rel = relative_logit_delta(lt, lc)
        agr = top1_agreement(lt, lc)
        w_max = float(rel.max().item())
        w_agree = float(agr.float().mean().item())
        max_rel_all = max(max_rel_all, w_max)
        agree_n += int(agr.sum().item())
        pos_n += int(agr.numel())
        windows_done += 1
        per_window.append(
            {
                "key": list(key),
                "positions": int(agr.numel()),
                "max_rel_logit_delta": w_max,
                "top1_agreement": w_agree,
            }
        )
        processor.add_predicted_tokens_to_context(
            map_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
        )
        print(
            f"window={key} pos={L} max_rel={w_max:.4e} top1={w_agree:.6f} cum_pos={pos_n}",
            flush=True,
        )
        if args.max_positions > 0 and pos_n >= args.max_positions:
            break

    top1 = (agree_n / pos_n) if pos_n else 0.0
    gate_rel = max_rel_all <= MAX_REL_GATE
    gate_top1 = top1 >= TOP1_GATE
    gate_n = pos_n >= MIN_POSITIONS
    report = {
        "section": 38,
        "rung": 1,
        "kind": "teacher_forced_logit_agreement",
        "campaign_tip": "55949274",
        "auth_fp16_main_tps": 366.11,
        "precision": args.precision,
        "dump_profile": str(profile_path),
        "teacher_engine": "optimized",
        "candidate_engine": "turbo",
        "candidate_preset": "turbo-tier2-fused-step-s38-v1",
        "positions": pos_n,
        "windows": windows_done,
        "max_rel_logit_delta": max_rel_all,
        "top1_agreement": top1,
        "fwd_s": time.perf_counter() - t_fwd,
        "elapsed_s": time.perf_counter() - t0,
        "gates": {
            "max_rel_leq_1e-2": gate_rel,
            "top1_geq_99_5pct": gate_top1,
            "positions_geq_100k": gate_n,
            "tier2_quality_gate_pass": bool(gate_rel and gate_top1 and gate_n),
            "quality_gate_looks_reachable": bool(gate_rel and gate_top1),
        },
        "thresholds": {
            "max_rel_logit_delta": MAX_REL_GATE,
            "top1_agreement": TOP1_GATE,
            "min_positions": MIN_POSITIONS,
        },
        "per_window": per_window,
        "no_500_claim": True,
        "optimized_default_unchanged": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gates"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    if not (gate_rel and gate_top1):
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
