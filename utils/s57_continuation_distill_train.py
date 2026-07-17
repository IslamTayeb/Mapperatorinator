#!/usr/bin/env python3
"""§57 continuation(+timing) distill — adapt multsong 2L draft on short lookback + timing.

Reuses §37/§43 CE+KL plumbing. Materializes:
  - map₀ (anti-regression)
  - map_rest from turbo short-continuation dumps (s55; med gen≈7)
  - timing windows (was never in §37/§43 train)

Offline held-out gates before any turbo wire. Not a TPS / 500 claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from slider import Beatmap

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_s37():
    path = REPO / "utils" / "s37_tiny_draft_train_smoke.py"
    spec = importlib.util.spec_from_file_location("s37_tiny_draft_train_smoke", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s37 = _load_s37()
DRAFT_INIT_LAYERS = s37.DRAFT_INIT_LAYERS
PRIMARY_GAMMA = s37.PRIMARY_GAMMA
acceptance_alpha = s37.acceptance_alpha
apply_temp_top_p = s37.apply_temp_top_p
build_two_layer_draft = s37.build_two_layer_draft
distill_ce_kl_loss = s37.distill_ce_kl_loss
expected_accepted = s37.expected_accepted
find_osu = s37.find_osu
find_profile = s37.find_profile
get_decoder = s37.get_decoder
iter_map_windows = s37.iter_map_windows
load_args = s37.load_args
load_dump_tokens = s37.load_dump_tokens
prepare_contexts = s37.prepare_contexts
teacher_force_logits = s37.teacher_force_logits

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.tokenizer import ContextType  # noqa: E402

__import__("config")

DEFAULT_SHARDS = Path(
    "/work/imt11/Mapperatorinator/runs/s57-continuation-shards/continuation_shards.json"
)
DEFAULT_RESUME = Path(
    "/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_multsong.pt"
)
S55_RUN = Path("/work/imt11/Mapperatorinator/runs/s55-turbo-fp16-scout-50152542")
TIP_RUN = Path(
    "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
)
AUDIO_SALVALAI = Path("/work/imt11/Mapperatorinator/data/salvalai.mp3")
FIVE_SONG_AUDIO = {
    "pegasus": Path("/work/imt11/Mapperatorinator/data/five-song-profile/pegasus.mp3"),
    "lambada": Path("/work/imt11/Mapperatorinator/data/five-song-profile/lambada.mp3"),
}


def prepare_timing_contexts(processor, args_inf, generation_config, song_length: float):
    """Timing as out_context (generate), matching v32 sequential first pass."""
    gen_in, gen_out, req_special = processor._get_viable_template(
        in_context=[],
        out_context=[ContextType.TIMING],
        extra_in_context=None,
        gamemode=generation_config.gamemode,
    )
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
    cond_kwargs = {k: v for k, v in model_kwargs.items() if k != "song_position"}
    in_ctx = processor.get_in_context(
        in_context=gen_in,
        beatmap_path=None,
        extra_in_context=None,
        song_length=song_length,
    )
    out_ctx = processor.get_out_context(
        out_context=gen_out,
        generation_config=generation_config,
        given_context=[],
        beatmap_path=None,
        extra_in_context=None,
        song_length=song_length,
        verbose=False,
    )
    timing_i = next(i for i, c in enumerate(out_ctx) if c["context_type"] == ContextType.TIMING)
    return in_ctx, out_ctx, timing_i, req_special, cond_kwargs


def iter_timing_windows(
    *,
    processor,
    sequences,
    dumps,
    in_ctx,
    out_ctx,
    timing_i,
    req_special,
    song_length,
    max_windows: int,
):
    timing_ctx = out_ctx[timing_i]
    done = 0
    for sequence_index, (frames, frame_time) in enumerate(zip(*sequences[:2])):
        if done >= max_windows:
            break
        key = ("timing", sequence_index)
        if key not in dumps:
            continue
        dump_tokens = dumps[key]
        if len(dump_tokens) < 1:
            continue
        trim_lookback = sequence_index != 0 and processor.types_first and processor.lookback_time > 0
        trim_lookahead = sequence_index != len(sequences[0]) - 1
        frames_p = processor.prepare_frames(frames)
        frame_time_v = frame_time.item()
        cond_prompt, uncond_prompt = processor.get_prompts(
            processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
            processor.prepare_context_sequences(out_ctx[: timing_i + 1], frame_time_v, True, req_special),
        )
        [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
        prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())].cpu()
        gen = torch.tensor(dump_tokens, dtype=torch.long)
        full = torch.cat([prompt_ids, gen], dim=0).unsqueeze(0)
        if full.shape[1] < 2:
            processor.add_predicted_tokens_to_context(
                timing_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
            )
            continue
        dec_in = full[:, :-1]
        labels = full[:, 1:].clone()
        prompt_len = int(prompt_ids.numel())
        if prompt_len > 0:
            labels[:, : max(prompt_len - 1, 0)] = -100
        dec_attn = dec_in.ne(processor.tokenizer.pad_id)
        song_pos = None
        if processor.do_song_position_embed:
            song_pos = torch.tensor(
                [
                    [
                        frame_time_v / song_length,
                        (frame_time_v + processor.miliseconds_per_sequence) / song_length,
                    ]
                ],
                dtype=torch.float32,
            )
        yield {
            "sequence_index": sequence_index,
            "context_type": "timing",
            "role": "timing",
            "frames": frames_p,
            "dec_in": dec_in,
            "dec_attn": dec_attn,
            "labels": labels,
            "song_pos": song_pos,
            "prompt_len": prompt_len,
            "dump_tokens": dump_tokens,
            "gen": gen,
            "trim_lookback": trim_lookback,
            "trim_lookahead": trim_lookahead,
            "frame_time_v": frame_time_v,
        }
        processor.add_predicted_tokens_to_context(
            timing_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
        )
        done += 1


def eval_acceptance_sliced(
    teacher,
    draft,
    *,
    windows: list[dict],
    cond_kwargs_by_ctx: dict[str, dict],
    precision: str,
    temperature: float,
    top_p: float,
    gamma: int = PRIMARY_GAMMA,
) -> dict:
    teacher.eval()
    draft.eval()
    by_role: dict[str, list[float]] = {"map0": [], "map_rest": [], "timing": []}
    for w in windows:
        ctx = w.get("context_type", "map")
        cond_kwargs = cond_kwargs_by_ctx.get(ctx) or cond_kwargs_by_ctx.get("map")
        logits_p = teacher_force_logits(
            teacher,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            song_position=w["song_pos"],
            precision=precision,
            cond_kwargs=cond_kwargs,
        )
        logits_q = teacher_force_logits(
            draft,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            song_position=w["song_pos"],
            precision=precision,
            cond_kwargs=cond_kwargs,
        )
        start = max(w["prompt_len"] - 1, 0)
        end = start + len(w["dump_tokens"])
        lp = logits_p[0, start:end].cpu()
        lq = logits_q[0, start:end].cpu()
        if lp.numel() == 0:
            continue
        # Drop positions whose hard label is outside draft/teacher proj_out
        dump = torch.tensor(w["dump_tokens"], dtype=torch.long)
        in_vocab = dump.lt(lp.shape[-1])
        finite = torch.isfinite(lp).all(dim=-1) & torch.isfinite(lq).all(dim=-1) & in_vocab
        if int(finite.sum()) == 0:
            continue
        p = apply_temp_top_p(lp[finite], temperature, top_p)
        q = apply_temp_top_p(lq[finite], temperature, top_p)
        a = acceptance_alpha(p, q)
        role = w.get("role") or ("map0" if w.get("sequence_index") == 0 and ctx == "map" else ctx)
        if role not in by_role:
            by_role[role] = []
        by_role[role].extend(float(x) for x in a.tolist() if x == x)

    def _summ(alphas: list[float]) -> dict:
        mean_a = sum(alphas) / len(alphas) if alphas else 0.0
        return {
            "positions": len(alphas),
            "mean_alpha": mean_a,
            "E_accepted_per_step": expected_accepted(mean_a, gamma),
        }

    slices = {k: _summ(v) for k, v in by_role.items()}
    # Window-weighted path toward full-song: use token counts as weights
    tok = {k: slices[k]["positions"] for k in slices}
    e = {k: slices[k]["E_accepted_per_step"] for k in slices}
    map_tok = tok.get("map0", 0) + tok.get("map_rest", 0)
    map_e = (
        (e.get("map0", 1.0) * tok.get("map0", 0) + e.get("map_rest", 1.0) * tok.get("map_rest", 0))
        / map_tok
        if map_tok
        else 1.0
    )
    full_tok = map_tok + tok.get("timing", 0)
    full_e = (
        (map_e * map_tok + e.get("timing", 1.0) * tok.get("timing", 0)) / full_tok
        if full_tok
        else 1.0
    )
    return {
        "gamma": gamma,
        "slices": slices,
        "map_wE": map_e,
        "full_wE": full_e,
        "token_weights": tok,
    }


def gate_decision(holdout: dict, *, baseline_rest: float = 1.121) -> dict:
    rest = holdout["slices"].get("map_rest", {})
    timing = holdout["slices"].get("timing", {})
    map0 = holdout["slices"].get("map0", {})
    rest_e = float(rest.get("E_accepted_per_step") or 0.0)
    timing_e = float(timing.get("E_accepted_per_step") or 0.0)
    map0_e = float(map0.get("E_accepted_per_step") or 0.0)
    delta_rest = rest_e - baseline_rest
    rest_ok = rest_e >= 1.7 or (rest_e >= 1.63 and delta_rest >= 0.15)
    timing_ok = timing_e >= 1.3  # meaningful rise vs ≡1.0
    map0_ok = map0_e >= 1.7 or map0.get("positions", 0) == 0
    wire = bool(rest_ok and (timing_ok or False))  # timing must rise for wire; exit-gate is policy
    if rest_ok and timing_ok and map0_ok:
        decision = "GO_OFFLINE_GATES"
        text = "held-out map_rest + timing cleared; still need in-loop full-song before wire"
    elif rest_ok and not timing_ok:
        decision = "PARTIAL_MAP_REST_ONLY"
        text = (
            "map_rest improved; timing still dead — either more timing train or "
            "declare timing out of E-gate math before song-level 1.7 claim"
        )
    elif delta_rest >= 0.15:
        decision = "CONTINUE_TRAIN"
        text = "ΔE map_rest ≥ +0.15 but absolute E below aim; continue train"
    else:
        decision = "STOP_HELD_OUT"
        text = "held-out map_rest miss (ΔE < +0.15 and E < 1.63)"
    return {
        "decision": decision,
        "gate_text": text,
        "wire_runtime": False,  # never wire from offline alone
        "rest_ok": rest_ok,
        "timing_ok": timing_ok,
        "map0_ok": map0_ok,
        "map_rest_E": rest_e,
        "timing_E": timing_e,
        "map0_E": map0_e,
        "delta_map_rest_vs_s55": delta_rest,
        "map_wE": holdout.get("map_wE"),
        "full_wE": holdout.get("full_wE"),
    }


def materialize_from_shards(
    *,
    shards: dict,
    teacher,
    tokenizer,
    args_inf,
    song_filter: str | None,
    split: str,
) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Build train/eval windows. Returns (windows, cond_kwargs_by_ctx, metas).

    split=train  → shard split==train (includes map0)
    split=holdout → shard split==holdout plus s55 map0 (anti-regression probe)
    """
    windows_meta = shards["windows"]
    allow: dict[tuple[str, str, str], set[int]] = {}
    role_of: dict[tuple[str, str, str, int], str] = {}
    for r in windows_meta:
        if song_filter and r["song_id"] != song_filter:
            continue
        role = r["role"]
        rsplit = r.get("split")
        if split == "train":
            if rsplit != "train":
                continue
        elif split == "holdout":
            keep = rsplit == "holdout" or (
                role == "map0" and r["source"] == "s55_turbo"
            )
            if not keep:
                continue
        else:
            raise SystemExit(f"unknown split {split}")
        key = (r["song_id"], r["source"], r["context_type"])
        allow.setdefault(key, set()).add(int(r["sequence_index"]))
        role_of[(r["song_id"], r["source"], r["context_type"], int(r["sequence_index"]))] = role

    # Resolve profile/audio/osu per (song, source)
    source_paths = {
        ("salvalai", "s55_turbo"): {
            "audio": AUDIO_SALVALAI,
            "dump_run": S55_RUN,
            "profile": find_profile(S55_RUN),
            "osu": find_osu(S55_RUN),
        },
        ("salvalai", "tip_auth"): {
            "audio": AUDIO_SALVALAI,
            "dump_run": TIP_RUN,
            "profile": find_profile(TIP_RUN),
            "osu": find_osu(TIP_RUN),
        },
    }
    for song_id, audio in FIVE_SONG_AUDIO.items():
        prof = Path(
            f"/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/"
            f"after_separate/official/{song_id}/{song_id}.profile.json"
        )
        if prof.exists():
            source_paths[(song_id, "five_song")] = {
                "audio": audio,
                "dump_run": prof.parent,
                "profile": prof,
                "osu": find_osu(prof.parent) if list(prof.parent.glob("**/beatmap*.osu")) else prof,
            }

    all_windows: list[dict] = []
    cond_by_ctx: dict[str, dict] = {}
    metas: list[dict] = []

    for (song_id, source, ctx), seqs in sorted(allow.items()):
        paths = source_paths.get((song_id, source))
        if paths is None:
            print(f"WARN: no paths for {song_id}/{source}", flush=True)
            continue
        dumps = load_dump_tokens(paths["profile"])
        # Filter dumps to allowed seqs for this context
        dumps_f = {k: v for k, v in dumps.items() if k[0] == ctx and k[1] in seqs}
        if not dumps_f:
            continue
        args_inf.audio_path = str(paths["audio"])
        preprocessor = Preprocessor(args_inf, parallel=False)
        processor = Processor(args_inf, teacher, tokenizer)
        audio = preprocessor.load(str(paths["audio"]))
        sequences = preprocessor.segment(audio)
        song_length = sequences[2]
        generation_config = GenerationConfig(
            gamemode=args_inf.gamemode,
            difficulty=args_inf.difficulty,
            year=args_inf.year,
            hitsounded=args_inf.hitsounded,
            descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
        )
        if ctx == "map":
            in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = prepare_contexts(
                processor, args_inf, generation_config, song_length, paths["osu"]
            )
            cond_by_ctx["map"] = cond_kwargs
            # Materialize ALL map windows in order so lookback context is correct,
            # then keep only allowed seqs.
            full_dumps = {k: v for k, v in dumps.items() if k[0] == "map"}
            max_seq = max(seqs) if seqs else -1
            n_seq = len(sequences[0])
            kept = []
            for w in iter_map_windows(
                processor=processor,
                sequences=sequences,
                dumps=full_dumps,
                in_ctx=in_ctx,
                out_ctx=out_ctx,
                map_i=map_i,
                req_special=req_special,
                song_length=song_length,
                max_windows=n_seq,
            ):
                si = int(w["sequence_index"])
                if si > max_seq and not seqs:
                    break
                if si not in seqs:
                    continue
                role = role_of.get((song_id, source, "map", si), "map0" if si == 0 else "map_rest")
                w = dict(w)
                w["context_type"] = "map"
                w["role"] = role
                w["song_id"] = song_id
                w["source"] = source
                w["split"] = split
                kept.append(w)
            all_windows.extend(kept)
            metas.append(
                {
                    "song_id": song_id,
                    "source": source,
                    "context": "map",
                    "n_kept": len(kept),
                    "n_timing_points": n_timing,
                    "profile": str(paths["profile"]),
                }
            )
        elif ctx == "timing":
            in_ctx, out_ctx, timing_i, req_special, cond_kwargs = prepare_timing_contexts(
                processor, args_inf, generation_config, song_length
            )
            cond_by_ctx["timing"] = cond_kwargs
            full_dumps = {k: v for k, v in dumps.items() if k[0] == "timing"}
            kept = []
            for w in iter_timing_windows(
                processor=processor,
                sequences=sequences,
                dumps=full_dumps,
                in_ctx=in_ctx,
                out_ctx=out_ctx,
                timing_i=timing_i,
                req_special=req_special,
                song_length=song_length,
                max_windows=len(sequences[0]),
            ):
                si = int(w["sequence_index"])
                if si not in seqs:
                    continue
                w = dict(w)
                w["song_id"] = song_id
                w["source"] = source
                w["split"] = split
                kept.append(w)
            all_windows.extend(kept)
            metas.append(
                {
                    "song_id": song_id,
                    "source": source,
                    "context": "timing",
                    "n_kept": len(kept),
                    "profile": str(paths["profile"]),
                }
            )
    return all_windows, cond_by_ctx, metas


def sample_window(windows: list[dict], weights: dict[str, float], rng: random.Random) -> dict:
    ws = [max(weights.get(w["role"], 1.0), 0.0) for w in windows]
    total = sum(ws)
    if total <= 0:
        return rng.choice(windows)
    r = rng.random() * total
    acc = 0.0
    for w, wt in zip(windows, ws):
        acc += wt
        if r <= acc:
            return w
    return windows[-1]


def sanitize_labels(labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Mask pad/OOV targets. Timing dumps include ids ≥ vocab_size_out (proj_out=4069)."""
    out = labels.clone()
    # keep ignore_index (-100); drop anything outside [0, vocab_size)
    bad = out.ge(0) & out.ge(vocab_size)
    out = out.masked_fill(bad, -100)
    return out


def window_has_supervised(labels: torch.Tensor) -> bool:
    return bool(labels.ne(-100).any().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-out", type=Path, default=None)
    ap.add_argument("--resume-ckpt", type=Path, default=DEFAULT_RESUME)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--w-map0", type=float, default=1.0)
    ap.add_argument("--w-map-rest", type=float, default=4.0)
    ap.add_argument("--w-timing", type=float, default=3.0)
    ap.add_argument(
        "--song-filter",
        default="salvalai",
        help="primary song; empty = all songs in shards",
    )
    ap.add_argument("--baseline-rest-e", type=float, default=1.121)
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    shards = json.loads(args.shards.read_text())
    print(f"shards={args.shards} n_windows={shards.get('n_windows')}", flush=True)

    args_inf = load_args(
        args.config_name,
        [
            f"audio_path={AUDIO_SALVALAI}",
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
            f"temperature={args.temperature}",
            f"top_p={args.top_p}",
        ],
    )

    device = torch.device("cuda")
    print("loading teacher…", flush=True)
    t0 = time.perf_counter()
    teacher, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=device,
        max_batch_size=1,
        use_server=False,
        precision=args.precision,
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="v32",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    teacher.eval()
    if args.precision == "fp16":
        teacher = teacher.half()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"teacher_loaded_s={time.perf_counter() - t0:.1f}", flush=True)

    n_layers = len(get_decoder(teacher).layers)
    draft = build_two_layer_draft(teacher, DRAFT_INIT_LAYERS)
    draft = draft.float().to(device)
    train_precision = "fp32"
    resumed_from = None
    if args.resume_ckpt is not None and Path(args.resume_ckpt).exists():
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        state = ckpt.get("draft_state_dict", ckpt)
        missing, unexpected = draft.load_state_dict(state, strict=False)
        resumed_from = str(args.resume_ckpt)
        print(
            f"resumed_ckpt={resumed_from} missing={len(missing)} unexpected={len(unexpected)} "
            f"meta_init={ckpt.get('init_layers')}",
            flush=True,
        )
    else:
        print(f"WARN: resume ckpt missing ({args.resume_ckpt}); training from layer init", flush=True)

    song_filter = args.song_filter or None
    print("materialize_train…", flush=True)
    train_windows, cond_by_ctx, train_meta = materialize_from_shards(
        shards=shards,
        teacher=teacher,
        tokenizer=tokenizer,
        args_inf=args_inf,
        song_filter=song_filter,
        split="train",
    )
    print("materialize_holdout…", flush=True)
    hold_windows, cond_hold, hold_meta = materialize_from_shards(
        shards=shards,
        teacher=teacher,
        tokenizer=tokenizer,
        args_inf=args_inf,
        song_filter=song_filter,
        split="holdout",
    )
    # Merge cond kwargs
    for k, v in cond_hold.items():
        cond_by_ctx.setdefault(k, v)

    # proj_out vocab (timing dumps can exceed this; mask those targets)
    vocab_size = int(get_decoder(teacher).embed_tokens.num_embeddings)
    try:
        vocab_size = int(teacher.transformer.proj_out.weight.shape[0])
    except Exception:
        pass
    print(f"proj_out_vocab={vocab_size}", flush=True)
    oov_masked = 0
    kept_train = []
    for w in train_windows:
        w = dict(w)
        lab = sanitize_labels(w["labels"], vocab_size)
        oov_masked += int(((w["labels"] >= 0) & (w["labels"] >= vocab_size)).sum().item())
        w["labels"] = lab
        if window_has_supervised(lab):
            kept_train.append(w)
    train_windows = kept_train
    kept_hold = []
    for w in hold_windows:
        w = dict(w)
        # holdout eval uses dump_tokens logits slice — keep labels sanitized for consistency
        w["labels"] = sanitize_labels(w["labels"], vocab_size)
        kept_hold.append(w)
    hold_windows = kept_hold

    role_counts = {}
    for w in train_windows:
        role_counts[w["role"]] = role_counts.get(w["role"], 0) + 1
    print(
        f"train_windows={len(train_windows)} roles={role_counts} oov_label_positions_masked={oov_masked}",
        flush=True,
    )
    print(
        f"holdout_windows={len(hold_windows)} roles="
        f"{ {r: sum(1 for w in hold_windows if w['role']==r) for r in ('map0','map_rest','timing')} }",
        flush=True,
    )
    if not train_windows:
        raise SystemExit("no train windows")
    if not hold_windows:
        raise SystemExit("no holdout windows")

    weights = {"map0": args.w_map0, "map_rest": args.w_map_rest, "timing": args.w_timing}

    print("eval_baseline_holdout…", flush=True)
    baseline = eval_acceptance_sliced(
        teacher,
        draft,
        windows=hold_windows,
        cond_kwargs_by_ctx=cond_by_ctx,
        precision=args.precision,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(json.dumps({"baseline_holdout": baseline}, indent=2), flush=True)

    trainable = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    draft.train()
    t_train = time.perf_counter()
    losses: list[float] = []
    ce_losses: list[float] = []
    kl_losses: list[float] = []
    skipped = 0
    log_every = max(50, args.train_steps // 40)
    for step in range(args.train_steps):
        w = sample_window(train_windows, weights, rng)
        ctx = w.get("context_type", "map")
        cond_kwargs = cond_by_ctx.get(ctx) or cond_by_ctx["map"]
        opt.zero_grad(set_to_none=True)
        loss, ce_v, kl_v = distill_ce_kl_loss(
            draft,
            teacher,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            labels=w["labels"],
            song_position=w["song_pos"],
            draft_precision=train_precision,
            teacher_precision=args.precision,
            cond_kwargs=cond_kwargs,
            kl_weight=args.kl_weight,
            kl_temp=args.kl_temp,
        )
        if not torch.isfinite(loss):
            skipped += 1
            if step == 0 or (step + 1) % log_every == 0:
                print(f"step={step+1}/{args.train_steps} loss=nan skip role={w['role']}", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().float().cpu()))
        ce_losses.append(ce_v)
        kl_losses.append(kl_v)
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == args.train_steps:
            print(
                f"step={step+1}/{args.train_steps} loss={losses[-1]:.4f} "
                f"ce={ce_v:.4f} kl={kl_v:.4f} role={w['role']} seq={w['sequence_index']}",
                flush=True,
            )
    train_s = time.perf_counter() - t_train

    print("eval_after_holdout…", flush=True)
    after = eval_acceptance_sliced(
        teacher,
        draft,
        windows=hold_windows,
        cond_kwargs_by_ctx=cond_by_ctx,
        precision=train_precision,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gate = gate_decision(after, baseline_rest=args.baseline_rest_e)
    print(json.dumps({"after_holdout": after, "gate": gate}, indent=2), flush=True)

    if args.ckpt_out is not None:
        args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "draft_state_dict": draft.state_dict(),
                "init_layers": list(DRAFT_INIT_LAYERS),
                "schema": "s57-continuation-draft-v1",
                "train_steps": args.train_steps,
                "kl_weight": args.kl_weight,
                "kl_temp": args.kl_temp,
                "lr": args.lr,
                "resumed_from": resumed_from,
                "weights": weights,
                "tip_commit": "55949274",
                "gate": gate,
            },
            args.ckpt_out,
        )
        print(f"wrote_ckpt {args.ckpt_out}", flush=True)

    result = {
        "schema": "s57-continuation-distill-train-v1",
        "tip_commit": "55949274",
        "shards": str(args.shards),
        "resume_ckpt": resumed_from,
        "precision": args.precision,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "kl_weight": args.kl_weight,
        "kl_temp": args.kl_temp,
        "sample_weights": weights,
        "train_wall_seconds": train_s,
        "skipped_nonfinite_steps": skipped,
        "train_loss_last": losses[-1] if losses else None,
        "train_meta": train_meta,
        "holdout_meta": hold_meta,
        "baseline_holdout": baseline,
        "after_holdout": after,
        "gate": gate,
        "wire_runtime": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "note": (
            "§57 offline continuation(+timing) distill. "
            "Do not wire turbo until held-out map_rest clears and in-loop full-song confirms. "
            "Strict rejection-sampling remains merge candidate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)
    print(
        f"GATE {gate['decision']} map_rest_E={gate['map_rest_E']:.4f} "
        f"timing_E={gate['timing_E']:.4f} map_wE={gate['map_wE']:.4f} "
        f"full_wE={gate['full_wE']:.4f} wire=N",
        flush=True,
    )


if __name__ == "__main__":
    main()
