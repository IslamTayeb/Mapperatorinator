#!/usr/bin/env python3
"""§37 tiny-draft train + acceptance smoke (Track C).

Build a 2-layer same-width decoder draft (init from teacher layers [0,1]),
short CE on tip SALVALAI map dumps, then measure rejection-sampling
E[accepted/step] @ temp 0.9 / top-p 0.9 vs frozen teacher.

Acceptance probe only — gates turbo runtime. Not a production TPS claim.
TIER1 evidence pack required before any 500 / turbo ship claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf
from slider import Beatmap

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__import__("config")

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import (  # noqa: E402
    GenerationConfig,
    Preprocessor,
    Processor,
)
from osuT5.osuT5.tokenizer import ContextType  # noqa: E402

PRIMARY_GAMMA = 5
DRAFT_INIT_LAYERS = (0, 1)


def load_args(config_name: str, overrides: list[str]):
    config_dir = REPO / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(config_name=config_name.removeprefix("inference/"), overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def load_dump_tokens(profile_path: Path) -> dict[tuple[str, int], list[int]]:
    payload = json.loads(profile_path.read_text())
    out: dict[tuple[str, int], list[int]] = {}
    for rec in payload.get("generation", []):
        ids = rec.get("generated_token_ids")
        if not (isinstance(ids, list) and ids and isinstance(ids[0], int)):
            continue
        out[(str(rec.get("context_type")), int(rec.get("sequence_index")))] = [int(x) for x in ids]
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
    cands = [p for p in sorted(run_root.glob("**/beatmap*.osu")) if p.suffix == ".osu"]
    if not cands:
        raise SystemExit(f"no .osu under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def expected_accepted(alpha: float, gamma: int) -> float:
    if alpha >= 1.0 - 1e-12:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def apply_temp_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    scores = logits.float() / temperature
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_scores, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    remove = cumsum - sorted_probs > top_p
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, torch.finfo(sorted_scores.dtype).min)
    restored = torch.full_like(scores, torch.finfo(scores.dtype).min)
    restored.scatter_(-1, sorted_idx, sorted_scores)
    return F.softmax(restored, dim=-1)


def acceptance_alpha(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.minimum(p, q).sum(dim=-1)


def get_decoder(model: nn.Module) -> nn.Module:
    return model.transformer.model.decoder


def build_two_layer_draft(teacher: nn.Module, init_layers: tuple[int, ...] = DRAFT_INIT_LAYERS) -> nn.Module:
    """Deep-copy teacher; keep only two decoder layers (init from teacher indices)."""
    if len(init_layers) != 2:
        raise SystemExit(f"expected 2 init layers, got {init_layers}")
    draft = copy.deepcopy(teacher)
    dec_t = get_decoder(teacher)
    dec_d = get_decoder(draft)
    n = len(dec_t.layers)
    for i in init_layers:
        if i < 0 or i >= n:
            raise SystemExit(f"init layer {i} out of range for {n} layers")
    dec_d.layers = nn.ModuleList([copy.deepcopy(dec_t.layers[i]) for i in init_layers])
    # Freeze everything that is not draft decoder stack / final norm / lm head path
    for p in draft.parameters():
        p.requires_grad = False
    for p in dec_d.layers.parameters():
        p.requires_grad = True
    for p in dec_d.layer_norm.parameters():
        p.requires_grad = True
    # proj_out / decoder_embedder if present
    for name in ("proj_out", "lm_head", "decoder_embedder"):
        mod = getattr(draft, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    draft.train()
    return draft


@torch.no_grad()
def teacher_force_logits(
    model,
    *,
    frames: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    song_position: torch.Tensor | None,
    precision: str,
    cond_kwargs: dict[str, Any],
) -> torch.Tensor:
    kw = {}
    for k, v in cond_kwargs.items():
        if isinstance(v, torch.Tensor):
            vv = v.to(model.device)
            if k != "inputs" and vv.dtype == torch.float32 and precision == "fp16":
                vv = vv.half()
            kw[k] = vv
        else:
            kw[k] = v
    frames = frames.to(model.device)
    dec = decoder_input_ids.to(model.device)
    attn = decoder_attention_mask.to(model.device)
    if song_position is not None:
        kw["song_position"] = song_position.to(model.device)
    was_training = model.training
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(precision == "fp16")):
        out = model.forward(
            frames=frames,
            decoder_input_ids=dec,
            decoder_attention_mask=attn,
            use_cache=False,
            **kw,
        )
    if was_training:
        model.train()
    return out.logits.float()


def teacher_force_loss(
    model,
    *,
    frames: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    labels: torch.Tensor,
    song_position: torch.Tensor | None,
    precision: str,
    cond_kwargs: dict[str, Any],
) -> torch.Tensor:
    kw = {}
    for k, v in cond_kwargs.items():
        if isinstance(v, torch.Tensor):
            vv = v.to(model.device)
            if k != "inputs" and vv.dtype == torch.float32 and precision == "fp16":
                vv = vv.half()
            kw[k] = vv
        else:
            kw[k] = v
    frames = frames.to(model.device)
    dec = decoder_input_ids.to(model.device)
    attn = decoder_attention_mask.to(model.device)
    labels = labels.to(model.device)
    if song_position is not None:
        kw["song_position"] = song_position.to(model.device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(precision == "fp16")):
        out = model.forward(
            frames=frames,
            decoder_input_ids=dec,
            decoder_attention_mask=attn,
            labels=labels,
            use_cache=False,
            **kw,
        )
    if out.loss is None:
        # Fallback CE on logits vs labels
        logits = out.logits.float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return loss
    return out.loss


def prepare_contexts(processor, args_inf, generation_config, song_length: float, osu_path: Path):
    tip_beatmap = Beatmap.from_path(osu_path)
    timing = list(tip_beatmap.timing_points)
    if not timing:
        raise SystemExit(f"tip .osu has no timing points: {osu_path}")
    extra_in_context = {ContextType.TIMING: timing}
    output_type = [t for t in list(args_inf.output_type) if t != ContextType.TIMING]
    gen_in, gen_out, req_special = processor._get_viable_template(
        in_context=[ContextType.TIMING],
        out_context=output_type,
        extra_in_context=extra_in_context,
        gamemode=generation_config.gamemode,
    )
    model_kwargs = processor._get_model_cond_kwargs(generation_config)
    cond_kwargs = {k: v for k, v in model_kwargs.items() if k != "song_position"}
    in_ctx = processor.get_in_context(
        in_context=gen_in,
        beatmap_path=None,
        extra_in_context=extra_in_context,
        song_length=song_length,
    )
    out_ctx = processor.get_out_context(
        out_context=gen_out,
        generation_config=generation_config,
        given_context=[ContextType.TIMING],
        beatmap_path=None,
        extra_in_context=extra_in_context,
        song_length=song_length,
        verbose=False,
    )
    map_i = next(i for i, c in enumerate(out_ctx) if c["context_type"] == ContextType.MAP)
    return in_ctx, out_ctx, map_i, req_special, cond_kwargs, len(timing)


def iter_map_windows(
    *,
    processor,
    sequences,
    dumps,
    in_ctx,
    out_ctx,
    map_i,
    req_special,
    song_length,
    max_windows: int,
):
    map_ctx = out_ctx[map_i]
    done = 0
    for sequence_index, (frames, frame_time) in enumerate(zip(*sequences[:2])):
        if done >= max_windows:
            break
        key = ("map", sequence_index)
        if key not in dumps:
            raise SystemExit(f"missing map dump {key}")
        dump_tokens = dumps[key]
        if len(dump_tokens) < 1:
            continue
        trim_lookback = sequence_index != 0 and processor.types_first and processor.lookback_time > 0
        trim_lookahead = sequence_index != len(sequences[0]) - 1
        frames_p = processor.prepare_frames(frames)
        frame_time_v = frame_time.item()
        cond_prompt, uncond_prompt = processor.get_prompts(
            processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
            processor.prepare_context_sequences(out_ctx[: map_i + 1], frame_time_v, True, req_special),
        )
        [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
        prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())].cpu()
        gen = torch.tensor(dump_tokens, dtype=torch.long)
        full = torch.cat([prompt_ids, gen], dim=0).unsqueeze(0)
        if full.shape[1] < 2:
            processor.add_predicted_tokens_to_context(
                map_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
            )
            continue
        dec_in = full[:, :-1]
        labels = full[:, 1:].clone()
        # Only supervise map-token positions (after prompt)
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
            "map_ctx": map_ctx,
        }
        processor.add_predicted_tokens_to_context(
            map_ctx, gen, frame_time_v, trim_lookback, trim_lookahead
        )
        done += 1


def eval_acceptance(
    teacher,
    draft,
    *,
    windows: list[dict],
    cond_kwargs,
    precision: str,
    temperature: float,
    top_p: float,
) -> dict:
    teacher.eval()
    draft.eval()
    alphas: list[float] = []
    for w in windows:
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
        p = apply_temp_top_p(lp, temperature, top_p)
        q = apply_temp_top_p(lq, temperature, top_p)
        a = acceptance_alpha(p, q)
        alphas.extend(float(x) for x in a.tolist())
    mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
    e_primary = expected_accepted(mean_alpha, PRIMARY_GAMMA)
    if e_primary >= 1.8:
        decision = "GO_SECTION_36_TURBO_SCAFFOLD"
        gate = "E>=1.8 → scaffold turbo runtime (still need TIER1 before 500)"
    elif e_primary >= 1.3:
        decision = "CONTINUE_TRAIN_OR_TURBO_WITH_TREE"
        gate = "1.3–1.8 → continue train / tree drafts; TIER1 before ship"
    else:
        decision = "CONTINUE_TRAIN_OR_EAGLE"
        gate = "E<1.3 → more train or EAGLE revisit; no turbo claim"
    return {
        "positions": len(alphas),
        "mean_alpha": mean_alpha,
        "E_accepted_per_step_primary": e_primary,
        "E_accepted_per_step_by_gamma": {str(g): expected_accepted(mean_alpha, g) for g in range(3, 9)},
        "gate_decision": decision,
        "gate_text": gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path("/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"),
    )
    ap.add_argument("--audio", type=Path, default=Path("/work/imt11/Mapperatorinator/data/salvalai.mp3"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt-out", type=Path, default=None)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-map-windows", type=int, default=16, help="windows for train+eval smoke")
    ap.add_argument("--train-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    profile_path = find_profile(args.dump_run)
    dumps = load_dump_tokens(profile_path)
    map_keys = sorted(k for k in dumps if k[0] == "map")
    print(f"dump={profile_path}", flush=True)
    print(f"map_windows_available={len(map_keys)}", flush=True)

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
    print(f"teacher_loaded_s={time.perf_counter()-t0:.1f}", flush=True)

    n_layers = len(get_decoder(teacher).layers)
    if n_layers != 12:
        raise SystemExit(f"expected 12 decoder layers, got {n_layers}")
    print(f"draft_init_layers={list(DRAFT_INIT_LAYERS)} of {n_layers}", flush=True)
    draft = build_two_layer_draft(teacher, DRAFT_INIT_LAYERS)
    # Train draft in fp32 even when teacher eval is fp16 — fp16 CE blew up (50146182 NaN).
    draft = draft.float()
    draft.to(device)
    train_precision = "fp32"

    preprocessor = Preprocessor(args_inf, parallel=False)
    # Processor binds to teacher for prompt/tokenization helpers only
    processor = Processor(args_inf, teacher, tokenizer)
    audio = preprocessor.load(str(args.audio))
    sequences = preprocessor.segment(audio)
    song_length = sequences[2]
    print(f"sequences={len(sequences[0])} song_ms={song_length}", flush=True)

    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    osu_path = find_osu(args.dump_run)
    print(f"timing_source_osu={osu_path}", flush=True)

    # Fresh contexts for train materialization
    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = prepare_contexts(
        processor, args_inf, generation_config, song_length, osu_path
    )
    print(f"timing_points={n_timing}", flush=True)

    max_windows = min(args.max_map_windows, len(map_keys)) if args.max_map_windows > 0 else len(map_keys)
    windows = list(
        iter_map_windows(
            processor=processor,
            sequences=sequences,
            dumps=dumps,
            in_ctx=in_ctx,
            out_ctx=out_ctx,
            map_i=map_i,
            req_special=req_special,
            song_length=song_length,
            max_windows=max_windows,
        )
    )
    print(f"materialized_windows={len(windows)}", flush=True)
    if not windows:
        raise SystemExit("no train windows")

    # Baseline acceptance (init draft, no train)
    print("eval_baseline…", flush=True)
    baseline = eval_acceptance(
        teacher,
        draft,
        windows=windows,
        cond_kwargs=cond_kwargs,
        precision=args.precision,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"baseline mean_alpha={baseline['mean_alpha']:.4f} "
        f"E={baseline['E_accepted_per_step_primary']:.4f}",
        flush=True,
    )

    trainable = [p for p in draft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    draft.train()
    t_train = time.perf_counter()
    losses: list[float] = []
    skipped_nonfinite = 0
    for step in range(args.train_steps):
        w = windows[step % len(windows)]
        opt.zero_grad(set_to_none=True)
        loss = teacher_force_loss(
            draft,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            labels=w["labels"],
            song_position=w["song_pos"],
            precision=train_precision,
            cond_kwargs=cond_kwargs,
        )
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            losses.append(float("nan"))
            if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.train_steps:
                print(f"step={step+1}/{args.train_steps} loss=nan (skip)", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().float().cpu()))
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.train_steps:
            print(f"step={step+1}/{args.train_steps} loss={losses[-1]:.4f}", flush=True)
    train_s = time.perf_counter() - t_train

    print("eval_after_train…", flush=True)
    after = eval_acceptance(
        teacher,
        draft,
        windows=windows,
        cond_kwargs=cond_kwargs,
        precision=train_precision,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"after mean_alpha={after['mean_alpha']:.4f} "
        f"E={after['E_accepted_per_step_primary']:.4f} "
        f"gate={after['gate_decision']}",
        flush=True,
    )

    if args.ckpt_out is not None:
        args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "draft_state_dict": draft.state_dict(),
                "init_layers": list(DRAFT_INIT_LAYERS),
                "train_steps": args.train_steps,
                "tip_commit": "55949274",
            },
            args.ckpt_out,
        )
        print(f"wrote_ckpt {args.ckpt_out}", flush=True)

    result = {
        "schema": "s37-tiny-draft-train-smoke-v1",
        "tip_commit": "55949274",
        "dump_run": str(args.dump_run),
        "dump_profile": str(profile_path),
        "model_path": str(args_inf.model_path),
        "precision": args.precision,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "teacher_decoder_layers": n_layers,
        "draft_decoder_layers": 2,
        "draft_init_layers": list(DRAFT_INIT_LAYERS),
        "draft_config": "2-layer same-width CE distill smoke",
        "map_windows": len(windows),
        "train_steps": args.train_steps,
        "lr": args.lr,
        "train_wall_seconds": train_s,
        "train_precision": train_precision,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "skipped_nonfinite_steps": skipped_nonfinite,
        "prior_smoke_job": "50146182",
        "fix": "fp32 draft train + grad clip (fp16 CE NaN)",
        "baseline_acceptance": baseline,
        "after_train_acceptance": after,
        "gate_decision": after["gate_decision"],
        "gate_text": after["gate_text"],
        "E_accepted_per_step_primary": after["E_accepted_per_step_primary"],
        "mean_alpha": after["mean_alpha"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "note": (
            "§37 smoke only. Not production TPS. TIER1 pack required before turbo/500 claim. "
            "INT8 hybrid is not FP16."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
