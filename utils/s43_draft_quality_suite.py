#!/usr/bin/env python3
"""§43 draft-quality suite (W4) — offline acceptance / distill / sweeps.

No turbo generate_window / canary / §39 wiring.

Phases:
  (a) E[accepted/step] for tip SALVALAI-trained draft on ≥2 held-out songs
  (b) multi-song distill shards + longer CE/KL train (optional widen via more steps)
  (c) K/γ + draft-temperature sweeps from dumped logits → acceptance-vs-cost
  (d) simulate tree drafts (K>1) and cheaper 1-layer / half-width cost proxies

Not a production TPS claim. Does not wire inference_engine=turbo.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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
eval_acceptance = s37.eval_acceptance
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

FIVE_SONG_ROOT = Path(
    "/work/imt11/Mapperatorinator/runs/five-song-profile-20260702-232516-8a2de72/"
    "after_separate/official"
)
DATA_ROOT = Path("/work/imt11/Mapperatorinator/data/five-song-profile")
TIP_SALVALAI_DUMP = Path(
    "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
)
TIP_DRAFT_CKPT = Path(
    "/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt"
)

SONG_CATALOG = {
    "salvalai": {
        "audio": DATA_ROOT / "salvalai.mp3",
        "profile_dir": FIVE_SONG_ROOT / "salvalai",
        "role": "train_ref",
        "alt_dump_run": TIP_SALVALAI_DUMP,  # tip FP16 auth preferred when present
    },
    "pegasus": {
        "audio": DATA_ROOT / "pegasus.mp3",
        "profile_dir": FIVE_SONG_ROOT / "pegasus",
        "role": "multi_train",
    },
    "lambada": {
        "audio": DATA_ROOT / "lambada.mp3",
        "profile_dir": FIVE_SONG_ROOT / "lambada",
        "role": "multi_train",
    },
    "ela-ke-leitada": {
        "audio": DATA_ROOT / "ela-ke-leitada.mp3",
        "profile_dir": FIVE_SONG_ROOT / "ela-ke-leitada",
        "role": "held_out",
    },
    "nube-negra": {
        "audio": DATA_ROOT / "nube-negra.mp3",
        "profile_dir": FIVE_SONG_ROOT / "nube-negra",
        "role": "held_out",
    },
}


def build_n_layer_draft(teacher: nn.Module, init_layers: tuple[int, ...]) -> nn.Module:
    draft = copy.deepcopy(teacher)
    dec_t = get_decoder(teacher)
    dec_d = get_decoder(draft)
    n = len(dec_t.layers)
    for i in init_layers:
        if i < 0 or i >= n:
            raise SystemExit(f"init layer {i} out of range for {n} layers")
    dec_d.layers = nn.ModuleList([copy.deepcopy(dec_t.layers[i]) for i in init_layers])
    for p in draft.parameters():
        p.requires_grad = False
    for p in dec_d.layers.parameters():
        p.requires_grad = True
    for p in dec_d.layer_norm.parameters():
        p.requires_grad = True
    for name in ("proj_out", "lm_head", "decoder_embedder"):
        mod = getattr(draft, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
    draft.train()
    return draft


def resolve_song_paths(song_id: str) -> dict[str, Path]:
    meta = SONG_CATALOG[song_id]
    dump_run = meta.get("alt_dump_run")
    if dump_run is not None and Path(dump_run).exists():
        profile = find_profile(Path(dump_run))
        osu = find_osu(Path(dump_run))
        return {"audio": meta["audio"], "profile": profile, "osu": osu, "dump_run": Path(dump_run)}
    profile_dir = meta["profile_dir"]
    profiles = sorted(profile_dir.glob("*.profile.json"))
    if not profiles:
        raise SystemExit(f"no profile for {song_id} under {profile_dir}")
    profile = profiles[0]
    osus = sorted((profile_dir / "output").glob("beatmap*.osu")) if (profile_dir / "output").exists() else []
    if not osus:
        osus = sorted(profile_dir.glob("**/beatmap*.osu"))
    if not osus:
        raise SystemExit(f"no .osu for {song_id}")
    return {"audio": meta["audio"], "profile": profile, "osu": osus[0], "dump_run": profile_dir}


def materialize_song_windows(
    *,
    song_id: str,
    teacher,
    tokenizer,
    args_inf,
    max_map_windows: int,
) -> tuple[list[dict], dict[str, Any], dict[str, Any]]:
    paths = resolve_song_paths(song_id)
    dumps = load_dump_tokens(paths["profile"])
    map_keys = sorted(k for k in dumps if k[0] == "map")
    preprocessor = Preprocessor(args_inf, parallel=False)
    # bind processor to teacher for prompt helpers
    processor = Processor(args_inf, teacher, tokenizer)
    # override audio
    args_inf.audio_path = str(paths["audio"])
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
    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = prepare_contexts(
        processor, args_inf, generation_config, song_length, paths["osu"]
    )
    max_w = min(max_map_windows, len(map_keys)) if max_map_windows > 0 else len(map_keys)
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
            max_windows=max_w,
        )
    )
    meta = {
        "song_id": song_id,
        "role": SONG_CATALOG[song_id]["role"],
        "audio": str(paths["audio"]),
        "profile": str(paths["profile"]),
        "osu": str(paths["osu"]),
        "dump_run": str(paths["dump_run"]),
        "map_windows_available": len(map_keys),
        "map_windows_used": len(windows),
        "song_ms": float(song_length),
        "timing_points": n_timing,
        "n_sequences": len(sequences[0]),
    }
    return windows, cond_kwargs, meta


@torch.no_grad()
def collect_map_logits(
    teacher,
    draft,
    *,
    windows: list[dict],
    cond_kwargs,
    teacher_precision: str,
    draft_precision: str,
) -> dict[str, Any]:
    """Dump per-position teacher/draft logits on map tokens (float16 on CPU)."""
    teacher.eval()
    draft.eval()
    packs: list[dict[str, Any]] = []
    for w in windows:
        logits_p = teacher_force_logits(
            teacher,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            song_position=w["song_pos"],
            precision=teacher_precision,
            cond_kwargs=cond_kwargs,
        )
        logits_q = teacher_force_logits(
            draft,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            song_position=w["song_pos"],
            precision=draft_precision,
            cond_kwargs=cond_kwargs,
        )
        start = max(w["prompt_len"] - 1, 0)
        end = start + len(w["dump_tokens"])
        lp = logits_p[0, start:end].half().cpu()
        lq = logits_q[0, start:end].half().cpu()
        if lp.numel() == 0:
            continue
        packs.append(
            {
                "sequence_index": w["sequence_index"],
                "logits_teacher": lp,
                "logits_draft": lq,
                "n_tokens": int(lp.shape[0]),
            }
        )
    return {
        "n_windows": len(packs),
        "n_tokens": sum(p["n_tokens"] for p in packs),
        "windows": packs,
    }


def alphas_from_logit_pack(pack: dict[str, Any], temperature: float, top_p: float) -> list[float]:
    """Per-position α; drop rows with non-finite teacher/draft logits."""
    alphas: list[float] = []
    for w in pack["windows"]:
        lt = w["logits_teacher"].float()
        ld = w["logits_draft"].float()
        finite = torch.isfinite(lt).all(dim=-1) & torch.isfinite(ld).all(dim=-1)
        if int(finite.sum()) == 0:
            continue
        p = apply_temp_top_p(lt[finite], temperature, top_p)
        q = apply_temp_top_p(ld[finite], temperature, top_p)
        a = acceptance_alpha(p, q)
        alphas.extend(float(x) for x in a.tolist() if x == x)  # finite
    return alphas


def summarize_alphas(alphas: list[float], gamma: int = PRIMARY_GAMMA) -> dict[str, Any]:
    mean_alpha = sum(alphas) / len(alphas) if alphas else 0.0
    e = expected_accepted(mean_alpha, gamma)
    return {
        "positions": len(alphas),
        "mean_alpha": mean_alpha,
        "E_accepted_per_step": e,
        "E_by_gamma": {str(g): expected_accepted(mean_alpha, g) for g in range(3, 9)},
    }


def simulate_tree_accepted(
    alphas: list[float],
    *,
    gamma: int,
    k_branches: int,
    n_trials: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """MC: K independent γ-drafts; take longest accepted prefix; +1 bonus token.

    Per-position accept ~ Bern(α_t) using cycling position alphas (teacher-forced α).
    """
    if not alphas:
        return {"E_accepted_per_step": 1.0, "trials": 0}
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor(alphas, dtype=torch.float32)
    n = a.numel()
    totals = []
    for trial in range(n_trials):
        start = int(torch.randint(0, n, (1,), generator=g).item())
        best = 0
        for _b in range(k_branches):
            accepted = 0
            for j in range(gamma):
                idx = (start + j) % n
                u = torch.rand((), generator=g)
                if float(u) <= float(a[idx]):
                    accepted += 1
                else:
                    break
            best = max(best, accepted)
        totals.append(best + 1)  # +1 verified bonus / resample token
    mean_e = sum(totals) / len(totals)
    return {
        "E_accepted_per_step": mean_e,
        "trials": n_trials,
        "gamma": gamma,
        "k_branches": k_branches,
        "method": "mc_independent_branches_teacher_forced_alpha",
    }


def cost_units(*, gamma: int, k_branches: int, draft_rel_cost: float) -> float:
    """Relative step cost in verify-forward units.

    Assume one batched verify ≈ 1.0 and each draft token costs draft_rel_cost
    relative to a verify token (2-layer same-width ≈ 2/12 ≈ 0.167 of full teacher;
    we report both absolute-ish and normalized-to-2layer).
    """
    return float(k_branches) * float(gamma) * float(draft_rel_cost) + 1.0


def acceptance_vs_cost_table(
    alphas: list[float],
    *,
    draft_name: str,
    draft_rel_cost: float,
    temperatures: list[float],
    gammas: list[int],
    ks: list[int],
    top_p: float,
    logit_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temp in temperatures:
        if logit_pack is not None and abs(temp - 0.9) > 1e-9:
            a_list = alphas_from_logit_pack(logit_pack, temp, top_p)
        else:
            a_list = alphas
        base = summarize_alphas(a_list, PRIMARY_GAMMA)
        for gamma in gammas:
            for k in ks:
                if k == 1:
                    e = expected_accepted(base["mean_alpha"], gamma)
                    method = "leviathan_closed_form"
                else:
                    sim = simulate_tree_accepted(a_list, gamma=gamma, k_branches=k)
                    e = sim["E_accepted_per_step"]
                    method = sim["method"]
                c = cost_units(gamma=gamma, k_branches=k, draft_rel_cost=draft_rel_cost)
                rows.append(
                    {
                        "draft": draft_name,
                        "draft_rel_cost": draft_rel_cost,
                        "temperature": temp,
                        "top_p": top_p,
                        "gamma": gamma,
                        "K": k,
                        "mean_alpha": base["mean_alpha"],
                        "E_accepted_per_step": e,
                        "cost_verify_units": c,
                        "E_over_cost": e / c if c > 0 else 0.0,
                        "method": method,
                    }
                )
    return rows


def build_multi_song_shards(song_ids: list[str], out_path: Path) -> dict[str, Any]:
    windows_meta = []
    n_tokens = 0
    for sid in song_ids:
        paths = resolve_song_paths(sid)
        dumps = load_dump_tokens(paths["profile"])
        rows = []
        for (ctx, idx), ids in sorted(dumps.items()):
            if ctx != "map":
                continue
            rows.append(
                {
                    "song_id": sid,
                    "context_type": ctx,
                    "sequence_index": idx,
                    "generated_token_ids": ids,
                    "generated_tokens": len(ids),
                }
            )
            n_tokens += len(ids)
        windows_meta.append(
            {
                "song_id": sid,
                "profile": str(paths["profile"]),
                "n_map_windows": len(rows),
                "n_tokens": sum(r["generated_tokens"] for r in rows),
            }
        )
        # write per-song shard
        song_out = out_path.parent / f"distill_shards_{sid}.json"
        song_out.write_text(
            json.dumps(
                {
                    "schema": "s43-distill-shards-v1",
                    "song_id": sid,
                    "source_profile": str(paths["profile"]),
                    "n_windows": len(rows),
                    "n_tokens": sum(r["generated_tokens"] for r in rows),
                    "windows": rows,
                },
                indent=2,
            )
            + "\n"
        )
    payload = {
        "schema": "s43-multi-song-distill-index-v1",
        "songs": windows_meta,
        "n_songs": len(song_ids),
        "n_tokens_total": n_tokens,
        "note": "Hard-label shards for §43 multi-song CE/KL. Acceptance claimed separately.",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def pick_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer E≥1.8, then maximize E/cost; else E≥1.3; else best E/cost."""
    if not rows:
        return {}
    def key(r):
        return (r["E_over_cost"], r["E_accepted_per_step"], -r["cost_verify_units"])

    ge18 = [r for r in rows if r["E_accepted_per_step"] >= 1.8]
    ge13 = [r for r in rows if r["E_accepted_per_step"] >= 1.3]
    pool = ge18 or ge13 or rows
    best = max(pool, key=key)
    return {
        "recommended": best,
        "pool": "E>=1.8" if ge18 else ("E>=1.3" if ge13 else "best_E_over_cost_below_floor"),
        "n_candidates": len(pool),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draft-ckpt", type=Path, default=TIP_DRAFT_CKPT)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-map-windows", type=int, default=0, help="0=all")
    ap.add_argument(
        "--held-out-songs",
        default="ela-ke-leitada,nube-negra",
        help="comma songs never in tip train; used for (a)",
    )
    ap.add_argument(
        "--multi-train-songs",
        default="salvalai,pegasus,lambada",
        help="songs for multi-song distill (b)",
    )
    ap.add_argument("--multi-train-steps", type=int, default=4000)
    ap.add_argument("--cheap-train-steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument(
        "--skip-multi-train-if-heldout-e-ge",
        type=float,
        default=1.8,
        help="skip multi-song train when mean held-out E already clears this",
    )
    ap.add_argument("--force-multi-train", action="store_true")
    ap.add_argument("--skip-logit-dump", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logit_dir = out_dir / "logit_dumps"
    logit_dir.mkdir(parents=True, exist_ok=True)

    held_out = [s.strip() for s in args.held_out_songs.split(",") if s.strip()]
    multi_train = [s.strip() for s in args.multi_train_songs.split(",") if s.strip()]
    eval_songs = ["salvalai"] + held_out

    print("building multi-song shards…", flush=True)
    shard_index = build_multi_song_shards(sorted(set(multi_train + held_out + ["salvalai"])), out_dir / "distill_shards_index.json")

    args_inf = load_args(
        "profile_salvalai",
        [
            f"audio_path={SONG_CATALOG['salvalai']['audio']}",
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
    print(f"loading tip draft ckpt {args.draft_ckpt}", flush=True)
    tip_draft = build_two_layer_draft(teacher, DRAFT_INIT_LAYERS).float().to(device)
    ckpt = torch.load(args.draft_ckpt, map_location="cpu")
    state = ckpt.get("draft_state_dict", ckpt)
    missing, unexpected = tip_draft.load_state_dict(state, strict=False)
    print(f"tip_draft missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    tip_draft.eval()

    # --- (a) held-out acceptance for tip SALVALAI-only draft ---
    phase_a: dict[str, Any] = {"schema": "s43-phase-a-heldout-v1", "songs": {}}
    song_windows: dict[str, tuple[list[dict], dict]] = {}
    tip_alphas: dict[str, list[float]] = {}
    tip_logit_packs: dict[str, dict[str, Any]] = {}

    for sid in eval_songs:
        print(f"materialize+eval tip_draft song={sid}", flush=True)
        windows, cond_kwargs, meta = materialize_song_windows(
            song_id=sid,
            teacher=teacher,
            tokenizer=tokenizer,
            args_inf=args_inf,
            max_map_windows=args.max_map_windows,
        )
        song_windows[sid] = (windows, cond_kwargs)
        acc = eval_acceptance(
            teacher,
            tip_draft,
            windows=windows,
            cond_kwargs=cond_kwargs,
            precision=args.precision,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # collect alphas via logit pack (also used for sweeps)
        if not args.skip_logit_dump:
            pack = collect_map_logits(
                teacher,
                tip_draft,
                windows=windows,
                cond_kwargs=cond_kwargs,
                teacher_precision=args.precision,
                draft_precision="fp32",
            )
            tip_logit_packs[sid] = pack
            torch.save(pack, logit_dir / f"tip2layer_{sid}.pt")
            tip_alphas[sid] = alphas_from_logit_pack(pack, args.temperature, args.top_p)
        else:
            # reconstruct alphas only through eval (no per-pos store)
            tip_alphas[sid] = [acc["mean_alpha"]] * max(acc["positions"], 1)

        phase_a["songs"][sid] = {
            **meta,
            "acceptance": acc,
            "mean_alpha_from_dump": summarize_alphas(tip_alphas[sid])["mean_alpha"],
        }
        print(
            f"  {sid}: E={acc['E_accepted_per_step_primary']:.4f} "
            f"alpha={acc['mean_alpha']:.4f} pos={acc['positions']}",
            flush=True,
        )

    held_e = [
        phase_a["songs"][s]["acceptance"]["E_accepted_per_step_primary"] for s in held_out
    ]
    phase_a["held_out_songs"] = held_out
    phase_a["held_out_mean_E"] = sum(held_e) / len(held_e) if held_e else None
    phase_a["held_out_min_E"] = min(held_e) if held_e else None
    phase_a["train_song_E_salvalai"] = phase_a["songs"]["salvalai"]["acceptance"][
        "E_accepted_per_step_primary"
    ]
    phase_a["tip_train_job"] = "50146289"
    phase_a["tip_train_reported_E"] = 2.921
    (out_dir / "phase_a_heldout.json").write_text(json.dumps(phase_a, indent=2) + "\n")

    # --- (b) multi-song train if needed ---
    phase_b: dict[str, Any] = {
        "schema": "s43-phase-b-multsong-v1",
        "multi_train_songs": multi_train,
        "shard_index": shard_index,
        "skipped": False,
    }
    multi_draft = None
    do_multi = args.force_multi_train or (
        phase_a["held_out_mean_E"] is not None
        and phase_a["held_out_mean_E"] < args.skip_multi_train_if_heldout_e_ge
    )
    if not do_multi:
        phase_b["skipped"] = True
        phase_b["reason"] = (
            f"held_out_mean_E={phase_a['held_out_mean_E']:.4f} "
            f">= {args.skip_multi_train_if_heldout_e_ge}"
        )
        print(f"skip multi-train: {phase_b['reason']}", flush=True)
    else:
        print("multi-song train…", flush=True)
        multi_draft = build_two_layer_draft(teacher, DRAFT_INIT_LAYERS).float().to(device)
        multi_draft.load_state_dict(state, strict=False)
        # concatenate windows; use per-window cond_kwargs from last song in loop — must keep pairs
        paired: list[tuple[dict, dict]] = []
        for sid in multi_train:
            if sid in song_windows:
                wins, ck = song_windows[sid]
            else:
                wins, ck, meta = materialize_song_windows(
                    song_id=sid,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    args_inf=args_inf,
                    max_map_windows=args.max_map_windows,
                )
                song_windows[sid] = (wins, ck)
                print(f"  materialized train song {sid} windows={len(wins)}", flush=True)
            for w in wins:
                paired.append((w, ck))
        # train with song-specific cond_kwargs each step
        multi_draft.train()
        trainable = [p for p in multi_draft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=args.lr)
        losses: list[float] = []
        skipped = 0
        t_train = time.perf_counter()
        log_every = max(50, args.multi_train_steps // 40)
        for step in range(args.multi_train_steps):
            w, ck = paired[step % len(paired)]
            opt.zero_grad(set_to_none=True)
            loss, ce_v, kl_v = distill_ce_kl_loss(
                multi_draft,
                teacher,
                frames=w["frames"],
                decoder_input_ids=w["dec_in"],
                decoder_attention_mask=w["dec_attn"],
                labels=w["labels"],
                song_position=w["song_pos"],
                draft_precision="fp32",
                teacher_precision=args.precision,
                cond_kwargs=ck,
                kl_weight=args.kl_weight,
                kl_temp=args.kl_temp,
            )
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(float(loss.detach().float().cpu()))
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == args.multi_train_steps:
                print(
                    f"  multi_step={step+1}/{args.multi_train_steps} loss={losses[-1]:.4f} "
                    f"ce={ce_v:.4f} kl={kl_v:.4f}",
                    flush=True,
                )
        ckpt_path = out_dir / "draft_multsong.pt"
        torch.save(
            {
                "draft_state_dict": multi_draft.state_dict(),
                "init_layers": list(DRAFT_INIT_LAYERS),
                "train_steps": args.multi_train_steps,
                "train_songs": multi_train,
                "resumed_from": str(args.draft_ckpt),
                "tip_commit": "55949274",
                "schema": "s43-multi-song-draft-v1",
            },
            ckpt_path,
        )
        multi_draft.eval()
        phase_b["train"] = {
            "train_steps": args.multi_train_steps,
            "train_wall_seconds": time.perf_counter() - t_train,
            "train_loss_first": losses[0] if losses else None,
            "train_loss_last": losses[-1] if losses else None,
            "skipped_nonfinite_steps": skipped,
            "n_window_pairs": len(paired),
            "ckpt": str(ckpt_path),
        }
        phase_b["after_train_acceptance"] = {}
        for sid in eval_songs:
            wins, ck = song_windows[sid]
            acc = eval_acceptance(
                teacher,
                multi_draft,
                windows=wins,
                cond_kwargs=ck,
                precision=args.precision,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            phase_b["after_train_acceptance"][sid] = acc
            print(
                f"  multi after {sid}: E={acc['E_accepted_per_step_primary']:.4f}",
                flush=True,
            )
            pack = collect_map_logits(
                teacher,
                multi_draft,
                windows=wins,
                cond_kwargs=ck,
                teacher_precision=args.precision,
                draft_precision="fp32",
            )
            torch.save(pack, logit_dir / f"multi2layer_{sid}.pt")

    (out_dir / "phase_b_multsong.json").write_text(json.dumps(phase_b, indent=2) + "\n")

    # --- (d) cheaper drafts: 1-layer train + half-width cost proxy ---
    print("cheap 1-layer draft…", flush=True)
    phase_d: dict[str, Any] = {"schema": "s43-phase-d-cheap-drafts-v1"}
    # use multi-train songs if available else salvalai windows
    train_sids = multi_train if multi_train else ["salvalai"]
    paired_cheap: list[tuple[dict, dict]] = []
    for sid in train_sids:
        if sid not in song_windows:
            wins, ck, _meta = materialize_song_windows(
                song_id=sid,
                teacher=teacher,
                tokenizer=tokenizer,
                args_inf=args_inf,
                max_map_windows=args.max_map_windows,
            )
            song_windows[sid] = (wins, ck)
        wins, ck = song_windows[sid]
        for w in wins:
            paired_cheap.append((w, ck))

    one_layer = build_n_layer_draft(teacher, (0,)).float().to(device)
    # quick train
    one_layer.train()
    trainable = [p for p in one_layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    t_cheap = time.perf_counter()
    losses = []
    for step in range(args.cheap_train_steps):
        w, ck = paired_cheap[step % len(paired_cheap)]
        opt.zero_grad(set_to_none=True)
        loss, ce_v, kl_v = distill_ce_kl_loss(
            one_layer,
            teacher,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            labels=w["labels"],
            song_position=w["song_pos"],
            draft_precision="fp32",
            teacher_precision=args.precision,
            cond_kwargs=ck,
            kl_weight=args.kl_weight,
            kl_temp=args.kl_temp,
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.detach().float().cpu()))
    cheap_ckpt = out_dir / "draft_1layer.pt"
    torch.save(
        {
            "draft_state_dict": one_layer.state_dict(),
            "init_layers": [0],
            "draft_decoder_layers": 1,
            "train_steps": args.cheap_train_steps,
            "train_songs": train_sids,
            "schema": "s43-one-layer-draft-v1",
        },
        cheap_ckpt,
    )
    one_layer.eval()
    phase_d["one_layer"] = {
        "init_layers": [0],
        "train_steps": args.cheap_train_steps,
        "train_wall_seconds": time.perf_counter() - t_cheap,
        "train_loss_last": losses[-1] if losses else None,
        "ckpt": str(cheap_ckpt),
        "rel_cost_vs_2layer": 0.5,
        "acceptance": {},
    }
    one_alphas: dict[str, list[float]] = {}
    for sid in eval_songs:
        wins, ck = song_windows[sid]
        acc = eval_acceptance(
            teacher,
            one_layer,
            windows=wins,
            cond_kwargs=ck,
            precision=args.precision,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        phase_d["one_layer"]["acceptance"][sid] = acc
        pack = collect_map_logits(
            teacher,
            one_layer,
            windows=wins,
            cond_kwargs=ck,
            teacher_precision=args.precision,
            draft_precision="fp32",
        )
        torch.save(pack, logit_dir / f"one1layer_{sid}.pt")
        one_alphas[sid] = alphas_from_logit_pack(pack, args.temperature, args.top_p)
        print(f"  1layer {sid}: E={acc['E_accepted_per_step_primary']:.4f}", flush=True)

    # Half-width: cost simulation only. Acceptance proxy = 1-layer (similar FLOP budget).
    phase_d["half_width_2layer_sim"] = {
        "note": (
            "No architecture train — cost_rel=0.5 vs 2-layer full-width. "
            "Acceptance proxy uses measured 1-layer alphas (similar decode FLOP budget)."
        ),
        "rel_cost_vs_2layer": 0.5,
        "acceptance_proxy": "one_layer",
    }
    (out_dir / "phase_d_cheap.json").write_text(json.dumps(phase_d, indent=2) + "\n")

    # --- (c)+(d) sweeps / tree / cost curves ---
    print("offline sweeps…", flush=True)
    temperatures = [0.7, 0.9, 1.1]
    gammas = [3, 4, 5, 6, 7, 8]
    ks = [1, 2, 4]
    # Relative draft cost vs verify (= full teacher decode token): 2/12 and 1/12
    draft_configs = [
        ("tip_2layer_salvalai", tip_alphas, tip_logit_packs, 2.0 / float(n_layers)),
        ("one_layer", one_alphas, {s: None for s in eval_songs}, 1.0 / float(n_layers)),
        (
            "half_width_2layer_sim",
            one_alphas,
            {s: None for s in eval_songs},
            (2.0 / float(n_layers)) * 0.5,
        ),
    ]
    # attach real logit packs for tip; for others reload if present
    for sid in eval_songs:
        p = logit_dir / f"one1layer_{sid}.pt"
        if p.exists():
            # keep None in loop — load below
            pass

    all_rows: list[dict[str, Any]] = []
    per_song_tables: dict[str, Any] = {}
    for sid in eval_songs:
        song_rows: list[dict[str, Any]] = []
        for name, alpha_map, pack_map, rel in draft_configs:
            alphas = alpha_map.get(sid, [])
            pack = None
            if name == "tip_2layer_salvalai":
                pack = tip_logit_packs.get(sid)
            elif name == "one_layer":
                pp = logit_dir / f"one1layer_{sid}.pt"
                pack = torch.load(pp, map_location="cpu") if pp.exists() else None
            rows = acceptance_vs_cost_table(
                alphas,
                draft_name=name,
                draft_rel_cost=rel,
                temperatures=temperatures,
                gammas=gammas,
                ks=ks,
                top_p=args.top_p,
                logit_pack=pack,
            )
            for r in rows:
                r["song_id"] = sid
            song_rows.extend(rows)
            all_rows.extend(rows)
        per_song_tables[sid] = {
            "n_rows": len(song_rows),
            "recommendation": pick_recommendation(song_rows),
            "top5_by_E_over_cost": sorted(song_rows, key=lambda r: r["E_over_cost"], reverse=True)[:5],
        }

    # Primary recommendation: average E/cost over held-out songs for each config key
    agg: dict[tuple, list[float]] = defaultdict(list)
    agg_e: dict[tuple, list[float]] = defaultdict(list)
    meta_row: dict[tuple, dict] = {}
    for r in all_rows:
        if r["song_id"] not in held_out:
            continue
        key = (r["draft"], r["temperature"], r["gamma"], r["K"])
        agg[key].append(r["E_over_cost"])
        agg_e[key].append(r["E_accepted_per_step"])
        meta_row[key] = r
    agg_rows = []
    for key, vals in agg.items():
        e_vals = agg_e[key]
        base = meta_row[key]
        agg_rows.append(
            {
                "draft": key[0],
                "temperature": key[1],
                "gamma": key[2],
                "K": key[3],
                "draft_rel_cost": base["draft_rel_cost"],
                "held_out_mean_E": sum(e_vals) / len(e_vals),
                "held_out_mean_E_over_cost": sum(vals) / len(vals),
                "cost_verify_units": base["cost_verify_units"],
                "method": base["method"],
            }
        )
    rec = pick_recommendation(
        [
            {
                **r,
                "E_accepted_per_step": r["held_out_mean_E"],
                "E_over_cost": r["held_out_mean_E_over_cost"],
            }
            for r in agg_rows
        ]
    )

    phase_c = {
        "schema": "s43-phase-c-sweeps-v1",
        "temperatures": temperatures,
        "gammas": gammas,
        "ks": ks,
        "top_p": args.top_p,
        "held_out_songs": held_out,
        "per_song": per_song_tables,
        "held_out_aggregate_rows": sorted(
            agg_rows, key=lambda r: r["held_out_mean_E_over_cost"], reverse=True
        ),
        "recommendation": rec,
        "cost_model": (
            "cost = K*γ*draft_rel + 1 verify; draft_rel = n_draft_layers/n_teacher_layers "
            "(half-width multiplies by 0.5). Tree K>1 uses MC on teacher-forced α."
        ),
    }
    (out_dir / "phase_c_sweeps.json").write_text(json.dumps(phase_c, indent=2) + "\n")
    (out_dir / "acceptance_vs_cost_rows.json").write_text(json.dumps(all_rows, indent=2) + "\n")

    # Compact acceptance table for notes
    table = {
        "schema": "s43-acceptance-table-v1",
        "tip_draft_ckpt": str(args.draft_ckpt),
        "tip_train_job": "50146289",
        "tip_train_E_salvalai_reported": 2.921,
        "phase_a_salvalai_E": phase_a["train_song_E_salvalai"],
        "phase_a_held_out": {
            s: phase_a["songs"][s]["acceptance"]["E_accepted_per_step_primary"] for s in held_out
        },
        "phase_a_held_out_mean_E": phase_a["held_out_mean_E"],
        "phase_b_skipped": phase_b.get("skipped"),
        "phase_b_after": {
            s: phase_b.get("after_train_acceptance", {}).get(s, {}).get("E_accepted_per_step_primary")
            for s in eval_songs
        }
        if not phase_b.get("skipped")
        else None,
        "one_layer_held_out": {
            s: phase_d["one_layer"]["acceptance"][s]["E_accepted_per_step_primary"] for s in held_out
        },
        "recommendation": rec,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "note": (
            "§43 draft quality only. No turbo runtime wiring. Not a 500 TPS claim. "
            "Half-width is cost simulation with 1-layer acceptance proxy."
        ),
    }
    (out_dir / "acceptance_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2), flush=True)
    print(f"WROTE {out_dir}", flush=True)


if __name__ == "__main__":
    main()
