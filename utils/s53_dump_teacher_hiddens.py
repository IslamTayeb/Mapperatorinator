#!/usr/bin/env python3
"""§53 — dump teacher decoder hiddens + logits for EAGLE head probes.

Replays tip-authored map token dumps under teacher-force with
``output_hidden_states=True``. Uses tip SALVALAI auth dumps by default and
optionally one held-out song (nube-negra).

Not a TPS / TIER1 claim. Tip stays 55949274 / 366.11.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

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

SONG_CATALOG = {
    "salvalai": {
        "audio": DATA_ROOT / "salvalai.mp3",
        "alt_audio": Path("/work/imt11/Mapperatorinator/data/salvalai.mp3"),
        "profile_dir": FIVE_SONG_ROOT / "salvalai",
        "alt_dump_run": TIP_SALVALAI_DUMP,
        "role": "train_ref",
    },
    "nube-negra": {
        "audio": DATA_ROOT / "nube-negra.mp3",
        "profile_dir": FIVE_SONG_ROOT / "nube-negra",
        "role": "held_out",
    },
    "ela-ke-leitada": {
        "audio": DATA_ROOT / "ela-ke-leitada.mp3",
        "profile_dir": FIVE_SONG_ROOT / "ela-ke-leitada",
        "role": "held_out",
    },
}


@torch.no_grad()
def teacher_force_logits_and_hidden(
    model,
    *,
    frames: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    song_position: torch.Tensor | None,
    precision: str,
    cond_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float16-cpu logits [T,V] and last decoder hidden [T,D]."""
    kw: dict[str, Any] = {}
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
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(precision == "fp16")
    ):
        out = model.forward(
            frames=frames,
            decoder_input_ids=dec,
            decoder_attention_mask=attn,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **kw,
        )
    if was_training:
        model.train()
    logits = out.logits[0].float().cpu().half()
    hs = out.decoder_hidden_states
    if not hs:
        raise RuntimeError("decoder_hidden_states empty; output_hidden_states failed")
    hidden = hs[-1][0].float().cpu().half()
    if logits.shape[0] != hidden.shape[0]:
        raise RuntimeError(
            f"logits/hidden length mismatch {logits.shape[0]} vs {hidden.shape[0]}"
        )
    return logits, hidden


def resolve_song(song_id: str) -> dict[str, Path]:
    meta = SONG_CATALOG[song_id]
    audio = Path(meta["audio"])
    if not audio.exists() and meta.get("alt_audio") is not None:
        audio = Path(meta["alt_audio"])
    dump_run = meta.get("alt_dump_run")
    if dump_run is not None and Path(dump_run).exists():
        profile = s37.find_profile(Path(dump_run))
        osu = s37.find_osu(Path(dump_run))
        return {
            "audio": audio,
            "profile": profile,
            "osu": osu,
            "dump_run": Path(dump_run),
        }
    profile_dir = Path(meta["profile_dir"])
    profiles = sorted(profile_dir.glob("*.profile.json"))
    if not profiles:
        raise SystemExit(f"no profile for {song_id} under {profile_dir}")
    profile = profiles[0]
    osus = (
        sorted((profile_dir / "output").glob("beatmap*.osu"))
        if (profile_dir / "output").exists()
        else []
    )
    if not osus:
        osus = sorted(profile_dir.glob("**/beatmap*.osu"))
    if not osus:
        raise SystemExit(f"no .osu for {song_id}")
    return {
        "audio": audio,
        "profile": profile,
        "osu": osus[0],
        "dump_run": profile_dir,
    }


def dump_song(
    *,
    song_id: str,
    teacher,
    tokenizer,
    args_inf,
    precision: str,
    max_map_windows: int,
) -> dict[str, Any]:
    paths = resolve_song(song_id)
    dumps = s37.load_dump_tokens(paths["profile"])
    map_keys = sorted(k for k in dumps if k[0] == "map")
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
    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = s37.prepare_contexts(
        processor, args_inf, generation_config, song_length, paths["osu"]
    )
    max_w = min(max_map_windows, len(map_keys)) if max_map_windows > 0 else len(map_keys)
    windows_in = list(
        s37.iter_map_windows(
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
    out_windows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for w in windows_in:
        logits, hidden = teacher_force_logits_and_hidden(
            teacher,
            frames=w["frames"],
            decoder_input_ids=w["dec_in"],
            decoder_attention_mask=w["dec_attn"],
            song_position=w["song_pos"],
            precision=precision,
            cond_kwargs=cond_kwargs,
        )
        # Keep supervised map region (after prompt) to shrink packs.
        prompt_len = int(w["prompt_len"])
        start = max(prompt_len - 1, 0)
        logits = logits[start:]
        hidden = hidden[start:]
        if logits.shape[0] < 2:
            continue
        out_windows.append(
            {
                "sequence_index": int(w["sequence_index"]),
                "prompt_len": prompt_len,
                "n_tokens": int(logits.shape[0]),
                "logits_teacher": logits,
                "hidden_teacher": hidden,
            }
        )
    elapsed = time.perf_counter() - t0
    return {
        "schema": "s53-teacher-hidden-pack-v1",
        "song_id": song_id,
        "role": SONG_CATALOG[song_id]["role"],
        "audio": str(paths["audio"]),
        "profile": str(paths["profile"]),
        "osu": str(paths["osu"]),
        "dump_run": str(paths["dump_run"]),
        "tip_token_source": "exact-rope-device-state-fp16-auth-49964133"
        if song_id == "salvalai"
        else "five-song-profile",
        "precision": precision,
        "timing_points": n_timing,
        "n_windows": len(out_windows),
        "n_tokens": sum(int(w["n_tokens"]) for w in out_windows),
        "dump_seconds": elapsed,
        "windows": out_windows,
        "note": (
            "Teacher-force last decoder hidden + logits on tip-authored map dumps. "
            "For EAGLE linear_fit / smoke_head / in_loop probes only."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--songs",
        default="salvalai,nube-negra",
        help="comma-separated song ids from catalog",
    )
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max-map-windows", type=int, default=48)
    ap.add_argument("--heldout-max-windows", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    song_ids = [s.strip() for s in args.songs.split(",") if s.strip()]
    for sid in song_ids:
        if sid not in SONG_CATALOG:
            raise SystemExit(f"unknown song {sid}; catalog={sorted(SONG_CATALOG)}")

    # First song audio for hydra compose; overridden per song later.
    first_audio = resolve_song(song_ids[0])["audio"]
    args_inf = s37.load_args(
        args.config_name,
        [
            f"audio_path={first_audio}",
            "device=cuda",
            f"precision={args.precision}",
            "attn_implementation=sdpa",
            # Tip weights + TF dump path used by §37/§43 (same teacher as optimized tip).
            "inference_engine=v32",
            "use_server=false",
            "parallel=false",
            "cfg_scale=1.0",
            "num_beams=1",
            f"seed={args.seed}",
            "profile_inference=false",
            "temperature=0.9",
            "top_p=0.9",
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.precision == "fp16" and device.type == "cuda":
        teacher = teacher.half()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"teacher_loaded_s={time.perf_counter() - t0:.1f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "schema": "s53-teacher-hidden-index-v1",
        "tip_commit": "55949274",
        "auth_fp16_main_tps": 366.11,
        "songs": {},
    }
    combined_windows: list[dict[str, Any]] = []
    for sid in song_ids:
        max_w = (
            args.max_map_windows
            if SONG_CATALOG[sid]["role"] == "train_ref"
            else args.heldout_max_windows
        )
        print(f"dumping song={sid} max_windows={max_w}", flush=True)
        pack = dump_song(
            song_id=sid,
            teacher=teacher,
            tokenizer=tokenizer,
            args_inf=args_inf,
            precision=args.precision,
            max_map_windows=max_w,
        )
        out_pt = args.out_dir / f"hidden_pack_{sid}.pt"
        torch.save(pack, out_pt)
        meta = {
            "path": str(out_pt),
            "role": pack["role"],
            "n_windows": pack["n_windows"],
            "n_tokens": pack["n_tokens"],
            "dump_seconds": pack["dump_seconds"],
            "profile": pack["profile"],
        }
        index["songs"][sid] = meta
        print(json.dumps(meta, indent=2), flush=True)
        for w in pack["windows"]:
            combined_windows.append(
                {
                    "song_id": sid,
                    "role": pack["role"],
                    "sequence_index": w["sequence_index"],
                    "n_tokens": w["n_tokens"],
                    "logits_teacher": w["logits_teacher"],
                    "hidden_teacher": w["hidden_teacher"],
                }
            )

    combined = {
        "schema": "s53-teacher-hidden-pack-v1",
        "song_id": "combined",
        "role": "train+heldout",
        "n_windows": len(combined_windows),
        "n_tokens": sum(int(w["n_tokens"]) for w in combined_windows),
        "windows": combined_windows,
        "songs": list(index["songs"].keys()),
        "tip_commit": "55949274",
    }
    combined_path = args.out_dir / "hidden_pack_combined.pt"
    torch.save(combined, combined_path)
    index["combined_pack"] = str(combined_path)
    index["n_windows_combined"] = combined["n_windows"]
    index["n_tokens_combined"] = combined["n_tokens"]
    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {combined_path}", flush=True)
    print(f"wrote {index_path}", flush=True)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
