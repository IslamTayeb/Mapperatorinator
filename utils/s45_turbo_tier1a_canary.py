#!/usr/bin/env python3
"""§45 TIER1a greedy canary gate for turbo integrator.

Requires turbo vs optimized bit-exact match for ≥500 generated tokens × N seeds.
Hard-fails on any mismatch. Not a 500 TPS claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "utils") not in sys.path:
    sys.path.insert(0, str(REPO / "utils"))

# Reuse §36 window/engine helpers.
import s36_turbo_speculative_smoke as s36  # noqa: E402


def run_one_seed(
    *,
    args_inf: Any,
    model_kwargs: dict[str, Any],
    meta: dict[str, Any],
    draft_ckpt: Path,
    precision: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    prompt_len = int(meta["prompt_len"])
    max_length = prompt_len + int(max_new_tokens)
    generate_kwargs = {
        "precision": precision,
        "do_sample": False,
        "num_beams": 1,
        "top_p": 0.9,
        "max_length": max_length,
        "cfg_scale": 1.0,
        "timeshift_bias": 0,
        "types_first": False,
        "temperature": 1.0,
        "lookback_time": 0.0,
        "lookahead_time": 0.0,
        "context_type": "map",
        "sync_model_timing": True,
        "pad_token_id": None,
    }

    def _clone_kwargs(src: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in src.items():
            out[key] = value.clone() if isinstance(value, torch.Tensor) else value
        return out

    turbo_run = s36.run_engine_window(
        engine_name="turbo",
        args_inf=args_inf,
        draft_ckpt=draft_ckpt,
        model_kwargs=_clone_kwargs(model_kwargs),
        generate_kwargs=dict(generate_kwargs),
    )
    torch.cuda.empty_cache()
    opt_run = s36.run_engine_window(
        engine_name="optimized",
        args_inf=args_inf,
        draft_ckpt=None,
        model_kwargs=_clone_kwargs(model_kwargs),
        generate_kwargs=dict(generate_kwargs),
    )

    turbo_tok = turbo_run["result_tokens"]
    opt_tok = opt_run["result_tokens"]
    # Gate on the first ≥max_new_tokens generated ids (after prompt).
    turbo_gen = turbo_tok[prompt_len:]
    opt_gen = opt_tok[prompt_len:]
    need = int(max_new_tokens)
    length_ok = len(turbo_gen) >= need and len(opt_gen) >= need
    match_prefix = bool(length_ok) and turbo_gen[:need] == opt_gen[:need]
    match_full = turbo_tok == opt_tok
    first_mismatch = None
    window = None
    if not match_prefix:
        n = min(len(turbo_tok), len(opt_tok))
        first_mismatch = next((i for i in range(n) if turbo_tok[i] != opt_tok[i]), n)
        lo = max(0, int(first_mismatch) - 2)
        hi = min(n, int(first_mismatch) + 3)
        window = {
            "prompt_len": prompt_len,
            "abs_index": first_mismatch,
            "gen_index": int(first_mismatch) - int(prompt_len),
            "turbo_slice": turbo_tok[lo:hi],
            "optimized_slice": opt_tok[lo:hi],
        }

    turbo_gen_count = int(turbo_run["stats"].get("generated_tokens") or len(turbo_gen))
    seed_ok = bool(match_prefix)
    match = match_prefix

    row = {
        "seed": seed,
        "match": match,
        "seed_ok": seed_ok,
        "length_ok": length_ok,
        "turbo_len": len(turbo_tok),
        "optimized_len": len(opt_tok),
        "turbo_generated_tokens": turbo_gen_count,
        "optimized_generated_tokens": int(
            opt_run["stats"].get("generated_tokens") or len(opt_gen)
        ),
        "first_mismatch": first_mismatch,
        "mismatch_window": window,
        "turbo_wall_s": turbo_run["wall_s"],
        "optimized_wall_s": opt_run["wall_s"],
        "turbo_gamma": turbo_run["stats"].get("turbo_gamma"),
        "turbo_accepted_per_verify": turbo_run["stats"].get("turbo_accepted_per_verify"),
        "turbo_speculative": turbo_run["stats"].get("turbo_speculative"),
        "turbo_teacher_aligned_greedy": turbo_run["stats"].get(
            "turbo_teacher_aligned_greedy"
        ),
        "runtime_meta": turbo_run.get("runtime_meta"),
    }
    print(
        f"seed={seed} match={match} seed_ok={seed_ok} "
        f"turbo_gen={turbo_gen_count} opt_gen={row['optimized_generated_tokens']} "
        f"first_mismatch={first_mismatch}",
        flush=True,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--dump-run", type=Path, required=True)
    ap.add_argument("--draft-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument(
        "--seeds",
        type=str,
        default="12345,23456,34567",
        help="comma-separated seeds (default 3)",
    )
    args = ap.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) < 1:
        raise SystemExit("need at least one seed")

    os.chdir(REPO)
    os.environ.setdefault("MAPPERATORINATOR_TURBO_VERIFY_FASTPATH", "1")
    os.environ.setdefault("MAPPERATORINATOR_TURBO_TEACHER_ALIGNED", "1")
    os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)

    args_inf = s36.load_args(
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
            f"seed={seeds[0]}",
            "profile_inference=false",
        ],
    )

    from inference import load_model_with_engine
    from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor

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
    osu_path = s36.find_osu(args.dump_run)
    model_kwargs, meta = s36.build_first_map_window_kwargs(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )
    print(
        f"prompt_len={meta['prompt_len']} max_new_tokens={args.max_new_tokens} "
        f"seeds={seeds} draft={args.draft_ckpt}",
        flush=True,
    )
    del helper_model, processor
    torch.cuda.empty_cache()

    rows = []
    for seed in seeds:
        rows.append(
            run_one_seed(
                args_inf=args_inf,
                model_kwargs=model_kwargs,
                meta=meta,
                draft_ckpt=args.draft_ckpt,
                precision=args.precision,
                max_new_tokens=args.max_new_tokens,
                seed=seed,
            )
        )

    all_ok = all(r["seed_ok"] for r in rows)
    payload = {
        "section": "45",
        "track": "C",
        "gate": "TIER1a_greedy_canary",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} branch --show-current").read().strip(),
        "draft_ckpt": str(args.draft_ckpt),
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "seeds": seeds,
        "meta": meta,
        "seed_results": rows,
        "tier1a_ok": all_ok,
        "campaign_tip": "55949274",
        "auth_fp16_main_tps": 366.11,
        "note": (
            "TIER1a integrator gate only. No 500 claim. Full TIER1 pack "
            "(§44 harness) still required before turbo ship."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)
    print(f"tier1a_ok={all_ok}", flush=True)
    if not all_ok:
        raise SystemExit("TIER1a greedy canary FAIL (≥500 tok × seeds)")


if __name__ == "__main__":
    main()
