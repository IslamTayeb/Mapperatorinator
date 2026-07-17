#!/usr/bin/env python3
"""§40 TIER1a diagnosis: dump teacher logits at first_mismatch for

(a) one-token eager sequential decode
(b) batched K-verify (turbo `_verify_draft_tokens`)

Classify rejection-rule / indexing bug vs batch-variance numerics.
Not a TPS claim. Does not touch optimized/ bit-exact path.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "utils"))

__import__("config")

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.inference.turbo import speculate as spec  # noqa: E402
from s36_turbo_speculative_smoke import (  # noqa: E402
    build_first_map_window_kwargs,
    find_osu,
    load_args,
)


def _git(*git_args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *git_args], text=True
    ).strip()


def _topk(logits_1d: torch.Tensor, k: int = 8) -> list[dict[str, float | int]]:
    vals, idxs = torch.topk(logits_1d.float(), k=min(k, logits_1d.numel()))
    return [
        {"token": int(idxs[i].item()), "logit": float(vals[i].item())}
        for i in range(vals.numel())
    ]


def _compare_logits(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    diff = (a - b).abs()
    argmax_a = int(torch.argmax(a).item())
    argmax_b = int(torch.argmax(b).item())
    return {
        "argmax_a": argmax_a,
        "argmax_b": argmax_b,
        "argmax_match": argmax_a == argmax_b,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "a_at_b_argmax": float(a[argmax_b].item()),
        "b_at_a_argmax": float(b[argmax_a].item()),
        "logit_gap_a": float((a[argmax_a] - a[argmax_b]).item()),
        "logit_gap_b": float((b[argmax_b] - b[argmax_a]).item()),
        "topk_a": _topk(a),
        "topk_b": _topk(b),
    }


def _fresh_teacher_state(teacher, model_kwargs: dict[str, Any], prompt_ids: torch.Tensor):
    cache = spec.get_turbo_cache(cfg_scale=1.0)
    mk = spec._move_model_kwargs(teacher, model_kwargs)
    mk.pop("decoder_input_ids", None)
    mk["past_key_values"] = cache
    mk["use_cache"] = True
    last, mk = spec._prefill(teacher, prompt_ids=prompt_ids, model_kwargs=mk)
    return last, mk, cache


def _fresh_draft_state(draft, model_kwargs: dict[str, Any], prompt_ids: torch.Tensor, encoder_outputs):
    cache = spec.get_turbo_cache(cfg_scale=1.0)
    mk = spec._move_model_kwargs(draft, model_kwargs)
    mk.pop("decoder_input_ids", None)
    mk["past_key_values"] = cache
    mk["use_cache"] = True
    if encoder_outputs is not None:
        mk["encoder_outputs"] = encoder_outputs
    last, mk = spec._prefill(draft, prompt_ids=prompt_ids, model_kwargs=mk)
    return last, mk, cache


def _advance_one(
    model,
    *,
    sequences: torch.Tensor,
    last_logits: torch.Tensor,
    model_kwargs: dict[str, Any],
    processors,
    token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
    processed = spec._apply_processors(
        processors, sequences, last_logits.unsqueeze(0)
    ).squeeze(0)
    tok = int(torch.argmax(processed).item()) if token is None else int(token)
    tok_t = torch.tensor([[tok]], device=sequences.device, dtype=sequences.dtype)
    sequences = torch.cat([sequences, tok_t], dim=-1)
    out, mk = spec._forward_decoder(
        model, decoder_input_ids=sequences, model_kwargs=model_kwargs
    )
    last = out.logits[:, -1, :].float().squeeze(0).clone()
    return sequences, last, mk, tok


@torch.no_grad()
def dump_at_mismatch(
    *,
    teacher,
    draft,
    tokenizer,
    model_kwargs: dict[str, Any],
    gamma: int,
    timeshift_bias: float,
    lookback_time: float,
    mismatch_abs: int,
    known_opt_prefix: list[int],
) -> dict[str, Any]:
    processors = spec.build_structural_processors(
        tokenizer,
        timeshift_bias=float(timeshift_bias),
        types_first=False,
        lookback_time=float(lookback_time),
        device=teacher.device,
    )

    base_mk = spec._move_model_kwargs(teacher, model_kwargs)
    prompt_ids = base_mk.pop("decoder_input_ids")
    prompt_len = int(prompt_ids.shape[1])
    if mismatch_abs <= prompt_len:
        raise SystemExit(f"mismatch_abs={mismatch_abs} must be > prompt_len={prompt_len}")

    gen_index = mismatch_abs - prompt_len
    verify_start = prompt_len + (gen_index // gamma) * gamma
    idx_in_window = mismatch_abs - verify_start

    # --- Build optimized-matching prefix tokens up through mismatch-1 ---
    teacher_last, teacher_mk, _ = _fresh_teacher_state(teacher, model_kwargs, prompt_ids)
    enc = teacher_mk.get("encoder_outputs")
    draft_last, draft_mk, _ = _fresh_draft_state(draft, model_kwargs, prompt_ids, enc)
    sequences = prompt_ids.clone()
    prefix_tokens: list[int] = []
    while int(sequences.shape[1]) < mismatch_abs:
        gi = int(sequences.shape[1]) - prompt_len
        force = known_opt_prefix[gi] if gi < len(known_opt_prefix) else None
        sequences, teacher_last, teacher_mk, tok = _advance_one(
            teacher,
            sequences=sequences,
            last_logits=teacher_last,
            model_kwargs=teacher_mk,
            processors=processors,
            token=force,
        )
        prefix_tokens.append(tok)
        sequences_d = sequences  # same ids
        draft_out, draft_mk = spec._forward_decoder(
            draft, decoder_input_ids=sequences_d, model_kwargs=draft_mk
        )
        draft_last = draft_out.logits[:, -1, :].float().squeeze(0).clone()

    eager_processed = spec._apply_processors(
        processors, sequences, teacher_last.unsqueeze(0)
    ).squeeze(0)
    eager_raw = teacher_last.clone()

    # --- Batched K-verify from verify_start with forced opt-prefix draft ---
    teacher_last_v, teacher_mk_v, _ = _fresh_teacher_state(
        teacher, model_kwargs, prompt_ids
    )
    sequences_v = prompt_ids.clone()
    while int(sequences_v.shape[1]) < verify_start:
        gi = int(sequences_v.shape[1]) - prompt_len
        sequences_v, teacher_last_v, teacher_mk_v, _ = _advance_one(
            teacher,
            sequences=sequences_v,
            last_logits=teacher_last_v,
            model_kwargs=teacher_mk_v,
            processors=processors,
            token=known_opt_prefix[gi],
        )

    # Forced draft = known opt tokens for the window, then greedy pad.
    force_ids: list[int] = []
    for i in range(gamma):
        gi = (verify_start - prompt_len) + i
        if gi < len(known_opt_prefix):
            force_ids.append(int(known_opt_prefix[gi]))
        elif gi < len(prefix_tokens):
            force_ids.append(int(prefix_tokens[gi]))
        else:
            force_ids.append(int(known_opt_prefix[-1]))

    p_logits_f, _, _, _ = spec._verify_draft_tokens(
        teacher,
        prompt_ids=sequences_v,
        draft_ids=force_ids,
        last_logits=teacher_last_v,
        model_kwargs=teacher_mk_v,
        processors=processors,
        greedy=True,
        temperature=0.0,
        top_p=0.9,
    )
    batched_at_mismatch = p_logits_f[idx_in_window]
    cmp = _compare_logits(eager_processed, batched_at_mismatch)

    # Per-position eager vs batched across the forced window.
    teacher_last_e, teacher_mk_e, _ = _fresh_teacher_state(
        teacher, model_kwargs, prompt_ids
    )
    sequences_e = prompt_ids.clone()
    while int(sequences_e.shape[1]) < verify_start:
        gi = int(sequences_e.shape[1]) - prompt_len
        sequences_e, teacher_last_e, teacher_mk_e, _ = _advance_one(
            teacher,
            sequences=sequences_e,
            last_logits=teacher_last_e,
            model_kwargs=teacher_mk_e,
            processors=processors,
            token=known_opt_prefix[gi],
        )

    per_pos = []
    for i in range(gamma):
        eager_i = spec._apply_processors(
            processors, sequences_e, teacher_last_e.unsqueeze(0)
        ).squeeze(0)
        c = _compare_logits(eager_i, p_logits_f[i])
        per_pos.append(
            {
                "window_i": i,
                "abs_index": verify_start + i,
                "force_draft_token": int(force_ids[i]),
                "eager_argmax": c["argmax_a"],
                "batched_argmax": c["argmax_b"],
                "argmax_match": c["argmax_match"],
                "max_abs_diff": c["max_abs_diff"],
                "mean_abs_diff": c["mean_abs_diff"],
                "logit_gap_eager": c["logit_gap_a"],
                "logit_gap_batched": c["logit_gap_b"],
                "topk_eager": c["topk_a"][:5],
                "topk_batched": c["topk_b"][:5],
            }
        )
        sequences_e, teacher_last_e, teacher_mk_e, _ = _advance_one(
            teacher,
            sequences=sequences_e,
            last_logits=teacher_last_e,
            model_kwargs=teacher_mk_e,
            processors=processors,
            token=force_ids[i],
        )

    # Live draft path (actual turbo draft proposals) from verify_start.
    teacher_last_l, teacher_mk_l, _ = _fresh_teacher_state(
        teacher, model_kwargs, prompt_ids
    )
    enc_l = teacher_mk_l.get("encoder_outputs")
    draft_last_l, draft_mk_l, _ = _fresh_draft_state(
        draft, model_kwargs, prompt_ids, enc_l
    )
    sequences_l = prompt_ids.clone()
    while int(sequences_l.shape[1]) < verify_start:
        gi = int(sequences_l.shape[1]) - prompt_len
        tok = int(known_opt_prefix[gi])
        sequences_l, teacher_last_l, teacher_mk_l, _ = _advance_one(
            teacher,
            sequences=sequences_l,
            last_logits=teacher_last_l,
            model_kwargs=teacher_mk_l,
            processors=processors,
            token=tok,
        )
        draft_out, draft_mk_l = spec._forward_decoder(
            draft, decoder_input_ids=sequences_l, model_kwargs=draft_mk_l
        )
        draft_last_l = draft_out.logits[:, -1, :].float().squeeze(0).clone()

    draft_ids, _, _, _, _ = spec._draft_k_tokens(
        draft,
        prompt_ids=sequences_l,
        last_logits=draft_last_l,
        model_kwargs=draft_mk_l,
        gamma=gamma,
        greedy=True,
        temperature=0.0,
        top_p=0.9,
        processors=processors,
        rng=None,
        max_new_tokens=64,
        eos_ids=set(),
    )
    p_logits_live, _, _, _ = spec._verify_draft_tokens(
        teacher,
        prompt_ids=sequences_l,
        draft_ids=draft_ids,
        last_logits=teacher_last_l,
        model_kwargs=teacher_mk_l,
        processors=processors,
        greedy=True,
        temperature=0.0,
        top_p=0.9,
    )
    live_teacher = [int(torch.argmax(p_logits_live[i]).item()) for i in range(len(draft_ids))]
    n_acc = 0
    residual = None
    for i, tok in enumerate(draft_ids):
        if int(tok) == live_teacher[i]:
            n_acc += 1
        else:
            residual = live_teacher[i]
            break

    mismatch_rows = [p for p in per_pos if not p["argmax_match"]]
    # Row 0 uses cached last_logits (not batched matmul over draft); if it
    # already disagrees hard, state/indexing is broken. Later-only flips with
    # small gaps are classic SDPA/FP16 batch variance.
    row0_bad = bool(per_pos) and (not per_pos[0]["argmax_match"]) and (
        per_pos[0]["max_abs_diff"] > 1.0
    )
    huge = [p for p in mismatch_rows if p["max_abs_diff"] > 5.0]
    if row0_bad or (huge and len(mismatch_rows) >= 2):
        classification = "rejection_rule_or_indexing_bug"
    elif mismatch_rows:
        classification = "batch_variance_numerics"
    else:
        classification = "no_logit_divergence_at_window_investigate_elsewhere"

    return {
        "prompt_len": prompt_len,
        "mismatch_abs": mismatch_abs,
        "gen_index": gen_index,
        "verify_start": verify_start,
        "idx_in_window": idx_in_window,
        "gamma": gamma,
        "prefix_tokens_forced": prefix_tokens,
        "eager_at_mismatch": {
            "argmax": int(torch.argmax(eager_processed).item()),
            "topk": _topk(eager_processed),
            "raw_topk": _topk(eager_raw),
        },
        "batched_forced_at_mismatch": {
            "argmax": int(torch.argmax(batched_at_mismatch).item()),
            "topk": _topk(batched_at_mismatch),
            "force_draft_ids": force_ids,
        },
        "compare_eager_vs_batched_forced": cmp,
        "per_position_window": per_pos,
        "live_draft_ids": draft_ids,
        "live_draft_teacher_argmax": live_teacher,
        "live_n_accepted": n_acc,
        "live_residual": residual,
        "classification": classification,
        "smoke_expected": {"turbo_token": 2236, "optimized_token": 2213},
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
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--mismatch-abs", type=int, default=110)
    ap.add_argument("--known-opt-prefix", default="12,1648,2213,2749,3878")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    os.chdir(REPO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)

    known = [int(x) for x in args.known_opt_prefix.split(",") if x.strip()]
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
    del helper_model, processor
    torch.cuda.empty_cache()

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
    if runtime is None or not hasattr(runtime, "draft"):
        raise SystemExit("turbo runtime/draft missing")
    draft = runtime.draft
    teacher.eval()
    draft.eval()

    payload = dump_at_mismatch(
        teacher=teacher,
        draft=draft,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        gamma=args.gamma,
        timeshift_bias=0.0,
        lookback_time=0.0,
        mismatch_abs=args.mismatch_abs,
        known_opt_prefix=known,
    )
    payload["meta"] = meta
    payload["commit"] = _git("rev-parse", "HEAD")
    payload["branch"] = _git("branch", "--show-current")
    payload["draft_ckpt"] = str(args.draft_ckpt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"classification={payload['classification']}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
