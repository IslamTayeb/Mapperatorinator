#!/usr/bin/env python3
"""§41 TIER1a probe: graph-aligned teacher vs optimized at mismatch@110.

Forces optimized prefix [12,1648], compares argmax at gen_index=2 for:
- turbo graph-aligned sequential Q=1 teacher (§41)
- optimized engine greedy continuation

Gate: turbo argmax == optimized argmax (target 2213 from §40 dump).
Not a full 500-token × 3-seed canary; budget probe only.
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
from osuT5.osuT5.inference.turbo.verify_fastpath import (  # noqa: E402
    allocate_teacher_static_cache,
    build_teacher_verify_fastpath,
    teacher_aligned_runtime_context,
)
from s36_turbo_speculative_smoke import (  # noqa: E402
    build_first_map_window_kwargs,
    find_osu,
    load_args,
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def _topk(logits: torch.Tensor, k: int = 8) -> list[dict[str, float | int]]:
    vals, idxs = torch.topk(logits.float().reshape(-1), k=min(k, logits.numel()))
    return [
        {"token": int(idxs[i].item()), "logit": float(vals[i].item())}
        for i in range(vals.numel())
    ]


@torch.no_grad()
def aligned_teacher_argmax_at(
    teacher,
    *,
    model_kwargs: dict[str, Any],
    prompt_ids: torch.Tensor,
    forced_prefix: list[int],
    precision: str,
) -> dict[str, Any]:
    """Prefill + force prefix via sequential Q=1; return next-step argmax."""
    os.environ.setdefault("MAPPERATORINATOR_TURBO_VERIFY_FASTPATH", "1")
    os.environ.setdefault("MAPPERATORINATOR_TURBO_TEACHER_ALIGNED", "1")
    os.environ.setdefault("MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH", "1")
    fp = build_teacher_verify_fastpath(teacher)
    if fp is None:
        raise RuntimeError("verify fastpath disabled")
    cache = allocate_teacher_static_cache(teacher, cfg_scale=1.0)
    mk = spec._move_model_kwargs(teacher, model_kwargs)
    mk.pop("decoder_input_ids", None)
    mk["past_key_values"] = cache
    mk["use_cache"] = True
    with teacher_aligned_runtime_context(precision=precision):
        last, mk = spec._prefill(teacher, prompt_ids=prompt_ids, model_kwargs=mk)
        # Prefer stashed encoder_outputs; fall back to keeping frames (eager only).
        if mk.get("encoder_outputs") is None:
            raise RuntimeError("aligned canary prefill missing encoder_outputs")
        for key in ("frames", "inputs", "input_features", "input_values", "inputs_embeds"):
            mk.pop(key, None)
        ids = prompt_ids
        for tok in forced_prefix:
            # Logits before consuming tok are for verifying tok; after Q=1 we
            # advance. For mismatch@110 we need logits AFTER [12,1648].
            ids = torch.cat(
                [
                    ids,
                    torch.tensor([[int(tok)]], device=ids.device, dtype=ids.dtype),
                ],
                dim=-1,
            )
            out, mk = fp.forward_q1(decoder_input_ids=ids, model_kwargs=mk)
            last = out.logits[:, -1, :].float().squeeze(0)
    argmax = int(torch.argmax(last).item())
    return {
        "argmax": argmax,
        "topk": _topk(last),
        "q1_steps": fp.q1_steps,
        "graph_captures": fp.graph_captures,
        "graph_replays": fp.graph_replays,
        "meta": fp.profile_metadata(),
    }


@torch.no_grad()
def optimized_argmax_at(
    *,
    args_inf,
    model_kwargs: dict[str, Any],
    prompt_ids: torch.Tensor,
    forced_prefix: list[int],
    max_new: int,
) -> dict[str, Any]:
    bound, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision="fp16",
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="optimized",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    raw, runtime = unwrap_engine_binding(bound)
    raw.eval()
    session = runtime.new_context_state()
    # Force prefix by concatenating into decoder_input_ids then generate 1 token.
    forced = torch.tensor([forced_prefix], device=prompt_ids.device, dtype=prompt_ids.dtype)
    seed_ids = torch.cat([prompt_ids, forced], dim=-1)
    mk = {
        k: (v.to(raw.device) if isinstance(v, torch.Tensor) else v)
        for k, v in model_kwargs.items()
    }
    # Optimized generate_window expects `inputs` (not frames).
    if "frames" in mk and "inputs" not in mk:
        mk["inputs"] = mk["frames"]
    mk["decoder_input_ids"] = seed_ids
    mk["decoder_attention_mask"] = torch.ones_like(seed_ids)
    gw = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "temperature": 0.0,
        "precision": "fp16",
        "cfg_scale": 1.0,
        "num_beams": 1,
        "context_type": "map",
        "sync_model_timing": True,
    }
    result, stats = runtime.generate_window(
        model=raw,
        tokenizer=tokenizer,
        model_kwargs=mk,
        generate_kwargs=gw,
        context_state=session,
    )
    gen = result[0, seed_ids.shape[1] : seed_ids.shape[1] + max_new].tolist()
    return {
        "generated": [int(x) for x in gen],
        "first_argmax": int(gen[0]) if gen else None,
        "stats_backend": stats.get("decoder_loop_backend"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audio", type=Path, default=Path("/work/imt11/Mapperatorinator/data/salvalai.mp3"))
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        ),
    )
    ap.add_argument("--config-name", default="profile_salvalai")
    args = ap.parse_args()

    os.chdir(REPO)
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_FASTPATH"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_TEACHER_ALIGNED"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH"] = "1"

    args_inf = load_args(
        args.config_name,
        [
            f"audio_path={args.audio}",
            "device=cuda",
            "precision=fp16",
            "attn_implementation=sdpa",
            "inference_engine=v32",
            "use_server=false",
            "parallel=false",
            "cfg_scale=1.0",
            "num_beams=1",
            "seed=12345",
            "profile_inference=false",
        ],
    )
    helper, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision="fp16",
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="v32",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    helper = helper.half().eval()
    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, helper, tokenizer)
    sequences = preprocessor.segment(preprocessor.load(str(args.audio)))
    song_length = float(sequences[2])
    osu_path = find_osu(args.dump_run)
    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    model_kwargs, meta = build_first_map_window_kwargs(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )
    prompt_ids = model_kwargs["decoder_input_ids"].to("cuda")
    forced = [12, 1648]  # §40 mismatch prefix

    # Teacher for aligned path (same weights as optimized tip).
    teacher_bound, _tok = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision="fp16",
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="optimized",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    teacher, _ = unwrap_engine_binding(teacher_bound)
    teacher.eval()

    aligned = aligned_teacher_argmax_at(
        teacher,
        model_kwargs=model_kwargs,
        prompt_ids=prompt_ids,
        forced_prefix=forced,
        precision="fp16",
    )
    # Always persist aligned result even if optimized compare crashes.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = {
        "section": 41,
        "workstream": "W2",
        "commit": _git("rev-parse", "HEAD"),
        "prompt_len": meta["prompt_len"],
        "forced_prefix": forced,
        "s40_reference": {"turbo_eager": 2236, "optimized": 2213},
        "graph_aligned_teacher": aligned,
    }
    try:
        opt = optimized_argmax_at(
            args_inf=args_inf,
            model_kwargs=model_kwargs,
            prompt_ids=prompt_ids,
            forced_prefix=forced,
            max_new=1,
        )
        match = aligned["argmax"] == opt["first_argmax"]
        # Also compare to sealed §40 optimized reference token.
        match_ref = aligned["argmax"] == 2213
        out = {
            **partial,
            "optimized_continuation": opt,
            "argmax_match": match,
            "argmax_match_s40_ref_2213": match_ref,
            "canary_path_status": "PASS" if match else "FAIL",
            "note": (
                "Budget probe at mismatch@110. Full ≥500×3 canary after PASS. "
                "No 500 claim. Tip 55949274/366.11."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        match_ref = aligned["argmax"] == 2213
        out = {
            **partial,
            "optimized_continuation_error": f"{type(exc).__name__}: {exc}",
            "argmax_match": None,
            "argmax_match_s40_ref_2213": match_ref,
            "canary_path_status": "PARTIAL_ALIGNED_ONLY",
            "note": (
                "Aligned teacher ran; optimized compare failed. "
                "Treat argmax_match_s40_ref_2213 as interim signal."
            ),
        }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if out.get("canary_path_status") == "PASS":
        return 0
    if out.get("argmax_match_s40_ref_2213"):
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
