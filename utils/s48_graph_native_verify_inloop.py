#!/usr/bin/env python3
"""§48 W-VG in-loop c_verify gate (graph-native K=γ).

Runs a real sampled turbo speculative window and reports CUDA-event
teacher-verify forward time from inside the cycle (not harness-only).

Gate: c_verify ≤ 1.2× tip Q=1 (≤2.22 ms when Q1=1.853 ms).
Kill: ratio > 1.35× after graph-native attempt → STOP.

Not a 500 claim. Campaign tip 55949274 / 366.11.
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

import s36_turbo_speculative_smoke as s36  # noqa: E402
from s41_verify_fastpath_microbench import (  # noqa: E402
    measure_optimized_q1_step,
    _move_kwargs,
)
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.runtime_profiling import generation_profile_context  # noqa: E402

TARGET_RATIO = 1.2
KILL_RATIO = 1.35
# Auth tip Q1 from §41 job 50147970 when local Q1 measure fails.
FALLBACK_Q1_MS = 1.853
GATE_MS = FALLBACK_Q1_MS * TARGET_RATIO  # 2.2236


def _clone_kwargs(src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in src.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument(
        "--draft-ckpt",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt"
        ),
    )
    ap.add_argument(
        "--dump-run",
        type=Path,
        default=Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        ),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--gamma", type=int, default=3)
    args = ap.parse_args()

    os.chdir(REPO)
    os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)
    os.environ["MAPPERATORINATOR_TURBO_GAMMA"] = str(int(args.gamma))
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_FASTPATH"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE"] = "1"
    # Sampled path: aligned Q1 must stay inert so K=γ graph-native is hit.
    os.environ["MAPPERATORINATOR_TURBO_TEACHER_ALIGNED"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_CUDA_TIMER"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_STEP_PROFILE"] = "1"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args_inf = s36.load_args(
        args.config_name,
        [
            f"audio_path={args.audio}",
            "device=cuda",
            "precision=fp16",
            "attn_implementation=sdpa",
            "inference_engine=turbo",
            "use_server=false",
            "parallel=false",
            "cfg_scale=1.0",
            "num_beams=1",
            f"seed={args.seed}",
            "profile_inference=false",
        ],
    )

    preprocessor = Preprocessor(args_inf, parallel=False)
    # Need a processor/tokenizer for window build; load turbo binding once.
    bound, tokenizer = load_model_with_engine(
        ckpt_path=args_inf.model_path,
        t5_args=args_inf.train,
        device=torch.device("cuda"),
        max_batch_size=1,
        use_server=False,
        precision="fp16",
        attn_implementation="sdpa",
        eval_mode=True,
        inference_engine="turbo",
        gamemode=args_inf.gamemode,
        auto_select_gamemode_model=True,
    )
    raw, runtime = unwrap_engine_binding(bound)
    raw.eval()
    processor = Processor(args_inf, bound, tokenizer)
    sequences = preprocessor.segment(preprocessor.load(str(args.audio)))
    song_length = float(sequences[2])
    osu_path = s36.find_osu(args.dump_run)
    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    model_kwargs, meta = s36.build_first_map_window_kwargs(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )

    # Q1 baseline (same stack as §41).
    q1_ms = None
    q1_meta: dict[str, Any] = {}
    try:
        with generation_profile_context(
            q1_bmm_cross_attention=True,
            native_q1_self_attention=True,
            native_q1_rope_cache_self_attention=True,
            native_cross_mlp_tail=True,
            optimized_expected_dtype=torch.float16,
        ):
            # Fresh optimized binding for fair Q1 (avoid turbo draft side effects).
            opt_bound, _tok = load_model_with_engine(
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
            opt_model, _ = unwrap_engine_binding(opt_bound)
            opt_model.eval()
            mk = _move_kwargs(opt_model, _clone_kwargs(model_kwargs))
            prompt_ids = mk.pop("decoder_input_ids")
            q1_meta = measure_optimized_q1_step(
                opt_model,
                prompt_ids=prompt_ids,
                model_kwargs=mk,
                warmup=5,
                iters=30,
            )
            q1_ms = float(q1_meta["q1_step_ms"])
            del opt_bound, opt_model
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        q1_meta = {"error": f"{type(exc).__name__}: {exc}"}
        q1_ms = FALLBACK_Q1_MS

    prompt_len = int(meta["prompt_len"])
    max_length = prompt_len + int(args.max_new_tokens)
    generate_kwargs = {
        "precision": "fp16",
        "do_sample": True,
        "num_beams": 1,
        "top_p": 0.9,
        "max_length": max_length,
        "cfg_scale": 1.0,
        "timeshift_bias": 0,
        "types_first": False,
        "temperature": 0.9,
        "lookback_time": 0.0,
        "lookahead_time": 0.0,
        "context_type": "map",
        "sync_model_timing": True,
        "pad_token_id": getattr(tokenizer, "pad_id", None),
    }

    session = runtime.new_context_state()
    # Ensure persistent verify_fp is armed before the window.
    verify_fp = runtime.ensure_verify_fastpath(raw)
    session.verify_fastpath = verify_fp
    if verify_fp is not None:
        verify_fp.forward_ms_total = 0.0
        verify_fp.forward_timed_calls = 0
        verify_fp.graph_captures = 0
        verify_fp.graph_replays = 0
        verify_fp.graph_native_replays = 0
        verify_fp.prepare_inputs_calls = 0
        verify_fp.eager_calls = 0

    result, stats = runtime.generate_window(
        model=raw,
        tokenizer=tokenizer,
        model_kwargs=_clone_kwargs(model_kwargs),
        generate_kwargs=generate_kwargs,
        context_state=session,
    )

    # Second window with same session/runtime — captures must not rebuild from scratch.
    captures_after_w1 = (
        int(verify_fp.graph_captures) if verify_fp is not None else None
    )
    replays_after_w1 = (
        int(verify_fp.graph_native_replays) if verify_fp is not None else None
    )
    result2, stats2 = runtime.generate_window(
        model=raw,
        tokenizer=tokenizer,
        model_kwargs=_clone_kwargs(model_kwargs),
        generate_kwargs={
            **generate_kwargs,
            "max_length": prompt_len + max(64, args.max_new_tokens // 4),
        },
        context_state=session,
    )
    captures_after_w2 = (
        int(verify_fp.graph_captures) if verify_fp is not None else None
    )

    c_verify_ms = None
    if verify_fp is not None and verify_fp.forward_timed_calls > 0:
        c_verify_ms = float(verify_fp.forward_ms_total / verify_fp.forward_timed_calls)
    phase_verify_ms = stats.get("turbo_verify_ms_per_verify")

    ratio = None
    if c_verify_ms is not None and q1_ms and q1_ms > 0:
        ratio = float(c_verify_ms) / float(q1_ms)

    gate_pass = ratio is not None and ratio <= TARGET_RATIO
    kill = ratio is not None and ratio > KILL_RATIO
    path = stats.get("turbo_verify_path")
    path_ok = path == "graph_native_k"
    prepare_ok = (
        verify_fp is not None and int(verify_fp.prepare_inputs_calls) == 0
    )
    persistent_ok = (
        captures_after_w1 is not None
        and captures_after_w2 is not None
        and captures_after_w2 <= captures_after_w1 + 4  # allow a few new buckets
    )

    decision = "PASS" if gate_pass and path_ok else ("STOP_KILL" if kill else "MISS")
    if not path_ok and decision == "PASS":
        decision = "MISS_PATH"

    out = {
        "section": 48,
        "worker": "W-VG",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(
            f"git -C {REPO} branch --show-current"
        ).read().strip(),
        "campaign_tip": "55949274",
        "campaign_tip_fp16_main_tps": 366.11,
        "gamma": int(args.gamma),
        "max_new_tokens": int(args.max_new_tokens),
        "q1_step_ms": q1_ms,
        "q1_meta": q1_meta,
        "c_verify_ms_inloop": c_verify_ms,
        "c_verify_ratio_vs_q1": ratio,
        "phase_verify_ms_per_verify": phase_verify_ms,
        "gate_ms_1_2x": float(q1_ms) * TARGET_RATIO if q1_ms else GATE_MS,
        "target_ratio": TARGET_RATIO,
        "kill_ratio": KILL_RATIO,
        "gate_pass": gate_pass,
        "kill": kill,
        "decision": decision,
        "turbo_verify_path": path,
        "path_hit_graph_native": path_ok,
        "prepare_inputs_calls": (
            int(verify_fp.prepare_inputs_calls) if verify_fp is not None else None
        ),
        "prepare_inputs_hot_path_clean": prepare_ok,
        "graph_captures_w1": captures_after_w1,
        "graph_captures_w2": captures_after_w2,
        "graph_native_replays_w1": replays_after_w1,
        "persistent_cache_ok": persistent_ok,
        "window1_stats": {
            k: stats.get(k)
            for k in (
                "turbo_verify_steps",
                "turbo_accepted_tokens",
                "turbo_accepted_per_verify",
                "turbo_verify_path",
                "turbo_verify_graph_native",
                "turbo_verify_graph_native_replays",
                "turbo_verify_graph_captures",
                "turbo_verify_forward_ms_avg",
                "turbo_verify_prepare_inputs_calls",
                "turbo_verify_ms_per_verify",
            )
        },
        "window2_stats": {
            k: stats2.get(k)
            for k in (
                "turbo_verify_steps",
                "turbo_verify_path",
                "turbo_verify_graph_captures",
                "turbo_verify_graph_native_replays",
            )
        },
        "tokens_w1": int(result.shape[-1]) if hasattr(result, "shape") else None,
        "tokens_w2": int(result2.shape[-1]) if hasattr(result2, "shape") else None,
        "note": (
            "In-loop CUDA-event c_verify on sampled turbo cycle. "
            "No 500 claim. Tip unchanged 55949274/366.11."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    # Always 0 for slurm harvest; decision lives in JSON.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
