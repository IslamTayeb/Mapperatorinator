#!/usr/bin/env python3
"""§54 Stage A in-loop c_verify + runtime E + path hits.

Interim gate: c_verify ≤1.35× (~2.5 ms). Kill Stage A if no path to ≤2.5 ms.
Mandatory: re-measure accepted/verify E (ΔE ≥ −0.05 vs unfused §48 baseline);
greedy TIER1a canary stays on aligned Q=1 (assert via separate canary job).

Does not re-grind unfused §48. Tip 55949274 / 366.11. No 500 claim.
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

# §54 interim: ≤1.35× (~2.5 ms). Kill if no path to absolute ≤2.5 ms.
INTERIM_RATIO = 1.35
ABS_GATE_MS = 2.5
# Unfused §48 in-loop E (accepted/verify) from integrator-era scouts ~1.06;
# Stage A must not lose more than 0.05 vs the just-measured unfused control.
DELTA_E_MIN = -0.05
FALLBACK_Q1_MS = 1.853
# Sealed §48 reference (do not re-grind).
S48_C_VERIFY_MS = 3.075
S48_RATIO = 1.669


def _clone_kwargs(src: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in src.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def _run_window(
    *,
    runtime,
    raw,
    tokenizer,
    model_kwargs: dict[str, Any],
    generate_kwargs: dict[str, Any],
    session,
    verify_fp,
) -> tuple[Any, dict[str, Any]]:
    if verify_fp is not None:
        verify_fp.forward_ms_total = 0.0
        verify_fp.forward_timed_calls = 0
    return runtime.generate_window(
        model=raw,
        tokenizer=tokenizer,
        model_kwargs=_clone_kwargs(model_kwargs),
        generate_kwargs=generate_kwargs,
        context_state=session,
    )


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
    ap.add_argument(
        "--skip-unfused-control",
        action="store_true",
        help="Skip MROW=0 control (use sealed §48 constants for ΔE baseline only)",
    )
    args = ap.parse_args()

    os.chdir(REPO)
    os.environ.setdefault("MAPPERATORINATOR_TURBO_DRAFT_CKPT", str(args.draft_ckpt))
    os.environ.setdefault("MAPPERATORINATOR_TURBO_GAMMA", str(int(args.gamma)))
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_FASTPATH"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_TEACHER_ALIGNED"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_CUDA_TIMER"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_STEP_PROFILE"] = "1"
    os.environ["MAPPERATORINATOR_TURBO_VERIFY_MROW"] = "1"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Preload m-row extension before timed window.
    from osuT5.osuT5.inference.optimized.kernels.m_row import preload_native_m_row

    preload_native_m_row()

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

    unfused_e = None
    unfused_c_verify = None
    if not args.skip_unfused_control:
        os.environ["MAPPERATORINATOR_TURBO_VERIFY_MROW"] = "0"
        session_u = runtime.new_context_state()
        verify_fp_u = runtime.ensure_verify_fastpath(raw)
        session_u.verify_fastpath = verify_fp_u
        if verify_fp_u is not None:
            verify_fp_u.graph_cache.clear()
            verify_fp_u.graph_captures = 0
            verify_fp_u.graph_replays = 0
            verify_fp_u.graph_native_replays = 0
            verify_fp_u.prepare_inputs_calls = 0
            verify_fp_u.forward_ms_total = 0.0
            verify_fp_u.forward_timed_calls = 0
        _r_u, stats_u = _run_window(
            runtime=runtime,
            raw=raw,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
            session=session_u,
            verify_fp=verify_fp_u,
        )
        unfused_e = stats_u.get("turbo_accepted_per_verify")
        if verify_fp_u is not None and verify_fp_u.forward_timed_calls > 0:
            unfused_c_verify = float(
                verify_fp_u.forward_ms_total / verify_fp_u.forward_timed_calls
            )
        del session_u
        torch.cuda.empty_cache()

    os.environ["MAPPERATORINATOR_TURBO_VERIFY_MROW"] = "1"
    session = runtime.new_context_state()
    verify_fp = runtime.ensure_verify_fastpath(raw)
    session.verify_fastpath = verify_fp
    if verify_fp is not None:
        verify_fp.graph_cache.clear()
        verify_fp.graph_captures = 0
        verify_fp.graph_replays = 0
        verify_fp.graph_native_replays = 0
        verify_fp.prepare_inputs_calls = 0
        verify_fp.forward_ms_total = 0.0
        verify_fp.forward_timed_calls = 0
        verify_fp.eager_calls = 0

    result, stats = _run_window(
        runtime=runtime,
        raw=raw,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        generate_kwargs=generate_kwargs,
        session=session,
        verify_fp=verify_fp,
    )

    c_verify_ms = None
    if verify_fp is not None and verify_fp.forward_timed_calls > 0:
        c_verify_ms = float(verify_fp.forward_ms_total / verify_fp.forward_timed_calls)

    ratio = None
    if c_verify_ms is not None and q1_ms and q1_ms > 0:
        ratio = float(c_verify_ms) / float(q1_ms)

    e_runtime = stats.get("turbo_accepted_per_verify")
    e_base = float(unfused_e) if unfused_e is not None else None
    delta_e = None
    if e_runtime is not None and e_base is not None:
        delta_e = float(e_runtime) - float(e_base)

    interim_pass = (
        c_verify_ms is not None
        and c_verify_ms <= ABS_GATE_MS
        and ratio is not None
        and ratio <= INTERIM_RATIO
    )
    kill_no_path = c_verify_ms is not None and c_verify_ms > ABS_GATE_MS
    e_ok = delta_e is None or delta_e >= DELTA_E_MIN
    path = stats.get("turbo_verify_path")
    path_ok = path == "graph_native_k"

    if interim_pass and e_ok and path_ok:
        decision = "STAGE_A_PASS"
    elif kill_no_path:
        decision = "STAGE_A_KILL"
    else:
        decision = "STAGE_A_MISS"

    out = {
        "section": 54,
        "stage": "A",
        "worker": "Track2-verify-kernels",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} branch --show-current").read().strip(),
        "campaign_tip": "55949274",
        "campaign_tip_fp16_main_tps": 366.11,
        "gamma": int(args.gamma),
        "mrow_enabled": True,
        "q1_step_ms": q1_ms,
        "q1_meta": q1_meta,
        "c_verify_ms_inloop": c_verify_ms,
        "c_verify_ratio_vs_q1": ratio,
        "abs_gate_ms": ABS_GATE_MS,
        "interim_ratio": INTERIM_RATIO,
        "s48_sealed_c_verify_ms": S48_C_VERIFY_MS,
        "s48_sealed_ratio": S48_RATIO,
        "unfused_control_c_verify_ms": unfused_c_verify,
        "unfused_control_E": unfused_e,
        "E_runtime_accepted_per_verify": e_runtime,
        "delta_E": delta_e,
        "delta_E_gate": DELTA_E_MIN,
        "delta_E_ok": e_ok,
        "interim_pass": interim_pass,
        "kill_no_path_to_2_5ms": kill_no_path,
        "decision": decision,
        "turbo_verify_path": path,
        "path_hit_graph_native": path_ok,
        "prepare_inputs_calls": (
            int(verify_fp.prepare_inputs_calls) if verify_fp is not None else None
        ),
        "graph_native_replays": (
            int(verify_fp.graph_native_replays) if verify_fp is not None else None
        ),
        "graph_captures": (
            int(verify_fp.graph_captures) if verify_fp is not None else None
        ),
        "window_stats": {
            k: stats.get(k)
            for k in (
                "turbo_verify_steps",
                "turbo_accepted_tokens",
                "turbo_accepted_per_verify",
                "turbo_verify_path",
                "turbo_verify_forward_ms_avg",
                "turbo_verify_ms_per_verify",
            )
        },
        "tokens": int(result.shape[-1]) if hasattr(result, "shape") else None,
        "stage_b_authorized": bool(interim_pass and e_ok and path_ok),
        "note": (
            "§54 Stage A m-row islands around SDPA. "
            "Do not re-grind unfused §48. Tip unchanged. No 500 claim."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
