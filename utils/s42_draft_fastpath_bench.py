#!/usr/bin/env python3
"""§42 draft fastpath microbench: c_draft ms/token on optimized kernels + CUDA graph.

Measures 2-layer draft graph-replay wall vs a same-node 4-layer teacher main step.
Target: c_draft ≤ 0.15× main ≈ ≤0.4 ms/token on 2080 Ti.

Turbo-only. Not a production TPS / 500 claim.
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
import hydra
from omegaconf import DictConfig, OmegaConf

REPO = Path(os.environ.get("PROBE_REPO", Path(__file__).resolve().parents[1])).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__import__("config")

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.inference.turbo.draft_fastpath import (  # noqa: E402
    DraftFastpathRunner,
    time_graph_replay_ms,
)
TIP_FP16_MAIN_MS = 1000.0 / 366.11  # ~2.731 ms/token auth tip
TARGET_RATIO = 0.15
TARGET_MS = 0.4


def load_args(config_name: str, overrides: list[str]):
    config_dir = REPO / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            config_name=config_name.removeprefix("inference/"), overrides=overrides
        )
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def find_osu(run_root: Path) -> Path:
    cands = [p for p in sorted(run_root.glob("**/beatmap*.osu")) if p.suffix == ".osu"]
    if not cands:
        # Fall back to tip auth artifact if present.
        tip = Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        )
        cands = [p for p in sorted(tip.glob("**/beatmap*.osu")) if p.suffix == ".osu"]
    if not cands:
        raise SystemExit(f"no .osu under {run_root}")
    for p in cands:
        if "candidate_first" in str(p):
            return p
    return cands[0]


def build_first_map_window_kwargs(
    processor, args_inf, generation_config, song_length, osu_path, sequences
):
    sys.path.insert(0, str(REPO / "utils"))
    import s37_tiny_draft_train_smoke as s37  # type: ignore

    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = s37.prepare_contexts(
        processor, args_inf, generation_config, song_length, osu_path
    )
    sequence_index = 0
    frames, frame_time = sequences[0][sequence_index], sequences[1][sequence_index]
    frames_p = processor.prepare_frames(frames)
    frame_time_v = frame_time.item()
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
        processor.prepare_context_sequences(
            out_ctx[: map_i + 1], frame_time_v, True, req_special
        ),
    )
    [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
    prompt_ids = prompt[
        0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())
    ].cpu()
    if frames_p.ndim == 1:
        frames_p = frames_p.unsqueeze(0)
    model_kwargs: dict[str, Any] = {
        "inputs": frames_p,
        "decoder_input_ids": prompt_ids.unsqueeze(0),
        "decoder_attention_mask": torch.ones_like(prompt_ids.unsqueeze(0)),
        **cond_kwargs,
    }
    if processor.do_song_position_embed:
        model_kwargs["song_position"] = torch.tensor(
            [
                [
                    frame_time_v / song_length,
                    (frame_time_v + processor.miliseconds_per_sequence) / song_length,
                ]
            ],
            dtype=torch.float32,
        )
    meta = {
        "prompt_len": int(prompt_ids.numel()),
        "timing_points": n_timing,
        "sequence_index": sequence_index,
        "context_type": "map",
    }
    return model_kwargs, meta


def _ensure_encoder_outputs(model, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    mk = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_kwargs.items()
    }
    if "inputs" in mk and "frames" not in mk:
        mk["frames"] = mk.pop("inputs")
    if mk.get("encoder_outputs") is not None:
        return mk
    prompt = mk.get("decoder_input_ids")
    if prompt is None:
        raise RuntimeError("model_kwargs missing decoder_input_ids")
    # One forward to populate encoder_outputs via prepare_inputs path.
    from osuT5.osuT5.inference.turbo.speculate import _prefill, _move_model_kwargs

    mk2 = _move_model_kwargs(model, model_kwargs)
    prompt_ids = mk2.pop("decoder_input_ids")
    _logits, mk2 = _prefill(model, prompt_ids=prompt_ids, model_kwargs=mk2)
    return mk2


def bench_model_graph(
    *,
    model,
    model_kwargs: dict[str, Any],
    prompt_ids: torch.LongTensor,
    dtype: torch.dtype,
    prefixes: list[int],
    warmup: int,
    iters: int,
    label: str,
) -> dict[str, Any]:
    runner = DraftFastpathRunner(
        model,
        dtype=dtype,
        enable_native_kernels=True,
        cuda_graph=True,
    )
    bind_kwargs = dict(model_kwargs)
    bind_kwargs.pop("decoder_input_ids", None)
    runner.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
    runner.prefill()
    rows = []
    for prefix in prefixes:
        # Fresh runner per prefix so timing starts from a clean graph capture.
        r = DraftFastpathRunner(
            model,
            dtype=dtype,
            enable_native_kernels=True,
            cuda_graph=True,
        )
        r.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
        r.prefill()
        row = time_graph_replay_ms(
            r, prefix_length=prefix, warmup=warmup, iters=iters
        )
        row["label"] = label
        row["layers"] = r.layer_count
        rows.append(row)
        print(
            f"[{label}] prefix={prefix} bucket={row['bucket_prefix_length']} "
            f"ms/token={row['ms_per_token']:.4f} layers={r.layer_count}",
            flush=True,
        )
    ms_values = [float(r["ms_per_token"]) for r in rows]
    return {
        "label": label,
        "layers": int(rows[0]["layers"]) if rows else None,
        "rows": rows,
        "median_ms_per_token": float(sorted(ms_values)[len(ms_values) // 2]),
        "min_ms_per_token": float(min(ms_values)),
        "max_ms_per_token": float(max(ms_values)),
        "dispatch_counts_last": rows[-1]["dispatch_counts"] if rows else {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--audio", default=os.environ.get("MAPPERATORINATOR_AUDIO", ""))
    ap.add_argument(
        "--draft-ckpt",
        default=os.environ.get(
            "MAPPERATORINATOR_TURBO_DRAFT_CKPT",
            "/work/imt11/Mapperatorinator/runs/s37-tiny-draft-train-50146289/draft_train.pt",
        ),
    )
    ap.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--prefixes", default="128,256,512")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--osu-hint",
        type=Path,
        default=None,
        help="Optional existing .osu for context prep (else tip auth artifact)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for §42 draft fastpath bench")

    overrides = [
        f"precision={args.precision}",
        "device=cuda",
        "attn_implementation=sdpa",
        "inference_engine=turbo",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
    ]
    if args.audio:
        overrides.append(f"audio_path={args.audio}")

    os.environ["MAPPERATORINATOR_TURBO_DRAFT_CKPT"] = str(args.draft_ckpt)
    os.environ.setdefault("MAPPERATORINATOR_TURBO_DRAFT_FASTPATH", "1")

    args_inf = load_args(args.config_name, overrides)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32

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

    audio_path = Path(args.audio) if args.audio else Path(args_inf.audio_path)
    preprocessor = Preprocessor(args_inf, parallel=False)
    processor = Processor(args_inf, teacher, tokenizer)
    audio = preprocessor.load(str(audio_path))
    sequences = preprocessor.segment(audio)
    song_length = sequences[2]
    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )
    # Prefer an existing tip .osu for context; generation of a full map is out of scope.
    osu_path = args.osu_hint
    if osu_path is None:
        tip_root = Path(
            "/work/imt11/Mapperatorinator/runs/exact-rope-device-state-fp16-auth-49964133"
        )
        osu_path = find_osu(tip_root)
    model_kwargs, meta = build_first_map_window_kwargs(
        processor,
        args_inf,
        generation_config,
        song_length,
        osu_path,
        sequences,
    )
    prompt_ids = model_kwargs["decoder_input_ids"].to(teacher.device)

    # Populate shared encoder outputs from teacher once.
    teacher_mk = _ensure_encoder_outputs(teacher, model_kwargs)
    enc = teacher_mk.get("encoder_outputs")
    if enc is None:
        raise SystemExit("failed to materialize teacher encoder_outputs")

    draft_mk = {
        key: value
        for key, value in teacher_mk.items()
        if key
        not in {
            "past_key_values",
            "cache_position",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
    }
    draft_mk["encoder_outputs"] = enc

    teacher_mk_bench = dict(draft_mk)

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
    # Ensure prompt fits under the smallest prefix.
    prompt_len = int(prompt_ids.shape[1])
    prefixes = [p for p in prefixes if p >= prompt_len]
    if not prefixes:
        raise SystemExit(
            f"all prefixes < prompt_len={prompt_len}; got {args.prefixes}"
        )

    print(
        json.dumps(
            {
                "host": os.uname().nodename,
                "cuda": torch.cuda.get_device_name(0),
                "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
                "prompt_len": prompt_len,
                "prefixes": prefixes,
                "draft_ckpt": str(args.draft_ckpt),
            },
            indent=2,
        ),
        flush=True,
    )

    t0 = time.perf_counter()
    draft_report = bench_model_graph(
        model=draft,
        model_kwargs=draft_mk,
        prompt_ids=prompt_ids,
        dtype=dtype,
        prefixes=prefixes,
        warmup=args.warmup,
        iters=args.iters,
        label="draft_2layer",
    )
    main_report = bench_model_graph(
        model=teacher,
        model_kwargs=teacher_mk_bench,
        prompt_ids=prompt_ids,
        dtype=dtype,
        prefixes=prefixes,
        warmup=args.warmup,
        iters=args.iters,
        label="main_4layer",
    )
    wall_s = time.perf_counter() - t0

    draft_ms = float(draft_report["median_ms_per_token"])
    main_ms = float(main_report["median_ms_per_token"])
    ratio = draft_ms / main_ms if main_ms > 0 else float("inf")
    tip_ratio = draft_ms / TIP_FP16_MAIN_MS

    summary = {
        "section": 42,
        "track": "C",
        "workstream": "W3",
        "precision": args.precision,
        "host": os.uname().nodename,
        "cuda_device": torch.cuda.get_device_name(0),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(
            f"git -C {REPO} branch --show-current"
        ).read().strip(),
        "draft_ckpt": str(args.draft_ckpt),
        "prompt_meta": meta,
        "prompt_len": prompt_len,
        "prefixes": prefixes,
        "warmup": args.warmup,
        "iters": args.iters,
        "draft": draft_report,
        "main": main_report,
        "c_draft_ms_per_token_median": draft_ms,
        "c_main_ms_per_token_median": main_ms,
        "ratio_vs_measured_main": ratio,
        "ratio_vs_tip_fp16_2p731ms": tip_ratio,
        "target_ratio": TARGET_RATIO,
        "target_ms": TARGET_MS,
        "gate_pass_vs_measured_main": bool(ratio <= TARGET_RATIO),
        "gate_pass_vs_absolute_ms": bool(draft_ms <= TARGET_MS),
        "bench_wall_s": wall_s,
        "tip_fp16_main_ms_reference": TIP_FP16_MAIN_MS,
        "note": (
            "§42 draft fastpath microbench (CUDA graph replay). "
            "Not a 500 / TIER1 ship claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass_vs_measured_main"] and not summary["gate_pass_vs_absolute_ms"]:
        print(
            f"GATE MISS: c_draft={draft_ms:.4f}ms "
            f"ratio_vs_main={ratio:.3f} (need ≤{TARGET_RATIO} or ≤{TARGET_MS}ms)",
            flush=True,
        )
        return 2
    print(
        f"GATE: c_draft={draft_ms:.4f}ms ratio_vs_main={ratio:.3f} "
        f"abs_pass={summary['gate_pass_vs_absolute_ms']} "
        f"ratio_pass={summary['gate_pass_vs_measured_main']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
