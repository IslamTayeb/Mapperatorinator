#!/usr/bin/env python3
"""§49 W-DG microbench: γ=3 draft chain in ONE CUDA graph (ms total).

Gate: 3 drafts ≤1.2 ms total. Kill if draft chain >2 ms.
Uses §43 1-layer draft by default; ports §42 StaticCache/q1 fastpath.
Philox-safe in-graph sample (§29c). Persistent chain-graph cache.
Not a 500 / TIER1 ship claim.
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
from osuT5.osuT5.inference.turbo.draft_chain_graph import (  # noqa: E402
    GATE_MS_TOTAL,
    KILL_MS_TOTAL,
    DraftChainGraphRunner,
    time_chain_graph_ms,
)

DEFAULT_DRAFT = (
    "/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/draft_1layer.pt"
)
TIP_FP16_MAIN_MS = 1000.0 / 366.11


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
    from transformers.modeling_outputs import BaseModelOutput

    from osuT5.osuT5.inference.cache_utils import get_cache
    from osuT5.osuT5.inference.turbo.speculate import _forward_decoder, _move_model_kwargs

    if model_kwargs.get("encoder_outputs") is not None:
        return {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_kwargs.items()
        }
    mk2 = _move_model_kwargs(model, model_kwargs)
    prompt_ids = mk2.pop("decoder_input_ids")
    mk2["past_key_values"] = get_cache(model, 1, 1, 1.0)
    mk2["use_cache"] = True
    outputs, mk2 = _forward_decoder(
        model, decoder_input_ids=prompt_ids, model_kwargs=mk2
    )
    if mk2.get("encoder_outputs") is None:
        enc = getattr(outputs, "encoder_last_hidden_state", None)
        if enc is None:
            raise RuntimeError("teacher forward returned no encoder_last_hidden_state")
        mk2["encoder_outputs"] = BaseModelOutput(last_hidden_state=enc)
    mk2["decoder_input_ids"] = prompt_ids
    return mk2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--audio", default=os.environ.get("MAPPERATORINATOR_AUDIO", ""))
    ap.add_argument("--draft-ckpt", default=os.environ.get(
        "MAPPERATORINATOR_TURBO_DRAFT_CKPT", DEFAULT_DRAFT
    ))
    ap.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--prefixes", default="128,256")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--osu-hint", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for §49 draft chain bench")

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
    os.environ.setdefault("MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH", "1")

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

    prefixes = [int(x) for x in args.prefixes.split(",") if x.strip()]
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
                "gamma": args.gamma,
                "draft_ckpt": str(args.draft_ckpt),
                "gate_ms_total": GATE_MS_TOTAL,
                "kill_ms_total": KILL_MS_TOTAL,
            },
            indent=2,
        ),
        flush=True,
    )

    bind_kwargs = dict(draft_mk)
    bind_kwargs.pop("decoder_input_ids", None)
    rows = []
    t0 = time.perf_counter()
    for prefix in prefixes:
        runner = DraftChainGraphRunner(
            draft,
            dtype=dtype,
            gamma=int(args.gamma),
            temperature=float(args.temperature),
            enable_native_kernels=True,
            cuda_graph_warmup=2,
        )
        runner.bind(model_kwargs=bind_kwargs, prompt_ids=prompt_ids)
        runner.prefill()
        # JIT native kernels once before chain capture.
        runner.fastpath.warm_native_kernels()
        row = time_chain_graph_ms(
            runner,
            prefix_length=prefix,
            warmup=args.warmup,
            iters=args.iters,
        )
        row["label"] = "draft_chain"
        row["layers"] = runner.fastpath.layer_count
        print(
            f"[chain] prefix={prefix} ms_chain={row['ms_per_chain']:.4f} "
            f"ms/tok={row['ms_per_token']:.4f} "
            f"gate_pass={row['gate_pass']} kill={row['kill']} "
            f"captures={row['chain_captures']} "
            f"cache_size={row['persistent_cache_size']}",
            flush=True,
        )
        rows.append(row)

    wall_s = time.perf_counter() - t0
    ms_values = [float(r["ms_per_chain"]) for r in rows]
    median_ms = float(sorted(ms_values)[len(ms_values) // 2])
    min_ms = float(min(ms_values))
    max_ms = float(max(ms_values))
    gate_pass = bool(median_ms <= GATE_MS_TOTAL)
    kill = bool(median_ms > KILL_MS_TOTAL)

    summary = {
        "section": 49,
        "track": "C",
        "workstream": "W-DG",
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
        "gamma": int(args.gamma),
        "temperature": float(args.temperature),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "ms_per_chain_median": median_ms,
        "ms_per_chain_min": min_ms,
        "ms_per_chain_max": max_ms,
        "ms_per_token_median": median_ms / float(args.gamma),
        "gate_ms_total": GATE_MS_TOTAL,
        "kill_ms_total": KILL_MS_TOTAL,
        "gate_pass": gate_pass,
        "kill": kill,
        "decision": (
            "KILL" if kill else ("PASS" if gate_pass else "MISS_UNDER_KILL")
        ),
        "bench_wall_s": wall_s,
        "tip_fp16_main_ms_reference": TIP_FP16_MAIN_MS,
        "campaign_tip": "55949274",
        "auth_fp16_main_tps": 366.11,
        "note": (
            "§49 graphed draft chain microbench (ONE graph, γ steps, "
            "Philox in-graph sample + embed feedback). Not a 500 claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    if kill:
        print(
            f"KILL: ms_chain_median={median_ms:.4f} > {KILL_MS_TOTAL} ms",
            flush=True,
        )
        return 3
    if not gate_pass:
        print(
            f"GATE MISS: ms_chain_median={median_ms:.4f} "
            f"(need ≤{GATE_MS_TOTAL} ms; kill>{KILL_MS_TOTAL})",
            flush=True,
        )
        return 2
    print(
        f"GATE PASS: ms_chain_median={median_ms:.4f} ≤ {GATE_MS_TOTAL} ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
