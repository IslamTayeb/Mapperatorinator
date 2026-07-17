#!/usr/bin/env python3
"""§41 budget-gate microbench: c_verify(K) vs one optimized Q=1 step.

Measures wall time of K-token teacher verify (StaticCache + active-prefix,
optional CUDA graph) against one optimized single-token CUDA-graph step on
the same FP16 teacher. Target: c_verify / c_step ≤ 1.2 for K in {4,5,8}.

Not a full-song reciprocal. Not a 500 claim.
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

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__import__("config")

from inference import load_model_with_engine  # noqa: E402
from osuT5.osuT5.inference import GenerationConfig, Preprocessor, Processor  # noqa: E402
from osuT5.osuT5.inference.cache_utils import get_cache  # noqa: E402
from osuT5.osuT5.inference.engine_binding import unwrap_engine_binding  # noqa: E402
from osuT5.osuT5.inference.optimized.single.decode_loop import (  # noqa: E402
    _bucketed_prefix_length,
    _capture_decode_cuda_graph,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
)
from osuT5.osuT5.inference.turbo.verify_fastpath import (  # noqa: E402
    TeacherVerifyFastpath,
    allocate_teacher_static_cache,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context  # noqa: E402
from osuT5.osuT5.tokenizer import ContextType  # noqa: E402


def load_args(config_name: str, overrides: list[str]):
    config_dir = REPO / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            config_name=config_name.removeprefix("inference/"), overrides=overrides
        )
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def build_first_map_window(processor, args_inf, generation_config, song_length, osu_path, sequences):
    sys.path.insert(0, str(REPO / "utils"))
    import s37_tiny_draft_train_smoke as s37  # type: ignore

    in_ctx, out_ctx, map_i, req_special, cond_kwargs, n_timing = s37.prepare_contexts(
        processor, args_inf, generation_config, song_length, osu_path
    )
    frames, frame_time = sequences[0][0], sequences[1][0]
    frames_p = processor.prepare_frames(frames)
    frame_time_v = frame_time.item()
    cond_prompt, uncond_prompt = processor.get_prompts(
        processor.prepare_context_sequences(in_ctx, frame_time_v, False, req_special),
        processor.prepare_context_sequences(out_ctx[: map_i + 1], frame_time_v, True, req_special),
    )
    [prompt, _uncond], _max_len = processor.pad_prompts([cond_prompt, uncond_prompt])
    prompt_ids = prompt[0, : int(prompt[0].ne(processor.tokenizer.pad_id).sum().item())]
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
    return model_kwargs, {
        "prompt_len": int(prompt_ids.numel()),
        "timing_points": n_timing,
    }


def _move_kwargs(model, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value
        for key, value in model_kwargs.items()
    }
    out = {
        key: (
            value.to(model.dtype)
            if key != "inputs"
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            else value
        )
        for key, value in out.items()
    }
    if "inputs" in out:
        frames = out.pop("inputs")
        out["frames"] = out.get("frames", frames)
    return out


@torch.no_grad()
def prefill(model, prompt_ids: torch.Tensor, model_kwargs: dict[str, Any]):
    mk = dict(model_kwargs)
    mk.pop("cache_position", None)
    mk["decoder_attention_mask"] = torch.ones(
        (1, int(prompt_ids.shape[1])),
        device=prompt_ids.device,
        dtype=torch.long,
    )
    model_inputs = model.prepare_inputs_for_generation(prompt_ids, **mk)
    if "cache_position" in model_inputs:
        mk["cache_position"] = model_inputs["cache_position"]
    outputs = model(**model_inputs, return_dict=True)
    mk = model._update_model_kwargs_for_generation(
        outputs, mk, is_encoder_decoder=True
    )
    return outputs.logits[:, -1, :].float().squeeze(0), mk


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def measure_optimized_q1_step(
    model,
    *,
    prompt_ids: torch.Tensor,
    model_kwargs: dict[str, Any],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Prefill + capture one active-prefix Q=1 CUDA graph; time replay."""
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0)
    mk = dict(model_kwargs)
    mk["past_key_values"] = cache
    mk["use_cache"] = True
    _, mk = prefill(model, prompt_ids, mk)

    # Append one dummy token position for decode step timing.
    tok = torch.tensor([[1]], device=prompt_ids.device, dtype=prompt_ids.dtype)
    ids = torch.cat([prompt_ids, tok], dim=-1)
    mk["decoder_attention_mask"] = torch.ones(
        (1, int(ids.shape[1])), device=ids.device, dtype=torch.long
    )
    mk.pop("cache_position", None)
    model_inputs = model.prepare_inputs_for_generation(ids, **mk)
    cur_len = int(ids.shape[1])
    max_cache = int(cache.get_max_cache_shape())
    prefix = _bucketed_prefix_length(cur_len, 64, max_cache)
    static_inputs = _clone_static_graph_inputs(model_inputs)
    graph, outputs, capture_s = _capture_decode_cuda_graph(
        model,
        static_inputs,
        active_prefix_length=prefix,
        warmup=1,
    )

    def replay():
        _copy_static_graph_inputs(static_inputs, model_inputs)
        graph.replay()
        _ = outputs.logits

    ms = timed_cuda(replay, warmup=warmup, iters=iters) * 1000.0
    return {
        "q1_step_ms": ms,
        "prefix_bucket": prefix,
        "capture_seconds": capture_s,
        "graph_signature": list(_cuda_graph_signature(prefix, model_inputs)),
    }


@torch.no_grad()
def measure_verify_k(
    model,
    *,
    prompt_ids: torch.Tensor,
    model_kwargs: dict[str, Any],
    k: int,
    use_cuda_graph: bool,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Time K-token teacher verify via §41 TeacherVerifyFastpath."""
    cache = allocate_teacher_static_cache(model, cfg_scale=1.0)
    mk = dict(model_kwargs)
    mk["past_key_values"] = cache
    mk["use_cache"] = True
    _, mk = prefill(model, prompt_ids, mk)

    draft = torch.randint(
        low=10,
        high=100,
        size=(1, k),
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    )
    ids = torch.cat([prompt_ids, draft], dim=-1)
    fp = TeacherVerifyFastpath(
        model=model,
        bucket_size=64,
        use_cuda_graph=use_cuda_graph,
    )

    # One setup forward (capture or eager) then crop back for fair timed iters.
    outputs, mk_after = fp.forward_k(
        decoder_input_ids=ids,
        model_kwargs=dict(mk),
        k=k,
        allow_graph=use_cuda_graph,
    )
    _ = outputs.logits
    # Zero-tail crop for StaticCache so timed iters rewrite the same slots.
    L = int(prompt_ids.shape[1])
    self_cache = cache.self_attention_cache
    for layer in self_cache.layers:
        if getattr(layer, "is_initialized", False) and L < layer.keys.shape[2]:
            layer.keys[:, :, L:, :].zero_()
            layer.values[:, :, L:, :].zero_()

    # Rebuild kwargs length to L for timed path.
    mk_timed = dict(mk)
    mk_timed["past_key_values"] = cache
    mk_timed["decoder_attention_mask"] = torch.ones(
        (1, L), device=prompt_ids.device, dtype=torch.long
    )
    mk_timed.pop("cache_position", None)

    def one_verify():
        # Crop each iter so StaticCache slots L: are rewritten cleanly.
        for layer in self_cache.layers:
            if getattr(layer, "is_initialized", False) and L < layer.keys.shape[2]:
                layer.keys[:, :, L:, :].zero_()
                layer.values[:, :, L:, :].zero_()
        mk_local = dict(mk_timed)
        outs, _ = fp.forward_k(
            decoder_input_ids=ids,
            model_kwargs=mk_local,
            k=k,
            allow_graph=use_cuda_graph,
        )
        _ = outs.logits

    ms = timed_cuda(one_verify, warmup=warmup, iters=iters) * 1000.0
    return {
        "k": k,
        "verify_ms": ms,
        "use_cuda_graph": use_cuda_graph,
        "eager_calls": fp.eager_calls,
        "graph_replays": fp.graph_replays,
        "graph_captures": fp.graph_captures,
        "logits_shape": list(outputs.logits.shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="profile_salvalai")
    ap.add_argument("--audio", default=os.environ.get("MAPPERATORINATOR_AUDIO", ""))
    ap.add_argument("--osu", default=os.environ.get("MAPPERATORINATOR_OSU", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="4,5,8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--cuda-graph-verify", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for §41 microbench")

    overrides = [
        "device=cuda",
        "precision=fp16",
        "attn_implementation=sdpa",
        "inference_engine=optimized",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
        "seed=12345",
        "profile_inference=false",
    ]
    if args.audio:
        overrides.append(f"audio_path={args.audio}")
    args_inf = load_args(args.config_name, overrides)

    model_binding, tokenizer = load_model_with_engine(
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
    model, _runtime = unwrap_engine_binding(model_binding)
    model.eval()

    preprocessor = Preprocessor(args_inf)
    processor = Processor(args_inf, model_binding, tokenizer)
    audio = Path(args_inf.audio_path)
    song_length = preprocessor.get_song_length(str(audio))
    sequences = preprocessor.load(str(audio))

    osu_path = Path(args.osu) if args.osu else None
    if osu_path is None or not osu_path.is_file():
        tip_roots = [
            Path(
                "/work/imt11/Mapperatorinator/runs/"
                "shared-rope-device-state-fp16-auth-49964133"
            ),
            Path(
                "/work/imt11/Mapperatorinator/runs/"
                "exact-rope-device-state-fp16-auth-49964133"
            ),
        ]
        cands: list[Path] = []
        for tip_osu in tip_roots:
            if tip_osu.is_dir():
                cands.extend(sorted(tip_osu.glob("**/beatmap*.osu")))
        if not cands:
            raise SystemExit(
                "Need --osu or tip dump under *-rope-device-state-fp16-auth-49964133"
            )
        osu_path = cands[0]

    generation_config = GenerationConfig(
        gamemode=args_inf.gamemode,
        difficulty=args_inf.difficulty,
        year=args_inf.year,
        hitsounded=args_inf.hitsounded,
        descriptors=list(args_inf.descriptors) if args_inf.descriptors else None,
    )

    model_kwargs_cpu, meta = build_first_map_window(
        processor, args_inf, generation_config, song_length, osu_path, sequences
    )
    model_kwargs = _move_kwargs(model, model_kwargs_cpu)
    prompt_ids = model_kwargs.pop("decoder_input_ids")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    results: dict[str, Any] = {
        "section": 41,
        "track": "C",
        "workstream": "W2",
        "precision": "fp16",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "prompt_len": meta["prompt_len"],
        "osu": str(osu_path),
        "warmup": args.warmup,
        "iters": args.iters,
        "target_ratio": 1.2,
        "context_type": ContextType.MAP.value,
        "note": "Budget-gate microbench only. No 500 claim. Tip 55949274/366.11.",
    }

    # Arm the same native decode contexts optimized MAP generation uses.
    with generation_profile_context(
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
        native_cross_mlp_tail=True,
        optimized_expected_dtype=torch.float16,
    ):
        q1 = measure_optimized_q1_step(
            model,
            prompt_ids=prompt_ids,
            model_kwargs=model_kwargs,
            warmup=args.warmup,
            iters=args.iters,
        )
        results["optimized_q1_step"] = q1
        q1_ms = float(q1["q1_step_ms"])

        verify_rows = []
        for k in ks:
            row = measure_verify_k(
                model,
                prompt_ids=prompt_ids,
                model_kwargs=model_kwargs,
                k=k,
                use_cuda_graph=bool(args.cuda_graph_verify),
                warmup=args.warmup,
                iters=args.iters,
            )
            row["ratio_vs_q1"] = float(row["verify_ms"]) / q1_ms if q1_ms > 0 else None
            row["gate_pass"] = (
                row["ratio_vs_q1"] is not None and row["ratio_vs_q1"] <= 1.2
            )
            verify_rows.append(row)
        results["verify"] = verify_rows
        results["best_ratio"] = min(
            (r["ratio_vs_q1"] for r in verify_rows if r["ratio_vs_q1"] is not None),
            default=None,
        )
        results["all_gate_pass"] = all(r["gate_pass"] for r in verify_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    # Always 0: gate lives in JSON for the ledger (budget microbench).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
