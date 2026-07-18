#!/usr/bin/env python3
"""§58 turbo-on-tiger first gate: c_verify(K) / c_q1 on tiger's uniform decode graph.

Measures CUDA-event replay of PR #120 ``CUDAGraphDecoder`` at q_len=1 vs q_len=K
on the same StaticCache bucket — no fused kernels, no active-prefix overlay,
no §54 Stage B grind.

Expect ~1.1–1.3×. Not a full-song reciprocal. Not a 500 claim.
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
from transformers.modeling_outputs import BaseModelOutput

REPO = Path(os.environ.get("PROBE_REPO", ".")).resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

__import__("config")

from inference import load_model_with_server  # noqa: E402
from osuT5.osuT5.inference.cache_utils import get_cache  # noqa: E402
from osuT5.osuT5.inference.compiled_decode import (  # noqa: E402
    CUDAGraphDecoder,
    CaptureError,
    _pick_bucket,
    _reset_cache,
    get_k_decoder,
    neutralize_dynamic_rope,
)
from osuT5.osuT5.inference.preprocessor import Preprocessor  # noqa: E402
from osuT5.osuT5.inference.server import precompute_encoder_outputs  # noqa: E402
from osuT5.osuT5.inference.turbo.kv_rollback import rewind_self_cache  # noqa: E402


def load_args(config_name: str, overrides: list[str]):
    config_dir = REPO / "configs" / "inference"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = hydra.compose(
            config_name=config_name.removeprefix("inference/"), overrides=overrides
        )
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_cuda_events(fn, *, warmup: int, iters: int) -> float:
    """Return mean milliseconds per call via CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    sync()
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        sync()
        times.append(float(start.elapsed_time(end)))
    return sum(times) / len(times)


def build_window(
    model,
    tokenizer,
    args,
    audio: Path,
    *,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Real encoder hidden + synthetic decoder prompt for graph timing."""
    preprocessor = Preprocessor(args)
    samples = preprocessor.load(str(audio))
    sequences, _times, _song_len = preprocessor.segment(samples)
    frames = sequences[:1].contiguous()  # first window only
    if frames.ndim == 1:
        frames = frames.unsqueeze(0)

    enc_cpu = precompute_encoder_outputs(model, frames, {}, None)
    enc_hidden = enc_cpu.to(model.device, dtype=model.dtype).contiguous()

    sos = int(getattr(tokenizer, "sos_id", 1))
    prompt = torch.full(
        (1, int(prompt_len)), sos, dtype=torch.long, device=model.device
    )
    # Mild variation so RoPE/cache writes are not completely degenerate.
    vocab = int(model.config.vocab_size)
    noise = torch.randint(
        low=10, high=min(200, vocab - 1), size=(1, max(1, prompt_len - 1)),
        device=model.device, dtype=torch.long,
    )
    prompt[:, 1:] = noise
    meta = {
        "prompt_len": int(prompt_len),
        "enc_shape": list(enc_hidden.shape),
        "frames_shape": list(frames.shape),
        "audio": str(audio),
    }
    return prompt, enc_hidden, meta


@torch.no_grad()
def prefill_cache(model, cache, prompt: torch.Tensor, enc_hidden: torch.Tensor) -> None:
    _reset_cache(cache)
    enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
    cache_position = torch.arange(prompt.shape[1], device=prompt.device)
    inputs = model.prepare_inputs_for_generation(
        prompt,
        past_key_values=cache,
        use_cache=True,
        encoder_outputs=enc_out,
        decoder_attention_mask=torch.ones_like(prompt),
        cache_position=cache_position,
    )
    model(**inputs)


@torch.no_grad()
def measure_q1(
    model,
    *,
    prompt: torch.Tensor,
    enc_hidden: torch.Tensor,
    cache_len: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    decoder = CUDAGraphDecoder(
        model, batch_size=1, enc_shape=enc_hidden.shape, dtype=model.dtype, q_len=1
    )
    cache = get_cache(model, batch_size=1, num_beams=1, cfg_scale=1.0, max_cache_len=cache_len)
    t_cap0 = time.perf_counter()
    try:
        decoder.capture(cache)
    except Exception as exc:  # noqa: BLE001
        raise CaptureError(str(exc)) from exc
    capture_s = time.perf_counter() - t_cap0

    prefill_cache(model, cache, prompt, enc_hidden)
    decoder.set_encoder_hidden(enc_hidden)
    prefix = int(prompt.shape[1])
    tok = torch.tensor([[42]], device=prompt.device, dtype=torch.long)
    cache_pos = torch.tensor([prefix], device=prompt.device, dtype=torch.long)

    def step():
        # Rewind the single write slot so successive replays stay at prefix.
        rewind_self_cache(cache, prefix, occupied_end=prefix + 1)
        decoder.replay(tok, cache_pos)

    ms = timed_cuda_events(step, warmup=warmup, iters=iters)
    return {
        "q1_step_ms": ms,
        "capture_seconds": capture_s,
        "path": "tiger_cuda_graph_q1",
        "cache_len": cache_len,
        "prefix_len": prefix,
    }


@torch.no_grad()
def measure_verify_k(
    model,
    *,
    prompt: torch.Tensor,
    enc_hidden: torch.Tensor,
    cache_len: int,
    k: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    decoder, cache = get_k_decoder(
        model,
        batch_size=1,
        cfg_scale=1.0,
        enc_shape=enc_hidden.shape,
        dtype=model.dtype,
        cache_len=cache_len,
        q_len=k,
    )
    # Fresh prefill on the K-graph's cache object.
    prefill_cache(model, cache, prompt, enc_hidden)
    decoder.set_encoder_hidden(enc_hidden)
    prefix = int(prompt.shape[1])
    tokens = torch.randint(
        low=10, high=100, size=(1, k), device=prompt.device, dtype=torch.long
    )
    cache_pos = torch.arange(prefix, prefix + k, device=prompt.device, dtype=torch.long)

    def step():
        rewind_self_cache(cache, prefix, occupied_end=prefix + k)
        decoder.replay(tokens, cache_pos)

    ms = timed_cuda_events(step, warmup=warmup, iters=iters)
    return {
        "k": k,
        "c_verify_ms": ms,
        "path": "tiger_cuda_graph_qk",
        "cache_len": cache_len,
        "prefix_len": prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="v32")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", default="3,5")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra hydra overrides (precision=fp16 use_server=false ...)",
    )
    args_cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for §58 c_verify scout")

    overrides = [
        "use_server=false",
        "precision=fp16",
        "fast_decoder_loop=true",
        "attn_implementation=sdpa",
        "cfg_scale=1.0",
        "num_beams=1",
        *args_cli.overrides,
    ]
    args = load_args(args_cli.config_name, overrides)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=1,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        eval_mode=True,
        gamemode=getattr(args, "gamemode", 0),
        auto_select_gamemode_model=getattr(args, "auto_select_gamemode_model", True),
        fast_decoder_loop=True,
    )
    model.eval()
    neutralize_dynamic_rope(model)

    prompt, enc_hidden, window_meta = build_window(
        model, tokenizer, args, args_cli.audio, prompt_len=args_cli.prompt_len
    )
    model_max = int(model.config.max_target_positions)
    cache_len = _pick_bucket(int(prompt.shape[1]) + 64, model_max)

    q1 = measure_q1(
        model,
        prompt=prompt,
        enc_hidden=enc_hidden,
        cache_len=cache_len,
        warmup=args_cli.warmup,
        iters=args_cli.iters,
    )
    ks = [int(x) for x in args_cli.ks.split(",") if x.strip()]
    verify_rows: list[dict[str, Any]] = []
    best_ratio = None
    best_k = None
    for k in ks:
        row = measure_verify_k(
            model,
            prompt=prompt,
            enc_hidden=enc_hidden,
            cache_len=cache_len,
            k=k,
            warmup=args_cli.warmup,
            iters=args_cli.iters,
        )
        ratio = float(row["c_verify_ms"]) / float(q1["q1_step_ms"])
        row["c_verify_ratio_vs_q1"] = ratio
        row["gate_expect_lo"] = 1.1
        row["gate_expect_hi"] = 1.3
        row["in_expect_band"] = 1.1 <= ratio <= 1.3
        verify_rows.append(row)
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_k = k

    out = {
        "section": 58,
        "track": "turbo-on-tiger",
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} branch --show-current").read().strip(),
        "gpu": torch.cuda.get_device_name(0),
        "precision": args.precision,
        "window": window_meta,
        "cache_len": cache_len,
        "q1": q1,
        "verify": verify_rows,
        "best_k": best_k,
        "best_ratio": best_ratio,
        "gate": "c_verify / q1 expect ~1.1–1.3× on tiger uniform path",
        "note": (
            "Tiger CUDAGraphDecoder q_len=K vs q_len=1. No fused verify. "
            "Optimized tip 55949274 frozen. No 500 claim."
        ),
        "campaign_tip_frozen": "55949274",
    }
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    args_cli.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    if best_ratio is None:
        raise SystemExit("no K measured")
    print(
        f"BEST k={best_k} c_verify_ratio={best_ratio:.4f} "
        f"q1={q1['q1_step_ms']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
