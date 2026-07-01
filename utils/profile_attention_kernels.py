"""Profile SDPA vs FlashAttention kernels for Mapperatorinator-like shapes.

This utility is intentionally standalone and opt-in. It does not load model
weights or audio; it uses fixed tensors that match the VarWhisper small v3
attention shape so backend overhead can be inspected without sampling noise.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


@dataclasses.dataclass(frozen=True)
class AttentionCase:
    name: str
    kind: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int
    causal: bool
    note: str


@dataclasses.dataclass
class CaseTensors:
    q_sdpa: torch.Tensor
    k_sdpa: torch.Tensor
    v_sdpa: torch.Tensor
    q_fa2: torch.Tensor
    qkv_fa2: torch.Tensor | None
    kv_fa2: torch.Tensor | None


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _tensor_shape(tensor: torch.Tensor | None) -> list[int] | None:
    if tensor is None:
        return None
    return list(tensor.shape)


def _time_attr(event: object, *names: str) -> float:
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return float(value)
    return 0.0


@contextlib.contextmanager
def _nvtx_record(name: str):
    pushed = False
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        pushed = True
    with record_function(name):
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()


def build_cases(heads: int, head_dim: int) -> list[AttentionCase]:
    cases: list[AttentionCase] = []
    for kv_len in (64, 128, 512, 1024, 2048):
        cases.append(
            AttentionCase(
                name=f"decode_self_kv{kv_len}",
                kind="self_decode",
                batch=1,
                heads=heads,
                q_len=1,
                kv_len=kv_len,
                head_dim=head_dim,
                causal=False,
                note="Single-token cached decoder self-attention. Causal is disabled here so SDPA and FA2 use the same maskless math.",
            )
        )
    for kv_len in (512, 1024, 2048):
        cases.append(
            AttentionCase(
                name=f"cross_attn_kv{kv_len}",
                kind="cross",
                batch=1,
                heads=heads,
                q_len=1,
                kv_len=kv_len,
                head_dim=head_dim,
                causal=False,
                note="Single-token decoder cross-attention against encoder states.",
            )
        )
    for seq_len in (128, 512, 1024, 2048):
        cases.append(
            AttentionCase(
                name=f"prefill_self_len{seq_len}",
                kind="self_prefill",
                batch=1,
                heads=heads,
                q_len=seq_len,
                kv_len=seq_len,
                head_dim=head_dim,
                causal=True,
                note="Full causal self-attention prefill-like case.",
            )
        )
    cases.extend(
        [
            AttentionCase(
                name="batch8_decode_self_kv512",
                kind="self_decode",
                batch=8,
                heads=heads,
                q_len=1,
                kv_len=512,
                head_dim=head_dim,
                causal=False,
                note="Batch-size what-if for single-token cached decoder self-attention.",
            ),
            AttentionCase(
                name="batch8_cross_attn_kv2048",
                kind="cross",
                batch=8,
                heads=heads,
                q_len=1,
                kv_len=2048,
                head_dim=head_dim,
                causal=False,
                note="Batch-size what-if for decoder cross-attention.",
            ),
            AttentionCase(
                name="batch8_prefill_self_len256",
                kind="self_prefill",
                batch=8,
                heads=heads,
                q_len=256,
                kv_len=256,
                head_dim=head_dim,
                causal=True,
                note="Batch-size what-if for causal prefill.",
            ),
        ]
    )
    return cases


def make_tensors(case: AttentionCase, dtype: torch.dtype, device: torch.device) -> CaseTensors:
    q_base = torch.randn(
        case.batch,
        case.heads,
        case.q_len,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    k_base = torch.randn(
        case.batch,
        case.heads,
        case.kv_len,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    v_base = torch.randn(
        case.batch,
        case.heads,
        case.kv_len,
        case.head_dim,
        dtype=dtype,
        device=device,
    )
    q_fa2 = q_base.transpose(1, 2).contiguous()
    if case.kind == "self_prefill" and case.q_len == case.kv_len:
        qkv_fa2 = torch.stack(
            (
                q_base.transpose(1, 2),
                k_base.transpose(1, 2),
                v_base.transpose(1, 2),
            ),
            dim=2,
        ).contiguous()
        kv_fa2 = None
    else:
        qkv_fa2 = None
        kv_fa2 = torch.stack(
            (k_base.transpose(1, 2), v_base.transpose(1, 2)),
            dim=2,
        ).contiguous()
    return CaseTensors(
        q_sdpa=q_base.contiguous(),
        k_sdpa=k_base.contiguous(),
        v_sdpa=v_base.contiguous(),
        q_fa2=q_fa2,
        qkv_fa2=qkv_fa2,
        kv_fa2=kv_fa2,
    )


def sdpa_attention_only(case: AttentionCase, tensors: CaseTensors) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        tensors.q_sdpa,
        tensors.k_sdpa,
        tensors.v_sdpa,
        dropout_p=0.0,
        is_causal=case.causal,
    )


def sdpa_repo_like(case: AttentionCase, tensors: CaseTensors) -> torch.Tensor:
    out = sdpa_attention_only(case, tensors)
    return out.transpose(1, 2).contiguous().view(case.batch, case.q_len, case.heads * case.head_dim)


def make_fa2_callables(case: AttentionCase, tensors: CaseTensors, deterministic: bool) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    try:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func, flash_attn_qkvpacked_func
    except Exception as exc:  # pragma: no cover - exercised on GPU envs
        raise RuntimeError("flash-attn is required for backend=fa2") from exc

    def attention_only() -> torch.Tensor:
        if tensors.qkv_fa2 is not None:
            return flash_attn_qkvpacked_func(
                tensors.qkv_fa2,
                dropout_p=0.0,
                causal=case.causal,
                window_size=(-1, -1),
                deterministic=deterministic,
            )
        if tensors.kv_fa2 is None:
            raise RuntimeError("FA2 kv tensor was not prepared")
        return flash_attn_kvpacked_func(
            tensors.q_fa2,
            tensors.kv_fa2,
            dropout_p=0.0,
            causal=case.causal,
            window_size=(-1, -1),
            deterministic=deterministic,
        )

    def repo_like() -> torch.Tensor:
        q = tensors.q_sdpa.transpose(1, 2).contiguous()
        if case.kind == "self_prefill" and case.q_len == case.kv_len:
            qkv = torch.stack(
                (
                    q,
                    tensors.k_sdpa.transpose(1, 2),
                    tensors.v_sdpa.transpose(1, 2),
                ),
                dim=2,
            ).contiguous()
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=0.0,
                causal=case.causal,
                window_size=(-1, -1),
                deterministic=deterministic,
            )
        else:
            kv = torch.stack(
                (
                    tensors.k_sdpa.transpose(1, 2),
                    tensors.v_sdpa.transpose(1, 2),
                ),
                dim=2,
            ).contiguous()
            out = flash_attn_kvpacked_func(
                q,
                kv,
                dropout_p=0.0,
                causal=case.causal,
                window_size=(-1, -1),
                deterministic=deterministic,
            )
        return out.view(case.batch, case.q_len, case.heads * case.head_dim)

    return attention_only, repo_like


def benchmark_cuda(
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / iters)

    return {
        "avg_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": samples,
    }


def profile_callable(
    fn: Callable[[], torch.Tensor],
    label: str,
    profile_iters: int,
    profile_top: int,
    trace_path: Path | None,
) -> dict[str, object]:
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profile_kwargs = {
        "activities": activities,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
    }
    try:
        profiler_context = profile(**profile_kwargs, acc_events=True)
    except TypeError:
        profiler_context = profile(**profile_kwargs)

    with profiler_context as prof:
        for _ in range(profile_iters):
            with _nvtx_record(label):
                fn()
        torch.cuda.synchronize()

    key_averages = prof.key_averages(group_by_input_shape=True)
    try:
        table = key_averages.table(sort_by="self_cuda_time_total", row_limit=profile_top)
    except Exception:
        table = key_averages.table(sort_by="self_device_time_total", row_limit=profile_top)
    events: list[dict[str, object]] = []
    for event in key_averages:
        cuda_total = _time_attr(event, "cuda_time_total", "device_time_total")
        self_cuda_total = _time_attr(event, "self_cuda_time_total", "self_device_time_total")
        cpu_total = _time_attr(event, "cpu_time_total")
        self_cpu_total = _time_attr(event, "self_cpu_time_total")
        if cuda_total <= 0 and self_cuda_total <= 0 and cpu_total <= 0:
            continue
        events.append(
            {
                "key": event.key,
                "count": int(getattr(event, "count", 0)),
                "cuda_time_total_us": cuda_total,
                "self_cuda_time_total_us": self_cuda_total,
                "cpu_time_total_us": cpu_total,
                "self_cpu_time_total_us": self_cpu_total,
                "input_shapes": getattr(event, "input_shapes", None),
            }
        )

    events.sort(key=lambda item: float(item["self_cuda_time_total_us"]), reverse=True)
    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))
    return {
        "label": label,
        "profile_iters": profile_iters,
        "table": table,
        "events": events[:profile_top],
        "trace_path": str(trace_path) if trace_path is not None else None,
    }


def environment_metadata() -> dict[str, object]:
    device = torch.device("cuda")
    meta: dict[str, object] = {
        "created_at_unix": time.time(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_device_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_count": torch.cuda.device_count(),
        "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        meta["sdpa_cudnn_enabled"] = bool(torch.backends.cuda.cudnn_sdp_enabled())
    try:
        import flash_attn

        meta["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
    except Exception as exc:
        meta["flash_attn_import_error"] = repr(exc)
    return meta


def _result_summary_line(result: dict[str, object]) -> str:
    case = result["case"]
    backend = result["backend"]
    mode = result["mode"]
    timing = result["timing"]
    assert isinstance(timing, dict)
    avg_ms = float(timing["avg_ms"])
    min_ms = float(timing["min_ms"])
    return f"{case:28s} {backend:4s} {mode:14s} avg={avg_ms:9.4f}ms min={min_ms:9.4f}ms"


def write_text_summary(path: Path, results: list[dict[str, object]]) -> None:
    lines = ["Mapperatorinator attention kernel profile", ""]
    lines.append("Per-case timings:")
    for result in results:
        lines.append(_result_summary_line(result))

    lines.extend(["", "FA2 / SDPA ratios by case and mode:"])
    by_key: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    for result in results:
        by_key.setdefault((str(result["case"]), str(result["mode"])), {})[str(result["backend"])] = result
    for (case, mode), backend_results in sorted(by_key.items()):
        sdpa = backend_results.get("sdpa")
        fa2 = backend_results.get("fa2")
        if not sdpa or not fa2:
            continue
        sdpa_timing = sdpa["timing"]
        fa2_timing = fa2["timing"]
        assert isinstance(sdpa_timing, dict)
        assert isinstance(fa2_timing, dict)
        ratio = float(fa2_timing["avg_ms"]) / float(sdpa_timing["avg_ms"])
        lines.append(f"{case:28s} {mode:14s} fa2/sdpa={ratio:7.3f}x")

    lines.extend(["", "Top profiler tables:"])
    for result in results:
        prof = result.get("profiler")
        if not isinstance(prof, dict):
            continue
        lines.extend(
            [
                "",
                f"## {prof['label']}",
                str(prof["table"]),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def filter_cases(cases: Iterable[AttentionCase], filters: list[str]) -> list[AttentionCase]:
    if not filters:
        return list(cases)
    selected = []
    for case in cases:
        if any(part in case.name for part in filters):
            selected.append(case)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile SDPA and FlashAttention 2 for VarWhisper-like attention shapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("attention-kernel-profile"))
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--backends", default="sdpa,fa2", help="Comma-separated subset of sdpa,fa2.")
    parser.add_argument("--modes", default="attention_only,repo_like", help="Comma-separated subset of attention_only,repo_like.")
    parser.add_argument("--case-filter", default="", help="Comma-separated substrings; empty means all cases.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--profile-top", type=int, default=30)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument(
        "--trace-cases",
        default="decode_self_kv512,cross_attn_kv2048,prefill_self_len512,batch8_decode_self_kv512",
        help="Comma-separated case-name substrings that should export Chrome traces.",
    )
    parser.add_argument("--trace-all", action="store_true")
    parser.add_argument("--deterministic-flash-attn", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this on a GPU Slurm allocation, not a login node or local Mac.")

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    backends = _parse_csv(args.backends)
    modes = _parse_csv(args.modes)
    case_filters = _parse_csv(args.case_filter)
    trace_filters = _parse_csv(args.trace_cases)

    invalid_backends = set(backends) - {"sdpa", "fa2"}
    invalid_modes = set(modes) - {"attention_only", "repo_like"}
    if invalid_backends:
        raise ValueError(f"Unknown backends: {sorted(invalid_backends)}")
    if invalid_modes:
        raise ValueError(f"Unknown modes: {sorted(invalid_modes)}")
    if "fa2" in backends and dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention 2 requires --dtype fp16 or bf16.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    cases = filter_cases(build_cases(heads=args.heads, head_dim=args.head_dim), case_filters)
    if not cases:
        raise ValueError("No attention cases matched --case-filter.")

    metadata = environment_metadata()
    metadata["shape_defaults"] = {
        "model_family": "VarWhisper small v3",
        "d_model": args.heads * args.head_dim,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "source_seq_len_reference": 2048,
        "target_seq_len_reference": 2560,
    }
    print(json.dumps({"metadata": metadata}, sort_keys=True), flush=True)

    results: list[dict[str, object]] = []
    device = torch.device("cuda")
    for case in cases:
        tensors = make_tensors(case, dtype=dtype, device=device)
        fa2_callables: tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]] | None = None
        if "fa2" in backends:
            fa2_callables = make_fa2_callables(case, tensors, deterministic=args.deterministic_flash_attn)

        for backend in backends:
            for mode in modes:
                label = f"{backend}.{mode}.{case.name}"
                if backend == "sdpa":
                    fn = sdpa_attention_only if mode == "attention_only" else sdpa_repo_like
                    callable_fn = lambda fn=fn, case=case, tensors=tensors: fn(case, tensors)
                else:
                    if fa2_callables is None:
                        raise RuntimeError("FA2 callables were not initialized")
                    callable_fn = fa2_callables[0] if mode == "attention_only" else fa2_callables[1]

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                timing = benchmark_cuda(
                    callable_fn,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_reserved = torch.cuda.max_memory_reserved()
                should_trace = args.trace_all or any(part in case.name for part in trace_filters)
                trace_path = args.output_dir / f"{_safe_name(label)}.trace.json" if should_trace and not args.skip_profiler else None
                profiler_result = None
                if not args.skip_profiler and args.profile_iters > 0:
                    profiler_result = profile_callable(
                        callable_fn,
                        label=label,
                        profile_iters=args.profile_iters,
                        profile_top=args.profile_top,
                        trace_path=trace_path,
                    )

                result: dict[str, object] = {
                    "case": case.name,
                    "case_details": dataclasses.asdict(case),
                    "backend": backend,
                    "mode": mode,
                    "dtype": args.dtype,
                    "timing": timing,
                    "memory": {
                        "peak_allocated_bytes": peak_allocated,
                        "peak_reserved_bytes": peak_reserved,
                    },
                    "tensor_shapes": {
                        "q_sdpa": _tensor_shape(tensors.q_sdpa),
                        "k_sdpa": _tensor_shape(tensors.k_sdpa),
                        "v_sdpa": _tensor_shape(tensors.v_sdpa),
                        "q_fa2": _tensor_shape(tensors.q_fa2),
                        "qkv_fa2": _tensor_shape(tensors.qkv_fa2),
                        "kv_fa2": _tensor_shape(tensors.kv_fa2),
                    },
                    "profiler": profiler_result,
                }
                results.append(result)
                print(_result_summary_line(result), flush=True)

    payload = {
        "metadata": metadata,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "cases": [dataclasses.asdict(case) for case in cases],
        "results": results,
    }
    json_path = args.output_dir / "attention_kernel_profile.json"
    txt_path = args.output_dir / "attention_kernel_profile.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_text_summary(txt_path, results)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {txt_path}", flush=True)


if __name__ == "__main__":
    main()
