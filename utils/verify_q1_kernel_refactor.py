from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


PREFIXES = (128, 576, 640)
HEADS = 6
HEAD_DIM = 64
MAX_CACHE_LENGTH = 832


def _dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision {name!r}")


def _random_tensor(
    generator: torch.Generator,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        dtype=dtype,
        device="cuda",
    )


def _install_repo(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    kernel = repo_root / "osuT5/osuT5/inference/optimized/kernels/q1_attention.py"
    if not kernel.is_file():
        raise FileNotFoundError(f"missing q1 kernel at {kernel}")
    sys.path.insert(0, str(repo_root))


def _assert_untouched_cache(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    write_position: int,
) -> None:
    if not torch.equal(before[..., :write_position, :], after[..., :write_position, :]):
        raise RuntimeError("fused q1 kernel changed cache entries before the active slot")
    if not torch.equal(
        before[..., write_position + 1 :, :],
        after[..., write_position + 1 :, :],
    ):
        raise RuntimeError("fused q1 kernel changed cache entries after the active slot")


def capture(repo_root: Path, precision: str, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("q1 kernel capture requires CUDA")
    _install_repo(repo_root)
    from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
        native_q1_attention,
        native_q1_rope_cache_attention,
        preload_native_q1_attention,
    )

    dtype = _dtype(precision)
    generator = torch.Generator(device="cpu").manual_seed(20260713)
    query = _random_tensor(generator, (1, HEADS, 1, HEAD_DIM), dtype=dtype)
    keys = _random_tensor(
        generator,
        (1, HEADS, MAX_CACHE_LENGTH, HEAD_DIM),
        dtype=dtype,
    )
    values = _random_tensor(
        generator,
        (1, HEADS, MAX_CACHE_LENGTH, HEAD_DIM),
        dtype=dtype,
    )
    qkv = _random_tensor(generator, (1, 1, 3, HEADS, HEAD_DIM), dtype=dtype)
    cos = _random_tensor(generator, (1, 1, HEAD_DIM), dtype=dtype)
    sin = _random_tensor(generator, (1, 1, HEAD_DIM), dtype=dtype)
    base_cache_keys = _random_tensor(
        generator,
        (1, HEADS, MAX_CACHE_LENGTH, HEAD_DIM),
        dtype=dtype,
    )
    base_cache_values = _random_tensor(
        generator,
        (1, HEADS, MAX_CACHE_LENGTH, HEAD_DIM),
        dtype=dtype,
    )

    preload_native_q1_attention()
    states: dict[str, dict[str, torch.Tensor]] = {}
    for prefix in PREFIXES:
        mask = torch.linspace(-0.25, 0.0, prefix, device="cuda", dtype=torch.float32)
        plain_output = native_q1_attention(
            query,
            keys[..., :prefix, :],
            values[..., :prefix, :],
            mask,
        )
        plain_repeat = native_q1_attention(
            query,
            keys[..., :prefix, :],
            values[..., :prefix, :],
            mask,
        )
        if not torch.equal(plain_output, plain_repeat):
            raise RuntimeError(f"plain q1 output is not deterministic at prefix {prefix}")

        write_position = prefix - 1
        cache_position = torch.tensor([write_position], device="cuda", dtype=torch.long)
        cache_keys = base_cache_keys.clone()
        cache_values = base_cache_values.clone()
        fused_output = native_q1_rope_cache_attention(
            qkv,
            cache_keys,
            cache_values,
            cos,
            sin,
            cache_position,
            mask,
            prefix,
        )
        torch.cuda.synchronize()
        _assert_untouched_cache(
            base_cache_keys,
            cache_keys,
            write_position=write_position,
        )
        _assert_untouched_cache(
            base_cache_values,
            cache_values,
            write_position=write_position,
        )

        repeat_keys = base_cache_keys.clone()
        repeat_values = base_cache_values.clone()
        fused_repeat = native_q1_rope_cache_attention(
            qkv,
            repeat_keys,
            repeat_values,
            cos,
            sin,
            cache_position,
            mask,
            prefix,
        )
        torch.cuda.synchronize()
        if not torch.equal(fused_output, fused_repeat):
            raise RuntimeError(f"fused q1 output is not deterministic at prefix {prefix}")
        if not torch.equal(cache_keys, repeat_keys) or not torch.equal(
            cache_values,
            repeat_values,
        ):
            raise RuntimeError(f"fused q1 cache write is not deterministic at prefix {prefix}")

        states[str(prefix)] = {
            "plain_output": plain_output.cpu(),
            "fused_output": fused_output.cpu(),
            "cache_keys": cache_keys.cpu(),
            "cache_values": cache_values.cpu(),
        }

    payload = {
        "schema_version": 1,
        "precision": precision,
        "prefixes": PREFIXES,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "max_cache_length": MAX_CACHE_LENGTH,
        "states": states,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _compare_values(reference: Any, candidate: Any, path: str, failures: list[str]) -> None:
    if isinstance(reference, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            failures.append(f"{path}: candidate is not a tensor")
        elif reference.dtype != candidate.dtype:
            failures.append(
                f"{path}: dtype differs ({reference.dtype} != {candidate.dtype})"
            )
        elif reference.shape != candidate.shape:
            failures.append(
                f"{path}: shape differs ({tuple(reference.shape)} != {tuple(candidate.shape)})"
            )
        elif not torch.equal(reference, candidate):
            max_abs = float((reference.float() - candidate.float()).abs().max().item())
            failures.append(f"{path}: tensor differs (max_abs={max_abs})")
        return
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            failures.append(f"{path}: candidate is not a mapping")
            return
        if set(reference) != set(candidate):
            failures.append(f"{path}: mapping keys differ")
            return
        for key in sorted(reference):
            _compare_values(reference[key], candidate[key], f"{path}.{key}", failures)
        return
    if reference != candidate:
        failures.append(f"{path}: value differs ({reference!r} != {candidate!r})")


def compare(reference_path: Path, candidate_path: Path, json_output: Path | None) -> None:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    failures: list[str] = []
    _compare_values(reference, candidate, "capture", failures)
    report = {
        "exact": not failures,
        "failures": failures,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(rendered + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--repo-root", type=Path, required=True)
    capture_parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.repo_root, args.precision, args.output)
    else:
        compare(args.reference, args.candidate, args.json_output)


if __name__ == "__main__":
    main()
