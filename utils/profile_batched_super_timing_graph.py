"""Profile eager-parallel versus opt-in batched CUDA-graph super timing."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


SCHEMA_VERSION = 1
MIN_HEADROOM_BYTES = 512 * 1024 * 1024


def _load_args(config_name: str, overrides: list[str]):
    import hydra
    from omegaconf import DictConfig, OmegaConf

    __import__("config")
    config_dir = REPO_ROOT / "configs" / "inference"
    resolved_name = config_name.removeprefix("inference/")
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name=resolved_name, overrides=overrides)
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ("git", "-C", str(REPO_ROOT), *args),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_args(args, *, batch_size: int, mode: str) -> None:
    required = {
        "device": "cuda",
        "attn_implementation": "sdpa",
        "use_server": False,
        "inference_engine": "v32",
        "parallel": False,
        "super_timing": False,
        "timer_num_beams": 1,
        "timer_cfg_scale": 1.0,
    }
    mismatches = {
        key: {"actual": getattr(args, key), "required": value}
        for key, value in required.items()
        if getattr(args, key) != value
    }
    if mismatches:
        raise ValueError(
            "batched super-timing profiler requires a neutral public config: "
            f"{mismatches}"
        )
    if mode not in {"eager", "graph"}:
        raise ValueError("mode must be eager or graph")
    if args.precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be fp32 or fp16")
    if int(args.timer_iterations) <= 0:
        raise ValueError("timer_iterations must be positive")
    if batch_size <= 0 or batch_size > int(args.max_batch_size):
        raise ValueError(
            f"batch_size must be in [1, {int(args.max_batch_size)}]"
        )


def _fixed_audio_offsets(generator, *, seed: int) -> list[int]:
    half_window_ms = int(generator.miliseconds_per_sequence // 2)
    if half_window_ms <= 0:
        raise RuntimeError("super-timing window must have a positive duration")
    rng = np.random.RandomState(seed)
    return [
        int(rng.randint(-half_window_ms, half_window_ms))
        for _ in range(int(generator.iterations))
    ]


def _window_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for record in records:
        iteration = record.get("super_timing_iteration")
        start = record.get("batch_start_index")
        batch_size = record.get("batch_size")
        prompts = record.get("prompt_sha256_per_sample")
        prompt_counts = record.get("prompt_tokens_per_sample")
        raw_tokens = record.get("raw_generated_token_ids_per_sample")
        generated_counts = record.get("generated_tokens_per_sample")
        if not isinstance(iteration, int) or not isinstance(start, int):
            raise RuntimeError("generation record is missing iteration/window indices")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise RuntimeError("generation record has an invalid batch size")
        row_fields = (prompts, prompt_counts, raw_tokens, generated_counts)
        if any(not isinstance(value, list) or len(value) != batch_size for value in row_fields):
            raise RuntimeError("generation record has incomplete per-row evidence")
        for row in range(batch_size):
            tokens = raw_tokens[row]
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) for token in tokens
            ):
                raise RuntimeError("raw generated token evidence is malformed")
            windows.append(
                {
                    "iteration": iteration,
                    "window_index": start + row,
                    "prompt_sha256": prompts[row],
                    "prompt_tokens": int(prompt_counts[row]),
                    "raw_generated_token_ids": tokens,
                    "generated_tokens": int(generated_counts[row]),
                }
            )
    windows.sort(key=lambda row: (row["iteration"], row["window_index"]))
    identities = [
        (row["iteration"], row["window_index"])
        for row in windows
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate super-timing window evidence")
    return windows


def _dispatch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = (
        "native_q1_rope_cache_self_attention",
        "native_q1_self_attention",
        "q1_bmm_cross_attention",
        "native_cross_mlp_tail",
    )
    hits = {family: 0 for family in families}
    modes: set[str] = set()
    policies: set[str] = set()
    for record in records:
        mode = record.get("optimized_dispatch_mode")
        if mode is not None:
            modes.add(str(mode))
        policy = record.get("optimized_dispatch_policy")
        if policy is not None:
            policies.add(json.dumps(policy, sort_keys=True))
        record_hits = record.get("optimized_dispatch_capture_hits") or {}
        for family in families:
            hits[family] += int(record_hits.get(family, 0) or 0)
    return {
        "capture_hits": hits,
        "modes": sorted(modes),
        "policies": [json.loads(value) for value in sorted(policies)],
    }


def _graph_summary(
    runtime,
    *,
    expected_batch_sizes: set[int],
) -> dict[str, Any]:
    if runtime is None:
        return {
            "count": 0,
            "capture_seconds": 0.0,
            "decode_replays": 0,
            "entries": [],
            "cache_states": [],
            "cache_ownership_pass": True,
        }
    session = runtime.new_context_state()
    entries: list[dict[str, Any]] = []
    for entry in session.graph_cache.values():
        static_inputs = entry.get("static_inputs") or {}
        batch_tensor = None
        for key in ("decoder_input_ids", "input_ids", "inputs_embeds"):
            value = static_inputs.get(key)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                batch_tensor = value
                break
        if batch_tensor is None:
            batch_tensor = next(
                (
                    value
                    for value in static_inputs.values()
                    if isinstance(value, torch.Tensor) and value.ndim > 1
                ),
                None,
            )
        floating = next(
            (
                value
                for value in static_inputs.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            ),
            None,
        )
        entries.append(
            {
                "batch_size": (
                    int(batch_tensor.shape[0]) if batch_tensor is not None else None
                ),
                "active_prefix_length": int(entry["active_prefix_length"]),
                "dtype": str(floating.dtype) if floating is not None else None,
                "device": str(floating.device) if floating is not None else None,
                "capture_seconds": float(entry["capture_seconds"]),
                "decode_replays": int(entry["decode_replays"]),
            }
        )
    entries.sort(
        key=lambda row: (
            row["batch_size"] or -1,
            row["active_prefix_length"],
            row["dtype"] or "",
        )
    )
    cache_states: list[dict[str, Any]] = []
    ownership_pass = True
    cache_object_ids: set[int] = set()
    tensor_storage_pointers: set[int] = set()
    observed_batch_sizes: set[int] = set()
    for signature, cache in session.caches.items():
        expected_batch = int(signature[1]) * int(signature[2]) * int(signature[3])
        observed_batch_sizes.add(int(signature[1]))
        cache_unique = id(cache) not in cache_object_ids
        cache_object_ids.add(id(cache))
        families: dict[str, list[torch.Tensor]] = {}
        for attention_name, attention_cache in (
            ("self", cache.self_attention_cache),
            ("cross", cache.cross_attention_cache),
        ):
            layers = getattr(attention_cache, "layers", [])
            for field in ("keys", "values"):
                families[f"{attention_name}_{field}"] = [
                    value
                    for layer in layers
                    if isinstance((value := getattr(layer, field, None)), torch.Tensor)
                ]
        family_nonempty = all(families.values())
        tensor_rows: list[dict[str, Any]] = []
        tensors_pass = family_nonempty
        current_storage_pointers: set[int] = set()
        for family, tensors in families.items():
            for tensor in tensors:
                pointer = int(tensor.untyped_storage().data_ptr())
                unique_within_state = pointer not in current_storage_pointers
                unique_across_states = pointer not in tensor_storage_pointers
                current_storage_pointers.add(pointer)
                tensor_pass = all((
                    tensor.ndim >= 1,
                    int(tensor.shape[0]) == expected_batch,
                    str(tensor.dtype) == signature[4],
                    str(tensor.device) == signature[5],
                    unique_within_state,
                    unique_across_states,
                ))
                tensors_pass &= tensor_pass
                tensor_rows.append({
                    "family": family,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "unique_storage_within_state": unique_within_state,
                    "unique_storage_across_states": unique_across_states,
                    "pass": tensor_pass,
                })
        tensor_storage_pointers.update(current_storage_pointers)
        row_pass = cache_unique and tensors_pass
        ownership_pass &= row_pass
        cache_states.append(
            {
                "batch_size": int(signature[1]),
                "num_beams": int(signature[2]),
                "cfg_multiplier": int(signature[3]),
                "dtype": signature[4],
                "device": signature[5],
                "cache_object_unique": cache_unique,
                "families_nonempty": family_nonempty,
                "tensors": tensor_rows,
                "ownership_pass": row_pass,
            }
        )
    cache_states.sort(key=lambda row: (row["batch_size"], row["dtype"]))
    observed_batches_pass = observed_batch_sizes == expected_batch_sizes
    ownership_pass &= observed_batches_pass
    return {
        "count": session.graph_count,
        "capture_seconds": session.graph_capture_seconds,
        "decode_replays": session.graph_decode_replays,
        "entries": entries,
        "cache_states": cache_states,
        "expected_cache_batch_sizes": sorted(expected_batch_sizes),
        "observed_cache_batch_sizes": sorted(observed_batch_sizes),
        "observed_cache_batches_pass": observed_batches_pass,
        "cache_ownership_pass": ownership_pass,
    }


def compare_exact(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    contract_fields = (
        "precision",
        "batch_size",
        "seed",
        "timer_iterations",
        "timer_num_beams",
        "timer_cfg_scale",
        "audio_sha256",
        "model_path",
        "git_commit",
        "torch_version",
        "cuda_version",
        "cuda_device",
        "public_wiring",
        "phase",
        "repetition",
    )
    checks = {
        "reference_is_eager": reference["metadata"]["mode"] == "eager",
        "candidate_is_graph": candidate["metadata"]["mode"] == "graph",
        **{
            f"contract_{field}": (
                field in reference["metadata"]
                and field in candidate["metadata"]
                and reference["metadata"][field]
                == candidate["metadata"][field]
            )
            for field in contract_fields
        },
        "audio_offsets": (
            reference["workload"]["audio_offsets_ms"]
            == candidate["workload"]["audio_offsets_ms"]
        ),
        "per_window_tokens": (
            reference["workload"]["windows"]
            == candidate["workload"]["windows"]
        ),
        "raw_histograms": (
            reference["output"]["raw_histograms"]
            == candidate["output"]["raw_histograms"]
        ),
        "smoothed_histograms": (
            reference["output"]["smoothed_histograms"]
            == candidate["output"]["smoothed_histograms"]
        ),
        "tpbs": reference["output"]["tpbs"] == candidate["output"]["tpbs"],
        "measure_counts": (
            reference["output"]["measure_counts"]
            == candidate["output"]["measure_counts"]
        ),
        "final_events": (
            reference["output"]["final_events"]
            == candidate["output"]["final_events"]
        ),
        "final_event_times": (
            reference["output"]["final_event_times"]
            == candidate["output"]["final_event_times"]
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


@torch.no_grad()
def profile(args, *, mode: str, batch_size: int, seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("batched super-timing profiling requires CUDA")
    _assert_args(args, batch_size=batch_size, mode=mode)

    from inference import (
        compile_args,
        get_config,
        load_model_with_server,
        setup_inference_environment,
    )
    from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
    from osuT5.osuT5.inference.profiler import InferenceProfiler
    from osuT5.osuT5.inference.super_timing_generator import SuperTimingGenerator

    compile_args(args, verbose=False)
    setup_inference_environment(seed)
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=False,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=False,
    )
    model.generation_config.disable_compile = True
    expected_dtype = torch.float32 if args.precision == "fp32" else torch.float16
    if model.dtype != expected_dtype:
        raise TypeError(
            f"loaded model dtype {model.dtype}, expected {expected_dtype}"
        )

    runtime = None
    bound_model = model
    if mode == "graph":
        from osuT5.osuT5.inference.optimized.single.engine import (
            build_experimental_batched_super_timing_runtime,
        )

        runtime = build_experimental_batched_super_timing_runtime(
            args.precision,
            nominal_batch_size=batch_size,
        )
        bound_model = InferenceEngineBinding(model, runtime)

    profiler = InferenceProfiler(enabled=True)
    generator = SuperTimingGenerator(args, bound_model, tokenizer, profiler=profiler)
    generator.processor.max_batch_size = batch_size
    if generator.processor.num_beams != 1 or generator.processor.cfg_scale != 1.0:
        raise RuntimeError("super-timing generator changed the greedy batch contract")
    offsets = _fixed_audio_offsets(generator, seed=seed)
    generator.profile_audio_offsets = offsets
    generation_config, _ = get_config(args)
    audio = generator.preprocessor.load(args.audio_path)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_before = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    events, event_times = generator.generate(
        audio,
        generation_config,
        verbose=False,
    )
    torch.cuda.synchronize()
    complete_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(model.device).total_memory)
    diagnostics = generator.last_profile_diagnostics
    if diagnostics is None:
        raise RuntimeError("super-timing diagnostics were not captured")
    if diagnostics["final_event_times"] != [int(value) for value in event_times]:
        raise RuntimeError("final event-time diagnostics disagree with output")

    records = profiler.generation
    windows = _window_manifest(records)
    total_generated = sum(row["generated_tokens"] for row in windows)
    generation_wall = sum(float(row["wall_seconds"]) for row in records)
    model_wall = sum(float(row["model_elapsed_seconds"]) for row in records)
    prompt_wall = sum(float(row.get("prompt_wall_seconds", 0.0)) for row in records)
    active_row_seconds = sum(
        float(row["wall_seconds"]) * int(row["batch_size"])
        for row in records
    )
    warm_records = [
        row
        for row in records
        if float(row.get("decode_graph_capture_seconds_delta", 0.0) or 0.0)
        == 0.0
    ]
    capture_affected_records = [
        row
        for row in records
        if float(row.get("decode_graph_capture_seconds_delta", 0.0) or 0.0)
        != 0.0
    ]
    batches_by_iteration: dict[int, list[int]] = {}
    for row in records:
        batches_by_iteration.setdefault(
            int(row["super_timing_iteration"]),
            [],
        ).append(int(row["batch_size"]))
    nominal_unused = sum(
        batch_size - actual
        for batches in batches_by_iteration.values()
        for actual in batches
    )
    expected_cache_batch_sizes = {
        actual
        for batches in batches_by_iteration.values()
        for actual in batches
    }
    graph = _graph_summary(
        runtime,
        expected_batch_sizes=expected_cache_batch_sizes,
    )
    dispatch = _dispatch_summary(records)
    if mode == "graph" and batch_size > 1:
        if any(dispatch["capture_hits"].values()):
            raise RuntimeError("B>1 graph run executed a specialized B1 dispatch")
        if dispatch["modes"] != ["framework_batch"]:
            raise RuntimeError("B>1 graph run did not record framework dispatch")
    if mode == "graph" and batch_size == 1:
        if dispatch["modes"] != ["accepted_batch1"]:
            raise RuntimeError("B1 graph run did not preserve accepted dispatch")
        if dispatch["capture_hits"]["q1_bmm_cross_attention"] <= 0:
            raise RuntimeError("B1 graph run did not execute accepted q1-BMM cross")

    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "mode": mode,
            "batch_size": batch_size,
            "precision": args.precision,
            "seed": seed,
            "timer_iterations": int(args.timer_iterations),
            "timer_num_beams": int(args.timer_num_beams),
            "timer_cfg_scale": float(args.timer_cfg_scale),
            "public_wiring": False,
            "audio_path": str(args.audio_path),
            "audio_sha256": _sha256_file(args.audio_path),
            "model_path": str(args.model_path),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "git_dirty": bool(_git_value("status", "--porcelain")),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "timing": {
            "complete_super_timing_seconds": complete_seconds,
            "generation_outer_wall_seconds": generation_wall,
            "model_elapsed_seconds": model_wall,
            "prompt_stack_seconds": prompt_wall,
            "graph_capture_seconds": graph["capture_seconds"],
            "model_wall_excluding_capture_seconds": max(
                0.0,
                model_wall - graph["capture_seconds"],
            ),
            "warm_decode_wall_seconds": sum(
                float(row["model_elapsed_seconds"]) for row in warm_records
            ),
            "warm_decode_record_count": len(warm_records),
            "capture_affected_decode_wall_seconds": sum(
                float(row["model_elapsed_seconds"])
                for row in capture_affected_records
            ),
            "capture_affected_decode_record_count": len(capture_affected_records),
            "aggregate_tokens_per_second": (
                total_generated / complete_seconds if complete_seconds > 0 else 0.0
            ),
            "mean_active_row_tokens_per_second": (
                total_generated / active_row_seconds
                if active_row_seconds > 0
                else 0.0
            ),
        },
        "workload": {
            "audio_offsets_ms": offsets,
            "iterations": int(args.timer_iterations),
            "window_count": len(windows),
            "generated_tokens": total_generated,
            "batches_by_iteration": {
                str(key): value for key, value in sorted(batches_by_iteration.items())
            },
            "tail_batch_sizes": sorted(
                {
                    actual
                    for batches in batches_by_iteration.values()
                    for actual in batches
                    if actual != batch_size
                }
            ),
            "nominal_unused_tail_capacity": nominal_unused,
            "captured_inactive_lanes": 0,
            "windows": windows,
        },
        "graphs": graph,
        "dispatch": dispatch,
        "memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_bytes": total_memory,
            "headroom_bytes": total_memory - peak_reserved,
            "minimum_headroom_bytes": MIN_HEADROOM_BYTES,
            "headroom_pass": total_memory - peak_reserved >= MIN_HEADROOM_BYTES,
        },
        "output": diagnostics,
        "gates": {
            "cache_ownership_pass": graph["cache_ownership_pass"],
            "graph_capture_pass": (
                mode == "eager"
                or (graph["count"] > 0 and graph["decode_replays"] > 0)
            ),
            "headroom_pass": total_memory - peak_reserved >= MIN_HEADROOM_BYTES,
        },
    }
    report["gates"]["pass"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    args = _load_args(cli.config_name, list(cli.overrides))
    report = profile(
        args,
        mode=cli.mode,
        batch_size=cli.batch_size,
        seed=cli.seed,
    )
    report["metadata"].update({
        "phase": cli.phase,
        "repetition": cli.repetition,
        "order": cli.order,
    })
    if cli.reference_report is not None:
        reference = json.loads(cli.reference_report.read_text(encoding="utf-8"))
        report["parity"] = compare_exact(reference, report)
        report["gates"]["exact_parity_pass"] = report["parity"]["pass"]
        report["gates"]["pass"] = all(
            value for key, value in report["gates"].items() if key != "pass"
        )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "metadata": report["metadata"],
        "timing": report["timing"],
        "graphs": {
            key: report["graphs"][key]
            for key in ("count", "capture_seconds", "decode_replays")
        },
        "memory": report["memory"],
        "gates": report["gates"],
        "parity": report.get("parity"),
    }, indent=2))
    if not report["gates"]["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
