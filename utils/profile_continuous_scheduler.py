from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INFERENCE_DIR = REPO_ROOT / "osuT5" / "osuT5" / "inference"


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_continuous_batching = _load_module_from_path(
    "_mapperatorinator_continuous_batching",
    INFERENCE_DIR / "continuous_batching.py",
)
_generation_compatibility = _load_module_from_path(
    "_mapperatorinator_generation_compatibility",
    INFERENCE_DIR / "generation_compatibility.py",
)

ContinuousBatchRequest = _continuous_batching.ContinuousBatchRequest
ContinuousBatchScheduler = _continuous_batching.ContinuousBatchScheduler
ContinuousBatchSchedulerConfig = _continuous_batching.ContinuousBatchSchedulerConfig
generation_compatibility_key = _generation_compatibility.generation_compatibility_key

CONTINUOUS_SCHEDULER_TOKEN_STATUS = "scheduler_only_scripted_tokens"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CPU-only continuous-batching scheduler dry run. This does not execute model generation "
            "and is only verifier/planning infrastructure for future server batching work."
        )
    )
    parser.add_argument("--requests-json", type=Path, default=None, help="Optional scripted request list JSON.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for the dry-run manifest.")
    parser.add_argument("--suite-id", default=None, help="Stable suite id. Defaults to a random short id.")
    parser.add_argument("--max-active-sequences", type=int, default=2)
    parser.add_argument("--max-wait-ms", type=int, default=0)
    parser.add_argument("--prefill-policy", default="serial", choices=["serial", "batch_prefill"])
    parser.add_argument("--decode-order-policy", default="arrival_order", choices=["arrival_order", "round_robin"])
    parser.add_argument(
        "--rng-policy",
        default="serial_global",
        choices=["serial_global", "per_request_generator", "documented_drift"],
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser.parse_args()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _runtime_environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
    }


def _token_sha256(tokens: list[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_request_specs() -> list[dict[str, Any]]:
    generate_kwargs = {
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 20,
        "temperature": 1.0,
        "num_beams": 1,
        "cfg_scale": 1.0,
    }
    return [
        {
            "request_id": "song0-window0",
            "prompt_tokens": 4,
            "max_new_tokens": 4,
            "eos_token_ids": [99],
            "script_tokens": [10, 11, 12, 99],
            "generate_kwargs": generate_kwargs,
            "metadata": {"song_id": "song0", "window_index": 0},
            "initial_rng_state_hash": "synthetic-rng-before-song0",
            "final_rng_state_hash": "synthetic-rng-after-song0",
            "logits_processor_state_hash": "synthetic-logits-song0",
            "cache_state_hash": "synthetic-cache-song0",
        },
        {
            "request_id": "song1-window0",
            "prompt_tokens": 4,
            "max_new_tokens": 3,
            "eos_token_ids": [99],
            "script_tokens": [20, 21, 22],
            "generate_kwargs": generate_kwargs,
            "metadata": {"song_id": "song1", "window_index": 0},
            "initial_rng_state_hash": "synthetic-rng-before-song1",
            "final_rng_state_hash": "synthetic-rng-after-song1",
            "logits_processor_state_hash": "synthetic-logits-song1",
            "cache_state_hash": "synthetic-cache-song1",
        },
        {
            "request_id": "song2-window0",
            "prompt_tokens": 4,
            "max_new_tokens": 2,
            "eos_token_ids": [99],
            "script_tokens": [30, 99],
            "generate_kwargs": generate_kwargs,
            "metadata": {"song_id": "song2", "window_index": 0},
            "initial_rng_state_hash": "synthetic-rng-before-song2",
            "final_rng_state_hash": "synthetic-rng-after-song2",
            "logits_processor_state_hash": "synthetic-logits-song2",
            "cache_state_hash": "synthetic-cache-song2",
        },
    ]


def _load_request_specs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _default_request_specs()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--requests-json must contain a JSON list of request objects.")
    return payload


def _build_request(spec: dict[str, Any]) -> ContinuousBatchRequest:
    try:
        request_id = str(spec["request_id"])
        prompt_tokens = int(spec["prompt_tokens"])
        max_new_tokens = int(spec["max_new_tokens"])
        script_tokens = [int(token) for token in spec["script_tokens"]]
    except KeyError as exc:
        raise ValueError(f"Missing request field: {exc.args[0]}") from exc
    generate_kwargs = dict(spec.get("generate_kwargs") or {})
    return ContinuousBatchRequest(
        request_id=request_id,
        compatibility_key=generation_compatibility_key(generate_kwargs),
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_ids=tuple(int(token) for token in spec.get("eos_token_ids", [])),
        script_tokens=script_tokens,
        metadata=dict(spec.get("metadata") or {}),
        initial_rng_state_hash=spec.get("initial_rng_state_hash"),
        final_rng_state_hash=spec.get("final_rng_state_hash"),
        logits_processor_state_hash=spec.get("logits_processor_state_hash"),
        cache_state_hash=spec.get("cache_state_hash"),
    )


def _request_manifest_row(request: dict[str, Any]) -> dict[str, Any]:
    generated_tokens = [int(token) for token in request.get("generated_tokens", [])]
    return {
        **request,
        "generated_token_sha256": _token_sha256(generated_tokens),
        "token_equivalence_status": CONTINUOUS_SCHEDULER_TOKEN_STATUS,
    }


def _aggregate_report(report: dict[str, Any], scheduler_wall_seconds: float) -> dict[str, Any]:
    requests = report.get("requests", [])
    stop_reason_counts = Counter(request.get("stop_reason") for request in requests)
    completed_requests = sum(1 for request in requests if request.get("stop_reason") is not None)
    generated_tokens = sum(int(request.get("generated_token_count", 0) or 0) for request in requests)
    cache_slot_events = report.get("cache_slot_events", [])
    return {
        "result_class": "continuous_scheduler_dry_run",
        "model_generation_executed": False,
        "request_count": len(requests),
        "completed_request_count": completed_requests,
        "total_generated_tokens": generated_tokens,
        "scheduler_cpu_wall_seconds": scheduler_wall_seconds,
        "scheduler_tokens_per_cpu_second": (
            generated_tokens / scheduler_wall_seconds if scheduler_wall_seconds > 0 else 0.0
        ),
        "active_batch_size_histogram": report.get("active_batch_size_histogram", {}),
        "stop_reason_counts": {str(key): value for key, value in sorted(stop_reason_counts.items(), key=lambda item: str(item[0]))},
        "cache_slot_acquire_count": sum(1 for event in cache_slot_events if event.get("event") == "acquire"),
        "cache_slot_release_count": sum(1 for event in cache_slot_events if event.get("event") == "release"),
    }


def run_scheduler_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    request_specs = _load_request_specs(args.requests_json)
    scheduler = ContinuousBatchScheduler(
        ContinuousBatchSchedulerConfig(
            max_active_sequences=args.max_active_sequences,
            max_wait_ms=args.max_wait_ms,
            prefill_policy=args.prefill_policy,
            decode_order_policy=args.decode_order_policy,
            rng_policy=args.rng_policy,
        )
    )
    for spec in request_specs:
        scheduler.enqueue(_build_request(spec))

    started = time.perf_counter()
    report = scheduler.run_until_idle(max_steps=args.max_steps).to_dict()
    scheduler_wall_seconds = time.perf_counter() - started
    request_rows = [_request_manifest_row(request) for request in report.get("requests", [])]
    report["requests"] = request_rows

    suite_id = args.suite_id or uuid.uuid4().hex[:8]
    return {
        "schema_version": 1,
        "suite_id": suite_id,
        "run_kind": "continuous_scheduler_dry_run",
        "result_class": "continuous_scheduler_dry_run",
        "model_generation_executed": False,
        "token_equivalence_status": CONTINUOUS_SCHEDULER_TOKEN_STATUS,
        "equivalence_scope": (
            "scheduler-only scripted-token lifecycle evidence; no model, sampling, RNG consumption, "
            "logits processors, cache tensors, or output files are executed"
        ),
        "requests_json_path": str(args.requests_json) if args.requests_json is not None else None,
        "request_count": len(request_rows),
        "config": report["config"],
        "compatibility_key": report["compatibility_key"],
        "active_batch_size_histogram": report["active_batch_size_histogram"],
        "steps": report["steps"],
        "requests": request_rows,
        "cache_slot_events": report["cache_slot_events"],
        "aggregate": _aggregate_report(report, scheduler_wall_seconds),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "runtime_environment": _runtime_environment(),
    }


def main() -> None:
    args = _parse_args()
    manifest = run_scheduler_dry_run(args)
    output_root = args.output_root or REPO_ROOT / "runs" / f"continuous-scheduler-dry-run-{manifest['suite_id']}"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "continuous_scheduler_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[continuous-scheduler] manifest saved to {manifest_path}")
    print(
        "[continuous-scheduler] requests={requests}, tokens={tokens}, active_hist={hist}".format(
            requests=manifest["aggregate"]["request_count"],
            tokens=manifest["aggregate"]["total_generated_tokens"],
            hist=manifest["active_batch_size_histogram"],
        )
    )


if __name__ == "__main__":
    main()
