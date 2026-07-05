from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import (  # noqa: E402
    compile_args,
    file_artifact_metadata,
    generate,
    get_config,
    get_default_logger,
    get_server_address,
    get_server_runtime_key,
    load_model_with_server,
    setup_inference_environment,
    should_load_separate_timing_model,
)
from osuT5.osuT5.inference.profiler import InferenceProfiler  # noqa: E402
from osuT5.osuT5.inference.server import InferenceClient  # noqa: E402
from utils.inference_profile_metrics import first_record_breakdown  # noqa: E402
from utils.profile_inference_suite import (  # noqa: E402
    _aggregate_batch_summaries,
    _load_song_entries,
    _profile_batch_summary,
    _summary_for_label,
    _token_sha256,
    _flatten_token_ids,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run concurrent full-song requests through the existing static IPC InferenceServer. "
            "This measures server/static batching throughput, not single-song TPS."
        )
    )
    parser.add_argument("--config-name", default="profile_salvalai", help="Hydra inference config name.")
    parser.add_argument("--song-list", required=True, help="YAML/JSON/text song list with at least five songs.")
    parser.add_argument("--suite-id", default=None, help="Stable suite id. Defaults to a random short id.")
    parser.add_argument("--output-root", default=None, help="Directory for per-request outputs and manifest.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of concurrent passes over the song list.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent client requests. Defaults to the song count.",
    )
    parser.add_argument(
        "--launch-stagger-seconds",
        type=float,
        default=0.0,
        help="Delay between request submissions; default 0 launches the batch as tightly as Python threads allow.",
    )
    parser.add_argument(
        "--server-start-timeout-seconds",
        type=float,
        default=900.0,
        help="Fail if the owner server socket is not ready within this many seconds.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=7200.0,
        help="Fail an individual server request if no response arrives within this many seconds.",
    )
    parser.add_argument(
        "--suite-timeout-seconds",
        type=float,
        default=7200.0,
        help="Fail the concurrent suite if all requests do not finish within this many seconds.",
    )
    parser.add_argument(
        "--server-idle-timeout-seconds",
        type=float,
        default=7200.0,
        help="Keep the owner server alive between request bursts for this many seconds.",
    )
    parser.add_argument(
        "--allow-short-suite",
        action="store_true",
        help="Allow fewer than 5 songs for harness smoke tests.",
    )
    parser.add_argument(
        "--allow-existing-server",
        action="store_true",
        help=(
            "Permit reuse of an existing IPC server socket. Default is to fail loudly so stale "
            "servers with different runtime flags cannot contaminate batching profiles."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. max_batch_size=5 use_server=true.",
    )
    return parser.parse_args()


def _compose_config(config_name: str, overrides: list[str]) -> Any:
    config_dir = REPO_ROOT / "configs" / "inference"
    with hydra.initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return hydra.compose(config_name=config_name, overrides=overrides)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _runtime_environment() -> dict[str, Any]:
    env_keys = [
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "TORCH_LOGS",
        "TMPDIR",
    ]
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "argv": sys.argv,
        "python": sys.version,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "env": {key: os.environ.get(key) for key in env_keys},
    }


def _server_socket_paths(args: Any) -> dict[str, str]:
    server_runtime_key = get_server_runtime_key(
        max_batch_size=args.max_batch_size,
        server_batch_timeout=args.server_batch_timeout,
        device=args.device,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        generation_compile=args.inference_generation_compile,
    )
    paths = {
        "main": get_server_address(
            args.model_path,
            lora_path=args.lora_path,
            gamemode=args.gamemode,
            auto_select_gamemode_model=args.auto_select_gamemode_model,
            server_runtime_key=server_runtime_key,
        )
    }
    if should_load_separate_timing_model(args):
        paths["timing"] = get_server_address(
            args.model_path,
            gamemode=args.gamemode,
            auto_select_gamemode_model=False,
            server_runtime_key=server_runtime_key,
        )
    return paths


def _validate_no_existing_server(socket_paths: dict[str, str]) -> None:
    if os.name != "posix":
        return
    existing = {name: path for name, path in socket_paths.items() if Path(path).exists()}
    if not existing:
        return
    details = ", ".join(f"{name}={path}" for name, path in sorted(existing.items()))
    raise RuntimeError(
        "Refusing to run static server batch profile with an existing IPC socket. "
        "This harness needs a fresh server so runtime flags, RNG state, and metadata are known. "
        f"Existing sockets: {details}. Stop the old server/remove stale sockets, or pass "
        "--allow-existing-server only for an explicitly documented reuse run."
    )


def _server_config_fingerprint(args: Any) -> dict[str, Any]:
    return {
        "model_path": str(args.model_path),
        "lora_path": str(args.lora_path) if args.lora_path is not None else None,
        "gamemode": args.gamemode,
        "auto_select_gamemode_model": bool(args.auto_select_gamemode_model),
        "device": str(args.device),
        "precision": str(args.precision),
        "attn_implementation": str(args.attn_implementation),
        "max_batch_size": int(args.max_batch_size),
        "server_batch_timeout": float(args.server_batch_timeout),
        "inference_generation_compile": bool(args.inference_generation_compile),
    }


def _validate_server_identity(
        args: Any,
        *,
        expected_socket_paths: dict[str, str],
        expected_config_fingerprint: dict[str, Any],
) -> None:
    socket_paths = _server_socket_paths(args)
    if socket_paths != expected_socket_paths:
        raise RuntimeError(
            "Static server batch request resolved different server socket paths. "
            f"expected={expected_socket_paths}, actual={socket_paths}"
        )
    fingerprint = _server_config_fingerprint(args)
    if fingerprint != expected_config_fingerprint:
        raise RuntimeError(
            "Static server batch request changed server-loading configuration. "
            f"expected={expected_config_fingerprint}, actual={fingerprint}"
        )


def _validate_server_sockets_ready(socket_paths: dict[str, str]) -> None:
    if os.name != "posix":
        return
    missing = {name: path for name, path in socket_paths.items() if not Path(path).exists()}
    if missing:
        details = ", ".join(f"{name}={path}" for name, path in sorted(missing.items()))
        raise RuntimeError(
            "Static server worker refused to auto-start a server because expected socket(s) are missing: "
            f"{details}"
        )


def _load_server_assets(
        args: Any,
        *,
        allow_auto_start: bool,
        connect_timeout: float | None,
        request_timeout: float | None,
        idle_timeout: float,
) -> dict[str, Any]:
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=True,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
        server_allow_auto_start=allow_auto_start,
        server_connect_timeout=connect_timeout,
        server_request_timeout=request_timeout,
        server_idle_timeout=idle_timeout,
        server_batch_timeout=args.server_batch_timeout,
    )
    timing_model, timing_tokenizer = None, None
    if should_load_separate_timing_model(args):
        timing_model, timing_tokenizer = load_model_with_server(
            args.model_path,
            args.train,
            args.device,
            max_batch_size=args.max_batch_size,
            use_server=True,
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            gamemode=args.gamemode,
            auto_select_gamemode_model=False,
            generation_compile=args.inference_generation_compile,
            server_allow_auto_start=allow_auto_start,
            server_connect_timeout=connect_timeout,
            server_request_timeout=request_timeout,
            server_idle_timeout=idle_timeout,
            server_batch_timeout=args.server_batch_timeout,
        )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "timing_model": timing_model,
        "timing_tokenizer": timing_tokenizer,
    }


def _ensure_server(asset: Any, timeout: float | None) -> None:
    if isinstance(asset, InferenceClient):
        asset.ensure_server(timeout=timeout)


def _shutdown_server(asset: Any) -> None:
    if isinstance(asset, InferenceClient):
        asset.shutdown_server()


def _run_request(
        *,
        raw_args: Any,
        suite_dir: Path,
        suite_id: str,
        suite_song_count: int,
        suite_repeat_count: int,
        expected_socket_paths: dict[str, str],
        expected_config_fingerprint: dict[str, Any],
        connect_timeout: float | None,
        request_timeout: float | None,
        idle_timeout: float,
        run_index: int,
        repeat_index: int,
        song_entry: dict[str, Any],
) -> dict[str, Any]:
    run_args = copy.deepcopy(raw_args)
    if run_args.seed is None:
        raise ValueError("seed must be set before static server batch runs.")
    run_args.use_server = True
    run_args.parallel = False
    run_args.audio_path = str(song_entry["audio_path"])
    run_args.beatmap_path = str(song_entry.get("beatmap_path") or "")
    if "seed" in song_entry:
        run_args.seed = int(song_entry["seed"])
    if "start_time" in song_entry:
        run_args.start_time = int(song_entry["start_time"])
    if "end_time" in song_entry:
        run_args.end_time = int(song_entry["end_time"])

    run_dir = suite_dir / str(song_entry["output_subdir"]) / f"repeat{repeat_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_args.output_path = str(run_dir)
    run_args.profile_output_path = str(run_dir / f"run{run_index:02d}.profile.json")
    compile_args(run_args, verbose=False)
    _validate_server_identity(
        run_args,
        expected_socket_paths=expected_socket_paths,
        expected_config_fingerprint=expected_config_fingerprint,
    )
    _validate_server_sockets_ready(expected_socket_paths)

    request_start = time.perf_counter()
    assets = _load_server_assets(
        run_args,
        allow_auto_start=False,
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
        idle_timeout=idle_timeout,
    )
    generation_config, beatmap_config = get_config(run_args)
    profiler = InferenceProfiler.from_args(run_args)
    profiler.set_metadata(
        suite_id=suite_id,
        run_kind="static_server_batch",
        suite_run_index=run_index,
        run_index=run_index,
        repeat_index=repeat_index,
        song_index=int(song_entry["song_index"]),
        song_id=song_entry["song_id"],
        suite_song_count=suite_song_count,
        suite_repeat_count=suite_repeat_count,
        rng_reset_policy="server_global_rng_shared_across_concurrent_requests",
        requested_seed=run_args.seed,
        server_seed_applied=False,
        token_equivalence_status="not_checked_shared_server_rng",
        server_batch_claim_scope="static_ipc_concurrent_full_song_requests",
        warmup_excluded=False,
    )

    try:
        _, result_path = generate(
            run_args,
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=assets["model"],
            tokenizer=assets["tokenizer"],
            timing_model=assets["timing_model"],
            timing_tokenizer=assets["timing_tokenizer"],
            profiler=profiler,
            logger=get_default_logger(),
            verbose=False,
        )
    finally:
        for key in ("model", "timing_model"):
            client = assets.get(key)
            if isinstance(client, InferenceClient) and client.conn:
                client.conn.close()

    request_wall_seconds = time.perf_counter() - request_start
    profile_path = Path(run_args.profile_output_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    main_summary = _summary_for_label(profile, "main_generation")
    timing_summary = _summary_for_label(profile, "timing_context")
    main_record_breakdown = first_record_breakdown(profile, "main_generation")
    profile_metadata = profile.get("metadata", {})
    result_file_sha256 = profile_metadata.get("result_file_sha256")
    result_file_size_bytes = profile_metadata.get("result_file_size_bytes")
    if result_file_sha256 is None and result_path is not None and Path(result_path).is_file():
        result_metadata = file_artifact_metadata(result_path)
        result_file_sha256 = result_metadata.get("result_file_sha256")
        result_file_size_bytes = result_metadata.get("result_file_size_bytes")
    tokens = _flatten_token_ids(profile, "main_generation")
    return {
        "run_index": run_index,
        "repeat_index": repeat_index,
        "song_index": int(song_entry["song_index"]),
        "song_id": song_entry["song_id"],
        "audio_path": str(song_entry["audio_path"]),
        "beatmap_path": str(song_entry.get("beatmap_path") or ""),
        "start_time": run_args.start_time,
        "end_time": run_args.end_time,
        "seed": run_args.seed,
        "requested_seed": run_args.seed,
        "server_seed_applied": False,
        "token_equivalence_status": "not_checked_shared_server_rng",
        "request_wall_seconds": request_wall_seconds,
        "result_path": str(result_path),
        "result_file_sha256": result_file_sha256,
        "result_file_size_bytes": result_file_size_bytes,
        "profile_path": str(profile_path),
        "sequence_count": profile_metadata.get("sequence_count"),
        "song_length_ms": profile_metadata.get("song_length_ms"),
        "main_generated_tokens": int(main_summary.get("generated_tokens", 0) or 0),
        "main_model_elapsed_seconds": float(main_summary.get("model_elapsed_seconds", 0.0) or 0.0),
        "main_wall_seconds": float(main_summary.get("wall_seconds", 0.0) or 0.0),
        "main_tokens_per_second": float(main_summary.get("tokens_per_second", 0.0) or 0.0),
        "main_first_record": main_record_breakdown["first_record"],
        "main_remaining_records": main_record_breakdown["remaining_records"],
        "timing_generated_tokens": int(timing_summary.get("generated_tokens", 0) or 0),
        "timing_model_elapsed_seconds": float(timing_summary.get("model_elapsed_seconds", 0.0) or 0.0),
        "timing_wall_seconds": float(timing_summary.get("wall_seconds", 0.0) or 0.0),
        "timing_tokens_per_second": float(timing_summary.get("tokens_per_second", 0.0) or 0.0),
        "main_token_count": len(tokens) if tokens is not None else None,
        "main_token_sha256": _token_sha256(tokens),
        "generation_batch_summary": _profile_batch_summary(profile),
    }


def _aggregate_runs(runs: list[dict[str, Any]], scheduler_wall_seconds: float) -> dict[str, Any]:
    main_tokens = sum(int(run.get("main_generated_tokens") or 0) for run in runs)
    timing_tokens = sum(int(run.get("timing_generated_tokens") or 0) for run in runs)
    request_walls = sorted(float(run.get("request_wall_seconds") or 0.0) for run in runs)
    request_wall = sum(request_walls)
    main_model_elapsed = sum(float(run.get("main_model_elapsed_seconds") or 0.0) for run in runs)
    timing_model_elapsed = sum(float(run.get("timing_model_elapsed_seconds") or 0.0) for run in runs)
    batching = _aggregate_batch_summaries(runs)
    server_batch_observed = False
    for label_summary in batching.get("by_label", {}).values():
        hist = label_summary.get("server_unique_batch_size_histogram") or label_summary.get("server_batch_size_histogram")
        if isinstance(hist, dict) and any(int(size) > 1 and int(count) > 0 for size, count in hist.items()):
            server_batch_observed = True
            break
    return {
        "runs": len(runs),
        "result_class": "static_server_batch" if server_batch_observed else "static_server_no_batch_observed",
        "server_batch_observed": server_batch_observed,
        "same_calculation": False,
        "throughput_claim_scope": "static_ipc_concurrent_full_song_requests",
        "token_equivalence_status": "not_checked_shared_server_rng",
        "main_generated_tokens": main_tokens,
        "timing_generated_tokens": timing_tokens,
        "scheduler_wall_seconds": scheduler_wall_seconds,
        "request_wall_seconds_sum": request_wall,
        "request_wall_seconds_max": max(request_walls) if request_walls else 0.0,
        "request_wall_seconds_p95": _nearest_rank_percentile(request_walls, 0.95),
        "main_model_elapsed_seconds_sum": main_model_elapsed,
        "timing_model_elapsed_seconds_sum": timing_model_elapsed,
        "main_tokens_per_scheduler_second": (
            main_tokens / scheduler_wall_seconds if scheduler_wall_seconds > 0 else 0.0
        ),
        "timing_tokens_per_scheduler_second": (
            timing_tokens / scheduler_wall_seconds if scheduler_wall_seconds > 0 else 0.0
        ),
        "main_tokens_per_request_model_second_attributed": (
            main_tokens / main_model_elapsed if main_model_elapsed > 0 else 0.0
        ),
        "timing_tokens_per_request_model_second_attributed": (
            timing_tokens / timing_model_elapsed if timing_model_elapsed > 0 else 0.0
        ),
        "batching": batching,
    }


def _nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * percentile + 0.999999) - 1))
    return sorted_values[index]


def main() -> None:
    cli_args = _parse_args()
    if cli_args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if cli_args.launch_stagger_seconds < 0:
        raise ValueError("--launch-stagger-seconds must be non-negative.")
    if cli_args.server_start_timeout_seconds <= 0:
        raise ValueError("--server-start-timeout-seconds must be positive.")
    if cli_args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be positive.")
    if cli_args.suite_timeout_seconds <= 0:
        raise ValueError("--suite-timeout-seconds must be positive.")
    if cli_args.server_idle_timeout_seconds <= 0:
        raise ValueError("--server-idle-timeout-seconds must be positive.")

    cfg = _compose_config(cli_args.config_name, cli_args.overrides)
    raw_args = OmegaConf.to_object(cfg)
    raw_args.use_server = True
    raw_args.parallel = False
    if raw_args.generate_positions:
        raise ValueError("static server batch profiling currently requires generate_positions=false.")
    if not raw_args.profile_inference:
        raise ValueError("profile_inference must be true for static server batch profiling.")
    if not raw_args.profile_record_token_ids:
        raise ValueError("profile_record_token_ids must be true for token hash reporting.")

    song_entries = _load_song_entries(cli_args.song_list)
    if len(song_entries) < 5 and not cli_args.allow_short_suite:
        raise ValueError("static server batch profiling expects at least 5 songs.")
    raw_args.audio_path = str(song_entries[0]["audio_path"])
    raw_args.beatmap_path = str(song_entries[0].get("beatmap_path") or "")
    compile_args(raw_args)
    max_workers = cli_args.max_workers or len(song_entries)
    if max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    socket_paths = _server_socket_paths(raw_args)
    if not cli_args.allow_existing_server:
        _validate_no_existing_server(socket_paths)
    server_config_fingerprint = _server_config_fingerprint(raw_args)

    suite_id = cli_args.suite_id or uuid.uuid4().hex[:12]
    suite_dir = Path(cli_args.output_root or raw_args.output_path) / f"static-server-batch-{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    setup_inference_environment(raw_args.seed)
    set_seed(raw_args.seed)
    owner_assets = _load_server_assets(
        raw_args,
        allow_auto_start=True,
        connect_timeout=cli_args.server_start_timeout_seconds,
        request_timeout=cli_args.request_timeout_seconds,
        idle_timeout=cli_args.server_idle_timeout_seconds,
    )
    try:
        _ensure_server(owner_assets["model"], cli_args.server_start_timeout_seconds)
        if owner_assets.get("timing_model") is not None:
            _ensure_server(owner_assets["timing_model"], cli_args.server_start_timeout_seconds)

        suite_items: list[tuple[int, int, dict[str, Any]]] = []
        for repeat_index in range(cli_args.repeats):
            for song_entry in song_entries:
                suite_items.append((repeat_index, len(suite_items), song_entry))

        runs: list[dict[str, Any]] = []
        scheduler_start = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = []
        try:
            for repeat_index, run_index, song_entry in suite_items:
                futures.append(
                    executor.submit(
                        _run_request,
                        raw_args=raw_args,
                        suite_dir=suite_dir,
                        suite_id=suite_id,
                        suite_song_count=len(song_entries),
                        suite_repeat_count=cli_args.repeats,
                        expected_socket_paths=socket_paths,
                        expected_config_fingerprint=server_config_fingerprint,
                        connect_timeout=cli_args.server_start_timeout_seconds,
                        request_timeout=cli_args.request_timeout_seconds,
                        idle_timeout=cli_args.server_idle_timeout_seconds,
                        run_index=run_index,
                        repeat_index=repeat_index,
                        song_entry=song_entry,
                    )
                )
                if cli_args.launch_stagger_seconds > 0:
                    time.sleep(cli_args.launch_stagger_seconds)
            try:
                for future in as_completed(futures, timeout=cli_args.suite_timeout_seconds):
                    run = future.result()
                    runs.append(run)
                    print(
                        "[static-server] run {run_index}: song={song_id}, main={tokens} tokens, "
                        "request_wall={wall:.3f}s, attributed_model_tok/s={tok_s:.3f}".format(
                            run_index=run["run_index"],
                            song_id=run["song_id"],
                            tokens=run["main_generated_tokens"],
                            wall=run["request_wall_seconds"],
                            tok_s=run["main_tokens_per_second"],
                        )
                    )
            except FuturesTimeoutError as exc:
                for future in futures:
                    future.cancel()
                raise TimeoutError(
                    f"Static server batch suite timed out after {cli_args.suite_timeout_seconds:.1f}s "
                    f"with {len(runs)} / {len(futures)} requests completed."
                ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        scheduler_wall_seconds = time.perf_counter() - scheduler_start
    finally:
        _shutdown_server(owner_assets.get("timing_model"))
        _shutdown_server(owner_assets.get("model"))

    runs.sort(key=lambda run: int(run["run_index"]))
    manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "run_kind": "static_server_batch",
        "config_name": cli_args.config_name,
        "overrides": cli_args.overrides,
        "song_list_path": cli_args.song_list,
        "song_count": len(song_entries),
        "repeats": cli_args.repeats,
        "max_workers": max_workers,
        "launch_stagger_seconds": cli_args.launch_stagger_seconds,
        "server_batch_timeout_seconds": raw_args.server_batch_timeout,
        "server_start_timeout_seconds": cli_args.server_start_timeout_seconds,
        "request_timeout_seconds": cli_args.request_timeout_seconds,
        "suite_timeout_seconds": cli_args.suite_timeout_seconds,
        "server_idle_timeout_seconds": cli_args.server_idle_timeout_seconds,
        "allow_existing_server": cli_args.allow_existing_server,
        "server_socket_paths": socket_paths,
        "server_config_fingerprint": server_config_fingerprint,
        "rng_reset_policy": "server_global_rng_shared_across_concurrent_requests",
        "same_calculation": False,
        "throughput_claim_scope": "static_ipc_concurrent_full_song_requests",
        "token_equivalence_status": "not_checked_shared_server_rng",
        "equivalence_scope": (
            "static server batching is throughput evidence only unless compared against "
            "per-song token/output baselines from the same server/RNG policy"
        ),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "runtime_environment": _runtime_environment(),
        "songs": song_entries,
        "runs": runs,
        "aggregate": _aggregate_runs(runs, scheduler_wall_seconds),
    }
    manifest_path = suite_dir / "static_server_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[static-server] manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
