from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
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
    get_default_logger,
    get_config,
    load_diff_model,
    load_model_with_server,
    setup_inference_environment,
    should_load_separate_timing_model,
)
from utils.inference_profile_metrics import first_record_breakdown  # noqa: E402
from osuT5.osuT5.inference.profiler import InferenceProfiler  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a same-process inference profiling suite with one model load."
    )
    parser.add_argument("--config-name", default="profile_salvalai", help="Hydra inference config name.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeated runs.")
    parser.add_argument(
        "--run-kind",
        default="warm_repeat",
        choices=["warm_repeat", "serial_multi_song"],
        help="Result class recorded in suite/profile metadata.",
    )
    parser.add_argument("--suite-id", default=None, help="Stable suite id. Defaults to a random short id.")
    parser.add_argument("--output-root", default=None, help="Directory for per-run outputs and suite_manifest.json.")
    parser.add_argument(
        "--song-list",
        default=None,
        help=(
            "YAML/JSON/text song list for serial_multi_song. YAML/JSON may be a list or a dict "
            "with a songs list; each item is an audio path string or a dict with audio_path, "
            "optional song_id/id, output_subdir, beatmap_path, seed, start_time, and end_time."
        ),
    )
    parser.add_argument(
        "--allow-short-suite",
        action="store_true",
        help="Allow serial_multi_song with fewer than 5 songs for harness smoke tests.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=0,
        help="Increment seed by this amount per run. Default 0 resets every run to the same seed.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides for the inference config, e.g. inference_active_prefix_decode_loop=true.",
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


def _safe_path_part(value: str, *, fallback: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_", ".") else "-" for c in value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or fallback


def _load_song_entries(song_list_path: str | None) -> list[dict[str, Any]]:
    if song_list_path is None:
        return []

    path = Path(song_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Song list does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Song list must be a file: {path}")

    if path.suffix.lower() in {".yaml", ".yml", ".json"}:
        data = OmegaConf.to_object(OmegaConf.load(path))
    else:
        data = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if isinstance(data, dict):
        if "songs" not in data:
            raise ValueError("Song list mapping must contain a 'songs' key.")
        data = data["songs"]
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Song list must contain at least one song.")

    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(data):
        if isinstance(raw_entry, str):
            entry = {"audio_path": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            raise ValueError(f"Song list item {index} must be a string or mapping.")

        audio_path = entry.get("audio_path")
        if not audio_path:
            raise ValueError(f"Song list item {index} is missing audio_path.")

        song_id = str(entry.get("song_id") or entry.get("id") or Path(str(audio_path)).stem or f"song{index:02d}")
        output_subdir = str(
            entry.get("output_subdir")
            or f"{index:02d}-{_safe_path_part(song_id, fallback=f'song{index:02d}')}"
        )
        normalized = {
            "song_index": index,
            "song_id": song_id,
            "audio_path": str(audio_path),
            "output_subdir": _safe_path_part(output_subdir, fallback=f"song{index:02d}"),
        }
        if entry.get("beatmap_path"):
            normalized["beatmap_path"] = str(entry["beatmap_path"])
        if entry.get("seed") is not None:
            normalized["seed"] = int(entry["seed"])
        if entry.get("start_time") is not None:
            normalized["start_time"] = int(entry["start_time"])
        if entry.get("end_time") is not None:
            normalized["end_time"] = int(entry["end_time"])
        entries.append(normalized)

    return entries


def _load_models(args: Any) -> dict[str, Any]:
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        lora_path=args.lora_path,
        gamemode=args.gamemode,
        auto_select_gamemode_model=args.auto_select_gamemode_model,
        generation_compile=args.inference_generation_compile,
    )

    timing_model, timing_tokenizer = None, None
    if should_load_separate_timing_model(args):
        print("Using base model for timing generation.")
        timing_model, timing_tokenizer = load_model_with_server(
            args.model_path,
            args.train,
            args.device,
            max_batch_size=args.max_batch_size,
            use_server=args.use_server,
            precision=args.precision,
            attn_implementation=args.attn_implementation,
            gamemode=args.gamemode,
            auto_select_gamemode_model=False,
            generation_compile=args.inference_generation_compile,
        )

    diff_model, diff_tokenizer, refine_model = None, None, None
    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)
        if os.path.exists(args.diff_refine_ckpt):
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion, args.device)[0]
        if args.compile:
            diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=True)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "timing_model": timing_model,
        "timing_tokenizer": timing_tokenizer,
        "diff_model": diff_model,
        "diff_tokenizer": diff_tokenizer,
        "refine_model": refine_model,
    }


def _summary_for_label(profile: dict[str, Any], label: str) -> dict[str, Any]:
    return profile.get("summary", {}).get("generation_by_label", {}).get(label, {})


def _flatten_token_ids(profile: dict[str, Any], label: str) -> list[int] | None:
    tokens: list[int] = []
    saw_tokens = False
    for record in profile.get("generation", []):
        if record.get("profile_label") != label:
            continue
        if "generated_token_ids" in record:
            value = record.get("generated_token_ids")
            if value is None:
                continue
            saw_tokens = True
            tokens.extend(int(token) for token in value)
        elif "generated_token_ids_per_sample" in record:
            value = record.get("generated_token_ids_per_sample")
            if value is None:
                continue
            saw_tokens = True
            for sample in value:
                tokens.extend(int(token) for token in sample)
    return tokens if saw_tokens else None


def _profile_batch_summary(profile: dict[str, Any]) -> dict[str, Any]:
    records = profile.get("generation", [])
    by_label: dict[str, dict[str, Any]] = {}
    for record in records:
        label = str(record.get("profile_label") or "unknown")
        label_summary = by_label.setdefault(
            label,
            {
                "records": 0,
                "modes": {},
                "batch_size_histogram": {},
                "server_batch_size_histogram": {},
                "server_batch_count": 0,
                "server_request_record_count": 0,
                "server_total_queue_wait_seconds": 0.0,
                "server_max_queue_wait_seconds": 0.0,
                "server_batching_modes": {},
                "server_elapsed_seconds_attributions": {},
                "server_batches": [],
            },
        )
        label_summary["records"] += 1
        mode = str(record.get("mode") or "unknown")
        label_summary["modes"][mode] = int(label_summary["modes"].get(mode, 0)) + 1
        batch_size = record.get("batch_size")
        if batch_size is not None:
            key = str(int(batch_size))
            label_summary["batch_size_histogram"][key] = int(
                label_summary["batch_size_histogram"].get(key, 0)
            ) + 1
        server_mode = record.get("server_batching_mode")
        if server_mode is not None:
            server_mode = str(server_mode)
            label_summary["server_batching_modes"][server_mode] = int(
                label_summary["server_batching_modes"].get(server_mode, 0)
            ) + 1
        attribution = record.get("server_elapsed_seconds_attribution")
        if attribution is not None:
            attribution = str(attribution)
            label_summary["server_elapsed_seconds_attributions"][attribution] = int(
                label_summary["server_elapsed_seconds_attributions"].get(attribution, 0)
            ) + 1
        server_sizes = record.get("server_batch_sizes")
        if isinstance(server_sizes, list) and server_sizes:
            label_summary["server_request_record_count"] += 1
            label_summary["server_batch_count"] += len(server_sizes)
            server_ids = record.get("server_batch_ids")
            request_counts = record.get("server_batch_request_counts")
            work_items = record.get("server_batch_work_items")
            for size in server_sizes:
                key = str(int(size))
                label_summary["server_batch_size_histogram"][key] = int(
                    label_summary["server_batch_size_histogram"].get(key, 0)
                ) + 1
            for index, size in enumerate(server_sizes):
                batch_id = (
                    server_ids[index]
                    if isinstance(server_ids, list) and index < len(server_ids)
                    else None
                )
                request_count = (
                    request_counts[index]
                    if isinstance(request_counts, list) and index < len(request_counts)
                    else None
                )
                work_item = (
                    work_items[index]
                    if isinstance(work_items, list) and index < len(work_items)
                    else None
                )
                label_summary["server_batches"].append({
                    "batch_id": int(batch_id) if batch_id is not None else None,
                    "batch_size": int(size),
                    "request_count": int(request_count) if request_count is not None else None,
                    "work_items": int(work_item) if work_item is not None else None,
                })
        queue_wait = float(record.get("server_total_queue_wait_seconds") or 0.0)
        label_summary["server_total_queue_wait_seconds"] += queue_wait
        label_summary["server_max_queue_wait_seconds"] = max(
            float(label_summary["server_max_queue_wait_seconds"]),
            float(record.get("server_max_queue_wait_seconds") or 0.0),
        )
    return {"by_label": by_label}


def _token_sha256(tokens: list[int] | None) -> str | None:
    if tokens is None:
        return None
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _aggregate_runs(selected: list[dict[str, Any]]) -> dict[str, Any]:
    generated_tokens = sum(int(run.get("main_generated_tokens") or 0) for run in selected)
    model_elapsed_seconds = sum(float(run.get("main_model_elapsed_seconds") or 0.0) for run in selected)
    wall_seconds = sum(float(run.get("main_wall_seconds") or 0.0) for run in selected)
    return {
        "runs": len(selected),
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": generated_tokens / model_elapsed_seconds if model_elapsed_seconds > 0 else 0.0,
        "wall_tokens_per_second": generated_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "first_records": _aggregate_run_segments(selected, "main_first_record"),
        "remaining_records": _aggregate_run_segments(selected, "main_remaining_records"),
    }


def _aggregate_run_segments(selected: list[dict[str, Any]], key: str) -> dict[str, Any]:
    segments = [
        segment
        for run in selected
        for segment in [run.get(key)]
        if isinstance(segment, dict)
    ]
    generated_tokens = sum(int(segment.get("generated_tokens") or 0) for segment in segments)
    model_elapsed_seconds = sum(float(segment.get("model_elapsed_seconds") or 0.0) for segment in segments)
    wall_seconds = sum(float(segment.get("wall_seconds") or 0.0) for segment in segments)
    records = sum(int(segment.get("records", 1) or 0) for segment in segments)
    return {
        "runs": len(segments),
        "records": records,
        "generated_tokens": generated_tokens,
        "model_elapsed_seconds": model_elapsed_seconds,
        "wall_seconds": wall_seconds,
        "tokens_per_second": generated_tokens / model_elapsed_seconds if model_elapsed_seconds > 0 else 0.0,
        "wall_tokens_per_second": generated_tokens / wall_seconds if wall_seconds > 0 else 0.0,
    }


def _aggregate(runs: list[dict[str, Any]], *, start_index: int) -> dict[str, Any]:
    return _aggregate_runs(runs[start_index:])


def _warmup_excluded_run_indices(runs: list[dict[str, Any]]) -> list[int]:
    if not any(int(run.get("repeat_index", run.get("run_index", 0))) > 0 for run in runs):
        return []
    return [
        int(run["run_index"])
        for run in runs
        if int(run.get("repeat_index", run.get("run_index", 0))) == 0
    ]


def _warmed_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if int(run.get("repeat_index", run.get("run_index", 0))) > 0
    ]


def _aggregate_by_song(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_song: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        by_song.setdefault(int(run["song_index"]), []).append(run)

    summaries = []
    for song_index, song_runs in sorted(by_song.items()):
        first = song_runs[0]
        summaries.append({
            "song_index": song_index,
            "song_id": first.get("song_id"),
            "audio_path": first.get("audio_path"),
            "runs": len(song_runs),
            "all_runs": _aggregate(song_runs, start_index=0),
            "warmed_runs": _aggregate_runs(_warmed_runs(song_runs)) if _warmed_runs(song_runs) else None,
            "token_equivalence": [
                run.get("token_equivalence_to_song_baseline") for run in song_runs
            ],
        })
    return summaries


def _aggregate_batch_summaries(runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for run in runs:
        batch_summary = run.get("generation_batch_summary")
        if not isinstance(batch_summary, dict):
            continue
        by_label = batch_summary.get("by_label")
        if not isinstance(by_label, dict):
            continue
        for label, label_summary in by_label.items():
            target = aggregate.setdefault(
                str(label),
                {
                    "records": 0,
                    "modes": {},
                    "batch_size_histogram": {},
                    "server_batch_size_histogram": {},
                    "server_batch_count": 0,
                    "server_request_record_count": 0,
                    "server_total_queue_wait_seconds": 0.0,
                    "server_max_queue_wait_seconds": 0.0,
                    "server_batching_modes": {},
                    "server_elapsed_seconds_attributions": {},
                    "server_batch_count_attributed": 0,
                    "server_unique_batch_size_histogram": {},
                    "_seen_server_batch_ids": set(),
                },
            )
            target["records"] += int(label_summary.get("records", 0) or 0)
            target["server_batch_count_attributed"] += int(label_summary.get("server_batch_count", 0) or 0)
            target["server_request_record_count"] += int(
                label_summary.get("server_request_record_count", 0) or 0
            )
            target["server_total_queue_wait_seconds"] += float(
                label_summary.get("server_total_queue_wait_seconds", 0.0) or 0.0
            )
            target["server_max_queue_wait_seconds"] = max(
                float(target["server_max_queue_wait_seconds"]),
                float(label_summary.get("server_max_queue_wait_seconds", 0.0) or 0.0),
            )
            for field in (
                "modes",
                "batch_size_histogram",
                "server_batch_size_histogram",
                "server_batching_modes",
                "server_elapsed_seconds_attributions",
            ):
                values = label_summary.get(field)
                if not isinstance(values, dict):
                    continue
                for key, value in values.items():
                    target[field][str(key)] = int(target[field].get(str(key), 0)) + int(value)
            server_batches = label_summary.get("server_batches")
            if isinstance(server_batches, list):
                for batch in server_batches:
                    if not isinstance(batch, dict):
                        continue
                    batch_id = batch.get("batch_id")
                    batch_size = batch.get("batch_size")
                    if batch_id is None or batch_size is None:
                        continue
                    seen_key = str(batch_id)
                    if seen_key in target["_seen_server_batch_ids"]:
                        continue
                    target["_seen_server_batch_ids"].add(seen_key)
                    target["server_batch_count"] += 1
                    size_key = str(int(batch_size))
                    target["server_unique_batch_size_histogram"][size_key] = int(
                        target["server_unique_batch_size_histogram"].get(size_key, 0)
                    ) + 1
            elif int(label_summary.get("server_batch_count", 0) or 0) > 0:
                target["server_batch_count"] += int(label_summary.get("server_batch_count", 0) or 0)

    for label_summary in aggregate.values():
        label_summary.pop("_seen_server_batch_ids", None)
    return {"by_label": aggregate}


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


def _write_manifest(
        suite_dir: Path,
        *,
        suite_id: str,
        config_name: str,
        overrides: list[str],
        run_kind: str,
        seed_step: int,
        song_list_path: str | None,
        songs: list[dict[str, Any]],
        runs: list[dict[str, Any]],
) -> Path:
    manifest = {
        "schema_version": 3,
        "suite_id": suite_id,
        "run_kind": run_kind,
        "config_name": config_name,
        "overrides": overrides,
        "rng_reset_policy": "accelerate.set_seed(song_seed + repeat_index * seed_step) before each generate()",
        "seed_step": seed_step,
        "song_list_path": song_list_path,
        "song_count": len(songs),
        "songs": songs,
        "warmup_excluded": _warmup_excluded_run_indices(runs),
        "first_song_cold_indices": _warmup_excluded_run_indices(runs),
        "warmed_run_indices": [int(run["run_index"]) for run in _warmed_runs(runs)],
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "runtime_environment": _runtime_environment(),
        "runs": runs,
        "aggregate": {
            "all_runs": _aggregate(runs, start_index=0),
            "warmed_runs": _aggregate_runs(_warmed_runs(runs)) if _warmed_runs(runs) else None,
            "by_song": _aggregate_by_song(runs),
            "batching": _aggregate_batch_summaries(runs),
        },
    }
    manifest_path = suite_dir / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    cli_args = _parse_args()
    if cli_args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    cfg = _compose_config(cli_args.config_name, cli_args.overrides)
    raw_args = OmegaConf.to_object(cfg)
    song_entries = _load_song_entries(cli_args.song_list)
    load_args = copy.deepcopy(raw_args)
    if cli_args.run_kind == "serial_multi_song":
        if not song_entries:
            raise ValueError("serial_multi_song requires --song-list with an explicit multi-song list.")
        if len(song_entries) < 5 and not cli_args.allow_short_suite:
            raise ValueError(
                "serial_multi_song expects at least 5 songs. Use --allow-short-suite only for harness smoke tests."
            )
        load_args.audio_path = str(song_entries[0]["audio_path"])
        load_args.beatmap_path = str(song_entries[0].get("beatmap_path") or "")
    elif song_entries:
        raise ValueError("--song-list is only valid with --run-kind serial_multi_song.")

    compile_args(load_args)

    if not load_args.profile_inference:
        raise ValueError("profile_inference must be true for suite profiling.")
    if not load_args.profile_record_token_ids:
        raise ValueError("profile_record_token_ids must be true for suite token-equivalence checks.")
    if load_args.use_server:
        raise ValueError("profile_inference_suite requires use_server=false until server reseeding is explicit.")

    suite_id = cli_args.suite_id or uuid.uuid4().hex[:12]
    suite_dir = Path(cli_args.output_root or load_args.output_path) / f"{cli_args.run_kind}-{suite_id}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    setup_inference_environment(load_args.seed)
    assets = _load_models(load_args)

    logger = get_default_logger()
    if cli_args.run_kind == "warm_repeat":
        song_entries = [{
            "song_index": 0,
            "song_id": Path(str(load_args.audio_path)).stem or "song00",
            "audio_path": str(load_args.audio_path),
            "output_subdir": "",
        }]

    baseline_tokens_by_song: dict[int, list[int] | None] = {}
    runs: list[dict[str, Any]] = []

    suite_items: list[tuple[int, int, dict[str, Any]]] = []
    for repeat_index in range(cli_args.repeats):
        for song_entry in song_entries:
            suite_items.append((repeat_index, int(song_entry["song_index"]), song_entry))

    for run_index, (repeat_index, song_index, song_entry) in enumerate(suite_items):
        run_args = copy.deepcopy(raw_args)
        if run_args.seed is None:
            raise ValueError("seed must be set before suite runs.")
        seed_base = int(song_entry.get("seed", run_args.seed))
        run_args.seed = seed_base + repeat_index * cli_args.seed_step
        run_args.audio_path = str(song_entry["audio_path"])
        run_args.beatmap_path = str(song_entry.get("beatmap_path") or "")
        if "start_time" in song_entry:
            run_args.start_time = int(song_entry["start_time"])
        if "end_time" in song_entry:
            run_args.end_time = int(song_entry["end_time"])

        if cli_args.run_kind == "warm_repeat":
            run_dir = suite_dir / f"run{run_index:02d}"
        else:
            run_dir = suite_dir / str(song_entry["output_subdir"]) / f"repeat{repeat_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_args.output_path = str(run_dir)
        run_args.profile_output_path = str(run_dir / f"run{run_index:02d}.profile.json")
        compile_args(run_args, verbose=False)
        if cli_args.run_kind == "serial_multi_song" and run_args.gamemode != load_args.gamemode:
            raise ValueError(
                "serial_multi_song loads one model for the whole suite and cannot mix gamemodes "
                f"(first compiled gamemode={load_args.gamemode}, song {song_entry['song_id']} "
                f"compiled gamemode={run_args.gamemode})."
            )

        set_seed(run_args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        profiler = InferenceProfiler.from_args(run_args)
        profiler.set_metadata(
            suite_id=suite_id,
            run_kind=cli_args.run_kind,
            suite_run_index=run_index,
            run_index=run_index,
            repeat_index=repeat_index,
            song_index=song_index,
            song_id=song_entry["song_id"],
            suite_song_count=len(song_entries),
            suite_repeat_count=cli_args.repeats,
            rng_reset_policy="accelerate.set_seed before generate",
            seed_step=cli_args.seed_step,
            warmup_excluded=repeat_index == 0 and cli_args.repeats > 1,
        )

        generation_config, beatmap_config = get_config(run_args)

        print(
            f"[suite] run {run_index + 1}/{len(suite_items)}: "
            f"song={song_entry['song_id']} repeat={repeat_index} seed={run_args.seed}"
        )
        _, result_path = generate(
            run_args,
            generation_config=generation_config,
            beatmap_config=beatmap_config,
            model=assets["model"],
            tokenizer=assets["tokenizer"],
            timing_model=assets["timing_model"],
            timing_tokenizer=assets["timing_tokenizer"],
            diff_model=assets["diff_model"],
            diff_tokenizer=assets["diff_tokenizer"],
            refine_model=assets["refine_model"],
            profiler=profiler,
            logger=logger,
        )

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
        token_sha256 = _token_sha256(tokens)
        if song_index not in baseline_tokens_by_song:
            baseline_tokens_by_song[song_index] = tokens
            token_equivalence = "baseline"
        elif cli_args.seed_step != 0:
            token_equivalence = "not_checked_seed_changed"
        elif baseline_tokens_by_song[song_index] is None or tokens is None:
            token_equivalence = "not_checked"
        else:
            token_equivalence = "PASS" if tokens == baseline_tokens_by_song[song_index] else "FAIL"

        run_record = {
            "run_index": run_index,
            "repeat_index": repeat_index,
            "song_index": song_index,
            "song_id": song_entry["song_id"],
            "audio_path": str(song_entry["audio_path"]),
            "beatmap_path": str(song_entry.get("beatmap_path") or ""),
            "start_time": run_args.start_time,
            "end_time": run_args.end_time,
            "seed": run_args.seed,
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
            "main_token_sha256": token_sha256,
            "token_equivalence_to_song_baseline": token_equivalence,
            "token_equivalence_to_run0": token_equivalence if song_index == 0 else "not_applicable",
            "generation_batch_summary": _profile_batch_summary(profile),
        }
        print(
            "[suite] run {run_index}: song={song_id}, main={tokens} tokens, model={seconds:.3f}s, "
            "tok/s={tok_s:.3f}, equivalence={equivalence}".format(
                run_index=run_index,
                song_id=song_entry["song_id"],
                tokens=run_record["main_generated_tokens"],
                seconds=run_record["main_model_elapsed_seconds"],
                tok_s=run_record["main_tokens_per_second"],
                equivalence=token_equivalence,
            )
        )
        runs.append(run_record)

    manifest_path = _write_manifest(
        suite_dir,
        suite_id=suite_id,
        config_name=cli_args.config_name,
        overrides=cli_args.overrides,
        run_kind=cli_args.run_kind,
        seed_step=cli_args.seed_step,
        song_list_path=cli_args.song_list,
        songs=song_entries,
        runs=runs,
    )
    print(f"[suite] manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
