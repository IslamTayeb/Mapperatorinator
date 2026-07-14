"""Run a five-song confirmation suite with one persistent model/runtime."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from osuT5.osuT5.inference.profiler import InferenceProfiler
from utils.run_approximate_weight_only import _initialize_with_evidence, _load_args
from utils.run_final_confirmation import (
    ConfirmationRuntime,
    ConfirmationState,
    _candidate_decode_context,
    _sha256_file,
)


SCHEMA_VERSION = "mapperatorinator.serial-final-confirmation.v1"


def _songs(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("song manifest schema_version must be 1")
    songs = payload.get("songs")
    if not isinstance(songs, list) or len(songs) != 5:
        raise ValueError("serial confirmation requires exactly five songs")
    parsed = []
    names = set()
    for item in songs:
        if not isinstance(item, dict):
            raise ValueError("song manifest entries must be objects")
        name, audio = item.get("name"), item.get("audio_path")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("song names must be non-empty and unique")
        path_value = Path(str(audio)).expanduser().resolve()
        if not path_value.is_file():
            raise ValueError(f"song audio is missing: {path_value}")
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or _sha256_file(path_value) != expected_sha:
            raise ValueError(f"song audio hash mismatch: {name}")
        names.add(name)
        parsed.append({"name": name, "audio_path": str(path_value), "sha256": expected_sha})
    return parsed


def run(
    config_name: str,
    overrides: list[str],
    *,
    song_manifest: Path,
    output_root: Path,
    evidence_path: Path,
    candidate: bool,
    initialization_path: Path | None,
) -> None:
    if candidate != (initialization_path is not None):
        raise ValueError("candidate suite requires exactly one initialization path")
    import torch
    import inference

    songs = _songs(song_manifest)
    output_root.mkdir(parents=True, exist_ok=False)
    process_started = time.time_ns()
    first_args = _load_args(
        config_name,
        [*overrides, f"audio_path={songs[0]['audio_path']}", f"output_path={output_root / songs[0]['name']}"],
    )
    inference.compile_args(first_args, verbose=False)
    inference.setup_inference_environment(first_args.seed)
    model_load_started = time.perf_counter()
    binding, tokenizer = inference.load_model_with_engine(
        first_args.model_path,
        first_args.train,
        first_args.device,
        max_batch_size=first_args.max_batch_size,
        use_server=first_args.use_server,
        precision=first_args.precision,
        attn_implementation=first_args.attn_implementation,
        lora_path=first_args.lora_path,
        gamemode=first_args.gamemode,
        auto_select_gamemode_model=first_args.auto_select_gamemode_model,
        inference_engine=first_args.inference_engine,
    )
    separate_timing_model = inference.should_load_separate_timing_model(first_args)
    timing_binding = None
    timing_tokenizer = None
    if separate_timing_model:
        timing_binding, timing_tokenizer = inference.load_model_with_engine(
            first_args.model_path,
            first_args.train,
            first_args.device,
            max_batch_size=first_args.max_batch_size,
            use_server=first_args.use_server,
            precision=first_args.precision,
            attn_implementation=first_args.attn_implementation,
            gamemode=first_args.gamemode,
            auto_select_gamemode_model=False,
            inference_engine=first_args.inference_engine,
        )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    if not isinstance(binding, InferenceEngineBinding):
        raise TypeError("serial confirmation requires an optimized engine binding")
    if timing_binding is not None and not isinstance(timing_binding, InferenceEngineBinding):
        raise TypeError("serial confirmation requires an optimized timing engine binding")
    initialization = None
    if candidate:
        initializer = getattr(binding.runtime, "initialize_approximate_weight_only", None)
        if initializer is None:
            raise RuntimeError("candidate runtime lacks approximate-weight initialization")
        initialization = _initialize_with_evidence(initializer, binding.raw_model)
        assert initialization_path is not None
        initialization_path.parent.mkdir(parents=True, exist_ok=True)
        initialization_path.write_text(
            json.dumps(initialization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    state = ConfirmationState()
    runtime = ConfirmationRuntime(
        binding.runtime,
        mode="natural",
        manifest=None,
        state=state,
    )
    binding = InferenceEngineBinding(binding.raw_model, runtime)
    timing_runtime = None
    if timing_binding is not None:
        timing_runtime = ConfirmationRuntime(
            timing_binding.runtime,
            mode="natural",
            manifest=None,
            state=state,
        )
        timing_binding = InferenceEngineBinding(timing_binding.raw_model, timing_runtime)

    runs: list[dict[str, Any]] = []
    for index, song in enumerate(songs):
        song_output = output_root / song["name"]
        song_output.mkdir(parents=True, exist_ok=False)
        args = _load_args(
            config_name,
            [*overrides, f"audio_path={song['audio_path']}", f"output_path={song_output}"],
        )
        profiler = InferenceProfiler.from_args(args)
        with profiler.stage("compile_args"):
            inference.compile_args(args, verbose=False)
        with profiler.stage("setup_inference_environment"):
            inference.setup_inference_environment(args.seed)
        if (
            args.model_path != first_args.model_path
            or args.precision != first_args.precision
            or args.inference_engine != first_args.inference_engine
            or inference.should_load_separate_timing_model(args) != separate_timing_model
        ):
            raise RuntimeError("serial confirmation songs changed model topology")
        with profiler.stage("build_generation_config"):
            generation_config, beatmap_config = inference.get_config(args)
        gc.collect()
        torch.cuda.synchronize()
        allocated_before = int(torch.cuda.memory_allocated())
        reserved_before = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        record_start = len(runtime.records)
        with _candidate_decode_context(candidate):
            _, result_path = inference.generate(
                args,
                generation_config=generation_config,
                beatmap_config=beatmap_config,
                model=binding,
                tokenizer=tokenizer,
                timing_model=timing_binding,
                timing_tokenizer=timing_tokenizer,
                profiler=profiler,
                verbose=False,
            )
        torch.cuda.synchronize()
        record_end = len(runtime.records)
        gc.collect()
        torch.cuda.synchronize()
        result_path = Path(result_path).resolve()
        profile_path = result_path.with_name(result_path.name + ".profile.json")
        if not profile_path.is_file():
            raise RuntimeError(f"serial song profile is missing: {profile_path}")
        song_records = runtime.records[record_start:record_end]
        logical = {
            label: sum(
                int(record["logical_steps"])
                for record in song_records
                if record["profile_label"] == label
            )
            for label in ("timing_context", "main_generation")
        }
        if min(logical.values()) <= 0:
            raise RuntimeError(f"serial song {song['name']} lacks timing/main work")
        runs.append(
            {
                "index": index,
                "name": song["name"],
                "audio_path": song["audio_path"],
                "audio_sha256": song["sha256"],
                "profile_path": str(profile_path),
                "profile_sha256": _sha256_file(profile_path),
                "result_path": str(result_path),
                "result_sha256": _sha256_file(result_path),
                "logical_steps": logical,
                "cuda_memory": {
                    "allocated_bytes_before": allocated_before,
                    "allocated_bytes_after": int(torch.cuda.memory_allocated()),
                    "allocated_bytes_delta": int(torch.cuda.memory_allocated()) - allocated_before,
                    "reserved_bytes_before": reserved_before,
                    "reserved_bytes_after": int(torch.cuda.memory_reserved()),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                },
            }
        )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "run_uuid": str(uuid.uuid4()),
        "pid": __import__("os").getpid(),
        "process_started_time_ns": process_started,
        "process_finished_time_ns": time.time_ns(),
        "candidate": candidate,
        "startup": {
            "model_load_seconds": model_load_seconds,
            "separate_timing_model": separate_timing_model,
            "loaded_runtime_count": 2 if timing_runtime is not None else 1,
            "candidate_initialization_seconds": (
                float(initialization["initialization_wall_seconds"])
                if initialization is not None
                else 0.0
            ),
        },
        "song_manifest_path": str(song_manifest.resolve()),
        "song_manifest_sha256": _sha256_file(song_manifest),
        "initialization": initialization,
        "runs": runs,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--song-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--initialization-path", type=Path)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        song_manifest=parsed.song_manifest,
        output_root=parsed.output_root,
        evidence_path=parsed.evidence_path,
        candidate=parsed.candidate,
        initialization_path=parsed.initialization_path,
    )


if __name__ == "__main__":
    main()
