from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = 1
RUN_ORDER = (
    ("baseline_first", "baseline"),
    ("candidate_second", "overlap"),
    ("candidate_first", "overlap"),
    ("baseline_second", "baseline"),
)
PROFILE_LABELS = ("timing_context", "main_generation")
MIN_COLD_WALL_SAVING_SECONDS = 0.5
CONTRACT_METADATA_KEYS = (
    "model_path",
    "audio_path",
    "seed",
    "precision",
    "attn_implementation",
    "inference_engine",
    "use_server",
    "parallel",
    "temperature",
    "timing_temperature",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "cfg_scale",
    "in_context",
    "output_type",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _assert_scout_args(args) -> None:
    required = {
        "inference_engine": "optimized",
        "device": "cuda",
        "attn_implementation": "sdpa",
        "use_server": False,
        "parallel": False,
        "profile_inference": True,
        "profile_pass_kind": "untraced_control",
        "generate_positions": False,
    }
    mismatches = {
        key: {"actual": getattr(args, key), "required": expected}
        for key, expected in required.items()
        if getattr(args, key) != expected
    }
    if mismatches:
        raise ValueError(
            "audio/model overlap scout requires the untraced optimized runtime: "
            f"{mismatches}"
        )


def _record_stage(profiler, name: str, function, *args, **kwargs):
    with profiler.stage(name):
        return function(*args, **kwargs)


def _profile_signature(profile: dict[str, Any]) -> dict[str, Any]:
    generation = profile.get("generation")
    if not isinstance(generation, list):
        raise ValueError("inference profile is missing generation records")
    labels: dict[str, Any] = {}
    for label in PROFILE_LABELS:
        records = [
            record
            for record in generation
            if isinstance(record, dict) and record.get("profile_label") == label
        ]
        if not records:
            raise ValueError(f"inference profile is missing {label} records")
        token_material = []
        stopping_material = []
        for index, record in enumerate(records):
            groups = record.get("generated_token_ids_per_sample")
            if groups is None:
                tokens = record.get("generated_token_ids")
                if not isinstance(tokens, list):
                    raise ValueError(f"{label} record {index} has no token IDs")
                groups = [tokens]
            if not isinstance(groups, list) or any(
                not isinstance(group, list) for group in groups
            ):
                raise ValueError(f"{label} record {index} has invalid token groups")
            parsed_groups = [
                [int(token) for token in group]
                for group in groups
            ]
            generated = int(record.get("generated_tokens", 0) or 0)
            counts = [len(group) for group in parsed_groups]
            if generated != sum(counts):
                raise ValueError(
                    f"{label} record {index} token count is internally inconsistent"
                )
            key = [
                record.get("context_type"),
                record.get("mode"),
                record.get(
                    "sequence_index",
                    record.get("batch_start_index", index),
                ),
            ]
            token_material.append(parsed_groups)
            stopping_material.append(
                {
                    "record_key": key,
                    "generated_tokens": generated,
                    "counts": counts,
                    "final_tokens": [group[-1] if group else None for group in parsed_groups],
                }
            )
        labels[label] = {
            "token_stream_sha256": _canonical_sha256(token_material),
            "stopping_sha256": _canonical_sha256(stopping_material),
            "generated_tokens": sum(
                item["generated_tokens"] for item in stopping_material
            ),
            "records": len(records),
        }
    return labels


def _manifest_from_success(
    *,
    mode: str,
    result_path: Path,
    profile_path: Path,
    inner_wall_seconds: float,
    audio_task,
    prepared_audio,
) -> dict[str, Any]:
    profile = _load_json(profile_path)
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("inference profile is missing metadata")
    audio_metadata = {
        key: metadata.get(key)
        for key in (
            "audio_array_sha256",
            "audio_array_dtype",
            "audio_array_shape",
            "audio_array_nbytes",
        )
    }
    if any(value is None for value in audio_metadata.values()):
        raise ValueError("inference profile is missing decoded-audio metadata")
    worker = None
    if prepared_audio is not None:
        if prepared_audio.metadata != audio_metadata:
            raise RuntimeError(
                "prepared audio changed between worker completion and generation"
            )
        worker = {
            **prepared_audio.metadata,
            "worker_wall_seconds": prepared_audio.worker_wall_seconds,
            "worker_started_at_perf_counter_seconds": (
                prepared_audio.worker_started_at_perf_counter_seconds
            ),
            "worker_finished_at_perf_counter_seconds": (
                prepared_audio.worker_finished_at_perf_counter_seconds
            ),
            "worker_thread_name": prepared_audio.worker_thread_name,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "performance_conclusion_allowed": True,
        "mode": mode,
        "inner_request_wall_seconds": inner_wall_seconds,
        "audio": audio_metadata,
        "audio_worker": worker,
        "audio_task_closed": audio_task is None or audio_task.closed,
        "result_path": str(result_path),
        "result_size_bytes": result_path.stat().st_size,
        "result_sha256": _sha256_file(result_path),
        "profile_path": str(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "profile_contract": {
            key: metadata.get(key) for key in CONTRACT_METADATA_KEYS
        },
        "generation_signature": _profile_signature(profile),
        "profile_stage_wall_seconds": profile.get("summary", {}).get(
            "stage_wall_seconds",
            {},
        ),
    }


def run_single(
    *,
    mode: str,
    config_name: str,
    overrides: list[str],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if mode not in {"baseline", "overlap"}:
        raise ValueError("single-run mode must be baseline or overlap")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    complete_started = time.perf_counter()
    audio_task = None
    prepared_audio = None
    try:
        from inference import (
            compile_args,
            generate,
            get_config,
            load_model_with_engine,
            setup_inference_environment,
            should_load_separate_timing_model,
        )
        from osuT5.osuT5.inference import Preprocessor
        from osuT5.osuT5.inference.profiler import InferenceProfiler

        explicit_overrides = [
            *overrides,
            f"output_path={output_dir}",
            "profile_inference=true",
            "profile_detail_ranges=false",
            "profile_cuda_capture=false",
            "profile_pass_kind=untraced_control",
        ]
        args = _load_args(config_name, explicit_overrides)
        profiler = InferenceProfiler.from_args(args)
        _record_stage(profiler, "compile_args", compile_args, args, False)
        _assert_scout_args(args)
        _record_stage(
            profiler,
            "setup_inference_environment",
            setup_inference_environment,
            args.seed,
        )

        if mode == "overlap":
            from osuT5.osuT5.inference.optimized.audio_model_overlap_scout import (
                AudioPreparationTask,
            )

            preprocessor = _record_stage(
                profiler,
                "prepare_audio_loader",
                Preprocessor,
                args,
                args.parallel,
            )
            with profiler.stage("submit_audio_load"):
                audio_task = AudioPreparationTask(
                    preprocessor.load,
                    args.audio_path,
                )

        model, tokenizer = _record_stage(
            profiler,
            "load_main_model",
            load_model_with_engine,
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
            inference_engine=args.inference_engine,
        )

        timing_model, timing_tokenizer = None, None
        if should_load_separate_timing_model(args):
            timing_model, timing_tokenizer = _record_stage(
                profiler,
                "load_timing_model",
                load_model_with_engine,
                args.model_path,
                args.train,
                args.device,
                max_batch_size=args.max_batch_size,
                use_server=args.use_server,
                precision=args.precision,
                attn_implementation=args.attn_implementation,
                gamemode=args.gamemode,
                auto_select_gamemode_model=False,
                inference_engine=args.inference_engine,
            )

        if mode == "overlap":
            from osuT5.osuT5.inference.optimized.audio_model_overlap_scout import (
                preload_accepted_native_extensions,
            )

            _record_stage(
                profiler,
                "preload_accepted_native_extensions",
                preload_accepted_native_extensions,
            )

        generation_config, beatmap_config = _record_stage(
            profiler,
            "build_generation_config",
            get_config,
            args,
        )

        if audio_task is not None:
            with profiler.stage("await_audio_load"):
                prepared_audio = audio_task.result()
            audio_task.close()

        _, result_path = generate(
            args,
            generation_config=generation_config,
            beatmap_path=args.beatmap_path,
            beatmap_config=beatmap_config,
            model=model,
            tokenizer=tokenizer,
            timing_model=timing_model,
            timing_tokenizer=timing_tokenizer,
            profiler=profiler,
            preloaded_audio=(
                None if prepared_audio is None else prepared_audio.samples
            ),
            verbose=False,
        )
        result_path = Path(result_path)
        profile_path = profiler.default_output_path(result_path)
        manifest = _manifest_from_success(
            mode=mode,
            result_path=result_path,
            profile_path=profile_path,
            inner_wall_seconds=time.perf_counter() - complete_started,
            audio_task=audio_task,
            prepared_audio=prepared_audio,
        )
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        if audio_task is not None:
            audio_task.close()
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "setup_failure",
            "performance_conclusion_allowed": False,
            "mode": mode,
            "inner_request_wall_seconds": time.perf_counter() - complete_started,
            "audio_task_closed": audio_task is None or audio_task.closed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(manifest_path, failure)
        raise


def _run_child(
    *,
    python: Path,
    run_name: str,
    mode: str,
    config_name: str,
    overrides: list[str],
    run_root: Path,
) -> dict[str, Any]:
    run_dir = run_root / run_name
    manifest = run_dir / "manifest.json"
    output_dir = run_dir / "output"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "single",
        "--mode",
        mode,
        "--config-name",
        config_name,
        "--output-dir",
        str(output_dir),
        "--manifest-path",
        str(manifest),
        *overrides,
    ]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record = {
        "run_name": run_name,
        "mode": mode,
        "command": command,
        "exit_code": completed.returncode,
        "cold_process_wall_seconds": time.perf_counter() - started,
        "manifest_path": str(manifest),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    _write_json(run_dir / "process.json", record)
    return record


def _all_equal(values: Sequence[Any]) -> bool:
    return bool(values) and all(value == values[0] for value in values[1:])


def analyze_reciprocal(
    records: Sequence[dict[str, Any]],
    *,
    minimum_saving_seconds: float = MIN_COLD_WALL_SAVING_SECONDS,
) -> dict[str, Any]:
    if len(records) != len(RUN_ORDER):
        raise ValueError(f"reciprocal analysis requires {len(RUN_ORDER)} runs")
    expected = list(RUN_ORDER)
    actual = [(record.get("run_name"), record.get("mode")) for record in records]
    if actual != expected:
        raise ValueError(f"reciprocal run order changed: expected {expected}, got {actual}")
    failed_processes = [
        record["run_name"]
        for record in records
        if int(record.get("exit_code", -1)) != 0
    ]
    manifests = []
    for record in records:
        path = Path(record["manifest_path"])
        if not path.is_file():
            failed_processes.append(record["run_name"])
            continue
        manifests.append(_load_json(path))
    setup_pass = (
        not failed_processes
        and len(manifests) == len(records)
        and all(manifest.get("status") == "success" for manifest in manifests)
        and all(manifest.get("audio_task_closed") is True for manifest in manifests)
    )
    if not setup_pass:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "setup_failure",
            "performance_conclusion_allowed": False,
            "promotion_pass": False,
            "failed_processes": sorted(set(failed_processes)),
        }

    audio_hashes = [manifest["audio"]["audio_array_sha256"] for manifest in manifests]
    contracts = [manifest["profile_contract"] for manifest in manifests]
    results = [manifest["result_sha256"] for manifest in manifests]
    token_hashes = {
        label: [
            manifest["generation_signature"][label]["token_stream_sha256"]
            for manifest in manifests
        ]
        for label in PROFILE_LABELS
    }
    stopping_hashes = {
        label: [
            manifest["generation_signature"][label]["stopping_sha256"]
            for manifest in manifests
        ]
        for label in PROFILE_LABELS
    }
    exactness = {
        "audio_array_hash_pass": _all_equal(audio_hashes),
        "profile_contract_pass": _all_equal(contracts),
        "result_osu_hash_pass": _all_equal(results),
        "token_stream_pass": all(_all_equal(values) for values in token_hashes.values()),
        "stopping_pass": all(_all_equal(values) for values in stopping_hashes.values()),
    }
    exactness_pass = all(exactness.values())

    by_name = {record["run_name"]: record for record in records}
    savings = [
        float(by_name["baseline_first"]["cold_process_wall_seconds"])
        - float(by_name["candidate_second"]["cold_process_wall_seconds"]),
        float(by_name["baseline_second"]["cold_process_wall_seconds"])
        - float(by_name["candidate_first"]["cold_process_wall_seconds"]),
    ]
    mean_saving = sum(savings) / len(savings)
    performance = {
        "minimum_mean_cold_wall_saving_seconds": minimum_saving_seconds,
        "paired_cold_wall_savings_seconds": savings,
        "mean_cold_wall_saving_seconds": mean_saving,
        "no_reciprocal_order_regression": all(value >= 0 for value in savings),
        "threshold_pass": mean_saving >= minimum_saving_seconds,
    }
    performance_pass = (
        performance["no_reciprocal_order_regression"]
        and performance["threshold_pass"]
    )
    promotion_pass = exactness_pass and performance_pass
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if promotion_pass else "stop",
        "performance_conclusion_allowed": True,
        "promotion_pass": promotion_pass,
        "exactness_pass": exactness_pass,
        "exactness": exactness,
        "performance_pass": performance_pass,
        "performance": performance,
        "runs": {
            record["run_name"]: {
                "mode": record["mode"],
                "cold_process_wall_seconds": record["cold_process_wall_seconds"],
                "inner_request_wall_seconds": manifests[index][
                    "inner_request_wall_seconds"
                ],
                "profile_stage_wall_seconds": manifests[index][
                    "profile_stage_wall_seconds"
                ],
                "audio_worker": manifests[index]["audio_worker"],
            }
            for index, record in enumerate(records)
        },
    }


def _analysis_text(report: dict[str, Any]) -> str:
    if not report.get("performance_conclusion_allowed"):
        return (
            "audio_model_overlap_status=SETUP_FAILURE\n"
            "performance_conclusion_allowed=false\n"
            f"failed_processes={report.get('failed_processes', [])}\n"
        )
    performance = report["performance"]
    return "\n".join(
        [
            f"audio_model_overlap_status={report['status'].upper()}",
            f"promotion_pass={str(report['promotion_pass']).lower()}",
            f"exactness_pass={str(report['exactness_pass']).lower()}",
            f"performance_pass={str(report['performance_pass']).lower()}",
            "paired_cold_wall_savings_seconds="
            + ",".join(
                f"{value:.6f}"
                for value in performance["paired_cold_wall_savings_seconds"]
            ),
            "mean_cold_wall_saving_seconds="
            f"{performance['mean_cold_wall_saving_seconds']:.6f}",
            "minimum_mean_cold_wall_saving_seconds="
            f"{performance['minimum_mean_cold_wall_saving_seconds']:.6f}",
            "",
        ]
    )


def run_reciprocal(
    *,
    python: Path,
    config_name: str,
    overrides: list[str],
    run_root: Path,
    minimum_saving_seconds: float,
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"reciprocal run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    records = []
    for run_name, mode in RUN_ORDER:
        record = _run_child(
            python=python,
            run_name=run_name,
            mode=mode,
            config_name=config_name,
            overrides=overrides,
            run_root=run_root,
        )
        records.append(record)
        if record["exit_code"] != 0:
            break
    report = analyze_reciprocal(
        records,
        minimum_saving_seconds=minimum_saving_seconds,
    ) if len(records) == len(RUN_ORDER) else {
        "schema_version": SCHEMA_VERSION,
        "status": "setup_failure",
        "performance_conclusion_allowed": False,
        "promotion_pass": False,
        "failed_processes": [records[-1]["run_name"]],
    }
    _write_json(run_root / "analysis.json", report)
    (run_root / "analysis.txt").write_text(
        _analysis_text(report),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile opt-in audio/model-load overlap in fresh processes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--mode", choices=("baseline", "overlap"), required=True)
    single.add_argument("--config-name", default="profile_salvalai")
    single.add_argument("--output-dir", type=Path, required=True)
    single.add_argument("--manifest-path", type=Path, required=True)
    single.add_argument("overrides", nargs="*")

    reciprocal = subparsers.add_parser("reciprocal")
    reciprocal.add_argument("--python", type=Path, default=Path(sys.executable))
    reciprocal.add_argument("--config-name", default="profile_salvalai")
    reciprocal.add_argument("--run-root", type=Path, required=True)
    reciprocal.add_argument(
        "--minimum-saving-seconds",
        type=float,
        default=MIN_COLD_WALL_SAVING_SECONDS,
    )
    reciprocal.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    if args.command == "single":
        run_single(
            mode=args.mode,
            config_name=args.config_name,
            overrides=args.overrides,
            output_dir=args.output_dir,
            manifest_path=args.manifest_path,
        )
        return
    if args.minimum_saving_seconds < 0:
        parser.error("--minimum-saving-seconds must be non-negative")
    if not args.python.is_file():
        parser.error(f"--python is not a file: {args.python}")
    report = run_reciprocal(
        python=args.python,
        config_name=args.config_name,
        overrides=args.overrides,
        run_root=args.run_root,
        minimum_saving_seconds=args.minimum_saving_seconds,
    )
    raise SystemExit(0 if report.get("promotion_pass") else 1)


if __name__ == "__main__":
    main()
