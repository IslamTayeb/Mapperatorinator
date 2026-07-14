from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_ORDER = (
    "cached_first",
    "direct_first",
    "direct_second",
    "cached_second",
)
ANALYZER_ROLES = {
    "baseline_first": "cached_first",
    "candidate_first": "direct_first",
    "candidate_second": "direct_second",
    "baseline_second": "cached_second",
}
EXPECTED_RETAINED_COMPOSITION = (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
)


def _profile_path(output: Path) -> Path:
    profiles = sorted(output.rglob("*.profile.json"))
    if len(profiles) != 1:
        raise RuntimeError(f"expected one profile under {output}, got {profiles}")
    return profiles[0]


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {name} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _validate_extension_evidence(
    records: dict[str, Any],
    manifest_extensions: dict[str, Any],
    *,
    mode: str,
) -> float:
    expected_mode = "direct" if mode == "direct" else "load_inline"
    if set(records) != set(manifest_extensions):
        raise RuntimeError(
            f"{mode} extension set mismatch: "
            f"expected {sorted(manifest_extensions)}, got {sorted(records)}"
        )
    for name, expected in manifest_extensions.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"{mode} extension record {name} must be an object")
        for key in ("source_sha256", "library_sha256", "functions"):
            if record.get(key) != expected.get(key):
                raise RuntimeError(f"{mode} extension {name} differs in {key}")
        if record.get("mode") != expected_mode:
            raise RuntimeError(
                f"{mode} extension {name} used {record.get('mode')}, "
                f"expected {expected_mode}"
            )
        load_seconds = record.get("load_seconds")
        if (
            isinstance(load_seconds, bool)
            or not isinstance(load_seconds, (int, float))
            or not math.isfinite(float(load_seconds))
            or float(load_seconds) <= 0.0
        ):
            raise RuntimeError(
                f"{mode} extension {name} lacks a positive load duration"
            )
    return sum(float(record["load_seconds"]) for record in records.values())


def _validate_retained_initialization(
    payload: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    if payload.get("result_class") != "documented-drift":
        raise RuntimeError(f"{role} changed the retained runtime result class")
    if payload.get("exactness_claim") is not False:
        raise RuntimeError(f"{role} must not claim exactness against accepted FP32")
    if payload.get("combined_runtime") != EXPECTED_RETAINED_COMPOSITION:
        raise RuntimeError(f"{role} did not execute the retained K1+INT8 composition")
    shared = payload.get("shared_rope")
    if not isinstance(shared, dict):
        raise RuntimeError(f"{role} shared-RoPE evidence is missing")
    expected = {
        "version": "shared-decoder-rope-v1",
        "scope": "main-model-only",
        "incremental_exactness_claim": True,
        "original_decoder_forward_required": True,
    }
    failures = {
        key: {"expected": value, "actual": shared.get(key)}
        for key, value in expected.items()
        if shared.get(key) != value
    }
    stats = shared.get("stats")
    if not isinstance(stats, dict):
        failures["stats"] = {"expected": "object", "actual": type(stats).__name__}
    else:
        for key in (
            "forwards",
            "computes",
            "reuses",
            "expected_computes",
            "expected_reuses",
        ):
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                failures[f"stats.{key}"] = {
                    "expected": "positive integer",
                    "actual": value,
                }
        if stats.get("computes") != stats.get("expected_computes"):
            failures["stats.computes"] = {
                "expected": stats.get("expected_computes"),
                "actual": stats.get("computes"),
            }
        if stats.get("reuses") != stats.get("expected_reuses"):
            failures["stats.reuses"] = {
                "expected": stats.get("expected_reuses"),
                "actual": stats.get("reuses"),
            }
    if failures:
        raise RuntimeError(f"{role} shared-RoPE evidence is invalid: {failures}")
    overlay = payload.get("int8_mlp_overlay")
    if not isinstance(overlay, dict):
        raise RuntimeError(f"{role} INT8 MLP initialization evidence is missing")
    expected_overlay = {
        "version": "per-row-symmetric-int8-mlp-v1",
        "result_class": "documented-drift",
        "exactness_claim": False,
        "scope": "main-model-decoder-mlp-only",
        "composition_control": "k4-split-kv-mixed-weight-shared-rope-v1",
        "replaces_effective_regions": ["mlp_fc1", "mlp_fc2"],
        "retained_unused_fp16_regions": ["mlp_fc1", "mlp_fc2"],
        "fp32_activations_norm_bias_reductions_residual_outputs": True,
        "quantization": "symmetric-per-output-row",
        "dispatch_counter": "int8_weight_mlp_tail",
    }
    overlay_failures = {
        key: {"expected": value, "actual": overlay.get(key)}
        for key, value in expected_overlay.items()
        if overlay.get(key) != value
    }
    for key in ("extension_init_seconds", "weight_pack_seconds", "packed_weight_bytes"):
        value = overlay.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            overlay_failures[key] = {
                "expected": "finite non-negative",
                "actual": value,
            }
    if overlay_failures:
        raise RuntimeError(f"{role} INT8 MLP evidence is invalid: {overlay_failures}")
    return {
        "combined_runtime": EXPECTED_RETAINED_COMPOSITION,
        "result_class": "documented-drift",
        "exactness_claim": False,
        "shared_rope": shared,
        "int8_mlp_overlay": expected_overlay,
    }


def _tree_size_bytes(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"storage root is missing: {root}")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"storage root contains an unsupported symlink: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def _validate_build_result(
    path: Path,
    *,
    expected_sha256: str,
    expected_commit: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    from osuT5.osuT5.inference.optimized.kernels.native_extension import (
        _sha256_file,
    )

    path = path.resolve()
    if _sha256_file(path) != expected_sha256:
        raise RuntimeError("native extension build-result hash mismatch")
    payload = _load_json(path, name="native extension build result")
    if payload.get("source_commit") != expected_commit:
        raise RuntimeError("native extension build result commit mismatch")
    build_seconds = payload.get("build_seconds")
    if (
        isinstance(build_seconds, bool)
        or not isinstance(build_seconds, (int, float))
        or not math.isfinite(float(build_seconds))
        or float(build_seconds) <= 0.0
    ):
        raise RuntimeError("native extension build result lacks build_seconds")
    if payload.get("packaged") != manifest:
        raise RuntimeError("native extension build result packaged manifest mismatch")
    records = payload.get("extensions")
    manifest_extensions = manifest.get("extensions")
    if not isinstance(records, dict) or not isinstance(manifest_extensions, dict):
        raise RuntimeError("native extension build result lacks extension records")
    if set(records) != set(manifest_extensions):
        raise RuntimeError("native extension build result extension set mismatch")
    for name, expected in manifest_extensions.items():
        record = records.get(name)
        if not isinstance(record, dict) or record.get("mode") != "load_inline":
            raise RuntimeError(
                f"native extension build result {name} is not a JIT build"
            )
        for key in ("source_sha256", "library_sha256", "functions"):
            if record.get(key) != expected.get(key):
                raise RuntimeError(
                    f"native extension build result {name} differs in {key}"
                )
        load_seconds = record.get("load_seconds")
        if (
            isinstance(load_seconds, bool)
            or not isinstance(load_seconds, (int, float))
            or not math.isfinite(float(load_seconds))
            or float(load_seconds) <= 0.0
        ):
            raise RuntimeError(
                f"native extension build result {name} lacks build/load duration"
            )
    return payload


def evaluate_gate(
    analysis: dict[str, Any],
    process_walls: dict[str, float],
    *,
    minimum_cold_saving_seconds: float,
    build_result: dict[str, Any] | None = None,
    extension_load_seconds: dict[str, float] | None = None,
    storage_bytes: dict[str, int] | None = None,
) -> dict[str, Any]:
    if set(process_walls) != set(RUN_ORDER):
        raise ValueError("process wall roles are incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in process_walls.values()
    ):
        raise ValueError("process walls must be positive finite numbers")
    cached = [process_walls["cached_first"], process_walls["cached_second"]]
    direct = [process_walls["direct_first"], process_walls["direct_second"]]
    cached_median = statistics.median(cached)
    direct_median = statistics.median(direct)
    cold_saving = cached_median - direct_median
    metrics = analysis.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("reciprocal analysis metrics are missing")
    warm = metrics.get("complete_request_wall_seconds")
    cold_profile = metrics.get("cold_process_outer_wall_seconds")
    if not isinstance(warm, dict) or not isinstance(cold_profile, dict):
        raise ValueError("reciprocal warm/cold metrics are missing")
    warm_delta = float(warm["candidate_minus_baseline"])
    cold_profile_saving = -float(cold_profile["candidate_minus_baseline"])
    profile_cold_walls = cold_profile.get("run_values")
    if not isinstance(profile_cold_walls, dict) or set(profile_cold_walls) != set(
        RUN_ORDER
    ):
        raise ValueError("profile cold-process walls are incomplete")
    wall_reconciliation = {
        role: {
            "external_process_wall_seconds": float(process_walls[role]),
            "profile_cold_outer_wall_seconds": float(profile_cold_walls[role]),
            "python_startup_and_unprofiled_seconds": (
                float(process_walls[role]) - float(profile_cold_walls[role])
            ),
        }
        for role in RUN_ORDER
    }
    invalid_reconciliation = {
        role: values
        for role, values in wall_reconciliation.items()
        if values["python_startup_and_unprofiled_seconds"] < -0.05
    }
    if invalid_reconciliation:
        raise ValueError(
            "external process wall is shorter than profiled cold outer wall: "
            f"{invalid_reconciliation}"
        )
    parity = analysis.get("parity")
    if not isinstance(parity, dict):
        raise ValueError("reciprocal parity is missing")
    exact_pass = bool(
        parity.get("cross_candidate_exact")
        and parity.get("required_exact_labels_pass")
        and parity.get("required_exact_dispatch_labels_pass")
        and parity.get("output_divergence", {}).get("final_map_equal")
    )
    cold_pass = cold_saving >= minimum_cold_saving_seconds
    warm_pass = warm_delta <= 0.0
    result = {
        "schema_version": 1,
        "minimum_cold_saving_seconds": minimum_cold_saving_seconds,
        "process_wall_seconds": process_walls,
        "cached_process_median_seconds": cached_median,
        "direct_process_median_seconds": direct_median,
        "complete_cold_wall_saving_seconds": cold_saving,
        "profile_cold_outer_saving_seconds": cold_profile_saving,
        "process_wall_reconciliation": wall_reconciliation,
        "warm_complete_request_candidate_minus_cached_seconds": warm_delta,
        "cold_saving_pass": cold_pass,
        "warm_no_regression_pass": warm_pass,
        "exact_parity_pass": exact_pass,
        "pass": cold_pass and warm_pass and exact_pass,
    }
    if build_result is not None:
        result["native_extension_build_seconds"] = float(
            build_result["build_seconds"]
        )
    if extension_load_seconds is not None:
        if set(extension_load_seconds) != set(RUN_ORDER):
            raise ValueError("extension-load timing roles are incomplete")
        cached_load = [
            float(extension_load_seconds["cached_first"]),
            float(extension_load_seconds["cached_second"]),
        ]
        direct_load = [
            float(extension_load_seconds["direct_first"]),
            float(extension_load_seconds["direct_second"]),
        ]
        result["native_extension_load_seconds"] = {
            "runs": extension_load_seconds,
            "cached_median": statistics.median(cached_load),
            "direct_median": statistics.median(direct_load),
            "direct_saving": (
                statistics.median(cached_load) - statistics.median(direct_load)
            ),
        }
    if storage_bytes is not None:
        invalid_storage = {
            key: value
            for key, value in storage_bytes.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        }
        if invalid_storage:
            raise ValueError(
                f"invalid native extension storage sizes: {invalid_storage}"
            )
        result["native_extension_storage_bytes"] = storage_bytes
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from osuT5.osuT5.inference.optimized.kernels.native_extension import (
        MANIFEST_ENV,
        validate_packaged_manifest,
    )
    from utils.analyze_reciprocal_full_song_candidate import analyze
    from utils.validate_k4_profile_contract import validate as validate_k4
    from utils.validate_k1_remainder_profile import validate as validate_k1
    from utils.validate_int8_mlp_full_song_profile import (
        validate_profile as validate_int8_mlp,
    )
    from utils.validate_weight_only_full_song_profile import (
        validate_profile as validate_weight,
    )

    run_root = args.run_root.resolve()
    if run_root.exists():
        raise RuntimeError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    preflight = validate_packaged_manifest(
        args.manifest,
        expected_source_commit=args.commit,
        expected_manifest_sha256=args.manifest_sha256,
        extension_cache_root=args.extension_cache_root,
    )
    external_manifest = _load_json(args.manifest, name="native extension manifest")
    build_result = _validate_build_result(
        args.build_result,
        expected_sha256=args.build_result_sha256,
        expected_commit=args.commit,
        manifest=external_manifest,
    )
    (run_root / "manifest-validation.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package = run_root / "prebuilt-extension-package"
    shutil.copytree(args.manifest.resolve().parent, package)
    local_manifest = package / args.manifest.name
    local_cache = run_root / "cached-extension-cache"
    shutil.copytree(args.extension_cache_root, local_cache)
    copied_preflight = validate_packaged_manifest(
        local_manifest,
        expected_source_commit=args.commit,
        expected_manifest_sha256=args.manifest_sha256,
        extension_cache_root=local_cache,
    )
    (run_root / "copied-manifest-validation.json").write_text(
        json.dumps(copied_preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(args.build_result, run_root / "native-extension-build.json")
    storage_bytes = {
        "prebuilt_package": _tree_size_bytes(package),
        "cached_jit_tree": _tree_size_bytes(local_cache),
    }
    common_overrides = (
        f"audio_path={args.audio}",
        "device=cuda",
        "precision=fp32",
        "attn_implementation=sdpa",
        "inference_engine=optimized",
        "use_server=false",
        "parallel=false",
        "cfg_scale=1.0",
        "num_beams=1",
        "seed=12345",
        "super_timing=false",
        "generate_positions=false",
        "profile_inference=true",
        "profile_detail_ranges=false",
        "profile_cuda_capture=false",
        "profile_pass_kind=untraced_control",
    )
    if "inference_generation_compile:" in (REPO_ROOT / "config.py").read_text():
        common_overrides += ("inference_generation_compile=false",)

    process_walls: dict[str, float] = {}
    extension_load_seconds: dict[str, float] = {}
    profile_paths: dict[str, Path] = {}
    initialization_evidence: dict[str, dict[str, Any]] = {}
    retained_initialization: dict[str, dict[str, Any]] = {}
    manifest_extensions = preflight["extensions"]
    for role in RUN_ORDER:
        mode = "direct" if role.startswith("direct") else "cached"
        output = run_root / role
        init_json = run_root / "initialization" / f"{role}.json"
        extension_json = run_root / "extension-load" / f"{role}.json"
        compiler_cache = run_root / "compiler-cache" / role
        env = os.environ.copy()
        env["TORCH_EXTENSIONS_DIR"] = str(local_cache)
        env["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache / "torch_inductor")
        env["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
        env["CUDA_CACHE_PATH"] = str(compiler_cache / "cuda")
        env["TORCH_CUDA_ARCH_LIST"] = "7.5"
        env["PYTHONHASHSEED"] = "0"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if mode == "direct":
            env[MANIFEST_ENV] = str(local_manifest)
        else:
            env.pop(MANIFEST_ENV, None)
        for directory in (
            compiler_cache / "torch_inductor",
            compiler_cache / "triton",
            compiler_cache / "cuda",
            init_json.parent,
            extension_json.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python),
            str(
                REPO_ROOT
                / "utils"
                / "run_k4_shared_rope_k1_remainder_int8_mlp_weight_only.py"
            ),
            "--config-name",
            "profile_salvalai",
            "--output-init-json",
            str(init_json),
            "--output-extension-json",
            str(extension_json),
            f"output_path={output}",
            *common_overrides,
        ]
        stdout_path = run_root / f"{role}.stdout.txt"
        stderr_path = run_root / f"{role}.stderr.txt"
        started = time.perf_counter()
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        process_walls[role] = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"{role} fresh process failed ({completed.returncode}); "
                f"see {stdout_path} and {stderr_path}"
            )
        profile = _profile_path(output)
        profile_payload = _load_json(profile, name=f"{role} profile")
        validate_weight(profile_payload, role="candidate")
        validate_k4(profile_payload, role="candidate", block_size=4)
        validate_k1(profile_payload, role="candidate")
        validate_int8_mlp(profile_payload, role="candidate")
        extension_load_seconds[role] = _validate_extension_evidence(
            _load_json(extension_json, name=f"{role} extension evidence"),
            manifest_extensions,
            mode=mode,
        )
        if not init_json.is_file() or not init_json.stat().st_size:
            raise RuntimeError(f"{role} initialization evidence is missing")
        initialization_evidence[role] = _load_json(
            init_json,
            name=f"{role} initialization evidence",
        )
        retained_initialization[role] = _validate_retained_initialization(
            initialization_evidence[role],
            role=role,
        )
        profile_paths[role] = profile

    first_retained = retained_initialization[RUN_ORDER[0]]
    if any(value != first_retained for value in retained_initialization.values()):
        raise RuntimeError("cached/direct retained runtime initialization diverged")

    analysis_paths = {
        analyzer_role: profile_paths[run_role]
        for analyzer_role, run_role in ANALYZER_ROLES.items()
    }
    analysis = analyze(
        analysis_paths,
        mode="relaxed",
        required_exact_labels=("timing_context", "main_generation"),
        required_exact_dispatch_labels=("timing_context", "main_generation"),
    )
    (run_root / "reciprocal-analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = evaluate_gate(
        analysis,
        process_walls,
        minimum_cold_saving_seconds=args.minimum_cold_saving_seconds,
        build_result=build_result,
        extension_load_seconds=extension_load_seconds,
        storage_bytes=storage_bytes,
    )
    gate["retained_runtime_initialization"] = first_retained
    gate["performance"] = {
        key: analysis["metrics"][key]
        for key in (
            "timing_model_seconds",
            "timing_tokens",
            "timing_tps",
            "timing_outer_stage_wall_seconds",
            "timing_postprocess_seconds",
            "main_model_seconds",
            "main_tokens",
            "main_tps",
            "fixed_8294_main_seconds",
            "main_outer_stage_wall_seconds",
            "final_postprocess_write_seconds",
            "complete_request_wall_seconds",
            "cold_process_outer_wall_seconds",
            "setup_stage_sum_seconds",
            "graph_capture_seconds",
            "setup_plus_capture_seconds",
            "total_postprocess_seconds",
            "peak_cuda_memory_allocated_mb",
        )
    }
    gate["exactness"] = {
        "comparison": "cached-jit-versus-direct-prebuilt-same-runtime",
        "accepted_fp32_exactness_claim": False,
        "token_and_stopping_divergence": analysis["parity"][
            "token_and_stopping_divergence"
        ],
        "dispatch_cache_topology": analysis["parity"][
            "dispatch_cache_topology"
        ],
        "output_divergence": analysis["parity"]["output_divergence"],
    }
    (run_root / "aot-gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not gate["pass"]:
        raise SystemExit(3)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--build-result", type=Path, required=True)
    parser.add_argument("--build-result-sha256", required=True)
    parser.add_argument("--extension-cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--minimum-cold-saving-seconds", type=float, default=0.5)
    args = parser.parse_args()
    gate = run(args)
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
