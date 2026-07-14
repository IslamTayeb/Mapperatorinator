from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_EXTENSIONS = (
    "mapperatorinator_q1_attention",
    "mapperatorinator_native_decoder_layer",
    "mapperatorinator_weight_only_fp16_v1",
)


def _probe_callable(function: Any) -> dict[str, str]:
    try:
        value = function()
    except BaseException as exc:
        return {
            "kind": "exception",
            "type": type(exc).__name__,
            "message": str(exc).splitlines()[0],
        }
    return {"kind": "return", "type": type(value).__name__, "message": ""}


def worker(mode: str, manifest: Path) -> dict[str, Any]:
    from osuT5.osuT5.inference.optimized.kernels.native_extension import MANIFEST_ENV

    if mode == "direct":
        os.environ[MANIFEST_ENV] = str(manifest.resolve())
    elif mode == "cached":
        os.environ.pop(MANIFEST_ENV, None)
    else:
        raise ValueError(f"unsupported native extension load mode {mode!r}")
    started = time.perf_counter()
    from osuT5.osuT5.inference.optimized.kernels import (
        decoder_layer,
        q1_attention,
        weight_only,
    )
    from osuT5.osuT5.inference.optimized.kernels.native_extension import (
        loaded_extension_records,
    )

    modules = (
        q1_attention.preload_native_q1_attention(),
        decoder_layer.preload_native_decoder_layer(),
        weight_only.preload_weight_only_extension(),
    )
    elapsed = time.perf_counter() - started
    records = loaded_extension_records()
    probes: dict[str, dict[str, dict[str, str]]] = {}
    for name, module in zip(EXPECTED_EXTENSIONS, modules, strict=True):
        record = records[name]
        probes[name] = {
            symbol: _probe_callable(getattr(module, symbol))
            for symbol in record["functions"]
        }
    return {
        "mode": mode,
        "seconds": elapsed,
        "records": records,
        "probes": probes,
    }


def _comparable(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "extensions": {
            name: {
                "source_sha256": record["source_sha256"],
                "library_sha256": record["library_sha256"],
                "functions": record["functions"],
            }
            for name, record in run["records"].items()
        },
        "probes": run["probes"],
    }


def summarize(
    runs: list[dict[str, Any]],
    *,
    minimum_saving_seconds: float,
) -> dict[str, Any]:
    cached = [run for run in runs if run.get("mode") == "cached"]
    direct = [run for run in runs if run.get("mode") == "direct"]
    if not cached or len(cached) != len(direct):
        raise ValueError("benchmark requires paired cached and direct runs")
    reference = _comparable(cached[0])
    parity_failures = []
    for index, run in enumerate(runs):
        if _comparable(run) != reference:
            parity_failures.append({"index": index, "mode": run.get("mode")})
    cached_median = statistics.median(float(run["seconds"]) for run in cached)
    direct_median = statistics.median(float(run["seconds"]) for run in direct)
    saving = cached_median - direct_median
    parity_pass = not parity_failures
    saving_pass = saving >= minimum_saving_seconds
    return {
        "schema_version": 1,
        "minimum_saving_seconds": minimum_saving_seconds,
        "cached_seconds": [float(run["seconds"]) for run in cached],
        "direct_seconds": [float(run["seconds"]) for run in direct],
        "cached_median_seconds": cached_median,
        "direct_median_seconds": direct_median,
        "saving_seconds": saving,
        "parity_failures": parity_failures,
        "parity_pass": parity_pass,
        "saving_pass": saving_pass,
        "pass": parity_pass and saving_pass,
        "runs": runs,
    }


def _run_worker(mode: str, manifest: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--manifest",
        str(manifest.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native extension {mode} worker failed ({completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native extension {mode} worker returned invalid JSON: {completed.stdout!r}"
        ) from exc


def benchmark(
    manifest: Path,
    *,
    rounds: int,
    minimum_saving_seconds: float,
) -> dict[str, Any]:
    if rounds <= 0:
        raise ValueError("benchmark rounds must be positive")
    runs = []
    for round_index in range(rounds):
        order = ("cached", "direct") if round_index % 2 == 0 else ("direct", "cached")
        for mode in order:
            runs.append(_run_worker(mode, manifest))
    return summarize(runs, minimum_saving_seconds=minimum_saving_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--minimum-saving-seconds", type=float, default=0.5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("cached", "direct"))
    args = parser.parse_args()
    if args.worker:
        if args.mode is None:
            parser.error("--worker requires --mode")
        print(json.dumps(worker(args.mode, args.manifest), sort_keys=True))
        return
    if args.output is None:
        parser.error("benchmark mode requires --output")
    result = benchmark(
        args.manifest,
        rounds=args.rounds,
        minimum_saving_seconds=args.minimum_saving_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
