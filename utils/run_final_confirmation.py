"""Fresh-process natural and fixed-work inference confirmation runner.

This utility is benchmark-only.  It wraps the selected optimized runtime after
model loading and leaves the production selector and default dispatch untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.engine_binding import InferenceEngineBinding
from utils.final_confirmation_runtime import load_runtime_plugin, native_extension_evidence
from utils.fixed_seed_inference import (
    fixed_seed_processor_generation,
    rng_state_fingerprints,
)
from utils.run_approximate_weight_only import _load_args


SCHEMA_VERSION = "mapperatorinator.final-confirmation-run.v1"
MANIFEST_SCHEMA_VERSION = "mapperatorinator.fixed-work-manifest.v1"
MODES = ("natural", "record-fixed-work", "replay-fixed-work")
PROFILE_LABEL_BY_CONTEXT = {"timing": "timing_context", "map": "main_generation"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("fixed-work manifest schema is missing or unsupported")
    labels = payload.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(PROFILE_LABEL_BY_CONTEXT.values()):
        raise ValueError("fixed-work manifest must contain timing_context and main_generation")
    for label, raw_counts in labels.items():
        if (
            not isinstance(raw_counts, list)
            or not raw_counts
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw_counts)
        ):
            raise ValueError(f"fixed-work manifest {label} counts must be positive integers")
    return payload


class ConfirmationRuntime:
    def __init__(
        self,
        runtime: Any,
        *,
        mode: str,
        manifest: dict[str, Any] | None,
        indices: defaultdict[str, int] | None = None,
        records: list[dict[str, Any]] | None = None,
        device_resident_fixed_work: bool = False,
    ) -> None:
        self._runtime = runtime
        self._mode = mode
        self._manifest = manifest
        self._indices = indices if indices is not None else defaultdict(int)
        self.records = records if records is not None else []
        if not isinstance(device_resident_fixed_work, bool):
            raise TypeError("device_resident_fixed_work must be boolean")
        self._device_resident_fixed_work = device_resident_fixed_work

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def profile_metadata(self) -> dict[str, Any]:
        metadata = dict(self._runtime.profile_metadata())
        metadata.update(
            {
                "final_confirmation_mode": self._mode,
                "fixed_work_manifest_schema": (
                    self._manifest.get("schema_version")
                    if self._manifest is not None
                    else None
                ),
                "device_resident_fixed_work": self._device_resident_fixed_work,
            }
        )
        return metadata

    def generate_window(self, **kwargs: Any):
        generate_kwargs = dict(kwargs["generate_kwargs"])
        context_type = generate_kwargs.get("context_type")
        try:
            label = PROFILE_LABEL_BY_CONTEXT[str(context_type)]
        except KeyError as exc:
            raise RuntimeError(
                f"confirmation runner does not support context_type={context_type!r}"
            ) from exc
        model_kwargs = kwargs["model_kwargs"]
        prompt = model_kwargs.get("decoder_input_ids")
        if prompt is None or getattr(prompt, "ndim", None) != 2 or prompt.shape[0] != 1:
            raise RuntimeError("confirmation runner requires one decoder prompt row")
        prompt_width = int(prompt.shape[1])
        index = self._indices[label]
        target_steps = None
        fixed_work_eos_ids: list[int] = []
        original_eos = None
        engine_module = None
        if self._mode == "replay-fixed-work":
            assert self._manifest is not None
            counts = self._manifest["labels"][label]
            if index >= len(counts):
                raise RuntimeError(f"fixed-work replay has too many {label} windows")
            target_steps = int(counts[index])
            if not self._device_resident_fixed_work:
                generate_kwargs["max_length"] = prompt_width + target_steps
            from osuT5.osuT5.inference.optimized.single import engine as engine_module
            from osuT5.osuT5.event import ContextType

            original_eos = engine_module.eos_token_ids
            fixed_work_eos_ids = original_eos(
                kwargs["tokenizer"],
                lookback_time=float(generate_kwargs.get("lookback_time", 0.0)),
                lookahead_time=float(generate_kwargs.get("lookahead_time", 0.0)),
                context_type=ContextType(str(context_type)),
            )
            # Keep one impossible EOS id so K8 retains a valid EosTokenCriteria
            # while executing the fixed logical target. Natural EOS ids are
            # recorded above and used to trim the consumer-facing result.
            engine_module.eos_token_ids = lambda *args, **kwargs: [-1]
        call_kwargs = dict(kwargs)
        call_kwargs["generate_kwargs"] = generate_kwargs
        fixed_target_context = nullcontext()
        if target_steps is not None and self._device_resident_fixed_work:
            from osuT5.osuT5.inference.optimized.single.k8_runtime import (
                fixed_work_target_steps,
            )

            fixed_target_context = fixed_work_target_steps(target_steps)
        try:
            with fixed_target_context:
                result, stats = self._runtime.generate_window(**call_kwargs)
        finally:
            if engine_module is not None and original_eos is not None:
                engine_module.eos_token_ids = original_eos
        if getattr(result, "ndim", None) != 2 or result.shape[0] != 1:
            raise RuntimeError("confirmation runtime returned an invalid token tensor")
        logical_steps = int(result.shape[1]) - prompt_width
        if logical_steps <= 0:
            raise RuntimeError("confirmation runtime generated no logical steps")
        if target_steps is not None and logical_steps != target_steps:
            raise RuntimeError(
                f"fixed-work {label}[{index}] executed {logical_steps} steps, expected {target_steps}"
            )
        consumer_steps = logical_steps
        if target_steps is not None and fixed_work_eos_ids:
            generated = result[0, prompt_width:]
            eos = torch.tensor(
                fixed_work_eos_ids,
                dtype=generated.dtype,
                device=generated.device,
            )
            positions = torch.nonzero(torch.isin(generated, eos), as_tuple=False)
            if positions.numel() > 0:
                consumer_steps = int(positions[0, 0].item()) + 1
                result = result[:, : prompt_width + consumer_steps]
                from osuT5.osuT5.inference.generation_utils import build_generation_stats

                rebuilt = build_generation_stats(
                    result,
                    model_kwargs,
                    getattr(kwargs["tokenizer"], "pad_id", None),
                    float(stats["elapsed_seconds"]),
                )
                stats = {**stats, **rebuilt}
            stats["fixed_work_logical_steps"] = logical_steps
        self.records.append(
            {
                "profile_label": label,
                "window_index": index,
                "prompt_width": prompt_width,
                "logical_steps": logical_steps,
                "consumer_steps": consumer_steps,
                "declared_generated_tokens": int(stats.get("generated_tokens", 0)),
                "target_steps": target_steps,
            }
        )
        self._indices[label] += 1
        return result, stats

    def validate_complete(self) -> None:
        if self._mode != "replay-fixed-work":
            return
        assert self._manifest is not None
        for label, counts in self._manifest["labels"].items():
            if self._indices[label] != len(counts):
                raise RuntimeError(
                    f"fixed-work replay consumed {self._indices[label]} {label} windows, expected {len(counts)}"
                )


def _recorded_labels(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {}
    for label in PROFILE_LABEL_BY_CONTEXT.values():
        entries = [record for record in records if record["profile_label"] == label]
        if not entries:
            raise RuntimeError(f"confirmation run recorded no {label} windows")
        if [entry["window_index"] for entry in entries] != list(range(len(entries))):
            raise RuntimeError(f"confirmation run {label} window indices are not contiguous")
        labels[label] = [int(entry["logical_steps"]) for entry in entries]
    return labels


def run(
    config_name: str,
    overrides: list[str],
    *,
    mode: str,
    candidate: bool,
    runtime_spec_path: Path | None,
    manifest_path: Path | None,
    evidence_path: Path,
    initialization_path: Path | None,
    expected_main_steps: int,
) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if expected_main_steps <= 0:
        raise ValueError("expected_main_steps must be positive")
    if mode == "natural" and manifest_path is not None:
        raise ValueError("natural confirmation must not receive a fixed-work manifest")
    if mode != "natural" and manifest_path is None:
        raise ValueError("fixed-work confirmation requires --manifest-path")
    if candidate != (runtime_spec_path is not None):
        raise ValueError("candidate mode requires exactly one runtime spec")
    if candidate != (initialization_path is not None):
        raise ValueError("candidate mode requires exactly one runtime evidence path")

    import torch
    import inference

    args = _load_args(config_name, overrides)
    if args.inference_engine != "optimized" or args.precision != "fp32":
        raise ValueError("final confirmation requires optimized FP32 storage/runtime")
    if args.use_server or args.parallel or not args.profile_inference:
        raise ValueError("final confirmation requires local sequential profiling")
    manifest = _load_manifest(manifest_path) if mode == "replay-fixed-work" else None
    if manifest is not None:
        actual = sum(manifest["labels"]["main_generation"])
        if actual != expected_main_steps:
            raise ValueError(
                f"fixed-work manifest has {actual} main steps, expected {expected_main_steps}"
            )

    original_loader = inference.load_model_with_engine
    wrapped_runtimes: list[ConfirmationRuntime] = []
    shared_indices: defaultdict[str, int] = defaultdict(int)
    shared_records: list[dict[str, Any]] = []
    plugin = load_runtime_plugin(runtime_spec_path)

    def loader(*loader_args: Any, **loader_kwargs: Any):
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        if not isinstance(binding, InferenceEngineBinding):
            raise TypeError("final confirmation requires an optimized engine binding")
        binding = plugin.transform_binding(binding, binding_index=len(wrapped_runtimes))
        wrapped_runtime = ConfirmationRuntime(
            binding.runtime,
            mode=mode,
            manifest=manifest,
            indices=shared_indices,
            records=shared_records,
            device_resident_fixed_work=bool(
                getattr(plugin, "device_resident_fixed_work", False)
            ),
        )
        wrapped_runtimes.append(wrapped_runtime)
        return InferenceEngineBinding(binding.raw_model, wrapped_runtime), tokenizer

    inference.load_model_with_engine = loader
    process_started = time.time_ns()
    torch.cuda.reset_peak_memory_stats()
    rng_before = rng_state_fingerprints()
    try:
        with plugin, fixed_seed_processor_generation(inference, base_seed=args.seed):
            _, result_path = inference.main(args)
    finally:
        inference.load_model_with_engine = original_loader
    rng_after = rng_state_fingerprints()
    if not wrapped_runtimes:
        raise RuntimeError("confirmation model was never loaded")
    wrapped_runtimes[0].validate_complete()
    labels = _recorded_labels(shared_records)
    runtime_evidence = plugin.evidence()
    if initialization_path is not None:
        initialization_path.parent.mkdir(parents=True, exist_ok=True)
        initialization_path.write_text(
            json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if mode == "record-fixed-work":
        assert manifest_path is not None
        if sum(labels["main_generation"]) != expected_main_steps:
            raise RuntimeError(
                "recorded baseline main work changed: "
                f"{sum(labels['main_generation'])} != {expected_main_steps}"
            )
        manifest_payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "expected_main_steps": expected_main_steps,
            "labels": labels,
            "audio_path": str(Path(args.audio_path).resolve()),
            "seed": int(args.seed),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = manifest_payload

    torch.cuda.synchronize()
    result_path = Path(result_path).resolve()
    profiles = sorted(result_path.parent.glob(f"{result_path.name}.profile.json"))
    if len(profiles) != 1:
        raise RuntimeError(f"expected one profile beside {result_path}, got {profiles}")
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "run_uuid": str(uuid.uuid4()),
        "pid": os.getpid(),
        "process_started_time_ns": process_started,
        "process_finished_time_ns": time.time_ns(),
        "mode": mode,
        "candidate": candidate,
        "profile_path": str(profiles[0].resolve()),
        "profile_sha256": _sha256_file(profiles[0]),
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path),
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path is not None else None,
        "expected_main_steps": expected_main_steps,
        "labels": {
            label: {
                "windows": len(counts),
                "logical_steps": sum(counts),
            }
            for label, counts in labels.items()
        },
        "records": shared_records,
        "runtime": runtime_evidence,
        "runtime_spec_path": (
            str(runtime_spec_path.resolve()) if runtime_spec_path is not None else None
        ),
        "runtime_spec_sha256": (
            _sha256_file(runtime_spec_path) if runtime_spec_path is not None else None
        ),
        "rng": {
            "before": rng_before,
            "after": rng_after,
        },
        "cuda_memory": {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "initialization": runtime_evidence["initialization"],
        "native_extensions": native_extension_evidence(),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--runtime-spec", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--initialization-path", type=Path)
    parser.add_argument("--expected-main-steps", type=int, default=8294)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        mode=parsed.mode,
        candidate=parsed.candidate,
        runtime_spec_path=parsed.runtime_spec,
        manifest_path=parsed.manifest_path,
        evidence_path=parsed.evidence_path,
        initialization_path=parsed.initialization_path,
        expected_main_steps=parsed.expected_main_steps,
    )


if __name__ == "__main__":
    main()
