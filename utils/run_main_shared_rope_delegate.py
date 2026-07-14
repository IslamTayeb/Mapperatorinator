"""Scope exact shared decoder RoPE to the first, main-model binding."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from typing import Callable

from osuT5.osuT5.inference.optimized.scout.shared_rope import (
    SharedRopeStats,
    shared_decoder_rope_context,
)


SHARED_STAGE_COMPOSITION_VERSION = (
    "k4-split-kv-mixed-weight-shared-rope-stage-precision-v1"
)


def _validated_shared_rope_evidence(stats: SharedRopeStats) -> dict:
    evidence = stats.as_dict()
    if evidence["module_count"] <= 1 or evidence["group_count"] <= 0:
        raise RuntimeError("shared RoPE did not discover a reusable decoder group")
    if evidence["forwards"] <= 0:
        raise RuntimeError("shared RoPE did not observe a main-model forward")
    if evidence["computes"] != evidence["expected_computes"]:
        raise RuntimeError("shared RoPE compute accounting is inconsistent")
    if evidence["reuses"] != evidence["expected_reuses"]:
        raise RuntimeError("shared RoPE reuse accounting is inconsistent")
    if evidence["reuses"] <= 0:
        raise RuntimeError("shared RoPE did not eliminate a redundant computation")
    return {
        "version": "shared-decoder-rope-v1",
        "scope": "main-model-only",
        "incremental_exactness_claim": True,
        "original_decoder_forward_required": True,
        "stats": evidence,
    }


def _enrich_initialization(
    path: Path,
    stats: SharedRopeStats,
    *,
    arm: str,
    composition_version: str,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("delegate initialization evidence is missing or invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("delegate initialization evidence must be an object")
    if "shared_rope" in payload or "combined_runtime" in payload:
        raise RuntimeError("delegate initialization evidence already contains shared wiring")
    payload["combined_runtime"] = composition_version
    payload["composition_arm"] = arm
    payload["shared_rope"] = _validated_shared_rope_evidence(stats)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_with_main_shared_rope(
    delegate: Callable[[str, list[str], Path], None],
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    *,
    arm: str,
    composition_version: str = SHARED_STAGE_COMPOSITION_VERSION,
) -> None:
    """Run a two-model delegate while patching only its first/main model.

    Both stage-aware delegates own K4/K1 selection and mixed-weight lifecycle.
    This outer layer owns only the exact shared-RoPE context, avoiding a second
    decode-block context or a second decoder implementation.
    """

    if arm not in {"control", "candidate"}:
        raise ValueError("shared stage arm must be control or candidate")
    if not isinstance(composition_version, str) or not composition_version:
        raise ValueError("shared stage composition version must be non-empty")

    import inference

    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    main_stats = SharedRopeStats()
    loaded_bindings = 0

    def shared_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings
        if loaded_bindings > 1:
            raise RuntimeError("shared stage composition observed an extra model load")
        expected_main = loaded_bindings == 0
        if loader_kwargs.get("auto_select_gamemode_model") is not expected_main:
            raise RuntimeError("shared stage composition model load order changed")
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if expected_main:
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=main_stats)
            )
        return binding, tokenizer

    inference.load_model_with_engine = shared_loader
    try:
        delegate(config_name, overrides, output_init_json)
        if loaded_bindings != 2:
            raise RuntimeError(
                "shared stage composition expected exactly main and timing bindings"
            )
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    _enrich_initialization(
        output_init_json,
        main_stats,
        arm=arm,
        composition_version=composition_version,
    )


__all__ = [
    "SHARED_STAGE_COMPOSITION_VERSION",
    "run_with_main_shared_rope",
]
