"""Compose K=4, mixed weights, split-KV, and exact shared decoder RoPE."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.scout.shared_rope import (  # noqa: E402
    SharedRopeStats,
    shared_decoder_rope_context,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from utils.run_approximate_weight_only import run as run_weight_only  # noqa: E402


COMPOSITION_VERSION = "k4-split-kv-mixed-weight-shared-rope-v1"


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


def _enrich_initialization_evidence(
    output_init_json: Path,
    stats: SharedRopeStats,
    *,
    composition_version: str = COMPOSITION_VERSION,
) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "mixed-weight initialization evidence is missing or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("mixed-weight initialization evidence must be an object")
    if "shared_rope" in payload or "combined_runtime" in payload:
        raise RuntimeError("combined initialization evidence already contains wiring")
    payload["combined_runtime"] = composition_version
    payload["shared_rope"] = _validated_shared_rope_evidence(stats)
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    output_extension_json: Path | None = None,
    *,
    graph_remainders: bool = False,
    weight_runner=None,
    composition_version: str = COMPOSITION_VERSION,
) -> None:
    """Install every candidate while sharing RoPE only on the main model.

    ``inference.main`` loads the main binding before its separate timing binding.
    The outer loader patch therefore scopes shared RoPE to the first raw model;
    ``run_weight_only`` retains ownership of mixed-weight initialization on that
    same binding. K=4 is process-scoped and continues to own both decode loops.
    """

    import inference

    if weight_runner is None:
        weight_runner = run_weight_only

    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    main_stats = SharedRopeStats()
    loaded_bindings = 0

    def shared_rope_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=main_stats)
            )
        return binding, tokenizer

    inference.load_model_with_engine = shared_rope_loader
    try:
        install_options = {"block_size": 4}
        if graph_remainders:
            install_options["graph_remainders"] = True
        with install_k8_candidate(**install_options):
            weight_runner(config_name, overrides, output_init_json)
        if loaded_bindings < 2:
            raise RuntimeError(
                "combined full-song runner expected separate main and timing bindings"
            )
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    _enrich_initialization_evidence(
        output_init_json,
        main_stats,
        composition_version=composition_version,
    )
    if output_extension_json is not None:
        from osuT5.osuT5.inference.optimized.kernels.native_extension import (
            loaded_extension_records,
        )

        records = loaded_extension_records()
        if not records:
            raise RuntimeError("shared-RoPE runtime loaded no native extensions")
        output_extension_json.parent.mkdir(parents=True, exist_ok=True)
        output_extension_json.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-init-json", type=Path, required=True)
    parser.add_argument("--output-extension-json", type=Path)
    parser.add_argument("overrides", nargs="*")
    parsed = parser.parse_args()
    run(
        parsed.config_name,
        parsed.overrides,
        parsed.output_init_json,
        parsed.output_extension_json,
    )


if __name__ == "__main__":
    main()
