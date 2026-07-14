"""Compose one cross-region delta with the current K4 mixed/shared-RoPE base."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osuT5.osuT5.inference.optimized.kernels.weight_only_runtime import (  # noqa: E402
    CROSS_FP16_PACKED,
    CROSS_SPLIT8,
)
from osuT5.osuT5.inference.optimized.scout.shared_rope import (  # noqa: E402
    SharedRopeStats,
    shared_decoder_rope_context,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    install_k8_candidate,
)
from utils.run_approximate_weight_only import run_with_initializer  # noqa: E402
from utils.run_k4_shared_rope_approximate_weight_only import (  # noqa: E402
    _validated_shared_rope_evidence,
)


CROSS_CANDIDATE_MODES = frozenset({CROSS_FP16_PACKED, CROSS_SPLIT8})


def _enrich_evidence(
    output_init_json: Path,
    *,
    mode: str,
    stats: SharedRopeStats,
) -> None:
    try:
        payload = json.loads(output_init_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cross candidate initialization evidence is invalid") from exc
    if not isinstance(payload, dict):
        raise TypeError("cross candidate initialization evidence must be an object")
    cross = payload.get("cross_candidate")
    if not isinstance(cross, dict) or cross.get("mode") != mode:
        raise RuntimeError("cross candidate initialization mode was not recorded")
    if "shared_rope" in payload or "combined_runtime" in payload:
        raise RuntimeError("cross candidate evidence already contains composition wiring")
    payload["combined_runtime"] = f"k4-split-kv-mixed-weight-shared-rope-{mode}-v1"
    payload["shared_rope"] = _validated_shared_rope_evidence(stats)
    payload["cross_runtime"] = {
        **cross,
        "incremental_control": "k4-split-kv-mixed-weight-shared-rope-v1",
        "original_decoder_forward_required": True,
    }
    output_init_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    config_name: str,
    overrides: list[str],
    output_init_json: Path,
    *,
    mode: str,
) -> None:
    if mode not in CROSS_CANDIDATE_MODES:
        raise ValueError(
            f"cross candidate mode must be one of {sorted(CROSS_CANDIDATE_MODES)}"
        )
    import inference

    original_loader = inference.load_model_with_engine
    stack = ExitStack()
    main_stats = SharedRopeStats()
    loaded_bindings = 0

    def candidate_loader(*loader_args, **loader_kwargs):
        nonlocal loaded_bindings
        binding, tokenizer = original_loader(*loader_args, **loader_kwargs)
        loaded_bindings += 1
        if loaded_bindings == 1:
            stack.enter_context(
                shared_decoder_rope_context(binding.raw_model, stats=main_stats)
            )
        return binding, tokenizer

    inference.load_model_with_engine = candidate_loader
    try:
        with install_k8_candidate(block_size=4):
            run_with_initializer(
                config_name,
                overrides,
                output_init_json,
                initializer_name="initialize_approximate_weight_only_cross",
                initializer_kwargs={"mode": mode},
            )
        if loaded_bindings < 2:
            raise RuntimeError(
                "cross candidate expected separate main and timing bindings"
            )
    finally:
        inference.load_model_with_engine = original_loader
        stack.close()
    _enrich_evidence(output_init_json, mode=mode, stats=main_stats)


__all__ = ["CROSS_CANDIDATE_MODES", "run"]
