"""§37/§43 N-layer same-width draft construction for turbo.

§43 perf default is 1-layer (init_layers=(0,)) with γ=3; quality alt is
2-layer multi-song. Ckpt payload carries init_layers.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

DRAFT_INIT_LAYERS = (0, 1)
DEFAULT_DRAFT_CKPT_ENV = "MAPPERATORINATOR_TURBO_DRAFT_CKPT"


def get_decoder(model: nn.Module) -> nn.Module:
    return model.transformer.model.decoder


def build_n_layer_draft(
    teacher: nn.Module,
    init_layers: tuple[int, ...] = DRAFT_INIT_LAYERS,
) -> nn.Module:
    if len(init_layers) < 1:
        raise ValueError(f"expected >=1 init layers, got {init_layers}")
    draft = copy.deepcopy(teacher)
    dec_t = get_decoder(teacher)
    dec_d = get_decoder(draft)
    n = len(dec_t.layers)
    for i in init_layers:
        if i < 0 or i >= n:
            raise ValueError(f"init layer {i} out of range for {n} layers")
    dec_d.layers = nn.ModuleList([copy.deepcopy(dec_t.layers[i]) for i in init_layers])
    # Keep config in sync so cache / layer-count helpers see the draft depth.
    n_layers = len(init_layers)
    for cfg in (
        getattr(draft, "config", None),
        getattr(getattr(draft, "transformer", None), "config", None),
        getattr(getattr(draft, "config", None), "backbone_config", None),
    ):
        if cfg is not None and hasattr(cfg, "decoder_layers"):
            cfg.decoder_layers = n_layers
    for p in draft.parameters():
        p.requires_grad = False
    draft.eval()
    return draft


def build_two_layer_draft(
    teacher: nn.Module,
    init_layers: tuple[int, ...] = DRAFT_INIT_LAYERS,
) -> nn.Module:
    """Backward-compatible alias for build_n_layer_draft."""
    return build_n_layer_draft(teacher, init_layers)


def resolve_draft_ckpt(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(DEFAULT_DRAFT_CKPT_ENV)
    if env:
        return Path(env)
    raise FileNotFoundError(
        f"turbo draft ckpt required via {DEFAULT_DRAFT_CKPT_ENV} or explicit path"
    )


def load_draft_from_ckpt(
    teacher: nn.Module,
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    path = resolve_draft_ckpt(ckpt_path)
    payload = torch.load(path, map_location="cpu")
    state = payload.get("draft_state_dict", payload)
    init_layers = tuple(payload.get("init_layers", DRAFT_INIT_LAYERS))
    draft = build_n_layer_draft(teacher, init_layers)
    # Timing / non-gamemode teachers can differ in vocab size from the §37
    # gamemode=0 distill ckpt. Load only shape-compatible tensors.
    model_sd = draft.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in model_sd and model_sd[key].shape == value.shape
    }
    skipped_shape = sorted(set(state) - set(filtered))
    missing, unexpected = draft.load_state_dict(filtered, strict=False)
    if device is not None:
        draft = draft.to(device)
    if dtype is not None:
        draft = draft.to(dtype=dtype)
    draft.eval()
    meta = {
        "ckpt_path": str(path),
        "init_layers": list(init_layers),
        "n_layers": len(init_layers),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_shape_keys": skipped_shape,
        "loaded_param_count": len(filtered),
        "train_steps": payload.get("train_steps"),
        "tip_commit": payload.get("tip_commit"),
    }
    return draft, meta
