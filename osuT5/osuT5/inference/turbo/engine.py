"""Turbo runtime (Track C §52 integrator: §47+§49 path-hit scout).

Immutable preset: §43 1-layer draft (K=1, γ=3, temp=0.9) + §49 graphed draft
chain + §47 keep-KV + §48 graph-native verify (no grind). Not bit-exact.
Campaign tip remains 55949274 / FP16 366.11.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from ..engine_binding import InferenceEngineBinding
from .draft import DEFAULT_DRAFT_CKPT_ENV, load_draft_from_ckpt
from .rejection import apply_temp_top_p, reject_sample_prefix
from .speculate import speculative_generate_window

TURBO_PRESET_VERSION = "turbo-integrator-s52-kv-dg-v1"
PRIMARY_GAMMA = 3


@dataclass
class TurboDecodeSession:
    """Request-local mutable state for turbo speculative decode.

    Teacher StaticCache + verify_fp persist across windows so CUDA-graph /
    bucket entries are not rebuilt per window (§47).
    """

    accepted_tokens_total: int = 0
    verify_steps: int = 0
    draft_calls: int = 0
    teacher_cache: Any | None = None
    draft_cache: Any | None = None
    verify_fp: Any | None = None
    teacher_forwards: int = 0
    teacher_verify_forwards: int = 0
    teacher_extra_q1_forwards: int = 0
    teacher_accepted_reforwards: int = 0
    draft_accepted_reforwards: int = 0
    draft_chain_runner: Any | None = None


@dataclass(frozen=True, slots=True)
class TurboPreset:
    version: str
    precision: str
    gamma: int
    temperature: float
    top_p: float
    result_class: str = "distribution-equivalent-pending-tier1"


TURBO_PRESETS = MappingProxyType(
    {
        "fp16": TurboPreset(
            version=TURBO_PRESET_VERSION,
            precision="fp16",
            gamma=PRIMARY_GAMMA,
            temperature=0.9,
            top_p=0.9,
        ),
        "fp32": TurboPreset(
            version=TURBO_PRESET_VERSION,
            precision="fp32",
            gamma=PRIMARY_GAMMA,
            temperature=0.9,
            top_p=0.9,
        ),
    }
)


@dataclass
class TurboRuntime:
    """Opt-in turbo runtime bound to a teacher + §37 tiny draft."""

    preset: TurboPreset
    draft: Any
    draft_meta: dict[str, Any]
    speculative_generate_wired: bool = True

    def new_context_state(self) -> TurboDecodeSession:
        return TurboDecodeSession()

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_effective_config_version": self.preset.version,
            "turbo_runtime_owner": "osuT5.osuT5.inference.turbo.engine",
            "turbo_result_class": self.preset.result_class,
            "turbo_precision": self.preset.precision,
            "turbo_gamma": self.preset.gamma,
            "turbo_temperature": self.preset.temperature,
            "turbo_top_p": self.preset.top_p,
            "turbo_draft_ckpt": self.draft_meta.get("ckpt_path"),
            "turbo_draft_init_layers": self.draft_meta.get("init_layers"),
            "turbo_speculative_generate_window": (
                "wired" if self.speculative_generate_wired else "scaffold_pending"
            ),
            "turbo_tier1_required": True,
            "turbo_verify_fastpath_default": True,
            "turbo_tree_k": 1,
            "note": (
                "§52 integrator: §47 keep-KV + §49 graphed draft chain + §48 "
                "graph-native verify (no VG grind). Sampled path DOCUMENTED "
                "DRIFT under §34; greedy TIER1a stays crop-rebuild/aligned-Q1. "
                "Campaign tip 55949274/366.11. No 500 claim."
            ),
        }

    def reject_sample_prefix(
        self,
        *,
        teacher_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids,
        rng: torch.Generator | None = None,
    ) -> tuple[int, int | None]:
        p = apply_temp_top_p(teacher_logits, self.preset.temperature, self.preset.top_p)
        q = apply_temp_top_p(draft_logits, self.preset.temperature, self.preset.top_p)
        return reject_sample_prefix(
            p_probs=p,
            q_probs=q,
            draft_token_ids=draft_token_ids,
            rng=rng,
        )

    def generate_window(
        self,
        *,
        model,
        tokenizer,
        model_kwargs: dict[str, Any],
        generate_kwargs: dict[str, Any],
        context_state: TurboDecodeSession,
    ):
        if not isinstance(context_state, TurboDecodeSession):
            raise TypeError("turbo runtime requires TurboDecodeSession")
        allow_fallback = (
            os.environ.get("MAPPERATORINATOR_TURBO_ALLOW_TEACHER_FALLBACK", "") == "1"
        )
        if not self.speculative_generate_wired:
            if not allow_fallback:
                raise RuntimeError(
                    "turbo speculative generate_window is not wired. "
                    f"draft_ckpt={self.draft_meta.get('ckpt_path')}"
                )
            from ..server import model_generate

            gw = dict(generate_kwargs)
            for k in ("collect_strict_exactness", "sync_model_timing"):
                gw.pop(k, None)
            result, stats = model_generate(model, tokenizer, model_kwargs, gw)
            stats = dict(stats or {})
            stats["turbo_scaffold"] = True
            stats["turbo_teacher_fallback"] = True
            stats["turbo_speculative"] = False
            context_state.verify_steps += 1
            return result, stats

        gamma = self.preset.gamma
        gamma_env = os.environ.get("MAPPERATORINATOR_TURBO_GAMMA", "").strip()
        if gamma_env:
            gamma = max(1, int(gamma_env))
        return speculative_generate_window(
            teacher=model,
            draft=self.draft,
            tokenizer=tokenizer,
            model_kwargs=model_kwargs,
            generate_kwargs=dict(generate_kwargs),
            gamma=gamma,
            temperature=self.preset.temperature,
            top_p=self.preset.top_p,
            session=context_state,
        )


def load_turbo_runtime(
    *,
    teacher: Any,
    precision: str,
    draft_ckpt: str | None = None,
) -> TurboRuntime:
    if precision not in TURBO_PRESETS:
        raise ValueError(f"turbo supports precision fp16|fp32, got {precision!r}")
    dtype = torch.float16 if precision == "fp16" else torch.float32
    device = getattr(teacher, "device", None)
    draft, meta = load_draft_from_ckpt(
        teacher,
        draft_ckpt,
        device=device,
        dtype=dtype,
    )
    return TurboRuntime(
        preset=TURBO_PRESETS[precision],
        draft=draft,
        draft_meta=meta,
        speculative_generate_wired=True,
    )


def load_turbo_engine(
    *,
    model_loader,
    loader_kwargs: dict[str, Any],
):
    """Load teacher via injected loader, attach turbo runtime + draft."""
    if not callable(model_loader):
        raise TypeError("turbo inference requires an injected model_loader.")
    if not isinstance(loader_kwargs, dict):
        raise TypeError("turbo inference requires loader_kwargs.")
    precision = str(loader_kwargs.get("precision", "fp32"))
    teacher, tokenizer = model_loader(**loader_kwargs)
    from ..engine_binding import unwrap_engine_binding

    raw_teacher, _inner = unwrap_engine_binding(teacher)
    draft_ckpt = os.environ.get(DEFAULT_DRAFT_CKPT_ENV)
    runtime = load_turbo_runtime(
        teacher=raw_teacher,
        precision=precision,
        draft_ckpt=draft_ckpt,
    )
    return InferenceEngineBinding(raw_model=raw_teacher, runtime=runtime), tokenizer
