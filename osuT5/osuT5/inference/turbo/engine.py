"""Turbo-on-tiger runtime (§58 STEP 2): strict rejection-sampling window.

Teacher = tiger PR #120 compiled decode. Verify = ``CUDAGraphDecoder`` at
q_len=K. Keep-accepted-KV via StaticCache rewind. Graphed draft chain when
armed. Tip ``55949274`` stays frozen elsewhere — no §54 fused verify.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from .draft import DEFAULT_DRAFT_CKPT_ENV, load_draft_from_ckpt
from .rejection import apply_temp_top_p, reject_sample_prefix
from .speculate import speculative_generate_window, structural_processors_enabled
from .tiger_verify import TigerVerifySession

TURBO_PRESET_VERSION = "turbo-on-tiger-pr120-s58-window-v1"
PRIMARY_GAMMA = 5


@dataclass
class TurboDecodeSession:
    accepted_tokens_total: int = 0
    verify_steps: int = 0
    draft_calls: int = 0
    teacher_cache: Any | None = None
    draft_cache: Any | None = None
    tiger_verify: TigerVerifySession | None = None
    draft_chain_runner: Any | None = None
    window_stats: list[dict[str, Any]] = field(default_factory=list)


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


def turbo_env_enabled() -> bool:
    raw = os.environ.get("MAPPERATORINATOR_TURBO", "").strip().lower()
    return raw in {"1", "true", "on", "yes"}


@dataclass
class TurboRuntime:
    preset: TurboPreset
    draft: Any
    draft_meta: dict[str, Any]
    speculative_generate_wired: bool = True

    def new_context_state(self) -> TurboDecodeSession:
        return TurboDecodeSession(tiger_verify=TigerVerifySession())

    def profile_metadata(self) -> dict[str, Any]:
        return {
            "turbo_effective_config_version": self.preset.version,
            "turbo_runtime_owner": "osuT5.osuT5.inference.turbo.engine",
            "turbo_base": "tiger14n feat/compiled-decode (PR #120)",
            "turbo_verify_path": "tiger CUDAGraphDecoder q_len=K",
            "turbo_result_class": self.preset.result_class,
            "turbo_precision": self.preset.precision,
            "turbo_gamma": self.preset.gamma,
            "turbo_temperature": self.preset.temperature,
            "turbo_top_p": self.preset.top_p,
            "turbo_draft_ckpt": self.draft_meta.get("ckpt_path"),
            "turbo_draft_init_layers": self.draft_meta.get("init_layers"),
            "turbo_structural_processors": structural_processors_enabled(),
            "turbo_speculative_generate_window": (
                "wired" if self.speculative_generate_wired else "scaffold_pending"
            ),
            "turbo_tier1_required": True,
            "note": (
                "§58 turbo-on-tiger STEP 2: rejection-sampling + keep-KV + "
                "graphed draft on tiger decode. No §54 fused verify. "
                "Tip 55949274 frozen."
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
        p = apply_temp_top_p(
            teacher_logits, self.preset.temperature, self.preset.top_p
        )
        q = apply_temp_top_p(
            draft_logits, self.preset.temperature, self.preset.top_p
        )
        return reject_sample_prefix(
            p_probs=p, q_probs=q, draft_token_ids=draft_token_ids, rng=rng
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
                    "turbo-on-tiger speculative generate_window not wired. "
                    f"draft_ckpt={self.draft_meta.get('ckpt_path')}"
                )
            from ..compiled_decode import model_generate_compiled

            return model_generate_compiled(
                model, tokenizer, model_kwargs, dict(generate_kwargs)
            )

        gamma = self.preset.gamma
        gamma_env = os.environ.get("MAPPERATORINATOR_TURBO_GAMMA", "").strip()
        if gamma_env:
            gamma = max(1, int(gamma_env))
        result, stats = speculative_generate_window(
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
        context_state.window_stats.append(dict(stats))
        return result, stats


def load_turbo_runtime(
    *,
    teacher: Any,
    precision: str,
    draft_ckpt: str | None = None,
) -> TurboRuntime:
    if precision not in TURBO_PRESETS:
        raise ValueError(f"turbo-on-tiger supports fp16|fp32, got {precision!r}")
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
    if not callable(model_loader):
        raise TypeError("turbo inference requires an injected model_loader.")
    if not isinstance(loader_kwargs, dict):
        raise TypeError("turbo inference requires loader_kwargs.")
    precision = str(loader_kwargs.get("precision", "fp16"))
    loaded = model_loader(**loader_kwargs)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        teacher, tokenizer = loaded
    else:
        teacher, tokenizer = loaded, None
    draft_ckpt = os.environ.get(DEFAULT_DRAFT_CKPT_ENV)
    runtime = load_turbo_runtime(
        teacher=teacher,
        precision=precision,
        draft_ckpt=draft_ckpt,
    )
    return (teacher, runtime), tokenizer


def attach_turbo_to_processor(processor, *, precision: str | None = None) -> TurboRuntime | None:
    """Bind a turbo runtime + session onto a tiger Processor (env-gated path).

    Returns None (and marks ``turbo_disabled``) when the draft ckpt is not
    shape-compatible with this teacher — e.g. timing models vs map draft.
    """
    prec = precision or getattr(processor, "precision", "fp16")
    try:
        runtime = load_turbo_runtime(
            teacher=processor.model,
            precision=str(prec),
            draft_ckpt=os.environ.get(DEFAULT_DRAFT_CKPT_ENV),
        )
    except Exception:
        processor.turbo_disabled = True
        return None
    loaded = int(runtime.draft_meta.get("loaded_param_count") or 0)
    if loaded < 50:
        processor.turbo_disabled = True
        return None
    processor.turbo_runtime = runtime
    processor.turbo_session = runtime.new_context_state()
    processor.turbo_disabled = False
    return runtime


__all__ = [
    "TurboDecodeSession",
    "TurboPreset",
    "TurboRuntime",
    "TURBO_PRESETS",
    "attach_turbo_to_processor",
    "load_turbo_engine",
    "load_turbo_runtime",
    "turbo_env_enabled",
]
