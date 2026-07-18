"""Turbo-on-tiger runtime scaffold (§58).

Teacher = tiger PR #120 compiled decode (uniform HF + CUDA graphs).
Verify = ``CUDAGraphDecoder`` at q_len=K. Draft chain / keep-KV / rejection
land next; this module binds the preset and exposes verify helpers only until
the full Leviathan loop is wired onto ``model_generate_compiled``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .rejection import apply_temp_top_p, reject_sample_prefix
from .tiger_verify import TigerVerifySession

TURBO_PRESET_VERSION = "turbo-on-tiger-pr120-s58-scaffold-v1"
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
    preset: TurboPreset
    teacher: Any
    tokenizer: Any
    speculative_generate_wired: bool = False

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
            "turbo_speculative_generate_window": (
                "wired" if self.speculative_generate_wired else "scaffold_pending"
            ),
            "turbo_tier1_required": True,
            "note": (
                "§58 turbo-on-tiger: rejection-sampling + keep-KV + graphed draft "
                "scaffold on tiger decode. No §54 fused verify. Tip 55949274 frozen."
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


def load_turbo_engine(*, model_loader, loader_kwargs: dict[str, Any]):
    precision = str(loader_kwargs.get("precision", "fp16"))
    if precision not in TURBO_PRESETS:
        raise ValueError(f"turbo-on-tiger supports fp16|fp32, got {precision}")
    teacher = model_loader()
    # tokenizer loaded separately by caller in tiger stack; stash None until wired
    return TurboRuntime(
        preset=TURBO_PRESETS[precision],
        teacher=teacher,
        tokenizer=None,
        speculative_generate_wired=False,
    )
  raise TypeError("turbo runtime requires TurboDecodeSession")
        # Until speculative window is wired post-c_verify, fall back only when
        # explicitly allowed (smoke / teacher baseline).
        allow_fallback = (
            os.environ.get("MAPPERATORINATOR_TURBO_ALLOW_TEACHER_FALLBACK", "") == "1"
        )
        if not self.speculative_generate_wired:
            if not allow_fallback:
                raise RuntimeError(
                    "turbo-on-tiger speculative generate_window pending c_verify "
                    f"gate. draft_ckpt={self.draft_meta.get('ckpt_path')}"
                )
            from ..compiled_decode import model_generate_compiled

            gw = dict(generate_kwargs)
            return model_generate_compiled(model, tokenizer, model_kwargs, gw)

        raise RuntimeError("speculative window not yet ported on tiger base")


def load_turbo_runtime(
    *,
    teacher: Any,
    precision: str,
    draft_ckpt: str | None = None,
) -> TurboRuntime:
    if precision not in TURBO_PRESETS:
        raise ValueError(f"turbo supports precision fp16|fp32|bf16, got {precision!r}")
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
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
        speculative_generate_wired=False,
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
    teacher, tokenizer = model_loader(**loader_kwargs)
    draft_ckpt = os.environ.get(DEFAULT_DRAFT_CKPT_ENV)
    runtime = load_turbo_runtime(
        teacher=teacher,
        precision=precision,
        draft_ckpt=draft_ckpt,
    )
    # Lightweight binding: (model, runtime) tuple until tiger grows engine_binding.
    return (teacher, runtime), tokenizer
