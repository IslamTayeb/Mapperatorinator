"""Custom decode loop for osuT5 using direct CUDA graph capture.

model.generate() rebuilds tensors and runs host-syncing logits processors every
step, which leaves this small decoder GPU-underutilised. This module replaces it
with a tight loop:
  1. prefill  — eager, processes the prompt, fills the StaticCache
  2. decode   — the single-token forward is captured into a CUDA graph and
                replayed per token. prepare_inputs runs on CPU between replays
                (cheap), logits processors + top-p sampling run in Python.

The decode graph is shape-static (batch, 1 token) and is captured ONCE then
reused across all windows. The encoder hidden state is a static buffer rewritten
per window; the StaticCache is reset (not reallocated) per window.

Output is quality-equivalent (not bit-identical) to HF generate: the sampling RNG
draw order differs, and the batched encoder precompute perturbs the encoder
hidden states at the last bit, so sampled tokens can diverge. Greedy decoding
matches HF token-for-token.

T1 safety rails (§59):
  - Fast path requires SDPA (FA2 auto-select breaks capture via dynamic KV trim).
  - Capture failure is loud and does not silently latch OFF unless explicitly allowed.
  - Graph+StaticCache entries are LRU-evicted under a VRAM budget (8–12GB cards).
  - CFG left-pad prompt masks are preserved into decode (stock HF extends the 2D
    pad mask; the prior graph path passed mask=None and attended pad K/V).

T3 compile-then-capture (pivot package):
  - Optional ``torch.compile`` of the shape-static decode step **before** CUDA
    graph capture (``fullgraph=True``, ``dynamic=False``,
    ``mode="max-autotune-no-cudagraphs"``). Never ``reduce-overhead`` near
    manual capture (§22 Inductor-in-capture lesson).
  - Warm every bucket (incl. any future turbo ``q_len``) before its capture.
  - Shape-static per-token sampling tail (monotonic hoist + temperature) is
    also compiled when enabled.
  - Opt-in via ``MAPPERATORINATOR_COMPILE_DECODE=1`` (default off so tiger
    plain-graph behavior is unchanged unless T3 jobs enable it).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers import LogitsProcessorList, TemperatureLogitsWarper

from .cache_utils import get_cache
from .server import get_eos_token_id, _build_generation_stats, build_logits_processors, model_generate
from ..event import ContextType, EventType
from .logit_processors import MonotonicTimeShiftLogitsProcessor

# T3: never use reduce-overhead next to manual CUDAGraph capture.
_COMPILE_MODE = "max-autotune-no-cudagraphs"
_COMPILE_WARMUP_ITERS = 5
_EAGER_CAPTURE_WARMUP_ITERS = 3


class CaptureError(RuntimeError):
    """CUDA graph capture failed (unsupported GPU/backend or model architecture)."""


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def compile_decode_enabled() -> bool:
    """T3 opt-in: compile-then-capture decode step + sampling tail."""
    return _env_flag("MAPPERATORINATOR_COMPILE_DECODE", default=False)


def _pin_inductor_cache_dir() -> str | None:
    """Pin TORCHINDUCTOR_CACHE_DIR per install when unset (cold-start hygiene).

    Does not ship arch/driver-bound artifacts; only uses an existing install-local
    cache root if the operator provided one via env, else a job-local default under
    TMPDIR / TORCH_EXTENSIONS_DIR parent.
    """
    existing = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if existing:
        os.makedirs(existing, exist_ok=True)
        return existing
    base = os.environ.get("TORCH_EXTENSIONS_DIR") or os.environ.get("TMPDIR") or "/tmp"
    cache = os.path.join(base, "torchinductor")
    os.makedirs(cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
    return cache


def _compile_callable(fn, *, label: str):
    """Compile ``fn`` with the binding T3 knobs; regional fallback on failure.

    NEVER uses ``reduce-overhead`` (conflicts with manual CUDAGraph capture).
    """
    _pin_inductor_cache_dir()
    kwargs: dict = {
        "fullgraph": True,
        "dynamic": False,
        "mode": _COMPILE_MODE,
    }
    try:
        compiled = torch.compile(fn, **kwargs)
        return compiled, {"mode": _COMPILE_MODE, "fullgraph": True, "fallback": None}
    except Exception as e:
        warnings.warn(
            f"T3 torch.compile({label}) fullgraph/{_COMPILE_MODE} failed ({e!r}); "
            "trying regional/default fallback (still no reduce-overhead).",
            RuntimeWarning,
            stacklevel=2,
        )
    # Regional / softer fallback — still shape-static intent, no cudagraphs-in-compile.
    try:
        compiled = torch.compile(fn, fullgraph=False, dynamic=False, mode="default")
        return compiled, {
            "mode": "default",
            "fullgraph": False,
            "fallback": "regional_or_default",
        }
    except Exception as e:
        warnings.warn(
            f"T3 torch.compile({label}) fallback also failed ({e!r}); using eager.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fn, {"mode": "eager", "fullgraph": False, "fallback": "eager"}


def _graph_cache_budget_bytes() -> int:
    """VRAM budget for captured decode graphs + StaticCaches.

    Default 2 GiB — sized for 8–12GB consumer cards that also hold weights.
    Override with MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB.
    """
    raw = os.environ.get("MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB", "2048").strip()
    try:
        mb = int(raw)
    except ValueError as e:
        raise ValueError(
            f"MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB must be an int, got {raw!r}"
        ) from e
    if mb <= 0:
        raise ValueError("MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB must be > 0")
    return mb * 1024 * 1024


def _nbytes(obj) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    total = 0
    if isinstance(obj, (list, tuple)):
        for item in obj:
            total += _nbytes(item)
        return total
    if isinstance(obj, dict):
        for item in obj.values():
            total += _nbytes(item)
        return total
    key_cache = getattr(obj, "key_cache", None)
    value_cache = getattr(obj, "value_cache", None)
    if key_cache is not None or value_cache is not None:
        total += _nbytes(key_cache)
        total += _nbytes(value_cache)
        return total
    self_attn = getattr(obj, "self_attention_cache", None)
    cross_attn = getattr(obj, "cross_attention_cache", None)
    if self_attn is not None or cross_attn is not None:
        total += _nbytes(self_attn)
        total += _nbytes(cross_attn)
        return total
    return 0


def _entry_nbytes(decoder, cache) -> int:
    total = _nbytes(cache)
    for name in (
        "static_token",
        "static_cache_pos",
        "static_enc_hidden",
        "static_logits",
        "_static_inputs",
    ):
        total += _nbytes(getattr(decoder, name, None))
    # CUDAGraph storage is opaque; charge a conservative multiple of the
    # static workspace so multi-bucket captures cannot silently OOM 8–12GB cards.
    return int(total * 1.25) + (8 << 20)


def require_sdpa_for_fast_path(model) -> str:
    """Fast-path capture requires SDPA. FA2's dynamic KV trim breaks graphs."""
    impl = None
    config = getattr(model, "config", None)
    if config is not None:
        impl = getattr(config, "_attn_implementation", None)
    if impl is None:
        transformer = getattr(model, "transformer", None)
        tconfig = getattr(transformer, "config", None) if transformer is not None else None
        if tconfig is not None:
            impl = getattr(tconfig, "_attn_implementation", None)
    impl = impl or "unknown"
    if impl != "sdpa":
        raise CaptureError(
            f"fast decoder loop requires attn_implementation='sdpa' (got {impl!r}). "
            "FA2 auto-select breaks CUDA graph capture (dynamic KV slice at "
            "modeling_varwhisper FA2 path) and previously latched the fast path OFF "
            "process-wide. Pass attn_implementation=sdpa or let inference setup "
            "force SDPA when fast_decoder_loop is enabled."
        )
    return impl


def _merge_cfg_prompt_mask(
    decoder_attention_mask,
    negative_prompt_attention_mask,
    batch_size: int,
    prompt_len: int,
    device,
):
    """Build (eff_batch, prompt_len) 1=valid mask for CFG-doubled prefill."""
    if decoder_attention_mask is None:
        cond = torch.ones((batch_size, prompt_len), dtype=torch.long, device=device)
    else:
        cond = decoder_attention_mask.to(device=device, dtype=torch.long)
        if cond.shape != (batch_size, prompt_len):
            raise ValueError(
                f"decoder_attention_mask shape {tuple(cond.shape)} != "
                f"({batch_size}, {prompt_len})"
            )
    if negative_prompt_attention_mask is None:
        uncond = torch.ones_like(cond)
    else:
        uncond = negative_prompt_attention_mask.to(device=device, dtype=torch.long)
        if uncond.shape != (batch_size, prompt_len):
            raise ValueError(
                f"negative_prompt_attention_mask shape {tuple(uncond.shape)} != "
                f"({batch_size}, {prompt_len})"
            )
    # prepare_inputs_for_generation places uncond in the first half.
    return torch.cat([uncond, cond], dim=0)


def _needs_pad_aware_decode(prompt_mask_2d: torch.Tensor | None) -> bool:
    """True when any left-pad (or other) zero exists in the prompt mask."""
    if prompt_mask_2d is None:
        return False
    return bool((prompt_mask_2d == 0).any().item())


def neutralize_dynamic_rope(model) -> int:
    """Make dynamic-RoPE decoders (e.g. Mapperatorinator-v30) CUDA-graph capturable.

    Dynamic RoPE runs `_dynamic_frequency_update` inside the forward, which does a
    data-dependent branch (`if seq_len > max_seq_len_cached`) and a buffer mutation.
    Both a device->host sync and in-capture state mutation are illegal during graph
    capture. But the branch only fires when the sequence grows past the rotary
    embedding's cached length, and that cache is sized to `max_target_positions`
    (see RoPEWhisperDecoder), which also caps generation length. So during decode
    the branch is *always* False: inv_freq is never recomputed and the update is a
    pure no-op whose only runtime effect is the illegal sync.

    Flipping rope_type to a static value skips that branch with bit-identical
    frequencies (inv_freq is untouched). Returns the number of modules neutralized.
    Only safe while the cache covers the whole decode range, so we require
    max_seq_len_cached >= max_target_positions before flipping.
    """
    max_target = getattr(getattr(model, "config", None), "max_target_positions", None)
    count = 0
    for module in model.modules():
        rope_type = getattr(module, "rope_type", None)
        if not (isinstance(rope_type, str) and "dynamic" in rope_type):
            continue
        cached = getattr(module, "max_seq_len_cached", None)
        if max_target is not None and cached is not None and cached < max_target:
            continue  # cache doesn't cover the decode range; leave dynamic behaviour
        module.rope_type = "default"
        count += 1
    return count


class _IncrementalMonotonicMask:
    """O(1)-per-step replacement for MonotonicTimeShiftLogitsProcessor.

    The stock processor re-scans the whole growing sequence every token (O(T^2)
    per window) to find the last TIME_SHIFT and whether it came after the last
    SOS. We track that as running per-row state instead: the mask forbids any
    time-shift token strictly smaller than the last emitted one (time can't go
    backwards), active only while a time-shift is the most recent structural
    event. Produces bit-identical masks to the stock processor (greedy-verified).
    """

    def __init__(self, tokenizer, device):
        self.ts_start = tokenizer.event_start[EventType.TIME_SHIFT]
        self.ts_end = tokenizer.event_end[EventType.TIME_SHIFT]
        self.sos_ids = torch.tensor(
            [tokenizer.sos_id] + list(getattr(tokenizer, "context_sos", {}).values()),
            dtype=torch.long, device=device)
        self.ts_vocab = torch.arange(self.ts_start, self.ts_end, device=device)
        self.last_ts_value = None  # (batch,) long
        self.active = None         # (batch,) bool
        self._compiled_tail = None
        self._compiled_tail_meta = None

    def init_from_prompt(self, input_ids):
        device = input_ids.device
        b, L = input_ids.shape
        is_ts = (input_ids >= self.ts_start) & (input_ids < self.ts_end)
        is_sos = torch.isin(input_ids, self.sos_ids)
        idx = torch.arange(L, device=device).expand(b, -1)
        last_ts_idx = torch.max(torch.where(is_ts, idx, -1), dim=1).values
        last_sos_idx = torch.max(torch.where(is_sos, idx, -1), dim=1).values
        rows = torch.arange(b, device=device)
        self.last_ts_value = torch.where(
            last_ts_idx != -1,
            input_ids[rows, last_ts_idx.clamp(min=0)] - self.ts_start,
            torch.zeros(b, dtype=torch.long, device=device))
        self.active = (last_ts_idx != -1) & (last_ts_idx > last_sos_idx)
        return self

    def apply(self, scores):
        # Forbid time-shift *tokens* below the last emitted one, for active rows.
        # ts_vocab holds token ids, so the threshold is ts_start + last value (the
        # id of the last time-shift), matching the stock processor exactly.
        thresh = (self.ts_start + self.last_ts_value).unsqueeze(1)
        invalid = (self.ts_vocab.unsqueeze(0) < thresh) & self.active.unsqueeze(1)
        scores[:, self.ts_start:self.ts_end].masked_fill_(invalid, -float('inf'))
        return scores

    def apply_with_temperature(self, scores, temperature: float):
        """Mono mask + uniform temperature (shape-static (B, V) sampling tail)."""
        if self._compiled_tail is not None:
            return self._compiled_tail(
                scores,
                self.ts_vocab,
                self.last_ts_value,
                self.active,
                self.ts_start,
                self.ts_end,
                float(temperature),
            )
        out = self.apply(scores)
        if temperature != 1.0:
            out = out / temperature
        return out

    def ensure_compiled_tail(self, *, batch_size: int, vocab_size: int):
        """Compile shape-static mono+temperature once (T3 sampling tail)."""
        if self._compiled_tail is not None or not compile_decode_enabled():
            return
        if not torch.cuda.is_available():
            return

        def _tail(scores, ts_vocab, last_ts_value, active, ts_start, ts_end, temperature):
            thresh = (ts_start + last_ts_value).unsqueeze(1)
            invalid = (ts_vocab.unsqueeze(0) < thresh) & active.unsqueeze(1)
            out = scores.clone()
            out[:, ts_start:ts_end].masked_fill_(invalid, -float("inf"))
            if temperature != 1.0:
                out = out / temperature
            return out

        compiled, meta = _compile_callable(_tail, label="sampling_tail")
        # Warm outside any CUDAGraph capture at the production (B, V) shape.
        device = self.ts_vocab.device
        b = max(1, int(batch_size))
        v = max(int(vocab_size), int(self.ts_end) + 1)
        warm_scores = torch.zeros((b, v), dtype=torch.float32, device=device)
        last = torch.zeros((b,), dtype=torch.long, device=device)
        active = torch.zeros((b,), dtype=torch.bool, device=device)
        for _ in range(_COMPILE_WARMUP_ITERS):
            compiled(warm_scores, self.ts_vocab, last, active, self.ts_start, self.ts_end, 1.0)
        torch.cuda.synchronize()
        self._compiled_tail = compiled
        self._compiled_tail_meta = meta

    def update(self, tok):
        t = tok.reshape(-1)
        is_ts = (t >= self.ts_start) & (t < self.ts_end)
        is_sos = torch.isin(t, self.sos_ids)
        self.last_ts_value = torch.where(is_ts, t - self.ts_start, self.last_ts_value)
        self.active = torch.where(is_ts, torch.ones_like(self.active),
                                  torch.where(is_sos, torch.zeros_like(self.active), self.active))


class CUDAGraphDecoder:
    """Captures a fixed-shape decode step into a CUDA graph.

    ``q_len=1`` is the production decode path (logits squeezed to ``(B, V)``).
    ``q_len=K>1`` is the turbo teacher-verify path on the same uniform HF+graph
    machinery (logits ``(B, K, V)``) — no fused-kernel / active-prefix overlay.
    """

    def __init__(self, model, batch_size, enc_shape, dtype, q_len: int = 1):
        self.model = model
        self.batch_size = batch_size
        self.q_len = int(q_len)
        if self.q_len < 1:
            raise ValueError("q_len must be >= 1")
        device = model.device
        self.vocab = model.config.vocab_size

        # Static input buffers
        self.static_token = torch.zeros((batch_size, self.q_len), dtype=torch.long, device=device)
        self.static_cache_pos = torch.zeros((self.q_len,), dtype=torch.long, device=device)
        # encoder hidden state as a static buffer (rewritten per window)
        self.static_enc_hidden = torch.zeros(enc_shape, dtype=dtype, device=device)
        # static output — keep q_len=1 squeezed for the existing decode loop
        if self.q_len == 1:
            self.static_logits = torch.zeros((batch_size, self.vocab),
                                             dtype=torch.float32, device=device)
        else:
            self.static_logits = torch.zeros((batch_size, self.q_len, self.vocab),
                                             dtype=torch.float32, device=device)
        self._graph = None
        self._cache = None  # set at capture; reused (reset) across windows
        self._pad_aware = False
        self._compiled_step = None
        self._compile_meta = None

    def capture(self, cache, *, pad_aware: bool = False, cache_len: int | None = None):
        """Warmup + capture the decode forward into a CUDA graph.

        T3: when ``MAPPERATORINATOR_COMPILE_DECODE=1``, Inductor-compile the
        shape-static step **before** capture (warm outside the graph), then
        capture the already-compiled kernels. Never ``reduce-overhead``.

        ``pad_aware=True`` keeps a full-length 2D decoder attention mask in the
        graph so left-pad (CFG) slots stay masked every replay. The 2D mask is
        intentional: prepare_inputs would expand it to a 4D mask frozen at the
        capture-time cache_position; passing 2D lets in-graph
        ``_update_causal_mask`` rebuild causal+pad from the live cache_position.
        """
        model = self.model
        self._cache = cache
        self._pad_aware = bool(pad_aware)
        require_sdpa_for_fast_path(model)

        # Discover model_inputs structure with a probe call (outside graph)
        enc_out = BaseModelOutput(last_hidden_state=self.static_enc_hidden)
        probe = model.prepare_inputs_for_generation(
            self.static_token,
            past_key_values=cache,
            use_cache=True,
            encoder_outputs=enc_out,
            cache_position=self.static_cache_pos,
            decoder_attention_mask=None,
        )
        self._input_keys = [k for k, v in probe.items() if isinstance(v, torch.Tensor)]
        # Static input buffers matching probe shapes/dtypes
        self._static_inputs = {}
        for k in self._input_keys:
            t = probe[k]
            self._static_inputs[k] = torch.empty_like(t)
            self._static_inputs[k].copy_(t)
        self._extra_kwargs = {k: v for k, v in probe.items() if not isinstance(v, torch.Tensor)}

        if self._pad_aware:
            if cache_len is None:
                cache_len = int(cache.get_max_cache_shape())
            device = model.device
            # Full-length 2D pad mask (1=valid). Kept 2D on purpose — see docstring.
            mask = torch.ones((self.batch_size, cache_len), dtype=torch.long, device=device)
            self._static_inputs["decoder_attention_mask"] = mask
            # Position ids for the current q_len window (refreshed every replay).
            pos = torch.zeros((self.batch_size, self.q_len), dtype=torch.long, device=device)
            self._static_inputs["decoder_position_ids"] = pos
            for key in ("decoder_attention_mask", "decoder_position_ids"):
                if key not in self._input_keys:
                    self._input_keys.append(key)

        def forward_only():
            outputs = model(**{**self._extra_kwargs, **self._static_inputs})
            if self.q_len == 1:
                self.static_logits.copy_(outputs.logits[:, -1, :].float())
            else:
                self.static_logits.copy_(outputs.logits.float())

        # Eager warmup on a side stream (required for cudagraph capture safety).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(_EAGER_CAPTURE_WARMUP_ITERS):
                forward_only()
        torch.cuda.current_stream().wait_stream(s)

        step_fn = forward_only
        if compile_decode_enabled():
            # §22: Inductor must finish compiling/warming OUTSIDE capture.
            compiled, meta = _compile_callable(forward_only, label=f"decode_q{self.q_len}")
            self._compiled_step = compiled
            self._compile_meta = meta
            s2 = torch.cuda.Stream()
            s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s2):
                for _ in range(_COMPILE_WARMUP_ITERS):
                    compiled()
            torch.cuda.current_stream().wait_stream(s2)
            torch.cuda.synchronize()
            step_fn = compiled

        # Capture (plain CUDAGraph — never reduce-overhead / cudagraphs-in-compile).
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            step_fn()

    def set_encoder_hidden(self, enc_hidden):
        """Copy a new encoder hidden state into the static buffer (per window)."""
        self.static_enc_hidden.copy_(enc_hidden)

    def replay(self, token, cache_position, *, pad_mask_2d=None, position_ids=None):
        """Write per-step inputs into the static buffers, then replay.

        For ``q_len=1`` the production hot path only refreshes
        ``decoder_input_ids`` + ``cache_position`` (mask/position_ids stay baked).
        Pad-aware / CFG graphs also refresh the 2D pad mask and position ids.
        For ``q_len=K`` verify we also refresh position ids when the probe
        exposed them, so the same graph can be replayed at different prefixes.
        """
        self._static_inputs["decoder_input_ids"].copy_(token)
        self._static_inputs["cache_position"].copy_(cache_position)
        if pad_mask_2d is not None:
            self._static_inputs["decoder_attention_mask"].copy_(pad_mask_2d)
            if position_ids is None:
                raise ValueError("pad-aware replay requires position_ids")
            self._static_inputs["decoder_position_ids"].copy_(position_ids)
        elif self.q_len > 1:
            for key in ("decoder_position_ids", "position_ids"):
                pos = self._static_inputs.get(key)
                if pos is not None and pos.shape[-1] == cache_position.numel():
                    pos.copy_(cache_position.unsqueeze(0))
        self._graph.replay()
        return self.static_logits  # (B, vocab) if q_len==1 else (B, q_len, vocab)

    def close(self):
        """Drop graph + static buffers so VRAM can be reclaimed on eviction."""
        self._graph = None
        self._cache = None
        self._static_inputs = {}
        self._compiled_step = None
        self._compile_meta = None


# Module-level LRU caches of (decoder, cache, nbytes) keyed by shape signature.
# Evicted under MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB (default 2048) shared
# across q_len=1 production graphs and q_len=K verify graphs.
_decoder_cache: OrderedDict = OrderedDict()
_k_decoder_cache: OrderedDict = OrderedDict()
_decoder_cache_bytes = 0
_k_decoder_cache_bytes = 0
_decoder_cache_evictions = 0
_k_decoder_cache_evictions = 0

# Decoder self-attention cache-length buckets. A static cache attends over its
# full allocated length every decode step, so we pick the smallest bucket that
# holds this window's prompt + generation headroom (a bigger cache = slower step).
# The last bucket is the model max.
_CACHE_BUCKETS = (384, 512, 768, 1024, 1280, 1536, 2048, 2560)
# Extra room reserved for generated tokens beyond the prompt when choosing a
# bucket. The prompt (lookback context) usually dominates the cache requirement;
# if a window generates past its bucket, model_generate_compiled retries it on a
# larger cache (see below), so this only needs to cover the common case.
_GEN_HEADROOM = 384


def _pick_bucket(prompt_len, model_max):
    """Smallest bucket >= prompt_len + headroom, capped at the model max."""
    need = min(prompt_len + _GEN_HEADROOM, model_max)
    for b in _CACHE_BUCKETS:
        if b >= need:
            return min(b, model_max)
    return model_max


def _bucket_ceil(need, model_max):
    """Smallest bucket >= need, capped at the model max (no generation headroom)."""
    for b in _CACHE_BUCKETS:
        if b >= need:
            return min(b, model_max)
    return model_max


def _cache_stats(which: str = "q1") -> dict:
    if which == "k":
        return {
            "entries": len(_k_decoder_cache),
            "bytes": _k_decoder_cache_bytes,
            "evictions": _k_decoder_cache_evictions,
            "budget_bytes": _graph_cache_budget_bytes(),
        }
    return {
        "entries": len(_decoder_cache),
        "bytes": _decoder_cache_bytes,
        "evictions": _decoder_cache_evictions,
        "budget_bytes": _graph_cache_budget_bytes(),
    }


def _total_cache_bytes() -> int:
    return _decoder_cache_bytes + _k_decoder_cache_bytes


def _evict_one(cache: OrderedDict, which: str) -> bool:
    """Evict the LRU entry from ``cache``. Returns False if empty."""
    global _decoder_cache_bytes, _decoder_cache_evictions
    global _k_decoder_cache_bytes, _k_decoder_cache_evictions
    if not cache:
        return False
    _key, (decoder, _cached, nbytes) = cache.popitem(last=False)
    try:
        decoder.close()
    except Exception:
        pass
    if which == "k":
        _k_decoder_cache_bytes = max(0, _k_decoder_cache_bytes - int(nbytes))
        _k_decoder_cache_evictions += 1
    else:
        _decoder_cache_bytes = max(0, _decoder_cache_bytes - int(nbytes))
        _decoder_cache_evictions += 1
    return True


def _evict_until_fit(cache: OrderedDict, which: str, _unused: str, need: int) -> None:
    """LRU-evict captured graphs until ``need`` bytes fit under the shared budget."""
    del _unused
    budget = _graph_cache_budget_bytes()
    # Prefer evicting from the cache we're inserting into; spill to the other.
    other = _k_decoder_cache if which == "q1" else _decoder_cache
    other_which = "k" if which == "q1" else "q1"
    while _total_cache_bytes() + need > budget:
        if cache and _evict_one(cache, which):
            continue
        if other and _evict_one(other, other_which):
            continue
        break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_decoder(model, batch_size, cfg_scale, enc_shape, dtype, cache_len, *, pad_aware: bool = False):
    global _decoder_cache_bytes
    key = (id(model), batch_size, cfg_scale, cache_len, bool(pad_aware))
    if key in _decoder_cache:
        _decoder_cache.move_to_end(key)
        decoder, cache, _nbytes = _decoder_cache[key]
        return decoder, cache
    require_sdpa_for_fast_path(model)
    cache = get_cache(model, batch_size, 1, cfg_scale, max_cache_len=cache_len)
    decoder = CUDAGraphDecoder(model, batch_size, enc_shape, dtype, q_len=1)
    try:
        decoder.capture(cache, pad_aware=pad_aware, cache_len=cache_len)
    except Exception as e:
        try:
            decoder.close()
        except Exception:
            pass
        raise CaptureError(str(e)) from e
    # After capture the cache holds garbage from the warmup forwards. The
    # decode loop calls _reset_cache + prefill before each window, which
    # restores correct state, so the capture corruption is harmless.
    nbytes = _entry_nbytes(decoder, cache)
    _evict_until_fit(_decoder_cache, "q1", "q1", nbytes)
    _decoder_cache[key] = (decoder, cache, nbytes)
    _decoder_cache_bytes += nbytes
    return decoder, cache


def get_k_decoder(model, batch_size, cfg_scale, enc_shape, dtype, cache_len, q_len: int):
    """Capture / reuse a tiger-uniform decode graph at ``q_len=K`` for turbo verify.

    Shares StaticCache sizing with the production bucketed path. Does not install
    fused kernels or active-prefix overlays — same HF forward as q_len=1.
    """
    global _k_decoder_cache_bytes
    q_len = int(q_len)
    if q_len < 1:
        raise ValueError("q_len must be >= 1")
    key = (id(model), batch_size, cfg_scale, cache_len, q_len)
    if key in _k_decoder_cache:
        _k_decoder_cache.move_to_end(key)
        decoder, cache, _nbytes = _k_decoder_cache[key]
        return decoder, cache
    require_sdpa_for_fast_path(model)
    cache = get_cache(model, batch_size, 1, cfg_scale, max_cache_len=cache_len)
    decoder = CUDAGraphDecoder(model, batch_size, enc_shape, dtype, q_len=q_len)
    try:
        decoder.capture(cache, pad_aware=False, cache_len=cache_len)
    except Exception as e:
        try:
            decoder.close()
        except Exception:
            pass
        raise CaptureError(str(e)) from e
    nbytes = _entry_nbytes(decoder, cache)
    _evict_until_fit(_k_decoder_cache, "k", "k", nbytes)
    _k_decoder_cache[key] = (decoder, cache, nbytes)
    _k_decoder_cache_bytes += nbytes
    return decoder, cache


def clear_decoder_caches() -> dict:
    """Drop all captured graphs (tests / budget scouts)."""
    global _decoder_cache_bytes, _k_decoder_cache_bytes
    while _decoder_cache:
        _key, (decoder, _c, _n) = _decoder_cache.popitem(last=False)
        try:
            decoder.close()
        except Exception:
            pass
    while _k_decoder_cache:
        _key, (decoder, _c, _n) = _k_decoder_cache.popitem(last=False)
        try:
            decoder.close()
        except Exception:
            pass
    _decoder_cache_bytes = 0
    _k_decoder_cache_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"q1": _cache_stats("q1"), "k": _cache_stats("k")}


def warm_all_decode_buckets(
    model,
    batch_size: int,
    cfg_scale: float,
    enc_shape,
    dtype,
    *,
    q_lens: tuple[int, ...] = (1,),
    pad_aware: bool = False,
    buckets: tuple[int, ...] | None = None,
) -> dict:
    """Compile+capture every cache bucket (and each ``q_len``) before measured work.

    Binding T3 rule: warm EVERY bucket — including any future turbo ``q_len`` —
    before its capture. Call from session cold-start (outside measured map).
    T4 turbo remains PARKED; pass extra ``q_lens`` only when those graphs exist.
    """
    require_sdpa_for_fast_path(model)
    neutralize_dynamic_rope(model)
    model_max = int(model.config.max_target_positions)
    if buckets is None:
        buckets = tuple(b for b in _CACHE_BUCKETS if b <= model_max)
        if model_max not in buckets:
            buckets = buckets + (model_max,)
    warmed = []
    t0 = time.perf_counter()
    seen = set()
    for cache_len in buckets:
        cl = min(int(cache_len), model_max)
        if cl in seen:
            continue
        seen.add(cl)
        for q_len in q_lens:
            q_len = int(q_len)
            if q_len < 1:
                raise ValueError("q_len must be >= 1")
            if q_len == 1:
                _get_decoder(
                    model, batch_size, cfg_scale, enc_shape, dtype, cl,
                    pad_aware=pad_aware,
                )
            else:
                get_k_decoder(model, batch_size, cfg_scale, enc_shape, dtype, cl, q_len)
            warmed.append({"cache_len": cl, "q_len": q_len, "pad_aware": bool(pad_aware)})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "warmed": warmed,
        "compile_decode": compile_decode_enabled(),
        "elapsed_s": time.perf_counter() - t0,
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "cache_stats": {"q1": _cache_stats("q1"), "k": _cache_stats("k")},
    }


def warmup_fast_decode_session(
    model,
    *,
    cfg_scale: float = 1.0,
    batch_size: int = 1,
    buckets: tuple[int, ...] | None = None,
    q_lens: tuple[int, ...] = (1,),
    raw_seq_len: int | None = None,
    encoder_batch_size: int = 16,
    verbose: bool = True,
) -> dict:
    """Cold-start: encoder cudnn warm + compile-then-capture every decode bucket.

    Counts as cold_start / load_jit proxy. Capture failures raise (no silent latch).
    """
    from .server import precompute_encoder_outputs

    if not torch.cuda.is_available():
        raise CaptureError("session warmup requires CUDA")
    neutralize_dynamic_rope(model)
    model.eval()
    if raw_seq_len is None:
        raise CaptureError(
            "warmup_fast_decode_session requires raw_seq_len "
            "(pass train src_seq_len-1 * hop_length from inference)"
        )

    t0 = time.perf_counter()
    frames = torch.zeros((max(1, encoder_batch_size), raw_seq_len), dtype=torch.float32, device="cpu")
    hidden = precompute_encoder_outputs(
        model, frames, {}, None, batch_size=encoder_batch_size,
    )
    enc_shape = (batch_size, int(hidden.shape[1]), int(hidden.shape[2]))
    encoder_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    warm = warm_all_decode_buckets(
        model,
        batch_size,
        cfg_scale,
        enc_shape,
        model.dtype,
        q_lens=q_lens,
        buckets=buckets,
    )
    capture_s = time.perf_counter() - t1
    stats = {
        "encoder_warmup_seconds": encoder_s,
        "capture_warmup_seconds": capture_s,
        "enc_shape": list(enc_shape),
        "raw_seq_len": raw_seq_len,
        **warm,
    }
    if verbose:
        print(
            f"Session warmup: encoder {encoder_s:.2f}s + "
            f"capture/compile {capture_s:.2f}s over {len(warm.get('warmed') or [])} "
            f"bucket entries (compile_decode={compile_decode_enabled()})"
        )
    return stats


def _reset_cache(cache):
    """Reset the cache buffers for reuse (avoid realloc).

    Must go through EncoderDecoderCache.reset() (not the sub-caches directly) so
    the per-layer ``is_updated`` cross-attention flags are cleared too. Otherwise
    prefill sees ``is_updated=True`` (set during graph-capture warmup) and reuses
    the just-zeroed cross-attention K/V instead of recomputing it from the new
    window's encoder outputs — producing garbage logits.
    """
    cache.reset()


# Set on the first capture failure when fallback is explicitly allowed.
# Default T1 posture: capture failure is LOUD and raises (no silent latch-off).
_capture_unsupported = False
_capture_failure_logged = False


def _loud_capture_failure(exc: BaseException) -> None:
    """Print a non-quiet capture failure (stderr + traceback). Never swallow."""
    global _capture_failure_logged
    msg = (
        "CUDA graph capture FAILED for the fast decoder loop. "
        "This previously latched the fast path OFF process-wide after a quiet "
        "print — often because attn_implementation auto-selected flash_attention_2 "
        "(FA2 dynamic KV trim is not capturable). Fix: force SDPA for the fast "
        f"path. Underlying error: {exc}"
    )
    print(msg, file=sys.stderr, flush=True)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _capture_failure_logged = True


@torch.no_grad()
def model_generate_compiled(model, tokenizer, model_kwargs, generate_kwargs):
    """Custom decode loop with CUDA-graph-captured forward.

    Expects precomputed encoder hidden states as model_kwargs['encoder_outputs']
    (a (B, L_enc, D) tensor, see server.precompute_encoder_outputs). Capture
    failure is loud (stderr + traceback). Fallback to stock generate is opt-in
    via MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1; otherwise the error raises
    and the fast path is not silently latched off.
    """
    global _capture_unsupported
    allow_fallback = _env_flag("MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK", default=False)
    if _capture_unsupported:
        if allow_fallback:
            return model_generate(model, tokenizer, model_kwargs, generate_kwargs)
        raise CaptureError(
            "fast decoder loop previously failed capture in this process; "
            "refusing silent stock fallback. Set "
            "MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1 to allow, or fix SDPA/"
            "capture prerequisites and restart the process."
        )
    # Shallow copies so a capture-failure fallback still sees the kwargs consumed below
    fallback_args = (dict(model_kwargs), dict(generate_kwargs))

    # Dynamic-RoPE models (e.g. v30) can't be graph-captured until their no-op
    # frequency update is disabled. Idempotent: a no-op once already flipped.
    neutralize_dynamic_rope(model)
    require_sdpa_for_fast_path(model)

    model_kwargs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in model_kwargs.items()}
    model_kwargs = {k: v.to(model.dtype) if k != "inputs" and isinstance(v, torch.Tensor)
                    and v.dtype == torch.float32 else v for k, v in model_kwargs.items()}
    encoder_outputs = BaseModelOutput(last_hidden_state=model_kwargs.pop('encoder_outputs'))

    precision = generate_kwargs.pop('precision', 'fp32')
    cfg_scale = generate_kwargs.pop('cfg_scale', 1.0)
    timeshift_bias = generate_kwargs.pop('timeshift_bias', 0)
    types_first = generate_kwargs.pop('types_first', False)
    temperature = generate_kwargs.pop('temperature', 1.0)
    timing_temperature = generate_kwargs.pop('timing_temperature', temperature)
    mania_column_temperature = generate_kwargs.pop('mania_column_temperature', temperature)
    taiko_hit_temperature = generate_kwargs.pop('taiko_hit_temperature', temperature)
    lookback_time = generate_kwargs.pop('lookback_time', 0.0)
    lookahead_time = generate_kwargs.pop('lookahead_time', 0.0)
    context_type = generate_kwargs.pop('context_type', None)
    do_sample = generate_kwargs.pop('do_sample', True)
    top_p = generate_kwargs.pop('top_p', 0.95)
    top_k = generate_kwargs.pop('top_k', 0)
    max_length = generate_kwargs.pop('max_length', 2560)
    if context_type is not None:
        context_type = ContextType(context_type)

    decoder_input_ids = model_kwargs["decoder_input_ids"]
    decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
    negative_prompt = model_kwargs.get("negative_prompt")
    negative_prompt_attention_mask = model_kwargs.get("negative_prompt_attention_mask")
    pad_token_id = getattr(tokenizer, 'pad_id', None)
    device = model.device

    use_cfg = cfg_scale > 1 and negative_prompt is not None
    eff_batch = decoder_input_ids.shape[0] * (2 if use_cfg else 1)
    eos_token_ids = get_eos_token_id(tokenizer, lookback_time=lookback_time,
                                     lookahead_time=lookahead_time, context_type=context_type)
    logits_processor_list = build_logits_processors(
        tokenizer, cfg_scale, timeshift_bias, types_first, temperature,
        timing_temperature, mania_column_temperature, taiko_hit_temperature,
        lookback_time, device,
    )
    # Hoist MonotonicTimeShift out of the per-step list into an O(1) incremental
    # masker. Only safe without CFG (CFG merges the 2-stream logits before the
    # monotonic mask and the merged-batch state bookkeeping differs); with CFG we
    # keep the stock O(T^2) processor.
    # T3: also hoist uniform TemperatureLogitsWarper into the compiled mono+temp
    # sampling tail when present (v32: monotonic hoist + temperature only).
    mono_mask = None
    hoist_temperature = None
    if cfg_scale <= 1:
        rest = LogitsProcessorList([p for p in logits_processor_list
                                    if not isinstance(p, MonotonicTimeShiftLogitsProcessor)])
        if len(rest) != len(logits_processor_list):  # one was actually removed
            mono_mask = _IncrementalMonotonicMask(tokenizer, device)
            logits_processor_list = rest
        if (
            compile_decode_enabled()
            and mono_mask is not None
            and len(logits_processor_list) == 1
            and isinstance(logits_processor_list[0], TemperatureLogitsWarper)
        ):
            hoist_temperature = float(logits_processor_list[0].temperature)
            logits_processor_list = LogitsProcessorList()
            mono_mask.ensure_compiled_tail(
                batch_size=decoder_input_ids.shape[0],
                vocab_size=int(model.config.vocab_size),
            )

    prompt_len = decoder_input_ids.shape[1]
    b0 = decoder_input_ids.shape[0]
    model_max = model.config.max_target_positions
    enc_hidden = encoder_outputs.last_hidden_state  # (B, L_enc, D)
    eos_t = torch.tensor(eos_token_ids, device=device, dtype=torch.long)
    # The true generation limit, matching the non-compiled path (which caps at
    # max_length and its full-size cache). We must never stop a window short of
    # this just because a bucket was too small.
    hard_cap = min(max_length, model_max)

    # CFG left-pad: build the doubled prompt mask. Pad-aware decode is enabled
    # whenever any zero exists (unequal cond/uncond lengths after pad_prompts).
    prompt_mask_2d = None
    if use_cfg:
        prompt_mask_2d = _merge_cfg_prompt_mask(
            decoder_attention_mask,
            negative_prompt_attention_mask,
            b0,
            prompt_len,
            device,
        )
    elif decoder_attention_mask is not None and _needs_pad_aware_decode(
        decoder_attention_mask.to(device=device, dtype=torch.long)
    ):
        prompt_mask_2d = decoder_attention_mask.to(device=device, dtype=torch.long)
    pad_aware = _needs_pad_aware_decode(prompt_mask_2d)

    def _run_window(cache_len):
        """Prefill + captured decode with a given static-cache length.

        Returns (generated_tokens, truncated) where ``truncated`` means the loop
        stopped because it filled the cache (``win_max``) rather than emitting EOS.
        A static cache attends over its full allocated length every step, so a right-sized
        cache is cheaper; we start small and only grow on truncation (below).
        """
        win_max = min(hard_cap, cache_len)
        decoder, cache = _get_decoder(
            model, eff_batch, cfg_scale, enc_hidden.shape, model.dtype, cache_len,
            pad_aware=pad_aware,
        )
        _reset_cache(cache)                    # fresh generation for this window
        decoder.set_encoder_hidden(enc_hidden)  # write this window's encoder state
        if mono_mask is not None:
            mono_mask.init_from_prompt(decoder_input_ids)  # reset per (re)attempt

        # Full-length 2D pad mask for pad-aware graphs (prompt zeros preserved;
        # generated positions filled with 1 as we go — matches stock HF extend).
        full_pad_mask = None
        if pad_aware:
            full_pad_mask = torch.zeros(
                (eff_batch, cache_len), dtype=torch.long, device=device
            )
            full_pad_mask[:, :prompt_len] = prompt_mask_2d

        # 1. PREFILL (eager) — process the whole prompt, fill the cache. Must use
        # the SAME cache object the graph references so decode steps see it.
        cache_position = torch.arange(prompt_len, device=device)
        prefill_inputs = model.prepare_inputs_for_generation(
            decoder_input_ids,
            past_key_values=cache,
            use_cache=True,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            negative_prompt=negative_prompt,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            cache_position=cache_position,
        )
        prefill_out = model(**prefill_inputs)
        last_logits = prefill_out.logits[:, -1, :].float()  # (B, vocab)

        # Preallocated id buffer: O(1) append + O(1) view of the sequence so far,
        # replacing the per-step growing torch.cat (O(T^2)/window). Processors that
        # still need history (types_first) read a cheap slice view.
        id_buffer = torch.empty((b0, win_max), dtype=torch.long, device=device)
        id_buffer[:, :prompt_len] = decoder_input_ids

        generated = []
        cur_len = prompt_len
        truncated = True
        # 2. DECODE LOOP — `last_logits` holds the logits for the current position
        # (prefill output on entry, else the previous replay). Sample, check EOS,
        # then replay for the next position. The EOS token is kept (appended) so
        # downstream trimming matches HF.
        while cur_len < win_max:
            if mono_mask is not None and hoist_temperature is not None:
                last_logits = mono_mask.apply_with_temperature(last_logits, hoist_temperature)
                proc_logits = last_logits
            else:
                if mono_mask is not None:
                    last_logits = mono_mask.apply(last_logits)
                proc_logits = logits_processor_list(id_buffer[:, :cur_len], last_logits)
            # Temperature is already applied by logits_processor_list (or the
            # compiled mono+temp tail), so _sample must NOT re-apply it — pass 1.0.
            # It still handles top-p / top-k, which the processor list omits
            # (matching model.generate's sampling step).
            next_token = _sample(proc_logits, do_sample, top_p, top_k, 1.0)

            id_buffer[:, cur_len] = next_token.reshape(b0)
            generated.append(next_token)
            if mono_mask is not None:
                mono_mask.update(next_token)
            cur_len += 1

            if torch.isin(next_token, eos_t).any().item():
                truncated = False
                break
            cache_position = torch.tensor([cur_len - 1], device=device, dtype=torch.long)
            replay_token = next_token
            if use_cfg and replay_token.shape[0] == b0:
                # CFG graph is (2B, 1); sample returns (B, 1) — broadcast both streams.
                replay_token = replay_token.repeat(2, 1)
            if pad_aware:
                # New token is always valid on both CFG streams (same sampled id).
                full_pad_mask[:, cur_len - 1] = 1
                # RoPE positions follow mask cumsum (not raw cache index) so left
                # pads do not shift generated positions — matches stock HF.
                pos_full = (full_pad_mask.cumsum(-1) - 1).clamp(min=0)
                position_ids = pos_full[:, cur_len - 1 : cur_len].contiguous()
                last_logits = decoder.replay(
                    replay_token, cache_position,
                    pad_mask_2d=full_pad_mask, position_ids=position_ids,
                )
            else:
                last_logits = decoder.replay(replay_token, cache_position)
        return generated, truncated

    start_time = time.perf_counter()
    try:
        # Start with a small bucket sized for the common short-generation case.
        cache_len = _pick_bucket(prompt_len, model_max)
        generated, truncated = _run_window(cache_len)
        # If the window filled its bucket without hitting EOS while the model was still
        # allowed to generate (bucket < hard_cap), the bucket was too small and we'd
        # otherwise emit a truncated sequence (seen with small lookahead, where windows
        # generate far more than the default headroom). Retry once with a cache big
        # enough for the full budget so compiled output matches the non-compiled path.
        if truncated and cache_len < hard_cap:
            cache_len = _bucket_ceil(hard_cap, model_max)
            generated, truncated = _run_window(cache_len)
    except CaptureError as e:
        _loud_capture_failure(e)
        _capture_unsupported = True
        if allow_fallback:
            print(
                "MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1: falling back to stock "
                f"generate after capture failure. ({e})",
                file=sys.stderr,
                flush=True,
            )
            return model_generate(model, tokenizer, *fallback_args)
        raise
    elapsed = time.perf_counter() - start_time

    gen_tensor = torch.cat(generated, dim=1) if generated else torch.empty(
        (decoder_input_ids.shape[0], 0), dtype=torch.long, device=device)
    result = torch.cat([decoder_input_ids, gen_tensor], dim=1).cpu()
    stats = _build_generation_stats(result, model_kwargs, pad_token_id, elapsed)
    return result, stats


def _sample(logits, do_sample, top_p, top_k, temperature):
    """Top-p (nucleus) sampling. Returns (B, 1) token tensor."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k)
        thresh = vals[..., -1, None]
        logits = torch.where(logits < thresh, torch.full_like(logits, -float('inf')), logits)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float('inf'))
        logits = torch.full_like(logits, -float('inf'))
        logits.scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    flat = probs.view(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1)
    return sampled.view(probs.size(0), -1)
