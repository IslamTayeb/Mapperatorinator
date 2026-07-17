"""§49 W-DG: γ-step draft chain in ONE CUDA graph (sample + embed feedback).

Ports §42 StaticCache / q1 fastpath and chains γ decode steps inside a single
captured graph with Philox-safe in-graph multinomial (§29c pattern) and
token→embedding feedback via ``decoder_input_ids``.

Under keep-KV, the γ-th forward's logits stay in ``logits_ws`` for the next
cycle's first sample (the previously discarded tail).

Turbo-only. Persistent chain-graph cache lives on the session (cross-window).
Not a 500 / TIER1 ship claim.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..optimized.single.decode_loop import (
    _bucketed_prefix_length,
    _clone_static_graph_inputs,
    _copy_static_graph_inputs,
    _cuda_graph_signature,
)
from ..optimized.single.runtime_context import active_prefix_self_attention_context
from .draft_fastpath import (
    ACTIVE_PREFIX_BUCKET_SIZE,
    DraftFastpathRunner,
    _align_decoder_mask,
    _strip_audio_inputs,
)
from .rejection import apply_temp_top_p

DRAFT_CHAIN_GRAPH_ENV = "MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH"
DEFAULT_GAMMA = 3
DEFAULT_TOP_P = 0.9
GATE_MS_TOTAL = 1.2
KILL_MS_TOTAL = 2.0


def draft_chain_graph_enabled() -> bool:
    raw = os.environ.get(DRAFT_CHAIN_GRAPH_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


@dataclass
class DraftChainGraphStats:
    chain_captures: int = 0
    chain_replays: int = 0
    capture_seconds: float = 0.0
    last_chain_ms: float = 0.0
    hit_counters: dict[str, int] = field(
        default_factory=lambda: {
            "draft_chain_graph_replay": 0,
            "draft_chain_graph_capture": 0,
            "draft_chain_eager_fallback": 0,
        }
    )


def _vocab_size(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("vocab_size", "decoder_vocab_size"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    # Fall back to lm_head out features.
    lm = getattr(model, "lm_head", None) or getattr(
        getattr(model, "transformer", None), "lm_head", None
    )
    if lm is not None and hasattr(lm, "out_features"):
        return int(lm.out_features)
    raise RuntimeError("cannot resolve draft vocab size")


def _zero_self_cache_tail(cache, start: int) -> None:
    """Zero StaticCache self-attn slots from ``start`` (keep-KV rewind helper)."""
    self_cache = cache.self_attention_cache
    layers = getattr(self_cache, "layers", None) or []
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            # Older HF StaticCache layout.
            keys = getattr(self_cache, "key_cache", None)
            values = getattr(self_cache, "value_cache", None)
            if isinstance(keys, list):
                for k, v in zip(keys, values):
                    if k is not None and k.numel():
                        k[..., int(start) :, :].zero_()
                    if v is not None and v.numel():
                        v[..., int(start) :, :].zero_()
                break
            continue
        if keys is not None and keys.numel():
            keys[..., int(start) :, :].zero_()
        if values is not None and values.numel():
            values[..., int(start) :, :].zero_()
    if hasattr(self_cache, "cumulative_length"):
        try:
            self_cache.cumulative_length = int(start)
        except Exception:
            pass


class DraftChainGraphRunner:
    """γ-step draft chain: one CUDA graph = sample→embed→forward × γ."""

    def __init__(
        self,
        model,
        *,
        dtype: torch.dtype,
        gamma: int = DEFAULT_GAMMA,
        temperature: float = 0.9,
        top_p: float = DEFAULT_TOP_P,
        bucket_size: int = ACTIVE_PREFIX_BUCKET_SIZE,
        enable_native_kernels: bool = True,
        cuda_graph_warmup: int = 2,
    ):
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        self.model = model
        self.dtype = dtype
        self.gamma = int(gamma)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.bucket_size = int(bucket_size)
        self.cuda_graph_warmup = int(cuda_graph_warmup)
        self.fastpath = DraftFastpathRunner(
            model,
            dtype=dtype,
            bucket_size=bucket_size,
            enable_native_kernels=enable_native_kernels,
            cuda_graph=False,  # chain owns the graph
            cuda_graph_warmup=cuda_graph_warmup,
        )
        self.stats = DraftChainGraphStats()
        self._vocab = _vocab_size(model)
        self._device = model.device
        # Persistent across windows: stored on the ProductionDecodeSession.
        if not hasattr(self.fastpath.session, "draft_chain_graph_cache"):
            self.fastpath.session.draft_chain_graph_cache = {}
        self._logits_ws: torch.Tensor | None = None
        self._logits_steps: torch.Tensor | None = None  # [γ, V] after each forward
        self._tokens_ws: torch.Tensor | None = None
        self._token_in: torch.Tensor | None = None
        self._cache_pos: torch.Tensor | None = None
        self._seed_logits: torch.Tensor | None = None

    @property
    def session(self):
        return self.fastpath.session

    @property
    def chain_graph_cache(self) -> dict:
        return self.fastpath.session.draft_chain_graph_cache

    def bind(self, *, model_kwargs: dict[str, Any], prompt_ids: torch.LongTensor) -> None:
        self.fastpath.bind(model_kwargs=model_kwargs, prompt_ids=prompt_ids)
        self._device = prompt_ids.device
        self._ensure_workspaces()

    def _ensure_workspaces(self) -> None:
        device = self._device
        if self._logits_ws is None or self._logits_ws.device != device:
            self._logits_ws = torch.empty(
                (1, self._vocab), device=device, dtype=torch.float32
            )
            self._logits_steps = torch.empty(
                (self.gamma, self._vocab), device=device, dtype=torch.float32
            )
            self._tokens_ws = torch.zeros(
                (self.gamma,), device=device, dtype=torch.long
            )
            self._token_in = torch.zeros((1, 1), device=device, dtype=torch.long)
            self._cache_pos = torch.zeros((1,), device=device, dtype=torch.long)
            self._seed_logits = torch.empty(
                (1, self._vocab), device=device, dtype=torch.float32
            )

    def prefill(self) -> torch.Tensor:
        logits = self.fastpath.prefill()
        self._ensure_workspaces()
        assert self._seed_logits is not None and self._logits_ws is not None
        self._seed_logits.copy_(logits.view(1, -1))
        self._logits_ws.copy_(self._seed_logits)
        return logits

    def reset_cache(self) -> None:
        self.fastpath.reset_cache()

    def _build_static_decode_inputs(self, length: int) -> dict[str, Any]:
        """Prepare Q=1 decode inputs at ``length`` (full ids; HF slices via cache)."""
        assert self.fastpath._input_ids is not None
        assert self.fastpath._model_kwargs is not None
        mk = self.fastpath._model_kwargs
        mk.pop("cache_position", None)
        _align_decoder_mask(mk, length=int(length), device=self._device)
        # Ensure ids length matches.
        ids = self.fastpath._input_ids
        if int(ids.shape[1]) != int(length):
            raise RuntimeError(
                f"input_ids len {ids.shape[1]} != requested chain start {length}"
            )
        model_inputs = self.model.prepare_inputs_for_generation(ids, **mk)
        model_inputs = _strip_audio_inputs(model_inputs)
        return model_inputs

    def _chain_signature(self, active_prefix_length: int, model_inputs: dict) -> tuple:
        base = _cuda_graph_signature(active_prefix_length, model_inputs)
        return ("draft_chain", self.gamma, self.temperature, self.top_p, base)

    def _sample_into(
        self,
        logits_ws: torch.Tensor,
        token_out: torch.Tensor,
        token_in: torch.Tensor,
    ) -> None:
        """Philox-graphable temp+top-p sample (matches offline probe / rejection q)."""
        if self.temperature <= 1e-5:
            token_out.copy_(torch.argmax(logits_ws, dim=-1).view_as(token_out))
        else:
            probs = apply_temp_top_p(logits_ws, self.temperature, self.top_p)
            sampled = torch.multinomial(probs, num_samples=1).view_as(token_out)
            token_out.copy_(sampled)
        token_in.copy_(token_out.view(1, 1))

    def _run_chain_body(
        self,
        *,
        static_inputs: dict[str, Any],
        active_prefix_length: int,
        logits_ws: torch.Tensor,
        logits_steps: torch.Tensor,
        tokens_ws: torch.Tensor,
        token_in: torch.Tensor,
        cache_pos: torch.Tensor,
    ) -> None:
        """Unrolled γ: sample → write decoder_input_ids → forward → logits feedback."""
        for i in range(self.gamma):
            self._sample_into(logits_ws, tokens_ws[i : i + 1], token_in)
            # Embedding feedback: token id → model embed path via decoder_input_ids.
            if "decoder_input_ids" in static_inputs:
                static_inputs["decoder_input_ids"].copy_(token_in)
            elif "input_ids" in static_inputs:
                static_inputs["input_ids"].copy_(token_in)
            else:
                raise RuntimeError("static decode inputs missing decoder_input_ids")
            if "cache_position" in static_inputs:
                static_inputs["cache_position"].copy_(cache_pos)
            with active_prefix_self_attention_context(active_prefix_length):
                outputs = self.model(**static_inputs, return_dict=True)
            logits_ws.copy_(outputs.logits[:, -1, :].float())
            logits_steps[i].copy_(logits_ws.view(-1))
            cache_pos.add_(1)

    def _capture_chain_graph(
        self,
        *,
        static_inputs: dict[str, Any],
        active_prefix_length: int,
        start_len: int,
    ) -> dict[str, Any]:
        assert self._logits_ws is not None
        assert self._logits_steps is not None
        assert self._tokens_ws is not None
        assert self._token_in is not None
        assert self._cache_pos is not None
        assert self._seed_logits is not None

        device = self._logits_ws.device
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        generator = torch.cuda.default_generators[device_index]

        # Side-stream warmup so capture-time Philox offset is not the first sample.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(max(int(self.cuda_graph_warmup), 1)):
                self._logits_ws.copy_(self._seed_logits)
                self._cache_pos.fill_(int(start_len))
                self._run_chain_body(
                    static_inputs=static_inputs,
                    active_prefix_length=active_prefix_length,
                    logits_ws=self._logits_ws,
                    logits_steps=self._logits_steps,
                    tokens_ws=self._tokens_ws,
                    token_in=self._token_in,
                    cache_pos=self._cache_pos,
                )
                # Rewind KV + position so the next warmup/capture starts at start_len.
                if self.fastpath._cache is not None:
                    _zero_self_cache_tail(self.fastpath._cache, int(start_len))
                self._cache_pos.fill_(int(start_len))
                self._logits_ws.copy_(self._seed_logits)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        rng_before = torch.cuda.get_rng_state(device)
        self._logits_ws.copy_(self._seed_logits)
        self._cache_pos.fill_(int(start_len))
        if self.fastpath._cache is not None:
            _zero_self_cache_tail(self.fastpath._cache, int(start_len))

        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(generator)
        with torch.cuda.graph(graph):
            self._run_chain_body(
                static_inputs=static_inputs,
                active_prefix_length=active_prefix_length,
                logits_ws=self._logits_ws,
                logits_steps=self._logits_steps,
                tokens_ws=self._tokens_ws,
                token_in=self._token_in,
                cache_pos=self._cache_pos,
            )
        # Discard capture-time Philox advance (§29c).
        torch.cuda.set_rng_state(rng_before, device)
        # Rewind KV dirtied by capture so first replay is the real first use.
        if self.fastpath._cache is not None:
            _zero_self_cache_tail(self.fastpath._cache, int(start_len))
        self._cache_pos.fill_(int(start_len))
        self._logits_ws.copy_(self._seed_logits)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - started

        entry = {
            "graph": graph,
            "static_inputs": static_inputs,
            "active_prefix_length": int(active_prefix_length),
            "start_len": int(start_len),
            "gamma": int(self.gamma),
            "capture_seconds": float(capture_seconds),
            "logits_ws": self._logits_ws,
            "logits_steps": self._logits_steps,
            "tokens_ws": self._tokens_ws,
            "token_in": self._token_in,
            "cache_pos": self._cache_pos,
            "seed_logits": self._seed_logits,
        }
        self.stats.chain_captures += 1
        self.stats.capture_seconds += float(capture_seconds)
        self.stats.hit_counters["draft_chain_graph_capture"] += 1
        return entry

    def ensure_chain_graph(self, *, start_len: int | None = None) -> dict[str, Any]:
        """Capture or fetch persistent γ-chain graph for the current bucket."""
        if self.fastpath._input_ids is None or self.fastpath._model_kwargs is None:
            raise RuntimeError("bind()/prefill() required")
        self._ensure_workspaces()
        cur = int(self.fastpath._cur_len if start_len is None else start_len)
        # Active prefix must cover start + γ writes.
        cover = cur + self.gamma
        bucket = _bucketed_prefix_length(
            cover,
            self.bucket_size,
            self.fastpath._max_cache_len,
        )
        model_inputs = self._build_static_decode_inputs(cur)
        # Force Q=1 token buffer shapes for in-graph feedback.
        if "decoder_input_ids" in model_inputs:
            tok = model_inputs["decoder_input_ids"]
            if tok.ndim != 2 or tok.shape[-1] != 1:
                model_inputs["decoder_input_ids"] = tok[:, -1:].contiguous()
        if "cache_position" not in model_inputs or model_inputs["cache_position"] is None:
            model_inputs["cache_position"] = torch.tensor(
                [cur], device=self._device, dtype=torch.long
            )
        else:
            cp = model_inputs["cache_position"]
            if cp.numel() != 1:
                model_inputs["cache_position"] = torch.tensor(
                    [cur], device=self._device, dtype=torch.long
                )

        key = self._chain_signature(bucket, model_inputs)
        entry = self.chain_graph_cache.get(key)
        if entry is not None:
            return entry

        static_inputs = _clone_static_graph_inputs(model_inputs)
        # Align workspaces referenced by capture.
        assert self._cache_pos is not None
        self._cache_pos.fill_(cur)
        entry = self._capture_chain_graph(
            static_inputs=static_inputs,
            active_prefix_length=bucket,
            start_len=cur,
        )
        self.chain_graph_cache[key] = entry
        return entry

    @torch.no_grad()
    def replay_chain(
        self,
        *,
        seed_logits: torch.Tensor | None = None,
    ) -> tuple[list[int], torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Replay γ-chain graph for the speculative loop.

        Returns:
            draft_ids, q_probs[γ,V], logits_after[i]=logits after draft i,
            logits_after_γ (keep-KV next-cycle seed).
        """
        if self.fastpath._input_ids is None:
            raise RuntimeError("bind()/prefill() required")
        self._ensure_workspaces()
        assert self._seed_logits is not None
        assert self._logits_ws is not None
        assert self._logits_steps is not None
        assert self._tokens_ws is not None
        assert self._cache_pos is not None

        if seed_logits is not None:
            self._seed_logits.copy_(seed_logits.view(1, -1))
        # else: leave seed_logits as last keep-KV / prefill value

        start_len = int(self.fastpath._cur_len)
        entry = self.ensure_chain_graph(start_len=start_len)

        self._logits_ws.copy_(self._seed_logits)
        self._cache_pos.fill_(start_len)
        # Refresh static token/pos buffers from live prepare (same shapes).
        live = self._build_static_decode_inputs(start_len)
        if "decoder_input_ids" in live:
            live["decoder_input_ids"] = live["decoder_input_ids"][:, -1:].contiguous()
        if "cache_position" in live and live["cache_position"].numel() != 1:
            live["cache_position"] = torch.tensor(
                [start_len], device=self._device, dtype=torch.long
            )
        _copy_static_graph_inputs(entry["static_inputs"], live)

        with self.fastpath._profile_context():
            entry["graph"].replay()
        self.stats.chain_replays += 1
        self.stats.hit_counters["draft_chain_graph_replay"] += 1

        # Advance logical sequence by γ drafted tokens (host bookkeeping).
        tokens = self._tokens_ws.clone()
        new_ids = tokens.view(1, -1)
        self.fastpath._input_ids = torch.cat(
            [self.fastpath._input_ids, new_ids], dim=-1
        )
        self.fastpath._cur_len = int(self.fastpath._input_ids.shape[1])
        _align_decoder_mask(
            self.fastpath._model_kwargs,
            length=self.fastpath._cur_len,
            device=self._device,
        )
        # q_probs must match the proposal distribution used in-graph (temp+top-p).
        pre = torch.empty(
            (self.gamma, self._vocab),
            device=self._device,
            dtype=torch.float32,
        )
        pre[0].copy_(self._seed_logits.view(-1))
        if self.gamma > 1:
            pre[1:].copy_(self._logits_steps[:-1])
        if self.temperature <= 1e-5:
            # Greedy chain: one-hot at argmax (rejection path rarely uses this).
            q_probs = torch.zeros_like(pre)
            q_probs.scatter_(1, pre.argmax(dim=-1, keepdim=True), 1.0)
        else:
            q_probs = apply_temp_top_p(pre, self.temperature, self.top_p)
        logits_after = [self._logits_steps[i].clone() for i in range(self.gamma)]
        # Keep-KV: γ-th forward logits become next cycle seed.
        self._seed_logits.copy_(self._logits_ws)
        draft_ids = [int(t) for t in tokens.tolist()]
        return draft_ids, q_probs, logits_after, self._logits_ws.squeeze(0).clone()

    def rewind_keep_kv(self, keep_len: int) -> None:
        """O(1)-ish keep-KV rollback: zero self-KV past keep_len, trim ids."""
        if self.fastpath._cache is not None:
            _zero_self_cache_tail(self.fastpath._cache, int(keep_len))
        if self.fastpath._input_ids is not None:
            self.fastpath._input_ids = self.fastpath._input_ids[:, : int(keep_len)]
            self.fastpath._cur_len = int(keep_len)
            if self.fastpath._model_kwargs is not None:
                _align_decoder_mask(
                    self.fastpath._model_kwargs,
                    length=int(keep_len),
                    device=self._device,
                )

    def profile_metadata(self) -> dict[str, Any]:
        meta = self.fastpath.profile_metadata()
        meta.update(
            {
                "turbo_draft_chain_graph": True,
                "turbo_draft_chain_gamma": self.gamma,
                "turbo_draft_chain_temperature": self.temperature,
                "turbo_draft_chain_top_p": self.top_p,
                "turbo_draft_chain_captures": self.stats.chain_captures,
                "turbo_draft_chain_replays": self.stats.chain_replays,
                "turbo_draft_chain_hit_counters": dict(self.stats.hit_counters),
                "turbo_draft_chain_persistent_cache_size": len(self.chain_graph_cache),
            }
        )
        return meta


@torch.no_grad()
def time_chain_graph_ms(
    runner: DraftChainGraphRunner,
    *,
    prefix_length: int,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    """Microbench pure γ-chain CUDA graph replay (ms for all γ drafts).

    Host rewind / id bookkeeping is excluded from the timed region — only
    ``graph.replay()`` (sample+embed+forward ×γ) is measured, matching §42's
    graph-replay microbench style. KV is rewound between iters so each replay
    starts at ``prefix_length``.
    """
    if runner.fastpath._input_ids is None:
        raise RuntimeError("runner must be bound + prefills before timing")

    # Grow to prefix with eager fastpath decode (no chain graph).
    while runner.fastpath._cur_len < int(prefix_length):
        runner.fastpath.decode_token(1)
    seed = runner.fastpath.decode_token(1)
    runner.rewind_keep_kv(int(prefix_length))
    runner.fastpath._cur_len = int(prefix_length)
    if runner.fastpath._input_ids is not None:
        runner.fastpath._input_ids = runner.fastpath._input_ids[:, : int(prefix_length)]
    runner._ensure_workspaces()
    assert runner._seed_logits is not None
    assert runner._logits_ws is not None
    assert runner._cache_pos is not None
    runner._seed_logits.copy_(seed.view(1, -1))

    entry = runner.ensure_chain_graph(start_len=int(prefix_length))
    start_len = int(prefix_length)

    def _arm_for_replay() -> None:
        runner.rewind_keep_kv(start_len)
        runner.fastpath._cur_len = start_len
        if runner.fastpath._input_ids is not None:
            runner.fastpath._input_ids = runner.fastpath._input_ids[:, :start_len]
        runner._seed_logits.copy_(seed.view(1, -1))
        runner._logits_ws.copy_(runner._seed_logits)
        runner._cache_pos.fill_(start_len)
        live = runner._build_static_decode_inputs(start_len)
        if "decoder_input_ids" in live:
            live["decoder_input_ids"] = live["decoder_input_ids"][:, -1:].contiguous()
        if "cache_position" in live and live["cache_position"].numel() != 1:
            live["cache_position"] = torch.tensor(
                [start_len], device=runner._device, dtype=torch.long
            )
        _copy_static_graph_inputs(entry["static_inputs"], live)

    for _ in range(max(int(warmup), 0)):
        _arm_for_replay()
        with runner.fastpath._profile_context():
            entry["graph"].replay()
        runner.stats.chain_replays += 1
        runner.stats.hit_counters["draft_chain_graph_replay"] += 1

    # Timed region: pure graph replay only (rewind/arm outside events).
    times_ms: list[float] = []
    for _ in range(int(iters)):
        _arm_for_replay()
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        with runner.fastpath._profile_context():
            start_evt.record()
            entry["graph"].replay()
            end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(float(start_evt.elapsed_time(end_evt)))
        runner.stats.chain_replays += 1
        runner.stats.hit_counters["draft_chain_graph_replay"] += 1

    times_ms.sort()
    ms_chain = float(times_ms[len(times_ms) // 2])
    ms_mean = float(sum(times_ms) / len(times_ms))
    runner.stats.last_chain_ms = ms_chain
    bucket = _bucketed_prefix_length(
        int(prefix_length) + runner.gamma,
        runner.bucket_size,
        runner.fastpath._max_cache_len,
    )
    return {
        "prefix_length": int(prefix_length),
        "bucket_prefix_length": int(bucket),
        "gamma": int(runner.gamma),
        "ms_per_chain": ms_chain,
        "ms_per_chain_mean": ms_mean,
        "ms_per_chain_min": float(times_ms[0]),
        "ms_per_chain_max": float(times_ms[-1]),
        "ms_per_token": ms_chain / float(runner.gamma),
        "gate_ms_total": GATE_MS_TOTAL,
        "kill_ms_total": KILL_MS_TOTAL,
        "gate_pass": bool(ms_chain <= GATE_MS_TOTAL),
        "kill": bool(ms_chain > KILL_MS_TOTAL),
        "timing_mode": "pure_graph_replay",
        "warmup": int(warmup),
        "iters": int(iters),
        "chain_captures": int(runner.stats.chain_captures),
        "chain_replays": int(runner.stats.chain_replays),
        "hit_counters": dict(runner.stats.hit_counters),
        "persistent_cache_size": len(runner.chain_graph_cache),
        "dispatch_counts": dict(runner.fastpath.stats.dispatch_counts),
    }
