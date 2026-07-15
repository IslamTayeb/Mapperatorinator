"""Opt-in exact K-step CUDA-graph sampling feasibility probe.

This module is deliberately not imported by :mod:`optimized.scout`.  It tests
one narrow prerequisite for an exact multi-token runtime: one ordinary PyTorch
CUDA graph contains four *distinct* ``torch.multinomial`` calls, then a
terminal block is rolled back to the first EOS without changing the token,
cache, processor-state, or default CUDA-generator state that eager execution
would have produced.

The probe is not an inference runtime.  It uses toy FP32 decode state and has no
selector or default-dispatch wiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from typing import Any, Sequence

import torch


BLOCK_SIZE = 4
PAD_TOKEN_ID = 0
INITIAL_TOKEN_ID = 1
TOKEN_SENTINEL = -1
REQUIRED_CAPABILITY = (7, 5)
RNG_POLICY = "default_cuda_generator_torch_multinomial"
PROBE_BACKEND = "ordinary_pytorch_cuda_graph_four_distinct_steps"


@dataclass(frozen=True)
class ExactK4CaseResult:
    seed: int
    target_eos_position: int
    eos_token_ids: list[int]
    eager_full_tokens: list[int]
    graph_full_tokens: list[int]
    eager_terminal_tokens: list[int]
    restored_terminal_tokens: list[int]
    eager_offsets: list[int]
    per_step_offset_increments: list[int]
    graph_offset_increment: int
    detected_terminal_steps: int
    full_tokens_exact: bool
    full_cache_exact: bool
    full_processor_state_exact: bool
    full_stopping_exact: bool
    full_generator_state_exact: bool
    terminal_tokens_exact: bool
    terminal_cache_exact: bool
    terminal_processor_state_exact: bool
    terminal_stopping_exact: bool
    terminal_generator_state_exact: bool
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactK4ProbeResult:
    backend: str
    rng_policy: str
    torch_version: str
    torch_cuda_version: str
    device_name: str
    device_capability: tuple[int, int]
    float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    nvidia_tf32_override: str
    block_size: int
    cases: list[ExactK4CaseResult]
    exact_k4_feasible: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [case.to_dict() for case in self.cases]
        return payload


@dataclass
class _ToyState:
    tokens: torch.Tensor
    cache: torch.Tensor
    processor_state: torch.Tensor
    processor_snapshots: torch.Tensor
    current_length: torch.Tensor
    length_snapshots: torch.Tensor
    unfinished: torch.Tensor
    unfinished_snapshots: torch.Tensor
    stop_flags: torch.Tensor


def _state_sha256(state: torch.Tensor) -> str:
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
        raise TypeError("generator state must be a uint8 tensor")
    payload = state.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _enforce_strict_fp32_policy() -> dict[str, Any]:
    override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if override != "0":
        raise RuntimeError(
            "strict FP32 K4 probe requires NVIDIA_TF32_OVERRIDE=0 before process start"
        )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    precision = torch.get_float32_matmul_precision()
    cuda_allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    if precision != "highest" or cuda_allow_tf32 or cudnn_allow_tf32:
        raise RuntimeError("strict FP32 policy did not remain highest/TF32-disabled")
    return {
        "float32_matmul_precision": precision,
        "cuda_matmul_allow_tf32": cuda_allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
        "nvidia_tf32_override": override,
    }


def _validate_probe_inputs(
    seeds: Sequence[int],
    eos_positions: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(seeds, (str, bytes)) or isinstance(eos_positions, (str, bytes)):
        raise TypeError("seeds and eos_positions must be integer sequences")
    parsed_seeds = tuple(seeds)
    parsed_positions = tuple(eos_positions)
    if not parsed_seeds or len(parsed_seeds) != len(parsed_positions):
        raise ValueError("seeds and eos_positions must be nonempty and equally sized")
    for seed in parsed_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("every seed must be a non-negative integer")
    for position in parsed_positions:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or not 1 <= position <= BLOCK_SIZE
        ):
            raise ValueError(f"every EOS position must be in [1, {BLOCK_SIZE}]")
    if len(set(parsed_positions)) != len(parsed_positions):
        raise ValueError("EOS positions must be distinct so every terminal case is covered")
    return parsed_seeds, parsed_positions


def _validate_offset_schedule(offsets: Sequence[int]) -> list[int]:
    parsed = [int(value) for value in offsets]
    if len(parsed) != BLOCK_SIZE + 1:
        raise ValueError(
            f"offset schedule must contain {BLOCK_SIZE + 1} boundaries"
        )
    increments = [after - before for before, after in zip(parsed, parsed[1:])]
    if any(increment <= 0 for increment in increments):
        raise RuntimeError(f"multinomial offsets did not advance positively: {increments}")
    if len(set(increments)) != 1:
        raise RuntimeError(
            "fixed-shape multinomial offset increments were not constant: "
            f"{increments}"
        )
    return increments


def _validate_state(state: _ToyState, *, block_size: int = BLOCK_SIZE) -> None:
    if state.tokens.dtype != torch.long or state.tokens.ndim != 1:
        raise TypeError("token storage must be a rank-one int64 tensor")
    if state.tokens.numel() != block_size + 1:
        raise ValueError("token storage must contain the prompt plus one full block")
    if state.cache.dtype != torch.float32 or state.cache.ndim != 2:
        raise TypeError("cache storage must be a rank-two FP32 tensor")
    if state.cache.shape[0] != block_size:
        raise ValueError("cache storage must contain one row per block step")
    floating = (state.processor_state, state.processor_snapshots)
    if any(tensor.dtype != torch.float32 for tensor in floating):
        raise TypeError("processor state and snapshots must remain FP32")
    integer = (
        state.current_length,
        state.length_snapshots,
        state.unfinished,
        state.unfinished_snapshots,
    )
    if any(tensor.dtype != torch.long for tensor in integer):
        raise TypeError("length and unfinished state must remain int64")
    if state.stop_flags.dtype != torch.bool:
        raise TypeError("stopping flags must be boolean")
    tensors = (
        state.tokens,
        state.cache,
        state.processor_state,
        state.processor_snapshots,
        state.current_length,
        state.length_snapshots,
        state.unfinished,
        state.unfinished_snapshots,
        state.stop_flags,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all toy decode state must share one device")


def restore_terminal_state(
    state: _ToyState,
    *,
    valid_steps: int,
    block_size: int = BLOCK_SIZE,
) -> None:
    """Rollback graph overshoot while retaining the first ``valid_steps`` writes."""

    _validate_state(state, block_size=block_size)
    if isinstance(valid_steps, bool) or not isinstance(valid_steps, int):
        raise TypeError("valid_steps must be an integer")
    if not 1 <= valid_steps <= block_size:
        raise ValueError(f"valid_steps must be in [1, {block_size}]")
    state.tokens[1 + valid_steps :].fill_(TOKEN_SENTINEL)
    state.cache[valid_steps:].zero_()
    state.processor_state.copy_(state.processor_snapshots[valid_steps - 1])
    state.current_length.copy_(state.length_snapshots[valid_steps - 1])
    state.unfinished.copy_(state.unfinished_snapshots[valid_steps - 1])
    state.stop_flags[valid_steps:].zero_()
    state.processor_snapshots[valid_steps:].zero_()
    state.length_snapshots[valid_steps:].zero_()
    state.unfinished_snapshots[valid_steps:].zero_()


def _make_logits(device: torch.device) -> torch.Tensor:
    vocabulary = 2 * BLOCK_SIZE + 2
    logits = torch.full(
        (BLOCK_SIZE, vocabulary),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for step in range(BLOCK_SIZE):
        first = 2 + 2 * step
        logits[step, first] = 0.15 + 0.07 * step
        logits[step, first + 1] = -0.10 - 0.03 * step
    return logits


def _make_state(device: torch.device, vocabulary: int) -> _ToyState:
    state = _ToyState(
        tokens=torch.full(
            (BLOCK_SIZE + 1,),
            TOKEN_SENTINEL,
            dtype=torch.long,
            device=device,
        ),
        cache=torch.zeros(
            (BLOCK_SIZE, vocabulary),
            dtype=torch.float32,
            device=device,
        ),
        processor_state=torch.zeros((1,), dtype=torch.float32, device=device),
        processor_snapshots=torch.zeros(
            (BLOCK_SIZE, 1), dtype=torch.float32, device=device
        ),
        current_length=torch.ones((1,), dtype=torch.long, device=device),
        length_snapshots=torch.zeros(
            (BLOCK_SIZE, 1), dtype=torch.long, device=device
        ),
        unfinished=torch.ones((1,), dtype=torch.long, device=device),
        unfinished_snapshots=torch.zeros(
            (BLOCK_SIZE, 1), dtype=torch.long, device=device
        ),
        stop_flags=torch.zeros((BLOCK_SIZE,), dtype=torch.bool, device=device),
    )
    state.tokens[0] = INITIAL_TOKEN_ID
    _validate_state(state)
    return state


def _reset_state(state: _ToyState) -> None:
    state.tokens.fill_(TOKEN_SENTINEL)
    state.tokens[0] = INITIAL_TOKEN_ID
    state.cache.zero_()
    state.processor_state.zero_()
    state.processor_snapshots.zero_()
    state.current_length.fill_(1)
    state.length_snapshots.zero_()
    state.unfinished.fill_(1)
    state.unfinished_snapshots.zero_()
    state.stop_flags.zero_()


def _decode_step(
    *,
    step: int,
    logits: torch.Tensor,
    eos_mask: torch.Tensor,
    state: _ToyState,
) -> None:
    previous = state.tokens[step].to(dtype=torch.float32)
    state.cache[step].copy_(logits[step].nan_to_num(neginf=-100.0) + previous)
    scores = logits[step].clone()
    first = 2 + 2 * step
    scores[first].add_(state.processor_state[0] * 0.001)
    probabilities = torch.softmax(scores, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1).reshape(())
    was_unfinished = state.unfinished[0].clone()
    token = sampled * was_unfinished + PAD_TOKEN_ID * (1 - was_unfinished)
    state.tokens[step + 1].copy_(token)
    stopped = eos_mask[sampled].logical_and(was_unfinished.to(dtype=torch.bool))
    state.stop_flags[step].copy_(stopped)
    state.unfinished.mul_((~stopped).to(dtype=torch.long))
    state.processor_state.add_(token.to(dtype=torch.float32))
    state.current_length.add_(1)
    state.processor_snapshots[step].copy_(state.processor_state)
    state.length_snapshots[step].copy_(state.current_length)
    state.unfinished_snapshots[step].copy_(state.unfinished)


def _run_steps(
    *,
    logits: torch.Tensor,
    eos_mask: torch.Tensor,
    state: _ToyState,
    stop_on_eos: bool,
) -> list[int]:
    offsets = [int(torch.cuda._get_rng_state_offset(logits.device))]
    for step in range(BLOCK_SIZE):
        _decode_step(step=step, logits=logits, eos_mask=eos_mask, state=state)
        offsets.append(int(torch.cuda._get_rng_state_offset(logits.device)))
        if stop_on_eos and bool(state.stop_flags[step].item()):
            break
    return offsets


def _state_tokens(state: _ToyState) -> list[int]:
    return [int(value) for value in state.tokens.detach().cpu().tolist()]


def _capture_block(
    *,
    logits: torch.Tensor,
    eos_mask: torch.Tensor,
    state: _ToyState,
    generator: torch.Generator,
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    graph.register_generator_state(generator)
    with torch.cuda.graph(graph):
        for step in range(BLOCK_SIZE):
            _decode_step(step=step, logits=logits, eos_mask=eos_mask, state=state)
    return graph


def _detect_terminal_steps(stop_flags: torch.Tensor) -> int:
    indices = torch.nonzero(stop_flags, as_tuple=False).reshape(-1)
    if indices.numel() != 1:
        raise RuntimeError(
            "K4 toy block must record exactly one first-EOS flag; "
            f"found {int(indices.numel())}"
        )
    return int(indices[0].item()) + 1


def _run_case(
    *,
    seed: int,
    target_eos_position: int,
    device: torch.device,
) -> ExactK4CaseResult:
    logits = _make_logits(device)
    eos_mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
    first_eos = 2 + 2 * (target_eos_position - 1)
    eos_mask[first_eos : first_eos + 2] = True
    eos_token_ids = [first_eos, first_eos + 1]
    generator = torch.cuda.default_generators[device.index or 0]
    generator.manual_seed(seed)

    graph_state = _make_state(device, logits.shape[1])
    graph = _capture_block(
        logits=logits,
        eos_mask=eos_mask,
        state=graph_state,
        generator=generator,
    )
    post_capture_state = generator.get_state().clone()
    start_offset = int(generator.get_offset())

    eager_full = _make_state(device, logits.shape[1])
    generator.set_state(post_capture_state)
    eager_offsets = _run_steps(
        logits=logits,
        eos_mask=eos_mask,
        state=eager_full,
        stop_on_eos=False,
    )
    eager_full_generator_state = generator.get_state().clone()
    increments = _validate_offset_schedule(eager_offsets)

    _reset_state(graph_state)
    generator.set_state(post_capture_state)
    graph.replay()
    torch.cuda.synchronize(device)
    graph_end_offset = int(generator.get_offset())
    graph_full_generator_state = generator.get_state().clone()
    graph_full_tokens = _state_tokens(graph_state)
    full_tokens_exact = torch.equal(graph_state.tokens, eager_full.tokens)
    full_cache_exact = torch.equal(graph_state.cache, eager_full.cache)
    full_processor_state_exact = torch.equal(
        graph_state.processor_state, eager_full.processor_state
    ) and torch.equal(
        graph_state.processor_snapshots, eager_full.processor_snapshots
    )
    full_stopping_exact = (
        torch.equal(graph_state.stop_flags, eager_full.stop_flags)
        and torch.equal(graph_state.unfinished, eager_full.unfinished)
        and torch.equal(graph_state.current_length, eager_full.current_length)
    )
    full_generator_state_exact = _state_sha256(
        graph_full_generator_state
    ) == _state_sha256(eager_full_generator_state)
    graph_offset_increment = graph_end_offset - start_offset
    if graph_offset_increment != sum(increments):
        raise RuntimeError(
            "ordinary K4 graph did not consume the measured eager RNG offset: "
            f"graph={graph_offset_increment}, eager={sum(increments)}"
        )

    eager_terminal = _make_state(device, logits.shape[1])
    generator.set_state(post_capture_state)
    terminal_offsets = _run_steps(
        logits=logits,
        eos_mask=eos_mask,
        state=eager_terminal,
        stop_on_eos=True,
    )
    if len(terminal_offsets) != target_eos_position + 1:
        raise RuntimeError(
            f"eager terminal position changed: expected {target_eos_position}, "
            f"got {len(terminal_offsets) - 1}"
        )
    eager_terminal_generator_state = generator.get_state().clone()

    _reset_state(graph_state)
    generator.set_state(post_capture_state)
    graph.replay()
    torch.cuda.synchronize(device)
    detected = _detect_terminal_steps(graph_state.stop_flags)
    if detected != target_eos_position:
        raise RuntimeError(
            f"graph terminal position changed: expected {target_eos_position}, "
            f"got {detected}"
        )
    restore_terminal_state(graph_state, valid_steps=detected)
    corrected_offset = eager_offsets[detected]
    generator.set_offset(corrected_offset)
    restored_generator_state = generator.get_state().clone()

    terminal_tokens_exact = torch.equal(graph_state.tokens, eager_terminal.tokens)
    terminal_cache_exact = torch.equal(graph_state.cache, eager_terminal.cache)
    terminal_processor_state_exact = torch.equal(
        graph_state.processor_state, eager_terminal.processor_state
    )
    terminal_stopping_exact = (
        torch.equal(graph_state.stop_flags, eager_terminal.stop_flags)
        and torch.equal(graph_state.unfinished, eager_terminal.unfinished)
        and torch.equal(graph_state.current_length, eager_terminal.current_length)
    )
    terminal_generator_state_exact = _state_sha256(
        restored_generator_state
    ) == _state_sha256(eager_terminal_generator_state)
    exact_flags = (
        full_tokens_exact,
        full_cache_exact,
        full_processor_state_exact,
        full_stopping_exact,
        full_generator_state_exact,
        terminal_tokens_exact,
        terminal_cache_exact,
        terminal_processor_state_exact,
        terminal_stopping_exact,
        terminal_generator_state_exact,
    )
    return ExactK4CaseResult(
        seed=seed,
        target_eos_position=target_eos_position,
        eos_token_ids=eos_token_ids,
        eager_full_tokens=_state_tokens(eager_full),
        graph_full_tokens=graph_full_tokens,
        eager_terminal_tokens=_state_tokens(eager_terminal),
        restored_terminal_tokens=_state_tokens(graph_state),
        eager_offsets=eager_offsets,
        per_step_offset_increments=increments,
        graph_offset_increment=graph_offset_increment,
        detected_terminal_steps=detected,
        full_tokens_exact=full_tokens_exact,
        full_cache_exact=full_cache_exact,
        full_processor_state_exact=full_processor_state_exact,
        full_stopping_exact=full_stopping_exact,
        full_generator_state_exact=full_generator_state_exact,
        terminal_tokens_exact=terminal_tokens_exact,
        terminal_cache_exact=terminal_cache_exact,
        terminal_processor_state_exact=terminal_processor_state_exact,
        terminal_stopping_exact=terminal_stopping_exact,
        terminal_generator_state_exact=terminal_generator_state_exact,
        exact=all(exact_flags),
    )


def run_exact_k4_multinomial_probe(
    *,
    seeds: Sequence[int] = (7, 12345, 987654, 42),
    eos_positions: Sequence[int] = (1, 2, 3, 4),
) -> ExactK4ProbeResult:
    """Run the bounded strict-FP32 K4 exactness gate on an allocated 2080 Ti."""

    parsed_seeds, parsed_positions = _validate_probe_inputs(seeds, eos_positions)
    policy = _enforce_strict_fp32_policy()
    if not torch.cuda.is_available():
        raise RuntimeError("exact K4 probe requires an allocated CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability != REQUIRED_CAPABILITY:
        raise RuntimeError(
            "exact K4 probe is an RTX 2080 Ti/SM75 experiment; "
            f"found capability {capability}"
        )
    if not hasattr(torch.cuda.CUDAGraph, "register_generator_state"):
        raise RuntimeError("PyTorch does not expose CUDAGraph.register_generator_state")
    generator = torch.cuda.default_generators[device.index or 0]
    if not hasattr(generator, "get_offset") or not hasattr(generator, "set_offset"):
        raise RuntimeError("CUDA generator offset APIs are unavailable")
    if not hasattr(torch.cuda, "_get_rng_state_offset"):
        raise RuntimeError("PyTorch CUDA RNG-offset inspection API is unavailable")

    original_generator_state = generator.get_state().clone()
    try:
        cases = [
            _run_case(seed=seed, target_eos_position=position, device=device)
            for seed, position in zip(parsed_seeds, parsed_positions)
        ]
    finally:
        generator.set_state(original_generator_state)
    return ExactK4ProbeResult(
        backend=PROBE_BACKEND,
        rng_policy=RNG_POLICY,
        torch_version=str(torch.__version__),
        torch_cuda_version=str(torch.version.cuda),
        device_name=str(torch.cuda.get_device_name(device)),
        device_capability=capability,
        block_size=BLOCK_SIZE,
        cases=cases,
        exact_k4_feasible=all(case.exact for case in cases),
        **policy,
    )


__all__ = [
    "BLOCK_SIZE",
    "ExactK4CaseResult",
    "ExactK4ProbeResult",
    "PROBE_BACKEND",
    "RNG_POLICY",
    "restore_terminal_state",
    "run_exact_k4_multinomial_probe",
]
