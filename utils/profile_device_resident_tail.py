"""Size a fixed-block device-resident decode tail on real SALVALAI logits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

from osuT5.osuT5.inference.optimized.scout.device_tail import (  # noqa: E402
    DeviceSequenceState,
    fixed_block_tail,
)
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _accepted_main_session_run,
    _assert_scout_args,
    _load_args,
)


SCHEMA_VERSION = 1
BLOCK_SIZES = (8, 16)
BASELINE_MAIN_SECONDS = 30.068768849
FIXED_WORK_TOKENS = 8_294
PROMOTION_SAVING_SECONDS = 5.0
ACTIVE_PREFIX_BUCKET_SIZE = 64
DEFAULT_SAMPLES_PER_PREFIX = 3


def _git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return (
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _set_rng_state(
    state: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    cpu, cuda = state
    torch.random.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


def _rng_equal(
    left: tuple[torch.Tensor, list[torch.Tensor] | None],
    right: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> bool:
    if not torch.equal(left[0], right[0]):
        return False
    if left[1] is None or right[1] is None:
        return left[1] is right[1]
    return len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True)
    )


def _bucketed_prefix(cur_len: int, max_length: int) -> int:
    if cur_len <= 0 or max_length <= 0:
        raise ValueError("sequence lengths must be positive")
    return min(
        ((cur_len + ACTIVE_PREFIX_BUCKET_SIZE - 1) // ACTIVE_PREFIX_BUCKET_SIZE)
        * ACTIVE_PREFIX_BUCKET_SIZE,
        max_length,
    )


def _is_main_capture_label(profile_label: Any) -> bool:
    return profile_label == "main_generation"


def _clone_processor_list(processors) -> LogitsProcessorList:
    return LogitsProcessorList(copy.deepcopy(list(processors)))


def _mutable_tensor_templates(root: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot tensor-backed processor state for deterministic graph replay."""

    templates: list[tuple[torch.Tensor, torch.Tensor]] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            templates.append((value, value.detach().clone()))
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for item in attributes.values():
                visit(item)

    visit(root)
    return templates


@dataclass(slots=True)
class TailAnchor:
    prefix: int
    input_ids: torch.Tensor
    raw_logits_steps: list[torch.Tensor]
    rng_state: tuple[torch.Tensor, list[torch.Tensor] | None]
    logits_processors: LogitsProcessorList
    stopping_criteria: Any
    do_sample: bool
    pad_token_id: int
    eos_token_ids: torch.Tensor
    max_length: int


@dataclass(slots=True)
class WindowSchedule:
    window_index: int
    generated_steps: int
    decode_prefixes: list[int]


@dataclass(slots=True)
class CapturedTailGraph:
    graph: torch.cuda.CUDAGraph
    state: DeviceSequenceState
    outputs: tuple[torch.Tensor, torch.Tensor]
    static_model_inputs: dict[str, torch.Tensor]
    mutable_templates: list[tuple[torch.Tensor, torch.Tensor]]
    state_templates: dict[str, torch.Tensor]
    setup_seconds: float
    peak_vram_bytes: int

    def reset(self) -> None:
        self.state.sequence.copy_(self.state_templates["sequence"])
        self.state.physical_length.copy_(self.state_templates["physical_length"])
        self.state.logical_length.copy_(self.state_templates["logical_length"])
        self.state.unfinished.copy_(self.state_templates["unfinished"])
        for name, value in self.static_model_inputs.items():
            value.copy_(self.state_templates[f"static:{name}"])
        for tensor, template in self.mutable_templates:
            if tensor.is_cuda:
                tensor.copy_(template)


class _RawTailCapture(LogitsProcessor):
    """Transparent main-only processor that retains complete scheduled blocks."""

    def __init__(
        self,
        *,
        anchors: dict[int, dict[int, list[TailAnchor]]],
        processors: Any,
        stopping_criteria: Any,
        generation_config: Any,
        block_sizes: tuple[int, ...],
        window_index: int,
        samples_per_prefix: int,
    ) -> None:
        self.anchors = anchors
        self.processors = processors
        self.stopping_criteria = stopping_criteria
        self.generation_config = generation_config
        if not block_sizes or any(size < 1 for size in block_sizes):
            raise ValueError("tail capture block sizes must be positive")
        self.block_sizes = tuple(sorted(set(block_sizes)))
        self.window_index = window_index
        if samples_per_prefix < 1:
            raise ValueError("samples-per-prefix must be positive")
        self.samples_per_prefix = samples_per_prefix
        self.generated_steps = 0
        self.decode_prefixes: list[int] = []
        self.active: dict[int, TailAnchor | None] = {
            size: None for size in self.block_sizes
        }

    def _processors_without_self(self) -> LogitsProcessorList:
        return LogitsProcessorList(
            copy.deepcopy(
                [processor for processor in self.processors if processor is not self]
            )
        )

    def _start(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        prefix: int,
    ) -> TailAnchor:
        pad = getattr(self.generation_config, "_pad_token_tensor", None)
        eos = getattr(self.generation_config, "_eos_token_tensor", None)
        if not isinstance(pad, torch.Tensor) or pad.numel() != 1:
            raise RuntimeError("tail capture requires one tensor pad token")
        if not isinstance(eos, torch.Tensor) or eos.numel() < 1:
            raise RuntimeError("tail capture requires tensor EOS ids")
        return TailAnchor(
            prefix=prefix,
            input_ids=input_ids.detach().clone(),
            raw_logits_steps=[scores.detach().clone()],
            rng_state=_rng_state(),
            logits_processors=self._processors_without_self(),
            stopping_criteria=copy.deepcopy(self.stopping_criteria),
            do_sample=bool(self.generation_config.do_sample),
            pad_token_id=int(pad.detach().cpu().reshape(-1)[0]),
            eos_token_ids=eos.detach().clone().reshape(-1),
            max_length=int(self.generation_config.max_length),
        )

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise RuntimeError("device-tail capture requires batch size one")
        cur_len = int(input_ids.shape[1])
        max_length = int(self.generation_config.max_length)
        prefix = _bucketed_prefix(cur_len, max_length)
        self.generated_steps += 1
        # One prefill produces the first generated token in every window.  It
        # is not a decode-graph replay and must not be projected as K-block work.
        if self.generated_steps == 1:
            return scores
        self.decode_prefixes.append(prefix)
        for block_size in self.block_sizes:
            active = self.active[block_size]
            if active is not None:
                expected = active.input_ids.shape[1] + len(active.raw_logits_steps)
                if prefix != active.prefix or cur_len != expected:
                    active = None
                else:
                    active.raw_logits_steps.append(scores.detach().clone())
                    if len(active.raw_logits_steps) == block_size:
                        self.anchors.setdefault(block_size, {}).setdefault(
                            prefix,
                            [],
                        ).append(active)
                        active = None
                    self.active[block_size] = active
                    continue
            if (
                prefix - cur_len + 1 >= block_size
                and max_length - cur_len >= block_size
                and len(
                    self.anchors.setdefault(block_size, {}).get(prefix, [])
                )
                < self.samples_per_prefix
            ):
                active = self._start(input_ids, scores, prefix)
            self.active[block_size] = active
        return scores

    def schedule(self) -> WindowSchedule:
        return WindowSchedule(
            window_index=self.window_index,
            generated_steps=self.generated_steps,
            decode_prefixes=list(self.decode_prefixes),
        )


def _eager_dynamic_tail(
    anchor: TailAnchor,
    *,
    block_size: int,
    processors,
    stopping_criteria,
    read_status_each_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = anchor.input_ids.detach().clone()
    unfinished = torch.ones((1,), dtype=torch.bool, device=input_ids.device)
    generated = []
    continues = []
    pad = torch.tensor(
        anchor.pad_token_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    for raw_logits in anchor.raw_logits_steps[:block_size]:
        scores = processors(
            input_ids,
            raw_logits.clone(memory_format=torch.contiguous_format),
        )
        if anchor.do_sample:
            probabilities = nn.functional.softmax(scores, dim=-1)
            next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(scores, dim=-1)
        next_tokens = torch.where(unfinished, next_tokens, pad)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
        stopped = stopping_criteria(input_ids, None).to(dtype=torch.bool)
        unfinished = unfinished & ~stopped
        if read_status_each_step:
            # Match the production loop's scalar host decision.  The value is
            # deliberately not used to truncate this fixed-work control.
            bool((unfinished.max() == 0).item())
        generated.append(next_tokens)
        continues.append(unfinished.clone())
    return (
        input_ids,
        torch.stack(generated),
        unfinished,
        torch.stack(continues),
    )


def _new_device_state(anchor: TailAnchor, block_size: int) -> DeviceSequenceState:
    return DeviceSequenceState.allocate(
        anchor.input_ids,
        max_length=max(anchor.max_length, anchor.input_ids.shape[1] + block_size + 1),
        pad_token_id=anchor.pad_token_id,
        eos_token_ids=anchor.eos_token_ids,
    )


def _capture_candidate_graph(
    anchor: TailAnchor,
    *,
    block_size: int,
) -> CapturedTailGraph:
    state = _new_device_state(anchor, block_size)
    processors = _clone_processor_list(anchor.logits_processors)
    criteria = copy.deepcopy(anchor.stopping_criteria)
    mutable_templates = _mutable_tensor_templates((processors, criteria))
    start_length = int(anchor.input_ids.shape[1])
    static_model_inputs = {
        "decoder_input_ids": anchor.input_ids[:, -1:].detach().clone(),
        "cache_position": torch.full(
            (1,),
            start_length - 1,
            dtype=torch.long,
            device=anchor.input_ids.device,
        ),
        "position_ids": torch.full(
            (1, 1),
            start_length - 1,
            dtype=torch.long,
            device=anchor.input_ids.device,
        ),
    }
    template = {
        "sequence": state.sequence.detach().clone(),
        "physical_length": state.physical_length.detach().clone(),
        "logical_length": state.logical_length.detach().clone(),
        "unfinished": state.unfinished.detach().clone(),
        **{
            f"static:{name}": value.detach().clone()
            for name, value in static_model_inputs.items()
        },
    }

    def run():
        return fixed_block_tail(
            state=state,
            start_length=start_length,
            raw_logits_steps=anchor.raw_logits_steps[:block_size],
            logits_processor=processors,
            stopping_criteria=criteria,
            do_sample=anchor.do_sample,
            static_model_inputs=static_model_inputs,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    setup_started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    captured = CapturedTailGraph(
        graph=graph,
        state=state,
        outputs=outputs,
        static_model_inputs=static_model_inputs,
        mutable_templates=mutable_templates,
        state_templates=template,
        setup_seconds=setup_seconds,
        peak_vram_bytes=int(torch.cuda.max_memory_allocated()),
    )
    captured.reset()
    torch.cuda.synchronize()
    return captured


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _measure_eager_once(
    anchor: TailAnchor,
    *,
    block_size: int,
) -> dict[str, float]:
    processors = _clone_processor_list(anchor.logits_processors)
    criteria = copy.deepcopy(anchor.stopping_criteria)
    _set_rng_state(anchor.rng_state)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    _eager_dynamic_tail(
        anchor,
        block_size=block_size,
        processors=processors,
        stopping_criteria=criteria,
        read_status_each_step=True,
    )
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1_000.0
    event_ms = _event_ms(start_event, end_event)
    return {
        "wall_ms": wall_ms,
        "cuda_ms": event_ms,
    }


def _measure_graph_once(
    captured: CapturedTailGraph,
    anchor: TailAnchor,
) -> dict[str, float]:
    _set_rng_state(anchor.rng_state)
    captured.reset()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start_event.record()
    captured.graph.replay()
    end_event.record()
    # One explicit device-to-host completion decision per block.  It replaces
    # the production loop's one scalar read per token.
    bool(captured.state.unfinished.item())
    wall_ms = (time.perf_counter() - started) * 1_000.0
    event_ms = _event_ms(start_event, end_event)
    return {
        "wall_ms": wall_ms,
        "cuda_ms": event_ms,
    }


def _time_reciprocal(
    captured: CapturedTailGraph,
    anchor: TailAnchor,
    *,
    block_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    by_order: dict[str, dict[str, list[dict[str, float]]]] = {
        "eager_first": {"eager": [], "device_graph": []},
        "graph_first": {"eager": [], "device_graph": []},
    }
    for index in range(warmup + iterations):
        order_name = "eager_first" if index % 2 == 0 else "graph_first"
        variants = ("eager", "device_graph") if index % 2 == 0 else (
            "device_graph",
            "eager",
        )
        for variant in variants:
            sample = (
                _measure_eager_once(anchor, block_size=block_size)
                if variant == "eager"
                else _measure_graph_once(captured, anchor)
            )
            if index >= warmup:
                by_order[order_name][variant].append(sample)

    order_reports: dict[str, Any] = {}
    for order_name, values in by_order.items():
        if not values["eager"] or not values["device_graph"]:
            raise RuntimeError("reciprocal timing produced an empty order")
        variants = {}
        for variant, samples in values.items():
            wall = statistics.median(sample["wall_ms"] for sample in samples)
            cuda = statistics.median(sample["cuda_ms"] for sample in samples)
            variants[variant] = {
                "wall_ms_per_block": wall,
                "cuda_ms_per_block": cuda,
                "enqueue_and_sync_ms_per_block": max(0.0, wall - cuda),
            }
        order_reports[order_name] = {
            **variants,
            "saved_wall_ms_per_block": (
                variants["eager"]["wall_ms_per_block"]
                - variants["device_graph"]["wall_ms_per_block"]
            ),
        }
    conservative = min(
        report["saved_wall_ms_per_block"] for report in order_reports.values()
    )
    return {
        "orders": order_reports,
        "conservative_saved_wall_ms_per_block": conservative,
        "post_block_status_reads": 1,
        "static_model_input_handoff": True,
    }


def _profile_anchor(
    anchor: TailAnchor,
    *,
    block_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if len(anchor.raw_logits_steps) < block_size:
        raise ValueError("anchor does not contain the requested block")
    _set_rng_state(anchor.rng_state)
    eager = _eager_dynamic_tail(
        anchor,
        block_size=block_size,
        processors=_clone_processor_list(anchor.logits_processors),
        stopping_criteria=copy.deepcopy(anchor.stopping_criteria),
    )
    eager_rng = _rng_state()

    _set_rng_state(anchor.rng_state)
    captured = _capture_candidate_graph(
        anchor,
        block_size=block_size,
    )
    _set_rng_state(anchor.rng_state)
    captured.reset()
    torch.cuda.synchronize()
    captured.graph.replay()
    torch.cuda.synchronize()
    graph_rng = _rng_state()
    graph_generated, graph_continues = captured.outputs
    start_len = anchor.input_ids.shape[1]
    graph_sequence = captured.state.sequence[:, : start_len + block_size]
    checks = {
        "tokens_exact": bool(torch.equal(eager[1], graph_generated)),
        "sequence_exact": bool(torch.equal(eager[0], graph_sequence)),
        "unfinished_exact": bool(torch.equal(eager[2], captured.state.unfinished)),
        "continue_values_exact": bool(torch.equal(eager[3], graph_continues)),
        "rng_exact": _rng_equal(eager_rng, graph_rng),
        "handoff_token_exact": bool(
            torch.equal(
                captured.static_model_inputs["decoder_input_ids"].reshape(-1),
                eager[1][-1].reshape(-1),
            )
        ),
        "handoff_positions_exact": bool(
            torch.equal(
                captured.static_model_inputs["cache_position"],
                torch.full_like(
                    captured.static_model_inputs["cache_position"],
                    int(start_len + block_size - 1),
                ),
            )
            and torch.equal(
                captured.static_model_inputs["position_ids"],
                torch.full_like(
                    captured.static_model_inputs["position_ids"],
                    int(start_len + block_size - 1),
                ),
            )
        ),
        "no_early_stop": bool(torch.all(eager[3][:-1]).item())
        if block_size > 1
        else True,
        "finite_logits": all(
            bool(torch.isfinite(logits).all().item())
            for logits in anchor.raw_logits_steps[:block_size]
        ),
    }
    timing = _time_reciprocal(
        captured,
        anchor,
        block_size=block_size,
        warmup=warmup,
        iterations=iterations,
    )
    saved_wall_ms = timing["conservative_saved_wall_ms_per_block"]
    return {
        "prefix": anchor.prefix,
        "block_size": block_size,
        "start_length": int(start_len),
        "vocab_size": int(anchor.raw_logits_steps[0].shape[-1]),
        "processor_classes": [
            processor.__class__.__name__ for processor in anchor.logits_processors
        ],
        "stopping_classes": [
            criterion.__class__.__name__ for criterion in anchor.stopping_criteria
        ],
        "checks": checks,
        "pass": all(checks.values()),
        "reciprocal_timing": timing,
        "capture_setup_seconds": captured.setup_seconds,
        "peak_vram_bytes": captured.peak_vram_bytes,
        "saved_wall_ms_per_block": saved_wall_ms,
        "saved_wall_us_per_token": saved_wall_ms * 1_000.0 / block_size,
    }


def validate_live_graph_cache(graph_cache: dict[Any, dict[str, Any]]) -> dict[int, int]:
    if not isinstance(graph_cache, dict) or not graph_cache:
        raise ValueError("live graph cache must be a non-empty mapping")
    counts: dict[int, int] = {}
    for entry in graph_cache.values():
        if not isinstance(entry, dict):
            raise TypeError("live graph cache entries must be mappings")
        prefix = entry.get("active_prefix_length")
        count = entry.get("decode_replays")
        if (
            isinstance(prefix, bool)
            or not isinstance(prefix, int)
            or prefix <= 0
            or prefix in counts
        ):
            raise ValueError(f"invalid or duplicate live prefix {prefix!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"live prefix {prefix} has invalid replay count {count!r}")
        for required in ("graph", "outputs", "static_inputs"):
            if required not in entry:
                raise ValueError(f"live prefix {prefix} is missing {required}")
        counts[prefix] = count
    return dict(sorted(counts.items()))


def schedule_block_manifest(
    schedules: list[WindowSchedule],
    *,
    live_counts: dict[int, int],
) -> dict[str, Any]:
    if not schedules:
        raise ValueError("tail schedule requires at least one main window")
    scheduled_counts = {prefix: 0 for prefix in live_counts}
    runs: list[tuple[int, int, int]] = []
    generated_steps = 0
    for schedule in schedules:
        if schedule.generated_steps < 1:
            raise ValueError(f"main window {schedule.window_index} generated no tokens")
        if schedule.generated_steps != len(schedule.decode_prefixes) + 1:
            raise ValueError(
                f"main window {schedule.window_index} does not reconcile prefill/decode"
            )
        generated_steps += schedule.generated_steps
        start = 0
        while start < len(schedule.decode_prefixes):
            prefix = schedule.decode_prefixes[start]
            if prefix not in scheduled_counts:
                raise ValueError(f"scheduled prefix {prefix} is absent from live graph cache")
            end = start + 1
            while (
                end < len(schedule.decode_prefixes)
                and schedule.decode_prefixes[end] == prefix
            ):
                end += 1
            length = end - start
            scheduled_counts[prefix] += length
            runs.append((schedule.window_index, prefix, length))
            start = end
    if scheduled_counts != live_counts:
        raise RuntimeError(
            f"ordered main schedule differs from live graph counts: "
            f"scheduled={scheduled_counts}, live={live_counts}"
        )
    blocks: dict[str, Any] = {}
    for block_size in BLOCK_SIZES:
        prefixes: dict[str, Any] = {}
        for prefix, total in live_counts.items():
            prefix_runs = [length for _, value, length in runs if value == prefix]
            full_blocks = sum(length // block_size for length in prefix_runs)
            eligible = full_blocks * block_size
            prefixes[str(prefix)] = {
                "run_count": len(prefix_runs),
                "full_blocks": full_blocks,
                "eligible_replays": eligible,
                "remainder_replays": total - eligible,
                "total_replays": total,
            }
        blocks[str(block_size)] = {
            "prefixes": prefixes,
            "full_blocks": sum(value["full_blocks"] for value in prefixes.values()),
            "eligible_replays": sum(
                value["eligible_replays"] for value in prefixes.values()
            ),
            "remainder_replays": sum(
                value["remainder_replays"] for value in prefixes.values()
            ),
        }
    return {
        "window_count": len(schedules),
        "prefill_steps": len(schedules),
        "decode_replays": sum(live_counts.values()),
        "generated_steps": generated_steps,
        "run_count": len(runs),
        "blocks": blocks,
    }


def _aggregate_anchor_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("anchor aggregation requires samples")
    return {
        "samples": samples,
        "sample_count": len(samples),
        "pass": all(bool(sample["pass"]) for sample in samples),
        "conservative_saved_wall_us_per_token": min(
            float(sample["saved_wall_us_per_token"]) for sample in samples
        ),
        "capture_setup_seconds": sum(
            float(sample["capture_setup_seconds"]) for sample in samples
        ),
        "peak_vram_bytes": max(int(sample["peak_vram_bytes"]) for sample in samples),
    }


def _representative_anchors(
    anchors: list[TailAnchor],
    *,
    limit: int,
) -> list[TailAnchor]:
    if limit < 1:
        raise ValueError("samples-per-prefix must be positive")
    if len(anchors) <= limit:
        return list(anchors)
    indices = {
        round(index * (len(anchors) - 1) / (limit - 1))
        for index in range(limit)
    } if limit > 1 else {len(anchors) // 2}
    return [anchors[index] for index in sorted(indices)]


def summarize(
    buckets: dict[str, Any],
    *,
    live_counts: dict[int, int],
    schedule_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not buckets or not live_counts:
        raise ValueError("tail summary requires measured buckets and live counts")
    summaries: dict[str, Any] = {}
    total_replays = sum(live_counts.values())
    for block_size in BLOCK_SIZES:
        measured = {
            int(prefix): values[str(block_size)]
            for prefix, values in buckets.items()
            if str(block_size) in values
        }
        block_schedule = schedule_manifest["blocks"][str(block_size)]
        eligible_by_prefix = {
            int(prefix): int(values["eligible_replays"])
            for prefix, values in block_schedule["prefixes"].items()
        }
        measured_replays = sum(
            eligible_by_prefix.get(prefix, 0) for prefix in measured
        )
        saved_seconds = sum(
            eligible_by_prefix[prefix]
            * float(entry["conservative_saved_wall_us_per_token"])
            / 1_000_000.0
            for prefix, entry in measured.items()
        )
        eligible_replays = int(block_schedule["eligible_replays"])
        all_eligible_measured = measured_replays == eligible_replays
        component_pass = (
            saved_seconds >= PROMOTION_SAVING_SECONDS
            and eligible_replays > 0
            and all_eligible_measured
            and all(entry["pass"] for entry in measured.values())
        )
        summaries[str(block_size)] = {
            "measured_prefixes": sorted(measured),
            "measured_eligible_replays": measured_replays,
            "eligible_replays": eligible_replays,
            "remainder_replays": int(block_schedule["remainder_replays"]),
            "full_blocks": int(block_schedule["full_blocks"]),
            "total_replays": total_replays,
            "eligible_coverage_fraction": (
                measured_replays / eligible_replays if eligible_replays else 0.0
            ),
            "tail_component_runtime_headroom_seconds": saved_seconds,
            "all_correctness_pass": all(entry["pass"] for entry in measured.values()),
            "promotion_saving_seconds": PROMOTION_SAVING_SECONDS,
            "tail_component_headroom_pass": component_pass,
            "full_runtime_decision_available": False,
            "full_runtime_promotion_pass": False,
            "full_runtime_blockers": [
                "untraced_transition_budget_not_combined",
                "full_forward_graph_not_measured",
                "capture_setup_and_memory_not_projectable_to_full_runtime",
            ],
            "unmeasured_prefixes_assumed_saving_seconds": 0.0,
        }
    return summaries


@torch.no_grad()
def profile_device_resident_tail(
    args,
    *,
    output_path: Path,
    warmup: int,
    iterations: int,
    samples_per_prefix: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("device-resident tail profiler requires CUDA")
    if warmup < 1 or iterations < 2:
        raise ValueError("warmup must be positive and reciprocal iterations >= 2")
    if samples_per_prefix < 1:
        raise ValueError("samples-per-prefix must be positive")
    _assert_scout_args(args)
    from osuT5.osuT5.inference import Processor
    from osuT5.osuT5.inference.optimized.single import engine
    from osuT5.osuT5.inference.runtime_dispatch import attention_runtime_hooks

    anchors: dict[int, dict[int, list[TailAnchor]]] = {
        block_size: {} for block_size in BLOCK_SIZES
    }
    schedules: list[WindowSchedule] = []
    active_profile_label: str | None = None
    original_engine_generate = engine.active_prefix_decode_generate
    original_processor_generate = Processor.generate

    def tagged_processor_generate(processor, *positional, **kwargs):
        nonlocal active_profile_label
        previous = active_profile_label
        active_profile_label = kwargs.get("profile_label")
        try:
            return original_processor_generate(processor, *positional, **kwargs)
        finally:
            active_profile_label = previous

    def observed_generate(
        model,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        *positional,
        **kwargs,
    ):
        if not _is_main_capture_label(active_profile_label):
            return original_engine_generate(
                model,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                *positional,
                **kwargs,
            )
        hooks = attention_runtime_hooks()
        if hooks.q1_rope_cache_self_attention_forward is None:
            raise RuntimeError("main tail capture requires accepted native self attention")
        capture = _RawTailCapture(
            anchors=anchors,
            processors=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            block_sizes=BLOCK_SIZES,
            window_index=len(schedules),
            samples_per_prefix=samples_per_prefix,
        )
        logits_processor.insert(0, capture)
        try:
            return original_engine_generate(
                model,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                *positional,
                **kwargs,
            )
        finally:
            if capture in logits_processor:
                logits_processor.remove(capture)
            if capture.generated_steps:
                schedules.append(capture.schedule())

    Processor.generate = tagged_processor_generate
    engine.active_prefix_decode_generate = observed_generate
    try:
        run = _accepted_main_session_run(args, output_path=output_path)
    finally:
        engine.active_prefix_decode_generate = original_engine_generate
        Processor.generate = original_processor_generate
    live_counts = validate_live_graph_cache(run["session"].graph_cache)
    schedule_manifest = schedule_block_manifest(schedules, live_counts=live_counts)
    stats_generated = int(
        getattr(run["processor"], "last_generation_stats", {}).get(
            "generated_tokens",
            -1,
        )
    )
    if stats_generated != schedule_manifest["generated_steps"]:
        raise RuntimeError(
            "main generated-token stats differ from observed schedule: "
            f"stats={stats_generated}, schedule={schedule_manifest['generated_steps']}"
        )
    if stats_generated != FIXED_WORK_TOKENS:
        raise RuntimeError(
            f"fixed-work main requires {FIXED_WORK_TOKENS} generated steps, "
            f"got {stats_generated}"
        )
    measured: dict[str, Any] = {}
    for block_size in BLOCK_SIZES:
        scheduled_prefixes = schedule_manifest["blocks"][str(block_size)]["prefixes"]
        for prefix, prefix_schedule in scheduled_prefixes.items():
            prefix_int = int(prefix)
            expected_blocks = int(prefix_schedule["full_blocks"])
            captured = anchors[block_size].get(prefix_int, [])
            expected_samples = min(expected_blocks, samples_per_prefix)
            if len(captured) != expected_samples:
                raise RuntimeError(
                    f"K{block_size} prefix {prefix} captured {len(captured)} samples, "
                    f"expected {expected_samples} from {expected_blocks} scheduled blocks"
                )
            if not captured:
                continue
            representative = _representative_anchors(
                captured,
                limit=samples_per_prefix,
            )
            samples = [
                _profile_anchor(
                    anchor,
                    block_size=block_size,
                    warmup=warmup,
                    iterations=iterations,
                )
                for anchor in representative
            ]
            measured.setdefault(prefix, {})[str(block_size)] = {
                **_aggregate_anchor_samples(samples),
                "scheduled_full_blocks": expected_blocks,
                "scheduled_eligible_replays": int(
                    prefix_schedule["eligible_replays"]
                ),
            }
    summaries = summarize(
        measured,
        live_counts=live_counts,
        schedule_manifest=schedule_manifest,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "fixed-logit-device-tail-component-headroom",
            "production_wiring": False,
            "full_model_graph": False,
            "full_runtime_decision_available": False,
            "capture_provenance": "main_generation_only",
            "fixed_work_tokens": FIXED_WORK_TOKENS,
            "historical_baseline_main_seconds": BASELINE_MAIN_SECONDS,
            "commit": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "warmup": warmup,
            "iterations": iterations,
            "samples_per_prefix": samples_per_prefix,
            "sample_selection": "first_scheduled_blocks_per_prefix",
            "accepted_bucket_replay_counts": {
                str(prefix): count for prefix, count in live_counts.items()
            },
            "captured_prefixes_by_block": {
                str(block_size): sorted(values)
                for block_size, values in anchors.items()
            },
            "measured_prefixes": sorted(int(prefix) for prefix in measured),
            "result_path": run["result_path"],
        },
        "schedule": schedule_manifest,
        "buckets": measured,
        "summaries": summaries,
    }
    if not measured:
        raise RuntimeError("device-tail profiler captured no live main buckets")
    if not all(
        math.isfinite(float(summary["tail_component_runtime_headroom_seconds"]))
        for summary in summaries.values()
    ):
        raise RuntimeError("device-tail component headroom is invalid")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--samples-per-prefix",
        type=int,
        default=DEFAULT_SAMPLES_PER_PREFIX,
    )
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_device_resident_tail(
        args,
        output_path=cli.output_path,
        warmup=cli.warmup,
        iterations=cli.iterations,
        samples_per_prefix=cli.samples_per_prefix,
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metadata"], indent=2))
    print(json.dumps(report["summaries"], indent=2))
    if not any(
        bool(summary["tail_component_headroom_pass"])
        for summary in report["summaries"].values()
    ):
        raise SystemExit("STOP_DEVICE_TAIL_COMPONENT: no block cleared the 5s gate")


if __name__ == "__main__":
    main()
