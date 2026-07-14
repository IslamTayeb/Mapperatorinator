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
    validate_accepted_graph_cache,
)


SCHEMA_VERSION = 1
BLOCK_SIZES = (8, 16)
BASELINE_MAIN_SECONDS = 30.068768849
FIXED_WORK_TOKENS = 8_294
PROMOTION_SAVING_SECONDS = 5.0
ACTIVE_PREFIX_BUCKET_SIZE = 64


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


def _clone_processor_list(processors) -> LogitsProcessorList:
    return LogitsProcessorList(copy.deepcopy(list(processors)))


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


class _RawTailCapture(LogitsProcessor):
    """Transparent first processor that retains one real block per prefix."""

    def __init__(
        self,
        *,
        anchors: dict[int, TailAnchor],
        processors: Any,
        stopping_criteria: Any,
        generation_config: Any,
        block_size: int,
    ) -> None:
        self.anchors = anchors
        self.processors = processors
        self.stopping_criteria = stopping_criteria
        self.generation_config = generation_config
        self.block_size = block_size
        self.active: TailAnchor | None = None

    def _processors_without_self(self) -> LogitsProcessorList:
        return LogitsProcessorList(
            copy.deepcopy(
                [processor for processor in self.processors if processor is not self]
            )
        )

    def _start(self, input_ids: torch.Tensor, scores: torch.Tensor, prefix: int):
        pad = getattr(self.generation_config, "_pad_token_tensor", None)
        eos = getattr(self.generation_config, "_eos_token_tensor", None)
        if not isinstance(pad, torch.Tensor) or pad.numel() != 1:
            raise RuntimeError("tail capture requires one tensor pad token")
        if not isinstance(eos, torch.Tensor) or eos.numel() < 1:
            raise RuntimeError("tail capture requires tensor EOS ids")
        self.active = TailAnchor(
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
        if self.active is not None:
            expected = self.active.input_ids.shape[1] + len(
                self.active.raw_logits_steps
            )
            if prefix != self.active.prefix or cur_len != expected:
                self.active = None
            else:
                self.active.raw_logits_steps.append(scores.detach().clone())
                if len(self.active.raw_logits_steps) == self.block_size:
                    self.anchors.setdefault(prefix, self.active)
                    self.active = None
                return scores
        if prefix in self.anchors:
            return scores
        if (
            prefix - cur_len + 1 >= self.block_size
            and max_length - cur_len >= self.block_size
        ):
            self._start(input_ids, scores, prefix)
            if self.block_size == 1:
                self.anchors.setdefault(prefix, self.active)
                self.active = None
        return scores


def _eager_dynamic_tail(
    anchor: TailAnchor,
    *,
    block_size: int,
    processors,
    stopping_criteria,
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
) -> tuple[
    torch.cuda.CUDAGraph,
    DeviceSequenceState,
    tuple[torch.Tensor, torch.Tensor],
]:
    state = _new_device_state(anchor, block_size)
    processors = _clone_processor_list(anchor.logits_processors)
    criteria = copy.deepcopy(anchor.stopping_criteria)
    template = {
        "sequence": state.sequence.detach().clone(),
        "physical_length": state.physical_length.detach().clone(),
        "logical_length": state.logical_length.detach().clone(),
        "unfinished": state.unfinished.detach().clone(),
    }

    def run():
        state.sequence.copy_(template["sequence"])
        state.physical_length.copy_(template["physical_length"])
        state.logical_length.copy_(template["logical_length"])
        state.unfinished.copy_(template["unfinished"])
        return fixed_block_tail(
            state=state,
            start_length=int(anchor.input_ids.shape[1]),
            raw_logits_steps=anchor.raw_logits_steps[:block_size],
            logits_processor=processors,
            stopping_criteria=criteria,
            do_sample=anchor.do_sample,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    return graph, state, outputs


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


def _time_eager(
    anchor: TailAnchor,
    *,
    block_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    walls = []
    events = []
    for index in range(warmup + iterations):
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
        )
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1_000.0
        event_ms = _event_ms(start_event, end_event)
        if index >= warmup:
            walls.append(wall_ms)
            events.append(event_ms)
    return {
        "wall_ms_per_block": statistics.median(walls),
        "cuda_ms_per_block": statistics.median(events),
        "enqueue_and_sync_ms_per_block": max(
            0.0,
            statistics.median(walls) - statistics.median(events),
        ),
    }


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    anchor: TailAnchor,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    walls = []
    events = []
    for index in range(warmup + iterations):
        _set_rng_state(anchor.rng_state)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - started) * 1_000.0
        event_ms = _event_ms(start_event, end_event)
        if index >= warmup:
            walls.append(wall_ms)
            events.append(event_ms)
    return {
        "wall_ms_per_block": statistics.median(walls),
        "cuda_ms_per_block": statistics.median(events),
        "enqueue_and_sync_ms_per_block": max(
            0.0,
            statistics.median(walls) - statistics.median(events),
        ),
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
    graph, state, graph_outputs = _capture_candidate_graph(
        anchor,
        block_size=block_size,
    )
    _set_rng_state(anchor.rng_state)
    graph.replay()
    torch.cuda.synchronize()
    graph_rng = _rng_state()
    graph_generated, graph_continues = graph_outputs
    start_len = anchor.input_ids.shape[1]
    graph_sequence = state.sequence[:, : start_len + block_size]
    checks = {
        "tokens_exact": bool(torch.equal(eager[1], graph_generated)),
        "sequence_exact": bool(torch.equal(eager[0], graph_sequence)),
        "unfinished_exact": bool(torch.equal(eager[2], state.unfinished)),
        "continue_values_exact": bool(torch.equal(eager[3], graph_continues)),
        "rng_exact": _rng_equal(eager_rng, graph_rng),
        "no_early_stop": bool(torch.all(eager[3][:-1]).item())
        if block_size > 1
        else True,
        "finite_logits": all(
            bool(torch.isfinite(logits).all().item())
            for logits in anchor.raw_logits_steps[:block_size]
        ),
    }
    eager_timing = _time_eager(
        anchor,
        block_size=block_size,
        warmup=warmup,
        iterations=iterations,
    )
    graph_timing = _time_graph(
        graph,
        anchor,
        warmup=warmup,
        iterations=iterations,
    )
    saved_wall_ms = (
        eager_timing["wall_ms_per_block"] - graph_timing["wall_ms_per_block"]
    )
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
        "eager": eager_timing,
        "device_graph": graph_timing,
        "speedup": (
            eager_timing["wall_ms_per_block"]
            / graph_timing["wall_ms_per_block"]
        ),
        "saved_wall_ms_per_block": saved_wall_ms,
        "saved_wall_us_per_token": saved_wall_ms * 1_000.0 / block_size,
    }


def summarize(
    buckets: dict[str, Any],
    *,
    live_counts: dict[int, int],
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
        measured_replays = sum(live_counts.get(prefix, 0) for prefix in measured)
        saved_seconds = sum(
            live_counts[prefix]
            * float(entry["saved_wall_us_per_token"])
            / 1_000_000.0
            for prefix, entry in measured.items()
        )
        summaries[str(block_size)] = {
            "measured_prefixes": sorted(measured),
            "measured_replays": measured_replays,
            "total_replays": total_replays,
            "coverage_fraction": measured_replays / total_replays,
            "projected_main_saved_seconds": saved_seconds,
            "projected_main_seconds": BASELINE_MAIN_SECONDS - saved_seconds,
            "projected_fixed_work_tps": FIXED_WORK_TOKENS
            / (BASELINE_MAIN_SECONDS - saved_seconds),
            "all_correctness_pass": all(entry["pass"] for entry in measured.values()),
            "promotion_saving_seconds": PROMOTION_SAVING_SECONDS,
            "promotion_pass": (
                saved_seconds >= PROMOTION_SAVING_SECONDS
                and measured_replays > 0
                and all(entry["pass"] for entry in measured.values())
            ),
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
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("device-resident tail profiler requires CUDA")
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    _assert_scout_args(args)
    from osuT5.osuT5.inference.optimized.single import engine
    from osuT5.osuT5.inference.runtime_dispatch import attention_runtime_hooks

    anchors: dict[int, TailAnchor] = {}
    original = engine.active_prefix_decode_generate

    def observed_generate(
        model,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        *positional,
        **kwargs,
    ):
        hooks = attention_runtime_hooks()
        if hooks.q1_rope_cache_self_attention_forward is None:
            return original(
                model,
                input_ids,
                logits_processor,
                stopping_criteria,
                generation_config,
                *positional,
                **kwargs,
            )
        capture = _RawTailCapture(
            anchors=anchors,
            processors=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            block_size=max(BLOCK_SIZES),
        )
        logits_processor.insert(0, capture)
        try:
            return original(
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

    engine.active_prefix_decode_generate = observed_generate
    try:
        run = _accepted_main_session_run(args, output_path=output_path)
    finally:
        engine.active_prefix_decode_generate = original
    live_entries = validate_accepted_graph_cache(run["session"].graph_cache)
    live_counts = {
        prefix: int(entry["decode_replays"])
        for prefix, entry in live_entries.items()
    }
    measured: dict[str, Any] = {}
    for prefix, anchor in sorted(anchors.items()):
        if prefix not in live_counts:
            continue
        measured[str(prefix)] = {
            str(block_size): _profile_anchor(
                anchor,
                block_size=block_size,
                warmup=warmup,
                iterations=iterations,
            )
            for block_size in BLOCK_SIZES
            if len(anchor.raw_logits_steps) >= block_size
        }
    summaries = summarize(measured, live_counts=live_counts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "fixed-logit-device-tail-component-ceiling",
            "production_wiring": False,
            "full_model_graph": False,
            "fixed_work_tokens": FIXED_WORK_TOKENS,
            "baseline_main_seconds": BASELINE_MAIN_SECONDS,
            "commit": _git_head(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "manual"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(),
            "warmup": warmup,
            "iterations": iterations,
            "accepted_bucket_replay_counts": {
                str(prefix): count for prefix, count in live_counts.items()
            },
            "captured_prefixes": sorted(anchors),
            "measured_prefixes": sorted(int(prefix) for prefix in measured),
            "result_path": run["result_path"],
        },
        "buckets": measured,
        "summaries": summaries,
    }
    if not measured:
        raise RuntimeError("device-tail profiler captured no live main buckets")
    if not all(
        math.isfinite(float(summary["projected_main_seconds"]))
        and float(summary["projected_main_seconds"]) > 0
        for summary in summaries.values()
    ):
        raise RuntimeError("device-tail projection is invalid")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
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
    )
    cli.report_path.parent.mkdir(parents=True, exist_ok=True)
    cli.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metadata"], indent=2))
    print(json.dumps(report["summaries"], indent=2))
    if not any(
        bool(summary["promotion_pass"])
        for summary in report["summaries"].values()
    ):
        raise SystemExit("STOP_DEVICE_TAIL_COMPONENT: no block cleared the 5s gate")


if __name__ == "__main__":
    main()
