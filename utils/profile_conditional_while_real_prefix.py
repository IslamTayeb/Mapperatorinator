"""Gate CUDA conditional WHILE on one real accepted SALVALAI prefix graph."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from osuT5.osuT5.inference.optimized.single.conditional_while_scout import (  # noqa: E402
    ConditionalGraph,
)
from osuT5.osuT5.inference.optimized.single.decode_loop import (  # noqa: E402
    _clone_static_graph_inputs,
)
from osuT5.osuT5.inference.optimized.single.k8_runtime import (  # noqa: E402
    _ChildGraphSequence,
    _update_static_model_inputs,
)
from osuT5.osuT5.runtime_profiling import generation_profile_context  # noqa: E402
from utils.profile_native_prefix_dtype_scout import (  # noqa: E402
    _accepted_main_session_run,
    _all_cache_snapshots,
    _assert_scout_args,
    _cache_from_static_inputs,
    _cache_tensor,
    _load_args,
    _restore_all_cache,
    validate_accepted_graph_cache,
)


SCHEMA_VERSION = 1
VARIANTS = ("k1", "k4", "k8", "while")
BLOCK_SIZE = {"k1": 1, "k4": 4, "k8": 8, "while": None}
ACCEPTED_DISPATCH = {
    "precision": "fp32",
    "q1_bmm_cross_attention": True,
    "native_q1_self_attention": True,
    "native_q1_rope_cache_self_attention": True,
}


@dataclass(slots=True)
class LoopState:
    tokens: torch.Tensor
    physical_step: torch.Tensor
    logical_step: torch.Tensor
    unfinished: torch.Tensor
    forced_stop_after: torch.Tensor
    stop_step: torch.Tensor
    pad_token: torch.Tensor

    @classmethod
    def allocate(cls, *, max_steps: int, device: torch.device) -> "LoopState":
        if max_steps < 8:
            raise ValueError("real-prefix loop requires at least eight steps")
        return cls(
            tokens=torch.zeros((1, max_steps), dtype=torch.long, device=device),
            physical_step=torch.zeros((1,), dtype=torch.long, device=device),
            logical_step=torch.full(
                (1,), max_steps, dtype=torch.long, device=device
            ),
            unfinished=torch.ones((1,), dtype=torch.bool, device=device),
            forced_stop_after=torch.full(
                (1,), max_steps, dtype=torch.long, device=device
            ),
            stop_step=torch.full((1,), max_steps, dtype=torch.long, device=device),
            pad_token=torch.zeros((1,), dtype=torch.long, device=device),
        )

    def reset(self, *, fixed_steps: int, forced_stop_after: int) -> None:
        if not 1 <= forced_stop_after <= fixed_steps <= self.tokens.shape[1]:
            raise ValueError("forced stop must be inside the fixed-work capacity")
        self.tokens.zero_()
        self.physical_step.zero_()
        self.logical_step.fill_(fixed_steps)
        self.unfinished.fill_(True)
        self.forced_stop_after.fill_(forced_stop_after)
        self.stop_step.fill_(fixed_steps)


def _tail_step(
    *,
    logits: torch.Tensor,
    state: LoopState,
    static_inputs: dict[str, Any],
) -> torch.Tensor:
    token = torch.argmax(logits[:, -1, :], dim=-1).to(dtype=torch.long)
    active = torch.where(state.unfinished, token, state.pad_token)
    state.tokens.scatter_(
        1,
        state.physical_step.view(1, 1),
        active.view(1, 1),
    )
    next_step = state.physical_step + 1
    just_stopped = state.unfinished & (
        next_step >= state.forced_stop_after
    )
    state.logical_step.copy_(
        torch.where(just_stopped, next_step, state.logical_step)
    )
    state.unfinished.bitwise_and_(~just_stopped)
    state.physical_step.copy_(next_step)
    _update_static_model_inputs(static_inputs, active)
    return active


def _variant_launches(variant: str, logical_steps: int) -> int:
    if variant not in BLOCK_SIZE:
        raise ValueError(f"unknown conditional gate variant {variant!r}")
    if logical_steps <= 0:
        raise ValueError("logical_steps must be positive")
    block = BLOCK_SIZE[variant]
    return 1 if block is None else math.ceil(logical_steps / block)


def _hash_tensors(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        copy = value.detach().contiguous().cpu()
        digest.update(str(tuple(copy.shape)).encode())
        digest.update(str(copy.dtype).encode())
        digest.update(copy.numpy().tobytes())
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _accepted_context(prefix: int):
    return generation_profile_context(
        detail_ranges=False,
        active_prefix_self_attention_length=prefix,
        q1_bmm_cross_attention=True,
        native_q1_self_attention=True,
        native_q1_rope_cache_self_attention=True,
    )


def _restore(
    *,
    state: LoopState,
    fixed_steps: int,
    forced_stop_after: int,
    static_snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    cache: Any,
    cache_snapshots: list[Any],
    device: torch.device,
) -> None:
    torch.cuda.synchronize(device)
    _restore_all_cache(cache, cache_snapshots)
    for tensor, snapshot in static_snapshots:
        tensor.copy_(snapshot)
    state.reset(
        fixed_steps=fixed_steps,
        forced_stop_after=forced_stop_after,
    )
    torch.cuda.synchronize(device)


def _cache_hash(
    cache: Any,
    *,
    start_position: int,
    steps: int | None,
) -> str:
    values = []
    for layer_idx, _ in enumerate(cache.self_attention_cache.layers):
        for name in ("keys", "values"):
            tensor = _cache_tensor(cache, "self", layer_idx, name)
            if steps is None:
                values.append(tensor)
            else:
                values.append(
                    tensor[..., start_position : start_position + steps, :]
                )
    return _hash_tensors(values)


def _observation(
    *,
    state: LoopState,
    cache: Any,
    start_cache_position: int,
) -> dict[str, Any]:
    physical = int(state.physical_step.item())
    logical = int(state.logical_step.item())
    unfinished = bool(state.unfinished.item())
    visible = state.tokens[:, :logical]
    return {
        "physical_steps": physical,
        "logical_steps": logical,
        "wasted_steps": physical - logical,
        "unfinished": unfinished,
        "visible_tokens": visible.detach().cpu().reshape(-1).tolist(),
        "visible_token_sha256": _hash_tensors([visible]),
        "logical_cache_sha256": _cache_hash(
            cache,
            start_position=start_cache_position,
            steps=logical,
        ),
        "full_cache_sha256": _cache_hash(
            cache,
            start_position=start_cache_position,
            steps=None,
        ),
    }


def _measure(
    replay: Callable[[], None],
    *,
    launches: int,
    observe: Callable[[], dict[str, Any]],
    memory_device: torch.device,
) -> dict[str, Any]:
    if launches < 1:
        raise ValueError("launches must be positive")
    before = int(torch.cuda.memory_allocated(memory_device))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(launches):
        replay()
    end.record()
    end.synchronize()
    wall = time.perf_counter() - wall_start
    after = int(torch.cuda.memory_allocated(memory_device))
    return {
        "parent_launches": launches,
        "cuda_event_ms": float(start.elapsed_time(end)),
        "host_wall_seconds": wall,
        "memory_before_bytes": before,
        "memory_after_bytes": after,
        "memory_stable": before == after,
        **observe(),
    }


def _compare_case(
    rows: list[dict[str, Any]],
    *,
    expected_logical: int,
    fixed_steps: int,
) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    for row in rows:
        by_variant[row["variant"]].append(row)
    if any(len(values) != 2 for values in by_variant.values()):
        raise RuntimeError("each real-prefix variant requires reciprocal repeats")
    repeatable = {
        name: all(
            row["visible_token_sha256"] == values[0]["visible_token_sha256"]
            and row["logical_cache_sha256"] == values[0]["logical_cache_sha256"]
            and row["physical_steps"] == values[0]["physical_steps"]
            and row["logical_steps"] == values[0]["logical_steps"]
            for row in values
        )
        for name, values in by_variant.items()
    }
    first = {name: values[0] for name, values in by_variant.items()}
    visible_exact = len({row["visible_token_sha256"] for row in first.values()}) == 1
    logical_cache_exact = len({row["logical_cache_sha256"] for row in first.values()}) == 1
    logical_exact = all(
        row["logical_steps"] == expected_logical for row in first.values()
    )
    while_no_waste = first["while"]["physical_steps"] == expected_logical
    k1_no_waste = first["k1"]["physical_steps"] == expected_logical
    while_k1_full_cache_exact = (
        first["while"]["full_cache_sha256"]
        == first["k1"]["full_cache_sha256"]
    )
    memory_stable = all(row["memory_stable"] for row in rows)
    reciprocal_times = {
        name: max(row["cuda_event_ms"] for row in values)
        for name, values in by_variant.items()
    }
    physical_expected = {
        name: (
            expected_logical
            if name in {"k1", "while"}
            else math.ceil(expected_logical / int(BLOCK_SIZE[name]))
            * int(BLOCK_SIZE[name])
        )
        for name in VARIANTS
    }
    physical_exact = all(
        first[name]["physical_steps"] == physical_expected[name]
        for name in VARIANTS
    )
    passed = all((
        all(repeatable.values()),
        visible_exact,
        logical_cache_exact,
        logical_exact,
        physical_exact,
        while_no_waste,
        k1_no_waste,
        while_k1_full_cache_exact,
        memory_stable,
    ))
    return {
        "pass": passed,
        "expected_logical_steps": expected_logical,
        "fixed_work_capacity": fixed_steps,
        "repeatable": repeatable,
        "visible_tokens_exact": visible_exact,
        "logical_cache_exact": logical_cache_exact,
        "logical_steps_exact": logical_exact,
        "physical_steps_exact": physical_exact,
        "expected_physical_steps": physical_expected,
        "while_no_post_stop_waste": while_no_waste,
        "k1_no_post_stop_waste": k1_no_waste,
        "while_k1_full_cache_exact": while_k1_full_cache_exact,
        "memory_stable": memory_stable,
        "reciprocal_cuda_ms": reciprocal_times,
        "while_speedup_vs_pct": {
            name: (
                reciprocal_times[name] - reciprocal_times["while"]
            ) / reciprocal_times[name] * 100.0
            for name in ("k1", "k4", "k8")
        },
    }


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected real-prefix conditional schema")
    metadata = report.get("metadata")
    fixed = report.get("fixed_work")
    forced = report.get("forced_stop")
    if not isinstance(metadata, dict) or not isinstance(fixed, dict):
        raise TypeError("real-prefix report is incomplete")
    if metadata.get("accepted_dispatch") != ACCEPTED_DISPATCH:
        raise RuntimeError("accepted dispatch metadata changed")
    if not bool(fixed.get("pass")):
        raise RuntimeError("fixed-work conditional gate failed")
    if not isinstance(forced, dict) or not forced:
        raise RuntimeError("forced-stop conditional evidence is missing")
    if not all(bool(value.get("pass")) for value in forced.values()):
        raise RuntimeError("forced-stop conditional gate failed")
    if not bool(report.get("decision", {}).get("pass")):
        raise RuntimeError("real-prefix decision is inconsistent")


@torch.no_grad()
def profile_real_prefix(
    args,
    *,
    capture_output_path: Path,
    source_prefix: int,
    active_prefix: int,
    fixed_steps: int,
    forced_stops: tuple[int, ...],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("real-prefix conditional gate requires CUDA")
    if fixed_steps != 8:
        raise ValueError("first real-prefix gate is deliberately fixed at eight steps")
    if not forced_stops or any(not 1 <= value <= fixed_steps for value in forced_stops):
        raise ValueError("forced stops must be inside the fixed-work capacity")
    _assert_scout_args(args)
    run = _accepted_main_session_run(args, output_path=capture_output_path)
    model = run["model"]
    model.eval()
    accepted = validate_accepted_graph_cache(run["session"].graph_cache)
    if source_prefix not in accepted or active_prefix not in accepted:
        raise RuntimeError(
            "accepted run did not capture requested source/active prefixes "
            f"{source_prefix}/{active_prefix}"
        )
    if source_prefix >= active_prefix:
        raise ValueError("source prefix must precede the active test prefix")
    accepted_entry = accepted[source_prefix]
    static_inputs = _clone_static_graph_inputs(accepted_entry["static_inputs"])
    cache = _cache_from_static_inputs(static_inputs)
    cache_position = static_inputs.get("cache_position")
    if not isinstance(cache_position, torch.Tensor) or cache_position.numel() != 1:
        raise TypeError("real-prefix cache_position must be one tensor element")
    device = cache_position.device
    start_cache_position = int(cache_position.item())
    if start_cache_position + fixed_steps > active_prefix + 1:
        raise RuntimeError(
            "requested fixed work crosses the accepted active-prefix boundary: "
            f"start={start_cache_position}, steps={fixed_steps}, "
            f"active_prefix={active_prefix}"
        )
    state = LoopState.allocate(max_steps=fixed_steps, device=device)
    cache_snapshots = _all_cache_snapshots(cache, cache_position)
    static_snapshots = [
        (value, value.detach().clone())
        for value in static_inputs.values()
        if isinstance(value, torch.Tensor)
    ]

    capture_started = time.perf_counter()
    with torch.cuda.device(device):
        model_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(model_graph):
            with _accepted_context(active_prefix):
                outputs = model(**static_inputs, return_dict=True)
        tail_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(tail_graph):
            _tail_step(logits=outputs.logits, state=state, static_inputs=static_inputs)
        _restore(
            state=state,
            fixed_steps=fixed_steps,
            forced_stop_after=fixed_steps,
            static_snapshots=static_snapshots,
            cache=cache,
            cache_snapshots=cache_snapshots,
            device=device,
        )
        parents: dict[str, Any] = {
            "k1": _ChildGraphSequence.build(
                model_graph, tail_graph, block_size=1, device=device
            ),
            "k4": _ChildGraphSequence.build(
                model_graph, tail_graph, block_size=4, device=device
            ),
            "k8": _ChildGraphSequence.build(
                model_graph, tail_graph, block_size=8, device=device
            ),
            "while": ConditionalGraph.while_children(
                model_graph,
                tail_graph,
                unfinished=state.unfinished,
                physical_step=state.physical_step,
                stop_step=state.stop_step,
            ),
        }
        capture_seconds = time.perf_counter() - capture_started
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(device))

        def run_case(forced_stop_after: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            rows = []
            for name in VARIANTS:
                _restore(
                    state=state,
                    fixed_steps=fixed_steps,
                    forced_stop_after=forced_stop_after,
                    static_snapshots=static_snapshots,
                    cache=cache,
                    cache_snapshots=cache_snapshots,
                    device=device,
                )
                for _ in range(_variant_launches(name, forced_stop_after)):
                    parents[name].replay()
                torch.cuda.synchronize(device)
            for order in (VARIANTS, tuple(reversed(VARIANTS))):
                for name in order:
                    _restore(
                        state=state,
                        fixed_steps=fixed_steps,
                        forced_stop_after=forced_stop_after,
                        static_snapshots=static_snapshots,
                        cache=cache,
                        cache_snapshots=cache_snapshots,
                        device=device,
                    )
                    row = _measure(
                        parents[name].replay,
                        launches=_variant_launches(name, forced_stop_after),
                        observe=lambda: _observation(
                            state=state,
                            cache=cache,
                            start_cache_position=start_cache_position,
                        ),
                        memory_device=device,
                    )
                    row["variant"] = name
                    row["order"] = list(order)
                    rows.append(row)
            return rows, _compare_case(
                rows,
                expected_logical=forced_stop_after,
                fixed_steps=fixed_steps,
            )

        try:
            fixed_rows, fixed_summary = run_case(fixed_steps)
            forced_rows = {}
            forced_summary = {}
            for stop in forced_stops:
                rows, summary = run_case(stop)
                forced_rows[str(stop)] = rows
                forced_summary[str(stop)] = summary
        finally:
            parents["while"].close()
            for name in ("k8", "k4", "k1"):
                parents[name].close()
            _restore(
                state=state,
                fixed_steps=fixed_steps,
                forced_stop_after=fixed_steps,
                static_snapshots=static_snapshots,
                cache=cache,
                cache_snapshots=cache_snapshots,
                device=device,
            )

    passed = bool(fixed_summary["pass"]) and all(
        bool(value["pass"]) for value in forced_summary.values()
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "result_class": "real-prefix-conditional-while-component-gate",
            "production_wiring": False,
            "source": "accepted full SALVALAI graph inputs",
            "source_prefix": source_prefix,
            "active_prefix": active_prefix,
            "start_cache_position": start_cache_position,
            "fixed_steps": fixed_steps,
            "forced_stops": list(forced_stops),
            "accepted_dispatch": ACCEPTED_DISPATCH,
            "child_graphs": ["one accepted model step", "one argmax tail step"],
            "condition_owner": "device updater kernel after tail child",
            "counter_rng": False,
            "sampling_scope": "greedy component control only",
            "capture_seconds": capture_seconds,
            "peak_vram_bytes": peak_vram_bytes,
            "commit": _git_head(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "fixed_work": fixed_summary,
        "fixed_work_rounds": fixed_rows,
        "forced_stop": forced_summary,
        "forced_stop_rounds": forced_rows,
        "decision": {
            "pass": passed,
            "next_gate": (
                "natural 15-second candidate reciprocal"
                if passed
                else "stop before natural decode integration"
            ),
        },
    }
    _validate_report(report)
    return report


def _text_report(report: dict[str, Any]) -> str:
    fixed = report["fixed_work"]
    lines = [
        f"real_prefix_conditional_gate={'PASS' if report['decision']['pass'] else 'FAIL'}",
        f"source_prefix={report['metadata']['source_prefix']}",
        f"active_prefix={report['metadata']['active_prefix']}",
        f"start_cache_position={report['metadata']['start_cache_position']}",
        "fixed_reciprocal_cuda_ms="
        + json.dumps(fixed["reciprocal_cuda_ms"], sort_keys=True),
        "fixed_while_speedup_vs_pct="
        + json.dumps(fixed["while_speedup_vs_pct"], sort_keys=True),
    ]
    for stop, value in sorted(report["forced_stop"].items(), key=lambda row: int(row[0])):
        lines.append(
            f"forced_stop={stop} pass={value['pass']} "
            f"expected_physical_steps={json.dumps(value['expected_physical_steps'], sort_keys=True)}"
        )
    lines.append(f"next_gate={report['decision']['next_gate']}")
    return "\n".join(lines) + "\n"


def _parse_stops(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("forced stops must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("forced stops must be non-empty and unique")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="profile_salvalai")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-output-path", type=Path, required=True)
    parser.add_argument("--source-prefix", type=int, default=576)
    parser.add_argument("--active-prefix", type=int, default=640)
    parser.add_argument("--fixed-steps", type=int, default=8)
    parser.add_argument("--forced-stops", type=_parse_stops, default=(1, 3, 7))
    parser.add_argument("overrides", nargs="*")
    cli = parser.parse_args()
    overrides = list(cli.overrides)
    if not any(value.startswith("output_path=") for value in overrides):
        overrides.append(f"output_path={cli.capture_output_path}")
    args = _load_args(cli.config_name, overrides)
    report = profile_real_prefix(
        args,
        capture_output_path=cli.capture_output_path,
        source_prefix=cli.source_prefix,
        active_prefix=cli.active_prefix,
        fixed_steps=cli.fixed_steps,
        forced_stops=cli.forced_stops,
    )
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    (cli.output_dir / "real-prefix.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    text = _text_report(report)
    (cli.output_dir / "real-prefix.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
