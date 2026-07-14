"""Opt-in real-logit sizing gate for the K4 vocabulary/sampling tail."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import statistics
from typing import Any, Iterator


SCHEMA_VERSION = 1
RESULT_CLASS = "verifier-only-component-sizing"
RNG_POLICY = "counter_request_seed_window_prompt_v2"


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = math.ceil((len(ordered) - 1) * fraction)
    return int(ordered[index])


def _processor_descriptor(processor: Any) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    rows: list[tuple[str, tuple[Any, ...]]] = []
    for item in processor.processors:
        name = type(item).__name__
        attributes: list[Any] = []
        for field in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "types_first",
            "filter_value",
            "min_tokens_to_keep",
        ):
            value = getattr(item, field, None)
            if isinstance(value, (bool, int, float, str)):
                attributes.append((field, value))
        rows.append((name, tuple(attributes)))
    return tuple(rows)


@dataclass(slots=True)
class VocabSample:
    raw_logits: Any
    pre_top_p: Any
    post_top_p: Any
    threshold: float
    selected_token: int
    processor_descriptor: tuple[tuple[str, tuple[Any, ...]], ...]
    source: str

    @property
    def vocab_size(self) -> int:
        return int(self.pre_top_p.numel())

    @property
    def nucleus_size(self) -> int:
        import torch

        return int(torch.isfinite(self.post_top_p).sum().item())

    def validate(self) -> None:
        import torch

        for name, value in (
            ("raw_logits", self.raw_logits),
            ("pre_top_p", self.pre_top_p),
            ("post_top_p", self.post_top_p),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise ValueError(f"{name} must use CPU FP32 storage")
            if value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(f"{name} must have shape [1, vocab]")
        if not (self.raw_logits.shape == self.pre_top_p.shape == self.post_top_p.shape):
            raise ValueError("sample score tensors must share one shape")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("counter threshold must be inside (0, 1)")
        if not 0 <= self.selected_token < self.vocab_size:
            raise ValueError("selected token is outside the vocabulary")
        if self.source not in {"k4_graph_final_step", "eager_remainder"}:
            raise ValueError(f"unsupported sample source {self.source!r}")
        kept = torch.isfinite(self.post_top_p)
        if not bool(kept.any()) or not torch.equal(
            self.post_top_p[kept], self.pre_top_p[kept]
        ):
            raise RuntimeError("top-p must retain at least one unmodified score")
        removed = ~kept
        if bool(removed.any()) and not bool(
            torch.isneginf(self.post_top_p[removed]).all()
        ):
            raise RuntimeError("top-p may only remove scores by writing -inf")


class VocabSamplingObserver:
    """Bounded diagnostic observer; never used for authoritative run timing."""

    def __init__(self, *, max_samples: int) -> None:
        self.max_samples = _positive_int("max_samples", max_samples)
        self.samples: list[VocabSample] = []
        self.graph_buffers: dict[int, dict[str, Any]] = {}
        self.current: dict[str, Any] = {}
        self.current_descriptor: tuple[tuple[str, tuple[Any, ...]], ...] = ()
        self.capture_depth = 0
        self.total_graph_observations = 0
        self.total_eager_observations = 0
        self.wasted_graph_observations = 0
        self.dropped_observations = 0

    def begin_capture(self) -> None:
        if self.capture_depth != 0:
            raise RuntimeError("nested K4 tail capture is unsupported")
        self.capture_depth = 1
        self.current = {}

    def finish_capture(self, parent: Any, *, state: Any) -> None:
        if self.capture_depth != 1:
            raise RuntimeError("K4 tail capture observer is not active")
        self.capture_depth = 0
        required = {"raw_logits", "pre_top_p", "post_top_p", "threshold", "token"}
        if set(self.current) != required:
            raise RuntimeError(
                "K4 captured sampling buffers are incomplete: "
                f"expected {sorted(required)}, got {sorted(self.current)}"
            )
        self.graph_buffers[id(parent)] = {
            **self.current,
            "processor_descriptor": self.current_descriptor,
            "physical_length": state.physical_length,
            "logical_length": state.logical_length,
        }
        self.current = {}

    def abort_capture(self) -> None:
        self.capture_depth = 0
        self.current = {}

    def observe_processor(self, processor: Any, raw_logits: Any) -> None:
        self.current_descriptor = _processor_descriptor(processor)
        self.current["raw_logits"] = raw_logits

    def observe_top_p(self, before: Any, after: Any) -> None:
        if "pre_top_p" in self.current or "post_top_p" in self.current:
            raise RuntimeError("vocabulary scout supports exactly one top-p processor")
        self.current["pre_top_p"] = before
        self.current["post_top_p"] = after

    def observe_sampling(self, threshold: Any, token: Any) -> None:
        self.current["threshold"] = threshold
        self.current["token"] = token
        if self.capture_depth == 0:
            self.total_eager_observations += 1
            self._record(self.current, source="eager_remainder")
            self.current = {}

    def observe_graph_replay(self, parent: Any) -> None:
        buffers = self.graph_buffers.get(id(parent))
        if buffers is None:
            raise RuntimeError("K4 graph replay has no registered sampling buffers")
        self.total_graph_observations += 1
        physical = int(buffers["physical_length"].detach().cpu().item())
        logical = int(buffers["logical_length"].detach().cpu().item())
        if logical < physical:
            self.wasted_graph_observations += 1
            return
        self._record(buffers, source="k4_graph_final_step")

    def _record(self, buffers: dict[str, Any], *, source: str) -> None:
        import torch

        if len(self.samples) >= self.max_samples:
            self.dropped_observations += 1
            return
        sample = VocabSample(
            raw_logits=buffers["raw_logits"].detach().float().cpu().clone(),
            pre_top_p=buffers["pre_top_p"].detach().float().cpu().clone(),
            post_top_p=buffers["post_top_p"].detach().float().cpu().clone(),
            threshold=float(buffers["threshold"].detach().float().cpu().item()),
            selected_token=int(buffers["token"].detach().cpu().item()),
            processor_descriptor=tuple(
                buffers.get("processor_descriptor", self.current_descriptor)
            ),
            source=source,
        )
        sample.validate()
        self.samples.append(sample)


@contextmanager
def install_vocab_sampling_observer(
    observer: VocabSamplingObserver,
) -> Iterator[None]:
    """Patch only the opt-in K4 scout process and restore every callable."""

    import torch
    from transformers import TopPLogitsWarper

    from osuT5.osuT5.inference.optimized.single import k8_runtime

    if not isinstance(observer, VocabSamplingObserver):
        raise TypeError("observer must be VocabSamplingObserver")
    original_processor_call = k8_runtime.K8LogitsProcessor.__call__
    original_top_p_call = TopPLogitsWarper.__call__
    original_sample = k8_runtime._sample_counter
    original_capture = k8_runtime._capture_k8_entry
    original_replay = k8_runtime._ChildGraphSequence.replay

    def processor_call(self, raw_logits):
        observer.observe_processor(self, raw_logits)
        return original_processor_call(self, raw_logits)

    def top_p_call(self, input_ids, scores):
        result = original_top_p_call(self, input_ids, scores)
        observer.observe_top_p(scores, result)
        return result

    def sample_counter(scores, state):
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        threshold = state.counter_uniform().to(dtype=probabilities.dtype)
        cdf = probabilities.cumsum(dim=-1)
        token = (
            torch.searchsorted(cdf[0], threshold)
            .clamp(max=probabilities.shape[-1] - 1)
            .to(dtype=torch.long)
        )
        observer.observe_sampling(threshold, token)
        return token

    def capture(*args, **kwargs):
        observer.begin_capture()
        try:
            entry = original_capture(*args, **kwargs)
            observer.finish_capture(entry.parent, state=entry.state)
            return entry
        except Exception:
            observer.abort_capture()
            raise

    def replay(self):
        result = original_replay(self)
        observer.observe_graph_replay(self)
        return result

    k8_runtime.K8LogitsProcessor.__call__ = processor_call
    TopPLogitsWarper.__call__ = top_p_call
    k8_runtime._sample_counter = sample_counter
    k8_runtime._capture_k8_entry = capture
    k8_runtime._ChildGraphSequence.replay = replay
    try:
        yield
    finally:
        k8_runtime._ChildGraphSequence.replay = original_replay
        k8_runtime._capture_k8_entry = original_capture
        k8_runtime._sample_counter = original_sample
        TopPLogitsWarper.__call__ = original_top_p_call
        k8_runtime.K8LogitsProcessor.__call__ = original_processor_call


def _capture_graph(callable_, *, warmup: int):
    import torch

    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = callable_()
    # Capture records work but does not materialize a fresh value in every
    # output allocation.  Validate the same lifecycle used by production:
    # launch once, then synchronize before inspecting the static output.
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _time_graph(graph: Any, *, iterations: int, rounds: int) -> list[float]:
    import torch

    values: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / iterations)
    return values


def _sampling_with_threshold(scores: Any, threshold: Any):
    import torch

    probabilities = torch.nn.functional.softmax(scores, dim=-1)
    cdf = probabilities.cumsum(dim=-1)
    return (
        torch.searchsorted(cdf[0], threshold)
        .clamp(max=probabilities.shape[-1] - 1)
        .to(dtype=torch.long)
    )


def _top_p_kwargs(descriptor: dict[str, Any]) -> dict[str, Any]:
    top_p = descriptor.get("top_p")
    filter_value = descriptor.get("filter_value")
    min_tokens = descriptor.get("min_tokens_to_keep")
    if not isinstance(top_p, float) or not 0.0 < top_p < 1.0:
        raise RuntimeError("captured TopP descriptor lacks one valid top_p")
    if (
        isinstance(filter_value, bool)
        or not isinstance(filter_value, (int, float))
        or math.isnan(float(filter_value))
    ):
        raise RuntimeError("captured TopP descriptor lacks one valid filter_value")
    if (
        isinstance(min_tokens, bool)
        or not isinstance(min_tokens, int)
        or min_tokens < 1
    ):
        raise RuntimeError(
            "captured TopP descriptor lacks one valid min_tokens_to_keep"
        )
    return {
        "top_p": top_p,
        "filter_value": float(filter_value),
        "min_tokens_to_keep": min_tokens,
    }


def _require_recaptured_tensor(
    name: str,
    actual: Any,
    expected: Any,
    *,
    descriptor: dict[str, Any],
) -> None:
    import torch

    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()
    if torch.equal(actual_cpu, expected_cpu):
        return
    if actual_cpu.shape != expected_cpu.shape:
        raise RuntimeError(
            f"recaptured {name} shape differs from observed real tensor: "
            f"actual={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)} "
            f"descriptor={descriptor!r}"
        )
    equal = actual_cpu == expected_cpu
    mismatch_count = int((~equal).sum().item())
    finite_actual = torch.isfinite(actual_cpu)
    finite_expected = torch.isfinite(expected_cpu)
    finite_mask_mismatch_count = int((finite_actual ^ finite_expected).sum().item())
    jointly_finite = finite_actual & finite_expected
    finite_max_abs = (
        float(
            (actual_cpu[jointly_finite] - expected_cpu[jointly_finite])
            .abs()
            .max()
            .item()
        )
        if bool(jointly_finite.any())
        else 0.0
    )
    raise RuntimeError(
        f"recaptured {name} differs from observed real tensor: "
        f"mismatch_count={mismatch_count} "
        f"finite_mask_mismatch_count={finite_mask_mismatch_count} "
        f"finite_max_abs={finite_max_abs} descriptor={descriptor!r}"
    )


def benchmark_existing_tail(
    samples: list[VocabSample],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
) -> dict[str, Any]:
    """Time the existing graph-captured TopP and counter sampler on real scores."""

    import torch
    from transformers import TopPLogitsWarper

    if not torch.cuda.is_available():
        raise RuntimeError("vocabulary sampling component timing requires CUDA")
    _positive_int("warmup", warmup)
    _positive_int("iterations", iterations)
    _positive_int("rounds", rounds)
    if not samples:
        raise ValueError("component timing requires captured real-logit samples")
    for sample in samples:
        sample.validate()

    sizes = [sample.nucleus_size for sample in samples]
    targets = {
        "min": min(sizes),
        "p50": _percentile(sizes, 0.50),
        "p95": _percentile(sizes, 0.95),
        "max": max(sizes),
    }
    representatives: dict[str, VocabSample] = {}
    used: set[int] = set()
    for label, target in targets.items():
        candidates = sorted(
            enumerate(samples),
            key=lambda pair: (abs(pair[1].nucleus_size - target), pair[0]),
        )
        index, sample = (
            next((index, sample) for index, sample in candidates if index not in used)
            if len(used) < len(samples)
            else candidates[0]
        )
        used.add(index)
        representatives[label] = sample

    rows = []
    device = torch.device("cuda")
    for label, sample in representatives.items():
        top_p_rows = [
            attributes
            for name, attributes in sample.processor_descriptor
            if name == "TopPLogitsWarper"
        ]
        if len(top_p_rows) != 1:
            raise RuntimeError(
                "captured descriptor must contain exactly one TopP warper"
            )
        descriptor = dict(top_p_rows[0])
        top_p_kwargs = _top_p_kwargs(descriptor)
        pre = sample.pre_top_p.to(device=device)
        post = sample.post_top_p.to(device=device)
        threshold = torch.tensor(sample.threshold, dtype=torch.float32, device=device)
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        warper = TopPLogitsWarper(**top_p_kwargs)
        top_p_graph, top_p_output = _capture_graph(
            lambda: warper(input_ids, pre), warmup=warmup
        )
        # Check while this graph's output is still the most recently captured
        # allocation.  Do not depend on its storage surviving later captures.
        _require_recaptured_tensor(
            "TopP output",
            top_p_output,
            sample.post_top_p,
            descriptor=descriptor,
        )
        sampling_graph, sampling_output = _capture_graph(
            lambda: _sampling_with_threshold(post, threshold), warmup=warmup
        )
        if int(sampling_output.cpu().item()) != sample.selected_token:
            raise RuntimeError(
                "recaptured counter sampler differs from observed token: "
                f"actual={int(sampling_output.cpu().item())} "
                f"expected={sample.selected_token} descriptor={descriptor!r}"
            )
        combined_graph, combined_output = _capture_graph(
            lambda: _sampling_with_threshold(warper(input_ids, pre), threshold),
            warmup=warmup,
        )
        torch.cuda.synchronize()
        if int(combined_output.cpu().item()) != sample.selected_token:
            raise RuntimeError(
                "combined TopP+sampler graph differs from observed token: "
                f"actual={int(combined_output.cpu().item())} "
                f"expected={sample.selected_token} descriptor={descriptor!r}"
            )
        top_p_ms = _time_graph(top_p_graph, iterations=iterations, rounds=rounds)
        sampling_ms = _time_graph(sampling_graph, iterations=iterations, rounds=rounds)
        combined_ms = _time_graph(combined_graph, iterations=iterations, rounds=rounds)
        rows.append(
            {
                "representative": label,
                "nucleus_size": sample.nucleus_size,
                "top_p_ms_per_step_rounds": top_p_ms,
                "sampling_ms_per_step_rounds": sampling_ms,
                "combined_ms_per_step_rounds": combined_ms,
                "top_p_ms_per_step_worst": max(top_p_ms),
                "sampling_ms_per_step_worst": max(sampling_ms),
                "combined_ms_per_step_worst": max(combined_ms),
                "selected_token_exact": True,
                "top_p_mask_exact": True,
            }
        )
    return {
        "warmup": warmup,
        "iterations": iterations,
        "rounds": rounds,
        "reciprocal_order": ["top_p", "sampling", "combined"],
        "representatives": rows,
        "worst_combined_ms_per_step": max(
            row["combined_ms_per_step_worst"] for row in rows
        ),
    }


def summarize(
    observer: VocabSamplingObserver,
    component: dict[str, Any],
    *,
    fixed_main_steps: int,
    fixed_timing_steps: int,
    mixed_projection_ms_per_step: float,
    promotion_threshold_seconds: float,
) -> dict[str, Any]:
    fixed_main_steps = _positive_int("fixed_main_steps", fixed_main_steps)
    fixed_timing_steps = _positive_int("fixed_timing_steps", fixed_timing_steps)
    mixed_projection_ms_per_step = _finite_nonnegative(
        "mixed_projection_ms_per_step", mixed_projection_ms_per_step
    )
    promotion_threshold_seconds = _finite_nonnegative(
        "promotion_threshold_seconds", promotion_threshold_seconds
    )
    if not observer.samples:
        raise ValueError("vocabulary scout captured no samples")
    combined_ms = _finite_nonnegative(
        "worst_combined_ms_per_step", component.get("worst_combined_ms_per_step")
    )
    nucleus_sizes = [sample.nucleus_size for sample in observer.samples]
    vocab_sizes = {sample.vocab_size for sample in observer.samples}
    if len(vocab_sizes) != 1:
        raise RuntimeError(f"captured vocabulary size changed: {sorted(vocab_sizes)}")
    device_tail_main = combined_ms * fixed_main_steps / 1000.0
    projection_main = mixed_projection_ms_per_step * fixed_main_steps / 1000.0
    ideal_main = device_tail_main + projection_main
    request_steps = fixed_main_steps + fixed_timing_steps
    ideal_request = (
        (combined_ms + mixed_projection_ms_per_step) * request_steps / 1000.0
    )
    descriptors = sorted(
        {repr(sample.processor_descriptor) for sample in observer.samples}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "result_class": RESULT_CLASS,
        "production_wiring_changed": False,
        "authoritative_generation_timing": False,
        "observer_overhead_charged_to_performance": False,
        "rng_policy": RNG_POLICY,
        "rng_exact_with_k4_control": True,
        "sample_coverage": {
            "captured_samples": len(observer.samples),
            "max_samples": observer.max_samples,
            "graph_final_step_observations": observer.total_graph_observations,
            "eager_remainder_observations": observer.total_eager_observations,
            "post_eos_graph_observations_excluded": (
                observer.wasted_graph_observations
            ),
            "dropped_observations": observer.dropped_observations,
            "k4_graph_sampling_stride": 4,
            "graph_sample_semantics": (
                "final_physical step per K4 block, excluding blocks whose final "
                "step is after logical EOS"
            ),
        },
        "distribution": {
            "vocab_size": next(iter(vocab_sizes)),
            "nucleus_size_min": min(nucleus_sizes),
            "nucleus_size_p50": _percentile(nucleus_sizes, 0.50),
            "nucleus_size_p95": _percentile(nucleus_sizes, 0.95),
            "nucleus_size_max": max(nucleus_sizes),
            "processor_descriptors": descriptors,
        },
        "component_timing": component,
        "fixed_work_ceiling": {
            "fixed_main_steps": fixed_main_steps,
            "fixed_timing_steps": fixed_timing_steps,
            "existing_top_p_sampling_ms_per_step": combined_ms,
            "mixed_projection_ms_per_step": mixed_projection_ms_per_step,
            "ideal_main_top_p_sampling_seconds": device_tail_main,
            "ideal_main_projection_seconds": projection_main,
            "ideal_main_total_seconds": ideal_main,
            "ideal_complete_request_total_seconds": ideal_request,
            "promotion_threshold_seconds": promotion_threshold_seconds,
            "main_ceiling_clears_threshold": ideal_main >= promotion_threshold_seconds,
            "request_ceiling_clears_threshold": ideal_request
            >= promotion_threshold_seconds,
            "ceiling_interpretation": (
                "impossible zero-cost ceiling; any candidate retains nonzero work"
            ),
        },
        "decision": (
            "retain_for_candidate_kernel"
            if ideal_main >= promotion_threshold_seconds
            else "stop_below_main_component_gate"
        ),
    }
