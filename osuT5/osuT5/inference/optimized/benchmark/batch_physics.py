"""Validated measurement records for the merged-batch versus B1-lane scout."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..exactness import ExactnessResultClass


class BatchPhysicsExecutionFamily(str, Enum):
    MERGED_BATCH = "merged_batch"
    B1_LANE_POOL = "b1_lane_pool"


MERGED_STATE_OWNERSHIP_CONTRACT = "merged_fixed_slots_with_private_request_state"
LANE_STATE_OWNERSHIP_CONTRACT = (
    "independent_b1_stream_graph_buffer_generator_cache_cublas_workspace"
)


@dataclass(frozen=True)
class BatchPhysicsPlan:
    """The bounded execution-shape matrix required before scheduler work."""

    merged_batch_sizes: tuple[int, ...] = (1, 2, 5, 8)
    b1_lane_counts: tuple[int, ...] = (1, 2, 3, 4)
    result_class: ExactnessResultClass = ExactnessResultClass.EXACT_OUTPUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_class", ExactnessResultClass(self.result_class))
        if any(size <= 0 for size in self.merged_batch_sizes):
            raise ValueError("merged_batch_sizes must be positive.")
        if any(count <= 0 for count in self.b1_lane_counts):
            raise ValueError("b1_lane_counts must be positive.")

    def as_dict(self) -> dict[str, object]:
        return {
            "merged_batch_sizes": list(self.merged_batch_sizes),
            "b1_lane_counts": list(self.b1_lane_counts),
            "result_class": self.result_class.value,
            "status": "merged_one_token_verifier_only",
            "implemented_merged_one_token_batch_sizes": [1, 2, 5, 8],
            "planned_merged_batch_sizes": [],
            "lane_pool_status": "l1_validated_l2_verifier_pending_gpu",
            "validated_b1_lane_counts": [1],
            "implemented_pending_gpu_b1_lane_counts": [2],
            "planned_b1_lane_counts": [3, 4],
            "required_observation_fields": [
                "execution_family",
                "parallelism",
                "state_ownership_contract",
                "workload_contract_hash",
                "result_class",
                "seeds",
                "generated_tokens",
                "scheduler_wall_seconds",
                "model_seconds",
                "cuda_seconds",
                "peak_memory_bytes",
                "graph_capture_count",
                "graph_replay_count",
                "active_batch_size_histogram",
                "token_hashes",
                "final_rng_state_hashes",
                "stop_reasons",
            ],
            "bitwise_required_observation_fields": [
                "intermediate_state_hashes",
                "cache_state_hashes",
            ],
        }


@dataclass(frozen=True)
class BatchPhysicsObservation:
    """One untraced GPU observation with exactness and occupancy evidence."""

    execution_family: BatchPhysicsExecutionFamily
    parallelism: int
    state_ownership_contract: str
    workload_contract_hash: str
    result_class: ExactnessResultClass
    seeds: Mapping[str, int]
    generated_tokens: int
    scheduler_wall_seconds: float
    model_seconds: float
    cuda_seconds: float
    peak_memory_bytes: int
    graph_capture_count: int
    graph_replay_count: int
    active_batch_size_histogram: Mapping[int, int]
    token_hashes: Mapping[str, str]
    final_rng_state_hashes: Mapping[str, str]
    stop_reasons: Mapping[str, str]
    intermediate_state_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    cache_state_hashes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_family", BatchPhysicsExecutionFamily(self.execution_family))
        object.__setattr__(self, "result_class", ExactnessResultClass(self.result_class))
        if self.parallelism <= 0:
            raise ValueError("parallelism must be positive.")
        allowed_parallelism = (
            (1, 2, 5, 8)
            if self.execution_family is BatchPhysicsExecutionFamily.MERGED_BATCH
            else (1, 2, 3, 4)
        )
        if self.parallelism not in allowed_parallelism:
            raise ValueError(
                f"parallelism {self.parallelism} is outside the bounded "
                f"{self.execution_family.value} scout matrix {allowed_parallelism}."
            )
        required_state_contract = (
            MERGED_STATE_OWNERSHIP_CONTRACT
            if self.execution_family is BatchPhysicsExecutionFamily.MERGED_BATCH
            else LANE_STATE_OWNERSHIP_CONTRACT
        )
        if self.state_ownership_contract != required_state_contract:
            raise ValueError(
                f"{self.execution_family.value} requires state_ownership_contract="
                f"{required_state_contract!r}."
            )
        if not self.workload_contract_hash:
            raise ValueError("workload_contract_hash must be non-empty.")
        if self.generated_tokens <= 0:
            raise ValueError("generated_tokens must be positive.")
        for name in ("scheduler_wall_seconds", "model_seconds", "cuda_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        # Model-only and complete-step timings may come from separate replay
        # passes, so small measurement jitter can make model_seconds exceed the
        # later complete-step wall. Keep a bounded 5% consistency allowance;
        # larger inversions indicate incompatible intervals or malformed data.
        if self.model_seconds > self.scheduler_wall_seconds * 1.05:
            raise ValueError(
                "model_seconds cannot exceed scheduler_wall_seconds by more than 5%."
            )
        # CUDA time is measured inside the complete-step wall interval and
        # retains strict containment.
        if self.cuda_seconds > self.scheduler_wall_seconds:
            raise ValueError("cuda_seconds cannot exceed scheduler_wall_seconds.")
        if self.peak_memory_bytes <= 0:
            raise ValueError("peak_memory_bytes must be positive.")
        for name in ("graph_capture_count", "graph_replay_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not self.seeds or any(
                not request_id
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                for request_id, seed in self.seeds.items()
        ):
            raise TypeError("seeds must map non-empty request IDs to integer seeds.")
        if not self.active_batch_size_histogram or any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                for size, count in self.active_batch_size_histogram.items()
        ):
            raise ValueError("active_batch_size_histogram must contain positive integer counts.")
        if max(self.active_batch_size_histogram) > self.parallelism:
            raise ValueError("active batch size cannot exceed execution parallelism.")
        request_ids = set(self.seeds)
        if (
                not request_ids
                or request_ids != set(self.token_hashes)
                or request_ids != set(self.final_rng_state_hashes)
                or request_ids != set(self.stop_reasons)
        ):
            raise ValueError(
                "seeds, token_hashes, final_rng_state_hashes, and stop_reasons "
                "must cover the same requests."
            )
        if any(not key or not value for mapping in (
                self.token_hashes,
                self.final_rng_state_hashes,
                self.stop_reasons,
        )
               for key, value in mapping.items()):
            raise ValueError("request IDs, exactness hashes, and stop reasons must be non-empty.")
        bitwise_required = self.result_class is ExactnessResultClass.BITWISE_CALCULATION_EXACT
        self._validate_per_request_state_hashes(
            "intermediate_state_hashes",
            self.intermediate_state_hashes,
            request_ids,
            required=bitwise_required,
        )
        self._validate_per_request_state_hashes(
            "cache_state_hashes",
            self.cache_state_hashes,
            request_ids,
            required=bitwise_required,
        )

    @property
    def scheduler_wall_tokens_per_second(self) -> float:
        return self.generated_tokens / self.scheduler_wall_seconds

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["execution_family"] = self.execution_family.value
        result["result_class"] = self.result_class.value
        result["active_batch_size_histogram"] = {
            str(size): count for size, count in sorted(self.active_batch_size_histogram.items())
        }
        result["scheduler_wall_tokens_per_second"] = self.scheduler_wall_tokens_per_second
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatchPhysicsObservation":
        """Validate one JSON-compatible observation emitted by a GPU probe."""

        values = dict(payload)
        values.pop("scheduler_wall_tokens_per_second", None)
        values["active_batch_size_histogram"] = {
            int(size): int(count)
            for size, count in values["active_batch_size_histogram"].items()
        }
        values["seeds"] = {
            str(request_id): seed
            for request_id, seed in values["seeds"].items()
        }
        return cls(**values)

    @staticmethod
    def _validate_per_request_state_hashes(
            field_name: str,
            hashes: Mapping[str, Mapping[str, str]],
            request_ids: set[str],
            *,
            required: bool,
    ) -> None:
        if not hashes:
            if required:
                raise ValueError(
                    f"bitwise-calculation-exact requires per-request {field_name}."
                )
            return
        if set(hashes) != request_ids:
            raise ValueError(f"{field_name} must cover every request when provided.")
        if any(
                not request_id
                or not component_hashes
                or any(not name or not value for name, value in component_hashes.items())
                for request_id, component_hashes in hashes.items()
        ):
            raise ValueError(
                f"{field_name} must contain non-empty component names and hashes per request."
            )


@dataclass(frozen=True)
class SamplingComponentMeasurement:
    """One isolated, non-additive rowwise sampling component measurement."""

    component: str
    repeats: int
    wall_seconds: float
    cuda_seconds: float
    includes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "includes", tuple(self.includes))
        if not self.component:
            raise ValueError("component must be non-empty.")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive.")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive.")
        if not math.isfinite(self.cuda_seconds) or self.cuda_seconds < 0:
            raise ValueError("cuda_seconds must be finite and non-negative.")
        if self.cuda_seconds > self.wall_seconds:
            raise ValueError("cuda_seconds cannot exceed synchronized wall_seconds.")
        if not self.includes or any(not item for item in self.includes):
            raise ValueError("includes must contain non-empty operation labels.")

    @property
    def wall_seconds_per_step(self) -> float:
        return self.wall_seconds / self.repeats

    @property
    def cuda_seconds_per_step(self) -> float:
        return self.cuda_seconds / self.repeats

    @property
    def host_launch_and_sync_seconds_per_step(self) -> float:
        return max(0.0, self.wall_seconds_per_step - self.cuda_seconds_per_step)

    def as_dict(self, *, required_saving_seconds_per_step: float) -> dict[str, object]:
        return {
            "component": self.component,
            "repeats": self.repeats,
            "includes": list(self.includes),
            "wall_seconds": self.wall_seconds,
            "cuda_seconds": self.cuda_seconds,
            "wall_seconds_per_step": self.wall_seconds_per_step,
            "cuda_seconds_per_step": self.cuda_seconds_per_step,
            "host_launch_and_sync_seconds_per_step": (
                self.host_launch_and_sync_seconds_per_step
            ),
            "fantasy_free_wall_clears_required_saving": (
                self.wall_seconds_per_step >= required_saving_seconds_per_step
            ),
            "fantasy_free_cuda_clears_required_saving": (
                self.cuda_seconds_per_step >= required_saving_seconds_per_step
            ),
        }


def summarize_sampling_gap_target(
        *,
        batch_size: int,
        target_tokens_per_second: float,
        complete_wall_seconds_per_step: float,
        model_seconds_per_step: float,
) -> dict[str, float | int | bool]:
    """Compute the exact step-time saving required to reach the B8 target."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if target_tokens_per_second <= 0:
        raise ValueError("target_tokens_per_second must be positive.")
    if complete_wall_seconds_per_step <= 0 or model_seconds_per_step <= 0:
        raise ValueError("step times must be positive.")
    if model_seconds_per_step > complete_wall_seconds_per_step:
        raise ValueError("model step cannot exceed complete step for the same baseline.")
    target_step_seconds = batch_size / target_tokens_per_second
    required_saving = max(0.0, complete_wall_seconds_per_step - target_step_seconds)
    measured_gap = complete_wall_seconds_per_step - model_seconds_per_step
    return {
        "batch_size": batch_size,
        "target_tokens_per_second": target_tokens_per_second,
        "target_step_seconds": target_step_seconds,
        "baseline_complete_wall_seconds_per_step": complete_wall_seconds_per_step,
        "baseline_model_seconds_per_step": model_seconds_per_step,
        "baseline_complete_minus_model_gap_seconds_per_step": measured_gap,
        "required_saving_seconds_per_step": required_saving,
        "required_saving_fraction_of_measured_gap": (
            required_saving / measured_gap if measured_gap > 0 else math.inf
        ),
        "model_only_ceiling_tokens_per_second": batch_size / model_seconds_per_step,
        "model_only_ceiling_clears_target": (
            batch_size / model_seconds_per_step >= target_tokens_per_second
        ),
    }


def summarize_batched_processor_candidate(
        *,
        exactness_pass: bool,
        baseline_tokens_per_second: float,
        candidate_tokens_per_second: float,
        target_tokens_per_second: float = 500.0,
) -> dict[str, float | bool]:
    """Gate the identical-prompt B8 batched-processor microcandidate."""

    if baseline_tokens_per_second <= 0 or candidate_tokens_per_second <= 0:
        raise ValueError("throughput values must be positive.")
    if target_tokens_per_second <= 0:
        raise ValueError("target_tokens_per_second must be positive.")
    relative_gain = candidate_tokens_per_second / baseline_tokens_per_second - 1.0
    reaches_target = candidate_tokens_per_second >= target_tokens_per_second
    clears_keep_bar = relative_gain >= 0.05
    return {
        "exactness_pass": bool(exactness_pass),
        "baseline_complete_tokens_per_second": baseline_tokens_per_second,
        "candidate_complete_tokens_per_second": candidate_tokens_per_second,
        "target_tokens_per_second": target_tokens_per_second,
        "relative_complete_throughput_gain": relative_gain,
        "reaches_target_tokens_per_second": reaches_target,
        "clears_five_percent_gain_gate": clears_keep_bar,
        "promotion_gate_pass": bool(exactness_pass and reaches_target and clears_keep_bar),
    }


def compare_batch_physics_observations(
        baseline: BatchPhysicsObservation,
        candidate: BatchPhysicsObservation,
) -> dict[str, object]:
    """Compare exact observations and apply the campaign's five-percent scout bar."""

    if baseline.result_class is not candidate.result_class:
        raise ValueError(
            "batch-physics result classes differ: "
            f"{baseline.result_class.value} != {candidate.result_class.value}."
        )
    if baseline.workload_contract_hash != candidate.workload_contract_hash:
        raise ValueError("batch-physics workload contracts differ.")

    exactness_fields = {
        "seeds": baseline.seeds == candidate.seeds,
        "generated_tokens": baseline.generated_tokens == candidate.generated_tokens,
        "token_hashes": baseline.token_hashes == candidate.token_hashes,
        "final_rng_state_hashes": (
            baseline.final_rng_state_hashes == candidate.final_rng_state_hashes
        ),
        "stop_reasons": baseline.stop_reasons == candidate.stop_reasons,
        "intermediate_state_hashes": (
            baseline.intermediate_state_hashes == candidate.intermediate_state_hashes
        ),
        "cache_state_hashes": baseline.cache_state_hashes == candidate.cache_state_hashes,
    }
    failed_fields = [name for name, matched in exactness_fields.items() if not matched]
    if failed_fields:
        raise ValueError(
            "batch-physics observations are not exact-output comparable; mismatched fields: "
            + ", ".join(failed_fields)
        )
    baseline_tps = baseline.scheduler_wall_tokens_per_second
    candidate_tps = candidate.scheduler_wall_tokens_per_second
    relative_gain = candidate_tps / baseline_tps - 1.0
    result = {
        "baseline_execution_family": baseline.execution_family.value,
        "baseline_parallelism": baseline.parallelism,
        "candidate_execution_family": candidate.execution_family.value,
        "candidate_parallelism": candidate.parallelism,
        "result_classes": [baseline.result_class.value, candidate.result_class.value],
        "exactness_fields": exactness_fields,
        "baseline_scheduler_wall_tokens_per_second": baseline_tps,
        "candidate_scheduler_wall_tokens_per_second": candidate_tps,
        "relative_throughput_gain": relative_gain,
        "clears_five_percent_scout_bar": relative_gain >= 0.05,
        "decision": "continue_incremental_gates",
    }
    if relative_gain < 0.05:
        raise ValueError(
            "candidate fails the batch-physics keep gate: "
            f"relative throughput gain {relative_gain:.3%} is below 5%."
        )
    return result
