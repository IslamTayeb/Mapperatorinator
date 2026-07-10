"""Model-free compatibility analysis for accepted serial five-song profiles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REFERENCE_BUCKET = 128
MAXIMUM_ACCEPTED_PREFIX_BUCKET = 1024
MEASURED_COMPLETE_TPS = {
    1: 75.402,
    2: 149.341,
    5: 318.233,
    8: 439.636,
}
MEASURED_SHARED_B8_TPS = 497.32811465329866
MEASURED_OPTIMIZED_L1_TPS = 414.9046973558729
MEASURED_L2_MODEL_ONLY_TPS = 778.591
MEASURED_L2_COMPLETE_TPS = 397.903
MEASURED_B8_PRIVATE_PROCESSOR_STEP_MILLISECONDS = 17.760
MEASURED_B8_SHARED_PROCESSOR_STEP_MILLISECONDS = 14.696
ACCEPTED_FIVE_SONG_SERIAL_SCHEDULER_TPS = 210.230
ACCEPTED_FIVE_SONG_ACTIVE_WALL_TPS = 285.044
SYNTHETIC_B8_SERIAL_PREFILL_SECONDS = 0.34391318541020155
SYNTHETIC_B8_PACK_SECONDS = 0.04675073456019163


PROFILE_METADATA_CONTRACT_FIELDS = (
    "model_path",
    "precision",
    "attn_implementation",
    "inference_engine",
    "optimized_inference_mode",
    "use_server",
    "parallel",
    "inference_generation_compile",
    "inference_active_prefix_decode_loop",
    "inference_active_prefix_decode_bucket_size",
    "inference_active_prefix_decode_cuda_graph",
    "inference_active_prefix_decode_cuda_graph_warmup",
    "inference_active_prefix_decode_cuda_graph_min_decode_steps",
    "inference_stateful_monotonic_logits_processor",
    "inference_q1_bmm_cross_attention",
    "inference_decode_session_runtime",
    "inference_decode_session_cuda_graph",
    "inference_native_decode_kernels",
    "inference_native_q1_self_attention",
    "inference_native_q1_rope_cache_self_attention",
    "temperature",
    "timing_temperature",
    "mania_column_temperature",
    "taiko_hit_temperature",
    "timeshift_bias",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "cfg_scale",
    "lookback",
    "lookahead",
    "start_time",
    "end_time",
    "in_context",
    "output_type",
)

WINDOW_CONTRACT_FIELDS = (
    "profile_label",
    "mode",
    "context_type",
    "precision",
    "stateful_monotonic_logits_processor",
    "q1_bmm_cross_attention_enabled",
    "native_q1_self_attention_enabled",
    "native_q1_rope_cache_self_attention_enabled",
    "active_prefix_decode_loop_enabled",
    "active_prefix_decode_bucket_size",
    "active_prefix_decode_cuda_graph_enabled",
    "decode_session_runtime_enabled",
    "decode_session_cuda_graph_enabled",
    "trim_lookback",
    "trim_lookahead",
)


def stable_json_hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def bucketed_prefix_length(prefix_length: int, bucket_size: int) -> int:
    if prefix_length <= 0 or bucket_size <= 0:
        raise ValueError("prefix_length and bucket_size must be positive.")
    return ((prefix_length + bucket_size - 1) // bucket_size) * bucket_size


def _require_fields(values: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    missing = [field for field in fields if field not in values]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")


@dataclass(frozen=True)
class AcceptedWindow:
    song_id: str
    sequence_index: int
    prompt_tokens: int
    generated_token_ids: tuple[int, ...]
    static_contract: Mapping[str, Any]
    bucket_size: int

    def __post_init__(self) -> None:
        if not self.song_id:
            raise ValueError("song_id must be non-empty.")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative.")
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive.")
        if not self.generated_token_ids:
            raise ValueError("generated_token_ids must be non-empty.")
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be positive.")

    @property
    def main_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def decode_tokens(self) -> int:
        return max(0, self.main_tokens - 1)

    def decode_bucket(self, generated_position: int) -> int:
        """Bucket for the model call that produces output token at this position."""

        if generated_position <= 0 or generated_position >= self.main_tokens:
            raise ValueError("generated_position must name a post-prefill output token.")
        bucket = bucketed_prefix_length(
            self.prompt_tokens + generated_position,
            self.bucket_size,
        )
        if bucket > MAXIMUM_ACCEPTED_PREFIX_BUCKET:
            raise ValueError(
                f"accepted prefix bucket {bucket} exceeds the runtime cache limit "
                f"{MAXIMUM_ACCEPTED_PREFIX_BUCKET}."
            )
        return bucket

    def compatibility_payload(self, generated_position: int) -> dict[str, Any]:
        return {
            "static_contract": dict(self.static_contract),
            "active_prefix_bucket": self.decode_bucket(generated_position),
        }

    def compatibility_key(self, generated_position: int) -> str:
        return stable_json_hash(self.compatibility_payload(generated_position))


@dataclass(frozen=True)
class AcceptedSong:
    song_id: str
    profile_artifact_sha256: str
    metadata_contract: Mapping[str, Any]
    windows: tuple[AcceptedWindow, ...]

    def __post_init__(self) -> None:
        if not self.song_id or not self.profile_artifact_sha256:
            raise ValueError("song_id and profile_artifact_sha256 must be non-empty.")
        if not self.windows:
            raise ValueError("song must contain main-generation windows.")
        expected = list(range(len(self.windows)))
        actual = [window.sequence_index for window in self.windows]
        if actual != expected:
            raise ValueError(
                f"{self.song_id} main sequence indexes must be contiguous {expected}; got {actual}."
            )
        if any(window.song_id != self.song_id for window in self.windows):
            raise ValueError("window song IDs must match their owner.")


def accepted_song_from_profile(
        profile: Mapping[str, Any],
        *,
        profile_artifact_sha256: str,
) -> AcceptedSong:
    if int(profile.get("schema_version", -1)) < 1:
        raise ValueError("profile schema_version is missing or invalid.")
    metadata = profile.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("profile metadata must be a mapping.")
    required_metadata = (
        "song_id",
        "repeat_index",
        "run_kind",
        "profile_record_token_ids",
        "sequence_count",
        *PROFILE_METADATA_CONTRACT_FIELDS,
    )
    _require_fields(metadata, required_metadata, label="profile metadata")
    if int(metadata["repeat_index"]) != 1:
        raise ValueError("compatibility analysis requires accepted warmed repeat01 profiles.")
    if metadata["run_kind"] != "serial_multi_song":
        raise ValueError("profile must be a serial_multi_song denominator artifact.")
    if not bool(metadata["profile_record_token_ids"]):
        raise ValueError("profile must record exact generated token IDs.")
    song_id = str(metadata["song_id"])
    metadata_contract = {
        field: metadata[field]
        for field in PROFILE_METADATA_CONTRACT_FIELDS
    }
    generation = profile.get("generation")
    if not isinstance(generation, list):
        raise ValueError("profile generation must be a list.")
    windows: list[AcceptedWindow] = []
    for record in generation:
        if not isinstance(record, Mapping) or record.get("profile_label") != "main_generation":
            continue
        required_record = (
            "sequence_index",
            "prompt_tokens",
            "generated_tokens",
            "generated_token_ids",
            *WINDOW_CONTRACT_FIELDS,
        )
        _require_fields(record, required_record, label=f"{song_id} main-generation record")
        token_ids = record["generated_token_ids"]
        if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool)
                for token in token_ids
        ):
            raise TypeError(f"{song_id} generated_token_ids must be a list of integers.")
        if len(token_ids) != int(record["generated_tokens"]):
            raise ValueError(
                f"{song_id} sequence {record['sequence_index']} token count/hash source mismatch."
            )
        bucket_size = int(record["active_prefix_decode_bucket_size"])
        if bucket_size != int(metadata["inference_active_prefix_decode_bucket_size"]):
            raise ValueError(f"{song_id} record/metadata bucket sizes disagree.")
        static_contract = {
            "metadata": metadata_contract,
            "window": {field: record[field] for field in WINDOW_CONTRACT_FIELDS},
        }
        windows.append(AcceptedWindow(
            song_id=song_id,
            sequence_index=int(record["sequence_index"]),
            prompt_tokens=int(record["prompt_tokens"]),
            generated_token_ids=tuple(token_ids),
            static_contract=static_contract,
            bucket_size=bucket_size,
        ))
    windows.sort(key=lambda window: window.sequence_index)
    if len(windows) != int(metadata["sequence_count"]):
        raise ValueError(
            f"{song_id} expected {metadata['sequence_count']} main windows; got {len(windows)}."
        )
    return AcceptedSong(
        song_id=song_id,
        profile_artifact_sha256=profile_artifact_sha256,
        metadata_contract=metadata_contract,
        windows=tuple(windows),
    )


def request_manifest(songs: Sequence[AcceptedSong], *, order_sensitive: bool) -> dict[str, Any]:
    rows = [
        {
            "song_id": song.song_id,
            "profile_artifact_sha256": song.profile_artifact_sha256,
            "window_token_hashes": [
                stable_json_hash(window.generated_token_ids)
                for window in song.windows
            ],
        }
        for song in songs
    ]
    if not order_sensitive:
        rows.sort(key=lambda row: row["song_id"])
    return {
        "song_count": len(rows),
        "order_sensitive": order_sensitive,
        "songs": rows,
    }


def reciprocal_order(songs: Sequence[AcceptedSong]) -> tuple[AcceptedSong, ...]:
    return tuple(reversed(tuple(songs)))


def _advance_state(
        song: AcceptedSong,
        window_index: int,
        generated_position: int,
) -> tuple[int, int] | None:
    generated_position += 1
    while window_index < len(song.windows):
        window = song.windows[window_index]
        if generated_position < window.main_tokens:
            return window_index, generated_position
        window_index += 1
        generated_position = 1
        while window_index < len(song.windows) and song.windows[window_index].decode_tokens == 0:
            window_index += 1
    return None


def simulate_iteration_compatibility(
        songs: Sequence[AcceptedSong],
        *,
        policy: str = "largest_compatible_first",
) -> dict[str, Any]:
    if len(songs) < 1:
        raise ValueError("at least one song is required.")
    song_ids = [song.song_id for song in songs]
    if len(song_ids) != len(set(song_ids)):
        raise ValueError("one active dependency chain per distinct song is required.")
    metadata_hashes = {stable_json_hash(song.metadata_contract) for song in songs}
    if len(metadata_hashes) != 1:
        raise ValueError("accepted profiles do not share one runtime/sampling/kernel contract.")
    if policy not in {"largest_compatible_first", "synchronous_round_robin"}:
        raise ValueError("unknown compatibility scheduling policy.")

    states: dict[str, tuple[int, int]] = {}
    for song in songs:
        window_index = 0
        while window_index < len(song.windows) and song.windows[window_index].decode_tokens == 0:
            window_index += 1
        if window_index < len(song.windows):
            states[song.song_id] = (window_index, 1)
    songs_by_id = {song.song_id: song for song in songs}
    ready_request_histogram: Counter[int] = Counter()
    group_histogram: Counter[int] = Counter()
    tokens_by_group_size: Counter[int] = Counter()
    bucket_token_histogram: Counter[int] = Counter()
    group_events: list[dict[str, Any]] = []
    per_song_schedule: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    regroup_count = 0
    scheduler_step = 0
    while states:
        ready_request_histogram[len(states)] += 1
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for song_id in song_ids:
            if song_id not in states:
                continue
            song = songs_by_id[song_id]
            window_index, generated_position = states[song_id]
            window = song.windows[window_index]
            bucket = window.decode_bucket(generated_position)
            key = window.compatibility_key(generated_position)
            item = {
                "song_id": song_id,
                "sequence_index": window.sequence_index,
                "generated_position": generated_position,
                "bucket": bucket,
            }
            groups[key].append(item)
        ordered_group_keys = sorted(
            groups,
            key=lambda key: (
                -len(groups[key]),
                int(groups[key][0]["bucket"]),
                key,
            ),
        )
        selected_keys = (
            ordered_group_keys[:1]
            if policy == "largest_compatible_first"
            else ordered_group_keys
        )
        selected_song_ids: set[str] = set()
        for key in selected_keys:
            items = sorted(groups[key], key=lambda item: item["song_id"])
            size = len(items)
            group_histogram[size] += 1
            tokens_by_group_size[size] += size
            for item in items:
                bucket_token_histogram[item["bucket"]] += 1
            group_events.append({
                "scheduler_step": scheduler_step,
                "regroup": regroup_count,
                "compatibility_key": key,
                "bucket": items[0]["bucket"],
                "size": size,
                "song_ids": [item["song_id"] for item in items],
                "items": items,
            })
            selected_song_ids.update(item["song_id"] for item in items)
            for item in items:
                per_song_schedule[item["song_id"]].append((
                    item["sequence_index"],
                    item["generated_position"],
                    item["bucket"],
                ))
            scheduler_step += 1
        next_states = dict(states)
        for song_id in selected_song_ids:
            window_index, generated_position = states[song_id]
            advanced = _advance_state(
                songs_by_id[song_id],
                window_index,
                generated_position,
            )
            if advanced is not None:
                next_states[song_id] = advanced
            else:
                del next_states[song_id]
        states = next_states
        regroup_count += 1

    total_main_tokens = sum(window.main_tokens for song in songs for window in song.windows)
    total_decode_tokens = sum(window.decode_tokens for song in songs for window in song.windows)
    simulated_decode_tokens = sum(tokens_by_group_size.values())
    if total_decode_tokens != simulated_decode_tokens:
        raise AssertionError(
            f"decode token accounting mismatch: {total_decode_tokens} != {simulated_decode_tokens}."
        )
    max_group_size = max(group_histogram, default=0)
    schedule_hashes = {
        song_id: stable_json_hash(schedule)
        for song_id, schedule in sorted(per_song_schedule.items())
    }
    return {
        "launch_order": list(song_ids),
        "scheduler_policy": policy,
        "policy_contract": (
            "Regroup after every selected compatibility group. Largest-compatible-first "
            "chooses greatest group size, then lowest current prefix bucket, then stable "
            "compatibility-key hash. Synchronous-round-robin advances every current group "
            "once per regroup in that same deterministic order."
        ),
        "maximum_accepted_prefix_bucket": MAXIMUM_ACCEPTED_PREFIX_BUCKET,
        "scheduler_steps": scheduler_step,
        "regroup_count": regroup_count,
        "song_count": len(songs),
        "window_count": sum(len(song.windows) for song in songs),
        "total_main_tokens": total_main_tokens,
        "prefill_tokens": total_main_tokens - total_decode_tokens,
        "total_decode_tokens": total_decode_tokens,
        "ready_request_count_histogram": dict(sorted(ready_request_histogram.items())),
        "active_batch_size_histogram": dict(sorted(group_histogram.items())),
        "compatible_group_size_histogram": dict(sorted(group_histogram.items())),
        "decode_tokens_by_group_size": dict(sorted(tokens_by_group_size.items())),
        "bucket_decode_token_histogram": dict(sorted(bucket_token_histogram.items())),
        "max_compatible_group_size": max_group_size,
        "has_compatible_B8_group": max_group_size >= 8,
        "weighted_singleton_fraction": (
            tokens_by_group_size.get(1, 0) / total_decode_tokens
            if total_decode_tokens
            else 0.0
        ),
        "per_song_schedule_hashes": schedule_hashes,
        "group_events": group_events,
    }


def permutation_invariance_report(
        songs: Sequence[AcceptedSong],
        *,
        policy: str = "largest_compatible_first",
) -> dict[str, Any]:
    forward = simulate_iteration_compatibility(songs, policy=policy)
    reverse = simulate_iteration_compatibility(reciprocal_order(songs), policy=policy)
    invariant_fields = (
        "scheduler_policy",
        "policy_contract",
        "maximum_accepted_prefix_bucket",
        "scheduler_steps",
        "regroup_count",
        "song_count",
        "window_count",
        "total_main_tokens",
        "prefill_tokens",
        "total_decode_tokens",
        "active_batch_size_histogram",
        "ready_request_count_histogram",
        "compatible_group_size_histogram",
        "decode_tokens_by_group_size",
        "bucket_decode_token_histogram",
        "max_compatible_group_size",
        "has_compatible_B8_group",
        "weighted_singleton_fraction",
        "per_song_schedule_hashes",
    )
    field_matches = {field: forward[field] == reverse[field] for field in invariant_fields}
    return {
        "pass": all(field_matches.values()),
        "field_matches": field_matches,
        "forward_order": forward["launch_order"],
        "reverse_order": reverse["launch_order"],
        "forward_manifest_hash": stable_json_hash(
            request_manifest(songs, order_sensitive=True)
        ),
        "reverse_manifest_hash": stable_json_hash(
            request_manifest(reciprocal_order(songs), order_sensitive=True)
        ),
        "request_set_hash": stable_json_hash(
            request_manifest(songs, order_sensitive=False)
        ),
        "forward": forward,
        "reverse": reverse,
    }


def _measured_execution_options() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "optimized_lane_L1_complete",
            "batch_size": 1,
            "tokens_per_second": MEASURED_OPTIMIZED_L1_TPS,
            "job": "49547823",
            "commit": "08a59f5",
            "provenance": "reviewed normalized exact optimized L1 lane complete step",
        },
        {
            "name": "eager_merged_B1_complete",
            "batch_size": 1,
            "tokens_per_second": MEASURED_COMPLETE_TPS[1],
            "job": "49546220",
            "provenance": "exact eager merged-family B1 complete step",
        },
        {
            "name": "eager_merged_B2_complete",
            "batch_size": 2,
            "tokens_per_second": MEASURED_COMPLETE_TPS[2],
            "job": "49546893",
            "provenance": "exact eager merged B2 complete step",
        },
        {
            "name": "eager_merged_B5_complete",
            "batch_size": 5,
            "tokens_per_second": MEASURED_COMPLETE_TPS[5],
            "job": "49546977",
            "provenance": "exact eager merged B5 complete step",
        },
        {
            "name": "eager_merged_B8_complete",
            "batch_size": 8,
            "tokens_per_second": MEASURED_COMPLETE_TPS[8],
            "job": "49547025",
            "provenance": "exact eager merged B8 complete step",
        },
    )


def _best_measured_partition(size: int) -> tuple[float, tuple[str, ...]]:
    if size <= 0:
        raise ValueError("group size must be positive.")
    options = tuple(
        option
        for option in _measured_execution_options()
        if int(option["batch_size"]) <= size
    )
    best: list[tuple[float, tuple[str, ...]] | None] = [None] * (size + 1)
    best[0] = (0.0, ())
    for total in range(1, size + 1):
        candidates = []
        for option in options:
            batch = int(option["batch_size"])
            if batch > total or best[total - batch] is None:
                continue
            previous_seconds, previous_partition = best[total - batch]
            candidates.append((
                previous_seconds + batch / float(option["tokens_per_second"]),
                tuple(sorted((*previous_partition, str(option["name"])))),
            ))
        if candidates:
            best[total] = min(candidates, key=lambda candidate: candidate[0])
    if best[size] is None:
        raise AssertionError(f"no measured partition covers group size {size}.")
    return best[size]


def hybrid_l2_processor_ceiling(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Measured ceiling for a bounded L2 replay-plus-batched-processor scout.

    The proposed coordinator may remove control time, but it cannot make the
    already-measured concurrent model interval disappear.  Capping the ceiling
    at that interval avoids the impossible result obtained by subtracting the
    full B8 processor saving from the much smaller L2 complete step.
    """

    group_events = schedule.get("group_events")
    if not isinstance(group_events, list) or not group_events:
        raise ValueError("schedule must contain non-empty group_events.")
    total_decode_tokens = int(schedule["total_decode_tokens"])
    total_main_tokens = int(schedule["total_main_tokens"])
    window_count = int(schedule["window_count"])
    b2_events = sum(1 for event in group_events if int(event["size"]) == 2)
    b2_decode_tokens = 2 * b2_events
    b2_decode_token_fraction = b2_decode_tokens / total_decode_tokens

    measured_model_milliseconds = 2000.0 / MEASURED_L2_MODEL_ONLY_TPS
    measured_complete_milliseconds = 2000.0 / MEASURED_L2_COMPLETE_TPS
    measured_control_gap_milliseconds = (
        measured_complete_milliseconds - measured_model_milliseconds
    )
    target_500_milliseconds = 2000.0 / 500.0
    required_saving_milliseconds = max(
        0.0,
        measured_complete_milliseconds - target_500_milliseconds,
    )
    b8_observed_saving_milliseconds = (
        MEASURED_B8_PRIVATE_PROCESSOR_STEP_MILLISECONDS
        - MEASURED_B8_SHARED_PROCESSOR_STEP_MILLISECONDS
    )
    optimistic_hybrid_milliseconds = max(
        measured_model_milliseconds,
        measured_complete_milliseconds - b8_observed_saving_milliseconds,
    )
    optimistic_hybrid_tps = 2000.0 / optimistic_hybrid_milliseconds

    # Apply the optimistic L2 point to every compatible pair, including one
    # pair inside B3.  Remaining rows use the fastest measured exact L1 point.
    modeled_decode_seconds = 0.0
    hybrid_pair_events = 0
    for event in group_events:
        remaining = int(event["size"])
        pairs = remaining // 2
        hybrid_pair_events += pairs
        modeled_decode_seconds += pairs * optimistic_hybrid_milliseconds / 1000.0
        remaining -= pairs * 2
        modeled_decode_seconds += remaining / MEASURED_OPTIMIZED_L1_TPS
    setup_per_request = (
        SYNTHETIC_B8_SERIAL_PREFILL_SECONDS + SYNTHETIC_B8_PACK_SECONDS
    ) / 8.0
    full_linear_setup_seconds = setup_per_request * window_count
    schedule_decode_only_tps = total_main_tokens / modeled_decode_seconds
    schedule_full_setup_tps = total_main_tokens / (
        modeled_decode_seconds + full_linear_setup_seconds
    )

    materially_common = b2_decode_token_fraction >= 0.05
    return {
        "candidate": "concurrent_private_B1_replays_then_coordinator_batched_processor",
        "scope": "bounded_verifier_scout_only",
        "source_job": "49548733",
        "source_commit": "aa662c90",
        "measured_L2_model_only_tokens_per_second": MEASURED_L2_MODEL_ONLY_TPS,
        "measured_L2_complete_tokens_per_second": MEASURED_L2_COMPLETE_TPS,
        "measured_L2_model_only_step_milliseconds": measured_model_milliseconds,
        "measured_L2_complete_step_milliseconds": measured_complete_milliseconds,
        "measured_control_gap_milliseconds": measured_control_gap_milliseconds,
        "target_500_step_milliseconds": target_500_milliseconds,
        "required_saving_for_500_milliseconds": required_saving_milliseconds,
        "required_fraction_of_control_gap": (
            required_saving_milliseconds / measured_control_gap_milliseconds
        ),
        "B8_shared_processor_observed_saving_milliseconds": (
            b8_observed_saving_milliseconds
        ),
        "cross_shape_extrapolation_warning": (
            "The B8 shared-processor saving is evidence that the required L2 control saving "
            "is plausible, not an additive L2 timing measurement. The optimistic L2 ceiling "
            "is therefore capped at the measured L2 model-only interval."
        ),
        "optimistic_L2_step_milliseconds": optimistic_hybrid_milliseconds,
        "optimistic_L2_tokens_per_second": optimistic_hybrid_tps,
        "B2_event_count": b2_events,
        "B2_decode_tokens": b2_decode_tokens,
        "B2_decode_token_fraction": b2_decode_token_fraction,
        "B2_materially_common_at_five_percent": materially_common,
        "optimistic_schedule_hybrid_pair_event_count": hybrid_pair_events,
        "optimistic_schedule_decode_seconds": modeled_decode_seconds,
        "optimistic_schedule_decode_only_main_tokens_per_second": schedule_decode_only_tps,
        "optimistic_schedule_full_setup_main_tokens_per_second": schedule_full_setup_tps,
        "optimistic_schedule_clears_500_decode_only": schedule_decode_only_tps >= 500.0,
        "optimistic_schedule_clears_500_with_full_setup": schedule_full_setup_tps >= 500.0,
        "next_action": (
            "one_bounded_hybrid_L2_processor_verifier"
            if materially_common
            else "do_not_run_hybrid_L2_for_this_queue"
        ),
    }


def conservative_scheduler_ceiling(schedule: Mapping[str, Any]) -> dict[str, Any]:
    group_events = schedule.get("group_events")
    if not isinstance(group_events, list) or not group_events:
        raise ValueError("schedule must contain non-empty group_events.")
    modeled_decode_seconds = 0.0
    shape_applicable_tokens = 0
    partition_histogram: Counter[str] = Counter()
    for event in group_events:
        size = int(event["size"])
        seconds, partition = _best_measured_partition(size)
        modeled_decode_seconds += seconds
        partition_histogram["+".join(partition)] += 1
        if int(event["bucket"]) == REFERENCE_BUCKET:
            shape_applicable_tokens += size
    total_main_tokens = int(schedule["total_main_tokens"])
    total_decode_tokens = int(schedule["total_decode_tokens"])
    window_count = int(schedule["window_count"])
    setup_per_request = (
        SYNTHETIC_B8_SERIAL_PREFILL_SECONDS + SYNTHETIC_B8_PACK_SECONDS
    ) / 8.0
    full_linear_setup_seconds = setup_per_request * window_count
    initial_b8_setup_seconds = (
        SYNTHETIC_B8_SERIAL_PREFILL_SECONDS + SYNTHETIC_B8_PACK_SECONDS
    )
    schedule_decode_only_main_tps = total_main_tokens / modeled_decode_seconds
    schedule_full_setup_main_tps = total_main_tokens / (
        modeled_decode_seconds + full_linear_setup_seconds
    )
    incompatible_b8_decode_seconds = total_decode_tokens / MEASURED_SHARED_B8_TPS
    incompatible_b8_decode_only_main_tps = total_main_tokens / incompatible_b8_decode_seconds
    incompatible_b8_initial_setup_main_tps = total_main_tokens / (
        incompatible_b8_decode_seconds + initial_b8_setup_seconds
    )
    incompatible_b8_full_setup_main_tps = total_main_tokens / (
        incompatible_b8_decode_seconds + full_linear_setup_seconds
    )
    relative_to_serial = (
        schedule_full_setup_main_tps / ACCEPTED_FIVE_SONG_SERIAL_SCHEDULER_TPS - 1.0
    )
    relative_to_active = (
        schedule_full_setup_main_tps / ACCEPTED_FIVE_SONG_ACTIVE_WALL_TPS - 1.0
    )
    return {
        "reference_measurement_bucket": REFERENCE_BUCKET,
        "measured_complete_tokens_per_second": dict(MEASURED_COMPLETE_TPS),
        "measured_execution_options": list(_measured_execution_options()),
        "measured_shared_B8_decode_tokens_per_second": MEASURED_SHARED_B8_TPS,
        "measured_partition_histogram": dict(sorted(partition_histogram.items())),
        "shape_applicable_decode_tokens": shape_applicable_tokens,
        "shape_applicable_decode_token_fraction": shape_applicable_tokens / total_decode_tokens,
        "synthetic_bucket_extrapolation": shape_applicable_tokens != total_decode_tokens,
        "synthetic_bucket_warning": (
            "Measured merged points came from bucket 128. Reusing them for longer production "
            "buckets ignores prefix-dependent slowdown and is an optimistic extrapolation."
        ),
        "schedule_partition_decode_seconds": modeled_decode_seconds,
        "schedule_partition_decode_only_main_tokens_per_second": schedule_decode_only_main_tps,
        "synthetic_B8_setup_datum": {
            "serial_B1_prefill_seconds_for_8": SYNTHETIC_B8_SERIAL_PREFILL_SECONDS,
            "pack_seconds_for_8": SYNTHETIC_B8_PACK_SECONDS,
            "per_request_linearized_seconds": setup_per_request,
            "initial_scheduler_setup_seconds": initial_b8_setup_seconds,
            "full_queue_linearized_setup_seconds": full_linear_setup_seconds,
        },
        "schedule_partition_full_setup_main_tokens_per_second": schedule_full_setup_main_tps,
        "accepted_denominators": {
            "five_song_serial_scheduler_wall_main_tokens_per_second": (
                ACCEPTED_FIVE_SONG_SERIAL_SCHEDULER_TPS
            ),
            "five_song_active_wall_main_tokens_per_second": ACCEPTED_FIVE_SONG_ACTIVE_WALL_TPS,
        },
        "relative_gain_vs_serial_scheduler_wall": relative_to_serial,
        "clears_five_percent_vs_serial_scheduler_wall": relative_to_serial >= 0.05,
        "relative_gain_vs_active_wall": relative_to_active,
        "clears_five_percent_vs_active_wall": relative_to_active >= 0.05,
        "physically_incompatible_all_B8_fantasy": {
            "reason": (
                "The five-song dependency schedule never has eight active songs and has no B8 "
                "compatibility group; this is only an absolute optimistic bound."
            ),
            "decode_only_main_tokens_per_second": incompatible_b8_decode_only_main_tps,
            "initial_setup_inclusive_main_tokens_per_second": incompatible_b8_initial_setup_main_tps,
            "full_linearized_setup_main_tokens_per_second": incompatible_b8_full_setup_main_tps,
        },
        "clears_500_with_schedule_and_full_setup": schedule_full_setup_main_tps >= 500.0,
        "incompatible_B8_clears_500_with_initial_setup": (
            incompatible_b8_initial_setup_main_tps >= 500.0
        ),
    }


def analyze_five_song_profiles(songs: Sequence[AcceptedSong]) -> dict[str, Any]:
    if len(songs) != 5 or len({song.song_id for song in songs}) != 5:
        raise ValueError("the accepted compatibility gate requires exactly five distinct songs.")
    permutation = permutation_invariance_report(
        songs,
        policy="largest_compatible_first",
    )
    if not permutation["pass"]:
        raise AssertionError("reciprocal request order changed the compatibility schedule.")
    schedule = permutation["forward"]
    synchronous = permutation_invariance_report(
        songs,
        policy="synchronous_round_robin",
    )
    if not synchronous["pass"]:
        raise AssertionError("reciprocal request order changed the synchronous sensitivity schedule.")
    ceiling = conservative_scheduler_ceiling(schedule)
    hybrid_ceiling = hybrid_l2_processor_ceiling(schedule)
    can_advance = bool(
        schedule["has_compatible_B8_group"]
        and ceiling["clears_500_with_schedule_and_full_setup"]
        and ceiling["clears_five_percent_vs_serial_scheduler_wall"]
    )
    return {
        "schema_version": 1,
        "scope": "accepted_five_song_repeat01_model_free_compatibility",
        "request_manifest": request_manifest(songs, order_sensitive=True),
        "request_set_hash": permutation["request_set_hash"],
        "permutation_invariance": {
            key: value
            for key, value in permutation.items()
            if key not in {"forward", "reverse"}
        },
        "schedule": schedule,
        "policy_sensitivity": {
            "synchronous_round_robin": synchronous["forward"],
            "policy_independent_B8_impossibility": (
                "Only five dependency chains can be active; no scheduling policy can form B8."
            ),
        },
        "profile_shape_evidence_limit": (
            "Accepted profiles prove prompt token counts, bucket schedules, sampling/runtime "
            "contracts, and token IDs, but do not record encoder/frame/condition tensor shapes. "
            "Shape compatibility beyond the fixed smoke configuration remains unproven."
        ),
        "ceiling": ceiling,
        "next_candidate_ceiling": hybrid_ceiling,
        "campaign_next_action": hybrid_ceiling["next_action"],
        "can_advance_to_mixed_decode": can_advance,
        "decision": (
            "continue_to_bounded_mixed_decode"
            if can_advance
            else "stop_before_mixed_decode"
        ),
        "pass": True,
    }
