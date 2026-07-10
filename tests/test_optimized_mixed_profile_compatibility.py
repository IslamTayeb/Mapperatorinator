from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "osuT5/osuT5/inference/optimized/batch/mixed_profile_compatibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "test_mixed_profile_compatibility_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROFILE_METADATA_CONTRACT_FIELDS = MODULE.PROFILE_METADATA_CONTRACT_FIELDS
WINDOW_CONTRACT_FIELDS = MODULE.WINDOW_CONTRACT_FIELDS
AcceptedSong = MODULE.AcceptedSong
AcceptedWindow = MODULE.AcceptedWindow
accepted_song_from_profile = MODULE.accepted_song_from_profile
analyze_five_song_profiles = MODULE.analyze_five_song_profiles
bucketed_prefix_length = MODULE.bucketed_prefix_length
permutation_invariance_report = MODULE.permutation_invariance_report
request_manifest = MODULE.request_manifest
simulate_iteration_compatibility = MODULE.simulate_iteration_compatibility
stable_json_hash = MODULE.stable_json_hash
_best_measured_partition = MODULE._best_measured_partition


def _window(song_id: str, sequence_index: int, prompt: int, token_count: int) -> AcceptedWindow:
    return AcceptedWindow(
        song_id=song_id,
        sequence_index=sequence_index,
        prompt_tokens=prompt,
        generated_token_ids=tuple(range(token_count)),
        static_contract={"context": "map", "trim_lookahead": sequence_index == 0},
        bucket_size=64,
    )


def _song(song_id: str, *, prompt: int = 63, token_count: int = 3) -> AcceptedSong:
    return AcceptedSong(
        song_id=song_id,
        profile_artifact_sha256=f"sha-{song_id}",
        metadata_contract={"runtime": "same"},
        windows=(_window(song_id, 0, prompt, token_count),),
    )


def _profile(song_id: str, *, include_token_ids: bool = True):
    metadata = {field: f"value-{field}" for field in PROFILE_METADATA_CONTRACT_FIELDS}
    metadata.update({
        "song_id": song_id,
        "repeat_index": 1,
        "run_kind": "serial_multi_song",
        "profile_record_token_ids": True,
        "sequence_count": 1,
        "inference_active_prefix_decode_bucket_size": 64,
    })
    record = {field: f"value-{field}" for field in WINDOW_CONTRACT_FIELDS}
    record.update({
        "profile_label": "main_generation",
        "sequence_index": 0,
        "prompt_tokens": 63,
        "generated_tokens": 3,
        "active_prefix_decode_bucket_size": 64,
    })
    if include_token_ids:
        record["generated_token_ids"] = [10, 11, 12]
    return {
        "schema_version": 4,
        "metadata": metadata,
        "generation": [record],
    }


def test_bucket_schedule_starts_after_prefill_token():
    window = _window("song", 0, prompt=63, token_count=3)

    assert bucketed_prefix_length(64, 64) == 64
    assert window.decode_bucket(1) == 64
    assert window.decode_bucket(2) == 128
    with pytest.raises(ValueError, match="post-prefill"):
        window.decode_bucket(0)


def test_profile_parser_requires_exact_generated_token_ids():
    parsed = accepted_song_from_profile(
        _profile("song"),
        profile_artifact_sha256="artifact",
    )

    assert parsed.windows[0].generated_token_ids == (10, 11, 12)
    with pytest.raises(ValueError, match="missing required fields"):
        accepted_song_from_profile(
            _profile("song", include_token_ids=False),
            profile_artifact_sha256="artifact",
        )


def test_iteration_groups_one_active_window_per_song_and_counts_prefill_tokens():
    songs = tuple(_song(f"song-{index}", prompt=63 + index) for index in range(5))

    schedule = simulate_iteration_compatibility(songs)

    assert schedule["song_count"] == 5
    assert schedule["total_main_tokens"] == 15
    assert schedule["prefill_tokens"] == 5
    assert schedule["total_decode_tokens"] == 10
    assert schedule["scheduler_policy"] == "largest_compatible_first"
    assert "stable compatibility-key hash" in schedule["policy_contract"]
    assert sum(
        size * count
        for size, count in schedule["compatible_group_size_histogram"].items()
    ) == 10
    with pytest.raises(ValueError, match="distinct song"):
        simulate_iteration_compatibility((songs[0], songs[0]))


def test_reciprocal_order_preserves_schedule_and_request_set_hash():
    songs = tuple(_song(f"song-{index}") for index in range(5))

    report = permutation_invariance_report(songs)

    assert report["pass"] is True
    assert report["forward_manifest_hash"] != report["reverse_manifest_hash"]
    forward_set = stable_json_hash(request_manifest(songs, order_sensitive=False))
    reverse_set = stable_json_hash(request_manifest(tuple(reversed(songs)), order_sensitive=False))
    assert report["request_set_hash"] == forward_set == reverse_set


def test_five_song_analysis_stops_without_b8_or_setup_inclusive_headroom():
    songs = tuple(
        _song(f"song-{index}", token_count=900)
        for index in range(5)
    )

    report = analyze_five_song_profiles(songs)

    assert report["schedule"]["max_compatible_group_size"] == 5
    assert "group_events" not in report["schedule"]
    assert report["schedule"]["group_event_count"] == report["schedule"]["scheduler_steps"]
    assert len(report["schedule"]["group_events_sha256"]) == 64
    assert report["schedule"]["has_compatible_B8_group"] is False
    assert report["ceiling"]["clears_five_percent_vs_serial_scheduler_wall"] is True
    assert report["ceiling"]["clears_500_with_schedule_and_full_setup"] is False
    assert report["ceiling"]["incompatible_B8_clears_500_with_initial_setup"] is False
    assert report["can_advance_to_mixed_decode"] is False
    assert report["decision"] == "stop_before_mixed_decode"


def test_partition_uses_exact_optimized_l1_instead_of_old_eager_b1():
    seconds, partition = _best_measured_partition(3)

    assert seconds == pytest.approx(3 / MODULE.MEASURED_OPTIMIZED_L1_TPS)
    assert partition == ("optimized_lane_L1_complete",) * 3


def test_synchronous_round_robin_is_labeled_as_policy_sensitivity():
    songs = tuple(_song(f"song-{index}") for index in range(5))

    report = analyze_five_song_profiles(songs)
    sensitivity = report["policy_sensitivity"]["synchronous_round_robin"]

    assert sensitivity["scheduler_policy"] == "synchronous_round_robin"
    assert "advances every current group once" in sensitivity["policy_contract"]
    assert report["policy_sensitivity"]["policy_independent_B8_impossibility"]


def test_material_b2_schedule_authorizes_only_bounded_hybrid_ceiling():
    songs = (
        _song("pair-a", prompt=63, token_count=20),
        _song("pair-b", prompt=63, token_count=20),
        _song("other-a", prompt=191, token_count=3),
        _song("other-b", prompt=319, token_count=3),
        _song("other-c", prompt=447, token_count=3),
    )

    report = analyze_five_song_profiles(songs)
    ceiling = report["next_candidate_ceiling"]

    assert ceiling["B2_materially_common_at_five_percent"] is True
    assert ceiling["required_fraction_of_control_gap"] == pytest.approx(0.4176218075)
    assert ceiling["optimistic_L2_tokens_per_second"] == pytest.approx(778.591)
    assert report["campaign_next_action"] == "one_bounded_hybrid_L2_processor_verifier"
    assert report["can_advance_to_mixed_decode"] is False
