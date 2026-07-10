from __future__ import annotations

import json

import pytest

from osuT5.osuT5.inference.optimized.speculative import (
    STRUCTURAL_LIMITATIONS,
    aggregate_ngram_acceptance_metrics,
    find_generated_prefix_ngram_proposal,
    load_profile_token_windows,
    run_generated_prefix_ngram_profile_scout,
    simulate_generated_prefix_ngram_acceptance,
)
from utils.scout_ngram_acceptance import main as cli_main


def _profile(*windows: list[int], label: str = "main_generation") -> dict:
    return {
        "generation": [
            {
                "profile_label": label,
                "context_type": "MAP",
                "mode": "sequential",
                "sequence_index": index,
                "generated_tokens": len(tokens),
                "generated_token_ids": tokens,
            }
            for index, tokens in enumerate(windows)
        ]
    }


def test_ngram_lookup_prefers_longest_suffix_and_most_recent_occurrence():
    proposal = find_generated_prefix_ngram_proposal(
        (1, 2, 9, 1, 2, 8, 1, 2),
        max_proposal_tokens=3,
        max_ngram_length=4,
    )

    assert proposal is not None
    assert proposal.matched_ngram_length == 2
    assert proposal.source_start == 3
    assert proposal.token_ids == (8, 1, 2)


def test_ngram_lookup_is_empty_until_an_earlier_match_has_committed_following_tokens():
    assert find_generated_prefix_ngram_proposal(
        (4, 5),
        max_proposal_tokens=4,
        max_ngram_length=4,
    ) is None


@pytest.mark.parametrize("speculation_k", [2, 4, 8])
def test_structural_replay_accounts_empty_mismatch_and_full_accept_calls(speculation_k):
    result = simulate_generated_prefix_ngram_acceptance(
        (1, 2, 3, 1, 2, 3, 4),
        speculation_k=speculation_k,
        max_ngram_length=3,
    )

    assert result.serial_target_calls == 7
    assert result.speculative_target_calls == (
        result.empty_proposal_calls + result.mismatch_calls + result.full_accept_calls
    )
    assert result.target_calls_saved == 7 - result.speculative_target_calls
    assert result.serial_target_token_steps_saved == result.target_calls_saved
    assert result.accepted_tokens <= result.proposed_tokens
    assert result.accepted_token_coverage == pytest.approx(result.accepted_tokens / 7)


def test_mismatch_commits_accepted_prefix_plus_one_exact_target_fallback():
    result = simulate_generated_prefix_ngram_acceptance(
        (1, 2, 1, 2, 9),
        speculation_k=4,
        max_ngram_length=2,
    )

    assert result.empty_proposal_calls == 3
    assert result.mismatch_calls == 1
    assert result.full_accept_calls == 0
    assert result.proposed_tokens == 2
    assert result.accepted_tokens == 1
    assert result.speculative_target_calls == 4
    assert result.target_calls_saved == 1


def test_aggregate_rates_are_count_weighted_not_window_means():
    short = simulate_generated_prefix_ngram_acceptance(
        (1, 2),
        speculation_k=2,
        max_ngram_length=2,
    )
    long = simulate_generated_prefix_ngram_acceptance(
        (1, 2, 1, 2, 1, 2),
        speculation_k=2,
        max_ngram_length=2,
    )

    aggregate = aggregate_ngram_acceptance_metrics((short, long))

    assert aggregate.generated_tokens == 8
    assert aggregate.accepted_tokens == short.accepted_tokens + long.accepted_tokens
    assert aggregate.accepted_token_coverage == pytest.approx(aggregate.accepted_tokens / 8)


def test_profile_scout_reports_per_window_weighted_totals_limitations_and_advance(tmp_path):
    first = tmp_path / "first.profile.json"
    second = tmp_path / "second.profile.json"
    first.write_text(json.dumps(_profile([1, 2, 1, 2, 1, 2])), encoding="utf-8")
    second.write_text(json.dumps(_profile([3, 4, 3, 4, 3, 4])), encoding="utf-8")

    report = run_generated_prefix_ngram_profile_scout(
        (first, second),
        max_ngram_length=2,
    )

    assert report["result_class"] == "structural_ngram_acceptance_ceiling"
    assert report["claim_scope"] == "cpu_only_generated_prefix_replay_not_tps"
    assert report["window_count"] == 2
    assert len(report["windows"]) == 2
    assert set(report["weighted_totals_by_k"]) == {"2", "4", "8"}
    assert report["weighted_totals_by_k"]["2"]["generated_tokens"] == 12
    assert report["limitations"] == list(STRUCTURAL_LIMITATIONS)
    assert "q_len=K" in report["target_call_accounting_assumption"]
    assert report["triage"]["decision"] == "advance_to_gpu_q_len_k_numeric_and_cost_gate"


def test_profile_scout_can_reject_before_gpu_when_zero_cost_ceiling_is_below_bar(tmp_path):
    profile = tmp_path / "unique.profile.json"
    profile.write_text(json.dumps(_profile(list(range(32)))), encoding="utf-8")

    report = run_generated_prefix_ngram_profile_scout((profile,), max_ngram_length=4)

    assert all(
        totals["target_calls_saved"] == 0
        for totals in report["weighted_totals_by_k"].values()
    )
    assert report["triage"]["decision"] == "reject_before_gpu_below_structural_call_reduction_bar"


@pytest.mark.parametrize(
    "profile",
    [
        {"generation": [{"profile_label": "main_generation", "generated_tokens": 1}]},
        {"generation": [{
            "profile_label": "main_generation",
            "generated_tokens": 2,
            "generated_token_ids": [1],
        }]},
        {"generation": [{
            "profile_label": "main_generation",
            "generated_tokens": 1,
            "generated_token_ids": [True],
        }]},
    ],
)
def test_profile_token_evidence_fails_loudly(profile, tmp_path):
    path = tmp_path / "bad.profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="generated_token"):
        load_profile_token_windows(path, profile_label="main_generation")


def test_profile_with_no_selected_label_fails_loudly(tmp_path):
    path = tmp_path / "timing-only.profile.json"
    path.write_text(json.dumps(_profile([1, 2], label="timing_context")), encoding="utf-8")

    with pytest.raises(ValueError, match="no generation records"):
        load_profile_token_windows(path, profile_label="main_generation")


def test_cli_writes_the_same_structural_report_it_prints(tmp_path, capsys):
    profile = tmp_path / "input.profile.json"
    output = tmp_path / "nested" / "report.json"
    profile.write_text(json.dumps(_profile([1, 2, 1, 2, 1, 2])), encoding="utf-8")

    assert cli_main([
        str(profile),
        "--max-ngram-length", "2",
        "--speculation-k", "2", "4", "8",
        "--output-json", str(output),
    ]) == 0

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["claim_scope"] == "cpu_only_generated_prefix_replay_not_tps"
