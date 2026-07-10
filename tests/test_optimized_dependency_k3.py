from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = REPO_ROOT / "notes/inference-weighted-bucket-ceiling-report.json"
_MODULE_PATH = (
    REPO_ROOT / "osuT5/osuT5/inference/optimized/batch/dependency_k3.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_mapperatorinator_dependency_k3_test", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _parent() -> tuple[dict, str]:
    return _MODULE.load_parent_report(PARENT_PATH)


def _report() -> tuple[dict, dict, str]:
    parent, digest = _parent()
    return (
        _MODULE.build_dependency_aware_k3_report(
            parent, parent_file_sha256=digest
        ),
        parent,
        digest,
    )


def test_dependency_aware_k3_rejects_gpu_below_strong_bar() -> None:
    report, parent, digest = _report()
    _MODULE.validate_dependency_aware_k3_report(
        report, parent_report=parent, parent_file_sha256=digest
    )

    scenarios = report["scenarios"]
    assert scenarios["setup_free"][
        "scheduler_wall_main_tokens_per_second"
    ] == pytest.approx(640.7674443771291)
    assert scenarios["all_setup_charged"][
        "scheduler_wall_main_tokens_per_second"
    ] == pytest.approx(507.4479894201234)
    assert scenarios["dependency_aware"][
        "scheduler_wall_seconds"
    ] == pytest.approx(11.491027866143293)
    assert scenarios["dependency_aware"][
        "scheduler_wall_main_tokens_per_second"
    ] == pytest.approx(518.230402829809)
    assert scenarios["dependency_aware"][
        "headroom_over_500_fraction"
    ] == pytest.approx(0.03646080565961807)
    assert report["strong_gate"]["maximum_scheduler_wall_seconds"] == pytest.approx(
        11.342857142857143
    )
    assert report["strong_gate"]["pass"] is False
    assert report["decision"]["k3_gpu_scout_authorized"] is False


def test_dependency_aware_k3_rejects_exact_five_percent_equality() -> None:
    charged = _MODULE.STRONG_MAX_WALL_SECONDS - _MODULE.PARENT_K3_WALL_SECONDS
    scenario = _MODULE._scenario(
        policy="test_exact_strong_bar",
        charged_setup_seconds=charged,
        excluded_setup_seconds=0.0,
    )
    assert scenario["scheduler_wall_main_tokens_per_second"] == pytest.approx(525.0)
    assert scenario["strong_525_bar_pass"] is False


def test_dependency_aware_k3_transition_ledger_is_complete_and_hashed() -> None:
    report, _, _ = _report()
    ledger = report["transition_ledger"]
    rows = ledger["rows"]

    assert ledger["count"] == 45
    assert len(rows) == 45
    assert ledger["sha256"] == _MODULE.canonical_json_sha256(rows)
    assert rows[0]["song_id"] == "lambada"
    assert rows[0]["predecessor_sequence_index"] == 0
    assert rows[0]["sequence_index"] == 1
    assert rows[8]["sequence_index"] == 9
    assert rows[9]["song_id"] == "pegasus"
    assert rows[-1]["song_id"] == "nube-negra"
    assert rows[-1]["sequence_index"] == 9
    assert len({row["transition_index"] for row in rows}) == 45
    assert ledger["total_setup_seconds"] == pytest.approx(
        _MODULE.PER_WINDOW_SETUP_SECONDS * 45
    )


def test_dependency_aware_k3_rejects_parent_sha_or_source_mutation() -> None:
    parent, digest = _parent()
    with pytest.raises(ValueError, match="file SHA-256 changed"):
        _MODULE.build_dependency_aware_k3_report(
            parent, parent_file_sha256="0" * 64
        )

    mutated = copy.deepcopy(parent)
    mutated["five_request_ideal_k3"]["ideal_wall_seconds"] += 0.1
    with pytest.raises(ValueError, match="source-derived arithmetic"):
        _MODULE.build_dependency_aware_k3_report(
            mutated, parent_file_sha256=digest
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("decision", "k3_gpu_scout_authorized"), True),
        (("decision", "scheduler_or_runtime_wiring_authorized"), True),
        (("strong_gate", "pass"), True),
        (("transition_ledger", "count"), 44),
        (("workload", "dependent_transitions_charged"), 44),
    ],
)
def test_dependency_aware_k3_rejects_report_mutations(
        path: tuple[str, str], value: object,
) -> None:
    report, parent, digest = _report()
    report[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="source-derived arithmetic"):
        _MODULE.validate_dependency_aware_k3_report(
            report, parent_report=parent, parent_file_sha256=digest
        )


def test_dependency_aware_k3_rejects_transition_or_hash_mutation() -> None:
    report, parent, digest = _report()
    report["transition_ledger"]["rows"][17]["setup_seconds"] += 0.001
    report["transition_ledger"]["sha256"] = _MODULE.canonical_json_sha256(
        report["transition_ledger"]["rows"]
    )
    with pytest.raises(ValueError, match="source-derived arithmetic"):
        _MODULE.validate_dependency_aware_k3_report(
            report, parent_report=parent, parent_file_sha256=digest
        )
