from __future__ import annotations

import pytest

from utils.profile_combined_startup import CombinedStartupError, _comparison, summarize


def _row(
    mode: str,
    *,
    imported: float,
    native: float,
    optional: list[str] | None = None,
    library: str = "same",
) -> dict:
    return {
        "mode": mode,
        "repo": "/repo",
        "import_seconds": imported,
        "native_load_seconds": native,
        "ready_seconds": imported + native,
        "optional_modules_loaded": optional or [],
        "extensions": {
            "extension": {
                "library_sha256": library,
                "functions": ["call"],
                "probes": {
                    "call": {
                        "kind": "exception",
                        "type": "TypeError",
                        "message": "same",
                    }
                },
            }
        },
    }


def test_combined_summary_quantifies_independent_startup_wins() -> None:
    rows = {
        "lazy_parent": [_row("cached", imported=10.0, native=3.0)],
        "aot_parent": [_row("direct", imported=14.0, native=0.1, optional=["wandb"])],
        "combined": [_row("direct", imported=10.0, native=0.1)],
    }

    report = summarize(rows)

    assert report["status"] == "PASS"
    assert report["comparisons"]["lazy_parent"]["ready_seconds"]["saved_seconds"] == pytest.approx(2.9)
    assert report["comparisons"]["aot_parent"]["ready_seconds"]["saved_seconds"] == pytest.approx(4.0)
    assert report["independent_combination"]["predicted_ready_seconds"] == pytest.approx(10.1)
    assert report["independent_combination"]["interaction_seconds"] == pytest.approx(0.0)


def test_combined_summary_rejects_extension_drift_and_lazy_regression() -> None:
    rows = {
        "lazy_parent": [_row("cached", imported=10.0, native=3.0)],
        "aot_parent": [_row("direct", imported=14.0, native=0.1)],
        "combined": [_row("direct", imported=11.0, native=0.1, library="changed")],
    }

    report = summarize(rows)

    assert report["status"] == "FAIL"
    assert not report["checks"]["extension_parity"]
    assert not report["checks"]["lazy_import_retained"]


def test_combined_summary_requires_all_variants() -> None:
    with pytest.raises(CombinedStartupError, match="every parent"):
        summarize({"combined": [_row("direct", imported=1.0, native=0.1)]})


def test_comparison_reports_saved_time() -> None:
    assert _comparison(12.0, 9.0) == {
        "parent_seconds": 12.0,
        "combined_seconds": 9.0,
        "saved_seconds": 3.0,
        "improvement_pct": 25.0,
    }
