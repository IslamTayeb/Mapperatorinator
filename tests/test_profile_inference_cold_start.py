from __future__ import annotations

import pytest

from utils.profile_inference_cold_start import (
    ColdStartError,
    _improvement,
    _parse_import_time,
)


def test_parse_import_time_requires_and_extracts_inference() -> None:
    parsed = _parse_import_time(
        "\n".join(
            (
                "import time:       100 |        200 | dependency",
                "import time:       300 |       1000 | inference",
            )
        )
    )
    assert parsed == {"dependency": 200, "inference": 1000}

    with pytest.raises(ColdStartError, match="inference"):
        _parse_import_time("import time: 1 | 2 | dependency")


def test_improvement_reports_saved_seconds_and_percentage() -> None:
    assert _improvement(10.0, 7.5) == {
        "baseline_seconds": 10.0,
        "candidate_seconds": 7.5,
        "saved_seconds": 2.5,
        "improvement_pct": 25.0,
    }
