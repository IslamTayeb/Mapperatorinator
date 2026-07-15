import pytest

from utils.profile_compile_graph_cross_bmm import MODES, SENTINEL_BUCKETS, summarize


def _bucket(eager: float, compiled: float, *, exact: bool = True):
    return {
        "eager_outer_graph_ms_per_call": eager,
        "modes": {
            mode: {
                "pass": True,
                "compile_seconds": 2.0,
                "outer_capture_seconds": 0.1,
                "outer_graph_ms_per_call": compiled,
                "exact": exact,
                "max_abs": 0.0 if exact else 1e-5,
            }
            for mode in MODES
        },
    }


def test_summary_projects_twelve_layer_live_saving_and_setup() -> None:
    buckets = {str(prefix): _bucket(0.2, 0.1) for prefix in SENTINEL_BUCKETS}
    counts = {128: 10, 576: 20, 640: 30, 832: 5}

    report = summarize(buckets, live_counts=counts)

    assert report["measured_replays"] == 60
    assert report["total_live_replays"] == 65
    assert report["best_positive_mode"] in MODES
    expected = 12 * 60 * 0.1 / 1_000
    assert report["modes"]["default"]["projected_main_saving_seconds"] == pytest.approx(
        expected
    )
    assert report["modes"]["default"]["compile_plus_capture_setup_seconds"] == pytest.approx(
        6.3
    )


def test_summary_requires_all_sentinels_and_marks_failed_mode() -> None:
    buckets = {str(prefix): _bucket(0.2, 0.1) for prefix in SENTINEL_BUCKETS}
    buckets["576"]["modes"]["reduce-overhead"] = {
        "pass": False,
        "error": "nested graph",
    }
    counts = {prefix: 1 for prefix in SENTINEL_BUCKETS}

    report = summarize(buckets, live_counts=counts)
    assert report["modes"]["reduce-overhead"] == {
        "pass": False,
        "failure_buckets": [576],
    }

    with pytest.raises(ValueError, match="sentinel"):
        summarize({"128": buckets["128"]}, live_counts=counts)
