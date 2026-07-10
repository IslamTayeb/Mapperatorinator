from utils.profile_inference_suite import (
    _aggregate_batch_summaries,
    _generation_interval,
    _profile_batch_summary,
    _scheduler_wall_aggregate,
)
from utils.profile_static_server_batch import _aggregate_runs as _aggregate_static_server_runs


def test_profile_batch_summary_preserves_static_server_batch_metadata():
    profile = {
        "generation": [
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "batch_size": 1,
                "server_batching_mode": "static_ipc",
                "server_elapsed_seconds_attribution": "merged_batch_elapsed_replicated_per_request",
                "server_batch_ids": [11, 12],
                "server_batch_sizes": [5, 3],
                "server_batch_request_counts": [5, 3],
                "server_batch_work_items": [1, 1],
                "server_batch_elapsed_seconds": [1.5, 1.0],
                "server_queue_wait_seconds": [0.3, 0.2],
                "server_first_queue_wait_seconds": 0.3,
                "server_total_queue_wait_seconds": 0.5,
                "server_max_queue_wait_seconds": 0.4,
            },
            {
                "profile_label": "main_generation",
                "mode": "sequential",
                "batch_size": 1,
                "server_batching_mode": "static_ipc",
                "server_elapsed_seconds_attribution": "merged_batch_elapsed_replicated_per_request",
                "server_batch_ids": [11, 12],
                "server_batch_sizes": [5, 3],
                "server_batch_request_counts": [5, 3],
                "server_batch_work_items": [1, 1],
                "server_batch_elapsed_seconds": [1.5, 1.0],
                "server_queue_wait_seconds": [0.15, 0.1],
                "server_first_queue_wait_seconds": 0.15,
                "server_total_queue_wait_seconds": 0.25,
                "server_max_queue_wait_seconds": 0.2,
            },
        ]
    }

    label = _profile_batch_summary(profile)["by_label"]["main_generation"]

    assert label["records"] == 2
    assert label["modes"] == {"sequential": 2}
    assert label["batch_size_histogram"] == {"1": 2}
    assert label["server_batching_modes"] == {"static_ipc": 2}
    assert label["server_elapsed_seconds_attributions"] == {
        "merged_batch_elapsed_replicated_per_request": 2,
    }
    assert label["server_request_record_count"] == 2
    assert label["server_batch_count"] == 4
    assert label["server_batch_size_histogram"] == {"5": 2, "3": 2}
    assert label["server_total_queue_wait_seconds"] == 0.75
    assert label["server_max_queue_wait_seconds"] == 0.4
    assert label["server_total_first_queue_wait_seconds"] == 0.44999999999999996
    assert label["server_max_first_queue_wait_seconds"] == 0.3
    assert label["server_batches"] == [
        {"batch_id": 11, "batch_size": 5, "request_count": 5, "work_items": 1, "elapsed_seconds": 1.5, "queue_wait_seconds": 0.3},
        {"batch_id": 12, "batch_size": 3, "request_count": 3, "work_items": 1, "elapsed_seconds": 1.0, "queue_wait_seconds": 0.2},
        {"batch_id": 11, "batch_size": 5, "request_count": 5, "work_items": 1, "elapsed_seconds": 1.5, "queue_wait_seconds": 0.15},
        {"batch_id": 12, "batch_size": 3, "request_count": 3, "work_items": 1, "elapsed_seconds": 1.0, "queue_wait_seconds": 0.1},
    ]


def test_aggregate_batch_summaries_deduplicates_shared_server_batch_ids():
    run_summary = {
        "generation_batch_summary": {
            "by_label": {
                "main_generation": {
                    "records": 2,
                    "modes": {"sequential": 2},
                    "batch_size_histogram": {"1": 2},
                    "server_batch_size_histogram": {"5": 2, "3": 2},
                    "server_batch_count": 4,
                    "server_request_record_count": 2,
                    "server_total_queue_wait_seconds": 0.75,
                    "server_max_queue_wait_seconds": 0.4,
                    "server_total_first_queue_wait_seconds": 0.45,
                    "server_max_first_queue_wait_seconds": 0.3,
                    "server_batching_modes": {"static_ipc": 2},
                    "server_elapsed_seconds_attributions": {
                        "merged_batch_elapsed_replicated_per_request": 2,
                    },
                    "server_batches": [
                        {"batch_id": 11, "batch_size": 5, "request_count": 5, "work_items": 1, "elapsed_seconds": 1.5},
                        {"batch_id": 12, "batch_size": 3, "request_count": 3, "work_items": 1, "elapsed_seconds": 1.0},
                        {"batch_id": 11, "batch_size": 5, "request_count": 5, "work_items": 1, "elapsed_seconds": 1.5},
                        {"batch_id": 12, "batch_size": 3, "request_count": 3, "work_items": 1, "elapsed_seconds": 1.0},
                    ],
                }
            }
        }
    }

    label = _aggregate_batch_summaries([run_summary])["by_label"]["main_generation"]

    assert label["records"] == 2
    assert label["server_batch_count_attributed"] == 4
    assert label["server_batch_count"] == 2
    assert label["server_unique_batch_size_histogram"] == {"5": 1, "3": 1}
    assert label["server_batch_size_histogram"] == {"5": 2, "3": 2}
    assert label["server_request_record_count"] == 2
    assert label["server_total_queue_wait_seconds"] == 0.75
    assert label["server_max_queue_wait_seconds"] == 0.4
    assert label["server_total_first_queue_wait_seconds"] == 0.45
    assert label["server_max_first_queue_wait_seconds"] == 0.3
    assert label["server_unique_batch_elapsed_seconds_sum"] == 2.5
    assert label["server_unique_batch_elapsed_seconds_max"] == 1.5


def test_static_server_aggregate_classifies_no_batch_and_real_batch_runs():
    no_batch_run = {
        "main_generated_tokens": 100,
        "timing_generated_tokens": 10,
        "request_wall_seconds": 2.0,
        "main_model_elapsed_seconds": 1.0,
        "generation_batch_summary": {
            "by_label": {
                "main_generation": {
                    "records": 1,
                    "server_batch_count": 1,
                    "server_request_record_count": 1,
                    "server_batch_size_histogram": {"1": 1},
                    "server_batches": [
                        {"batch_id": 1, "batch_size": 1, "request_count": 1, "work_items": 1},
                    ],
                }
            }
        },
    }
    real_batch_run = {
        **no_batch_run,
        "generation_batch_summary": {
            "by_label": {
                "main_generation": {
                    "records": 1,
                    "server_batch_count": 1,
                    "server_request_record_count": 1,
                    "server_batch_size_histogram": {"5": 1},
                    "server_batches": [
                        {"batch_id": 2, "batch_size": 5, "request_count": 5, "work_items": 1},
                    ],
                }
            }
        },
    }

    no_batch = _aggregate_static_server_runs([no_batch_run], scheduler_wall_seconds=2.0)
    real_batch = _aggregate_static_server_runs([real_batch_run], scheduler_wall_seconds=2.0)

    assert no_batch["result_class"] == "static_server_no_batch_observed"
    assert not no_batch["server_batch_observed"]
    assert real_batch["result_class"] == "static_server_batch"
    assert real_batch["server_batch_observed"]
    assert real_batch["main_tokens_per_scheduler_second"] == 50.0


def test_suite_scheduler_wall_uses_first_main_start_to_last_main_finish():
    profile = {
        "generation": [
            {
                "profile_label": "main_generation",
                "generation_started_at_perf_counter_seconds": 10.0,
                "generation_finished_at_perf_counter_seconds": 11.0,
            },
            {
                "profile_label": "main_generation",
                "generation_started_at_perf_counter_seconds": 12.0,
                "generation_finished_at_perf_counter_seconds": 14.0,
            },
        ]
    }
    assert _generation_interval(profile, "main_generation") == {
        "started_at_perf_counter_seconds": 10.0,
        "finished_at_perf_counter_seconds": 14.0,
    }

    runs = [
        {
            "main_generated_tokens": 100,
            "timing_generated_tokens": 10,
            "main_generation_started_at_perf_counter_seconds": 10.0,
            "main_generation_finished_at_perf_counter_seconds": 14.0,
            "timing_context_started_at_perf_counter_seconds": 8.0,
            "timing_context_finished_at_perf_counter_seconds": 9.0,
        },
        {
            "main_generated_tokens": 200,
            "timing_generated_tokens": 20,
            "main_generation_started_at_perf_counter_seconds": 20.0,
            "main_generation_finished_at_perf_counter_seconds": 24.0,
            "timing_context_started_at_perf_counter_seconds": 18.0,
            "timing_context_finished_at_perf_counter_seconds": 19.0,
        },
    ]
    main = _scheduler_wall_aggregate(runs, include_timing_context=False)
    complete = _scheduler_wall_aggregate(runs, include_timing_context=True)

    assert main["definition"] == "first_main_start_to_last_main_finish"
    assert main["wall_seconds"] == 14.0
    assert main["main_generated_tokens"] == 300
    assert main["main_tokens_per_scheduler_second"] == 300 / 14
    assert complete["wall_seconds"] == 16.0
    assert complete["total_generated_tokens"] == 330
    assert complete["total_tokens_per_scheduler_second"] == 330 / 16


def test_suite_scheduler_wall_refuses_partial_legacy_intervals():
    runs = [
        {
            "main_generated_tokens": 100,
            "timing_generated_tokens": 10,
            "main_generation_started_at_perf_counter_seconds": 1.0,
            "main_generation_finished_at_perf_counter_seconds": 2.0,
        },
        {
            "main_generated_tokens": 100,
            "timing_generated_tokens": 10,
            "main_generation_started_at_perf_counter_seconds": None,
            "main_generation_finished_at_perf_counter_seconds": None,
        },
    ]

    assert _scheduler_wall_aggregate(runs, include_timing_context=False) is None
