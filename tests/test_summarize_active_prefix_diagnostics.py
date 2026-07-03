from utils.summarize_active_prefix_diagnostics import summarize_active_prefix_diagnostics


def test_summarize_active_prefix_diagnostics_aggregates_records():
    profile = {
        "generation": [
            {
                "profile_label": "main_generation",
                "generated_tokens": 3,
                "model_elapsed_seconds": 1.5,
                "active_prefix_decode_diagnostics": {
                    "decode_steps": 3,
                    "decode_forward_wall_cpu_s": 0.6,
                    "logits_processor_wall_cpu_s": 0.2,
                    "bucket_lengths_seen": [512],
                    "logits_processor_detail_wall_cpu_s": {
                        "0:ConditionalTemperatureLogitsWarper": 0.05,
                        "1:MonotonicTimeShiftLogitsProcessor": 0.15,
                    },
                    "logits_processor_detail_calls": {
                        "0:ConditionalTemperatureLogitsWarper": 3,
                        "1:MonotonicTimeShiftLogitsProcessor": 3,
                    },
                    "cuda_event_ms": {
                        "decode_forward.cuda_graph": 600.0,
                        "prepare_inputs": 100.0,
                    },
                    "cuda_event_calls": {
                        "decode_forward.cuda_graph": 3,
                        "prepare_inputs": 3,
                    },
                    "cuda_graph": {
                        "graphs": [
                            {
                                "active_prefix_length": 512,
                                "capture_seconds": 0.1,
                                "decode_replays": 2,
                                "static_input_shapes": {
                                    "decoder_input_ids": [1, 1],
                                    "cache_position": [1],
                                },
                            }
                        ]
                    },
                },
            },
            {
                "profile_label": "main_generation",
                "generated_tokens": 2,
                "model_elapsed_seconds": 1.0,
                "active_prefix_decode_diagnostics": {
                    "decode_steps": 2,
                    "decode_forward_wall_cpu_s": 0.4,
                    "bucket_lengths_seen": [512, 1024],
                    "logits_processor_detail_wall_cpu_s": {
                        "0:ConditionalTemperatureLogitsWarper": 0.04,
                    },
                    "logits_processor_detail_calls": {
                        "0:ConditionalTemperatureLogitsWarper": 2,
                    },
                    "cuda_event_ms": {
                        "decode_forward.cuda_graph": 400.0,
                    },
                    "cuda_event_calls": {
                        "decode_forward.cuda_graph": 2,
                    },
                    "cuda_graph": {
                        "graphs": [
                            {
                                "active_prefix_length": 512,
                                "capture_seconds": 0.2,
                                "decode_replays": 1,
                                "static_input_shapes": {
                                    "decoder_input_ids": [1, 1],
                                    "cache_position": [1],
                                },
                            },
                            {
                                "active_prefix_length": 1024,
                                "capture_seconds": 0.3,
                                "decode_replays": 1,
                                "static_input_shapes": {
                                    "decoder_input_ids": [1, 1],
                                    "cache_position": [1],
                                },
                            },
                        ]
                    },
                },
            },
            {
                "profile_label": "timing_generation",
                "generated_tokens": 9,
                "model_elapsed_seconds": 9.0,
            },
        ]
    }

    summary = summarize_active_prefix_diagnostics(profile, label="main_generation")

    assert summary["records"] == 2
    assert summary["records_with_diagnostics"] == 2
    assert summary["generated_tokens"] == 5
    assert summary["model_elapsed_seconds"] == 2.5
    assert summary["tokens_per_second"] == 2.0
    assert summary["numeric_totals"]["decode_steps"] == 5
    assert summary["wall_totals"]["decode_forward_wall_cpu_s"] == 1.0
    assert summary["processor_wall_cpu_s"]["0:ConditionalTemperatureLogitsWarper"] == 0.09
    assert summary["processor_calls"]["0:ConditionalTemperatureLogitsWarper"] == 5
    assert summary["cuda_event_ms"]["decode_forward.cuda_graph"] == 1000.0
    assert summary["cuda_event_calls"]["decode_forward.cuda_graph"] == 5
    ceilings = summary["cuda_event_single_range_ceilings"]
    assert ceilings[0]["name"] == "decode_forward.cuda_graph"
    assert ceilings[0]["event_seconds"] == 1.0
    assert ceilings[0]["calls"] == 5
    assert ceilings[0]["model_time_pct"] == 40.0
    assert ceilings[0]["above_5pct_keep_bar"] is True
    assert round(ceilings[0]["estimated_tokens_per_second_if_removed"], 6) == round(5 / 1.5, 6)
    assert ceilings[1]["name"] == "prepare_inputs"
    assert ceilings[1]["above_5pct_keep_bar"] is False
    assert summary["bucket_lengths_seen_counts"] == {"512": 2, "1024": 1}
    assert summary["cuda_graph_totals"]["graphs"] == 3
    assert summary["cuda_graph_totals"]["decode_replays"] == 4
    assert summary["cuda_graph_by_prefix"]["512"]["graphs"] == 2
    assert summary["cuda_graph_by_prefix"]["1024"]["capture_seconds"] == 0.3
    duplicate = summary["cuda_graph_duplicate_capture_ceiling"]
    assert duplicate["normalized_graphs"] == 2
    assert duplicate["graph_captures"] == 3
    assert duplicate["duplicate_graph_captures"] == 1
    assert round(duplicate["duplicate_capture_seconds"], 6) == 0.2
    assert round(duplicate["duplicate_capture_model_pct"], 6) == 8.0
    assert round(duplicate["estimated_tokens_per_second_without_duplicate_capture"], 6) == round(5 / 2.3, 6)
    assert duplicate["by_prefix"]["512"]["normalized_graphs"] == 1
    assert duplicate["by_prefix"]["512"]["duplicate_graphs"] == 1
    assert round(duplicate["by_prefix"]["512"]["duplicate_capture_seconds"], 6) == 0.2
