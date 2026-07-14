from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import build_runtime_nsight_manifest as manifest_builder
from utils.summarize_runtime_reprofile import (
    _divergence,
    _export_csvs,
    _pipeline,
    _stage,
)
from utils.validate_selected_runtime_nsight import _runtime_contract


def _profile(main_tokens: list[int]) -> dict:
    return {
        "metadata": {"precision": "fp32"},
        "summary": {
            "generation_by_label": {
                "timing_context": {
                    "records": 1,
                    "wall_seconds": 2.0,
                    "model_elapsed_seconds": 1.5,
                    "generated_tokens": 2,
                },
                "main_generation": {
                    "records": 1,
                    "wall_seconds": 4.0,
                    "model_elapsed_seconds": 3.0,
                    "generated_tokens": len(main_tokens),
                },
            },
            "stage_wall_seconds": {
                "timing_context_generation": 2.0,
                "timing_context_postprocess": 0.25,
                "main_generation": 4.0,
                "merge_generated_events": 0.5,
                "write_osu": 0.25,
                "audio_load": 1.0,
            },
        },
        "generation": [
            {
                "profile_label": "timing_context",
                "generated_tokens": 2,
                "generated_token_ids": [1, 2],
                "context_type": "TIMING",
                "mode": "timing",
            },
            {
                "profile_label": "main_generation",
                "generated_tokens": len(main_tokens),
                "generated_token_ids": main_tokens,
                "context_type": "MAP",
                "mode": "main",
            },
        ],
    }


def _evidence() -> dict:
    return {
        "labels": {
            "timing_context": {"logical_steps": 200},
            "main_generation": {"logical_steps": 8294},
        }
    }


def _osu(path: Path, x: int) -> Path:
    path.write_text(
        "osu file format v14\n\n"
        "[TimingPoints]\n"
        "0,500,4,2,0,100,1,0\n\n"
        "[HitObjects]\n"
        f"{x},192,1000,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )
    return path


def test_fixed_work_stage_uses_logical_steps_not_consumer_token_count():
    result = _stage(_profile([3, 4, 5]), _evidence(), "main_generation")

    assert result["logical_steps"] == 8294
    assert result["consumer_tokens"] == 3
    assert result["fixed_work_tokens_per_second"] == pytest.approx(8294 / 3)
    assert result["post_model_seconds"] == 1.0


def test_pipeline_separates_timing_main_postprocessing_and_other_request_wall():
    result = _pipeline(_profile([3]))

    assert result["request_profiled_stage_wall_seconds"] == 8.0
    assert result["timing_postprocessing_seconds"] == 0.25
    assert result["main_postprocessing_seconds"] == 0.75
    assert result["generation_stage_wall_seconds"] == 6.0
    assert result["other_profiled_stage_wall_seconds"] == 1.0


def test_relaxed_divergence_reports_tokens_stopping_structure_and_bytes(tmp_path):
    accepted = _profile([3, 4, 5])
    selected = _profile([3, 9])
    accepted_osu = _osu(tmp_path / "accepted.osu", 64)
    selected_osu = _osu(tmp_path / "selected.osu", 128)

    result = _divergence(accepted, selected, accepted_osu, selected_osu)

    main = result["labels"]["main_generation"]
    assert main["aligned_tokens"] == 2
    assert main["aligned_mismatches"] == 1
    assert main["aligned_mismatch_fraction"] == 0.5
    assert not main["token_stream_equal"]
    assert not main["stopping_equal"]
    assert not result["result_byte_identical"]
    assert result["finite_and_well_formed"]


def test_csv_export_keeps_kernel_family_launch_and_gap_evidence(tmp_path):
    timing = {
        "logical_steps": 200,
        "consumer_tokens": 190,
        "outer_wall_seconds": 2.0,
        "synchronized_model_seconds": 1.5,
        "post_model_seconds": 0.5,
        "fixed_work_tokens_per_second": 200 / 1.5,
    }
    main = {
        "logical_steps": 8294,
        "consumer_tokens": 8000,
        "outer_wall_seconds": 30.0,
        "synchronized_model_seconds": 25.0,
        "post_model_seconds": 5.0,
        "fixed_work_tokens_per_second": 331.76,
    }
    report = {
        "runs": {
            "selected_control": {
                "generation": {
                    "timing_generation": timing,
                    "main_generation": main,
                },
                "pipeline": {
                    "request_profiled_stage_wall_seconds": 34.0,
                    "timing_postprocessing_seconds": 0.5,
                    "main_postprocessing_seconds": 1.0,
                    "other_profiled_stage_wall_seconds": 0.5,
                },
            }
        }
    }
    analysis = {
        "runs": {
            "selected_node": {
                "stages": {
                    "main_generation": {
                        "status": "available",
                        "kernel_families": [
                            {
                                "family": "gemm_gemv_projection",
                                "kernel_count": 1,
                                "calls": 10,
                                "total_ns": 100,
                                "share_of_accumulated_kernel_time": {"percent": 50.0},
                            }
                        ],
                        "top_kernels": [
                            {
                                "raw_name": "hot_kernel",
                                "family": "gemm_gemv_projection",
                                "calls": 10,
                                "total_ns": 100,
                                "average_ns": 10,
                                "launch_shapes": [{"grid": "1 1 1", "block": "32 1 1"}],
                            }
                        ],
                        "trace_window": {
                            "wall_ns": 200,
                            "gpu_busy_union_ns": 150,
                            "gpu_idle_gap_union_ns": 50,
                        },
                        "cuda_api_report": {
                            "synchronization": {"calls": 2, "total_ns": 20}
                        },
                        "cuda_memory_size_report": {
                            "total_calls": 3,
                            "total_bytes": 64,
                        },
                        "diagnostic_counts": {"kernel_rows": 10, "runtime_rows": 12},
                    }
                }
            }
        }
    }

    outputs = _export_csvs(report, analysis, tmp_path)

    assert set(outputs) == {
        "stage_budget_csv",
        "request_pipeline_csv",
        "kernel_families_csv",
        "top_kernels_csv",
        "copy_sync_launch_gaps_csv",
    }
    assert "hot_kernel" in (tmp_path / "top_kernels.csv").read_text()
    assert "gpu_idle_gap_union_ns" in (
        tmp_path / "copy_sync_launch_gaps.csv"
    ).read_text()
    assert "complete_profiled_request_wall_seconds" in (
        tmp_path / "request_pipeline.csv"
    ).read_text()


def test_dcc_wrapper_is_fixed_work_runtime_spec_driven_and_report_only():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/dcc/profile_selected_runtime_nsight.sbatch").read_text()
    spec = json.loads(
        (root / "scripts/dcc/runtime_specs/k1_int8_fp16_cross.json").read_text()
    )

    assert "#SBATCH --time=01:00:00" in wrapper
    assert "EXPECTED_MAIN_STEPS=8294" in wrapper
    assert "MAPPERATORINATOR_RUNTIME_SPEC" in wrapper
    assert "selected_control" in wrapper
    assert "selected_budget" not in wrapper
    assert "selected_graph" in wrapper
    assert "selected_node" in wrapper
    assert "profile_pass_kind" in wrapper
    assert "cuda-graph-trace=\"$trace_level\"" in wrapper
    assert ".html" not in wrapper
    assert ".png" not in wrapper
    assert ".md" not in wrapper
    assert "PYTHONHASHSEED=0" in wrapper
    assert "runtime-nsight-$COMMIT" in wrapper
    assert "validate_selected_runtime_nsight.py" in wrapper
    assert "topology-contract.json" in wrapper
    assert "request_pipeline.csv" in wrapper
    assert spec["kwargs"]["block_size"] == 4
    assert spec["kwargs"]["graph_remainders"] is True
    assert spec["kwargs"]["initializer_name"] == (
        "initialize_approximate_int8_mlp_weight_only_cross"
    )
    assert spec["kwargs"]["initializer_kwargs"] == {
        "mode": "fp16_packed_projections"
    }
    assert spec["kwargs"]["minimum_bindings"] == 2
    assert spec["factory"].count(":") == 1


def test_selected_runtime_contract_requires_exact_composed_topology():
    stats = {
        "module_count": 12,
        "group_count": 1,
        "forwards": 10,
        "computes": 10,
        "expected_computes": 10,
        "reuses": 110,
        "expected_reuses": 110,
    }
    initialization = {
        "cross_candidate": {
            "mode": "fp16_packed_projections",
            "scope": "main-model-only",
            "attention_accumulation": "fp32",
            "production_selector_unchanged": True,
            "projection_delta_only": True,
            "accepted_q1_bmm": True,
            "incremental_exactness_required": True,
        },
        "int8_mlp_overlay": {
            "version": "per-row-symmetric-int8-mlp-v1",
            "scope": "main-model-decoder-mlp-only",
            "dispatch_counter": "int8_weight_mlp_tail",
        },
    }
    kwargs = {
        "block_size": 4,
        "graph_remainders": True,
        "initializer_name": "initialize_approximate_int8_mlp_weight_only_cross",
        "initializer_kwargs": {"mode": "fp16_packed_projections"},
        "shared_rope_binding_index": 0,
        "initializer_binding_index": 0,
        "minimum_bindings": 2,
    }
    evidence = {
        "candidate": True,
        "expected_main_steps": 8294,
        "labels": {"main_generation": {"logical_steps": 8294}},
        "runtime": {
            "name": "k4-k1-int8-fp16-packed-cross",
            "factory": "utils.final_confirmation_runtime:kblock_shared_rope_weight_plugin",
            "binding_count": 2,
            "spec": {"kwargs": kwargs},
            "initialization": initialization,
            "temporary_hooks": [
                {"kind": "block_decode", "block_size": 4, "graph_remainders": True},
                {"kind": "shared_decoder_rope", "binding_index": 0, "stats": stats},
                {
                    "kind": "runtime_initializer",
                    "binding_index": 0,
                    "name": "initialize_approximate_int8_mlp_weight_only_cross",
                    "kwargs": {"mode": "fp16_packed_projections"},
                },
            ],
        },
    }

    result = _runtime_contract(evidence, run_id="selected_control")
    assert result["binding_count"] == 2
    assert result["shared_rope"]["reuses"] == 110

    evidence["runtime"]["spec"]["kwargs"]["block_size"] = 1
    with pytest.raises(ValueError, match="spec.kwargs"):
        _runtime_contract(evidence, run_id="selected_control")


def test_manifest_builder_uses_paired_controls_without_explicit_fake_comparisons(
    tmp_path,
    monkeypatch,
):
    audio = tmp_path / "audio.mp3"
    runtime_spec = tmp_path / "runtime.json"
    fixed_manifest = tmp_path / "fixed.json"
    for path in (audio, runtime_spec, fixed_manifest):
        path.write_text(path.name, encoding="utf-8")

    def fake_run(run_id, evidence_path, **kwargs):
        del evidence_path, kwargs
        return {
            "run_id": run_id,
            "paired_control_run_id": manifest_builder.PAIRED_CONTROL.get(run_id),
        }

    monkeypatch.setattr(manifest_builder, "_run", fake_run)
    payload = manifest_builder.build(
        run_root=tmp_path,
        evidence_paths={
            run_id: tmp_path / f"{run_id}.json"
            for run_id in manifest_builder.RUN_IDS
        },
        commit="a" * 40,
        branch="codex/runtime",
        remote_ref="islamtayeb/codex/runtime",
        audio=audio,
        runtime_spec=runtime_spec,
        fixed_manifest=fixed_manifest,
    )

    by_id = {run["run_id"]: run for run in payload["runs"]}
    assert payload["comparisons"] == []
    assert by_id["selected_graph"]["paired_control_run_id"] == "selected_control"
    assert by_id["selected_node"]["paired_control_run_id"] == "selected_control"
