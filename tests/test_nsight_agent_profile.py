from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from utils import nsight_agent_profile as nsight


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def _kernel_summary(path: Path) -> Path:
    return _write_csv(
        path,
        [
            "Time (%)",
            "Total Time (ns)",
            "Instances",
            "Avg (ns)",
            "Min (ns)",
            "Max (ns)",
            "Name",
        ],
        [
            [66.7, 20, 1, 20, 20, 20, "exact::flash_attention_v2<float>"],
            [33.3, 10, 1, 10, 10, 10, "renamed_gemm_128x64"],
        ],
    )


def _kernel_trace(path: Path) -> Path:
    return _write_csv(
        path,
        ["Start (ns)", "Duration (ns)", "GridXYZ", "BlockXYZ", "Name"],
        [
            [0, 10, "1 1 1", "32 1 1", "renamed_gemm_128x64"],
            [5, 20, "2 1 1", "64 1 1", "exact::flash_attention_v2<float>"],
        ],
    )


def _cuda_api_summary(path: Path) -> Path:
    return _write_csv(
        path,
        [
            "Time (%)",
            "Total Time (ns)",
            "Num Calls",
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "StdDev (ns)",
            "Name",
        ],
        [
            [75, 30, 2, 15, 15, 10, 20, 5, "cudaLaunchKernel"],
            [25, 10, 1, 10, 10, 10, 10, 0, "cudaDeviceSynchronize"],
        ],
    )


def _memory_time_summary(path: Path) -> Path:
    return _write_csv(
        path,
        [
            "Time (%)",
            "Total Time (ns)",
            "Num Calls",
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "Operation",
        ],
        [[100, 50, 2, 25, 25, 20, 30, "Host-to-Device"]],
    )


def _memory_size_summary(path: Path) -> Path:
    return _write_csv(
        path,
        [
            "Total (MiB)",
            "Operations",
            "Avg (MiB)",
            "Med (MiB)",
            "Min (MiB)",
            "Max (MiB)",
            "Operation",
        ],
        [[3, 2, 1.5, 1.5, 1, 2, "Host-to-Device"]],
    )


def _nvtx_summary(path: Path) -> Path:
    return _write_csv(
        path,
        [
            "Time (%)",
            "Total Time (ns)",
            "Instances",
            "Avg (ns)",
            "Med (ns)",
            "Min (ns)",
            "Max (ns)",
            "Range",
        ],
        [[100, 80, 1, 80, 80, 80, 80, "mapperatorinator.stage.main_generation"]],
    )


def _nsys_sqlite(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER NOT NULL, end INTEGER, text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER NOT NULL, end INTEGER NOT NULL, demangledName INTEGER NOT NULL,
            gridX INTEGER NOT NULL, gridY INTEGER NOT NULL, gridZ INTEGER NOT NULL,
            blockX INTEGER NOT NULL, blockY INTEGER NOT NULL, blockZ INTEGER NOT NULL,
            correlationId INTEGER, graphNodeId INTEGER, graphId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER NOT NULL, end INTEGER NOT NULL, nameId INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER NOT NULL, end INTEGER NOT NULL, bytes INTEGER NOT NULL,
            copyKind INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER NOT NULL, end INTEGER NOT NULL, bytes INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
            start INTEGER NOT NULL, end INTEGER NOT NULL, syncType INTEGER NOT NULL,
            eventId INTEGER NOT NULL, correlationId INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start INTEGER NOT NULL, end INTEGER NOT NULL, graphId INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "mapperatorinator.stage.timing_context_generation"),
            (2, "mapperatorinator.stage.main_generation"),
            (3, "exact_kernel<float>"),
            (4, "cudaLaunchKernel"),
            (5, "cudaDeviceSynchronize"),
            (6, "generation.decode_graph_replay"),
        ],
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
        [
            (100, 200, "mapperatorinator.stage.timing_context_generation", None),
            (300, 500, "mapperatorinator.stage.main_generation", None),
            (320, 480, "generation.decode_graph_replay", None),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (120, 130, 3, 1, 1, 1, 32, 1, 1, 11, 21, 31),
            (330, 350, 3, 2, 1, 1, 64, 1, 1, 12, 22, 32),
            (490, 510, 3, 1, 1, 1, 32, 1, 1, 13, 23, 33),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
        [(110, 115, 4), (310, 315, 4), (360, 370, 5)],
    )
    connection.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'Host-to-Device')")
    connection.execute("INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (3, 'Stream sync')")
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (380, 390, 1024, 1)")
    connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (140, 145, 256)")
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (400, 410, 3, 7, 9)"
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, ?)",
        [(150, 160, 1), (340, 345, 2), (495, 505, 3)],
    )
    connection.commit()
    connection.close()
    return path


def _stage(*, attribution: dict | None = None) -> dict:
    stage = {
        "outer_wall_ns": 100,
        "synchronized_model_ns": 80,
        "generated_tokens": 8,
        "token_ids_sha256": "tokens",
        "stopping_sha256": "stopping",
        "cache_behavior_sha256": "cache",
        "trace_window_start_ns": 0,
        "trace_window_end_ns": 30,
    }
    if attribution is not None:
        stage["graph_attribution"] = attribution
    return stage


def _run(
    run_id: str,
    *,
    pass_kind: str,
    precision: str = "fp32",
    paired_control: str | None = None,
    artifacts: list[dict] | None = None,
    authoritative: bool | None = None,
    attribution: dict | None = None,
    group: str | None = None,
) -> dict:
    if authoritative is None:
        authoritative = pass_kind == "untraced_control"
    run = {
        "run_id": run_id,
        "run_group_id": group or f"{precision}:accepted",
        "precision": precision,
        "engine_variant": "accepted",
        "pass_kind": pass_kind,
        "authoritative_performance": authoritative,
        "paired_control_run_id": paired_control,
        "workload_contract": {"song": "same", "seed": 12345},
        "output_sha256": "output",
        "output_size_bytes": 123,
        "output_structure": {
            "timing_points": 3,
            "uninherited_timing_points": 2,
            "inherited_timing_points": 1,
            "hit_objects": 10,
            "hit_object_types": {
                "circles": 8,
                "sliders": 2,
                "spinners": 0,
                "holds": 0,
                "unknown": 0,
            },
            "malformed_lines": 0,
            "malformed_by_section": {"TimingPoints": 0, "HitObjects": 0},
            "nonfinite_values": 0,
            "numeric_validation_scope": (
                "timing_all_fields_and_hitobject_core_plus_type_specific_fields"
            ),
            "finite_and_well_formed": True,
        },
        "pipeline_stage_wall_ns": {
            "timing_context_generation": 100,
            "timing_context_postprocess": 7,
            "main_generation": 200,
            "merge_generated_events": 3,
            "resnap_events": 5,
            "postprocess_generate_osu": 11,
            "write_osu": 2,
        },
        "stages": {
            "main_generation": _stage(attribution=attribution),
            "timing_generation": _stage(),
        },
        "artifacts": artifacts or [],
    }
    return run


def _manifest(path: Path, runs: list[dict], **extra) -> Path:
    payload = {
        "schema_version": nsight.MANIFEST_SCHEMA_VERSION,
        "runs": runs,
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inference_profile(
    path: Path,
    *,
    precision: str = "fp32",
    main_tokens: list[int] | None = None,
    model_seconds: float = 1.0,
    dispatch_hits: int = 4,
    graph_capture_seconds: float = 0.5,
    graph_decode_replays: int = 12,
    pass_kind: str = "untraced_control",
) -> Path:
    main_tokens = [1, 2, 3] if main_tokens is None else main_tokens
    metadata = {
        "model_path": "model",
        "audio_path": "song.mp3",
        "beatmap_path": None,
        "seed": 12345,
        "precision": precision,
        "attn_implementation": "sdpa",
        "inference_engine": "optimized",
        "use_server": False,
        "parallel": False,
        "max_batch_size": 1,
        "temperature": 1.0,
        "timing_temperature": 1.0,
        "mania_column_temperature": 1.0,
        "taiko_hit_temperature": 1.0,
        "timeshift_bias": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "do_sample": True,
        "num_beams": 1,
        "cfg_scale": 1.0,
        "lookback": 0.5,
        "lookahead": 0.4,
        "start_time": None,
        "end_time": None,
        "in_context": ["TIMING"],
        "output_type": ["MAP"],
        "sequence_count": 1,
        "song_length_ms": 10_000,
        "optimized_effective_config_version": f"accepted-{precision}",
        "optimized_effective_config": {
            "precision": precision,
            "decode_session_cuda_graph": True,
        },
        "optimized_runtime_owner": "optimized.single.engine",
        "optimized_result_class": "documented-drift",
        "profile_pass_kind": pass_kind,
        "authoritative_performance": pass_kind == "untraced_control",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "nvidia_tf32_override": "0",
        "result_file_sha256": "osu-hash",
        "result_file_size_bytes": 500,
    }
    generation = []
    for label, context, tokens in (
        ("timing_context", "TIMING", [8, 9]),
        ("main_generation", "MAP", main_tokens),
    ):
        generation.append(
            {
                "profile_label": label,
                "context_type": context,
                "mode": "sequential",
                "sequence_index": 0,
                "precision": precision,
                "generated_tokens": len(tokens),
                "generated_tokens_per_sample": [len(tokens)],
                "generated_token_ids": tokens,
                "wall_seconds": model_seconds + 0.1,
                "model_elapsed_seconds": model_seconds,
                "decoder_loop_backend": "active_prefix_cuda_graph",
                "optimized_effective_config_version": f"accepted-{precision}",
                "optimized_dispatch_capture_hits": {"q1_bmm_cross_attention": dispatch_hits},
                "optimized_cuda_graphs": {
                    "graph_count": 1,
                    "decode_replays": graph_decode_replays,
                    "capture_seconds": graph_capture_seconds,
                    "buckets": [
                        {
                            "active_prefix_length": 128,
                            "decode_replays": graph_decode_replays,
                            "capture_seconds": graph_capture_seconds,
                        }
                    ],
                },
            }
        )
    path.write_text(
        json.dumps({"schema_version": 1, "metadata": metadata, "generation": generation}),
        encoding="utf-8",
    )
    return path


def test_strict_fp32_profile_contract_accepts_control_and_traced_passes(
    tmp_path: Path,
) -> None:
    control = _inference_profile(tmp_path / "control.json")
    traced = _inference_profile(
        tmp_path / "traced.json",
        pass_kind="nsys_graph",
    )

    control_report = nsight.validate_strict_fp32_profile(control)
    traced_report = nsight.validate_strict_fp32_profile(traced)

    assert control_report["status"] == "PASS"
    assert control_report["authoritative_performance"] is True
    assert traced_report["status"] == "PASS"
    assert traced_report["authoritative_performance"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("precision", "fp16"),
        ("float32_matmul_precision", "high"),
        ("cuda_matmul_allow_tf32", True),
        ("cudnn_allow_tf32", True),
        ("nvidia_tf32_override", None),
    ],
)
def test_strict_fp32_profile_contract_rejects_precision_or_tf32_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _inference_profile(tmp_path / f"{field}.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(nsight.NsightProfileError, match="strict FP32 profile contract"):
        nsight.validate_strict_fp32_profile(path)


def test_text_summary_separates_model_outer_trace_and_profiler_overhead(
    tmp_path: Path,
) -> None:
    control = _run("control", pass_kind="untraced_control")
    traced = _run(
        "traced",
        pass_kind="nsys_node",
        paired_control="control",
    )
    manifest = _manifest(tmp_path / "manifest.json", [control, traced])

    text = nsight.render_text_summary(nsight.analyze_manifest(manifest))

    assert "main_generation: tokens=8 model_s=" in text
    assert "outer_s=" in text
    assert "traced_tps_authoritative=false" in text
    assert "overhead traced vs control main_generation:" in text


def test_native_kernel_csv_preserves_exact_identity_and_classifies_family(tmp_path):
    summary_path = _kernel_summary(tmp_path / "kernels.csv")

    parsed = nsight.parse_kernel_summary(summary_path)

    assert parsed["accumulated_kernel_ns"] == 30
    assert [kernel["raw_name"] for kernel in parsed["kernels"]] == [
        "exact::flash_attention_v2<float>",
        "renamed_gemm_128x64",
    ]
    assert [kernel["family"] for kernel in parsed["kernels"]] == [
        "fmha_cross_attention",
        "gemm_gemv_projection",
    ]
    assert parsed["kernels"][0]["kernel_id"] != parsed["kernels"][1]["kernel_id"]
    assert parsed["column_resolution"]["canonical_fields"]["total_ns"]["source_column"] == "Total Time (ns)"


@pytest.mark.parametrize(
    ("name", "expected_family"),
    [
        (
            "void q1_rope_cache_attention_kernel<float, 256>(float const*)",
            "native_q1_self_rope_cache",
        ),
        ("q1_attention_kernel<__half, 128>", "native_q1_self_rope_cache"),
        ("fc1_gelu_warp_group_kernel<float>", "fused_fc1_gelu"),
        ("fc2_residual_warp_group_kernel<float>", "fused_fc2_residual"),
        (
            "fmha_cutlassF_f32_aligned_64x64_rf_sm75(PyTorchMemEffAttention::AttentionKernel)",
            "fmha_cross_attention",
        ),
        ("q1_bmm_cross_attention", "fmha_cross_attention"),
        ("volta_sgemm_128x64_tn", "gemm_gemv_projection"),
        ("one_token_out_projection_kernel", "gemm_gemv_projection"),
        (
            "void at::native::radixSortKVInPlace<(int)-2, float, long>()",
            "sampling_radix_sort",
        ),
        ("at::native::multinomial_cuda", "sampling_radix_sort"),
        ("at::native::vectorized_elementwise_kernel", "elementwise"),
        ("at::native::CatArrayBatchedCopy_contig", "memory"),
        ("unknown_vendor_kernel", "other"),
    ],
)
def test_kernel_family_classification_distinguishes_optimization_targets(
    name, expected_family
):
    family, _ = nsight.classify_kernel_family(name)

    assert family == expected_family


def test_osu_structure_summary_counts_objects_and_fails_visible_on_bad_values(tmp_path):
    path = tmp_path / "map.osu"
    path.write_text(
        """osu file format v14

[TimingPoints]
0,500,4,2,1,70,1,0
1000,-100,4,2,1,70,0,0

[HitObjects]
64,192,1000,1,0,0:0:0:0:
128,192,1200,2,0,B|256:192,1,120
nan,192,1400,1,0,0:0:0:0:
""",
        encoding="utf-8",
    )

    structure = nsight.summarize_osu_structure(path)

    assert structure["timing_points"] == 2
    assert structure["uninherited_timing_points"] == 1
    assert structure["inherited_timing_points"] == 1
    assert structure["hit_objects"] == 3
    assert structure["hit_object_types"]["circles"] == 1
    assert structure["hit_object_types"]["sliders"] == 1
    assert structure["malformed_lines"] == 1
    assert structure["nonfinite_values"] == 1
    assert not structure["finite_and_well_formed"]


def test_osu_structure_summary_rejects_incomplete_flags_fractional_types_and_bad_ends(
    tmp_path,
):
    path = tmp_path / "bad-map.osu"
    path.write_text(
        """osu file format v14

[TimingPoints]
0,500
100,500,4,2,1,70,2,0

[HitObjects]
64,192,1000,1.5,0,0:0:0:0:
64,192,1100,0,0,0:0:0:0:
256,192,1200,8,0,nan,0:0:0:0:
""",
        encoding="utf-8",
    )

    structure = nsight.summarize_osu_structure(path)

    assert structure["timing_points"] == 2
    assert structure["hit_objects"] == 3
    assert structure["malformed_by_section"] == {"TimingPoints": 2, "HitObjects": 3}
    assert structure["nonfinite_values"] == 1
    assert all(value == 0 for value in structure["hit_object_types"].values())
    assert not structure["finite_and_well_formed"]


def test_trace_uses_interval_union_not_accumulated_time_for_busy_share(tmp_path):
    trace_path = _kernel_trace(tmp_path / "trace.csv")

    parsed = nsight.parse_kernel_trace(trace_path)

    assert parsed["accumulated_kernel_ns"] == 30
    assert parsed["gpu_busy_union_ns"] == 25
    assert parsed["start_ns"] == 0
    assert parsed["end_ns"] == 25


def test_analyzer_has_explicit_denominators_and_unattributed_graph_replay(tmp_path):
    summary_path = _kernel_summary(tmp_path / "kernels.csv")
    trace_path = _kernel_trace(tmp_path / "trace.csv")
    artifacts = [
        {
            "role": "cuda_kernel_summary",
            "stage": "main_generation",
            "path": summary_path.name,
            "sha256": _sha256(summary_path),
        },
        {
            "role": "cuda_kernel_trace",
            "stage": "main_generation",
            "path": trace_path.name,
            "sha256": _sha256(trace_path),
        },
    ]
    control = _run("control", pass_kind="untraced_control")
    traced = _run(
        "graph",
        pass_kind="nsys_graph",
        paired_control="control",
        artifacts=artifacts,
        attribution={
            "method": "direct_nvtx_containment",
            "verified_on_replay": False,
            "kernel_regions": {"renamed_gemm_128x64": "mlp"},
        },
    )

    result = nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [control, traced]))
    stage = result["runs"]["graph"]["stages"]["main_generation"]

    assert stage["denominators"]["graph.main_generation.accumulated_kernel_ns"]["value_ns"] == 30
    assert stage["trace_window"]["kernel_accumulated_ns"] == 30
    assert stage["trace_window"]["gpu_busy_union_ns"] == 25
    assert stage["trace_window"]["gpu_idle_gap_union_ns"] == 5
    assert stage["trace_window"]["kernel_accumulated_share_of_window"]["percent"] == 100.0
    assert stage["top_kernels"][0]["launch_shapes"] == [
        {"grid": "2 1 1", "block": "64 1 1", "calls": 1}
    ]
    assert stage["top_kernels"][0]["trace_calls"] == 1
    semantic = stage["decoder_semantic_regions"]
    assert semantic["attribution"]["confidence"] == "none"
    assert semantic["regions"]["mlp"]["kernel_accumulated_ns"] == 0
    assert semantic["regions"]["unattributed"]["kernel_accumulated_ns"] == 30
    assert stage["kernel_families"] == [
        {
            "family": "fmha_cross_attention",
            "kernel_count": 1,
            "calls": 1,
            "total_ns": 20,
            "share_of_accumulated_kernel_time": {
                "numerator_ns": 20,
                "denominator_id": "graph.main_generation.accumulated_kernel_ns",
                "denominator_ns": 30,
                "percent": pytest.approx(200 / 3),
                "status": "available",
            },
        },
        {
            "family": "gemm_gemv_projection",
            "kernel_count": 1,
            "calls": 1,
            "total_ns": 10,
            "share_of_accumulated_kernel_time": {
                "numerator_ns": 10,
                "denominator_id": "graph.main_generation.accumulated_kernel_ns",
                "denominator_ns": 30,
                "percent": pytest.approx(100 / 3),
                "status": "available",
            },
        },
    ]
    pipeline = result["runs"]["graph"]["pipeline"]
    assert pipeline["timing_postprocessing"]["total_ns"] == 7
    assert pipeline["main_postprocessing"]["total_ns"] == 21
    assert result["comparisons"][0]["pass"]
    assert not result["comparisons"][0]["stages"]["main_generation"]["profiler_overhead"]["traced_tps_authoritative"]


def test_verified_replay_mapping_attributes_only_exact_names(tmp_path):
    summary_path = _kernel_summary(tmp_path / "kernels.csv")
    traced = _run(
        "graph",
        pass_kind="nsys_node",
        artifacts=[
            {
                "role": "cuda_kernel_summary",
                "stage": "main_generation",
                "path": summary_path.name,
            }
        ],
        attribution={
            "method": "verified_graph_node_correlation",
            "verified_on_replay": True,
            "kernel_regions": {"renamed_gemm_128x64": "mlp"},
        },
    )

    result = nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [traced]))
    semantic = result["runs"]["graph"]["stages"]["main_generation"]["decoder_semantic_regions"]

    assert semantic["attribution"]["confidence"] == "exact"
    assert semantic["regions"]["mlp"]["kernel_accumulated_ns"] == 10
    assert semantic["regions"]["unattributed"]["kernel_accumulated_ns"] == 20
    assert semantic["coverage"]["percent"] == pytest.approx(100 / 3)


def test_stage_scoped_non_kernel_reports_have_explicit_denominators(tmp_path):
    artifacts = []
    for role, path in (
        ("cuda_api_summary", _cuda_api_summary(tmp_path / "api.csv")),
        ("cuda_memory_time_summary", _memory_time_summary(tmp_path / "memory-time.csv")),
        ("cuda_memory_size_summary", _memory_size_summary(tmp_path / "memory-size.csv")),
        ("nvtx_summary", _nvtx_summary(tmp_path / "nvtx.csv")),
        (
            "gpu_time_util",
            _write_csv(
                tmp_path / "util.csv",
                ["Rule Name", "Utilization (%)"],
                [["GPU utilization", 95]],
            ),
        ),
    ):
        artifacts.append(
            {"role": role, "stage": "main_generation", "path": path.name}
        )
    traced = _run("node", pass_kind="nsys_node", artifacts=artifacts)

    result = nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [traced]))
    main = result["runs"]["node"]["stages"]["main_generation"]
    timing = result["runs"]["node"]["stages"]["timing_generation"]

    assert main["cuda_api_report"]["synchronization"][
        "share_of_report_accumulated_time"
    ]["percent"] == 25.0
    assert main["cuda_memory_time_report"]["operations"][0][
        "share_of_report_accumulated_time"
    ]["percent"] == 100.0
    assert main["cuda_memory_size_report"]["operations"][0][
        "share_of_report_total_bytes"
    ]["percent"] == 100.0
    assert main["nvtx_report"]["ranges"][0][
        "share_of_report_accumulated_time"
    ]["percent"] == 100.0
    assert main["analysis_reports"]["gpu_time_util"]["row_count"] == 1
    assert timing["cuda_api_report"]["status"] == "not_available"
    assert "node.main_generation.accumulated_cuda_api_ns" in main["denominators"]
    assert "node.main_generation.total_cuda_memory_bytes" in main["denominators"]


def test_stage_report_artifact_without_stage_fails_loudly(tmp_path):
    api = _cuda_api_summary(tmp_path / "api.csv")
    traced = _run(
        "node",
        pass_kind="nsys_node",
        artifacts=[{"role": "cuda_api_summary", "path": api.name}],
    )

    with pytest.raises(nsight.ArtifactError, match="must declare one stage"):
        nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [traced]))


def test_graph_pass_accepts_empty_optional_kernel_report(tmp_path):
    empty = tmp_path / "empty-kernels.csv"
    empty.write_bytes(b"")
    traced = _run(
        "graph",
        pass_kind="nsys_graph",
        artifacts=[
            {
                "role": "cuda_kernel_summary",
                "stage": "main_generation",
                "path": empty.name,
                "required": False,
            }
        ],
    )

    result = nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [traced]))
    main = result["runs"]["graph"]["stages"]["main_generation"]

    assert main["kernel_report"]["status"] == "empty"
    assert main["top_kernels"] == []
    assert main["decoder_semantic_regions"]["regions"]["unattributed"][
        "kernel_accumulated_ns"
    ] == 0


def test_traced_tps_can_never_be_authoritative(tmp_path):
    traced = _run("bad", pass_kind="nsys_graph", authoritative=True)

    with pytest.raises(nsight.NsightProfileError, match="cannot have authoritative TPS"):
        nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [traced]))


def test_run_group_cannot_mix_precisions(tmp_path):
    fp32 = _run("fp32", pass_kind="untraced_control", group="mixed")
    fp16 = _run("fp16", pass_kind="untraced_control", precision="fp16", group="mixed")

    with pytest.raises(nsight.NsightProfileError, match="mixes precision"):
        nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [fp32, fp16]))


def test_cross_precision_is_divergence_report_not_transparency_gate(tmp_path):
    fp32 = _run("fp32", pass_kind="untraced_control")
    fp16 = _run("fp16", pass_kind="untraced_control", precision="fp16")
    fp16["stages"]["main_generation"]["token_ids_sha256"] = "drift"
    fp16["output_structure"]["hit_objects"] = 9
    fp16["output_structure"]["hit_object_types"]["circles"] = 7
    manifest = _manifest(
        tmp_path / "manifest.json",
        [fp32, fp16],
        comparisons=[
            {
                "comparison_type": "cross_precision_divergence",
                "left_run_id": "fp32",
                "right_run_id": "fp16",
            }
        ],
    )

    result = nsight.analyze_manifest(manifest)
    comparison = result["comparisons"][0]

    assert comparison["comparison_type"] == "cross_precision_divergence"
    assert not comparison["is_transparency_gate"]
    assert not comparison["stages"]["main_generation"]["token_ids_sha256"]["equal"]
    assert not comparison["output_structure"]["fields"]["hit_objects"]["equal"]
    assert comparison["output_structure"]["fields"]["timing_points"]["equal"]
    assert result["runs"]["fp32"]["output_structure"]["finite_and_well_formed"]
    assert set(result["run_groups"]) == {"fp32:accepted", "fp16:accepted"}


def test_output_structure_manifest_rejects_inconsistent_counts(tmp_path):
    run = _run("bad-structure", pass_kind="untraced_control")
    run["output_structure"]["hit_objects"] += 1

    with pytest.raises(nsight.NsightProfileError, match="hit-object counts are inconsistent"):
        nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [run]))


def test_missing_or_renamed_required_column_fails_loudly(tmp_path):
    bad = _write_csv(
        tmp_path / "bad.csv",
        ["Total Time (ns)", "Launch Total", "Name"],
        [[10, 1, "kernel"]],
    )

    with pytest.raises(nsight.ColumnError, match="missing required columns"):
        nsight.parse_kernel_summary(bad)


def test_duration_alias_converts_microseconds_to_nanoseconds(tmp_path):
    trace = _write_csv(
        tmp_path / "trace.csv",
        ["Start (us)", "Duration (us)", "Kernel Name"],
        [[1.5, 2.25, "kernel"]],
    )

    parsed = nsight.parse_kernel_trace(trace)

    assert parsed["launches"][0]["start_ns"] == 1_500
    assert parsed["launches"][0]["duration_ns"] == 2_250


def test_cuda_api_summary_preserves_calls_timings_and_detects_sync(tmp_path):
    parsed = nsight.parse_cuda_api_summary(_cuda_api_summary(tmp_path / "cuda_api.csv"))

    assert parsed["total_calls"] == 3
    assert parsed["accumulated_api_ns"] == 40
    assert parsed["apis"][0]["name"] == "cudaLaunchKernel"
    assert parsed["apis"][0]["average_ns"] == 15
    assert parsed["apis"][0]["minimum_ns"] == 10
    assert parsed["apis"][0]["maximum_ns"] == 20
    assert parsed["synchronization"] == {
        "api_names": ["cudaDeviceSynchronize"],
        "calls": 1,
        "total_ns": 10,
    }


def test_cuda_memory_time_and_size_reports_preserve_operations_and_units(tmp_path):
    time_report = nsight.parse_cuda_memory_time_summary(
        _memory_time_summary(tmp_path / "memory_time.csv")
    )
    size_report = nsight.parse_cuda_memory_size_summary(
        _memory_size_summary(tmp_path / "memory_size.csv")
    )

    assert time_report["operations"] == [
        {
            "operation": "Host-to-Device",
            "calls": 2,
            "total_ns": 50,
            "average_ns": 25,
            "minimum_ns": 20,
            "maximum_ns": 30,
        }
    ]
    assert time_report["column_resolution"]["canonical_fields"]["calls"] == {
        "source_column": "Num Calls",
        "source_unit": None,
        "multiplier": None,
    }
    assert size_report["total_bytes"] == 3 * (1 << 20)
    assert size_report["operations"][0]["average_bytes"] == int(1.5 * (1 << 20))
    assert size_report["column_resolution"]["canonical_fields"]["total_bytes"] == {
        "source_column": "Total (MiB)",
        "source_unit": "bytes",
        "multiplier": 1 << 20,
    }


def test_nvtx_summary_preserves_exact_range_name(tmp_path):
    parsed = nsight.parse_nvtx_summary(_nvtx_summary(tmp_path / "nvtx.csv"))

    assert parsed["ranges"][0]["name"] == "mapperatorinator.stage.main_generation"
    assert parsed["ranges"][0]["calls"] == 1
    assert parsed["ranges"][0]["total_ns"] == 80


def test_analysis_csv_is_structured_with_column_provenance(tmp_path):
    analysis = _write_csv(
        tmp_path / "gpu_gaps.csv",
        ["Severity", "Gap Start (ns)", "Gap Duration (ns)", "Rule Name"],
        [["WARNING", 10, 20, "Long GPU gap"]],
    )

    parsed = nsight.parse_analysis_csv(analysis, analysis_kind="gpu_gaps")

    assert parsed["row_count"] == 1
    assert parsed["column_provenance"][1] == {
        "index": 1,
        "source_column": "Gap Start (ns)",
        "normalized_field": "gap_start_ns",
    }
    assert parsed["rows"][0]["values"] == {
        "severity": "WARNING",
        "gap_start_ns": "10",
        "gap_duration_ns": "20",
        "rule_name": "Long GPU gap",
    }


@pytest.mark.parametrize(
    ("parser", "headers", "rows"),
    [
        (
            nsight.parse_cuda_api_summary,
            ["Total Time (ns)", "API Total", "Avg (ns)", "Min (ns)", "Max (ns)", "Name"],
            [[10, 1, 10, 10, 10, "cudaLaunchKernel"]],
        ),
        (
            nsight.parse_cuda_memory_time_summary,
            ["Total Duration (ns)", "Operations", "Avg (ns)", "Min (ns)", "Max (ns)", "Operation"],
            [[10, 1, 10, 10, 10, "Memset"]],
        ),
        (
            nsight.parse_cuda_memory_size_summary,
            ["Bytes Total", "Operations", "Avg (B)", "Min (B)", "Max (B)", "Operation"],
            [[10, 1, 10, 10, 10, "Memset"]],
        ),
        (
            nsight.parse_nvtx_summary,
            ["Total Time (ns)", "Ranges", "Avg (ns)", "Min (ns)", "Max (ns)", "Range"],
            [[10, 1, 10, 10, 10, "range"]],
        ),
    ],
)
def test_non_kernel_reports_reject_renamed_required_columns(
    tmp_path,
    parser,
    headers,
    rows,
):
    path = _write_csv(tmp_path / f"bad-{parser.__name__}.csv", headers, rows)

    with pytest.raises(nsight.ColumnError, match="missing required columns"):
        parser(path)


def test_analysis_csv_rejects_normalized_column_collision(tmp_path):
    path = _write_csv(
        tmp_path / "bad-analysis.csv",
        ["Gap Start (ns)", "Gap Start ns"],
        [[1, 2]],
    )

    with pytest.raises(nsight.ColumnError, match="colliding normalized columns"):
        nsight.parse_analysis_csv(path, analysis_kind="gpu_gaps")


def test_artifact_hash_mismatch_is_fatal(tmp_path):
    summary_path = _kernel_summary(tmp_path / "kernels.csv")
    run = _run(
        "graph",
        pass_kind="nsys_graph",
        artifacts=[
            {
                "role": "cuda_kernel_summary",
                "stage": "main_generation",
                "path": summary_path.name,
                "sha256": "0" * 64,
            }
        ],
    )

    with pytest.raises(nsight.ArtifactError, match="SHA-256 mismatch"):
        nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [run]))


def test_optional_sqlite_metadata_is_integrity_checked(tmp_path):
    database = tmp_path / "report.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (name TEXT)")
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)",
        [("a",), ("b",)],
    )
    connection.commit()
    connection.close()
    run = _run(
        "control",
        pass_kind="untraced_control",
        artifacts=[{"role": "sqlite", "path": database.name}],
    )

    result = nsight.analyze_manifest(_manifest(tmp_path / "manifest.json", [run]))

    assert result["runs"]["control"]["sqlite"] == {
        "status": "available",
        "integrity_check": "ok",
        "table_rows": {"CUPTI_ACTIVITY_KIND_KERNEL": 2},
    }


def test_extract_sqlite_slices_exact_direct_text_stage_ranges(tmp_path):
    database = _nsys_sqlite(tmp_path / "report.sqlite")
    output_dir = tmp_path / "extracted"

    extraction = nsight.extract_sqlite_reports(database, output_dir)

    assert extraction["stage_assignment"] == (
        "exact_nvtx_text_resolution_and_timestamp_containment"
    )
    assert not extraction["semantic_graph_attribution_inferred"]
    assert extraction["stages"]["timing_generation"]["text_id"] is None
    assert extraction["stages"]["main_generation"]["counts"] == {
        "kernels": 1,
        "kernel_launches": 1,
        "cuda_graph_launches": 1,
        "cuda_api_calls": 2,
        "memory_operations": 1,
        "nvtx_ranges": 2,
        "synchronizations": 1,
        "sync_rows": 1,
    }
    trace = nsight.parse_kernel_trace(
        output_dir / "main_generation.cuda_kernel_trace.csv"
    )
    assert len(trace["launches"]) == 1
    assert trace["launches"][0]["start_ns"] == 330
    assert trace["launches"][0]["correlation_id"] == "12"
    assert trace["launches"][0]["graph_node_id"] == "22"
    assert trace["launches"][0]["graph_id"] == "32"
    api = nsight.parse_cuda_api_summary(
        output_dir / "main_generation.cuda_api_summary.csv"
    )
    assert api["synchronization"]["api_names"] == ["cudaDeviceSynchronize"]
    nvtx = nsight.parse_nvtx_summary(output_dir / "main_generation.nvtx_summary.csv")
    assert [entry["name"] for entry in nvtx["ranges"]] == [
        "mapperatorinator.stage.main_generation",
        "generation.decode_graph_replay",
    ]
    sync = nsight.parse_analysis_csv(
        output_dir / "main_generation.cuda_api_sync.csv",
        analysis_kind="cuda_api_sync",
    )
    assert sync["rows"][0]["values"]["sync_type"] == "Stream sync"
    assert json.loads((output_dir / "extraction.json").read_text(encoding="utf-8"))[
        "extraction_sha256"
    ] == extraction["extraction_sha256"]


def test_semantic_nvtx_views_do_not_claim_graph_replay_attribution(tmp_path):
    nvtx = _write_csv(
        tmp_path / "nvtx.csv",
        [
            "Time (%)",
            "Total Time (ns)",
            "Instances",
            "Avg (ns)",
            "Min (ns)",
            "Max (ns)",
            "Range",
        ],
        [
            [60, 60, 3, 20, 10, 30, "mapperatorinator.generation.sampling"],
            [40, 40, 2, 20, 15, 25, "mapperatorinator.decoder.layer0.self_attn_norm"],
            [30, 30, 2, 15, 10, 20, "mapperatorinator.decoder.layer0.self.residual"],
            [20, 20, 2, 10, 5, 15, "mapperatorinator.decoder.layer0.cross.residual"],
        ],
    )

    view = nsight._semantic_nvtx_views(nsight.parse_nvtx_summary(nvtx))

    assert not view["replay_attribution"]
    assert view["nested_ranges_may_overlap"]
    assert view["generation_regions"]["sampling"]["calls"] == 3
    assert view["decoder_regions"]["self_norm_qkv"][0]["total_ns"] == 40
    assert view["decoder_regions"]["self_out_residual"][0]["total_ns"] == 30
    assert view["decoder_regions"]["cross_out_residual"][0]["total_ns"] == 20


def test_extract_sqlite_requires_current_columns_and_unique_stage_ranges(tmp_path):
    database = _nsys_sqlite(tmp_path / "report.sqlite")
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
        (600, 700, "mapperatorinator.stage.main_generation", None),
    )
    connection.commit()
    connection.close()

    with pytest.raises(nsight.ArtifactError, match="expected exactly one NVTX range"):
        nsight.extract_sqlite_reports(database, tmp_path / "duplicate")

    bad = tmp_path / "bad.sqlite"
    connection = sqlite3.connect(bad)
    connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    with pytest.raises(nsight.ColumnError, match="missing required columns"):
        nsight.extract_sqlite_reports(bad, tmp_path / "bad-output")


def test_graph_level_sqlite_extraction_allows_zero_kernel_rows(tmp_path):
    database = _nsys_sqlite(tmp_path / "report.sqlite")
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
    connection.commit()
    connection.close()

    extraction = nsight.extract_sqlite_reports(
        database,
        tmp_path / "graph",
        graph_level=True,
    )

    assert extraction["stages"]["main_generation"]["counts"]["kernel_launches"] == 0
    kernel_artifact = next(
        artifact
        for artifact in extraction["artifacts"]
        if artifact["stage"] == "main_generation"
        and artifact["role"] == "cuda_kernel_summary"
    )
    assert kernel_artifact["allow_empty_rows"]
    assert kernel_artifact["row_count"] == 0


def test_graph_level_sqlite_extraction_allows_missing_kernel_table(tmp_path):
    database = _nsys_sqlite(tmp_path / "report.sqlite")
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    connection.commit()
    connection.close()

    extraction = nsight.extract_sqlite_reports(
        database,
        tmp_path / "graph-without-node-table",
        graph_level=True,
    )

    assert extraction["stages"]["timing_generation"]["counts"]["kernels"] == 0


def test_extract_sqlite_cli_writes_extraction_json(tmp_path):
    database = _nsys_sqlite(tmp_path / "report.sqlite")
    output_dir = tmp_path / "cli-extracted"

    assert nsight.main(
        [
            "extract-sqlite",
            "--sqlite",
            str(database),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    assert json.loads((output_dir / "extraction.json").read_text(encoding="utf-8"))[
        "schema_version"
    ] == "mapperatorinator.nsight-sqlite-extraction.v1"


def test_ncu_permission_failure_is_explicit_and_stops_targeted_work(tmp_path):
    log = tmp_path / "ncu.log"
    log.write_text("==ERROR== ERR_NVGPUCTRPERM permission denied\n", encoding="utf-8")
    run = _run("control", pass_kind="untraced_control")
    manifest = _manifest(
        tmp_path / "manifest.json",
        [run],
        ncu_probe={
            "attempted": True,
            "attempt_count": 1,
            "exit_code": 1,
            "log_path": log.name,
            "targeted_collections": [],
        },
    )

    result = nsight.analyze_manifest(manifest)

    assert result["status"] == "complete_with_ncu_unavailable"
    assert result["ncu"]["status"] == "permission_denied"
    assert result["ncu"]["error_code"] == "ERR_NVGPUCTRPERM"
    assert result["ncu"]["stop_enforced"]
    assert not result["ncu"]["usable"]

    with pytest.raises(nsight.NsightProfileError, match="targeted NCU collection"):
        nsight.classify_ncu_probe(
            {
                "attempted": True,
                "exit_code": 1,
                "output": "ERR_NVGPUCTRPERM",
                "targeted_collections": ["SpeedOfLight"],
            },
            tmp_path,
        )


def test_generic_ncu_failure_does_not_masquerade_as_permission_denied(tmp_path):
    result = nsight.classify_ncu_probe(
        {"attempted": True, "exit_code": 1, "output": "unknown option"},
        tmp_path,
    )

    assert result["status"] == "probe_failed"
    assert result["error_code"] is None
    assert not result["stop_enforced"]


def test_cli_writes_deterministic_json_and_concise_text(tmp_path):
    run = _run("control", pass_kind="untraced_control")
    manifest = _manifest(tmp_path / "manifest.json", [run])
    output = tmp_path / "summary.json"
    text = tmp_path / "summary.txt"

    assert nsight.main(
        [
            "analyze",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--text-output",
            str(text),
        ]
    ) == 0
    first = output.read_bytes()
    assert nsight.main(
        ["analyze", "--manifest", str(manifest), "--output", str(output)]
    ) == 0

    assert output.read_bytes() == first
    assert json.loads(first)["analysis_sha256"]
    assert "control main_generation" in text.read_text(encoding="utf-8")


def test_profile_transparency_is_exactness_only_and_ignores_performance(tmp_path):
    control = _inference_profile(tmp_path / "control.profile.json", model_seconds=1.0)
    traced = _inference_profile(tmp_path / "traced.profile.json", model_seconds=9.0)

    report = nsight.compare_profile_transparency(control, traced)

    assert report["pass"]
    assert report["performance_ignored_for_pass_fail"]
    assert report["label_checks"]["main_generation"]["pass"]
    assert report["label_checks"]["timing_context"]["pass"]
    assert report["artifact_checks"]["result_file_sha256"]["pass"]
    assert report["graph_cache_check"]["pass"]
    assert "model_elapsed_seconds" not in report["workload_checks"]


def test_profile_transparency_ignores_graph_capture_time_but_not_replay_shape(tmp_path):
    control = _inference_profile(
        tmp_path / "control.profile.json",
        graph_capture_seconds=0.5,
        graph_decode_replays=12,
    )
    different_capture = _inference_profile(
        tmp_path / "different-capture.profile.json",
        graph_capture_seconds=9.5,
        graph_decode_replays=12,
    )
    different_replays = _inference_profile(
        tmp_path / "different-replays.profile.json",
        graph_capture_seconds=0.5,
        graph_decode_replays=13,
    )

    capture_report = nsight.compare_profile_transparency(control, different_capture)
    replay_report = nsight.compare_profile_transparency(control, different_replays)

    assert capture_report["pass"]
    assert capture_report["graph_cache_check"]["pass"]
    assert not replay_report["pass"]
    assert replay_report["graph_cache_check"]["status"] == "FAIL"


def test_profile_transparency_detects_token_count_stop_and_graph_drift(tmp_path):
    control = _inference_profile(tmp_path / "control.profile.json")
    traced = _inference_profile(
        tmp_path / "traced.profile.json",
        main_tokens=[1, 7, 3, 4],
        dispatch_hits=5,
    )

    report = nsight.compare_profile_transparency(control, traced)

    assert not report["pass"]
    checks = report["label_checks"]["main_generation"]["checks"]
    assert checks["token_stream_sha256"]["status"] == "FAIL"
    assert checks["generated_tokens"]["status"] == "FAIL"
    assert checks["stopping_sha256"]["status"] == "FAIL"
    assert report["graph_cache_check"]["status"] == "FAIL"


def test_profile_transparency_rejects_cross_precision(tmp_path):
    control = _inference_profile(tmp_path / "control.profile.json", precision="fp32")
    traced = _inference_profile(tmp_path / "traced.profile.json", precision="fp16")

    with pytest.raises(nsight.NsightProfileError, match="rejects cross-precision"):
        nsight.compare_profile_transparency(control, traced)


def test_transparency_cli_emits_json_and_returns_one_on_drift(tmp_path):
    control = _inference_profile(tmp_path / "control.profile.json")
    traced = _inference_profile(tmp_path / "traced.profile.json", main_tokens=[1, 2, 4])
    output = tmp_path / "transparency.json"

    exit_code = nsight.main(
        [
            "transparency",
            "--control",
            str(control),
            "--traced",
            str(traced),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["pass"] is False
