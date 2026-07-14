from __future__ import annotations

import json
from pathlib import Path

from utils.analyze_persistent_graph_workspace import analyze


def _stage(name: str, start: float, wall: float) -> dict:
    return {
        "name": name,
        "started_at_perf_counter_seconds": start,
        "finished_at_perf_counter_seconds": start + wall,
        "wall_seconds": wall,
        "cuda_memory_allocated_mb": 100.0,
        "cuda_max_memory_allocated_mb": 200.0,
    }


def _profile(
    root: Path,
    label: str,
    *,
    capture_seconds: float,
    graph_delta: int,
    cross_request_hits: int,
) -> Path:
    result = root / f"{label}.osu"
    result.write_text("same map\n", encoding="utf-8")
    cache_storage = [
        {
            "state_signature": "sig",
            "kind": "self_attention_cache",
            "layer": 0,
            "name": "keys",
            "data_ptr": 123,
            "shape": [1, 2, 3, 4],
            "dtype": "torch.float32",
            "device": "cuda:0",
        }
    ]
    generation = []
    for index, profile_label in enumerate(("timing_context", "main_generation")):
        generation.append(
            {
                "profile_label": profile_label,
                "sequence_index": index,
                "model_elapsed_seconds": 2.0 + index,
                "generated_tokens": 10 + index,
                "output_tokens": 10 + index,
                "prompt_tokens": 2,
                "generated_token_ids": [index, index + 1],
                "decode_graph_capture_seconds_delta": capture_seconds / 2,
                "decode_graph_count_delta": graph_delta,
                "optimized_dispatch_mode": "selected",
                "optimized_dispatch_policy": {"self": "native"},
                "optimized_cuda_graphs": {
                    "persistent_workspace": {
                        "cross_request_graph_hits": cross_request_hits,
                        "cache_storage": cache_storage,
                    }
                },
                "cuda_memory_allocated_mb": 150.0,
                "cuda_max_memory_allocated_mb": 250.0,
            }
        )
    profile = root / f"{label}.profile.json"
    profile.write_text(
        json.dumps(
            {
                "metadata": {"result_path": str(result)},
                "generation": generation,
                "stages": [
                    _stage("validate_inputs", 10.0, 0.1),
                    _stage("timing_context_generation", 11.0, 2.5),
                    _stage("main_generation", 14.0, 3.5),
                    _stage("write_osu", 18.0, 0.2),
                ],
            }
        ),
        encoding="utf-8",
    )
    return profile


def _manifest(tmp_path: Path, *, warm_capture: float = 0.0) -> Path:
    cold = _profile(
        tmp_path,
        "cold",
        capture_seconds=0.4,
        graph_delta=1,
        cross_request_hits=0,
    )
    warm = _profile(
        tmp_path,
        "warm",
        capture_seconds=warm_capture,
        graph_delta=0,
        cross_request_hits=9,
    )
    pool = {
        "workspaces_created": 1,
        "workspaces_evicted": 0,
        "max_resident_slots": 1,
        "request_count": 2,
        "closed": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "cold": {
                        "profile_path": str(cold),
                        "process_call_wall_seconds": 7.0,
                    },
                    "warm": {
                        "profile_path": str(warm),
                        "process_call_wall_seconds": 6.0,
                    },
                },
                "final_pool_summary": {"main": pool, "timing": pool},
                "close_completed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_analyzer_accepts_exact_capture_free_warm_request(tmp_path: Path) -> None:
    report = analyze(_manifest(tmp_path))

    assert report["pass"] is True
    assert report["exactness_pass"] is True
    assert report["capture_gate"]["pass"] is True
    assert report["cross_request_graph_hits"] == 18


def test_analyzer_rejects_warm_recapture(tmp_path: Path) -> None:
    report = analyze(_manifest(tmp_path, warm_capture=0.01))

    assert report["pass"] is False
    assert report["capture_gate"]["pass"] is False
