from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/dcc/profile_nsight_inference.sbatch"


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_is_valid_bash_and_pins_clean_pushed_worktree() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = _source()

    assert "set -euo pipefail" in source
    assert "status --porcelain" in source
    assert 'rev-parse "$REMOTE_REF"' in source
    assert "MAPPERATORINATOR_COMMIT" in source
    assert "MAPPERATORINATOR_BRANCH" in source
    assert "MAPPERATORINATOR_REMOTE_REF" in source
    assert "expected one visible GPU" in source
    assert "2080 Ti" in source


def test_wrapper_collects_fp32_first_and_isolates_all_passes() -> None:
    source = _source()

    fp32 = source.index("run_precision fp32")
    fp16 = source.index("run_precision fp16")
    assert fp32 < fp16
    assert "full_control" in source
    assert "full_graph" in source
    assert "smoke_control" in source
    assert "smoke_node" in source
    assert "profile_salvalai_smoke15" in source
    assert "untraced_control" in source
    assert "nsys_graph" in source
    assert "nsys_node" in source


def test_graph_and_node_collection_use_bounded_low_overhead_options() -> None:
    source = _source()

    for option in (
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--backtrace=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        '--cuda-graph-trace="$trace_level"',
    ):
        assert option in source
    assert "extract_stage_reports" in source
    assert "utils/nsight_agent_profile.py extract-sqlite" in source
    assert "--graph-level" in source
    assert 'extraction_path = directory / "nsys/stages/extraction.json"' in source
    assert 'item["allow_empty_rows"]' in source


def test_wrapper_gates_transparency_before_next_precision() -> None:
    source = _source()

    body = source[source.index("run_precision()") : source.index("# FP32 must complete")]
    assert body.index('run_control "$precision" full') < body.index(
        'transparency_gate "$precision" full graph'
    )
    assert body.index('run_control "$precision" smoke') < body.index(
        'transparency_gate "$precision" smoke node'
    )
    assert "utils/nsight_agent_profile.py transparency" in source
    assert "utils/nsight_agent_profile.py analyze" in source


def test_ncu_permission_is_probed_once_and_permission_denial_stops_targeting() -> None:
    source = _source()

    assert source.count("sm__cycles_elapsed.avg") == 1
    denial = source.index("if grep -q 'ERR_NVGPUCTRPERM'")
    target = source.index("--section SpeedOfLight")
    assert denial < target
    assert "TARGETED_NCU=()" in source
    assert "--section MemoryWorkloadAnalysis" in source
    assert "--section LaunchStats" in source
    assert "--section Occupancy" in source
    assert "--set full" not in source
    assert "mapperatorinator.roi.inference_generation/" in source


def test_wrapper_keeps_generated_outputs_under_ignored_run_root() -> None:
    source = _source()

    assert 'RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"' in source
    assert "*.nsys-rep" not in source
    assert "manifest.json" in source
    assert "analysis.json" in source
    assert "analysis.txt" in source
    assert "sha256sums.txt" in source
