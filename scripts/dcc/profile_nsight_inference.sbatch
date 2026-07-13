#!/usr/bin/env bash
#SBATCH --job-name=nsight-inference
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "nsight-inference: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

capture_capability() {
  local output=$1
  shift
  set +e
  "$@" > "$output" 2> "$output.stderr.txt"
  local exit_code=$?
  set -e
  printf '%s\n' "$exit_code" > "$output.exit-code.txt"
  if [[ ! -s "$output" && ! -s "$output.stderr.txt" ]]; then
    die "capability probe produced no output: $*"
  fi
}

require_successful_capability() {
  local output=$1
  local exit_code
  exit_code=$(<"$output.exit-code.txt")
  [[ "$exit_code" == 0 ]] || die "required capability probe failed with exit code $exit_code: $output"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean profiler worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed profiler commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the profiler branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
RUN_LABEL=${RUN_LABEL:-nsight-agent-inference}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
CAPABILITIES="$RUN_ROOT/capabilities"
RUNS="$RUN_ROOT/runs"

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing profiler worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite existing run root $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "profiler worktree is dirty at $REPO"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "worktree HEAD does not match MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "pushed remote ref $REMOTE_REF does not match MAPPERATORINATOR_COMMIT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$CAPABILITIES" "$RUNS"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

for tool in ffmpeg ffprobe nvidia-smi nsys ncu dcgmi; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
NSYS=$(command -v nsys)
NCU=$(command -v ncu)
DCGMI=$(command -v dcgmi)

capture_capability "$CAPABILITIES/nsys-version.txt" "$NSYS" --version
require_successful_capability "$CAPABILITIES/nsys-version.txt"
capture_capability "$CAPABILITIES/nsys-profile-help.txt" "$NSYS" profile --help
capture_capability "$CAPABILITIES/nsys-stats-help.txt" "$NSYS" stats --help
capture_capability "$CAPABILITIES/nsys-help-reports.txt" "$NSYS" stats --help-reports
# Nsight Systems 2025.6 exits nonzero for this help request while still
# returning the complete supported-rule interface.  Nonempty captured output
# is authoritative for help probes; actual profiler commands remain fail-loud.
capture_capability "$CAPABILITIES/nsys-analyze-help.txt" "$NSYS" analyze --help
capture_capability "$CAPABILITIES/ncu-version.txt" "$NCU" --version
require_successful_capability "$CAPABILITIES/ncu-version.txt"
capture_capability "$CAPABILITIES/ncu-help.txt" "$NCU" --help
capture_capability "$CAPABILITIES/ncu-sets.txt" "$NCU" --list-sets
capture_capability "$CAPABILITIES/ncu-sections.txt" "$NCU" --list-sections
capture_capability "$CAPABILITIES/dcgmi-version.txt" "$DCGMI" --version
set +e
"$NSYS" recipe --help > "$CAPABILITIES/nsys-recipe-help.txt" 2>&1
NSYS_RECIPE_EXIT=$?
"$DCGMI" discovery -l > "$CAPABILITIES/dcgmi-discovery.txt" 2>&1
DCGMI_DISCOVERY_EXIT=$?
"$PYTHON" -c 'import ncu_report; print(ncu_report.__file__)' \
  > "$CAPABILITIES/ncu-python-report.txt" 2>&1
NCU_PYTHON_REPORT_EXIT=$?
set -e
printf '%s\n' "$NSYS_RECIPE_EXIT" > "$CAPABILITIES/nsys-recipe.exit-code.txt"
printf '%s\n' "$DCGMI_DISCOVERY_EXIT" > "$CAPABILITIES/dcgmi-discovery.exit-code.txt"
printf '%s\n' "$NCU_PYTHON_REPORT_EXIT" > "$CAPABILITIES/ncu-python-report.exit-code.txt"

for required in --capture-range --capture-range-end --cuda-graph-trace --sample --cpuctxsw --trace; do
  grep -q -- "$required" "$CAPABILITIES/nsys-profile-help.txt" || die "nsys profile does not support $required"
done
for required in --report --format; do
  grep -q -- "$required" "$CAPABILITIES/nsys-stats-help.txt" || die "nsys stats does not support $required"
done
for required in --section --metrics --launch-count --nvtx-include --graph-profiling; do
  grep -q -- "$required" "$CAPABILITIES/ncu-help.txt" || die "ncu does not support $required"
done
for section in SpeedOfLight MemoryWorkloadAnalysis LaunchStats Occupancy; do
  grep -q -- "$section" "$CAPABILITIES/ncu-sections.txt" || die "ncu section is unavailable: $section"
done

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected one visible GPU, got ${#GPU_NAMES[@]}"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "run_root=$RUN_ROOT"
  echo "nsys=$NSYS"
  echo "ncu=$NCU"
  echo "dcgmi=$DCGMI"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
cleanup() {
  kill "$SMI_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO"

INFERENCE_COMMAND=()
build_inference_command() {
  local precision=$1
  local config_name=$2
  local pass_kind=$3
  local detail_ranges=$4
  local cuda_capture=$5
  local output_dir=$6
  INFERENCE_COMMAND=(
    "$PYTHON" inference.py --config-name "$config_name"
    audio_path="$AUDIO"
    output_path="$output_dir"
    device=cuda
    precision="$precision"
    attn_implementation=sdpa
    inference_engine=optimized
    use_server=false
    parallel=false
    cfg_scale=1.0
    num_beams=1
    seed=12345
    profile_inference=true
    profile_detail_ranges="$detail_ranges"
    profile_cuda_capture="$cuda_capture"
    profile_pass_kind="$pass_kind"
  )
}

capture_run_paths() {
  local run_dir=$1
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "expected one profile JSON in $run_dir/output, got ${#profiles[@]}"
  local profile=${profiles[0]}
  local result=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$result"
  printf '%s\n' "$profile" > "$run_dir/profile-path.txt"
  printf '%s\n' "$result" > "$run_dir/result-path.txt"
}

run_control() {
  local precision=$1
  local scope=$2
  local config_name=$3
  local run_dir="$RUNS/$precision/${scope}_control"
  mkdir -p "$run_dir/output"
  build_inference_command "$precision" "$config_name" untraced_control false false "$run_dir/output"
  printf '%q ' "${INFERENCE_COMMAND[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "${INFERENCE_COMMAND[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  capture_run_paths "$run_dir"
}

report_supported() {
  # Parameterized reports are printed as e.g. ``cuda_gpu_kern_sum[:nvtx-name]``.
  grep -Eq "^[[:space:]]*$1(\\[|:|[[:space:]]|$)" "$CAPABILITIES/nsys-help-reports.txt"
}

export_nsys_report() {
  local rep=$1
  local report=$2
  local output=$3
  local required=$4
  if ! report_supported "$report"; then
    [[ "$required" == false ]] && return 0
    die "required nsys stats report is unavailable: $report"
  fi
  set +e
  "$NSYS" stats --report "$report" --format csv "$rep" \
    > "$output" 2> "$output.stderr.txt"
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    [[ "$required" == false ]] && return 0
    die "nsys stats report $report failed with exit code $exit_code"
  fi
  [[ "$required" == false ]] || require_nonempty "$output"
}

extract_stage_reports() {
  local sqlite_path=$1
  local output_dir=$2
  local trace_level=$3
  local args=(
    utils/nsight_agent_profile.py extract-sqlite
    --sqlite "$sqlite_path"
    --output-dir "$output_dir"
  )
  if [[ "$trace_level" == graph ]]; then
    args+=(--graph-level)
  fi
  "$PYTHON" "${args[@]}"
  require_nonempty "$output_dir/extraction.json"
}

export_nsys() {
  local run_dir=$1
  local trace_level=$2
  local rep="$run_dir/nsys/trace.nsys-rep"
  local sqlite_path="$run_dir/nsys/trace.sqlite"
  require_nonempty "$rep"
  "$NSYS" export --type sqlite --force-overwrite=true --output "$sqlite_path" "$rep" \
    > "$run_dir/nsys/export.stdout.txt" 2> "$run_dir/nsys/export.stderr.txt"
  require_nonempty "$sqlite_path"

  export_nsys_report "$rep" cuda_api_sum "$run_dir/nsys/cuda_api_sum.csv" true
  export_nsys_report "$rep" nvtx_sum "$run_dir/nsys/nvtx_sum.csv" true
  export_nsys_report "$rep" cuda_gpu_mem_time_sum "$run_dir/nsys/cuda_gpu_mem_time_sum.csv" false
  export_nsys_report "$rep" cuda_gpu_mem_size_sum "$run_dir/nsys/cuda_gpu_mem_size_sum.csv" false
  export_nsys_report "$rep" nvtx_gpu_proj_sum "$run_dir/nsys/nvtx_gpu_proj_sum.csv" false
  export_nsys_report "$rep" nvtx_gpu_proj_trace "$run_dir/nsys/nvtx_gpu_proj_trace.csv" false
  export_nsys_report "$rep" osrt_sum "$run_dir/nsys/osrt_sum.csv" false
  if [[ "$trace_level" == node ]]; then
    export_nsys_report "$rep" cuda_gpu_kern_sum "$run_dir/nsys/cuda_gpu_kern_sum.csv" true
    export_nsys_report "$rep" cuda_gpu_kern_gb_sum "$run_dir/nsys/cuda_gpu_kern_gb_sum.csv" false
    export_nsys_report "$rep" cuda_gpu_trace "$run_dir/nsys/cuda_gpu_trace.csv" true
    export_nsys_report "$rep" cuda_kern_exec_sum "$run_dir/nsys/cuda_kern_exec_sum.csv" false
    export_nsys_report "$rep" cuda_kern_exec_trace "$run_dir/nsys/cuda_kern_exec_trace.csv" false
  fi
  extract_stage_reports "$sqlite_path" "$run_dir/nsys/stages" "$trace_level"

  "$NSYS" analyze "$rep" > "$run_dir/nsys/analyze.txt" 2> "$run_dir/nsys/analyze.stderr.txt"
  for rule in cuda_api_sync gpu_gaps gpu_time_util; do
    if grep -q -- "$rule" "$CAPABILITIES/nsys-analyze-help.txt"; then
      "$NSYS" analyze --rule "$rule" "$rep" \
        > "$run_dir/nsys/analyze-$rule.txt" 2> "$run_dir/nsys/analyze-$rule.stderr.txt"
    fi
  done
}

run_nsys() {
  local precision=$1
  local scope=$2
  local config_name=$3
  local trace_level=$4
  local pass_kind=$5
  local run_dir="$RUNS/$precision/${scope}_${trace_level}"
  mkdir -p "$run_dir/output" "$run_dir/nsys"
  build_inference_command "$precision" "$config_name" "$pass_kind" true true "$run_dir/output"
  local nsys_args=(
    profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --cpuctxsw=none
    --backtrace=none
    --cuda-graph-trace="$trace_level"
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --wait=primary
    --force-overwrite=true
    --output="$run_dir/nsys/trace"
  )
  if grep -q -- '--cuda-event-trace' "$CAPABILITIES/nsys-profile-help.txt"; then
    nsys_args+=(--cuda-event-trace=false)
  fi
  printf '%q ' "$NSYS" "${nsys_args[@]}" "${INFERENCE_COMMAND[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "$NSYS" "${nsys_args[@]}" "${INFERENCE_COMMAND[@]}" \
    > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  capture_run_paths "$run_dir"
  export_nsys "$run_dir" "$trace_level"
}

transparency_gate() {
  local precision=$1
  local scope=$2
  local traced_kind=$3
  local control="$RUNS/$precision/${scope}_control/profile-path.txt"
  local traced="$RUNS/$precision/${scope}_${traced_kind}/profile-path.txt"
  local output="$RUNS/$precision/${scope}_${traced_kind}/transparency.json"
  "$PYTHON" utils/nsight_agent_profile.py transparency \
    --control "$(<"$control")" \
    --traced "$(<"$traced")" \
    --output "$output"
  require_nonempty "$output"
}

run_precision() {
  local precision=$1
  run_control "$precision" full profile_salvalai
  run_nsys "$precision" full profile_salvalai graph nsys_graph
  transparency_gate "$precision" full graph
  run_control "$precision" smoke profile_salvalai_smoke15
  run_nsys "$precision" smoke profile_salvalai_smoke15 node nsys_node
  transparency_gate "$precision" smoke node
}

# FP32 must complete its parity gates before FP16 begins.
run_precision fp32
run_precision fp16

NCU_PROBE_DIR="$RUN_ROOT/ncu-permission-probe"
mkdir -p "$NCU_PROBE_DIR"
set +e
"$NCU" \
  --target-processes application-only \
  --launch-count 1 \
  --metrics sm__cycles_elapsed.avg \
  --csv \
  --log-file "$NCU_PROBE_DIR/ncu-probe.log" \
  "$PYTHON" -c 'import torch; x=torch.ones(1024, device="cuda"); y=x+x; y.sum(); torch.cuda.synchronize()' \
  > "$NCU_PROBE_DIR/stdout.txt" 2> "$NCU_PROBE_DIR/stderr.txt"
NCU_PROBE_EXIT=$?
set -e
printf '%s\n' "$NCU_PROBE_EXIT" > "$NCU_PROBE_DIR/exit-code.txt"
touch "$NCU_PROBE_DIR/ncu-probe.log"
{
  cat "$NCU_PROBE_DIR/ncu-probe.log"
  cat "$NCU_PROBE_DIR/stdout.txt"
  cat "$NCU_PROBE_DIR/stderr.txt"
} > "$NCU_PROBE_DIR/combined.log"

TARGETED_NCU=()
NCU_FATAL=0
if grep -q 'ERR_NVGPUCTRPERM' "$NCU_PROBE_DIR/combined.log"; then
  echo "ncu_status=permission_denied" | tee "$NCU_PROBE_DIR/status.txt"
elif [[ $NCU_PROBE_EXIT -ne 0 ]]; then
  echo "ncu_status=probe_failed" | tee "$NCU_PROBE_DIR/status.txt"
  NCU_FATAL=1
else
  echo "ncu_status=available" | tee "$NCU_PROBE_DIR/status.txt"
  # FP32 is the authoritative optimization target.  Select exactly one kernel
  # from its fresh node-level trace and collect one bounded NCU profile.
  precision=fp32
  summary_dir="$RUNS/$precision/smoke_node/nsys/stages"
  hot_kernel_json="$RUN_ROOT/runs/$precision/hot-kernel.json"
  "$PYTHON" - "$summary_dir" "$hot_kernel_json" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
candidates = []
for stage in ("timing_generation", "main_generation"):
    path = root / f"{stage}.cuda_kernel_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates.append((float(row["Total Time (ns)"]), stage, row["Name"]))
if not candidates:
    raise SystemExit("no hot-kernel candidates in fresh node trace")
total, stage, name = max(candidates, key=lambda item: (item[0], item[2]))
output.write_text(
    json.dumps(
        {
            "raw_name": name,
            "stage": stage,
            "total_ns": total,
            "ncu_regex": "^" + re.escape(name) + "$",
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY
  hot_regex=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ncu_regex"])' "$hot_kernel_json")
  ncu_dir="$RUNS/$precision/smoke_ncu"
  mkdir -p "$ncu_dir/output" "$ncu_dir/ncu"
  build_inference_command "$precision" profile_salvalai_smoke15 ncu_kernel true false "$ncu_dir/output"
  printf '%q ' "$NCU" "${INFERENCE_COMMAND[@]}" > "$ncu_dir/command.txt"
  printf '\n' >> "$ncu_dir/command.txt"
  "$NCU" \
    --target-processes application-only \
    --graph-profiling node \
    --replay-mode kernel \
    --kernel-name-base demangled \
    --kernel-name "regex:$hot_regex" \
    --launch-count 1 \
    --nvtx \
    --nvtx-include 'mapperatorinator.roi.inference_generation/' \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --force-overwrite \
    --export "$ncu_dir/ncu/hot-kernel" \
    "${INFERENCE_COMMAND[@]}" \
    > "$ncu_dir/stdout.txt" 2> "$ncu_dir/stderr.txt"
  capture_run_paths "$ncu_dir"
  require_nonempty "$ncu_dir/ncu/hot-kernel.ncu-rep"
  "$NCU" --import "$ncu_dir/ncu/hot-kernel.ncu-rep" --page raw --csv \
    > "$ncu_dir/ncu/hot-kernel.raw.csv" 2> "$ncu_dir/ncu/import.stderr.txt"
  require_nonempty "$ncu_dir/ncu/hot-kernel.raw.csv"
  "$PYTHON" utils/nsight_agent_profile.py transparency \
    --control "$(<"$RUNS/$precision/smoke_control/profile-path.txt")" \
    --traced "$(<"$ncu_dir/profile-path.txt")" \
    --output "$ncu_dir/transparency.json"
  TARGETED_NCU+=("$precision:$(basename "$ncu_dir/ncu/hot-kernel.ncu-rep")")
fi

TARGETED_NCU_JSON=$(printf '%s\n' "${TARGETED_NCU[@]:-}" | "$PYTHON" -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')
export NSIGHT_RUN_ROOT="$RUN_ROOT"
export NSIGHT_REPO="$REPO"
export NSIGHT_COMMIT="$COMMIT"
export NSIGHT_BRANCH="$BRANCH"
export NSIGHT_REMOTE_REF="$REMOTE_REF"
export NSIGHT_AUDIO="$AUDIO"
export NSIGHT_NCU_PROBE_EXIT="$NCU_PROBE_EXIT"
export NSIGHT_TARGETED_NCU_JSON="$TARGETED_NCU_JSON"
"$PYTHON" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

from utils import nsight_agent_profile as nsight

root = Path(os.environ["NSIGHT_RUN_ROOT"]).resolve()
runs_root = root / "runs"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def relative(path):
    return str(path.resolve().relative_to(root))

def artifact(path, role, *, stage=None, required=True, allow_empty=False):
    item = {
        "role": role,
        "path": relative(path),
        "required": required,
        "allow_empty": allow_empty,
    }
    if path.exists() and path.is_file():
        item["sha256"] = sha(path)
    if stage is not None:
        item["stage"] = stage
    return item

def profile_stage(profile, label):
    signature = nsight._profile_label_signature(profile, label)
    if signature["status"] != "available" or not signature["self_consistent"]:
        raise SystemExit(f"profile label is unavailable or inconsistent: {label}")
    summary = profile.get("summary", {}).get("generation_by_label", {}).get(label)
    if not isinstance(summary, dict):
        raise SystemExit(f"missing generation summary for {label}")
    graph = nsight._profile_graph_cache_signature(profile, [label])
    return {
        "outer_wall_seconds": summary.get("wall_seconds"),
        "synchronized_model_seconds": summary.get("model_elapsed_seconds"),
        "generated_tokens": signature["generated_tokens"],
        "token_ids_sha256": signature["token_stream_sha256"],
        "stopping_sha256": signature["stopping_sha256"],
        "cache_behavior_sha256": graph["sha256"],
        "graph_attribution": {
            "method": "none",
            "verified_on_replay": False,
            "kernel_regions": {},
        },
    }

def pipeline_stage_wall_ns(profile):
    summary = profile.get("summary", {})
    raw = summary.get("stage_wall_seconds")
    if not isinstance(raw, dict):
        raise SystemExit("profile summary is missing stage_wall_seconds")
    result = {}
    for name, value in raw.items():
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise SystemExit(f"invalid pipeline stage duration for {name}: {value}")
        result[str(name)] = int(round(seconds * 1_000_000_000))
    return result

def add_run(result, precision, directory_name, scope, pass_kind, paired=None):
    directory = runs_root / precision / directory_name
    profile_path = Path((directory / "profile-path.txt").read_text().strip())
    result_path = Path((directory / "result-path.txt").read_text().strip())
    profile = nsight._load_inference_profile(profile_path)
    metadata = profile["metadata"]
    run_id = f"{precision}_{directory_name}"
    config = "profile_salvalai" if scope == "full" else "profile_salvalai_smoke15"
    artifacts = [
        artifact(profile_path, "inference_profile"),
        artifact(result_path, "result_osu"),
        artifact(directory / "stdout.txt", "process_stdout"),
        artifact(directory / "stderr.txt", "process_stderr", allow_empty=True),
    ]
    if pass_kind in {"nsys_graph", "nsys_node"}:
        artifacts.extend(
            [
                artifact(directory / "nsys/trace.nsys-rep", "nsys_rep"),
                artifact(directory / "nsys/trace.sqlite", "sqlite"),
                artifact(directory / "nsys/cuda_api_sum.csv", "nsys_cuda_api_sum"),
                artifact(directory / "nsys/nvtx_sum.csv", "nsys_nvtx_sum"),
                artifact(directory / "nsys/analyze.txt", "nsys_expert_analysis", allow_empty=True),
            ]
        )
    stages = {
        "timing_generation": profile_stage(profile, "timing_context"),
        "main_generation": profile_stage(profile, "main_generation"),
    }
    if pass_kind in {"nsys_graph", "nsys_node"}:
        extraction_path = directory / "nsys/stages/extraction.json"
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        artifacts.append(artifact(extraction_path, "stage_sqlite_extraction"))
        for extracted in extraction["artifacts"]:
            extracted_path = Path(extracted["path"])
            item = artifact(
                extracted_path,
                extracted["role"],
                stage=extracted["stage"],
            )
            item["row_count"] = int(extracted["row_count"])
            item["allow_empty_rows"] = bool(
                extracted.get("allow_empty_rows", False)
            )
            artifacts.append(item)
        for stage, extracted_stage in extraction["stages"].items():
            stages[stage]["trace_window_start_ns"] = int(
                extracted_stage["start_ns"]
            )
            stages[stage]["trace_window_end_ns"] = int(extracted_stage["end_ns"])
            stages[stage]["diagnostic_counts"] = extracted_stage["counts"]
    workload = {
        "commit": os.environ["NSIGHT_COMMIT"],
        "audio_sha256": sha(Path(os.environ["NSIGHT_AUDIO"])),
        "config_name": config,
        "precision": precision,
        "seed": 12345,
        "engine": "optimized",
        "attn_implementation": "sdpa",
    }
    result.append(
        {
            "run_id": run_id,
            "run_group_id": f"{precision}:accepted:{scope}",
            "precision": precision,
            "engine_variant": "accepted",
            "engine_preset_version": metadata.get("optimized_effective_config_version"),
            "pass_kind": pass_kind,
            "authoritative_performance": pass_kind == "untraced_control",
            "paired_control_run_id": f"{precision}_{paired}" if paired else None,
            "workload_contract": workload,
            "output_sha256": sha(result_path),
            "output_size_bytes": result_path.stat().st_size,
            "output_structure": nsight.summarize_osu_structure(result_path),
            "pipeline_stage_wall_ns": pipeline_stage_wall_ns(profile),
            "identity": {
                "config_name": config,
                "source_profile": relative(profile_path),
                "nsys_capture_scope": "timing_and_main_generation" if pass_kind.startswith("nsys_") else None,
                "stage_kernel_extraction": "exact NVTX wall containment" if pass_kind == "nsys_node" else None,
            },
            "stages": stages,
            "artifacts": artifacts,
        }
    )

runs = []
for precision in ("fp32", "fp16"):
    add_run(runs, precision, "full_control", "full", "untraced_control")
    add_run(runs, precision, "full_graph", "full", "nsys_graph", paired="full_control")
    add_run(runs, precision, "smoke_control", "smoke", "untraced_control")
    add_run(runs, precision, "smoke_node", "smoke", "nsys_node", paired="smoke_control")
    ncu_directory = runs_root / precision / "smoke_ncu"
    if ncu_directory.exists():
        add_run(runs, precision, "smoke_ncu", "smoke", "ncu_kernel", paired="smoke_control")
        ncu_run = runs[-1]
        ncu_run["artifacts"].extend(
            [
                artifact(ncu_directory / "ncu/hot-kernel.ncu-rep", "ncu_rep"),
                artifact(ncu_directory / "ncu/hot-kernel.raw.csv", "ncu_raw_csv"),
            ]
        )

targeted = json.loads(os.environ["NSIGHT_TARGETED_NCU_JSON"])
manifest = {
    "schema_version": nsight.MANIFEST_SCHEMA_VERSION,
    "provenance": {
        "repo": os.environ["NSIGHT_REPO"],
        "commit": os.environ["NSIGHT_COMMIT"],
        "branch": os.environ["NSIGHT_BRANCH"],
        "remote_ref": os.environ["NSIGHT_REMOTE_REF"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": os.environ.get("MAPPERATORINATOR_EXPECTED_GPU", "2080 Ti"),
        "precision_order": ["fp32", "fp16"],
    },
    "tool_capabilities": {
        "nsys_version_path": "capabilities/nsys-version.txt",
        "ncu_version_path": "capabilities/ncu-version.txt",
        "dcgmi_version_path": "capabilities/dcgmi-version.txt",
    },
    "runs": runs,
    "comparisons": [
        {
            "comparison_type": "cross_precision_divergence",
            "left_run_id": "fp32_full_control",
            "right_run_id": "fp16_full_control",
        }
    ],
    "ncu_probe": {
        "attempted": True,
        "attempt_count": 1,
        "exit_code": int(os.environ["NSIGHT_NCU_PROBE_EXIT"]),
        "log_path": "ncu-permission-probe/combined.log",
        "targeted_collections": targeted,
    },
}
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"$PYTHON" utils/nsight_agent_profile.py analyze \
  --manifest "$RUN_ROOT/manifest.json" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"
require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"

find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
require_nonempty "$RUN_ROOT/sha256sums.txt"
cat "$RUN_ROOT/analysis.txt"
[[ $NCU_FATAL -eq 0 ]] || die "ncu permission probe failed for a reason other than ERR_NVGPUCTRPERM"
