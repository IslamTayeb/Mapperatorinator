#!/usr/bin/env bash
#SBATCH --job-name=arena-runtime-nsight
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

# The selected K4/K1 runtime owns its cache capacity and natural stopping.
# Profiling therefore compares identical fixed-seed natural selected runs
# instead of rewriting MaxLengthCriteria through the generic fixed-work gate.
export RUN_LABEL=selected-arena-runtime-nsight
export MAPPERATORINATOR_SELECTED_RUNNER=utils/run_k4_shared_rope_fp16_cross_shared_arena_transition_timing.py
source "${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean profiler worktree}/scripts/dcc/profile_nsight_inference.sbatch"
exit 0

: <<'LEGACY_FIXED_WORK_HARNESS'

die() {
  echo "runtime-nsight-fixed7149: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
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
RUNTIME_SPEC_SOURCE=${MAPPERATORINATOR_RUNTIME_SPEC:-$REPO/scripts/dcc/runtime_specs/k1_int8_fp16_cross.json}
RUN_ROOT="$WORK/runs/selected-runtime-nsight-$SLURM_JOB_ID"
RUNS="$RUN_ROOT/runs"
CAPABILITIES="$RUN_ROOT/capabilities"
EXPECTED_MAIN_STEPS=7149

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing profiler worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ -s "$RUNTIME_SPEC_SOURCE" ]] || die "missing runtime spec at $RUNTIME_SPEC_SOURCE"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "profiler worktree is dirty"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD does not match commit"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "worktree branch changed"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "remote ref does not match commit"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions/runtime-nsight-$COMMIT" "$RUNS" "$CAPABILITIES"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/runtime-nsight-$COMMIT"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

for tool in ffmpeg ffprobe nvidia-smi nsys; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
NSYS=$(command -v nsys)
"$NSYS" --version > "$CAPABILITIES/nsys-version.txt" 2>&1
"$NSYS" profile --help > "$CAPABILITIES/nsys-profile-help.txt" 2>&1
for required in --capture-range --capture-range-end --cuda-graph-trace --trace; do
  grep -q -- "$required" "$CAPABILITIES/nsys-profile-help.txt" || die "nsys lacks $required"
done

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected one visible GPU"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-common" ]] || die "expected gpu-common partition"

cd "$REPO"
cp "$RUNTIME_SPEC_SOURCE" "$RUN_ROOT/runtime-spec.json"
RUNTIME_SPEC="$RUN_ROOT/runtime-spec.json"
"$PYTHON" - "$RUNTIME_SPEC" <<'PY'
import json
import sys
from pathlib import Path
from utils.final_confirmation_runtime import RUNTIME_SPEC_SCHEMA_VERSION, _import_factory

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema_version") != RUNTIME_SPEC_SCHEMA_VERSION:
    raise SystemExit("unsupported runtime spec schema")
if not isinstance(payload.get("name"), str) or not payload["name"].strip():
    raise SystemExit("runtime spec name must be non-empty")
if not isinstance(payload.get("kwargs"), dict):
    raise SystemExit("runtime spec kwargs must be an object")
_import_factory(payload.get("factory"))
PY

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
  echo "runtime_spec_sha256=$(sha256sum "$RUNTIME_SPEC" | awk '{print $1}')"
  echo "expected_main_steps=$EXPECTED_MAIN_STEPS"
  echo "run_root=$RUN_ROOT"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
cleanup() { kill "$SMI_PID" 2>/dev/null || true; }
trap cleanup EXIT

COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  inference_engine=optimized
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
)

CONFIRMATION_COMMAND=()
build_confirmation_command() {
  local run_id=$1
  local mode=$2
  local pass_kind=$3
  local detail_ranges=$4
  local cuda_capture=$5
  local candidate=$6
  local run_dir="$RUNS/$run_id"
  mkdir -p "$run_dir/output"
  CONFIRMATION_COMMAND=(
    "$PYTHON" utils/run_final_confirmation.py
    --mode "$mode"
    --manifest-path "$RUN_ROOT/fixed-work.json"
    --evidence-path "$run_dir/evidence.json"
    --expected-main-steps "$EXPECTED_MAIN_STEPS"
  )
  if [[ "$candidate" == true ]]; then
    CONFIRMATION_COMMAND+=(
      --candidate
      --runtime-spec "$RUNTIME_SPEC"
      --initialization-path "$run_dir/runtime-evidence.json"
    )
  fi
  CONFIRMATION_COMMAND+=(
    "${COMMON_OVERRIDES[@]}"
    output_path="$run_dir/output"
    profile_detail_ranges="$detail_ranges"
    profile_cuda_capture="$cuda_capture"
    profile_pass_kind="$pass_kind"
  )
}

run_confirmation() {
  local run_id=$1 mode=$2 pass_kind=$3 detail_ranges=$4 cuda_capture=$5 candidate=$6
  local run_dir="$RUNS/$run_id"
  build_confirmation_command "$run_id" "$mode" "$pass_kind" "$detail_ranges" "$cuda_capture" "$candidate"
  printf '%q ' "${CONFIRMATION_COMMAND[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "${CONFIRMATION_COMMAND[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  require_nonempty "$run_dir/evidence.json"
}

export_nsys_report() {
  local rep=$1 report=$2 output=$3 required=$4
  set +e
  "$NSYS" stats --report "$report" --format csv "$rep" > "$output" 2> "$output.stderr.txt"
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    [[ "$required" == false ]] && return 0
    die "nsys report $report failed with exit code $exit_code"
  fi
  [[ "$required" == false ]] || require_nonempty "$output"
}

export_nsys() {
  local run_id=$1 trace_level=$2
  local nsys_dir="$RUNS/$run_id/nsys"
  local rep="$nsys_dir/trace.nsys-rep"
  local sqlite="$nsys_dir/trace.sqlite"
  require_nonempty "$rep"
  "$NSYS" export --type sqlite --force-overwrite=true --output "$sqlite" "$rep" \
    > "$nsys_dir/export.stdout.txt" 2> "$nsys_dir/export.stderr.txt"
  require_nonempty "$sqlite"
  export_nsys_report "$rep" cuda_api_sum "$nsys_dir/cuda_api_sum.csv" true
  export_nsys_report "$rep" nvtx_sum "$nsys_dir/nvtx_sum.csv" true
  export_nsys_report "$rep" cuda_gpu_mem_time_sum "$nsys_dir/cuda_gpu_mem_time_sum.csv" false
  export_nsys_report "$rep" cuda_gpu_mem_size_sum "$nsys_dir/cuda_gpu_mem_size_sum.csv" false
  if [[ "$trace_level" == node ]]; then
    export_nsys_report "$rep" cuda_gpu_kern_sum "$nsys_dir/cuda_gpu_kern_sum.csv" true
    export_nsys_report "$rep" cuda_gpu_trace "$nsys_dir/cuda_gpu_trace.csv" true
  fi
  local extract_args=(
    utils/nsight_agent_profile.py extract-sqlite
    --sqlite "$sqlite"
    --output-dir "$nsys_dir/stages"
  )
  [[ "$trace_level" == graph ]] && extract_args+=(--graph-level)
  "$PYTHON" "${extract_args[@]}"
  require_nonempty "$nsys_dir/stages/extraction.json"
}

run_nsys() {
  local run_id=$1 trace_level=$2 pass_kind=$3
  local run_dir="$RUNS/$run_id"
  mkdir -p "$run_dir/output" "$run_dir/nsys"
  build_confirmation_command "$run_id" replay-fixed-work "$pass_kind" true true true
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
  printf '%q ' "$NSYS" "${nsys_args[@]}" "${CONFIRMATION_COMMAND[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "$NSYS" "${nsys_args[@]}" "${CONFIRMATION_COMMAND[@]}" \
    > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  require_nonempty "$run_dir/evidence.json"
  export_nsys "$run_id" "$trace_level"
}

# The accepted run creates the sole fixed-work manifest. Every selected arm
# replays that exact 7,149-main-step schedule through a runtime spec. This is
# the accepted natural work count on the pinned valid historical branch.
run_confirmation accepted_record record-fixed-work untraced_control false false false
require_nonempty "$RUN_ROOT/fixed-work.json"
run_confirmation selected_control replay-fixed-work untraced_control false false true

run_nsys selected_graph graph nsys_graph
run_nsys selected_node node nsys_node

profile_path() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile_path"])' "$RUNS/$1/evidence.json"
}
CONTROL_PROFILE=$(profile_path selected_control)
for pair in "selected_graph:graph" "selected_node:node"; do
  run_id=${pair%%:*}
  label=${pair##*:}
  "$PYTHON" utils/nsight_agent_profile.py transparency \
    --control "$CONTROL_PROFILE" \
    --traced "$(profile_path "$run_id")" \
    --output "$RUN_ROOT/transparency-$label.json"
  require_nonempty "$RUN_ROOT/transparency-$label.json"
done

"$PYTHON" utils/validate_selected_runtime_nsight.py \
  --selected-control-evidence "$RUNS/selected_control/evidence.json" \
  --selected-graph-evidence "$RUNS/selected_graph/evidence.json" \
  --selected-node-evidence "$RUNS/selected_node/evidence.json" \
  --output "$RUN_ROOT/topology-contract.json"
require_nonempty "$RUN_ROOT/topology-contract.json"

"$PYTHON" utils/build_runtime_nsight_manifest.py \
  --run-root "$RUN_ROOT" \
  --commit "$COMMIT" \
  --branch "$BRANCH" \
  --remote-ref "$REMOTE_REF" \
  --audio "$AUDIO" \
  --runtime-spec "$RUNTIME_SPEC" \
  --fixed-manifest "$RUN_ROOT/fixed-work.json" \
  --accepted-evidence "$RUNS/accepted_record/evidence.json" \
  --selected-control-evidence "$RUNS/selected_control/evidence.json" \
  --selected-graph-evidence "$RUNS/selected_graph/evidence.json" \
  --selected-node-evidence "$RUNS/selected_node/evidence.json" \
  --output "$RUN_ROOT/nsight-manifest.json"

"$PYTHON" utils/nsight_agent_profile.py analyze \
  --manifest "$RUN_ROOT/nsight-manifest.json" \
  --output "$RUN_ROOT/nsight-analysis.json" \
  --text-output "$RUN_ROOT/nsight-analysis.txt"

"$PYTHON" utils/summarize_runtime_reprofile.py \
  --accepted-evidence "$RUNS/accepted_record/evidence.json" \
  --selected-control-evidence "$RUNS/selected_control/evidence.json" \
  --selected-graph-evidence "$RUNS/selected_graph/evidence.json" \
  --selected-node-evidence "$RUNS/selected_node/evidence.json" \
  --topology-contract "$RUN_ROOT/topology-contract.json" \
  --graph-transparency "$RUN_ROOT/transparency-graph.json" \
  --node-transparency "$RUN_ROOT/transparency-node.json" \
  --nsight-analysis "$RUN_ROOT/nsight-analysis.json" \
  --csv-dir "$RUN_ROOT/csv" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

for required in \
  "$RUN_ROOT/analysis.json" \
  "$RUN_ROOT/analysis.txt" \
  "$RUN_ROOT/topology-contract.json" \
  "$RUN_ROOT/csv/stage_budget.csv" \
  "$RUN_ROOT/csv/request_pipeline.csv" \
  "$RUN_ROOT/csv/kernel_families.csv" \
  "$RUN_ROOT/csv/top_kernels.csv" \
  "$RUN_ROOT/csv/copy_sync_launch_gaps.csv"; do
  require_nonempty "$required"
done

find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
require_nonempty "$RUN_ROOT/sha256sums.txt"
cat "$RUN_ROOT/analysis.txt"
LEGACY_FIXED_WORK_HARNESS
