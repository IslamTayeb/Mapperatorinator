#!/usr/bin/env bash
#SBATCH --job-name=k8-reciprocal
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=03:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean DCC worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the pushed branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
MINIMUM_SMOKE_SPEEDUP_PCT=${MINIMUM_SMOKE_SPEEDUP_PCT:-0}
MINIMUM_SMOKE_COMPLETE_WALL_SPEEDUP_PCT=${MINIMUM_SMOKE_COMPLETE_WALL_SPEEDUP_PCT:-0}
FULL_FIXED_WORK_SAVING_SECONDS=${FULL_FIXED_WORK_SAVING_SECONDS:-5}
CANDIDATE_BLOCK_SIZE=${CANDIDATE_BLOCK_SIZE:-8}
CANDIDATE_RUNNER=${CANDIDATE_RUNNER:-utils/run_k8_candidate.py}
RUN_COUNTER_CONTROL=${RUN_COUNTER_CONTROL:-false}
RUN_FULL=${RUN_FULL:-true}
RUN_TAG=${RUN_TAG:-k8-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_TAG-${SLURM_JOB_ID:-manual}"

fail() {
  echo "$*" >&2
  exit 2
}

[[ -x "$ENV/bin/python" ]] || fail "missing environment Python at $ENV/bin/python"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || fail "missing worktree at $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  fail "K8 worktree is dirty"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || fail "worktree differs from COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || fail "worktree differs from BRANCH"
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || fail "missing fetched remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || fail "remote ref differs from COMMIT"
[[ -f "$AUDIO" ]] || fail "missing input audio at $AUDIO"

export PATH="$ENV/bin:$PATH"
for tool in ffmpeg ffprobe nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || fail "missing required executable: $tool"
done
"$ENV/bin/python" - \
  "$MINIMUM_SMOKE_SPEEDUP_PCT" \
  "$FULL_FIXED_WORK_SAVING_SECONDS" \
  "$MINIMUM_SMOKE_COMPLETE_WALL_SPEEDUP_PCT" \
  "$CANDIDATE_BLOCK_SIZE" <<'PY'
import sys

float(sys.argv[1])
if float(sys.argv[2]) < 0:
    raise SystemExit("FULL_FIXED_WORK_SAVING_SECONDS must be non-negative")
float(sys.argv[3])
if int(sys.argv[4]) not in (1, 4, 8):
    raise SystemExit("CANDIDATE_BLOCK_SIZE must be 1, 4, or 8")
import cuda.bindings.runtime  # noqa: F401
PY
[[ "$RUN_COUNTER_CONTROL" == true || "$RUN_COUNTER_CONTROL" == false ]] \
  || fail "RUN_COUNTER_CONTROL must be true or false"
[[ "$RUN_FULL" == true || "$RUN_FULL" == false ]] || fail "RUN_FULL must be true or false"
[[ -f "$REPO/$CANDIDATE_RUNNER" ]] || fail "missing candidate runner $CANDIDATE_RUNNER"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/torch_extensions" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$GPU_COUNT" -eq 1 ]] || fail "expected one allocated GPU, got $GPU_COUNT"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "$GPU_NAME" == *"2080 Ti"* ]] || fail "expected RTX 2080 Ti, got $GPU_NAME"

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "repo=$REPO"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "audio=$AUDIO"
  echo "gpu=$GPU_NAME"
  echo "python=$ENV/bin/python"
  echo "ffmpeg=$(command -v ffmpeg)"
  echo "ffprobe=$(command -v ffprobe)"
  echo "minimum_smoke_speedup_pct=$MINIMUM_SMOKE_SPEEDUP_PCT"
  echo "minimum_smoke_complete_wall_speedup_pct=$MINIMUM_SMOKE_COMPLETE_WALL_SPEEDUP_PCT"
  echo "full_fixed_work_saving_seconds=$FULL_FIXED_WORK_SAVING_SECONDS"
  echo "candidate_block_size=$CANDIDATE_BLOCK_SIZE"
  echo "candidate_runner=$CANDIDATE_RUNNER"
  echo "run_counter_control=$RUN_COUNTER_CONTROL"
  echo "run_full=$RUN_FULL"
  echo "run_root=$RUN_ROOT"
  "$ENV/bin/python" - <<'PY'
import torch
import transformers
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
PY
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$RUN_ROOT/preflight.txt"

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

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
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

run_profile() {
  local scope=$1
  local comparison=$2
  local runner=$3
  local order=$4
  local config=$5
  local output="$RUN_ROOT/$scope/$comparison/$order"
  local compiler_cache="$RUN_ROOT/compiler-cache/$scope/$comparison/$order"
  [[ ! -e "$compiler_cache" ]] || fail "compiler cache already exists: $compiler_cache"
  mkdir -p "$output" \
    "$compiler_cache/torch_inductor" "$compiler_cache/triton" "$compiler_cache/cuda"
  {
    echo "scope=$scope"
    echo "comparison=$comparison"
    echo "runner=$runner"
    echo "order=$order"
    echo "config=$config"
  } > "$output/command.txt"
  (
    export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"
    export TRITON_CACHE_DIR="$compiler_cache/triton"
    export CUDA_CACHE_PATH="$compiler_cache/cuda"
    cd "$REPO"
    "$ENV/bin/python" "$runner" --config-name "$config" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  [[ -n "$profile" && -s "$profile" ]] || fail "missing profile for $scope/$order"
  printf '%s\n' "$profile" > "$RUN_ROOT/$scope/$comparison/$order.profile-path.txt"
}

analyze_comparison() {
  local scope=$1
  local config=$2
  local comparison=$3
  local baseline_runner=$4
  local baseline_block_size=$5
  local scope_root="$RUN_ROOT/$scope/$comparison"
  mkdir -p "$scope_root"
  run_profile "$scope" "$comparison" "$baseline_runner" baseline_first "$config"
  run_profile "$scope" "$comparison" "$CANDIDATE_RUNNER" candidate_second "$config"
  run_profile "$scope" "$comparison" "$CANDIDATE_RUNNER" candidate_first "$config"
  run_profile "$scope" "$comparison" "$baseline_runner" baseline_second "$config"

  local baseline_first candidate_second candidate_first baseline_second
  baseline_first=$(<"$scope_root/baseline_first.profile-path.txt")
  candidate_second=$(<"$scope_root/candidate_second.profile-path.txt")
  candidate_first=$(<"$scope_root/candidate_first.profile-path.txt")
  baseline_second=$(<"$scope_root/baseline_second.profile-path.txt")

  local baseline_args=()
  if [[ -n "$baseline_block_size" ]]; then
    baseline_args=(--baseline-block-size "$baseline_block_size")
  fi
  set +e
  (
    cd "$REPO"
    "$ENV/bin/python" utils/summarize_inference_profile.py \
      --reciprocal "$baseline_first" "$candidate_second" "$candidate_first" "$baseline_second" \
      --labels timing_context,main_generation \
      --regression-tolerance-pct 5 \
      --json-output "$scope_root/strict-reciprocal.json"
  )
  local strict_status=$?
  set -e
  printf '%s\n' "$strict_status" > "$scope_root/strict-reciprocal.exit-code.txt"

  (
    cd "$REPO"
    "$ENV/bin/python" utils/validate_k8_reciprocal.py \
      --scope "$scope" \
      --baseline-first "$baseline_first" \
      --candidate-second "$candidate_second" \
      --candidate-first "$candidate_first" \
      --baseline-second "$baseline_second" \
      --minimum-smoke-speedup-pct "$MINIMUM_SMOKE_SPEEDUP_PCT" \
      --minimum-smoke-complete-wall-speedup-pct "$MINIMUM_SMOKE_COMPLETE_WALL_SPEEDUP_PCT" \
      --full-fixed-work-saving-seconds "$FULL_FIXED_WORK_SAVING_SECONDS" \
      --candidate-block-size "$CANDIDATE_BLOCK_SIZE" \
      "${baseline_args[@]}" \
      --json-output "$scope_root/gate.json" \
      --text-output "$scope_root/gate.txt"
  )
  local gate_status=$?
  set -e
  printf '%s\n' "$gate_status" > "$scope_root/gate.exit-code.txt"
  return "$gate_status"
}

natural_smoke_status=0
analyze_comparison smoke profile_salvalai_smoke15 natural inference.py "" \
  || natural_smoke_status=$?
counter_smoke_status=0
if [[ "$RUN_COUNTER_CONTROL" == true ]]; then
  analyze_comparison \
    smoke \
    profile_salvalai_smoke15 \
    counter-fixed \
    utils/run_k1_counter_control.py \
    1 \
    || counter_smoke_status=$?
fi
full_status=0
if [[ "$RUN_FULL" == true && "$natural_smoke_status" -eq 0 ]]; then
  analyze_comparison full profile_salvalai natural inference.py "" || full_status=$?
fi

if [[ "$natural_smoke_status" -ne 0 || "$counter_smoke_status" -ne 0 || "$full_status" -ne 0 ]]; then
  echo "block_reciprocal_gate=FAIL" >&2
  echo "natural_smoke_status=$natural_smoke_status" >&2
  echo "counter_smoke_status=$counter_smoke_status" >&2
  echo "full_status=$full_status" >&2
  exit 1
fi
echo "block_reciprocal_gate=PASS"
echo "artifacts=$RUN_ROOT"
