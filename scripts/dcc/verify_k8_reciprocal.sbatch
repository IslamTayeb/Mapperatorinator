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
FULL_FIXED_WORK_SAVING_SECONDS=${FULL_FIXED_WORK_SAVING_SECONDS:-5}
RUN_ROOT="$WORK/runs/k8-reciprocal-${SLURM_JOB_ID:-manual}"

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
"$ENV/bin/python" - "$MINIMUM_SMOKE_SPEEDUP_PCT" "$FULL_FIXED_WORK_SAVING_SECONDS" <<'PY'
import sys

float(sys.argv[1])
if float(sys.argv[2]) < 0:
    raise SystemExit("FULL_FIXED_WORK_SAVING_SECONDS must be non-negative")
import cuda.bindings.runtime  # noqa: F401
PY

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
  echo "full_fixed_work_saving_seconds=$FULL_FIXED_WORK_SAVING_SECONDS"
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
  local runner=$2
  local order=$3
  local config=$4
  local output="$RUN_ROOT/$scope/$order"
  local compiler_cache="$RUN_ROOT/compiler-cache/$scope/$order"
  [[ ! -e "$compiler_cache" ]] || fail "compiler cache already exists: $compiler_cache"
  mkdir -p "$output" \
    "$compiler_cache/torch_inductor" "$compiler_cache/triton" "$compiler_cache/cuda"
  {
    echo "scope=$scope"
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
  printf '%s\n' "$profile" > "$RUN_ROOT/$scope/$order.profile-path.txt"
}

analyze_scope() {
  local scope=$1
  local config=$2
  local scope_root="$RUN_ROOT/$scope"
  mkdir -p "$scope_root"
  run_profile "$scope" inference.py baseline_first "$config"
  run_profile "$scope" utils/run_k8_candidate.py candidate_second "$config"
  run_profile "$scope" utils/run_k8_candidate.py candidate_first "$config"
  run_profile "$scope" inference.py baseline_second "$config"

  local baseline_first candidate_second candidate_first baseline_second
  baseline_first=$(<"$scope_root/baseline_first.profile-path.txt")
  candidate_second=$(<"$scope_root/candidate_second.profile-path.txt")
  candidate_first=$(<"$scope_root/candidate_first.profile-path.txt")
  baseline_second=$(<"$scope_root/baseline_second.profile-path.txt")

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
      --full-fixed-work-saving-seconds "$FULL_FIXED_WORK_SAVING_SECONDS" \
      --json-output "$scope_root/gate.json" \
      --text-output "$scope_root/gate.txt"
  )
}

analyze_scope smoke profile_salvalai_smoke15
analyze_scope full profile_salvalai

echo "k8_reciprocal_gate=PASS"
echo "artifacts=$RUN_ROOT"
