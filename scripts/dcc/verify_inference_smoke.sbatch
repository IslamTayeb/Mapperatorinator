#!/usr/bin/env bash
#SBATCH --job-name=inference-smoke
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
BASELINE_REPO=${BASELINE_REPO:?Set BASELINE_REPO to a clean pushed worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:-$BASELINE_REPO}
BASELINE_COMMIT=${BASELINE_COMMIT:?Set BASELINE_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:-$BASELINE_COMMIT}
BASELINE_BRANCH=${BASELINE_BRANCH:-DETACHED}
CANDIDATE_BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH explicitly}
CANDIDATE_REMOTE=${CANDIDATE_REMOTE:-origin}
BASELINE_ENGINE=${BASELINE_ENGINE:-v32}
CANDIDATE_ENGINE=${CANDIDATE_ENGINE:-optimized}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai_smoke15}
PRECISION=${PRECISION:-fp32}
REGRESSION_TOLERANCE_PCT=${REGRESSION_TOLERANCE_PCT:-5}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-smoke}
RUN_ROOT="$WORK/runs/inference-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "missing environment Python at $ENV/bin/python" >&2
  exit 2
fi

for repo in "$BASELINE_REPO" "$CANDIDATE_REPO"; do
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "worktree is dirty at $repo" >&2
    git -C "$repo" status --short >&2
    exit 2
  fi
done
if [[ "$(git -C "$BASELINE_REPO" rev-parse HEAD)" != "$BASELINE_COMMIT" ]]; then
  echo "baseline worktree differs from BASELINE_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" != "$CANDIDATE_COMMIT" ]]; then
  echo "candidate worktree differs from CANDIDATE_COMMIT" >&2
  exit 2
fi
if [[ "$BASELINE_BRANCH" == DETACHED ]]; then
  if [[ -n "$(git -C "$BASELINE_REPO" branch --show-current)" ]]; then
    echo "baseline worktree must be detached" >&2
    exit 2
  fi
elif [[ "$(git -C "$BASELINE_REPO" branch --show-current)" != "$BASELINE_BRANCH" ]]; then
  echo "baseline worktree is not on BASELINE_BRANCH=$BASELINE_BRANCH" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" branch --show-current)" != "$CANDIDATE_BRANCH" ]]; then
  echo "candidate worktree is not on CANDIDATE_BRANCH=$CANDIDATE_BRANCH" >&2
  exit 2
fi
CANDIDATE_REMOTE_REF="refs/remotes/$CANDIDATE_REMOTE/$CANDIDATE_BRANCH"
if ! git -C "$CANDIDATE_REPO" show-ref --verify --quiet "$CANDIDATE_REMOTE_REF"; then
  echo "missing fetched candidate remote ref $CANDIDATE_REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" rev-parse "$CANDIDATE_REMOTE_REF")" != "$CANDIDATE_COMMIT" ]]; then
  echo "candidate remote ref differs from CANDIDATE_COMMIT" >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing input audio at $AUDIO" >&2
  exit 2
fi
if [[ "$BASELINE_ENGINE" != v32 && "$BASELINE_ENGINE" != optimized ]]; then
  echo "BASELINE_ENGINE must be v32 or optimized" >&2
  exit 2
fi
if [[ "$CANDIDATE_ENGINE" != v32 && "$CANDIDATE_ENGINE" != optimized ]]; then
  echo "CANDIDATE_ENGINE must be v32 or optimized" >&2
  exit 2
fi
if [[ "$PRECISION" != fp32 && "$PRECISION" != fp16 ]]; then
  echo "PRECISION must be fp32 or fp16" >&2
  exit 2
fi
"$ENV/bin/python" - "$REGRESSION_TOLERANCE_PCT" <<'PY'
import sys

value = float(sys.argv[1])
if value < 0:
    raise SystemExit("REGRESSION_TOLERANCE_PCT must be non-negative")
PY

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "baseline_commit=$BASELINE_COMMIT"
echo "candidate_commit=$CANDIDATE_COMMIT"
echo "baseline_branch=$BASELINE_BRANCH"
echo "candidate_branch=$CANDIDATE_BRANCH"
echo "candidate_remote_ref=$CANDIDATE_REMOTE_REF"
echo "baseline_engine=$BASELINE_ENGINE"
echo "candidate_engine=$CANDIDATE_ENGINE"
echo "profile_config=$PROFILE_CONFIG"
echo "precision=$PRECISION"
echo "regression_tolerance_pct=$REGRESSION_TOLERANCE_PCT"
echo "run_root=$RUN_ROOT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != *"2080 Ti"* ]]; then
  echo "expected RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi
nvidia-smi -L
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
  precision="$PRECISION"
  attn_implementation=sdpa
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
)

run_profile() {
  local repo=$1
  local engine=$2
  local label=$3
  local output="$RUN_ROOT/$label"
  local compiler_cache="$RUN_ROOT/compiler-cache/$label"
  local legacy_reset=()
  if [[ "$engine" == optimized ]] && grep -q 'inference_generation_compile:' "$repo/config.py"; then
    legacy_reset+=(inference_generation_compile=false)
  fi
  if [[ -e "$compiler_cache" ]]; then
    echo "compiler cache already exists for $label at $compiler_cache" >&2
    exit 2
  fi
  mkdir -p "$output" \
    "$compiler_cache/torch_inductor" "$compiler_cache/triton" "$compiler_cache/cuda"
  (
    export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"
    export TRITON_CACHE_DIR="$compiler_cache/triton"
    export CUDA_CACHE_PATH="$compiler_cache/cuda"
    cd "$repo"
    python inference.py --config-name "$PROFILE_CONFIG" \
      output_path="$output" \
      inference_engine="$engine" \
      "${COMMON_OVERRIDES[@]}" \
      "${legacy_reset[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing profile for $label" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$label.profile-path.txt"
}

run_profile "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_first
run_profile "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_second
run_profile "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_first
run_profile "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_second

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

(
  cd "$CANDIDATE_REPO"
  python utils/summarize_inference_profile.py \
    --reciprocal "$BASELINE_FIRST" "$CANDIDATE_SECOND" "$CANDIDATE_FIRST" "$BASELINE_SECOND" \
    --labels main_generation,timing_context \
    --regression-tolerance-pct "$REGRESSION_TOLERANCE_PCT" \
    --json-output "$RUN_ROOT/reciprocal-compare.json"
)

python - "$RUN_ROOT" "$BASELINE_ENGINE" "$CANDIDATE_ENGINE" "$PRECISION" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "baseline_first": sys.argv[2],
    "baseline_second": sys.argv[2],
    "candidate_first": sys.argv[3],
    "candidate_second": sys.argv[3],
}
precision = sys.argv[4]
for label, engine in expected.items():
    profile_path = Path((root / f"{label}.profile-path.txt").read_text().strip())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("metadata", {}).get("inference_engine") != engine:
        raise SystemExit(f"{label}: inference engine metadata mismatch")
    if profile.get("metadata", {}).get("precision") != precision:
        raise SystemExit(f"{label}: precision metadata mismatch")
print("inference_reciprocal_smoke_gate=PASS")
PY

find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*-compare.json' \) -print0 \
  | sort -z | xargs -0 sha256sum
