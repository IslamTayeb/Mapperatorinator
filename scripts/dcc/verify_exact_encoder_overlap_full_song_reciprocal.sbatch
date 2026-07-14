#!/usr/bin/env bash
#SBATCH --job-name=k1-encoder-overlap
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean overlap worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-k1-exact-encoder-overlap-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/shared-overlap-$COMMIT"

if [[ "$PROFILE_CONFIG" != profile_salvalai ]]; then
  echo "exact overlap gate requires profile_salvalai" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing overlap worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "overlap worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "overlap worktree commit mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "overlap worktree branch mismatch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote ref $REMOTE_REF does not match COMMIT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$AUDIO" ]]; then
  echo "missing Python or SALVALAI audio" >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing required audio tool: $tool" >&2
    exit 2
  }
done
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$TORCH_EXTENSIONS_DIR" \
  "$RUN_ROOT/preload" "$RUN_ROOT/initialization"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TORCH_EXTENSIONS_DIR
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo=$REPO"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "python=$PYTHON"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "exact overlap gate requires RTX 2080 Ti, got $GPU_NAME" >&2
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
      | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$RUN_ROOT/preload/evidence.json" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch

path = Path(sys.argv[1])
torch.cuda.synchronize()
started = time.perf_counter()
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import preload_native_decoder_layer
from osuT5.osuT5.inference.optimized.kernels.q1_attention import preload_native_q1_attention
from osuT5.osuT5.inference.optimized.kernels.weight_only import preload_weight_only_extension

preload_native_q1_attention()
preload_native_decoder_layer()
preload_weight_only_extension()
torch.cuda.synchronize()
path.write_text(
    json.dumps({"schema_version": 1, "wall_seconds": time.perf_counter() - started}, indent=2) + "\n",
    encoding="utf-8",
)
PY

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
  max_batch_size=16
  auto_select_gamemode_model=true
  super_timing=false
  generate_positions=false
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

run_profile() {
  local role=$1
  local mode=$2
  local output="$RUN_ROOT/$role"
  local evidence="$RUN_ROOT/$role.evidence.json"
  local initialization="$RUN_ROOT/initialization/$role.json"
  local compiler_cache="$RUN_ROOT/compiler-cache/$role"
  if [[ -e "$output" || -e "$evidence" || -e "$initialization" ]]; then
    echo "role output already exists: $role" >&2
    exit 2
  fi
  mkdir -p "$output" "$compiler_cache/torch_inductor" \
    "$compiler_cache/triton" "$compiler_cache/cuda"
  (
    export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"
    export TRITON_CACHE_DIR="$compiler_cache/triton"
    export CUDA_CACHE_PATH="$compiler_cache/cuda"
    "$PYTHON" utils/run_exact_encoder_overlap_full_song.py \
      --config-name "$PROFILE_CONFIG" \
      --mode "$mode" \
      --evidence-path "$evidence" \
      --initialization-path "$initialization" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" || ! -s "$profile" || ! -s "$evidence" || ! -s "$initialization" ]]; then
    echo "missing profile/evidence/initialization for $role" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$role.profile-path.txt"
  "$PYTHON" utils/validate_weight_only_full_song_profile.py \
    --profile "$profile" --role candidate \
    > "$RUN_ROOT/$role.weight-only-validation.json"
  "$PYTHON" utils/validate_k4_profile_contract.py \
    --profile "$profile" --role candidate --block-size 4 \
    > "$RUN_ROOT/$role.k4-validation.json"
  "$PYTHON" utils/validate_k1_remainder_profile.py \
    --profile "$profile" --role candidate \
    > "$RUN_ROOT/$role.k1-validation.json"
}

run_profile baseline_first baseline
run_profile candidate_second candidate
run_profile candidate_first candidate
run_profile baseline_second baseline

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

set +e
"$PYTHON" utils/analyze_exact_encoder_overlap_reciprocal.py \
  --baseline-first-profile "$BASELINE_FIRST" \
  --baseline-first-evidence "$RUN_ROOT/baseline_first.evidence.json" \
  --baseline-first-initialization "$RUN_ROOT/initialization/baseline_first.json" \
  --candidate-second-profile "$CANDIDATE_SECOND" \
  --candidate-second-evidence "$RUN_ROOT/candidate_second.evidence.json" \
  --candidate-second-initialization "$RUN_ROOT/initialization/candidate_second.json" \
  --candidate-first-profile "$CANDIDATE_FIRST" \
  --candidate-first-evidence "$RUN_ROOT/candidate_first.evidence.json" \
  --candidate-first-initialization "$RUN_ROOT/initialization/candidate_first.json" \
  --baseline-second-profile "$BASELINE_SECOND" \
  --baseline-second-evidence "$RUN_ROOT/baseline_second.evidence.json" \
  --baseline-second-initialization "$RUN_ROOT/initialization/baseline_second.json" \
  --base-seed 12345 \
  --minimum-request-saving-seconds 1.0 \
  --json-output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"
ANALYSIS_STATUS=$?
set -e
if [[ ! -s "$RUN_ROOT/analysis.json" || ! -s "$RUN_ROOT/analysis.txt" ]]; then
  echo "exact overlap analysis artifacts are missing" >&2
  exit 2
fi
cat "$RUN_ROOT/analysis.txt"
exit "$ANALYSIS_STATUS"
