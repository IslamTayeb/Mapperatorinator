#!/usr/bin/env bash
#SBATCH --job-name=fp16-timing-hybrid
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean pushed worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}
RUN_LABEL=${RUN_LABEL:-fp16-timing-hybrid-full-song-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/stage-precision-hybrid-$COMMIT"

if [[ "$PROFILE_CONFIG" != profile_salvalai ]]; then
  echo "stage-precision reciprocal requires profile_salvalai" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "worktree is dirty at $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "worktree commit differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "worktree branch differs from MAPPERATORINATOR_BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote ref differs from requested commit" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$AUDIO" ]]; then
  echo "missing environment Python or SALVALAI audio" >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done

mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/tmp" "$TORCH_EXTENSIONS_DIR" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "repo=$REPO"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "stage-precision reciprocal requires RTX 2080 Ti, got $GPU_NAME" >&2
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
  super_timing=false
  generate_positions=false
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

(
cd "$REPO"
"$PYTHON" - "$RUN_ROOT/extension-preload.json" "$COMMIT" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch

output, commit = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("extension preload requires CUDA")
torch.cuda.synchronize()
started = time.perf_counter()
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import preload_native_decoder_layer
from osuT5.osuT5.inference.optimized.kernels.q1_attention import preload_native_q1_attention
from osuT5.osuT5.inference.optimized.kernels.weight_only import preload_weight_only_extension

preload_native_q1_attention()
preload_native_decoder_layer()
preload_weight_only_extension()
torch.cuda.synchronize()
payload = {
    "schema_version": 1,
    "commit": commit,
    "wall_seconds": time.perf_counter() - started,
    "extensions": ["q1_attention", "decoder_layer", "weight_only"],
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
)

run_profile() {
  local role=$1
  local kind=$2
  local output="$RUN_ROOT/$role"
  local init_json="$RUN_ROOT/$role.initialization.json"
  local runner=utils/run_stage_aware_k4_mixed_control.py
  local validation_role=control
  if [[ "$kind" == candidate ]]; then
    runner=utils/run_k4_stage_precision_hybrid.py
    validation_role=candidate
  elif [[ "$kind" != control ]]; then
    echo "unknown run kind $kind" >&2
    exit 2
  fi
  if [[ -e "$output" || -e "$init_json" ]]; then
    echo "run paths already exist for $role" >&2
    exit 2
  fi
  mkdir -p "$output" "$RUN_ROOT/compiler-cache/$role"/{torch_inductor,triton,cuda}
  (
    export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/compiler-cache/$role/torch_inductor"
    export TRITON_CACHE_DIR="$RUN_ROOT/compiler-cache/$role/triton"
    export CUDA_CACHE_PATH="$RUN_ROOT/compiler-cache/$role/cuda"
    cd "$REPO"
    "$PYTHON" "$runner" \
      --config-name "$PROFILE_CONFIG" \
      --output-init-json "$init_json" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  if [[ ! -s "$init_json" ]]; then
    echo "$role did not produce initialization evidence" >&2
    exit 2
  fi
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" || ! -s "$profile" ]]; then
    echo "$role did not produce an inference profile" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$role.profile-path.txt"
  (
    cd "$REPO"
    "$PYTHON" utils/validate_stage_precision_hybrid_profile.py \
      --profile "$profile" \
      --role "$validation_role" \
      > "$RUN_ROOT/$role.hybrid-validation.json"
    "$PYTHON" utils/validate_k4_profile_contract.py \
      --profile "$profile" \
      --role candidate \
      --block-size 4 \
      --timing-block-size 1 \
      > "$RUN_ROOT/$role.k4-validation.json"
  )
}

run_profile baseline_first control
run_profile candidate_first candidate
run_profile candidate_second candidate
run_profile baseline_second control

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

(
  cd "$REPO"
  "$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
    --baseline-first-profile "$BASELINE_FIRST" \
    --candidate-first-profile "$CANDIDATE_FIRST" \
    --candidate-second-profile "$CANDIDATE_SECOND" \
    --baseline-second-profile "$BASELINE_SECOND" \
    --mode relaxed \
    --allow-mixed-stage-precision \
    --require-dispatch-declaration \
    --allow-dispatch-delta \
      'records.timing_context[[]*].optimized_effective_config_version' \
    --allow-dispatch-delta \
      'records.timing_context[[]*].optimized_dispatch_capture_hits' \
    --allow-dispatch-delta \
      'records.timing_context[[]*].optimized_dispatch_capture_hits.*' \
    --allow-dispatch-delta \
      'records.timing_context[[]*].native_cross_mlp_tail_*' \
    --allow-optional-dispatch-delta \
      'records.timing_context[[]*].optimized_cuda_graphs.*' \
    --allow-optional-dispatch-delta \
      'records.main_generation[[]*].optimized_dispatch_capture_hits' \
    --allow-optional-dispatch-delta \
      'records.main_generation[[]*].optimized_dispatch_capture_hits.*' \
    --allow-optional-dispatch-delta \
      'records.main_generation[[]*].optimized_cuda_graphs.*' \
    --json-output "$RUN_ROOT/reciprocal-analysis.json" \
    --text-output "$RUN_ROOT/reciprocal-analysis.txt"
)

for role in baseline_first candidate_first candidate_second baseline_second; do
  for suffix in initialization.json hybrid-validation.json k4-validation.json; do
    if [[ ! -s "$RUN_ROOT/$role.$suffix" ]]; then
      echo "missing required artifact $role.$suffix" >&2
      exit 2
    fi
  done
done
for artifact in extension-preload.json reciprocal-analysis.json reciprocal-analysis.txt; do
  if [[ ! -s "$RUN_ROOT/$artifact" ]]; then
    echo "missing required artifact $artifact" >&2
    exit 2
  fi
done

grep -q '^mode=relaxed$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.baseline_repeat_stable=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.candidate_repeat_stable=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^dispatch.declaration_pass=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.timing_model_seconds=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.main_model_seconds=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.complete_request_wall_seconds=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^divergence.timing_context=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^divergence.main_generation=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^divergence.final_map_equal=' "$RUN_ROOT/reciprocal-analysis.txt"
find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*analysis.json' -o -name '*validation.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
