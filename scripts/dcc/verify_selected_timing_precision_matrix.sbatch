#!/usr/bin/env bash
#SBATCH --job-name=timing-precision-matrix
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
REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean pushed worktree}
COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH}
REMOTE=${CANDIDATE_REMOTE:-origin}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/selected-timing-precision-matrix-${SLURM_JOB_ID:-manual}"
EXTENSIONS="$WORK/torch_extensions/timing-precision-matrix-${SLURM_JOB_ID:-manual}/$COMMIT"

if [[ ! -x "$PYTHON" || ! -f "$AUDIO" ]]; then
  echo "missing pinned Python or SALVALAI audio" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing candidate worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "candidate worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "candidate worktree commit mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "candidate worktree branch mismatch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched pushed ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed ref differs from candidate commit" >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "missing required tool: $tool" >&2
    exit 2
  }
done
mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/tmp" "$RUN_ROOT" "$EXTENSIONS"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR="$EXTENSIONS"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "timing precision matrix requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "repo=$REPO"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
nvidia-smi -L
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
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
  "$PYTHON" - "$RUN_ROOT/extension-setup.json" <<'PY'
import json
import time
from pathlib import Path
import sys
import torch

output = Path(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("native extension setup requires CUDA")
torch.cuda.synchronize()
started = time.perf_counter()
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import preload_native_decoder_layer
from osuT5.osuT5.inference.optimized.kernels.q1_attention import preload_native_q1_attention
from osuT5.osuT5.inference.optimized.kernels.weight_only import preload_weight_only_extension
from osuT5.osuT5.inference.optimized.scout.int8_mlp import preload_int8_mlp_extension
preload_native_q1_attention()
preload_native_decoder_layer()
preload_weight_only_extension()
preload_int8_mlp_extension()
torch.cuda.synchronize()
output.write_text(json.dumps({
    "schema_version": 1,
    "wall_seconds": time.perf_counter() - started,
    "peak_cuda_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
)

runner_for_arm() {
  case "$1" in
    control) printf '%s\n' utils/run_k4_shared_rope_fp16_cross.py ;;
    exact_fp32_native_self) printf '%s\n' utils/run_selected_timing_native_self.py ;;
    full_fp16) printf '%s\n' utils/run_selected_timing_full_fp16.py ;;
    fp16_weights_fp32_state) printf '%s\n' utils/run_selected_timing_fp16_weights.py ;;
    *) echo "unknown timing precision arm: $1" >&2; exit 2 ;;
  esac
}

run_arm() {
  local role=$1
  local arm=$2
  local runner
  runner=$(runner_for_arm "$arm")
  local output="$RUN_ROOT/$role"
  local init="$RUN_ROOT/$role.initialization.json"
  if [[ -e "$output" || -e "$init" ]]; then
    echo "run output already exists for $role" >&2
    exit 2
  fi
  mkdir -p "$output" "$RUN_ROOT/compiler-cache/$role/torch_inductor" \
    "$RUN_ROOT/compiler-cache/$role/triton" "$RUN_ROOT/compiler-cache/$role/cuda"
  (
    export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/compiler-cache/$role/torch_inductor"
    export TRITON_CACHE_DIR="$RUN_ROOT/compiler-cache/$role/triton"
    export CUDA_CACHE_PATH="$RUN_ROOT/compiler-cache/$role/cuda"
    cd "$REPO"
    "$PYTHON" "$runner" \
      --config-name profile_salvalai \
      --output-init-json "$init" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" || ! -s "$profile" || ! -s "$init" ]]; then
    echo "$role did not produce a profile and initialization evidence" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$role.profile-path.txt"
  case "$arm" in
    control|exact_fp32_native_self)
      local validation_role=control
      if [[ "$arm" == exact_fp32_native_self ]]; then
        validation_role=candidate
      fi
      (
        cd "$REPO"
        "$PYTHON" utils/validate_selected_timing_native_self_profile.py \
          --profile "$profile" --initialization "$init" \
          --role "$validation_role" > "$RUN_ROOT/$role.validation.json"
      )
      ;;
    full_fp16|fp16_weights_fp32_state)
      (
        cd "$REPO"
        "$PYTHON" utils/validate_selected_timing_precision_profile.py \
          --profile "$profile" --initialization "$init" \
          --mode "$arm" > "$RUN_ROOT/$role.validation.json"
      )
      ;;
  esac
}

run_arm control_first control
run_arm exact_first exact_fp32_native_self
run_arm full_fp16_first full_fp16
run_arm fp16_weights_first fp16_weights_fp32_state
run_arm fp16_weights_second fp16_weights_fp32_state
run_arm full_fp16_second full_fp16
run_arm exact_second exact_fp32_native_self
run_arm control_second control

CONTROL_FIRST=$(<"$RUN_ROOT/control_first.profile-path.txt")
CONTROL_SECOND=$(<"$RUN_ROOT/control_second.profile-path.txt")

analyze_arm() {
  local arm=$1
  local first=$2
  local second=$3
  shift 3
  (
    cd "$REPO"
    "$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
      --baseline-first-profile "$CONTROL_FIRST" \
      --candidate-first-profile "$first" \
      --candidate-second-profile "$second" \
      --baseline-second-profile "$CONTROL_SECOND" \
      --mode relaxed \
      --fixed-timing-tokens 821 \
      "$@" \
      --json-output "$RUN_ROOT/$arm.analysis.json" \
      --text-output "$RUN_ROOT/$arm.analysis.txt"
  )
}

analyze_arm exact_fp32_native_self \
  "$(<"$RUN_ROOT/exact_first.profile-path.txt")" \
  "$(<"$RUN_ROOT/exact_second.profile-path.txt")" \
  --require-exact-label timing_context \
  --require-exact-label main_generation \
  --require-exact-dispatch-label main_generation \
  --require-dispatch-declaration \
  --allow-dispatch-delta 'records.timing_context[[]*].optimized_dispatch_mode' \
  --allow-dispatch-delta 'records.timing_context[[]*].optimized_dispatch_policy.*' \
  --allow-dispatch-delta 'records.timing_context[[]*].optimized_timing_native_self.*' \
  --allow-dispatch-delta 'records.timing_context[[]*].optimized_dispatch_capture_hits' \
  --allow-dispatch-delta 'records.timing_context[[]*].optimized_dispatch_capture_hits.*' \
  --allow-optional-dispatch-delta 'records.timing_context[[]*].optimized_cuda_graphs.*'

analyze_arm full_fp16 \
  "$(<"$RUN_ROOT/full_fp16_first.profile-path.txt")" \
  "$(<"$RUN_ROOT/full_fp16_second.profile-path.txt")"

analyze_arm fp16_weights_fp32_state \
  "$(<"$RUN_ROOT/fp16_weights_first.profile-path.txt")" \
  "$(<"$RUN_ROOT/fp16_weights_second.profile-path.txt")"

(
  cd "$REPO"
  "$PYTHON" utils/summarize_selected_timing_precision_matrix.py \
    --exact-fp32-native-self "$RUN_ROOT/exact_fp32_native_self.analysis.json" \
    --full-fp16 "$RUN_ROOT/full_fp16.analysis.json" \
    --fp16-weights-fp32-state "$RUN_ROOT/fp16_weights_fp32_state.analysis.json" \
    --json-output "$RUN_ROOT/analysis.json" \
    --text-output "$RUN_ROOT/analysis.txt"
)

for artifact in \
  "$RUN_ROOT/extension-setup.json" \
  "$RUN_ROOT/exact_fp32_native_self.analysis.json" \
  "$RUN_ROOT/full_fp16.analysis.json" \
  "$RUN_ROOT/fp16_weights_fp32_state.analysis.json" \
  "$RUN_ROOT/analysis.json" \
  "$RUN_ROOT/analysis.txt"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing required matrix artifact: $artifact" >&2
    exit 2
  fi
done
for role in control_first exact_first full_fp16_first fp16_weights_first \
  fp16_weights_second full_fp16_second exact_second control_second; do
  if [[ ! -s "$RUN_ROOT/$role.validation.json" ]]; then
    echo "missing validation for $role" >&2
    exit 2
  fi
done

grep -q '^measurement.reciprocal=true$' "$RUN_ROOT/analysis.txt"
grep -q '^measurement.fixed_main_tokens=8294$' "$RUN_ROOT/analysis.txt"
grep -q '^measurement.fixed_timing_tokens=821$' "$RUN_ROOT/analysis.txt"
find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*.analysis.json' -o -name 'analysis.json' \
    -o -name '*.validation.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
