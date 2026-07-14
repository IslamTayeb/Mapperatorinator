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
EXTENSION_JOB_KEY=${SLURM_JOB_ID:-manual}
EXTENSION_CACHE_ROOT_OVERRIDE=${EXTENSION_CACHE_ROOT_OVERRIDE:-}
BASELINE_EXTENSION_KEY_OVERRIDE=${BASELINE_EXTENSION_KEY_OVERRIDE:-}
CANDIDATE_EXTENSION_KEY_OVERRIDE=${CANDIDATE_EXTENSION_KEY_OVERRIDE:-}

sanitize_extension_key() {
  local raw=$1
  local sanitized
  sanitized=$(printf '%s' "$raw" | sed 's/[^A-Za-z0-9._-]/_/g')
  if [[ -z "$sanitized" ]]; then
    echo "native extension commit key sanitized to empty: $raw" >&2
    exit 2
  fi
  printf '%s\n' "$sanitized"
}

BASELINE_EXTENSION_KEY=$(sanitize_extension_key "$BASELINE_COMMIT")
CANDIDATE_EXTENSION_KEY=$(sanitize_extension_key "$CANDIDATE_COMMIT")
if [[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]] \
  && [[ "$BASELINE_EXTENSION_KEY" == "$CANDIDATE_EXTENSION_KEY" ]]; then
  echo "distinct commits sanitize to the same native extension key" >&2
  echo "baseline_commit=$BASELINE_COMMIT" >&2
  echo "candidate_commit=$CANDIDATE_COMMIT" >&2
  echo "extension_key=$BASELINE_EXTENSION_KEY" >&2
  exit 2
fi
if [[ -z "$EXTENSION_CACHE_ROOT_OVERRIDE" ]]; then
  if [[ -n "$BASELINE_EXTENSION_KEY_OVERRIDE" || -n "$CANDIDATE_EXTENSION_KEY_OVERRIDE" ]]; then
    echo "extension key overrides require EXTENSION_CACHE_ROOT_OVERRIDE" >&2
    exit 2
  fi
  EXTENSION_CACHE_MODE=job_local
  EXTENSION_JOB_ROOT="$WORK/torch_extensions/reciprocal-$EXTENSION_JOB_KEY"
else
  if [[ "$EXTENSION_CACHE_ROOT_OVERRIDE" != /* ]]; then
    echo "EXTENSION_CACHE_ROOT_OVERRIDE must be absolute" >&2
    exit 2
  fi
  if [[ -z "$BASELINE_EXTENSION_KEY_OVERRIDE" || -z "$CANDIDATE_EXTENSION_KEY_OVERRIDE" ]]; then
    echo "existing extension cache reuse requires both commit key overrides" >&2
    exit 2
  fi
  if [[ "$BASELINE_EXTENSION_KEY_OVERRIDE" != "$BASELINE_EXTENSION_KEY" ]]; then
    echo "baseline extension key override does not match BASELINE_COMMIT" >&2
    exit 2
  fi
  if [[ "$CANDIDATE_EXTENSION_KEY_OVERRIDE" != "$CANDIDATE_EXTENSION_KEY" ]]; then
    echo "candidate extension key override does not match CANDIDATE_COMMIT" >&2
    exit 2
  fi
  if [[ ! -d "$EXTENSION_CACHE_ROOT_OVERRIDE" ]]; then
    echo "existing extension cache root does not exist" >&2
    exit 2
  fi
  EXTENSION_CACHE_MODE=existing_override
  EXTENSION_JOB_ROOT=$(realpath -e "$EXTENSION_CACHE_ROOT_OVERRIDE")
fi
BASELINE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$BASELINE_EXTENSION_KEY"
CANDIDATE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$CANDIDATE_EXTENSION_KEY"
if [[ "$EXTENSION_CACHE_MODE" == existing_override ]]; then
  for pair in \
    "$BASELINE_TORCH_EXTENSIONS_DIR:$BASELINE_EXTENSION_KEY" \
    "$CANDIDATE_TORCH_EXTENSIONS_DIR:$CANDIDATE_EXTENSION_KEY"; do
    extension_dir=${pair%%:*}
    expected_key=${pair##*:}
    if [[ ! -d "$extension_dir" ]]; then
      echo "missing existing native extension cache for commit key $expected_key" >&2
      exit 2
    fi
    resolved_extension_dir=$(realpath -e "$extension_dir")
    if [[ "$resolved_extension_dir" != "$EXTENSION_JOB_ROOT/$expected_key" ]]; then
      echo "native extension cache path/commit pairing is not exact" >&2
      exit 2
    fi
  done
fi

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
  "$BASELINE_TORCH_EXTENSIONS_DIR" "$CANDIDATE_TORCH_EXTENSIONS_DIR" \
  "$RUN_ROOT/native-extension-setup"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
echo "extension_cache_mode=$EXTENSION_CACHE_MODE"
echo "extension_job_root=$EXTENSION_JOB_ROOT"
echo "baseline_torch_extensions_dir=$BASELINE_TORCH_EXTENSIONS_DIR"
echo "candidate_torch_extensions_dir=$CANDIDATE_TORCH_EXTENSIONS_DIR"
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
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

preload_native_extensions() {
  local repo=$1
  local engine=$2
  local label=$3
  local extension_dir=$4
  local commit=$5
  local started_ns
  local finished_ns
  local status
  started_ns=$(date +%s%N)
  if [[ "$engine" == optimized ]]; then
    (
      export TORCH_EXTENSIONS_DIR="$extension_dir"
      cd "$repo"
      python - <<'PY'
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
    preload_native_decoder_layer,
)
from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
    preload_native_q1_attention,
)

preload_native_q1_attention()
preload_native_decoder_layer()
PY
    )
    status=preloaded
  else
    status=not_applicable
  fi
  finished_ns=$(date +%s%N)
  "$ENV/bin/python" - \
    "$RUN_ROOT/native-extension-setup/$label.json" \
    "$label" "$engine" "$repo" "$extension_dir" "$commit" \
    "$started_ns" "$finished_ns" "$status" <<'PY'
import json
import math
import sys
from pathlib import Path

(
    output,
    label,
    engine,
    repo,
    extension_dir,
    commit,
    started_ns,
    finished_ns,
    status,
) = sys.argv[1:]
started_ns = int(started_ns)
finished_ns = int(finished_ns)
wall_seconds = (finished_ns - started_ns) / 1_000_000_000
if finished_ns < started_ns or not math.isfinite(wall_seconds):
    raise SystemExit("invalid native extension preload wall")
payload = {
    "schema_version": 1,
    "label": label,
    "engine": engine,
    "repo": repo,
    "commit": commit,
    "torch_extensions_dir": extension_dir,
    "status": status,
    "started_at_unix_ns": started_ns,
    "finished_at_unix_ns": finished_ns,
    "wall_seconds": wall_seconds,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_profile() {
  local repo=$1
  local engine=$2
  local label=$3
  local extension_dir=$4
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
    export TORCH_EXTENSIONS_DIR="$extension_dir"
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

preload_native_extensions \
  "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_first \
  "$BASELINE_TORCH_EXTENSIONS_DIR" "$BASELINE_COMMIT"
run_profile \
  "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_first \
  "$BASELINE_TORCH_EXTENSIONS_DIR"
preload_native_extensions \
  "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_second \
  "$CANDIDATE_TORCH_EXTENSIONS_DIR" "$CANDIDATE_COMMIT"
run_profile \
  "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_second \
  "$CANDIDATE_TORCH_EXTENSIONS_DIR"
preload_native_extensions \
  "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_first \
  "$CANDIDATE_TORCH_EXTENSIONS_DIR" "$CANDIDATE_COMMIT"
run_profile \
  "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_first \
  "$CANDIDATE_TORCH_EXTENSIONS_DIR"
preload_native_extensions \
  "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_second \
  "$BASELINE_TORCH_EXTENSIONS_DIR" "$BASELINE_COMMIT"
run_profile \
  "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_second \
  "$BASELINE_TORCH_EXTENSIONS_DIR"

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
    metadata = profile.get("metadata", {})
    if metadata.get("profile_pass_kind") != "untraced_control":
        raise SystemExit(f"{label}: profile pass kind is not authoritative untraced")
    if metadata.get("authoritative_performance") is not True:
        raise SystemExit(f"{label}: authoritative_performance is not true")
    if metadata.get("profile_detail_ranges") is not False:
        raise SystemExit(f"{label}: profile_detail_ranges must be false")
    if metadata.get("profile_cuda_capture") is not False:
        raise SystemExit(f"{label}: profile_cuda_capture must be false")
print("inference_reciprocal_smoke_gate=PASS")
PY

find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*-compare.json' \) -print0 \
  | sort -z | xargs -0 sha256sum
