#!/usr/bin/env bash
#SBATCH --job-name=weight-only-full-recip
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
BASELINE_REPO=${BASELINE_REPO:?Set BASELINE_REPO to a clean pushed worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to a clean pushed worktree}
BASELINE_COMMIT=${BASELINE_COMMIT:?Set BASELINE_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
BASELINE_BRANCH=${BASELINE_BRANCH:-DETACHED}
CANDIDATE_BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH explicitly}
CANDIDATE_REMOTE=${CANDIDATE_REMOTE:-origin}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-weight-only-full-song-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
EXTENSION_JOB_KEY=${SLURM_JOB_ID:-manual}

if [[ "$PROFILE_CONFIG" != profile_salvalai ]]; then
  echo "full-song reciprocal requires PROFILE_CONFIG=profile_salvalai" >&2
  exit 2
fi

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
  exit 2
fi
EXTENSION_JOB_ROOT="$WORK/torch_extensions/reciprocal-$EXTENSION_JOB_KEY"
BASELINE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$BASELINE_EXTENSION_KEY"
CANDIDATE_TORCH_EXTENSIONS_DIR="$EXTENSION_JOB_ROOT/$CANDIDATE_EXTENSION_KEY"

validate_repo() {
  local repo=$1
  local commit=$2
  local branch=$3
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "worktree is dirty at $repo" >&2
    git -C "$repo" status --short >&2
    exit 2
  fi
  if [[ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]]; then
    echo "worktree $repo differs from expected commit $commit" >&2
    exit 2
  fi
  if [[ "$branch" == DETACHED ]]; then
    if [[ -n "$(git -C "$repo" branch --show-current)" ]]; then
      echo "worktree $repo must be detached" >&2
      exit 2
    fi
  elif [[ "$(git -C "$repo" branch --show-current)" != "$branch" ]]; then
    echo "worktree $repo differs from expected branch $branch" >&2
    exit 2
  fi
}

validate_repo "$BASELINE_REPO" "$BASELINE_COMMIT" "$BASELINE_BRANCH"
validate_repo "$CANDIDATE_REPO" "$CANDIDATE_COMMIT" "$CANDIDATE_BRANCH"
CANDIDATE_REMOTE_REF="refs/remotes/$CANDIDATE_REMOTE/$CANDIDATE_BRANCH"
if ! git -C "$CANDIDATE_REPO" show-ref --verify --quiet "$CANDIDATE_REMOTE_REF"; then
  echo "missing fetched candidate remote ref $CANDIDATE_REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" rev-parse "$CANDIDATE_REMOTE_REF")" != "$CANDIDATE_COMMIT" ]]; then
  echo "candidate remote ref differs from CANDIDATE_COMMIT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing environment Python at $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing input audio at $AUDIO" >&2
  exit 2
fi
export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$BASELINE_TORCH_EXTENSIONS_DIR" "$CANDIDATE_TORCH_EXTENSIONS_DIR" \
  "$RUN_ROOT/candidate-initialization"
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
echo "profile_config=$PROFILE_CONFIG"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "baseline_torch_extensions_dir=$BASELINE_TORCH_EXTENSIONS_DIR"
echo "candidate_torch_extensions_dir=$CANDIDATE_TORCH_EXTENSIONS_DIR"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "weight-only reciprocal requires RTX 2080 Ti, got $GPU_NAME" >&2
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
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

run_profile() {
  local role=$1
  local repo=$2
  local extension_dir=$3
  local candidate=$4
  local output="$RUN_ROOT/$role"
  local compiler_cache="$RUN_ROOT/compiler-cache/$role"
  local init_json="$RUN_ROOT/candidate-initialization/$role.json"
  local legacy_reset=()
  if grep -q 'inference_generation_compile:' "$repo/config.py"; then
    legacy_reset+=(inference_generation_compile=false)
  fi
  if [[ -e "$output" || -e "$compiler_cache" ]]; then
    echo "run paths already exist for $role" >&2
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
    if [[ "$candidate" == 1 ]]; then
      "$PYTHON" utils/run_approximate_weight_only.py \
        --config-name "$PROFILE_CONFIG" \
        --output-init-json "$init_json" \
        output_path="$output" \
        "${COMMON_OVERRIDES[@]}" \
        "${legacy_reset[@]}"
    else
      "$PYTHON" inference.py --config-name "$PROFILE_CONFIG" \
        output_path="$output" \
        "${COMMON_OVERRIDES[@]}" \
        "${legacy_reset[@]}"
    fi
  )
  if [[ "$candidate" == 1 && ! -s "$init_json" ]]; then
    echo "missing candidate initialization evidence for $role" >&2
    exit 2
  fi
  if [[ "$candidate" == 0 && -e "$init_json" ]]; then
    echo "baseline unexpectedly produced candidate initialization evidence" >&2
    exit 2
  fi
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing inference profile for $role" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$role.profile-path.txt"
}

run_profile baseline_first "$BASELINE_REPO" "$BASELINE_TORCH_EXTENSIONS_DIR" 0
run_profile candidate_first "$CANDIDATE_REPO" "$CANDIDATE_TORCH_EXTENSIONS_DIR" 1
run_profile candidate_second "$CANDIDATE_REPO" "$CANDIDATE_TORCH_EXTENSIONS_DIR" 1
run_profile baseline_second "$BASELINE_REPO" "$BASELINE_TORCH_EXTENSIONS_DIR" 0

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = {}
required_numbers = (
    "initialization_wall_seconds",
    "initialization_unattributed_seconds",
    "extension_init_seconds",
    "extension_allocated_bytes_delta",
    "extension_reserved_bytes_delta",
    "weight_pack_seconds",
    "weight_pack_allocated_bytes_delta",
    "weight_pack_reserved_bytes_delta",
    "retained_fp32_source_weight_bytes",
    "packed_weight_bytes",
)
for role in ("candidate_first", "candidate_second"):
    path = root / "candidate-initialization" / f"{role}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_class") != "documented-drift":
        raise SystemExit(f"{role}: candidate result class is not documented-drift")
    if payload.get("exactness_claim") is not False:
        raise SystemExit(f"{role}: candidate must not claim exactness")
    for key in required_numbers:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(f"{role}: missing numeric {key}")
        if not math.isfinite(float(value)):
            raise SystemExit(f"{role}: non-finite {key}")
    memory = payload.get("initialization_cuda_memory")
    if not isinstance(memory, dict):
        raise SystemExit(f"{role}: missing initialization_cuda_memory")
    for key in (
        "allocated_bytes_before",
        "allocated_bytes_after",
        "allocated_bytes_delta",
        "reserved_bytes_before",
        "reserved_bytes_after",
        "reserved_bytes_delta",
        "max_allocated_bytes_before",
        "max_allocated_bytes_after",
        "max_allocated_bytes_delta",
    ):
        if isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int):
            raise SystemExit(f"{role}: invalid initialization memory field {key}")
    runs[role] = payload

summary = {
    "schema_version": 1,
    "result_class": "documented-drift",
    "exactness_claim": False,
    "runs": runs,
    "medians": {
        key: statistics.median(float(runs[role][key]) for role in runs)
        for key in required_numbers
    },
}
(root / "candidate-initialization-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

(
  cd "$CANDIDATE_REPO"
  "$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
    --baseline-first-profile "$BASELINE_FIRST" \
    --candidate-first-profile "$CANDIDATE_FIRST" \
    --candidate-second-profile "$CANDIDATE_SECOND" \
    --baseline-second-profile "$BASELINE_SECOND" \
    --mode relaxed \
    --json-output "$RUN_ROOT/reciprocal-analysis.json" \
    --text-output "$RUN_ROOT/reciprocal-analysis.txt"
)

for artifact in \
  "$RUN_ROOT/candidate-initialization/candidate_first.json" \
  "$RUN_ROOT/candidate-initialization/candidate_second.json" \
  "$RUN_ROOT/candidate-initialization-summary.json" \
  "$RUN_ROOT/reciprocal-analysis.json" \
  "$RUN_ROOT/reciprocal-analysis.txt"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing required artifact: $artifact" >&2
    exit 2
  fi
done

grep -q '^mode=relaxed$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.claim=relaxed-nonexact$' "$RUN_ROOT/reciprocal-analysis.txt"
find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*analysis.json' -o -name '*summary.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
