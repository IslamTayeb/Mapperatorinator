#!/usr/bin/env bash
#SBATCH --job-name=split-weight-full-recip
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
BASELINE_BRANCH=${BASELINE_BRANCH:?Set BASELINE_BRANCH explicitly}
CANDIDATE_BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH explicitly}
CANDIDATE_REMOTE=${CANDIDATE_REMOTE:-origin}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}
BASELINE_RUNNER=${BASELINE_RUNNER:-}
CANDIDATE_RUNNER=${CANDIDATE_RUNNER:-utils/run_approximate_weight_only.py}
REQUIRE_EXACT_TIMING=${REQUIRE_EXACT_TIMING:-true}
REQUIRE_K4_CANDIDATE=${REQUIRE_K4_CANDIDATE:-false}
REQUIRE_SHARED_ROPE_INCREMENTAL=${REQUIRE_SHARED_ROPE_INCREMENTAL:-false}
REQUIRE_K1_REMAINDER_INCREMENTAL=${REQUIRE_K1_REMAINDER_INCREMENTAL:-false}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-split-weight-full-song-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
EXTENSION_JOB_KEY=${SLURM_JOB_ID:-manual}
EXTENSION_CACHE_ROOT_OVERRIDE=${EXTENSION_CACHE_ROOT_OVERRIDE:-}
BASELINE_EXTENSION_KEY_OVERRIDE=${BASELINE_EXTENSION_KEY_OVERRIDE:-}
CANDIDATE_EXTENSION_KEY_OVERRIDE=${CANDIDATE_EXTENSION_KEY_OVERRIDE:-}

if [[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]]; then
  echo "split+weight reciprocal requires one shared baseline/candidate commit" >&2
  exit 2
fi
if [[ "$BASELINE_BRANCH" != "$CANDIDATE_BRANCH" ]]; then
  echo "split+weight reciprocal requires one shared baseline/candidate branch" >&2
  exit 2
fi
BASELINE_REPO_RESOLVED=$(realpath -e "$BASELINE_REPO")
CANDIDATE_REPO_RESOLVED=$(realpath -e "$CANDIDATE_REPO")
if [[ "$BASELINE_REPO_RESOLVED" != "$CANDIDATE_REPO_RESOLVED" ]]; then
  echo "split+weight reciprocal requires one shared baseline/candidate worktree" >&2
  exit 2
fi

if [[ "$PROFILE_CONFIG" != profile_salvalai ]]; then
  echo "full-song reciprocal requires PROFILE_CONFIG=profile_salvalai" >&2
  exit 2
fi
if [[ "$CANDIDATE_RUNNER" = /* || "$CANDIDATE_RUNNER" == *".."* ]]; then
  echo "CANDIDATE_RUNNER must be a repository-relative path without .." >&2
  exit 2
fi
if [[ ! -f "$CANDIDATE_REPO/$CANDIDATE_RUNNER" ]]; then
  echo "missing candidate runner at $CANDIDATE_REPO/$CANDIDATE_RUNNER" >&2
  exit 2
fi
if [[ -n "$BASELINE_RUNNER" ]]; then
  if [[ "$BASELINE_RUNNER" = /* || "$BASELINE_RUNNER" == *".."* ]]; then
    echo "BASELINE_RUNNER must be a repository-relative path without .." >&2
    exit 2
  fi
  if [[ ! -f "$BASELINE_REPO/$BASELINE_RUNNER" ]]; then
    echo "missing baseline runner at $BASELINE_REPO/$BASELINE_RUNNER" >&2
    exit 2
  fi
fi
if [[ "$REQUIRE_EXACT_TIMING" != true && "$REQUIRE_EXACT_TIMING" != false ]]; then
  echo "REQUIRE_EXACT_TIMING must be true or false" >&2
  exit 2
fi
if [[ "$REQUIRE_K4_CANDIDATE" != true && "$REQUIRE_K4_CANDIDATE" != false ]]; then
  echo "REQUIRE_K4_CANDIDATE must be true or false" >&2
  exit 2
fi
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" != true \
  && "$REQUIRE_SHARED_ROPE_INCREMENTAL" != false ]]; then
  echo "REQUIRE_SHARED_ROPE_INCREMENTAL must be true or false" >&2
  exit 2
fi
if [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" != true \
  && "$REQUIRE_K1_REMAINDER_INCREMENTAL" != false ]]; then
  echo "REQUIRE_K1_REMAINDER_INCREMENTAL must be true or false" >&2
  exit 2
fi
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true \
  && "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  echo "shared-RoPE and K1-remainder incremental gates are mutually exclusive" >&2
  exit 2
fi
if [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  if [[ "$REQUIRE_K4_CANDIDATE" != true || "$REQUIRE_EXACT_TIMING" != true ]]; then
    echo "K1-remainder gate requires K4 and exact timing" >&2
    exit 2
  fi
  if [[ "$BASELINE_RUNNER" \
    != utils/run_k4_shared_rope_approximate_weight_only.py ]]; then
    echo "K1-remainder gate requires the exact shared-RoPE K4 control" >&2
    exit 2
  fi
  if [[ "$CANDIDATE_RUNNER" \
    != utils/run_k4_shared_rope_k1_remainder.py ]]; then
    echo "K1-remainder gate requires the captured-remainder runner" >&2
    exit 2
  fi
elif [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true ]]; then
  if [[ "$REQUIRE_K4_CANDIDATE" != true ]]; then
    echo "shared-RoPE incremental gate requires K4 contract validation" >&2
    exit 2
  fi
  if [[ "$REQUIRE_EXACT_TIMING" != true ]]; then
    echo "shared-RoPE incremental gate requires exact timing parity" >&2
    exit 2
  fi
  if [[ "$BASELINE_RUNNER" != utils/run_k4_approximate_weight_only.py ]]; then
    echo "shared-RoPE incremental gate requires the exact combined control runner" >&2
    exit 2
  fi
  if [[ "$CANDIDATE_RUNNER" \
    != utils/run_k4_shared_rope_approximate_weight_only.py ]]; then
    echo "shared-RoPE incremental gate requires the exact shared-RoPE runner" >&2
    exit 2
  fi
elif [[ "$REQUIRE_K4_CANDIDATE" == true ]]; then
  if [[ -n "$BASELINE_RUNNER" ]]; then
    echo "ordinary K4 gate requires the accepted baseline runner" >&2
    exit 2
  fi
  if [[ "$CANDIDATE_RUNNER" != utils/run_k4_approximate_weight_only.py ]]; then
    echo "K4 candidate gate requires the exact combined runner" >&2
    exit 2
  fi
  if [[ "$REQUIRE_EXACT_TIMING" != false ]]; then
    echo "K4 candidate gate requires documented timing drift" >&2
    exit 2
  fi
elif [[ -n "$BASELINE_RUNNER" \
  || "$CANDIDATE_RUNNER" != utils/run_approximate_weight_only.py ]]; then
  echo "non-K4 gate requires the default mixed-weight runner" >&2
  exit 2
fi
if [[ "$REQUIRE_EXACT_TIMING" == false && "$REQUIRE_K4_CANDIDATE" != true ]]; then
  echo "timing drift may be enabled only for the K4 candidate" >&2
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
  "$RUN_ROOT/candidate-initialization" "$RUN_ROOT/baseline-extension-setup"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "baseline_commit=$BASELINE_COMMIT"
echo "candidate_commit=$CANDIDATE_COMMIT"
echo "baseline_branch=$BASELINE_BRANCH"
echo "candidate_branch=$CANDIDATE_BRANCH"
echo "candidate_remote_ref=$CANDIDATE_REMOTE_REF"
echo "same_commit_gate=true"
echo "profile_config=$PROFILE_CONFIG"
echo "baseline_runner=${BASELINE_RUNNER:-accepted}"
echo "candidate_runner=$CANDIDATE_RUNNER"
echo "require_exact_timing=$REQUIRE_EXACT_TIMING"
echo "require_k4_candidate=$REQUIRE_K4_CANDIDATE"
echo "require_shared_rope_incremental=$REQUIRE_SHARED_ROPE_INCREMENTAL"
echo "require_k1_remainder_incremental=$REQUIRE_K1_REMAINDER_INCREMENTAL"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "extension_cache_mode=$EXTENSION_CACHE_MODE"
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
  super_timing=false
  generate_positions=false
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

preload_baseline_extensions() {
  local role=$1
  local output="$RUN_ROOT/baseline-extension-setup/$role.json"
  if [[ -e "$output" ]]; then
    echo "baseline extension setup evidence already exists for $role" >&2
    exit 2
  fi
  (
    export TORCH_EXTENSIONS_DIR="$BASELINE_TORCH_EXTENSIONS_DIR"
    cd "$BASELINE_REPO"
    "$PYTHON" - "$output" "$role" "$BASELINE_COMMIT" \
      "$BASELINE_TORCH_EXTENSIONS_DIR" <<'PY'
import json
import math
import sys
import time
from pathlib import Path

import torch

output, role, commit, extension_dir = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("baseline native extension setup requires CUDA")
torch.cuda.synchronize()
allocated_before = int(torch.cuda.memory_allocated())
reserved_before = int(torch.cuda.memory_reserved())
peak_before = int(torch.cuda.max_memory_allocated())
started = time.perf_counter()
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
    preload_native_decoder_layer,
)
from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
    preload_native_q1_attention,
)

preload_native_q1_attention()
preload_native_decoder_layer()
torch.cuda.synchronize()
finished = time.perf_counter()
allocated_after = int(torch.cuda.memory_allocated())
reserved_after = int(torch.cuda.memory_reserved())
peak_after = int(torch.cuda.max_memory_allocated())
wall_seconds = finished - started
if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
    raise SystemExit("invalid baseline native extension setup wall")
payload = {
    "schema_version": 1,
    "role": role,
    "commit": commit,
    "torch_extensions_dir": extension_dir,
    "status": "preloaded",
    "wall_seconds": wall_seconds,
    "cuda_memory": {
        "allocated_bytes_before": allocated_before,
        "allocated_bytes_after": allocated_after,
        "allocated_bytes_delta": allocated_after - allocated_before,
        "reserved_bytes_before": reserved_before,
        "reserved_bytes_after": reserved_after,
        "reserved_bytes_delta": reserved_after - reserved_before,
        "max_allocated_bytes_before": peak_before,
        "max_allocated_bytes_after": peak_after,
        "max_allocated_bytes_delta": peak_after - peak_before,
    },
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  )
  if [[ ! -s "$output" ]]; then
    echo "missing baseline extension setup evidence for $role" >&2
    exit 2
  fi
}

run_profile() {
  local role=$1
  local repo=$2
  local extension_dir=$3
  local candidate=$4
  local output="$RUN_ROOT/$role"
  local compiler_cache="$RUN_ROOT/compiler-cache/$role"
  local init_json="$RUN_ROOT/candidate-initialization/$role.json"
  local runtime_candidate=$candidate
  local runner=$CANDIDATE_RUNNER
  if [[ "$candidate" == 0 && -n "$BASELINE_RUNNER" ]]; then
    runtime_candidate=1
    runner=$BASELINE_RUNNER
  fi
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
    if [[ "$runtime_candidate" == 1 ]]; then
      "$PYTHON" "$CANDIDATE_REPO/$runner" \
        --config-name "$PROFILE_CONFIG" \
        --output-init-json "$init_json" \
        output_path="$output" \
        "${COMMON_OVERRIDES[@]}" \
        "${legacy_reset[@]}"
    else
      "$PYTHON" "$CANDIDATE_REPO/utils/run_fixed_seed_inference.py" \
        --config-name "$PROFILE_CONFIG" \
        --target-repo "$BASELINE_REPO" \
        output_path="$output" \
        "${COMMON_OVERRIDES[@]}" \
        "${legacy_reset[@]}"
    fi
  )
  if [[ "$runtime_candidate" == 1 && ! -s "$init_json" ]]; then
    echo "missing candidate initialization evidence for $role" >&2
    exit 2
  fi
  if [[ "$runtime_candidate" == 0 && -e "$init_json" ]]; then
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
  local validation_role=baseline
  if [[ "$runtime_candidate" == 1 ]]; then
    validation_role=candidate
  fi
  (
    cd "$CANDIDATE_REPO"
    "$PYTHON" utils/validate_weight_only_full_song_profile.py \
      --profile "$profile" \
      --role "$validation_role" \
      > "$RUN_ROOT/$role.weight-only-validation.json"
  )
  local k4_validation_role=baseline
  if [[ "$runtime_candidate" == 1 && "$REQUIRE_K4_CANDIDATE" == true ]]; then
    k4_validation_role=candidate
  fi
  (
    cd "$CANDIDATE_REPO"
    "$PYTHON" utils/validate_k4_profile_contract.py \
      --profile "$profile" \
      --role "$k4_validation_role" \
      --block-size 4 \
      > "$RUN_ROOT/$role.k4-validation.json"
  )
  if [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
    local remainder_role=control
    if [[ "$candidate" == 1 ]]; then
      remainder_role=candidate
    fi
    (
      cd "$CANDIDATE_REPO"
      "$PYTHON" utils/validate_k1_remainder_profile.py \
        --profile "$profile" \
        --role "$remainder_role" \
        > "$RUN_ROOT/$role.k1-remainder-validation.json"
    )
  fi
  (
    cd "$CANDIDATE_REPO"
    "$PYTHON" - "$profile" "$role" <<'PY'
import json
import sys

from utils.fixed_seed_inference import SEED_POLICY_VERSION, stage_seed

profile_path, role = sys.argv[1:]
metadata = json.loads(open(profile_path, encoding="utf-8").read()).get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit(f"{role}: profile metadata is missing")
if metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
    raise SystemExit(f"{role}: reciprocal seed policy is missing or wrong")
for label in ("timing_context", "main_generation"):
    expected_seed = stage_seed(12345, label)
    if metadata.get(f"reciprocal_seed_{label}") != expected_seed:
        raise SystemExit(f"{role}: reciprocal seed for {label} is wrong")
    for device in ("cpu", "cuda"):
        digest = metadata.get(f"reciprocal_rng_{device}_sha256_{label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"{role}: RNG fingerprint for {device}/{label} is missing")
PY
  )
}

preload_baseline_extensions baseline_first
run_profile baseline_first "$BASELINE_REPO" "$BASELINE_TORCH_EXTENSIONS_DIR" 0
run_profile candidate_first "$CANDIDATE_REPO" "$CANDIDATE_TORCH_EXTENSIONS_DIR" 1
run_profile candidate_second "$CANDIDATE_REPO" "$CANDIDATE_TORCH_EXTENSIONS_DIR" 1
preload_baseline_extensions baseline_second
run_profile baseline_second "$BASELINE_REPO" "$BASELINE_TORCH_EXTENSIONS_DIR" 0

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

"$PYTHON" - "$RUN_ROOT" "$BASELINE_COMMIT" \
  "$BASELINE_TORCH_EXTENSIONS_DIR" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_commit = sys.argv[2]
expected_extension_dir = sys.argv[3]
runs = {}
memory_fields = (
    "allocated_bytes_before",
    "allocated_bytes_after",
    "allocated_bytes_delta",
    "reserved_bytes_before",
    "reserved_bytes_after",
    "reserved_bytes_delta",
    "max_allocated_bytes_before",
    "max_allocated_bytes_after",
    "max_allocated_bytes_delta",
)
for role in ("baseline_first", "baseline_second"):
    path = root / "baseline-extension-setup" / f"{role}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != role or payload.get("status") != "preloaded":
        raise SystemExit(f"{role}: invalid baseline extension setup identity")
    if payload.get("commit") != expected_commit:
        raise SystemExit(f"{role}: baseline extension setup commit mismatch")
    if payload.get("torch_extensions_dir") != expected_extension_dir:
        raise SystemExit(f"{role}: baseline extension setup cache mismatch")
    wall = payload.get("wall_seconds")
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
    ):
        raise SystemExit(f"{role}: invalid baseline extension setup wall")
    memory = payload.get("cuda_memory")
    if not isinstance(memory, dict):
        raise SystemExit(f"{role}: missing baseline extension setup memory")
    for key in memory_fields:
        if isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int):
            raise SystemExit(f"{role}: invalid baseline setup memory field {key}")
    runs[role] = payload

summary = {
    "schema_version": 1,
    "commit": expected_commit,
    "torch_extensions_dir": expected_extension_dir,
    "runs": runs,
    "medians": {
        "wall_seconds": statistics.median(
            float(payload["wall_seconds"]) for payload in runs.values()
        ),
        "cuda_memory": {
            key: statistics.median(
                int(payload["cuda_memory"][key]) for payload in runs.values()
            )
            for key in memory_fields
        },
    },
}
(root / "baseline-extension-setup-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

INITIALIZATION_ROLES=candidate_first,candidate_second
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true \
  || "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  INITIALIZATION_ROLES=baseline_first,candidate_first,candidate_second,baseline_second
fi
"$PYTHON" - "$RUN_ROOT" "$INITIALIZATION_ROLES" \
  "$REQUIRE_SHARED_ROPE_INCREMENTAL" \
  "$REQUIRE_K1_REMAINDER_INCREMENTAL" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
roles = tuple(sys.argv[2].split(","))
require_shared_rope = sys.argv[3] == "true"
shared_rope_both_arms = sys.argv[4] == "true"
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
for role in roles:
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

if require_shared_rope or shared_rope_both_arms:
    shared_roles = roles if shared_rope_both_arms else (
        "candidate_first",
        "candidate_second",
    )
    for role in shared_roles:
        shared = runs[role].get("shared_rope")
        if not isinstance(shared, dict):
            raise SystemExit(f"{role}: missing shared-RoPE evidence")
        if shared.get("scope") != "main-model-only":
            raise SystemExit(f"{role}: shared-RoPE scope is wrong")
        if shared.get("incremental_exactness_claim") is not True:
            raise SystemExit(f"{role}: shared-RoPE exactness declaration is missing")
        stats = shared.get("stats")
        if not isinstance(stats, dict):
            raise SystemExit(f"{role}: shared-RoPE stats are missing")
        if stats.get("forwards", 0) <= 0 or stats.get("reuses", 0) <= 0:
            raise SystemExit(f"{role}: shared-RoPE did not execute")
        if stats.get("computes") != stats.get("expected_computes"):
            raise SystemExit(f"{role}: shared-RoPE compute accounting failed")
        if stats.get("reuses") != stats.get("expected_reuses"):
            raise SystemExit(f"{role}: shared-RoPE reuse accounting failed")
    if not shared_rope_both_arms:
        for role in ("baseline_first", "baseline_second"):
            if "shared_rope" in runs[role] or "combined_runtime" in runs[role]:
                raise SystemExit(
                    f"{role}: combined control unexpectedly enabled shared RoPE"
                )

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

ANALYSIS_MODE=relaxed
ANALYSIS_CLAIM=relaxed-nonexact
ANALYSIS_EXACT_ARGS=()
ANALYSIS_DISPATCH_ARGS=(
  --allow-optional-dispatch-delta 'optimized_result_class'
  --allow-dispatch-delta
    'records.main_generation[[]*].optimized_dispatch_capture_hits'
  --allow-dispatch-delta
    'records.main_generation[[]*].optimized_dispatch_capture_hits.*'
  --allow-dispatch-delta
    'records.main_generation[[]*].native_cross_mlp_tail_*'
  --allow-optional-dispatch-delta
    'records.main_generation[[]*].optimized_cuda_graphs.*'
)
if [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  ANALYSIS_EXACT_ARGS=(
    --require-exact-label timing_context
    --require-exact-label main_generation
  )
  ANALYSIS_DISPATCH_ARGS=(
    # The K1 parent is captured before the K4 parent for signatures with a
    # remainder.  That changes which parent capture owns the profiler's
    # capture-hit counters, without changing the child model graph or its
    # executed decoder dispatch.  The dedicated remainder validator below
    # still requires every remainder to move from eager to the K1 graph.
    --allow-dispatch-delta
      'records.*[[]*].optimized_dispatch_capture_hits'
    --allow-dispatch-delta
      'records.*[[]*].optimized_dispatch_capture_hits.*'
    --allow-dispatch-delta 'records.*[[]*].optimized_cuda_graphs.*'
  )
elif [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true ]]; then
  # Both arms retain the documented-drift mixed-weight runtime. Require an
  # exact incremental shared-RoPE delta without relabeling the overall runtime.
  ANALYSIS_MODE=relaxed
  ANALYSIS_CLAIM=relaxed-nonexact
  ANALYSIS_EXACT_ARGS=(
    --require-exact-label timing_context
    --require-exact-label main_generation
    --require-exact-dispatch-label timing_context
    --require-exact-dispatch-label main_generation
  )
  ANALYSIS_DISPATCH_ARGS=()
elif [[ "$REQUIRE_EXACT_TIMING" == true ]]; then
  ANALYSIS_EXACT_ARGS=(
    --require-exact-label timing_context
    --require-exact-dispatch-label timing_context
  )
else
  ANALYSIS_DISPATCH_ARGS+=(
    --allow-dispatch-delta
      'records.timing_context[[]*].optimized_dispatch_capture_hits'
    --allow-dispatch-delta
      'records.timing_context[[]*].optimized_dispatch_capture_hits.*'
    --allow-optional-dispatch-delta
      'records.timing_context[[]*].optimized_cuda_graphs.*'
  )
fi

(
  cd "$CANDIDATE_REPO"
  "$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
    --baseline-first-profile "$BASELINE_FIRST" \
    --candidate-first-profile "$CANDIDATE_FIRST" \
    --candidate-second-profile "$CANDIDATE_SECOND" \
    --baseline-second-profile "$BASELINE_SECOND" \
    --mode "$ANALYSIS_MODE" \
    "${ANALYSIS_EXACT_ARGS[@]}" \
    --require-dispatch-declaration \
    "${ANALYSIS_DISPATCH_ARGS[@]}" \
    --json-output "$RUN_ROOT/reciprocal-analysis.json" \
    --text-output "$RUN_ROOT/reciprocal-analysis.txt"
)

for artifact in \
  "$RUN_ROOT/baseline-extension-setup/baseline_first.json" \
  "$RUN_ROOT/baseline-extension-setup/baseline_second.json" \
  "$RUN_ROOT/baseline-extension-setup-summary.json" \
  "$RUN_ROOT/candidate-initialization/candidate_first.json" \
  "$RUN_ROOT/candidate-initialization/candidate_second.json" \
  "$RUN_ROOT/candidate-initialization-summary.json" \
  "$RUN_ROOT/baseline_first.weight-only-validation.json" \
  "$RUN_ROOT/candidate_first.weight-only-validation.json" \
  "$RUN_ROOT/candidate_second.weight-only-validation.json" \
  "$RUN_ROOT/baseline_second.weight-only-validation.json" \
  "$RUN_ROOT/baseline_first.k4-validation.json" \
  "$RUN_ROOT/candidate_first.k4-validation.json" \
  "$RUN_ROOT/candidate_second.k4-validation.json" \
  "$RUN_ROOT/baseline_second.k4-validation.json" \
  "$RUN_ROOT/reciprocal-analysis.json" \
  "$RUN_ROOT/reciprocal-analysis.txt"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing required artifact: $artifact" >&2
    exit 2
  fi
done
if [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  for role in baseline_first candidate_first candidate_second baseline_second; do
    artifact="$RUN_ROOT/$role.k1-remainder-validation.json"
    if [[ ! -s "$artifact" ]]; then
      echo "missing required K1-remainder artifact: $artifact" >&2
      exit 2
    fi
  done
fi
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true ]]; then
  for artifact in \
    "$RUN_ROOT/candidate-initialization/baseline_first.json" \
    "$RUN_ROOT/candidate-initialization/baseline_second.json"; do
    if [[ ! -s "$artifact" ]]; then
      echo "missing required incremental-control artifact: $artifact" >&2
      exit 2
    fi
  done
fi

grep -q "^mode=$ANALYSIS_MODE$" "$RUN_ROOT/reciprocal-analysis.txt"
grep -q "^parity.claim=$ANALYSIS_CLAIM$" "$RUN_ROOT/reciprocal-analysis.txt"
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true \
  || "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  grep -q '^parity.cross_candidate_exact=true$' \
    "$RUN_ROOT/reciprocal-analysis.txt"
fi
grep -q '^parity.baseline_repeat_stable=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.candidate_repeat_stable=true$' "$RUN_ROOT/reciprocal-analysis.txt"
EXPECTED_EXACT_LABELS=none
EXPECTED_EXACT_DISPATCH_LABELS=none
if [[ "$REQUIRE_SHARED_ROPE_INCREMENTAL" == true ]]; then
  EXPECTED_EXACT_LABELS=timing_context,main_generation
  EXPECTED_EXACT_DISPATCH_LABELS=$EXPECTED_EXACT_LABELS
elif [[ "$REQUIRE_K1_REMAINDER_INCREMENTAL" == true ]]; then
  EXPECTED_EXACT_LABELS=timing_context,main_generation
elif [[ "$REQUIRE_EXACT_TIMING" == true ]]; then
  EXPECTED_EXACT_LABELS=timing_context
  EXPECTED_EXACT_DISPATCH_LABELS=$EXPECTED_EXACT_LABELS
fi
grep -q "^parity.required_exact_labels=$EXPECTED_EXACT_LABELS$" \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.required_exact_labels_pass=true$' \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q "^parity.required_exact_dispatch_labels=$EXPECTED_EXACT_DISPATCH_LABELS$" \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.required_exact_dispatch_labels_pass=true$' \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^dispatch.declaration_pass=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.timing_model_seconds=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.main_model_seconds=' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.complete_request_wall_seconds=' \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.peak_cuda_memory_allocated_mb=' \
  "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^metric.setup_plus_capture_seconds=' \
  "$RUN_ROOT/reciprocal-analysis.txt"
find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*analysis.json' -o -name '*summary.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
