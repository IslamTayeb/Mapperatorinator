#!/usr/bin/env bash
#SBATCH --job-name=encoder-transition-recip
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

EXPECTED_BASELINE_COMMIT=367a9563a4d03dc15215c740fbd56cd9290d6d8b
ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
BASELINE_REPO=${BASELINE_REPO:?Set BASELINE_REPO to the detached current-main worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean pushed candidate worktree}
BASELINE_COMMIT=${BASELINE_COMMIT:-$EXPECTED_BASELINE_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
BASELINE_BRANCH=${BASELINE_BRANCH:-DETACHED}
CANDIDATE_BRANCH=${CANDIDATE_BRANCH:-codex/500tps-transition-cleanup}
BASELINE_REMOTE_REF=${BASELINE_REMOTE_REF:-refs/remotes/islamtayeb/main}
CANDIDATE_REMOTE=${CANDIDATE_REMOTE:-islamtayeb}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-encoder-transition-full-song-reciprocal}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
EXTENSION_ROOT="$WORK/torch_extensions/encoder-transition-${SLURM_JOB_ID:-manual}"

die() {
  echo "$*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] || die "this gate must run inside a Slurm allocation"
[[ "$PROFILE_CONFIG" == profile_salvalai ]] \
  || die "full-song reciprocal requires PROFILE_CONFIG=profile_salvalai"
[[ "$BASELINE_COMMIT" == "$EXPECTED_BASELINE_COMMIT" ]] \
  || die "baseline commit is stale; refresh/rebase the gate against current main"
[[ -x "$PYTHON" ]] || die "missing environment Python at $PYTHON"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"

validate_repo() {
  local repo=$1
  local commit=$2
  local branch=$3
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "missing worktree at $repo"
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    git -C "$repo" status --short >&2
    die "worktree is dirty at $repo"
  fi
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] \
    || die "worktree $repo differs from expected commit $commit"
  if [[ "$branch" == DETACHED ]]; then
    [[ -z "$(git -C "$repo" branch --show-current)" ]] \
      || die "baseline worktree must be detached"
  else
    [[ "$(git -C "$repo" branch --show-current)" == "$branch" ]] \
      || die "worktree $repo differs from expected branch $branch"
  fi
}

validate_repo "$BASELINE_REPO" "$BASELINE_COMMIT" "$BASELINE_BRANCH"
validate_repo "$CANDIDATE_REPO" "$CANDIDATE_COMMIT" "$CANDIDATE_BRANCH"
git -C "$BASELINE_REPO" show-ref --verify --quiet "$BASELINE_REMOTE_REF" \
  || die "baseline worktree lacks fetched ref $BASELINE_REMOTE_REF"
[[ "$(git -C "$BASELINE_REPO" rev-parse "$BASELINE_REMOTE_REF")" == "$BASELINE_COMMIT" ]] \
  || die "current main advanced; refresh baseline and rebase candidate"
CANDIDATE_REMOTE_REF="refs/remotes/$CANDIDATE_REMOTE/$CANDIDATE_BRANCH"
git -C "$CANDIDATE_REPO" show-ref --verify --quiet "$CANDIDATE_REMOTE_REF" \
  || die "candidate worktree lacks fetched ref $CANDIDATE_REMOTE_REF"
[[ "$(git -C "$CANDIDATE_REPO" rev-parse "$CANDIDATE_REMOTE_REF")" == "$CANDIDATE_COMMIT" ]] \
  || die "candidate remote ref differs from CANDIDATE_COMMIT"

[[ "${SLURM_JOB_ACCOUNT:-}" == romerolab ]] \
  || die "expected Slurm account romerolab, got ${SLURM_JOB_ACCOUNT:-missing}"
[[ "${SLURM_JOB_PARTITION:-}" == gpu-common ]] \
  || die "expected partition gpu-common, got ${SLURM_JOB_PARTITION:-missing}"
OTHER_GPU_JOBS=$(squeue -u "$USER" -h -t R,PD -o '%i|%t|%b|%j' \
  | awk -F'|' -v current="$SLURM_JOB_ID" '$1 != current && $3 ~ /gpu/ {print}')
[[ -z "$OTHER_GPU_JOBS" ]] || {
  echo "$OTHER_GPU_JOBS" >&2
  die "another GPU job is queued or running for this user"
}

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
AVAILABLE_KB=$(df -Pk "$WORK" | awk 'NR==2 {print $4}')
[[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || die "could not determine storage availability"
(( AVAILABLE_KB >= 5 * 1024 * 1024 )) \
  || die "less than 5 GiB available under $WORK"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT" "$EXTENSION_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$WORK/tmp"
export PYTHONHASHSEED=0
export MAX_JOBS=${MAX_JOBS:-8}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
[[ "$GPU_NAME" == "NVIDIA GeForce RTX 2080 Ti" ]] \
  || die "reciprocal gate requires RTX 2080 Ti, got $GPU_NAME"

"$PYTHON" - "$RUN_ROOT/preflight.json" "$BASELINE_COMMIT" "$CANDIDATE_COMMIT" \
  "$GPU_NAME" "$AVAILABLE_KB" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

import torch
import transformers

output, baseline, candidate, gpu, available_kb = sys.argv[1:]
payload = {
    "schema_version": 1,
    "host": platform.node(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_account": os.environ["SLURM_JOB_ACCOUNT"],
    "slurm_partition": os.environ["SLURM_JOB_PARTITION"],
    "gpu": gpu,
    "baseline_commit": baseline,
    "candidate_commit": candidate,
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "storage_available_kb": int(available_kb),
    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
}
if not payload["cuda_available"]:
    raise SystemExit("CUDA is unavailable")
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
nvidia-smi -q > "$RUN_ROOT/gpu.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$RUN_ROOT/slurm-job.txt"

sanitize_key() {
  printf '%s' "$1" | sed 's/[^A-Za-z0-9._-]/_/g'
}
BASELINE_EXTENSION_DIR="$EXTENSION_ROOT/$(sanitize_key "$BASELINE_COMMIT")"
CANDIDATE_EXTENSION_DIR="$EXTENSION_ROOT/$(sanitize_key "$CANDIDATE_COMMIT")"
[[ "$BASELINE_EXTENSION_DIR" != "$CANDIDATE_EXTENSION_DIR" ]] \
  || die "baseline and candidate extension directories must differ"
mkdir -p "$BASELINE_EXTENSION_DIR" "$CANDIDATE_EXTENSION_DIR"

preload_extensions() {
  local role=$1
  local repo=$2
  local commit=$3
  local extension_dir=$4
  local output="$RUN_ROOT/$role-extension-setup.json"
  (
    export TORCH_EXTENSIONS_DIR="$extension_dir"
    cd "$repo"
    "$PYTHON" - "$output" "$role" "$commit" "$extension_dir" <<'PY'
import json
import sys
import time
from pathlib import Path

import torch

output, role, commit, extension_dir = sys.argv[1:]
torch.cuda.synchronize()
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
payload = {
    "schema_version": 1,
    "role": role,
    "commit": commit,
    "torch_extensions_dir": extension_dir,
    "wall_seconds": time.perf_counter() - started,
    "status": "preloaded",
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  )
  [[ -s "$output" ]] || die "missing extension setup evidence for $role"
}

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

run_profile() {
  local role=$1
  local target_repo=$2
  local extension_dir=$3
  local output="$RUN_ROOT/$role"
  local compiler_cache="$RUN_ROOT/compiler-cache/$role"
  [[ ! -e "$output" && ! -e "$compiler_cache" ]] \
    || die "run paths already exist for $role"
  mkdir -p "$output" "$compiler_cache/torch_inductor" \
    "$compiler_cache/triton" "$compiler_cache/cuda"
  (
    export TORCH_EXTENSIONS_DIR="$extension_dir"
    export TORCHINDUCTOR_CACHE_DIR="$compiler_cache/torch_inductor"
    export TRITON_CACHE_DIR="$compiler_cache/triton"
    export CUDA_CACHE_PATH="$compiler_cache/cuda"
    cd "$target_repo"
    "$PYTHON" "$CANDIDATE_REPO/utils/run_fixed_seed_inference.py" \
      --config-name "$PROFILE_CONFIG" \
      --target-repo "$target_repo" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}"
  )
  mapfile -t profiles < <(find "$output" -name '*.profile.json' -type f | sort)
  [[ "${#profiles[@]}" == 1 ]] \
    || die "$role must produce exactly one profile, got ${#profiles[@]}"
  printf '%s\n' "${profiles[0]}" > "$RUN_ROOT/$role.profile-path.txt"
  (
    cd "$CANDIDATE_REPO"
    "$PYTHON" - "${profiles[0]}" "$role" <<'PY'
import json
import sys

from utils.fixed_seed_inference import SEED_POLICY_VERSION, stage_seed

profile_path, role = sys.argv[1:]
metadata = json.load(open(profile_path, encoding="utf-8")).get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit(f"{role}: profile metadata is missing")
if metadata.get("reciprocal_seed_policy") != SEED_POLICY_VERSION:
    raise SystemExit(f"{role}: reciprocal seed policy is missing or wrong")
for label in ("timing_context", "main_generation"):
    if metadata.get(f"reciprocal_seed_{label}") != stage_seed(12345, label):
        raise SystemExit(f"{role}: reciprocal seed for {label} is wrong")
    for device in ("cpu", "cuda"):
        digest = metadata.get(f"reciprocal_rng_{device}_sha256_{label}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SystemExit(f"{role}: RNG fingerprint for {device}/{label} is missing")
PY
  )
}

preload_extensions baseline "$BASELINE_REPO" "$BASELINE_COMMIT" \
  "$BASELINE_EXTENSION_DIR"
preload_extensions candidate "$CANDIDATE_REPO" "$CANDIDATE_COMMIT" \
  "$CANDIDATE_EXTENSION_DIR"

run_profile baseline_first "$BASELINE_REPO" "$BASELINE_EXTENSION_DIR"
run_profile candidate_first "$CANDIDATE_REPO" "$CANDIDATE_EXTENSION_DIR"
run_profile candidate_second "$CANDIDATE_REPO" "$CANDIDATE_EXTENSION_DIR"
run_profile baseline_second "$BASELINE_REPO" "$BASELINE_EXTENSION_DIR"

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

for role in candidate_first candidate_second; do
  profile=$(<"$RUN_ROOT/$role.profile-path.txt")
  (
    cd "$CANDIDATE_REPO"
    "$PYTHON" utils/validate_encoder_stabilization_profile.py \
      --profile "$profile" \
      --output "$RUN_ROOT/$role.encoder-stabilization.json"
  )
done

(
  cd "$CANDIDATE_REPO"
  "$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
    --baseline-first-profile "$BASELINE_FIRST" \
    --candidate-first-profile "$CANDIDATE_FIRST" \
    --candidate-second-profile "$CANDIDATE_SECOND" \
    --baseline-second-profile "$BASELINE_SECOND" \
    --mode exact-fp32 \
    --json-output "$RUN_ROOT/reciprocal-analysis.json" \
    --text-output "$RUN_ROOT/reciprocal-analysis.txt"
)

grep -q '^mode=exact-fp32$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^parity.cross_candidate_exact=true$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^dispatch.changed_paths=0$' "$RUN_ROOT/reciprocal-analysis.txt"
grep -q '^dispatch.declaration_pass=true$' "$RUN_ROOT/reciprocal-analysis.txt"

for artifact in \
  "$RUN_ROOT/preflight.json" \
  "$RUN_ROOT/baseline-extension-setup.json" \
  "$RUN_ROOT/candidate-extension-setup.json" \
  "$RUN_ROOT/candidate_first.encoder-stabilization.json" \
  "$RUN_ROOT/candidate_second.encoder-stabilization.json" \
  "$RUN_ROOT/reciprocal-analysis.json" \
  "$RUN_ROOT/reciprocal-analysis.txt"; do
  [[ -s "$artifact" ]] || die "missing required artifact: $artifact"
done

find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "encoder_transition_reciprocal_gate=PASS"
