#!/usr/bin/env bash
#SBATCH --job-name=aot-k1-int8-recip
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
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the exact clean worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
REMOTE_BRANCH=${MAPPERATORINATOR_REMOTE_BRANCH:?Set MAPPERATORINATOR_REMOTE_BRANCH}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
AOT_MANIFEST=${AOT_NATIVE_EXTENSION_MANIFEST:?Set the external prebuilt manifest path}
AOT_MANIFEST_SHA256=${AOT_NATIVE_EXTENSION_MANIFEST_SHA256:?Set the exact manifest hash}
AOT_BUILD_RESULT=${AOT_NATIVE_EXTENSION_BUILD_RESULT:?Set the external build result path}
AOT_BUILD_RESULT_SHA256=${AOT_NATIVE_EXTENSION_BUILD_RESULT_SHA256:?Set the exact build result hash}
AOT_EXTENSION_CACHE_ROOT=${AOT_NATIVE_EXTENSION_CACHE_ROOT:?Set the exact cached load_inline root}
RUN_ROOT="$WORK/runs/aot-k1-int8-full-song-reciprocal-${SLURM_JOB_ID:-manual}"
SMI_TEMP="$WORK/logs/.aot-k1-int8-full-song-reciprocal-${SLURM_JOB_ID:-manual}.nvidia-smi.csv"

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
  echo "worktree differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "worktree differs from MAPPERATORINATOR_BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$REMOTE_BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote ref differs from MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
for required in "$PYTHON" "$AUDIO" "$AOT_MANIFEST" "$AOT_BUILD_RESULT"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done
if [[ ! -d "$AOT_EXTENSION_CACHE_ROOT" ]]; then
  echo "missing AOT native extension cache root: $AOT_EXTENSION_CACHE_ROOT" >&2
  exit 2
fi
if [[ ! "$AOT_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "AOT manifest hash must be 64 lowercase hex characters" >&2
  exit 2
fi
if [[ ! "$AOT_BUILD_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "AOT build-result hash must be 64 lowercase hex characters" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache"
export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TORCH_CUDA_ARCH_LIST=7.5
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "repo=$REPO"
echo "audio=$AUDIO"
echo "aot_manifest=$AOT_MANIFEST"
echo "aot_manifest_sha256=$AOT_MANIFEST_SHA256"
echo "aot_build_result=$AOT_BUILD_RESULT"
echo "aot_build_result_sha256=$AOT_BUILD_RESULT_SHA256"
echo "aot_extension_cache_root=$AOT_EXTENSION_CACHE_ROOT"
echo "run_root=$RUN_ROOT"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "AOT full-song reciprocal requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$SMI_TEMP" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    if [[ -s "$SMI_TEMP" ]]; then
      mv "$SMI_TEMP" "$RUN_ROOT/nvidia-smi.csv"
    fi
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"
"$PYTHON" utils/run_aot_full_song_reciprocal.py \
  --python "$PYTHON" \
  --commit "$COMMIT" \
  --audio "$AUDIO" \
  --manifest "$AOT_MANIFEST" \
  --manifest-sha256 "$AOT_MANIFEST_SHA256" \
  --build-result "$AOT_BUILD_RESULT" \
  --build-result-sha256 "$AOT_BUILD_RESULT_SHA256" \
  --extension-cache-root "$AOT_EXTENSION_CACHE_ROOT" \
  --run-root "$RUN_ROOT" \
  --minimum-cold-saving-seconds 0.5

for artifact in \
  "$RUN_ROOT/manifest-validation.json" \
  "$RUN_ROOT/copied-manifest-validation.json" \
  "$RUN_ROOT/native-extension-build.json" \
  "$RUN_ROOT/reciprocal-analysis.json" \
  "$RUN_ROOT/aot-gate.json" \
  "$RUN_ROOT/extension-load/cached_first.json" \
  "$RUN_ROOT/extension-load/direct_first.json" \
  "$RUN_ROOT/extension-load/direct_second.json" \
  "$RUN_ROOT/extension-load/cached_second.json"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing required artifact: $artifact" >&2
    exit 2
  fi
done
"$PYTHON" - "$RUN_ROOT/aot-gate.json" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
if payload.get("pass") is not True:
    raise SystemExit("AOT full-song reciprocal gate did not pass")
if payload.get("exact_parity_pass") is not True:
    raise SystemExit("AOT full-song reciprocal exact parity did not pass")
if payload.get("cold_saving_pass") is not True:
    raise SystemExit("AOT full-song reciprocal cold saving did not pass")
if payload.get("warm_no_regression_pass") is not True:
    raise SystemExit("AOT full-song reciprocal warm request regressed")
retained = payload.get("retained_runtime_initialization")
if not isinstance(retained, dict) or retained.get("combined_runtime") != (
    "k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1"
):
    raise SystemExit("AOT full-song reciprocal lost the retained K1+INT8 topology")
if retained.get("result_class") != "documented-drift" or retained.get(
    "exactness_claim"
) is not False:
    raise SystemExit("AOT full-song reciprocal changed the retained drift claim")
exactness = payload.get("exactness")
if not isinstance(exactness, dict) or exactness.get(
    "accepted_fp32_exactness_claim"
) is not False:
    raise SystemExit("AOT full-song reciprocal mislabeled accepted-FP32 exactness")
performance = payload.get("performance")
for metric in (
    "fixed_8294_main_seconds",
    "complete_request_wall_seconds",
    "cold_process_outer_wall_seconds",
):
    if not isinstance(performance, dict) or metric not in performance:
        raise SystemExit(f"AOT full-song reciprocal is missing {metric}")
loads = payload.get("native_extension_load_seconds", {}).get("runs", {})
if set(loads) != {"cached_first", "direct_first", "direct_second", "cached_second"}:
    raise SystemExit("AOT full-song reciprocal extension load roles are incomplete")
PY
echo "AOT_NATIVE_EXTENSION_FULL_SONG_PASS"
