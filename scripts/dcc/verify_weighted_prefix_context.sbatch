#!/usr/bin/env bash
#SBATCH --job-name=weighted-prefix-context
#SBATCH --partition=common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
SHARED_REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated weighted-bucket DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed context-only verifier commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-weighted-bucket-verifier}
SOURCE_PROFILE="$WORK/runs/inference-denominator-five_smoke-49543717-a709b86/serial_multi_song-five_smoke-49543717-a709b86/lambada/repeat01/run05.profile.json"
SOURCE_AUDIO="$WORK/data/five-song-profile/lambada.mp3"
SOURCE_PROFILE_SHA=05e49f8799fb9611c2d5974877b83fc57940edafd45940ef7ed7b504ab12145f
SOURCE_AUDIO_SHA=1e692d9ffbc2a3051219ac19e96f2c32b428fa0e4fc36278022866e19e6e6183

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "isolated weighted-bucket checkout is missing at $REPO" >&2
  exit 2
fi
if [[ "$(realpath "$REPO")" == "$(realpath "$SHARED_REPO")" ]]; then
  echo "refusing to run weighted-prefix context replay from shared DCC checkout" >&2
  exit 2
fi
for path in "$SOURCE_PROFILE" "$SOURCE_AUDIO"; do
  if [[ ! -f "$path" ]]; then
    echo "weighted-prefix context source is missing at $path" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$SOURCE_PROFILE" | awk '{print $1}')" != "$SOURCE_PROFILE_SHA" ]]; then
  echo "weighted-prefix source profile SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE_AUDIO" | awk '{print $1}')" != "$SOURCE_AUDIO_SHA" ]]; then
  echo "weighted-prefix source audio SHA-256 mismatch" >&2
  exit 2
fi

COMMIT=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
if [[ "$COMMIT" != "$EXPECTED_COMMIT" || "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "checkout is $BRANCH@$COMMIT, expected $EXPECTED_BRANCH@$EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "weighted-bucket checkout is dirty; refusing context replay" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

RUN_ROOT="$WORK/runs/weighted-prefix-context-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/weighted-prefix-context.json"
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "source_profile=$SOURCE_PROFILE"
echo "source_audio=$SOURCE_AUDIO"
echo "run_root=$RUN_ROOT"

python -m py_compile \
  osuT5/osuT5/inference/optimized/batch/weighted_bucket.py \
  utils/verify_optimized_weighted_prefix_source.py

python utils/verify_optimized_weighted_prefix_source.py \
  --context-only \
  --config-name profile_salvalai_smoke15 \
  --source-profile "$SOURCE_PROFILE" \
  --source-audio "$SOURCE_AUDIO" \
  --report-path "$REPORT" \
  audio_path="$SOURCE_AUDIO" \
  output_path="$RUN_ROOT/output" \
  device=cpu \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=v32 \
  optimized_inference_mode=single \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  inference_generation_compile=false \
  inference_active_prefix_decode_loop=false \
  inference_active_prefix_decode_cuda_graph=false \
  inference_decode_session_runtime=false \
  inference_decode_session_cuda_graph=false \
  inference_native_decode_kernels=false \
  inference_native_q1_self_attention=false \
  inference_native_q1_rope_cache_self_attention=false

python - "$REPORT" <<'PY'
import json
import sys
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    canonical_json_sha256,
    validate_context_replay_report,
)

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
validate_context_replay_report(report)
digest = canonical_json_sha256(report)
print("context_replay_validation=PASS")
print(f"context_replay_report_sha256={digest}")
print("model_loaded=false")
print("cuda_used=false")
print("gpu_capture_authorized=false")
PY
sha256sum "$REPORT"
