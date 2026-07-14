#!/usr/bin/env bash
#SBATCH --job-name=top256-sampler-component
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
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean scout worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed scout commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the scout branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/top256-sampler-component-${SLURM_JOB_ID:-manual}"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing scout worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "scout worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "scout worktree commit mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "scout worktree branch mismatch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "remote ref does not match requested scout commit" >&2
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
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "top-256 sampler scout requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT/output" \
  "$WORK/torch_extensions/top256-sampler-$COMMIT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/top256-sampler-$COMMIT"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "run_root=$RUN_ROOT"
echo "python=$PYTHON"
echo "torch_extensions=$TORCH_EXTENSIONS_DIR"
nvidia-smi -L
scontrol show job -o "${SLURM_JOB_ID:?Slurm job id is required}"

cd "$REPO"
set +e
"$PYTHON" utils/run_k4_top256_sampler_scout.py \
  --config-name profile_salvalai \
  --output-init-json "$RUN_ROOT/initialization.json" \
  --output-report-json "$RUN_ROOT/component.json" \
  --output-report-text "$RUN_ROOT/component.txt" \
  --inference-output-dir "$RUN_ROOT/output" \
  --max-samples 512 \
  --warmup 100 \
  --iterations 1000 \
  --rounds 5 \
  --fixed-main-steps 8294 \
  --max-candidate-ms-per-step 0.035 \
  --minimum-saving-seconds 0.3 \
  audio_path="$AUDIO" \
  output_path="$RUN_ROOT/output" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  super_timing=false \
  generate_positions=false \
  profile_inference=true \
  profile_detail_ranges=false \
  profile_cuda_capture=false \
  profile_pass_kind=untraced_control
SCOUT_STATUS=$?
set -e

for required in \
  "$RUN_ROOT/component.json" \
  "$RUN_ROOT/component.txt" \
  "$RUN_ROOT/initialization.json"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required scout artifact: $required" >&2
    exit 2
  fi
done
PROFILE_COUNT=$(find "$RUN_ROOT/output" -type f -name '*.profile.json' | wc -l)
OSU_COUNT=$(find "$RUN_ROOT/output" -type f -name '*.osu' | wc -l)
if [[ "$PROFILE_COUNT" -ne 1 || "$OSU_COUNT" -ne 1 ]]; then
  echo "expected one profile and .osu; got profiles=$PROFILE_COUNT osu=$OSU_COUNT" >&2
  exit 2
fi
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
if [[ "$SCOUT_STATUS" -ne 0 && "$SCOUT_STATUS" -ne 3 ]]; then
  echo "top-256 sampler scout failed before producing a gate result" >&2
  exit "$SCOUT_STATUS"
fi
if [[ "$SCOUT_STATUS" -eq 3 ]]; then
  echo "STOP_COMPONENT_SCOUT"
  exit 3
fi
echo "PROMOTE_TO_RUNTIME_SCOUT"
