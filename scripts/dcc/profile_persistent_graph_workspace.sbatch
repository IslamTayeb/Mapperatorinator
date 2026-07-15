#!/usr/bin/env bash
#SBATCH --job-name=persistent-graph-pool
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
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
RUN_ROOT=${RUN_ROOT:-$WORK/runs/persistent-graph-workspace-${SLURM_JOB_ID:-manual}}

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
  echo "scout worktree is not at MAPPERATORINATOR_COMMIT=$COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "scout worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF"; then
  echo "missing fetched remote ref $REMOTE_REF" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed remote ref differs from MAPPERATORINATOR_COMMIT" >&2
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

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$WORK/tmp"
export TORCH_EXTENSIONS_DIR="$WORK/cache/torch_extensions/persistent-${COMMIT:0:12}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
mkdir -p "$TORCH_EXTENSIONS_DIR"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "repo=$REPO"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_ref=$REMOTE_REF"
echo "audio=$AUDIO"
echo "run_root=$RUN_ROOT"
echo "python=$PYTHON"
echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "persistent graph scout requires RTX 2080 Ti, got $GPU_NAME" >&2
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

cd "$REPO"
"$PYTHON" utils/run_persistent_graph_workspace_scout.py \
  --config-name profile_salvalai \
  --output-init-json "$RUN_ROOT/initialization.json" \
  --output-manifest "$RUN_ROOT/manifest.json" \
  "${COMMON_OVERRIDES[@]}"

for artifact in \
  "$RUN_ROOT/initialization.json" \
  "$RUN_ROOT/manifest.json"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing persistent graph artifact: $artifact" >&2
    exit 2
  fi
done

"$PYTHON" utils/analyze_persistent_graph_workspace.py \
  --manifest "$RUN_ROOT/manifest.json" \
  --output-json "$RUN_ROOT/analysis.json" \
  --output-text "$RUN_ROOT/analysis.txt"

for artifact in \
  "$RUN_ROOT/analysis.json" \
  "$RUN_ROOT/analysis.txt"; do
  if [[ ! -s "$artifact" ]]; then
    echo "missing persistent graph analysis: $artifact" >&2
    exit 2
  fi
done
grep -q '^pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^exactness_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^capture_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^cross_request_hits_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^initialization_topology_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^profile_topology_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^pool_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^memory_pass=true$' "$RUN_ROOT/analysis.txt"
grep -q '^close_pass=true$' "$RUN_ROOT/analysis.txt"
