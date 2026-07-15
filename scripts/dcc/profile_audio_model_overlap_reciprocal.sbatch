#!/usr/bin/env bash
#SBATCH --job-name=audio-model-overlap
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean scout worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed scout commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the scout branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-audio-model-overlap-reciprocal}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing executable Python at $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing scout worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "scout worktree is dirty at $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "scout worktree does not match MAPPERATORINATOR_COMMIT" >&2
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
if [[ ! -f "$AUDIO" ]]; then
  echo "missing audio at $AUDIO" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$(dirname "$RUN_ROOT")"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/audio-overlap-${COMMIT:0:12}"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "$PYTHON"):$PATH"
mkdir -p "$TORCH_EXTENSIONS_DIR"

for tool in ffmpeg ffprobe nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 2
  fi
done

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "run_root=$RUN_ROOT"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$WORK/logs/${RUN_LABEL}-preflight-${SLURM_JOB_ID:-manual}.txt"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != *"2080 Ti"* ]]; then
  echo "expected RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

cd "$REPO"
"$PYTHON" - <<'PY'
from osuT5.osuT5.inference.optimized.kernels.decoder_layer import (
    preload_native_decoder_layer,
)
from osuT5.osuT5.inference.optimized.kernels.q1_attention import (
    preload_native_q1_attention,
)
from osuT5.osuT5.inference.optimized.kernels.weight_only import (
    preload_weight_only_extension,
)
from osuT5.osuT5.inference.optimized.scout.int8_mlp import (
    preload_int8_mlp_extension,
)

preload_native_q1_attention()
preload_native_decoder_layer()
preload_weight_only_extension()
preload_int8_mlp_extension()
PY

set +e
"$PYTHON" utils/profile_audio_model_overlap.py reciprocal \
  --python "$PYTHON" \
  --config-name profile_salvalai \
  --run-root "$RUN_ROOT" \
  --minimum-saving-seconds 0.5 \
  audio_path="$AUDIO" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  super_timing=false \
  generate_positions=false
STATUS=$?
set -e

for artifact in analysis.json analysis.txt; do
  if [[ ! -s "$RUN_ROOT/$artifact" ]]; then
    echo "missing reciprocal artifact: $RUN_ROOT/$artifact" >&2
    exit 2
  fi
done

"$PYTHON" - "$RUN_ROOT/analysis.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not report.get("performance_conclusion_allowed"):
    raise SystemExit("audio/model overlap ended in setup failure; no performance conclusion")
print(f"audio_model_overlap_status={report['status']}")
print(f"promotion_pass={report['promotion_pass']}")
print(
    "mean_cold_wall_saving_seconds="
    f"{report['performance']['mean_cold_wall_saving_seconds']:.6f}"
)
PY

exit "$STATUS"
