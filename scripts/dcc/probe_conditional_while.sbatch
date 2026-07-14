#!/usr/bin/env bash
#SBATCH --job-name=conditional-while-probe
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
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean DCC worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the pushed branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_ROOT="$WORK/runs/conditional-while-probe-${SLURM_JOB_ID:-manual}"
NATIVE_BUILD_TOOLS=${MAPPERATORINATOR_NATIVE_BUILD_TOOLS:-$WORK/cache/native-build-tools}

fail() {
  echo "$*" >&2
  exit 2
}

[[ -x "$ENV/bin/python" ]] || fail "missing environment Python at $ENV/bin/python"
[[ -x "$ENV/bin/nvcc" ]] || fail "missing pinned nvcc at $ENV/bin/nvcc"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || fail "missing worktree at $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || fail "conditional worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || fail "worktree differs from COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || fail "worktree differs from BRANCH"
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || fail "missing fetched remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || fail "remote ref differs from COMMIT"
[[ -f "$AUDIO" ]] || fail "missing SALVALAI audio at $AUDIO"

export PATH="$ENV/bin:$PATH"
export PATH="$NATIVE_BUILD_TOOLS/bin:$PATH"
export PYTHONPATH="$NATIVE_BUILD_TOOLS${PYTHONPATH:+:$PYTHONPATH}"
for tool in ffmpeg ffprobe nvidia-smi; do
  command -v "$tool" >/dev/null 2>&1 || fail "missing required executable: $tool"
done
command -v ninja >/dev/null 2>&1 \
  || fail "missing ninja in native build tools at $NATIVE_BUILD_TOOLS"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$GPU_COUNT" -eq 1 ]] || fail "expected exactly one allocated GPU, got $GPU_COUNT"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "$GPU_NAME" == *"2080 Ti"* ]] || fail "expected RTX 2080 Ti, got $GPU_NAME"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/torch_extensions" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TORCH_CUDA_ARCH_LIST=7.5
export TMPDIR="$WORK/tmp"

finalize() {
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f -name '*.osu' -delete
    local invalid
    invalid=$(find "$RUN_ROOT" -type f ! -name '*.json' ! -name '*.txt' -print -quit)
    if [[ -n "$invalid" ]]; then
      echo "unexpected non-JSON/text artifact: $invalid" >&2
    fi
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "repo=$REPO"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "gpu=$GPU_NAME"
  echo "python=$ENV/bin/python"
  echo "nvcc=$ENV/bin/nvcc"
  echo "ninja=$(command -v ninja)"
  echo "native_build_tools=$NATIVE_BUILD_TOOLS"
  echo "audio=$AUDIO"
  echo "run_root=$RUN_ROOT"
  "$ENV/bin/nvcc" --version
  "$ENV/bin/python" - <<'PY'
import torch
import transformers
from cuda.bindings import runtime

required = (
    "cudaGraphConditionalHandleCreate",
    "cudaGraphAddNode",
    "cudaGraphNodeParams",
)
missing = [name for name in required if not hasattr(runtime, name)]
if missing:
    raise SystemExit(f"cuda.bindings is missing conditional APIs: {missing}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
PY
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$ENV/bin/python" utils/probe_cuda_conditional_while.py \
  --output-dir "$RUN_ROOT/capability" \
  --limits 1,3,10 \
  --replays 3 \
  --required-gpu-substring "2080 Ti"

"$ENV/bin/python" utils/profile_conditional_while_real_prefix.py \
  --config-name profile_salvalai \
  --output-dir "$RUN_ROOT/real-prefix" \
  --capture-output-path "$RUN_ROOT/real-prefix/baseline-capture" \
  --source-prefix 576 \
  --active-prefix 640 \
  --fixed-steps 8 \
  --forced-stops 1,3,7 \
  audio_path="$AUDIO" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  profile_inference=true \
  profile_detail_ranges=false \
  profile_cuda_capture=false \
  profile_pass_kind=untraced_control

find "$RUN_ROOT" -type f -name '*.osu' -delete
INVALID_ARTIFACT=$(find "$RUN_ROOT" -type f ! -name '*.json' ! -name '*.txt' -print -quit)
[[ -z "$INVALID_ARTIFACT" ]] \
  || fail "unexpected non-JSON/text artifact: $INVALID_ARTIFACT"
echo "conditional_while_staged_gate=PASS" | tee "$RUN_ROOT/decision.txt"
echo "artifacts=$RUN_ROOT"
