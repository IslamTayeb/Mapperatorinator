#!/usr/bin/env bash
#SBATCH --job-name=self-out-residual-component
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err
# Keep clear of authoritative confirmation pins.
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-9

set -euo pipefail

die() {
  echo "self-out-residual-component: $*" >&2
  exit 2
}

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the branch}
REMOTE=${MAPPERATORINATOR_REMOTE:-islamtayeb}
PRECISION=${PRECISION:-${MAPPERATORINATOR_PRECISION:-}}
[[ -n "$PRECISION" ]] || die "Set PRECISION or MAPPERATORINATOR_PRECISION to fp16 or fp32"
case "$PRECISION" in
  fp16|fp32) ;;
  *) die "PRECISION must be fp16 or fp32, got $PRECISION" ;;
esac

ALLOW_PARALLEL=${ALLOW_PARALLEL:-${MAPPERATORINATOR_ALLOW_PARALLEL:-1}}
RUN_LABEL=${RUN_LABEL:-self-out-residual-component-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in "$PYTHON" nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    die "missing required tool: $tool"
  fi
done
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  git -C "$REPO" status --short >&2
  die "scout worktree is dirty"
}
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree differs from MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree differs from MAPPERATORINATOR_BRANCH"
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  die "pushed $REMOTE branch does not match MAPPERATORINATOR_COMMIT"
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
# Unique per-job native build + temp dirs (ALLOW_PARALLEL-safe).
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/self-out-residual-$PRECISION-$COMMIT-${SLURM_JOB_ID:-manual}"
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" && -w "${SLURM_TMPDIR}" ]]; then
  export TMPDIR="$SLURM_TMPDIR/self-out-residual-$PRECISION"
elif [[ -d /tmp && -w /tmp ]]; then
  export TMPDIR="/tmp/mapperatorinator-self-out-residual-$PRECISION-${SLURM_JOB_ID:-manual}-$$"
else
  export TMPDIR="$WORK/tmp/self-out-residual-$PRECISION-${SLURM_JOB_ID:-manual}"
fi
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_CAPABILITY=$($PYTHON -c 'import torch; print("".join(map(str, torch.cuda.get_device_capability())))')
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" || "$GPU_CAPABILITY" != 75 ]]; then
  die "component requires RTX 2080 Ti SM75, got $GPU_NAME sm_$GPU_CAPABILITY"
fi

if [[ "$ALLOW_PARALLEL" != "1" ]]; then
  mapfile -t OTHER_GPU_JOBS < <(
    squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
      | awk -F'|' -v current="${SLURM_JOB_ID:-}" \
          '$1 != current && tolower($2) ~ /gpu/ { print }'
  )
  [[ ${#OTHER_GPU_JOBS[@]} -eq 0 ]] || {
    printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
    die "another GPU job is running or pending (set ALLOW_PARALLEL=1 to opt in)"
  }
fi

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote=$REMOTE"
  echo "precision=$PRECISION"
  echo "allow_parallel=$ALLOW_PARALLEL"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "run_root=$RUN_ROOT"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "tmpdir=$TMPDIR"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
set +e
"$PYTHON" utils/profile_self_out_residual_component.py \
  --report-path "$RUN_ROOT/component.json" \
  --precision "$PRECISION" \
  --warmup 100 \
  --iters 1000 \
  --seed 12345
STATUS=$?
set -e
echo "component_exit=$STATUS"
if [[ -f "$RUN_ROOT/component.json" ]]; then
  echo "wrote $RUN_ROOT/component.json"
  echo "HARVEST: summary.best_projected_main_saving_seconds / any_component_pass / stop_condition / evidence_mode=component_projection"
  sha256sum "$RUN_ROOT/component.json" || true
fi
exit "$STATUS"
