#!/usr/bin/env bash
#SBATCH --job-name=fp32-exact-k4-probe
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:05:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "fp32-exact-k4-probe: $*" >&2
  exit 2
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean probe worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed probe commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the probe branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed remote ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
RUN_LABEL=${RUN_LABEL:-strict-fp32-exact-k4}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

# This must exist before the first Python process or CUDA context.
export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "probe worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD does not match MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref does not match MAPPERATORINATOR_COMMIT"

for tool in nvidia-smi scontrol squeue sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
)
[[ ${#GPU_ROWS[@]} -eq 1 ]] || die "expected one visible GPU"
[[ "${GPU_ROWS[0]}" == *"2080 Ti"*", 7.5" ]] \
  || die "expected one RTX 2080 Ti/SM75, got ${GPU_ROWS[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
fi

mkdir -p "$WORK/logs" "$RUN_ROOT"
trap 'find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 | sort -z | xargs -0 -r sha256sum > "$RUN_ROOT/sha256sums.txt"' EXIT

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$PYTHON" utils/probe_exact_k4_multinomial.py \
  --output "$RUN_ROOT/analysis.json" \
  --seeds 7,12345,987654,42 \
  --eos-positions 1,2,3,4 \
  > "$RUN_ROOT/analysis.txt"

[[ -s "$RUN_ROOT/analysis.json" ]] || die "probe JSON is missing or empty"
[[ -s "$RUN_ROOT/analysis.txt" ]] || die "probe text is missing or empty"
echo "strict_fp32_exact_k4=PASS"
