#!/usr/bin/env bash
#SBATCH --job-name=s58-tiger-specwin
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §58 STEP 2: wire speculative window on tiger (draft→verify_k→reject→keep-KV).
# Prefer 2080 for parity with c_verify 50192694. ≤2 GPU with §57b.
# Unique TMPDIR / TORCH_EXTENSIONS. Tip 55949274 frozen. No push to Tiger14n.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s58-tiger-specwin-$JOB"
UNIQUE="$WORK/tmp/s58-tiger-specwin-$JOB"
EXT="$WORK/torch_extensions/s58-tiger-specwin-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
GAMMA=${MAPPERATORINATOR_TURBO_GAMMA:-3}

mkdir -p "$RUN_ROOT" "$UNIQUE" "$EXT" "$WORK/logs"
export PATH="$ENV/bin:$PATH"
export PROBE_REPO="$REPO"
export HF_HOME="$WORK/hf_home"
export TRANSFORMERS_CACHE="$WORK/hf_home/transformers"
export HF_HUB_CACHE="$WORK/hf_home/hub"
export XDG_CACHE_HOME="$WORK/cache"
export TMPDIR="$UNIQUE"
export TORCH_EXTENSIONS_DIR="$EXT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=$GAMMA"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  echo "section=58"
  echo "step=2"
  echo "campaign_tip=55949274"
  nvidia-smi -L
  squeue -u "$USER" -o "%.18i %.2t %.40j %b" | head -20 || true
} | tee "$RUN_ROOT/preflight.txt"

set +e
"$ENV/bin/python" utils/s58_tiger_spec_window_scout.py \
  --repo "$REPO" \
  --python "$ENV/bin/python" \
  --audio "$AUDIO" \
  --run-root "$RUN_ROOT" \
  --draft-ckpt "$DRAFT_CKPT" \
  --precision fp16 \
  --gamma "$GAMMA" \
  --seed 12345 \
  --min-e 1.05 \
  > "$RUN_ROOT/scout_stdout.txt" 2> "$RUN_ROOT/scout_stderr.txt"
RC=$?
set -e

if [[ -f "$RUN_ROOT/summary.json" ]]; then
  cp "$RUN_ROOT/summary.json" "$RUN_ROOT/summary.copy.json"
  cat "$RUN_ROOT/summary.json"
else
  echo '{"section":58,"step":2,"rc":'"$RC"',"decision":"FAIL_no_summary"}' | tee "$RUN_ROOT/summary.json"
fi

if [[ $RC -ne 0 ]]; then
  echo "s58 spec window scout failed rc=$RC" >&2
  tail -n 80 "$RUN_ROOT/scout_stderr.txt" >&2 || true
  tail -n 80 "$RUN_ROOT/stderr.txt" >&2 || true
  exit "$RC"
fi
echo DONE
