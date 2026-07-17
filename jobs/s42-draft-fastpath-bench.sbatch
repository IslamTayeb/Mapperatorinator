#!/usr/bin/env bash
#SBATCH --job-name=s42-draft-fastpath
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §42 W3: measure 2-layer draft c_draft ms/token on optimized kernels + CUDA graph.
# Unique TMPDIR / TORCH_EXTENSIONS. Turbo-only. Not a 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-draft-fastpath}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s42-draft-fastpath-$JOB"
UNIQUE="$WORK/tmp/s42-draft-$JOB"
EXT="$WORK/torch_extensions/s42-draft-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s37-tiny-draft-train-50146289/draft_train.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
OSU_HINT=${OSU_HINT:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}

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
export MAPPERATORINATOR_TURBO_DRAFT_CKPT="$DRAFT_CKPT"
export MAPPERATORINATOR_TURBO_DRAFT_FASTPATH=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  nvidia-smi -L
  squeue -u "$USER" || true
} | tee "$RUN_ROOT/preflight.txt"

# Resolve a concrete .osu under the tip auth dump if present.
OSU_PATH=$(find "$OSU_HINT" -type f -name 'beatmap*.osu' 2>/dev/null | head -n 1 || true)
OSU_ARGS=()
if [[ -n "${OSU_PATH}" ]]; then
  OSU_ARGS+=(--osu-hint "$OSU_PATH")
fi

"$ENV/bin/python" utils/s42_draft_fastpath_bench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --draft-ckpt "$DRAFT_CKPT" \
  --precision fp16 \
  --prefixes 128,256,512 \
  --warmup 10 \
  --iters 50 \
  --output "$RUN_ROOT/summary.json" \
  "${OSU_ARGS[@]}" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

cp -f "$RUN_ROOT/summary.json" "$RUN_ROOT/summary.json.bak" 2>/dev/null || true
echo DONE | tee -a "$RUN_ROOT/preflight.txt"
python - <<PY
import json
from pathlib import Path
p = Path("$RUN_ROOT/summary.json")
s = json.loads(p.read_text())
print(
    f"c_draft_ms={s['c_draft_ms_per_token_median']:.4f} "
    f"c_main_ms={s['c_main_ms_per_token_median']:.4f} "
    f"ratio={s['ratio_vs_measured_main']:.4f} "
    f"abs_pass={s['gate_pass_vs_absolute_ms']} "
    f"ratio_pass={s['gate_pass_vs_measured_main']}"
)
PY
