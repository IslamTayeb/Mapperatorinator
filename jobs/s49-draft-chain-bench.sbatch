#!/usr/bin/env bash
#SBATCH --job-name=s49-draft-chain
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §49 W-DG: γ=3 draft chain in ONE CUDA graph (Philox sample + embed feedback).
# Gate: ≤1.2 ms total; kill if >2 ms. Persistent caches. Not a 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-graphed-draft-chain}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s49-draft-chain-$JOB"
UNIQUE="$WORK/tmp/s49-draft-chain-$JOB"
EXT="$WORK/torch_extensions/s49-draft-chain-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
OSU_HINT=${OSU_HINT:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}
GAMMA=${GAMMA:-3}
TEMPERATURE=${TEMPERATURE:-0.9}

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
export MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=$GAMMA"
  echo "temperature=$TEMPERATURE"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  nvidia-smi -L
  squeue -u "$USER" || true
} | tee "$RUN_ROOT/preflight.txt"

OSU_PATH=$(find "$OSU_HINT" -type f -name 'beatmap*.osu' 2>/dev/null | head -n 1 || true)
OSU_ARGS=()
if [[ -n "${OSU_PATH}" ]]; then
  OSU_ARGS+=(--osu-hint "$OSU_PATH")
fi

set +e
"$ENV/bin/python" utils/s49_draft_chain_bench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --draft-ckpt "$DRAFT_CKPT" \
  --precision fp16 \
  --gamma "$GAMMA" \
  --temperature "$TEMPERATURE" \
  --prefixes 128,256 \
  --warmup 8 \
  --iters 40 \
  --output "$RUN_ROOT/summary.json" \
  "${OSU_ARGS[@]}" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"
RC=$?
set -e

cp -f "$RUN_ROOT/summary.json" "$RUN_ROOT/summary.json.bak" 2>/dev/null || true
echo "bench_rc=$RC" | tee -a "$RUN_ROOT/preflight.txt"
if [[ -f "$RUN_ROOT/summary.json" ]]; then
  python - <<PY
import json
from pathlib import Path
p = Path("$RUN_ROOT/summary.json")
s = json.loads(p.read_text())
print(
    f"ms_chain={s['ms_per_chain_median']:.4f} "
    f"ms_tok={s['ms_per_token_median']:.4f} "
    f"gate_pass={s['gate_pass']} kill={s['kill']} "
    f"decision={s['decision']}"
)
PY
fi
echo DONE | tee -a "$RUN_ROOT/preflight.txt"
exit "$RC"
