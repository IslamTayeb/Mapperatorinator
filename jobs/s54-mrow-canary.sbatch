#!/usr/bin/env bash
#SBATCH --job-name=s54-mrow-canary
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:40:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §54 Stage A: greedy TIER1a canary must stay unaffected (aligned Q=1 path).
# Unique TMPDIR/TORCH_EXTENSIONS. Coordinate ≤2 GPU with siblings.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-verify-fused-mrow}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s54-mrow-canary-$JOB"
UNIQUE="$WORK/tmp/s54-canary-$JOB"
EXT="$WORK/torch_extensions/s54-canary-$JOB"
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
DUMP=${MAPPERATORINATOR_DUMP_RUN:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}

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
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1
# m-row armed, but Q=1 canary must fall through to aligned q1 hooks.
export MAPPERATORINATOR_TURBO_VERIFY_MROW=1
# Q=1 canary falls through aligned hooks; keep Stage B off for clarity.
export MAPPERATORINATOR_TURBO_VERIFY_MROW_STAGE_B=0

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

"$ENV/bin/python" utils/s41_graph_aligned_canary_probe.py \
  --audio "$AUDIO" \
  --dump-run "$DUMP" \
  --out "$RUN_ROOT/canary_probe.json" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

cp -f "$RUN_ROOT/canary_probe.json" "$RUN_ROOT/summary.json"
"$ENV/bin/python" - <<PY | tee -a "$RUN_ROOT/preflight.txt"
import json
from pathlib import Path
d = json.loads(Path("$RUN_ROOT/canary_probe.json").read_text())
print("canary_decision=", d.get("decision") or d.get("pass") or d.get("status"))
print("keys=", sorted(d.keys())[:20])
PY
echo DONE
