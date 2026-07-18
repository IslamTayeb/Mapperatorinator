#!/usr/bin/env bash
#SBATCH --job-name=s58-tiger-cverify
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §58 turbo-on-tiger first gate: c_verify(K)/q1 on tiger uniform CUDAGraphDecoder.
# Coordinate with §57b: prefer 2080 (A5000 dumps may be running). ≤2 GPU total.
# Unique TMPDIR / TORCH_EXTENSIONS. No 500 claim. Tip 55949274 frozen.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s58-tiger-cverify-$JOB"
UNIQUE="$WORK/tmp/s58-tiger-cverify-$JOB"
EXT="$WORK/torch_extensions/s58-tiger-cverify-$JOB"
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}

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
  echo "base_expect=d01cdd2799891f78c871ff5889cfbb6a661a9e46"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  nvidia-smi -L
  squeue -u "$USER" -o "%.18i %.2t %.40j %b" | head -20 || true
} | tee "$RUN_ROOT/preflight.txt"

set +e
"$ENV/bin/python" utils/s58_tiger_c_verify_scout.py \
  --config-name v32 \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/c_verify.json" \
  --ks 3,5 \
  --warmup 5 \
  --iters 40 \
  --overrides precision=fp16 use_server=false \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"
RC=$?
set -e

"$ENV/bin/python" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
root = Path("$RUN_ROOT")
cv = None
p = root / "c_verify.json"
if p.is_file():
    cv = json.loads(p.read_text())
out = {
    "section": 58,
    "track": "turbo-on-tiger",
    "job_id": "$JOB",
    "commit": "$(git rev-parse HEAD)",
    "rc": $RC,
    "best_k": (cv or {}).get("best_k"),
    "best_ratio": (cv or {}).get("best_ratio"),
    "q1_ms": ((cv or {}).get("q1") or {}).get("q1_step_ms"),
    "verify": (cv or {}).get("verify"),
    "gpu": (cv or {}).get("gpu"),
    "gate": "c_verify/q1 expect ~1.1–1.3×",
    "campaign_tip_frozen": "55949274",
    "note": "First gate only. No fused verify. No 500 claim.",
}
print(json.dumps(out, indent=2))
PY

if [[ $RC -ne 0 ]]; then
  echo "c_verify scout failed rc=$RC" >&2
  tail -n 60 "$RUN_ROOT/stderr.txt" >&2 || true
  exit "$RC"
fi
echo DONE
