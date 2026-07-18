#!/usr/bin/env bash
#SBATCH --job-name=s59-t1-rails
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §59 T1 safety rails smoke: SDPA force + loud capture + cache eviction + CFG pad asserts.
# Unique TMPDIR / TORCH_EXTENSIONS. ≤2 GPU total with other live work.
# No 500 claim. Tip 55949274 frozen. No push to PR #120.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s59-t1-rails-$JOB"
UNIQUE="$WORK/tmp/s59-t1-rails-$JOB"
EXT="$WORK/torch_extensions/s59-t1-rails-$JOB"

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
# Default T1 posture: no silent capture fallback.
unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK || true

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  nvidia-smi -L
  squeue -u "$USER" -o "%.18i %.2t %.40j %b" | head -20 || true
} | tee "$RUN_ROOT/preflight.txt"

set +e
"$ENV/bin/python" utils/s59_t1_safety_rails_scout.py \
  --out "$RUN_ROOT/asserts.json" \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"
RC=$?
set -e

"$ENV/bin/python" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
root = Path("$RUN_ROOT")
asserts = None
p = root / "asserts.json"
if p.is_file():
    asserts = json.loads(p.read_text())
out = {
    "section": 59,
    "track": "t1-safety-rails",
    "job_id": "$JOB",
    "commit": "$(git rev-parse HEAD)",
    "rc": $RC,
    "ok": bool((asserts or {}).get("ok")),
    "decision": (asserts or {}).get("decision"),
    "checks": list(((asserts or {}).get("checks") or {}).keys()),
    "campaign_tip_frozen": "55949274",
    "note": "T1 rails smoke only. No 500 claim. Local vs PR#120.",
}
print(json.dumps(out, indent=2))
PY

if [[ $RC -ne 0 ]]; then
  echo "s59 t1 rails scout failed rc=$RC" >&2
  tail -n 80 "$RUN_ROOT/stderr.txt" >&2 || true
  exit "$RC"
fi
echo DONE
