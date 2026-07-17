#!/usr/bin/env bash
#SBATCH --job-name=s41-verify-fastpath
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §41 budget-gate: c_verify(K) vs optimized Q=1 step on 2080 Ti.
# Unique TMPDIR / TORCH_EXTENSIONS. Not a 500 claim / full reciprocal.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-verify-fastpath}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s41-verify-fastpath-$JOB"
UNIQUE="$WORK/tmp/s41-verify-$JOB"
EXT="$WORK/torch_extensions/s41-verify-$JOB"
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
OSU=${MAPPERATORINATOR_OSU:-}

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
export MAPPERATORINATOR_AUDIO="$AUDIO"

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

# Prefer tip FP16 dump .osu when present.
if [[ -z "$OSU" ]]; then
  mapfile -t osus < <(find "$WORK/runs/shared-rope-device-state-fp16-auth-49964133" -name 'beatmap*.osu' 2>/dev/null | sort)
  if [[ ${#osus[@]} -ge 1 ]]; then
    OSU="${osus[0]}"
  fi
fi

EXTRA=()
if [[ -n "$OSU" ]]; then
  EXTRA+=(--osu "$OSU")
fi

# Eager StaticCache path first (budget gate). Optional graph job can follow.
"$ENV/bin/python" utils/s41_verify_fastpath_microbench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/microbench_eager.json" \
  --ks 4,5,8 \
  --warmup 5 \
  --iters 40 \
  "${EXTRA[@]}" \
  | tee "$RUN_ROOT/stdout_eager.txt"

# Per-K CUDA graph path (same job; still budget microbench).
"$ENV/bin/python" utils/s41_verify_fastpath_microbench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/microbench_cudagraph.json" \
  --ks 4,5,8 \
  --warmup 5 \
  --iters 40 \
  --cuda-graph-verify \
  "${EXTRA[@]}" \
  | tee "$RUN_ROOT/stdout_cudagraph.txt"

"$ENV/bin/python" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
root = Path("$RUN_ROOT")
eager = json.loads((root / "microbench_eager.json").read_text())
graph = json.loads((root / "microbench_cudagraph.json").read_text())
out = {
    "section": 41,
    "job_id": "$JOB",
    "commit": eager.get("commit"),
    "optimized_q1_step_ms_eager_run": eager["optimized_q1_step"]["q1_step_ms"],
    "optimized_q1_step_ms_graph_run": graph["optimized_q1_step"]["q1_step_ms"],
    "eager": eager["verify"],
    "cudagraph": graph["verify"],
    "best_ratio_eager": eager.get("best_ratio"),
    "best_ratio_cudagraph": graph.get("best_ratio"),
    "gate_pass_eager": eager.get("all_gate_pass"),
    "gate_pass_cudagraph": graph.get("all_gate_pass"),
    "target_ratio": 1.2,
    "campaign_tip": "55949274",
    "campaign_tip_fp16_main_tps": 366.11,
    "note": "Budget-gate microbench only. No 500 / full reciprocal.",
}
print(json.dumps(out, indent=2))
PY

echo DONE
