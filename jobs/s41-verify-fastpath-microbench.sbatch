#!/usr/bin/env bash
#SBATCH --job-name=s41-verify-fastpath
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §41: (1) c_verify(K) vs Q=1 microbench  (2) graph-aligned canary probe @110
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

if [[ -z "$OSU" ]]; then
  mapfile -t osus < <(find "$DUMP" -name 'beatmap*.osu' 2>/dev/null | sort)
  if [[ ${#osus[@]} -ge 1 ]]; then
    OSU="${osus[0]}"
  fi
fi
EXTRA=()
if [[ -n "$OSU" ]]; then
  EXTRA+=(--osu "$OSU")
fi

set +e
"$ENV/bin/python" utils/s41_verify_fastpath_microbench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/microbench_eager.json" \
  --ks 4,5,8 \
  --warmup 5 \
  --iters 40 \
  "${EXTRA[@]}" \
  > "$RUN_ROOT/stdout_eager.txt" 2> "$RUN_ROOT/stderr_eager.txt"
EAGER_RC=$?

"$ENV/bin/python" utils/s41_verify_fastpath_microbench.py \
  --config-name profile_salvalai \
  --audio "$AUDIO" \
  --out "$RUN_ROOT/microbench_cudagraph.json" \
  --ks 4,5,8 \
  --warmup 5 \
  --iters 40 \
  --cuda-graph-verify \
  "${EXTRA[@]}" \
  > "$RUN_ROOT/stdout_cudagraph.txt" 2> "$RUN_ROOT/stderr_cudagraph.txt"
GRAPH_RC=$?

"$ENV/bin/python" utils/s41_graph_aligned_canary_probe.py \
  --audio "$AUDIO" \
  --dump-run "$DUMP" \
  --out "$RUN_ROOT/canary_probe.json" \
  > "$RUN_ROOT/stdout_canary.txt" 2> "$RUN_ROOT/stderr_canary.txt"
CANARY_RC=$?
set -e

"$ENV/bin/python" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
root = Path("$RUN_ROOT")

def load(name):
    p = root / name
    if not p.is_file():
        return None
    return json.loads(p.read_text())

eager = load("microbench_eager.json")
graph = load("microbench_cudagraph.json")
canary = load("canary_probe.json")
out = {
    "section": 41,
    "job_id": "$JOB",
    "commit": "$(git rev-parse HEAD)",
    "eager_rc": $EAGER_RC,
    "cudagraph_rc": $GRAPH_RC,
    "canary_rc": $CANARY_RC,
    "optimized_q1_step_ms": (eager or {}).get("optimized_q1_step", {}).get("q1_step_ms"),
    "eager": (eager or {}).get("verify"),
    "cudagraph": (graph or {}).get("verify"),
    "best_ratio_eager": (eager or {}).get("best_ratio"),
    "best_ratio_cudagraph": (graph or {}).get("best_ratio"),
    "gate_pass_eager": (eager or {}).get("all_gate_pass"),
    "gate_pass_cudagraph": (graph or {}).get("all_gate_pass"),
    "canary_path_status": (canary or {}).get("canary_path_status"),
    "canary_argmax_match": (canary or {}).get("argmax_match"),
    "canary": canary,
    "target_ratio": 1.2,
    "campaign_tip": "55949274",
    "campaign_tip_fp16_main_tps": 366.11,
    "note": "Budget-gate only. No 500 / full reciprocal. §40 STOP_ESCALATE owned here.",
}
print(json.dumps(out, indent=2))
PY

# Fail job if microbench crashed; canary FAIL is recorded but still surfaces.
if [[ $EAGER_RC -ne 0 && $GRAPH_RC -ne 0 ]]; then
  echo "both microbenches failed" >&2
  tail -n 40 "$RUN_ROOT/stderr_eager.txt" >&2 || true
  exit 2
fi
echo DONE
