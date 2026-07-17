#!/usr/bin/env bash
#SBATCH --job-name=s48-graph-native-verify
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §48 W-VG: in-loop c_verify for graph-native K=γ verify.
# Unique TMPDIR / TORCH_EXTENSIONS. Coordinate ≤2 concurrent GPU with siblings.
# Not a 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-graph-native-verify}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s48-graph-native-verify-$JOB"
UNIQUE="$WORK/tmp/s48-verify-$JOB"
EXT="$WORK/torch_extensions/s48-verify-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
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
export MAPPERATORINATOR_TURBO_DRAFT_CKPT="$DRAFT_CKPT"
export MAPPERATORINATOR_TURBO_GAMMA=3
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1
export MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_TIMER=1
export MAPPERATORINATOR_TURBO_STEP_PROFILE=1

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
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
  nvidia-smi -L
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
} | tee "$RUN_ROOT/preflight.txt"

set +e
"$ENV/bin/python" utils/s48_graph_native_verify_inloop.py \
  --audio "$AUDIO" \
  --draft-ckpt "$DRAFT_CKPT" \
  --dump-run "$DUMP" \
  --out "$RUN_ROOT/inloop.json" \
  --max-new-tokens 256 \
  --gamma 3 \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"
RC=$?
set -e

if [[ -f "$RUN_ROOT/inloop.json" ]]; then
  cp "$RUN_ROOT/inloop.json" "$RUN_ROOT/summary.json"
  "$ENV/bin/python" - <<PY | tee -a "$RUN_ROOT/preflight.txt"
import json
from pathlib import Path
p = Path("$RUN_ROOT/inloop.json")
d = json.loads(p.read_text())
print("decision=", d.get("decision"))
print("c_verify_ms=", d.get("c_verify_ms_inloop"))
print("ratio=", d.get("c_verify_ratio_vs_q1"))
print("path=", d.get("turbo_verify_path"))
print("prepare_inputs_calls=", d.get("prepare_inputs_calls"))
PY
else
  echo "missing inloop.json rc=$RC" | tee -a "$RUN_ROOT/preflight.txt"
  tail -n 80 "$RUN_ROOT/stderr.txt" || true
  exit 2
fi

echo DONE
exit $RC
