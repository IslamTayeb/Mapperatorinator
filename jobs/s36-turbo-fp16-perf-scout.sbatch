#!/usr/bin/env bash
#SBATCH --job-name=s36-turbo-fp16-scout
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# FP16 turbo speculative perf scout toward 500 TPS.
# Not authoritative for ship until TIER1 pack passes. Unique TMPDIR/EXT.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-tiny-draft}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s36-turbo-fp16-scout-$JOB"
UNIQUE="$WORK/tmp/s36-perf-$JOB"
EXT="$WORK/torch_extensions/s36-perf-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s37-tiny-draft-train-50146289/draft_train.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}

mkdir -p "$RUN_ROOT/output" "$UNIQUE" "$EXT" "$WORK/logs"
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
export MAPPERATORINATOR_PROFILE_OUTPUT="$RUN_ROOT/output"

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
} | tee "$RUN_ROOT/preflight.txt"

"$ENV/bin/python" inference.py \
  --config-name profile_salvalai \
  audio_path="$AUDIO" \
  output_path="$RUN_ROOT/output" \
  device=cuda \
  precision=fp16 \
  attn_implementation=sdpa \
  inference_engine=turbo \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  profile_inference=true \
  profile_detail_ranges=false \
  profile_cuda_capture=false \
  profile_pass_kind=untraced_control \
  do_sample=true \
  temperature=0.9 \
  top_p=0.9 \
  > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

mapfile -t profiles < <(find "$RUN_ROOT/output" -type f -name '*.profile.json' | sort)
echo "profiles=${#profiles[@]}" | tee -a "$RUN_ROOT/preflight.txt"
if [[ ${#profiles[@]} -lt 1 ]]; then
  echo "missing profile json" >&2
  tail -n 80 "$RUN_ROOT/stderr.txt" >&2 || true
  exit 2
fi

"$ENV/bin/python" - <<PY | tee "$RUN_ROOT/summary.json"
import json
from pathlib import Path
paths = [Path(p) for p in '''${profiles[*]}'''.split()]
rows = []
for p in paths:
    payload = json.loads(p.read_text())
    main = (
        (payload.get("summary") or {})
        .get("generation_by_label", {})
        .get("main_generation")
    )
    tps = tokens = elapsed = None
    if isinstance(main, dict):
        tps = main.get("tokens_per_second")
        tokens = main.get("generated_tokens")
        elapsed = main.get("elapsed_seconds")
    wall = (payload.get("summary") or {}).get("stage_wall_seconds", {}).get("main_generation")
    rows.append({
        "profile": str(p),
        "main_tps": tps,
        "generated_tokens": tokens,
        "elapsed_seconds": elapsed,
        "main_wall_seconds": wall,
        "turbo_accepted_per_verify": (
            main.get("turbo_accepted_per_verify") if isinstance(main, dict) else None
        ),
    })
best = max((r["main_tps"] for r in rows if r["main_tps"] is not None), default=None)
out = {
    "section": "36",
    "track": "C",
    "precision": "fp16",
    "engine": "turbo",
    "profiles": rows,
    "best_main_tps": best,
    "note": "Scout only. No 500 claim without TIER1 pack.",
    "campaign_tip_auth_fp16_main_tps": 366.11,
}
print(json.dumps(out, indent=2))
PY

echo DONE
