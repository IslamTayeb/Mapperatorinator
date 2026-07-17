#!/usr/bin/env bash
#SBATCH --job-name=s57b-turbo-dump
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §57b — multi-song short map_rest turbo dump (pegasus/lambada/…)。
# Matches §55 turbo FP16 recipe for short lookback rest windows (med gen≈7).
# Unique TMPDIR / TORCH_EXTENSIONS. Tip 55949274/366.11. No merge. No wire.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-integrator}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
SONG=${S57B_SONG:?set S57B_SONG=pegasus|lambada|...}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/five-song-profile/${SONG}.mp3}
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_multsong.pt}
RUN_ROOT="$WORK/runs/s57b-turbo-dump-${SONG}-$JOB"
UNIQUE="$WORK/tmp/s57b-dump-${SONG}-$JOB"
EXT="$WORK/torch_extensions/s57b-dump-${SONG}-$JOB"

if [[ ! -f "$AUDIO" ]]; then
  echo "ERROR: audio missing: $AUDIO" >&2
  exit 2
fi
if [[ ! -f "$DRAFT_CKPT" ]]; then
  echo "ERROR: draft ckpt missing: $DRAFT_CKPT" >&2
  exit 2
fi

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
export MAPPERATORINATOR_TURBO_GAMMA=5
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1
export MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE=1
export MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH=1
export MAPPERATORINATOR_TURBO_STEP_PROFILE=1
export MAPPERATORINATOR_TURBO_STRUCTURAL_PROCESSORS=0
export MAPPERATORINATOR_PROFILE_OUTPUT="$RUN_ROOT/output"
export S57B_RUN_ROOT="$RUN_ROOT"
export S57B_SONG="$SONG"

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "section=57b"
  echo "phase=turbo_short_rest_dump"
  echo "song=$SONG"
  echo "audio=$AUDIO"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=5"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
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

"$ENV/bin/python" - <<'PY' | tee "$RUN_ROOT/summary.json"
import json
import os
import statistics
from pathlib import Path

run = Path(os.environ["S57B_RUN_ROOT"])
song = os.environ["S57B_SONG"]
paths = sorted(Path(os.environ["MAPPERATORINATOR_PROFILE_OUTPUT"]).glob("**/*.profile.json"))
if not paths:
    raise SystemExit("no profile")

payload = json.loads(paths[0].read_text())
gens = [g for g in (payload.get("generation") or []) if isinstance(g, dict)]
by_role = {"map0": [], "map_rest": [], "timing": []}
for g in gens:
    ctx = str(g.get("context_type") or "")
    ids = g.get("generated_token_ids")
    if not (isinstance(ids, list) and ids and isinstance(ids[0], int)):
        continue
    n = int(g.get("generated_tokens") or len(ids))
    if ctx == "map":
        role = "map0" if int(g.get("sequence_index") or 0) == 0 else "map_rest"
    elif ctx == "timing":
        role = "timing"
    else:
        continue
    by_role[role].append(n)

def med(xs):
    return statistics.median(xs) if xs else None

summary = {
    "section": "57b",
    "phase": "turbo_short_rest_dump",
    "song_id": song,
    "profile": str(paths[0]),
    "n_generation_records": len(gens),
    "by_role": {
        role: {
            "n_windows": len(xs),
            "n_tokens": int(sum(xs)),
            "median_gen": med(xs),
        }
        for role, xs in by_role.items()
    },
    "campaign_tip": "55949274",
    "campaign_tip_auth_fp16_main_tps": 366.11,
    "note": (
        "Short map_rest turbo dump for §57b multi-song distill. "
        "Expect map_rest median_gen near s55≈7; not a TPS/500 claim."
    ),
}
print(json.dumps(summary, indent=2))
(run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
rest = summary["by_role"]["map_rest"]
if rest["n_windows"] < 8:
    raise SystemExit(f"too few map_rest windows: {rest}")
if rest["median_gen"] is None or rest["median_gen"] > 64:
    raise SystemExit(
        f"map_rest not short-continuation shaped: median_gen={rest['median_gen']}"
    )
print(f"OK song={song} map_rest n={rest['n_windows']} med={rest['median_gen']}")
PY

ln -sfn "$RUN_ROOT" "$WORK/runs/s57b-turbo-dump-${SONG}"
echo "STABLE $WORK/runs/s57b-turbo-dump-${SONG} -> $RUN_ROOT"
echo DONE
