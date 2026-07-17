#!/usr/bin/env bash
#SBATCH --job-name=s55-inloop-e-probe
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §55 Track1 bank: cheap in-loop E probe before SALVALAI scout.
# Prefer §43 multi-song 2-layer + offline-aligned sampling (no structural processors).
# Unique TMPDIR / TORCH_EXTENSIONS. Tip stays 55949274/366.11. No merge. No §44 yet.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-integrator}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s55-inloop-e-probe-$JOB"
UNIQUE="$WORK/tmp/s55-eprobe-$JOB"
EXT="$WORK/torch_extensions/s55-eprobe-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_multsong.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
DUMP_RUN=${DUMP_RUN:-$WORK/runs/exact-rope-device-state-fp16-auth-49964133}
MAX_NEW=${MAX_NEW_TOKENS:-512}
GAMMA=${MAPPERATORINATOR_TURBO_GAMMA:-5}

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
export MAPPERATORINATOR_TURBO_GAMMA="$GAMMA"
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1
export MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE=1
export MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH=1
export MAPPERATORINATOR_TURBO_STEP_PROFILE=1
# Align with §43 offline probe (temp/top-p only).
export MAPPERATORINATOR_TURBO_STRUCTURAL_PROCESSORS=0
export S55_RUN_ROOT="$RUN_ROOT"

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=$GAMMA"
  echo "max_new=$MAX_NEW"
  echo "structural_processors=0"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "section=55"
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
  nvidia-smi -L
} | tee "$RUN_ROOT/preflight.txt"

"$ENV/bin/python" "$REPO/utils/s36_turbo_speculative_smoke.py" \
  --dump-run "$DUMP_RUN" \
  --audio "$AUDIO" \
  --draft-ckpt "$DRAFT_CKPT" \
  --out "$RUN_ROOT/smoke.json" \
  --precision fp16 \
  --max-new-tokens "$MAX_NEW"

"$ENV/bin/python" - <<'PY' | tee "$RUN_ROOT/summary.json"
import json
import os
from pathlib import Path

run_root = Path(os.environ["S55_RUN_ROOT"])
job = os.environ.get("SLURM_JOB_ID", "manual")
payload = json.loads((run_root / "smoke.json").read_text())
stats = (payload.get("turbo") or {}).get("stats") or {}
e = float(stats.get("turbo_E_accepted_per_verify") or stats.get("turbo_accepted_per_verify") or 0.0)
e_mean = float(stats.get("turbo_E_mean_per_cycle") or e)
e_med = float(stats.get("turbo_E_median_per_cycle") or e)
verify = int(stats.get("turbo_verify_steps") or 0)
gate = e >= 1.7
out = {
    "section": "55",
    "phase": "inloop_e_probe",
    "job_id": job,
    "commit": payload.get("commit"),
    "branch": payload.get("branch"),
    "draft_ckpt": payload.get("draft_ckpt"),
    "gamma": stats.get("turbo_gamma"),
    "max_new_tokens": payload.get("max_new_tokens"),
    "structural_processors": stats.get("turbo_structural_processors"),
    "verify_steps": verify,
    "generated_tokens": stats.get("generated_tokens"),
    "E_accepted_per_verify": e,
    "E_mean_per_cycle": e_mean,
    "E_median_per_cycle": e_med,
    "cycle_glue_ms_per_verify": stats.get("turbo_cycle_glue_ms_per_verify")
    or stats.get("turbo_host_gap_ms_per_verify"),
    "draft_chain_enabled": stats.get("turbo_draft_chain_enabled"),
    "path_hit_counters": stats.get("turbo_path_hit_counters"),
    "gate_E_ge_1_7": gate,
    "recommend_scout": gate,
    "campaign_tip": "55949274",
    "campaign_tip_auth_fp16_main_tps": 366.11,
    "note": (
        "§55 cheap in-loop E probe (multsong 2-layer, offline-aligned, γ=5). "
        "Scout only if E>=1.7. Tip unchanged. No merge."
    ),
}
print(json.dumps(out, indent=2))
(run_root / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
if not gate:
    raise SystemExit(
        f"§55 in-loop E={e:.3f} < 1.7 — STOP Track1 bank; hand to §53/§54"
    )
print("§55 in-loop E gate PASS — scout authorized", flush=True)
PY

echo DONE
