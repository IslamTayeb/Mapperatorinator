#!/usr/bin/env bash
#SBATCH --job-name=s47-turbo-fp16-scout
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §47 W-KV sampled keep-accepted-KV FP16 scout.
# Gates: teacher_forwards/cycle==1; −10 ms/cycle vs §45 (~45 ms); path hits.
# Unique TMPDIR / TORCH_EXTENSIONS. No 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-keep-accepted-kv}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s47-turbo-fp16-scout-$JOB"
UNIQUE="$WORK/tmp/s47-perf-$JOB"
EXT="$WORK/torch_extensions/s47-perf-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
# §45 scout sustained median ~23 ms/tok → ~45 ms/cycle at E≈2; need −10 ms/cycle.
S45_MS_PER_CYCLE_REF=${S45_MS_PER_CYCLE_REF:-45.0}

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
export MAPPERATORINATOR_TURBO_GAMMA=3
export MAPPERATORINATOR_TURBO_VERIFY_FASTPATH=1
export MAPPERATORINATOR_TURBO_TEACHER_ALIGNED=1
export MAPPERATORINATOR_TURBO_VERIFY_CUDA_GRAPH=1
export MAPPERATORINATOR_TURBO_STEP_PROFILE=1
export MAPPERATORINATOR_PROFILE_OUTPUT="$RUN_ROOT/output"

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=3"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  echo "section=47"
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
  echo "s45_ms_per_cycle_ref=$S45_MS_PER_CYCLE_REF"
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

S45_REF="$S45_MS_PER_CYCLE_REF" "$ENV/bin/python" - <<'PY' | tee "$RUN_ROOT/summary.json"
import json
import os
from pathlib import Path

paths = sorted(Path(os.environ["MAPPERATORINATOR_PROFILE_OUTPUT"]).glob("**/*.profile.json"))
s45_ref = float(os.environ.get("S45_REF", "45.0"))
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
    def g(key, default=None):
        return main.get(key) if isinstance(main, dict) else default
    rows.append({
        "profile": str(p),
        "main_tps": tps,
        "generated_tokens": tokens,
        "elapsed_seconds": elapsed,
        "main_wall_seconds": wall,
        "turbo_accepted_per_verify": g("turbo_accepted_per_verify"),
        "turbo_kv_commit_mode": g("turbo_kv_commit_mode"),
        "turbo_canary_kv_mode": g("turbo_canary_kv_mode"),
        "turbo_teacher_forwards_per_cycle": g("turbo_teacher_forwards_per_cycle"),
        "turbo_teacher_extra_q1_per_cycle": g("turbo_teacher_extra_q1_per_cycle"),
        "turbo_teacher_accepted_reforwards_window": g(
            "turbo_teacher_accepted_reforwards_window"
        ),
        "turbo_draft_accepted_reforwards_window": g(
            "turbo_draft_accepted_reforwards_window"
        ),
        "turbo_rebuild_ms_per_verify": g("turbo_rebuild_ms_per_verify"),
        "turbo_verify_ms_per_verify": g("turbo_verify_ms_per_verify"),
        "turbo_draft_ms_per_verify": g("turbo_draft_ms_per_verify"),
        "turbo_phase_ms_per_token": g("turbo_phase_ms_per_token"),
        "turbo_verify_graph_captures": g("turbo_verify_graph_captures"),
        "turbo_verify_graph_replays": g("turbo_verify_graph_replays"),
        "turbo_verify_eager_calls": g("turbo_verify_eager_calls"),
    })

best = max((r["main_tps"] for r in rows if r["main_tps"] is not None), default=None)
# Prefer the main_generation row with turbo stats.
primary = next(
    (r for r in rows if r.get("turbo_kv_commit_mode") is not None),
    rows[0] if rows else {},
)
acc = primary.get("turbo_accepted_per_verify") or 0.0
phase_ms_tok = primary.get("turbo_phase_ms_per_token")
ms_cycle = None
if phase_ms_tok is not None and acc:
    ms_cycle = float(phase_ms_tok) * float(acc)
elif (
    primary.get("turbo_draft_ms_per_verify") is not None
    and primary.get("turbo_verify_ms_per_verify") is not None
):
    ms_cycle = (
        float(primary["turbo_draft_ms_per_verify"])
        + float(primary["turbo_verify_ms_per_verify"])
        + float(primary.get("turbo_rebuild_ms_per_verify") or 0.0)
    )

fwd = primary.get("turbo_teacher_forwards_per_cycle")
reforwards = primary.get("turbo_teacher_accepted_reforwards_window")
mode = primary.get("turbo_kv_commit_mode")
gate_fwd = fwd is not None and abs(float(fwd) - 1.0) < 1e-6
gate_no_reforward = reforwards is not None and int(reforwards) == 0
gate_mode = mode == "keep_accepted_o1_rewind"
gate_ms = ms_cycle is not None and (s45_ref - float(ms_cycle)) >= 10.0 - 1e-6
gates = {
    "teacher_forwards_per_cycle_eq_1": gate_fwd,
    "teacher_accepted_reforwards_eq_0": gate_no_reforward,
    "kv_mode_keep_accepted_o1_rewind": gate_mode,
    "minus_10_ms_per_cycle_vs_s45": gate_ms,
    "ms_per_cycle": ms_cycle,
    "s45_ms_per_cycle_ref": s45_ref,
    "ms_per_cycle_delta_vs_s45": (None if ms_cycle is None else s45_ref - float(ms_cycle)),
}
out = {
    "section": "47",
    "track": "C",
    "worker": "W-KV",
    "precision": "fp16",
    "engine": "turbo",
    "preset": "turbo-keep-accepted-kv-s47-v1",
    "draft_ckpt": os.environ.get("MAPPERATORINATOR_TURBO_DRAFT_CKPT"),
    "gamma": 3,
    "tree_k": 1,
    "profiles": rows,
    "best_main_tps": best,
    "gates": gates,
    "note": (
        "§47 W-KV scout. Sampled keep-accepted-KV is DOCUMENTED DRIFT under §34. "
        "No 500 claim. Tip unchanged 55949274/366.11."
    ),
    "campaign_tip": "55949274",
    "campaign_tip_auth_fp16_main_tps": 366.11,
}
print(json.dumps(out, indent=2))
if not (gate_fwd and gate_no_reforward and gate_mode):
    raise SystemExit("§47 structural gates MISS (forwards/reforward/mode)")
PY

echo DONE
