#!/usr/bin/env bash
#SBATCH --job-name=s55-turbo-fp16-scout
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §55 Track1 bank scout (ONLY after in-loop E≥1.7).
# merge base 44ab1f3e/d3cd6939 + draft chain + multsong 2-layer + offline-aligned.
# Mandatory: accepted/verify, NVTX draft/verify/accept/glue, graph hits, cycle-glue≤0.4ms.
# Unique TMPDIR / TORCH_EXTENSIONS. Tip stays 55949274/366.11. No merge.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-integrator}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s55-turbo-fp16-scout-$JOB"
UNIQUE="$WORK/tmp/s55-perf-$JOB"
EXT="$WORK/torch_extensions/s55-perf-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_multsong.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
AUTH_CEILING_E2=${AUTH_CEILING_E2:-461.68}
AUTH_CEILING_E24=${AUTH_CEILING_E24:-554.02}
# Authorizing projection band for E≥1.7 under keep-KV+DG (post-rung).
PROJ_LO=${PROJ_LO:-417}
PROJ_HI=${PROJ_HI:-436}

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
export AUTH_CEILING_E2="$AUTH_CEILING_E2"
export AUTH_CEILING_E24="$AUTH_CEILING_E24"
export PROJ_LO="$PROJ_LO"
export PROJ_HI="$PROJ_HI"
export S55_RUN_ROOT="$RUN_ROOT"

cd "$REPO"
{
  echo "host=$(hostname)"
  echo "job_id=$JOB"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "draft_ckpt=$DRAFT_CKPT"
  echo "gamma=5"
  echo "structural_processors=0"
  echo "tmpdir=$UNIQUE"
  echo "torch_extensions=$EXT"
  echo "audio=$AUDIO"
  echo "section=55"
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
  echo "auth_ceiling_e2=$AUTH_CEILING_E2"
  echo "auth_ceiling_e24=$AUTH_CEILING_E24"
  echo "proj_band=${PROJ_LO}-${PROJ_HI}"
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

paths = sorted(Path(os.environ["MAPPERATORINATOR_PROFILE_OUTPUT"]).glob("**/*.profile.json"))
auth_e2 = float(os.environ.get("AUTH_CEILING_E2", "461.68"))
auth_e24 = float(os.environ.get("AUTH_CEILING_E24", "554.02"))
proj_lo = float(os.environ.get("PROJ_LO", "417"))
proj_hi = float(os.environ.get("PROJ_HI", "436"))
proj_mid = 0.5 * (proj_lo + proj_hi)

def num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

windows = []
main_tps = None
main_wall = None
main_tokens = None
for p in paths:
    payload = json.loads(p.read_text())
    summary = payload.get("summary") or {}
    by_label = summary.get("generation_by_label") or {}
    main = by_label.get("main_generation")
    if isinstance(main, dict):
        main_tps = main.get("tokens_per_second", main_tps)
        main_tokens = main.get("generated_tokens", main_tokens)
        main_wall = main.get("elapsed_seconds", main_wall)
    wall = (summary.get("stage_wall_seconds") or {}).get("main_generation")
    if wall is not None:
        main_wall = wall
    gens = payload.get("generation") or []
    for g in gens:
        if not isinstance(g, dict):
            continue
        if not g.get("turbo_speculative") and g.get("turbo_kv_commit_mode") is None:
            continue
        windows.append(g)

steady = [w for w in windows if (w.get("turbo_verify_steps") or 0) >= 1]
if not steady:
    steady = windows

def med(vals):
    vals = [v for v in vals if v is not None]
    return (statistics.median(vals) if vals else None)

def mean(vals):
    vals = [v for v in vals if v is not None]
    return (statistics.mean(vals) if vals else None)

e_vals = [
    num(w.get("turbo_E_accepted_per_verify") or w.get("turbo_accepted_per_verify"))
    for w in steady
]
acc = med(e_vals)
acc_mean = mean(e_vals)
phase = med([num(w.get("turbo_phase_ms_per_token")) for w in steady])
draft_ms = med([num(w.get("turbo_draft_ms_per_verify")) for w in steady])
verify_ms = med([num(w.get("turbo_verify_ms_per_verify")) for w in steady])
rebuild_ms = med([num(w.get("turbo_rebuild_ms_per_verify")) for w in steady])
glue_ms = med([
    num(w.get("turbo_cycle_glue_ms_per_verify") or w.get("turbo_host_gap_ms_per_verify"))
    for w in steady
])
ms_cycle = None
if phase is not None and acc:
    ms_cycle = float(phase) * float(acc)
elif draft_ms is not None and verify_ms is not None:
    ms_cycle = float(draft_ms) + float(verify_ms) + float(rebuild_ms or 0.0)

fwd = med([num(w.get("turbo_teacher_forwards_per_cycle")) for w in steady])
last = steady[-1] if steady else {}
hits = dict(last.get("turbo_path_hit_counters") or {})
if not hits:
    hits = {
        "draft_chain_graph_replay": last.get("turbo_draft_chain_replays"),
        "draft_chain_graph_capture": last.get("turbo_draft_chain_captures"),
        "verify_graph_native_replays": last.get("turbo_verify_graph_native_replays"),
        "verify_graph_replays": last.get("turbo_verify_graph_replays"),
        "verify_prepare_inputs_calls": last.get("turbo_verify_prepare_inputs_calls"),
        "keep_accepted_o1_rewind_cycles": last.get(
            "turbo_keep_accepted_o1_rewind_cycles_window"
        ),
    }

chain_replays = int(hits.get("draft_chain_graph_replay") or 0)
native_replays = int(hits.get("verify_graph_native_replays") or 0)
kv_mode = last.get("turbo_kv_commit_mode")
path_ok = (
    chain_replays > 0
    and native_replays > 0
    and kv_mode == "keep_accepted_o1_rewind"
)
e_runtime = float(acc) if acc is not None else None
e_healthy = bool(e_runtime is not None and e_runtime >= 1.7)
ceiling_falsified = False
if e_runtime is not None and e_runtime < 1.7:
    ceiling_falsified = True
if not path_ok:
    ceiling_falsified = True

glue_ok = bool(glue_ms is not None and float(glue_ms) <= 0.4)
under_proj = False
if main_tps is not None and e_healthy:
    # >8% under 417–436 band mid → STOP diagnose wall
    under_proj = float(main_tps) < 0.92 * proj_mid

recommend_s44 = bool(
    main_tps is not None and float(main_tps) >= 384.0 and e_healthy and path_ok
)
graduate_turbo = bool(recommend_s44)
stop_diagnose = bool(
    (main_tps is not None and float(main_tps) < 200.0)
    or (e_healthy and under_proj)
    or (glue_ms is not None and float(glue_ms) > 0.4 and e_healthy)
)

# Per-window E log (compact).
per_window_e = [
    {
        "i": i,
        "E": num(w.get("turbo_E_accepted_per_verify") or w.get("turbo_accepted_per_verify")),
        "verify_steps": w.get("turbo_verify_steps"),
        "context_type": w.get("context_type"),
    }
    for i, w in enumerate(steady)
]

out = {
    "section": "55",
    "phase": "fp16_pathhit_scout",
    "track": "1",
    "precision": "fp16",
    "engine": "turbo",
    "preset": "turbo-integrator-s55-multsong-offline-v1",
    "draft_ckpt": os.environ.get("MAPPERATORINATOR_TURBO_DRAFT_CKPT"),
    "gamma": 5,
    "tree_k": 1,
    "structural_processors": False,
    "main_tps": main_tps,
    "main_wall_seconds": main_wall,
    "generated_tokens": main_tokens,
    "n_windows": len(windows),
    "n_steady_windows": len(steady),
    "median_accepted_per_verify": acc,
    "mean_accepted_per_verify": acc_mean,
    "per_window_E": per_window_e,
    "median_phase_ms_per_token": phase,
    "median_ms_per_cycle": ms_cycle,
    "median_draft_ms_per_verify": draft_ms,
    "median_verify_ms_per_verify": verify_ms,
    "median_rebuild_ms_per_verify": rebuild_ms,
    "median_cycle_glue_ms_per_verify": glue_ms,
    "cycle_glue_gate_le_0_4": glue_ok,
    "median_teacher_forwards_per_cycle": fwd,
    "kv_mode_last": kv_mode,
    "path_hit_counters_last": hits,
    "path_ok": path_ok,
    "authorizing_ceiling_e2": auth_e2,
    "authorizing_ceiling_e24": auth_e24,
    "projection_band": [proj_lo, proj_hi],
    "under_projection_8pct": under_proj,
    "ceiling_falsified": ceiling_falsified,
    "e_healthy": e_healthy,
    "recommend_s44": recommend_s44,
    "graduate_turbo_tip": graduate_turbo,
    "stop_diagnose": stop_diagnose,
    "campaign_tip": "55949274",
    "campaign_tip_auth_fp16_main_tps": 366.11,
    "note": (
        "§55 instrumented FP16 scout (multsong + offline-aligned + chain top-p). "
        "No 500 claim. Tip 55949274/366.11 unchanged unless graduate_turbo_tip. "
        "§44 only if main_tps>=384 and E healthy."
    ),
}
print(json.dumps(out, indent=2))
Path(os.environ["S55_RUN_ROOT"], "summary.json").write_text(json.dumps(out, indent=2) + "\n")
if main_tps is None:
    raise SystemExit("§55 scout missing main_tps")
if under_proj and e_healthy:
    raise SystemExit(
        f"§55 STOP diagnose wall: main_tps={main_tps:.2f} >8% under proj mid {proj_mid:.1f} with E>={1.7}"
    )
PY

echo DONE
