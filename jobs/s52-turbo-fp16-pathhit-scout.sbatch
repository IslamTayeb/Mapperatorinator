#!/usr/bin/env bash
#SBATCH --job-name=s52-turbo-fp16-scout
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §52 decision (A): §47 keep-KV + §49 draft chain (+ §48 graph-native verify, no grind).
# One FP16 SALVALAI instrumented scout. Mandatory path-hit counters.
# Unique TMPDIR / TORCH_EXTENSIONS. Tip stays 55949274/366.11. No 500 claim.

set -euo pipefail
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-integrator}
WORK=/work/imt11/Mapperatorinator
JOB=${SLURM_JOB_ID:-manual}
RUN_ROOT="$WORK/runs/s52-turbo-fp16-scout-$JOB"
UNIQUE="$WORK/tmp/s52-perf-$JOB"
EXT="$WORK/torch_extensions/s52-perf-$JOB"
DRAFT_CKPT=${DRAFT_CKPT:-$WORK/runs/s43-draft-quality-50147299/draft_1layer.pt}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
AUTH_CEILING_E2=${AUTH_CEILING_E2:-461.68}
AUTH_CEILING_E24=${AUTH_CEILING_E24:-554.02}

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
export MAPPERATORINATOR_TURBO_VERIFY_GRAPH_NATIVE=1
export MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH=1
export MAPPERATORINATOR_TURBO_STEP_PROFILE=1
export MAPPERATORINATOR_PROFILE_OUTPUT="$RUN_ROOT/output"
export AUTH_CEILING_E2="$AUTH_CEILING_E2"
export AUTH_CEILING_E24="$AUTH_CEILING_E24"

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
  echo "section=52"
  echo "campaign_tip=55949274"
  echo "auth_fp16_main_tps=366.11"
  echo "auth_ceiling_e2=$AUTH_CEILING_E2"
  echo "auth_ceiling_e24=$AUTH_CEILING_E24"
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

acc = med([num(w.get("turbo_accepted_per_verify")) for w in steady])
phase = med([num(w.get("turbo_phase_ms_per_token")) for w in steady])
draft_ms = med([num(w.get("turbo_draft_ms_per_verify")) for w in steady])
verify_ms = med([num(w.get("turbo_verify_ms_per_verify")) for w in steady])
rebuild_ms = med([num(w.get("turbo_rebuild_ms_per_verify")) for w in steady])
host_ms = med([num(w.get("turbo_host_gap_ms_per_verify")) for w in steady])
ms_cycle = None
if phase is not None and acc:
    ms_cycle = float(phase) * float(acc)
elif draft_ms is not None and verify_ms is not None:
    ms_cycle = float(draft_ms) + float(verify_ms) + float(rebuild_ms or 0.0)

fwd = med([num(w.get("turbo_teacher_forwards_per_cycle")) for w in steady])
# Aggregate path hits from last window (session-cumulative verify/draft counters).
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
prep_calls = int(hits.get("verify_prepare_inputs_calls") or 0)
kv_mode = last.get("turbo_kv_commit_mode")
path_ok = (
    chain_replays > 0
    and native_replays > 0
    and kv_mode == "keep_accepted_o1_rewind"
)
e_runtime = float(acc) if acc is not None else None
ceiling_falsified = False
if e_runtime is not None and e_runtime < 1.7:
    ceiling_falsified = True
if not path_ok:
    ceiling_falsified = True

recommend_s44 = bool(main_tps is not None and float(main_tps) >= 384.0)
stop_diagnose = bool(main_tps is not None and float(main_tps) < 200.0)

out = {
    "section": "52",
    "track": "C",
    "decision": "A",
    "precision": "fp16",
    "engine": "turbo",
    "preset": "turbo-integrator-s52-kv-dg-v1",
    "draft_ckpt": os.environ.get("MAPPERATORINATOR_TURBO_DRAFT_CKPT"),
    "gamma": 3,
    "tree_k": 1,
    "main_tps": main_tps,
    "main_wall_seconds": main_wall,
    "generated_tokens": main_tokens,
    "n_windows": len(windows),
    "n_steady_windows": len(steady),
    "median_accepted_per_verify": acc,
    "median_phase_ms_per_token": phase,
    "median_ms_per_cycle": ms_cycle,
    "median_draft_ms_per_verify": draft_ms,
    "median_verify_ms_per_verify": verify_ms,
    "median_rebuild_ms_per_verify": rebuild_ms,
    "median_host_ms_per_verify": host_ms,
    "median_teacher_forwards_per_cycle": fwd,
    "kv_mode_last": kv_mode,
    "path_hit_counters_last": hits,
    "path_ok": path_ok,
    "authorizing_ceiling_e2": auth_e2,
    "authorizing_ceiling_e24": auth_e24,
    "ceiling_falsified": ceiling_falsified,
    "recommend_s44": recommend_s44,
    "stop_diagnose": stop_diagnose,
    "campaign_tip": "55949274",
    "campaign_tip_auth_fp16_main_tps": 366.11,
    "note": (
        "§52 integrator path-hit scout (keep-KV + draft chain + graph-native verify). "
        "No 500 claim. Tip unchanged 55949274/366.11. No §44 unless main_tps>=384."
    ),
}
print(json.dumps(out, indent=2))
# Soft exit: always write summary; structural miss is recorded not fatal for Slurm.
if main_tps is None:
    raise SystemExit("§52 scout missing main_tps")
PY

echo DONE
