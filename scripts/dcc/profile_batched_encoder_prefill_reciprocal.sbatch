#!/usr/bin/env bash
# W3 Tiger B16 batched encoder prefill reciprocal vs tip optimized (FP16 2080 Ti).
# Documented-drift scout only. No V32/server change. No 500 claim. No tip demotion.
#SBATCH --job-name=w3-b16-encoder-prefill
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "w3-batched-encoder-prefill: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean W3 worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed W3 commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the W3 branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed W3 remote ref}
PRECISION=${MAPPERATORINATOR_PRECISION:-fp16}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
BATCH_SIZE=${MAPPERATORINATOR_ENCODER_BATCH_SIZE:-16}
STORAGE=${MAPPERATORINATOR_ENCODER_STORAGE:-cpu}
TIP_AUTH_TPS=${MAPPERATORINATOR_TIP_AUTH_TPS:-366.11}
MIN_WALL_GAIN_PCT=${MAPPERATORINATOR_MIN_WALL_GAIN_PCT:-5}
MAX_TPS_REGRESS_PCT=${MAPPERATORINATOR_MAX_TPS_REGRESS_PCT:-1}
MAX_TOKEN_MISMATCH_FRACTION=${MAPPERATORINATOR_MAX_TOKEN_MISMATCH_FRACTION:-0}
REQUIRE_OSU_EQUAL=${MAPPERATORINATOR_REQUIRE_OSU_EQUAL:-1}
RUN_LABEL=${RUN_LABEL:-w3-b16-encoder-prefill-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
JOB_TMP="$WORK/tmp/w3-${SLURM_JOB_ID}"
JOB_EXT="$WORK/torch_extensions/w3-${SLURM_JOB_ID}-${PRECISION}"

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] \
  || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
[[ "$PRECISION" == "fp16" || "$PRECISION" == "fp32" ]] \
  || die "MAPPERATORINATOR_PRECISION must be fp16 or fp32"
[[ "$STORAGE" == "cpu" || "$STORAGE" == "cuda" ]] \
  || die "MAPPERATORINATOR_ENCODER_STORAGE must be cpu or cuda"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "candidate worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD differs from MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree branch differs from MAPPERATORINATOR_BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref differs from MAPPERATORINATOR_COMMIT"

mkdir -p "$WORK/logs" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/cache/huggingface" "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export TMPDIR="$JOB_TMP"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected exactly one visible GPU"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] || {
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
}
if [[ ${#OTHER_GPU_JOBS[@]} -ne 0 ]]; then
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  echo "parallel_reciprocal_opt_in=1" >&2
fi

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "precision=$PRECISION"
  echo "batch_size=$BATCH_SIZE"
  echo "storage=$STORAGE"
  echo "tip_auth_tps=$TIP_AUTH_TPS"
  echo "min_wall_gain_pct=$MIN_WALL_GAIN_PCT"
  echo "max_tps_regress_pct=$MAX_TPS_REGRESS_PCT"
  echo "max_token_mismatch_fraction=$MAX_TOKEN_MISMATCH_FRACTION"
  echo "require_osu_equal=$REQUIRE_OSU_EQUAL"
  echo "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL"
  echo "run_root=$RUN_ROOT"
  echo "tmpdir=$TMPDIR"
  echo "torch_extensions=$TORCH_EXTENSIONS_DIR"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"
COMMON_ARGS=(
  --config-name profile_salvalai
  audio_path="$AUDIO"
  device=cuda
  precision="$PRECISION"
  attn_implementation=sdpa
  inference_engine=optimized
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_detail_ranges=false
  profile_cuda_capture=false
  profile_pass_kind=untraced_control
)

declare -A PROFILE_PATHS
declare -A OSU_PATHS
declare -A EVIDENCE_PATHS
run_role() {
  local role=$1
  local kind=$2
  local run_dir="$RUN_ROOT/$role"
  mkdir -p "$run_dir/output"
  local evidence="$run_dir/evidence.json"
  local command=(
    "$PYTHON" utils/run_batched_encoder_prefill_candidate.py
    --mode "$kind"
    --evidence-manifest "$evidence"
    --batch-size "$BATCH_SIZE"
    --storage "$STORAGE"
    -- "${COMMON_ARGS[@]}" output_path="$run_dir/output"
  )
  printf '%q ' "${command[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "$role produced ${#profiles[@]} profiles"
  local profile=${profiles[0]}
  local osu=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$osu"
  require_nonempty "$evidence"
  "$PYTHON" - "$evidence" "$kind" "$BATCH_SIZE" "$STORAGE" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
role = sys.argv[2]
batch_size = int(sys.argv[3])
storage = sys.argv[4]
assert payload["schema_version"] == 1
assert payload["role"] == role
assert payload["candidate"] == "tiger_b16_batched_encoder_prefill"
assert payload["enabled"] is (role == "candidate")
assert payload["batch_size"] == batch_size
assert payload["storage"] == storage
assert payload["result_class"] == "documented-drift"
assert payload["production_default"] is False
assert payload["tiger_pr"] == 120
if role == "candidate":
    evidence = payload["prefill_evidence"]
    assert isinstance(evidence, dict)
    assert evidence.get("labels_completed") == ["timing_context", "main_generation"]
    assert int(evidence.get("store_count", 0)) >= 1
    assert evidence.get("batch_size") == batch_size
    assert evidence.get("storage") == storage
else:
    assert payload["prefill_evidence"] is None
for key in ("rng_after_seed", "rng_after_inference"):
    rng = payload[key]
    assert set(rng) == {
        "numpy_sha256",
        "python_sha256",
        "torch_cpu_sha256",
        "torch_cuda_sha256",
    }
    assert all(
        isinstance(rng[name], str) and len(rng[name]) == 64
        for name in ("numpy_sha256", "python_sha256", "torch_cpu_sha256")
    )
    assert isinstance(rng["torch_cuda_sha256"], list) and rng["torch_cuda_sha256"]
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in rng["torch_cuda_sha256"]
    )
PY
  PROFILE_PATHS[$role]=$profile
  OSU_PATHS[$role]=$osu
  EVIDENCE_PATHS[$role]=$evidence
}

run_role baseline_first baseline
run_role candidate_first candidate
run_role candidate_second candidate
run_role baseline_second baseline

"$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
  --baseline-first-profile "${PROFILE_PATHS[baseline_first]}" \
  --baseline-first-osu "${OSU_PATHS[baseline_first]}" \
  --candidate-first-profile "${PROFILE_PATHS[candidate_first]}" \
  --candidate-first-osu "${OSU_PATHS[candidate_first]}" \
  --candidate-second-profile "${PROFILE_PATHS[candidate_second]}" \
  --candidate-second-osu "${OSU_PATHS[candidate_second]}" \
  --baseline-second-profile "${PROFILE_PATHS[baseline_second]}" \
  --baseline-second-osu "${OSU_PATHS[baseline_second]}" \
  --mode relaxed \
  --json-output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"
require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"

"$PYTHON" - \
  "$RUN_ROOT/analysis.json" \
  "${EVIDENCE_PATHS[candidate_first]}" \
  "${EVIDENCE_PATHS[candidate_second]}" \
  "$TIP_AUTH_TPS" \
  "$MIN_WALL_GAIN_PCT" \
  "$MAX_TPS_REGRESS_PCT" \
  "$MAX_TOKEN_MISMATCH_FRACTION" \
  "$REQUIRE_OSU_EQUAL" \
  "$RUN_ROOT/gate.json" <<'PY'
import json
import pathlib
import sys

analysis_path = pathlib.Path(sys.argv[1])
cand1 = json.loads(pathlib.Path(sys.argv[2]).read_text())
cand2 = json.loads(pathlib.Path(sys.argv[3]).read_text())
tip_auth_tps = float(sys.argv[4])
min_wall_gain_pct = float(sys.argv[5])
max_tps_regress_pct = float(sys.argv[6])
max_token_mismatch_fraction = float(sys.argv[7])
require_osu_equal = sys.argv[8] == "1"
gate_path = pathlib.Path(sys.argv[9])

report = json.loads(analysis_path.read_text())
assert report["mode"] == "relaxed"
assert report["parity"]["baseline_repeat_stable"] is True
assert report["parity"]["candidate_repeat_stable"] is True

wall = report["metrics"]["complete_request_wall_seconds"]
main_tps = report["metrics"]["main_tps"]
baseline_wall = float(wall["baseline_median"])
candidate_wall = float(wall["candidate_median"])
baseline_tps = float(main_tps["baseline_median"])
candidate_tps = float(main_tps["candidate_median"])
wall_gain_pct = (
    (baseline_wall - candidate_wall) / baseline_wall * 100.0 if baseline_wall > 0 else 0.0
)
tps_vs_baseline_pct = (
    (candidate_tps - baseline_tps) / baseline_tps * 100.0 if baseline_tps > 0 else 0.0
)
tps_vs_tip_auth_pct = (
    (candidate_tps - tip_auth_tps) / tip_auth_tps * 100.0 if tip_auth_tps > 0 else 0.0
)

token_div = report["parity"]["token_and_stopping_divergence"]
max_mismatch = max(
    float(token_div[label]["aligned_mismatch_fraction"]) for label in token_div
)
osu_equal = bool(report["parity"]["output_divergence"]["final_map_equal"])
cross_exact = bool(report["parity"]["cross_candidate_exact"])

failures = []
if max_mismatch > max_token_mismatch_fraction:
    failures.append(
        f"token mismatch fraction {max_mismatch:.6f} > {max_token_mismatch_fraction}"
    )
if require_osu_equal and not osu_equal:
    failures.append("final .osu bytes drifted beyond documented-drift policy")
if wall_gain_pct + 1e-12 < min_wall_gain_pct:
    failures.append(
        f"complete-request wall gain {wall_gain_pct:.3f}% < {min_wall_gain_pct}%"
    )
if tps_vs_baseline_pct < -max_tps_regress_pct:
    failures.append(
        f"main TPS vs same-job baseline regress {tps_vs_baseline_pct:.3f}% "
        f"< -{max_tps_regress_pct}%"
    )

decision = "PROMOTE_SCOUT" if not failures else "STOP"
gate = {
    "decision": decision,
    "result_class": "documented-drift",
    "production_default": False,
    "tip_auth_tps": tip_auth_tps,
    "baseline_complete_request_wall_seconds": baseline_wall,
    "candidate_complete_request_wall_seconds": candidate_wall,
    "complete_request_wall_gain_pct": wall_gain_pct,
    "baseline_main_tps": baseline_tps,
    "candidate_main_tps": candidate_tps,
    "main_tps_vs_baseline_pct": tps_vs_baseline_pct,
    "main_tps_vs_tip_auth_pct": tps_vs_tip_auth_pct,
    "tip_sanity_note": (
        "Do not demote tip auth if same-job baseline under-reads 366.11; "
        "wall/TPS gates are vs same-job tip optimized baseline."
    ),
    "cross_candidate_exact": cross_exact,
    "max_token_mismatch_fraction": max_mismatch,
    "final_map_equal": osu_equal,
    "candidate_prefill_store_count_first": cand1["prefill_evidence"]["store_count"],
    "candidate_prefill_store_count_second": cand2["prefill_evidence"]["store_count"],
    "candidate_precompute_seconds_first": cand1["prefill_evidence"][
        "total_complete_precompute_seconds"
    ],
    "candidate_precompute_seconds_second": cand2["prefill_evidence"][
        "total_complete_precompute_seconds"
    ],
    "failures": failures,
    "min_wall_gain_pct": min_wall_gain_pct,
    "max_tps_regress_pct": max_tps_regress_pct,
}
gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(json.dumps(gate, indent=2, sort_keys=True))
if decision != "PROMOTE_SCOUT":
    raise SystemExit(f"W3 gate STOP: {'; '.join(failures)}")
print("w3_batched_encoder_prefill_reciprocal=PASS")
PY
