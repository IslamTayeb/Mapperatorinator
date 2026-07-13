#!/usr/bin/env bash
#SBATCH --job-name=native-cross-mlp-reciprocal
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
BASELINE_REPO=${BASELINE_REPO:?Set BASELINE_REPO to a clean pushed worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:-$BASELINE_REPO}
BASELINE_COMMIT=${BASELINE_COMMIT:?Set BASELINE_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:-$BASELINE_COMMIT}
BASELINE_BRANCH=${BASELINE_BRANCH:?Set BASELINE_BRANCH}
CANDIDATE_BRANCH=${CANDIDATE_BRANCH:?Set CANDIDATE_BRANCH}
BASELINE_ENGINE=${BASELINE_ENGINE:-optimized}
CANDIDATE_ENGINE=${CANDIDATE_ENGINE:-optimized}
PROFILE_CONFIG=${PROFILE_CONFIG:?Set PROFILE_CONFIG explicitly to profile_salvalai_smoke15 or profile_salvalai}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-native-cross-mlp-smoke}
MIN_CANDIDATE_TPS=${MIN_CANDIDATE_TPS:-280}
MIN_MEAN_GAIN_PCT=${MIN_MEAN_GAIN_PCT:-5}
MIN_ORDER_GAIN_PCT=${MIN_ORDER_GAIN_PCT:-5}
REQUIRE_WALL_NONREGRESSION=${REQUIRE_WALL_NONREGRESSION:-1}
RUN_ROOT="$WORK/runs/inference-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

validate_repo() {
  local repo=$1
  local commit=$2
  local branch=$3
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "worktree is dirty at $repo" >&2
    git -C "$repo" status --short >&2
    exit 2
  fi
  if [[ "$(git -C "$repo" rev-parse HEAD)" != "$commit" ]]; then
    echo "worktree $repo differs from expected commit $commit" >&2
    exit 2
  fi
  if [[ "$(git -C "$repo" branch --show-current)" != "$branch" ]]; then
    echo "worktree $repo differs from expected branch $branch" >&2
    exit 2
  fi
}
validate_repo "$BASELINE_REPO" "$BASELINE_COMMIT" "$BASELINE_BRANCH"
validate_repo "$CANDIDATE_REPO" "$CANDIDATE_COMMIT" "$CANDIDATE_BRANCH"
if [[ ! -f "$AUDIO" ]]; then
  echo "missing input audio at $AUDIO" >&2
  exit 2
fi
export PATH="$(dirname "$PYTHON"):$PATH"
for tool in "$PYTHON" ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool" >&2
    exit 2
  fi
done
if [[ "$BASELINE_ENGINE" != v32 && "$BASELINE_ENGINE" != optimized ]]; then
  echo "BASELINE_ENGINE must be v32 or optimized" >&2
  exit 2
fi
if [[ "$CANDIDATE_ENGINE" != v32 && "$CANDIDATE_ENGINE" != optimized ]]; then
  echo "CANDIDATE_ENGINE must be v32 or optimized" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "baseline_commit=$BASELINE_COMMIT"
echo "candidate_commit=$CANDIDATE_COMMIT"
echo "baseline_branch=$BASELINE_BRANCH"
echo "candidate_branch=$CANDIDATE_BRANCH"
echo "baseline_engine=$BASELINE_ENGINE"
echo "candidate_engine=$CANDIDATE_ENGINE"
echo "profile_config=$PROFILE_CONFIG"
echo "run_root=$RUN_ROOT"
echo "min_candidate_tps=$MIN_CANDIDATE_TPS"
echo "min_mean_gain_pct=$MIN_MEAN_GAIN_PCT"
echo "min_order_gain_pct=$MIN_ORDER_GAIN_PCT"
echo "require_wall_nonregression=$REQUIRE_WALL_NONREGRESSION"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"
echo "nsys=$(command -v nsys || echo missing)"
echo "ncu=$(command -v ncu || echo missing)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi -L
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "native cross+MLP reciprocal requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
)

run_profile() {
  local repo=$1
  local engine=$2
  local label=$3
  local scout=$4
  local output="$RUN_ROOT/$label"
  local legacy_reset=()
  if [[ "$engine" == optimized ]] && grep -q 'inference_generation_compile:' "$repo/config.py"; then
    legacy_reset+=(inference_generation_compile=false)
  fi
  mkdir -p "$output"
  (
    cd "$repo"
    MAPPERATORINATOR_NATIVE_CROSS_MLP_SCOUT="$scout" \
    "$PYTHON" inference.py --config-name "$PROFILE_CONFIG" \
      output_path="$output" \
      inference_engine="$engine" \
      "${COMMON_OVERRIDES[@]}" \
      "${legacy_reset[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing profile for $label" >&2
    exit 2
  fi
  printf '%s\n' "$profile" > "$RUN_ROOT/$label.profile-path.txt"
}

run_profile "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_first 0
run_profile "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_second 1
run_profile "$CANDIDATE_REPO" "$CANDIDATE_ENGINE" candidate_first 1
run_profile "$BASELINE_REPO" "$BASELINE_ENGINE" baseline_second 0

BASELINE_FIRST=$(<"$RUN_ROOT/baseline_first.profile-path.txt")
CANDIDATE_SECOND=$(<"$RUN_ROOT/candidate_second.profile-path.txt")
CANDIDATE_FIRST=$(<"$RUN_ROOT/candidate_first.profile-path.txt")
BASELINE_SECOND=$(<"$RUN_ROOT/baseline_second.profile-path.txt")

(
  cd "$CANDIDATE_REPO"
  "$PYTHON" utils/summarize_inference_profile.py \
    --reciprocal "$BASELINE_FIRST" "$CANDIDATE_SECOND" "$CANDIDATE_FIRST" "$BASELINE_SECOND" \
    --labels main_generation,timing_context \
    --regression-tolerance-pct 5 \
    --json-output "$RUN_ROOT/reciprocal-compare.json"
)

"$PYTHON" - "$RUN_ROOT" "$BASELINE_ENGINE" "$CANDIDATE_ENGINE" \
  "$MIN_CANDIDATE_TPS" "$MIN_MEAN_GAIN_PCT" "$MIN_ORDER_GAIN_PCT" \
  "$REQUIRE_WALL_NONREGRESSION" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "baseline_first": sys.argv[2],
    "baseline_second": sys.argv[2],
    "candidate_first": sys.argv[3],
    "candidate_second": sys.argv[3],
}
profiles = {}
for label, engine in expected.items():
    profile_path = Path((root / f"{label}.profile-path.txt").read_text().strip())
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("metadata", {}).get("inference_engine") != engine:
        raise SystemExit(f"{label}: inference engine metadata mismatch")
    profiles[label] = profile

def records(profile, profile_label):
    return [
        row for row in profile.get("generation", [])
        if row.get("profile_label") == profile_label
    ]

def aggregate(profile, profile_label):
    rows = records(profile, profile_label)
    tokens = sum(int(row.get("generated_tokens", 0) or 0) for row in rows)
    model = sum(float(row.get("model_elapsed_seconds", 0.0) or 0.0) for row in rows)
    outer = sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in rows)
    return {"tokens": tokens, "model": model, "outer": outer, "tps": tokens / model}

for label in ("candidate_second", "candidate_first"):
    profile = profiles[label]
    main = records(profile, "main_generation")
    timing = records(profile, "timing_context")
    if not main or not timing:
        raise SystemExit(f"{label}: missing main or timing records")
    if not all(
        row.get("native_cross_mlp_tail_requested") is True
        and row.get("native_cross_mlp_tail_enabled") is True
        for row in main
    ):
        raise SystemExit(f"{label}: cross+MLP tail not enabled for every main record")
    if not all(
        row.get("native_cross_mlp_tail_requested") is True
        and row.get("native_cross_mlp_tail_enabled") is False
        and row.get("native_cross_mlp_tail_disabled_reason") == "timing_context"
        for row in timing
    ):
        raise SystemExit(f"{label}: cross+MLP timing disable metadata mismatch")

pairs = (
    ("baseline_first", "candidate_second"),
    ("baseline_second", "candidate_first"),
)
min_tps = float(sys.argv[4])
min_mean_gain = float(sys.argv[5])
min_order_gain = float(sys.argv[6])
require_wall = bool(int(sys.argv[7]))
gains = []
for baseline_label, candidate_label in pairs:
    baseline = aggregate(profiles[baseline_label], "main_generation")
    candidate = aggregate(profiles[candidate_label], "main_generation")
    if candidate["tps"] < min_tps:
        raise SystemExit(
            f"{candidate_label}: {candidate['tps']:.3f} TPS below {min_tps:.3f}"
        )
    gain = (candidate["tps"] - baseline["tps"]) / baseline["tps"] * 100.0
    gains.append(gain)
    if gain < min_order_gain:
        raise SystemExit(
            f"{candidate_label}: gain {gain:.3f}% below {min_order_gain:.3f}%"
        )
    if require_wall and candidate["outer"] > baseline["outer"]:
        raise SystemExit(
            f"{candidate_label}: main outer wall regressed "
            f"{baseline['outer']:.6f}->{candidate['outer']:.6f}"
        )
    baseline_stage = sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in profiles[baseline_label].get("stages", []))
    candidate_stage = sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in profiles[candidate_label].get("stages", []))
    if require_wall and candidate_stage > baseline_stage:
        raise SystemExit(
            f"{candidate_label}: stage wall regressed "
            f"{baseline_stage:.6f}->{candidate_stage:.6f}"
        )
if sum(gains) / len(gains) < min_mean_gain:
    raise SystemExit(
        f"mean reciprocal gain {sum(gains) / len(gains):.3f}% below "
        f"{min_mean_gain:.3f}%"
    )
print("native_cross_mlp_reciprocal_gate=PASS")
PY

find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*-compare.json' \) -print0 \
  | sort -z | xargs -0 sha256sum
