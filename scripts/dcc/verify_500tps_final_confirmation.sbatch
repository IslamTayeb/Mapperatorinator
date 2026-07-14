#!/usr/bin/env bash
#SBATCH --job-name=500tps-final-confirm
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=04:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=${MAPPERATORINATOR_ENV:-/hpc/group/romerolab/imt11/envs/mapperatorinator}
PYTHON=${MAPPERATORINATOR_PYTHON:-$ENV/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean DCC worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH}
REMOTE=${MAPPERATORINATOR_REMOTE:-origin}
RUN_LABEL=${RUN_LABEL:-500tps-final-confirmation}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
CANONICAL_SALVALAI=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
FIVE_SONG_ROOT=${FIVE_SONG_ROOT:-$WORK/data/five-song-profile}
NATURAL_REPETITIONS=${NATURAL_REPETITIONS:-2}
FIXED_REPETITIONS=${FIXED_REPETITIONS:-5}
EXPECTED_MAIN_STEPS=${EXPECTED_MAIN_STEPS:-8294}

if [[ "$NATURAL_REPETITIONS" != 2 || "$FIXED_REPETITIONS" != 5 ]]; then
  echo "authoritative final confirmation requires 2 natural and 5 fixed repetitions" >&2
  exit 2
fi
if [[ "$EXPECTED_MAIN_STEPS" != 8294 ]]; then
  echo "authoritative fixed-work control requires exactly 8294 main steps" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing environment Python: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing DCC worktree: $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "DCC worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "DCC worktree commit mismatch" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "DCC worktree branch mismatch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$BRANCH"
if ! git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || [[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" != "$COMMIT" ]]; then
  echo "pushed remote ref does not match exact confirmation commit" >&2
  exit 2
fi

export PATH="$(dirname "$PYTHON"):$PATH"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool" >&2; exit 2; }
done
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if [[ "$GPU_NAME" != "NVIDIA GeForce RTX 2080 Ti" ]]; then
  echo "final confirmation requires RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/tmp" "$WORK/torch_extensions/final-$COMMIT"
mkdir "$RUN_ROOT"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export TMPDIR="$WORK/tmp"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/final-$COMMIT"

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "repo=$REPO"
echo "run_root=$RUN_ROOT"
echo "gpu=$GPU_NAME"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job -o "$SLURM_JOB_ID"
fi
nvidia-smi -L
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

SONG_MANIFEST="$RUN_ROOT/five-songs.json"
"$PYTHON" - "$SONG_MANIFEST" "$CANONICAL_SALVALAI" "$FIVE_SONG_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, salvalai, root = sys.argv[1:]
paths = [
    ("lambada", Path(root) / "lambada.mp3"),
    ("pegasus", Path(root) / "pegasus.mp3"),
    ("ela-ke-leitada", Path(root) / "ela-ke-leitada.mp3"),
    ("salvalai", Path(salvalai)),
    ("nube-negra", Path(root) / "nube-negra.mp3"),
]
songs = []
for name, path in paths:
    if not path.is_file():
        raise SystemExit(f"missing five-song audio: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    songs.append({"name": name, "audio_path": str(path.resolve()), "sha256": digest})
Path(output).write_text(json.dumps({"schema_version": 1, "songs": songs}, indent=2, sort_keys=True) + "\n")
PY

COMMON_OVERRIDES=(
  device=cuda precision=fp32 attn_implementation=sdpa inference_engine=optimized
  use_server=false parallel=false cfg_scale=1.0 num_beams=1 seed=12345
  super_timing=false generate_positions=false profile_inference=true
  profile_detail_ranges=false profile_cuda_capture=false profile_pass_kind=untraced_control
)

run_fresh() {
  local root=$1
  local variant=$2
  local index=$3
  local mode=$4
  local manifest=${5:-}
  local dir="$root/$variant-$index"
  mkdir -p "$dir/output"
  local candidate_args=()
  local manifest_args=()
  if [[ "$variant" == candidate ]]; then
    candidate_args=(--candidate --initialization-path "$dir/initialization.json")
  fi
  if [[ -n "$manifest" ]]; then
    manifest_args=(--manifest-path "$manifest")
  fi
  (
    cd "$REPO"
    "$PYTHON" utils/run_final_confirmation.py \
      --config-name profile_salvalai \
      --mode "$mode" \
      --expected-main-steps "$EXPECTED_MAIN_STEPS" \
      --evidence-path "$dir/evidence.json" \
      "${candidate_args[@]}" \
      "${manifest_args[@]}" \
      audio_path="$CANONICAL_SALVALAI" \
      output_path="$dir/output" \
      "${COMMON_OVERRIDES[@]}"
  )
  [[ -s "$dir/evidence.json" ]] || { echo "missing evidence: $dir" >&2; exit 2; }
}

NATURAL_ROOT="$RUN_ROOT/natural-salvalai"
mkdir -p "$NATURAL_ROOT"
run_fresh "$NATURAL_ROOT" baseline 0 natural
run_fresh "$NATURAL_ROOT" candidate 0 natural
run_fresh "$NATURAL_ROOT" candidate 1 natural
run_fresh "$NATURAL_ROOT" baseline 1 natural

FIXED_ROOT="$RUN_ROOT/fixed-work-salvalai"
mkdir -p "$FIXED_ROOT/manifest-record/output"
FIXED_MANIFEST="$FIXED_ROOT/fixed-work-manifest.json"
(
  cd "$REPO"
  "$PYTHON" utils/run_final_confirmation.py \
    --config-name profile_salvalai \
    --mode record-fixed-work \
    --manifest-path "$FIXED_MANIFEST" \
    --expected-main-steps "$EXPECTED_MAIN_STEPS" \
    --evidence-path "$FIXED_ROOT/manifest-record/evidence.json" \
    audio_path="$CANONICAL_SALVALAI" \
    output_path="$FIXED_ROOT/manifest-record/output" \
    "${COMMON_OVERRIDES[@]}"
)
[[ -s "$FIXED_MANIFEST" ]] || { echo "fixed-work manifest missing" >&2; exit 2; }
for index in 0 1 2 3 4; do
  if (( index % 2 == 0 )); then
    run_fresh "$FIXED_ROOT" baseline "$index" replay-fixed-work "$FIXED_MANIFEST"
    run_fresh "$FIXED_ROOT" candidate "$index" replay-fixed-work "$FIXED_MANIFEST"
  else
    run_fresh "$FIXED_ROOT" candidate "$index" replay-fixed-work "$FIXED_MANIFEST"
    run_fresh "$FIXED_ROOT" baseline "$index" replay-fixed-work "$FIXED_MANIFEST"
  fi
done

for variant in baseline candidate; do
  dir="$RUN_ROOT/five-song-$variant"
  candidate_args=()
  if [[ "$variant" == candidate ]]; then
    candidate_args=(--candidate --initialization-path "$dir/initialization.json")
  fi
  (
    cd "$REPO"
    "$PYTHON" utils/run_serial_final_confirmation.py \
      --config-name profile_salvalai \
      --song-manifest "$SONG_MANIFEST" \
      --output-root "$dir/output" \
      --evidence-path "$dir/evidence.json" \
      "${candidate_args[@]}" \
      "${COMMON_OVERRIDES[@]}"
  )
done

(
  cd "$REPO"
  "$PYTHON" utils/analyze_final_confirmation.py \
    --natural-root "$NATURAL_ROOT" \
    --fixed-root "$FIXED_ROOT" \
    --baseline-suite-evidence "$RUN_ROOT/five-song-baseline/evidence.json" \
    --candidate-suite-evidence "$RUN_ROOT/five-song-candidate/evidence.json" \
    --natural-repetitions "$NATURAL_REPETITIONS" \
    --fixed-repetitions "$FIXED_REPETITIONS" \
    --json-output "$RUN_ROOT/analysis.json" \
    --text-output "$RUN_ROOT/analysis.txt"
)

for artifact in "$RUN_ROOT/analysis.json" "$RUN_ROOT/analysis.txt" "$RUN_ROOT/five-songs.json"; do
  [[ -s "$artifact" ]] || { echo "missing final artifact: $artifact" >&2; exit 2; }
done
cat "$RUN_ROOT/analysis.txt"
