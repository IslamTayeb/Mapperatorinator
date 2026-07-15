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
REMOTE_BRANCH=${MAPPERATORINATOR_REMOTE_BRANCH:-$BRANCH}
RUN_LABEL=${RUN_LABEL:-500tps-final-confirmation}
RUN_ROOT="$WORK/runs/$RUN_LABEL-${SLURM_JOB_ID:-manual}"
CANONICAL_SALVALAI=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
FIVE_SONG_ROOT=${FIVE_SONG_ROOT:-$WORK/data/five-song-profile}
NATURAL_REPETITIONS=${NATURAL_REPETITIONS:-5}
FIXED_REPETITIONS=${FIXED_REPETITIONS:-5}
EXPECTED_MAIN_STEPS=${EXPECTED_MAIN_STEPS:-8294}
REQUIRE_DIRECT_AOT=${REQUIRE_DIRECT_AOT:-1}
AOT_NATIVE_EXTENSION_MANIFEST=${AOT_NATIVE_EXTENSION_MANIFEST:-}
AOT_NATIVE_EXTENSION_MANIFEST_SHA256=${AOT_NATIVE_EXTENSION_MANIFEST_SHA256:-}
CANDIDATE_RUNTIME_SPEC_SOURCE=${CANDIDATE_RUNTIME_SPEC:-$REPO/scripts/dcc/runtime_specs/k1_int8_fp16_cross_shared_arena.json}
CANDIDATE_RUNTIME_NAME=${CANDIDATE_RUNTIME_NAME:-k4-split-kv-mixed-weight-shared-rope-k1-remainder-int8-mlp-v1}
CANDIDATE_RUNTIME_FACTORY=${CANDIDATE_RUNTIME_FACTORY:-utils.final_confirmation_runtime:kblock_shared_rope_weight_plugin}
CANDIDATE_RUNTIME_KWARGS_JSON=${CANDIDATE_RUNTIME_KWARGS_JSON:-'{"block_size":4,"graph_remainders":true,"initializer_name":"initialize_approximate_int8_mlp_weight_only","shared_rope_binding_index":0,"initializer_binding_index":0,"minimum_bindings":1}'}

if [[ "$NATURAL_REPETITIONS" != 5 || "$FIXED_REPETITIONS" != 5 ]]; then
  echo "authoritative final confirmation requires 5 natural and 5 fixed repetitions" >&2
  exit 2
fi
if [[ "$EXPECTED_MAIN_STEPS" != 8294 ]]; then
  echo "authoritative fixed-work control requires exactly 8294 main steps" >&2
  exit 2
fi
if [[ "$REQUIRE_DIRECT_AOT" != 0 && "$REQUIRE_DIRECT_AOT" != 1 ]]; then
  echo "REQUIRE_DIRECT_AOT must be 0 or 1" >&2
  exit 2
fi
if [[ "$REQUIRE_DIRECT_AOT" == 1 ]]; then
  if [[ ! -s "$AOT_NATIVE_EXTENSION_MANIFEST" ]]; then
    echo "authoritative confirmation requires a nonempty direct AOT manifest" >&2
    exit 2
  fi
  if [[ ! "$AOT_NATIVE_EXTENSION_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "AOT_NATIVE_EXTENSION_MANIFEST_SHA256 must be a lowercase SHA-256" >&2
    exit 2
  fi
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
CURRENT_BRANCH=$(git -C "$REPO" branch --show-current)
if [[ "$BRANCH" == DETACHED ]]; then
  if [[ -n "$CURRENT_BRANCH" ]]; then
    echo "DCC worktree must be detached" >&2
    exit 2
  fi
else
  if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "DCC worktree branch mismatch" >&2
    exit 2
  fi
fi
if [[ -z "$REMOTE_BRANCH" || "$REMOTE_BRANCH" == DETACHED ]]; then
  echo "MAPPERATORINATOR_REMOTE_BRANCH must name the pushed branch" >&2
  exit 2
fi
REMOTE_REF="refs/remotes/$REMOTE/$REMOTE_BRANCH"
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
export TORCH_CUDA_ARCH_LIST=7.5

if [[ "$REQUIRE_DIRECT_AOT" == 1 ]]; then
  (
    cd "$REPO"
    "$PYTHON" - \
      "$AOT_NATIVE_EXTENSION_MANIFEST" \
      "$COMMIT" \
      "$AOT_NATIVE_EXTENSION_MANIFEST_SHA256" \
      "$RUN_ROOT/aot-manifest-validation.json" <<'PY'
import json
import sys
from pathlib import Path

from osuT5.osuT5.inference.optimized.kernels.native_extension import (
    validate_direct_manifest,
)

manifest, commit, expected_hash, output = sys.argv[1:]
report = validate_direct_manifest(
    Path(manifest),
    expected_source_commit=commit,
    expected_manifest_sha256=expected_hash,
)
Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
PY
  )
  [[ -s "$RUN_ROOT/aot-manifest-validation.json" ]] || {
    echo "direct AOT manifest validation did not produce evidence" >&2
    exit 2
  }
  export MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST="$AOT_NATIVE_EXTENSION_MANIFEST"
else
  unset MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST
fi

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "remote_branch=$REMOTE_BRANCH"
echo "repo=$REPO"
echo "run_root=$RUN_ROOT"
echo "gpu=$GPU_NAME"
echo "native_extension_mode=$([[ "$REQUIRE_DIRECT_AOT" == 1 ]] && echo direct || echo load_inline)"
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

CANDIDATE_RUNTIME_SPEC_PATH="$RUN_ROOT/candidate-runtime-spec.json"
"$PYTHON" - \
  "$CANDIDATE_RUNTIME_SPEC_PATH" \
  "$CANDIDATE_RUNTIME_SPEC_SOURCE" \
  "$CANDIDATE_RUNTIME_NAME" \
  "$CANDIDATE_RUNTIME_FACTORY" \
  "$CANDIDATE_RUNTIME_KWARGS_JSON" <<'PY'
import json
import sys
from pathlib import Path

output, source, name, factory, kwargs_json = sys.argv[1:]
if source:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"candidate runtime spec is missing: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
else:
    payload = {
        "schema_version": "mapperatorinator.final-runtime-spec.v1",
        "name": name,
        "factory": factory,
        "kwargs": json.loads(kwargs_json),
    }
if not isinstance(payload, dict):
    raise SystemExit("candidate runtime spec root must be an object")
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
[[ -s "$CANDIDATE_RUNTIME_SPEC_PATH" ]] || {
  echo "candidate runtime spec was not created" >&2
  exit 2
}

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
    candidate_args=(
      --candidate
      --runtime-spec "$CANDIDATE_RUNTIME_SPEC_PATH"
      --initialization-path "$dir/runtime.json"
    )
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
for index in 0 1 2 3 4; do
  if (( index % 2 == 0 )); then
    run_fresh "$NATURAL_ROOT" baseline "$index" natural
    run_fresh "$NATURAL_ROOT" candidate "$index" natural
  else
    run_fresh "$NATURAL_ROOT" candidate "$index" natural
    run_fresh "$NATURAL_ROOT" baseline "$index" natural
  fi
done

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
    candidate_args=(
      --candidate
      --runtime-spec "$CANDIDATE_RUNTIME_SPEC_PATH"
      --initialization-path "$dir/runtime.json"
    )
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
