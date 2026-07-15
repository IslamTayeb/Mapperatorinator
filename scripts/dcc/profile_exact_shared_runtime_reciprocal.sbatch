#!/usr/bin/env bash
#SBATCH --job-name=exact-shared-runtime
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:35:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "exact-shared-runtime: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
BASELINE_REPO=${MAPPERATORINATOR_BASELINE_REPO:?Set the clean current-main worktree}
BASELINE_COMMIT=${MAPPERATORINATOR_BASELINE_COMMIT:?Set the exact current-main commit}
BASELINE_BRANCH=${MAPPERATORINATOR_BASELINE_BRANCH:?Set the baseline worktree branch}
BASELINE_REMOTE_REF=${MAPPERATORINATOR_BASELINE_REMOTE_REF:?Set the pushed baseline remote ref}
CANDIDATE_REPO=${MAPPERATORINATOR_CANDIDATE_REPO:?Set the clean promotion worktree}
CANDIDATE_COMMIT=${MAPPERATORINATOR_CANDIDATE_COMMIT:?Set the exact promotion commit}
CANDIDATE_BRANCH=${MAPPERATORINATOR_CANDIDATE_BRANCH:?Set the promotion worktree branch}
CANDIDATE_REMOTE_REF=${MAPPERATORINATOR_CANDIDATE_REMOTE_REF:?Set the pushed promotion remote ref}
PRECISION=${MAPPERATORINATOR_PRECISION:-fp32}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
RUN_LABEL=${RUN_LABEL:-exact-shared-runtime-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

case "$PRECISION" in
  fp32) ANALYSIS_MODE=exact-fp32 ;;
  fp16) ANALYSIS_MODE=exact-same-precision ;;
  *) die "MAPPERATORINATOR_PRECISION must be fp32 or fp16" ;;
esac

validate_worktree() {
  local label=$1 repo=$2 commit=$3 branch=$4 remote_ref=$5
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "$label worktree is missing: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || die "$label worktree is dirty"
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] \
    || die "$label worktree HEAD differs from its declared commit"
  [[ "$(git -C "$repo" branch --show-current)" == "$branch" ]] \
    || die "$label worktree branch differs from its declared branch"
  git -C "$repo" show-ref --verify --quiet "$remote_ref" \
    || die "$label pushed remote ref is missing: $remote_ref"
  [[ "$(git -C "$repo" rev-parse "$remote_ref")" == "$commit" ]] \
    || die "$label pushed remote ref differs from its declared commit"
}

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ "$BASELINE_COMMIT" != "$CANDIDATE_COMMIT" ]] \
  || die "baseline and candidate commits must differ"
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] \
  || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
validate_worktree baseline "$BASELINE_REPO" "$BASELINE_COMMIT" \
  "$BASELINE_BRANCH" "$BASELINE_REMOTE_REF"
validate_worktree candidate "$CANDIDATE_REPO" "$CANDIDATE_COMMIT" \
  "$CANDIDATE_BRANCH" "$CANDIDATE_REMOTE_REF"

mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/cache/torch_inductor" \
  "$WORK/cache/triton" "$WORK/cache/cuda" "$WORK/torch_extensions" \
  "$WORK/tmp" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
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
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j|%N' \
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
  echo "baseline_repo=$BASELINE_REPO"
  echo "baseline_commit=$BASELINE_COMMIT"
  echo "candidate_repo=$CANDIDATE_REPO"
  echo "candidate_commit=$CANDIDATE_COMMIT"
  echo "precision=$PRECISION"
  echo "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL"
  echo "run_root=$RUN_ROOT"
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
  local role=$1 kind=$2 repo commit
  case "$kind" in
    baseline) repo=$BASELINE_REPO; commit=$BASELINE_COMMIT ;;
    candidate) repo=$CANDIDATE_REPO; commit=$CANDIDATE_COMMIT ;;
    *) die "unknown reciprocal role kind: $kind" ;;
  esac
  local run_dir="$RUN_ROOT/$role"
  local output_dir="$run_dir/output"
  local evidence="$run_dir/evidence.json"
  local extension_dir="$WORK/torch_extensions/exact-shared-${commit}"
  local temp_dir="$WORK/tmp/exact-shared-${SLURM_JOB_ID}-${role}"
  mkdir -p "$output_dir" "$extension_dir" "$temp_dir"
  local command=(
    "$PYTHON" "$CANDIDATE_REPO/utils/run_external_inference_with_rng.py"
    --repo "$repo"
    --expected-commit "$commit"
    --role "$role"
    --evidence-manifest "$evidence"
    --
    "${COMMON_ARGS[@]}"
    output_path="$output_dir"
  )
  printf 'TORCH_EXTENSIONS_DIR=%q TMPDIR=%q ' "$extension_dir" "$temp_dir" \
    > "$run_dir/command.txt"
  printf '%q ' "${command[@]}" >> "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  local started finished
  started=$(date +%s.%N)
  TORCH_EXTENSIONS_DIR="$extension_dir" TMPDIR="$temp_dir" PYTHONPATH="$repo" \
    "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  finished=$(date +%s.%N)
  "$PYTHON" - "$run_dir/process-wall.json" "$started" "$finished" <<'PY'
import json
import pathlib
import sys

started = float(sys.argv[2])
finished = float(sys.argv[3])
if finished <= started:
    raise SystemExit("process wall interval is not positive")
pathlib.Path(sys.argv[1]).write_text(
    json.dumps({"started_epoch_seconds": started, "finished_epoch_seconds": finished,
                "wall_seconds": finished - started}, indent=2, sort_keys=True) + "\n"
)
PY
  mapfile -t profiles < <(find "$output_dir" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "$role produced ${#profiles[@]} profiles"
  local profile=${profiles[0]}
  local osu=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$osu"
  require_nonempty "$evidence"
  require_nonempty "$run_dir/process-wall.json"
  PROFILE_PATHS[$role]=$profile
  OSU_PATHS[$role]=$osu
  EVIDENCE_PATHS[$role]=$evidence
}

run_role baseline_first baseline
run_role candidate_first candidate
run_role candidate_second candidate
run_role baseline_second baseline

ANALYZER=(
  "$PYTHON" "$CANDIDATE_REPO/utils/analyze_reciprocal_full_song_candidate.py"
  --baseline-first-profile "${PROFILE_PATHS[baseline_first]}"
  --baseline-first-osu "${OSU_PATHS[baseline_first]}"
  --candidate-first-profile "${PROFILE_PATHS[candidate_first]}"
  --candidate-first-osu "${OSU_PATHS[candidate_first]}"
  --candidate-second-profile "${PROFILE_PATHS[candidate_second]}"
  --candidate-second-osu "${OSU_PATHS[candidate_second]}"
  --baseline-second-profile "${PROFILE_PATHS[baseline_second]}"
  --baseline-second-osu "${OSU_PATHS[baseline_second]}"
  --mode "$ANALYSIS_MODE"
  --allow-dispatch-delta 'optimized_effective_config*'
  --allow-dispatch-delta 'records.*.optimized_effective_config_version'
  --allow-dispatch-delta 'records.*.optimized_dispatch_capture_hits.shared_decoder_rope_*'
  --json-output "$RUN_ROOT/analysis.json"
  --text-output "$RUN_ROOT/analysis.txt"
)
"${ANALYZER[@]}"
require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"

"$PYTHON" - "$RUN_ROOT/analysis.json" "$PRECISION" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
precision = sys.argv[2]
assert report["parity"]["cross_candidate_exact"] is True
assert report["parity"]["dispatch_cache_topology"]["pass"] is True
assert {run["precision"] for run in report["runs"].values()} == {precision}
assert report["metrics"]["main_tokens"]["candidate_minus_baseline"] == 0.0
PY

"$PYTHON" - \
  "${EVIDENCE_PATHS[baseline_first]}" \
  "${EVIDENCE_PATHS[candidate_first]}" \
  "${EVIDENCE_PATHS[candidate_second]}" \
  "${EVIDENCE_PATHS[baseline_second]}" <<'PY'
import json
import pathlib
import sys

evidence = [json.loads(pathlib.Path(path).read_text()) for path in sys.argv[1:]]
for key in ("rng_after_seed", "rng_after_inference"):
    reference = evidence[0][key]
    assert all(payload[key] == reference for payload in evidence[1:]), key
PY

"$PYTHON" - \
  "${PROFILE_PATHS[baseline_first]}" \
  "${PROFILE_PATHS[candidate_first]}" <<'PY'
import json
import pathlib
import sys

baseline = json.loads(pathlib.Path(sys.argv[1]).read_text())
candidate = json.loads(pathlib.Path(sys.argv[2]).read_text())
baseline_config = baseline["metadata"]["optimized_effective_config"]
candidate_config = candidate["metadata"]["optimized_effective_config"]
assert "shared_decoder_rope" not in baseline_config
assert "preallocated_batch1_sequence_state" not in baseline_config
assert candidate_config["shared_decoder_rope"] is True
assert candidate_config["preallocated_batch1_sequence_state"] is True

for label in ("timing_context", "main_generation"):
    old = [row for row in baseline["generation"] if row.get("profile_label") == label]
    new = [row for row in candidate["generation"] if row.get("profile_label") == label]
    assert len(old) == len(new) and old
    for old_row, new_row in zip(old, new, strict=True):
        old_hits = old_row["optimized_dispatch_capture_hits"]
        new_hits = new_row["optimized_dispatch_capture_hits"]
        for key in (
            "native_q1_rope_cache_self_attention",
            "native_q1_self_attention",
            "q1_bmm_cross_attention",
            "native_cross_mlp_tail",
        ):
            assert new_hits[key] == old_hits[key], (label, key)
        assert new_hits["shared_decoder_rope_compute"] >= 0
        assert new_hits["shared_decoder_rope_reuse"] >= 0
PY

echo "exact_shared_runtime_reciprocal=PASS"
