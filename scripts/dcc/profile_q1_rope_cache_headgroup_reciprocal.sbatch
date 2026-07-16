#!/usr/bin/env bash
#SBATCH --job-name=q1-headgroup
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() { echo "q1-headgroup: $*" >&2; exit 2; }
require_nonempty() { [[ -s "$1" ]] || die "required artifact is missing or empty: $1"; }

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean candidate worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed candidate commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the candidate branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed candidate remote ref}
PRECISION=${MAPPERATORINATOR_PRECISION:-fp32}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
SHARED_ROPE_BASELINE=${MAPPERATORINATOR_SHARED_ROPE_BASELINE:-0}
RUN_LABEL=${RUN_LABEL:-q1-headgroup-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

case "$PRECISION" in
  fp32) ANALYSIS_MODE=exact-fp32 ;;
  fp16) ANALYSIS_MODE=exact-same-precision ;;
  *) die "MAPPERATORINATOR_PRECISION must be fp32 or fp16" ;;
esac
[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
[[ "$SHARED_ROPE_BASELINE" == 0 || "$SHARED_ROPE_BASELINE" == 1 ]] || die "MAPPERATORINATOR_SHARED_ROPE_BASELINE must be 0 or 1"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "candidate worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "worktree HEAD differs from MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "worktree branch differs from MAPPERATORINATOR_BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "pushed remote ref differs from MAPPERATORINATOR_COMMIT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache" HF_HOME="$WORK/cache/huggingface" TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor" TRITON_CACHE_DIR="$WORK/cache/triton" CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions/$SLURM_JOB_ID"
export TMPDIR="$WORK/tmp/reciprocal-$SLURM_JOB_ID"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
export TOKENIZERS_PARALLELISM=false NVIDIA_TF32_OVERRIDE=0

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"; done
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 && "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] || die "expected one $EXPECTED_GPU GPU"
mapfile -t OTHER_GPU_JOBS < <(squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' | awk -F'|' -v current="$SLURM_JOB_ID" '$1 != current && tolower($2) ~ /gpu/ { print }')
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] || die "another GPU job is running or pending for $USER"

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "precision=$PRECISION"
  echo "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL"
  echo "shared_rope_baseline=$SHARED_ROPE_BASELINE"
  echo "run_root=$RUN_ROOT"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
}
trap finalize EXIT

cd "$REPO"
COMMON_ARGS=(--config-name profile_salvalai audio_path="$AUDIO" device=cuda precision="$PRECISION" attn_implementation=sdpa inference_engine=optimized use_server=false parallel=false cfg_scale=1.0 num_beams=1 seed=12345 profile_inference=true profile_detail_ranges=false profile_cuda_capture=false profile_pass_kind=untraced_control)
declare -A PROFILE_PATHS OSU_PATHS EVIDENCE_PATHS
run_role() {
  local role=$1
  local kind=$2
  local run_dir="$RUN_ROOT/$role"
  mkdir -p "$run_dir/output"
  local evidence="$run_dir/evidence.json"
  local command=("$PYTHON" utils/run_exact_q1_rope_cache_headgroup.py --mode "$kind" --config-name profile_salvalai --rope-evidence-path "$run_dir/rope-evidence.json" --evidence-manifest "$evidence" "${COMMON_ARGS[@]:2}" output_path="$run_dir/output")
  printf '%q ' "${command[@]}" > "$run_dir/command.txt"; printf '\n' >> "$run_dir/command.txt"
  "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "$role produced ${#profiles[@]} profiles"
  # Split locals: under set -u, `local a=$x b=$a` leaves a unbound while expanding b.
  local profile=${profiles[0]}
  local osu=${profile%.profile.json}
  require_nonempty "$profile"; require_nonempty "$osu"; require_nonempty "$evidence"
  "$PYTHON" - "$evidence" "$kind" <<'PY'
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
role = sys.argv[2]
assert payload["schema_version"] == 1
assert payload["role"] == role
assert payload["candidate"] == "exact_q1_rope_cache_headgroup_on_tip"
assert payload["enabled"] == (role == "candidate")
assert payload["native_q1_rope_cache_headgroup_enabled"] == (role == "candidate")
assert payload["shared_rope_enabled"] is True
assert payload["device_sequence_state_enabled"] is True
assert isinstance(payload["decode_loop_calls"], int) and payload["decode_loop_calls"] > 0
assert isinstance(payload["headgroup_profile_calls"], int)
assert (payload["headgroup_profile_calls"] > 0) == (role == "candidate")
PY
  require_nonempty "$run_dir/rope-evidence.json"
  PROFILE_PATHS[$role]=$profile; OSU_PATHS[$role]=$osu; EVIDENCE_PATHS[$role]=$evidence
}

run_role baseline_first baseline
run_role candidate_first candidate
run_role candidate_second candidate
run_role baseline_second baseline

"$PYTHON" utils/analyze_reciprocal_full_song_candidate.py \
  --baseline-first-profile "${PROFILE_PATHS[baseline_first]}" --baseline-first-osu "${OSU_PATHS[baseline_first]}" \
  --candidate-first-profile "${PROFILE_PATHS[candidate_first]}" --candidate-first-osu "${OSU_PATHS[candidate_first]}" \
  --candidate-second-profile "${PROFILE_PATHS[candidate_second]}" --candidate-second-osu "${OSU_PATHS[candidate_second]}" \
  --baseline-second-profile "${PROFILE_PATHS[baseline_second]}" --baseline-second-osu "${OSU_PATHS[baseline_second]}" \
  --mode "$ANALYSIS_MODE" \
  --allow-dispatch-delta '*native_q1_rope_cache_headgroup*' \
  --allow-dispatch-delta '*q1_rope_cache_headgroup*' \
  --allow-dispatch-delta '*optimized_cuda_graphs*' \
  --json-output "$RUN_ROOT/analysis.json" --text-output "$RUN_ROOT/analysis.txt"
require_nonempty "$RUN_ROOT/analysis.json"; require_nonempty "$RUN_ROOT/analysis.txt"

"$PYTHON" - "$RUN_ROOT/analysis.json" "$PRECISION" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["parity"]["cross_candidate_exact"] is True
assert report["parity"]["dispatch_cache_topology"]["pass"] is True
assert not report["parity"]["dispatch_cache_topology"]["differences"]
assert {run["precision"] for run in report["runs"].values()} == {sys.argv[2]}
assert report["metrics"]["main_tokens"]["candidate_minus_baseline"] == 0.0
PY

"$PYTHON" - \
  "${EVIDENCE_PATHS[baseline_first]}" \
  "${EVIDENCE_PATHS[candidate_first]}" \
  "${EVIDENCE_PATHS[candidate_second]}" \
  "${EVIDENCE_PATHS[baseline_second]}" <<'PY'
import json, pathlib, sys
evidence = [json.loads(pathlib.Path(path).read_text()) for path in sys.argv[1:]]
for key in ("rng_after_seed", "rng_after_inference"):
    reference = evidence[0][key]
    assert all(payload[key] == reference for payload in evidence[1:]), key
PY
echo "q1_rope_cache_headgroup_reciprocal=PASS"
