#!/usr/bin/env bash
#SBATCH --job-name=shared-rope-full-song
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() {
  echo "shared-rope-full-song: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "missing or empty artifact: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside Slurm}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set the clean pushed candidate worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set the candidate commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set the candidate branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set the pushed candidate ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
PRECISION=${MAPPERATORINATOR_PRECISION:?Set MAPPERATORINATOR_PRECISION to fp32 or fp16}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
RUN_LABEL=${RUN_LABEL:-shared-rope-full-song-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ "$PRECISION" == fp32 || "$PRECISION" == fp16 ]] \
  || die "MAPPERATORINATOR_PRECISION must be fp32 or fp16"
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] \
  || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
[[ -f "$AUDIO" ]] || die "missing audio: $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree: $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "dirty worktree: $REPO"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "candidate commit mismatch"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "candidate branch mismatch"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed candidate ref"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed candidate ref mismatch"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
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

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do
  command -v "$tool" >/dev/null 2>&1 || die "missing executable: $tool"
done
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected one visible GPU"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] || {
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another user GPU job exists"
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
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "run_root=$RUN_ROOT"
  echo "precision=$PRECISION"
  echo "parallel_reciprocal_opt_in=$ALLOW_PARALLEL_RECIPROCAL"
  echo "NVIDIA_TF32_OVERRIDE=$NVIDIA_TF32_OVERRIDE"
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
  squeue -u "$USER" -o '%.18i %.40j %.10T %.24b'
  df -h "$WORK"
} | tee "$RUN_ROOT/preflight.txt"

capture_paths() {
  local run_dir=$1
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "expected one profile in $run_dir"
  local profile=${profiles[0]}
  local result=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$result"
  printf '%s\n' "$profile" > "$run_dir/profile-path.txt"
  printf '%s\n' "$result" > "$run_dir/result-path.txt"
}

run_process() {
  local repo=$1
  local precision=$2
  local side=$3
  local name=$4
  local pass_kind=$5
  local run_dir="$RUN_ROOT/$precision/$side/$name"
  mkdir -p "$run_dir/output"
  local entrypoint=(inference.py --config-name profile_salvalai)
  if [[ "$side" == candidate ]]; then
    entrypoint=(
      utils/run_strict_shared_rope.py
      --config-name profile_salvalai
      --evidence-path "$run_dir/candidate-evidence.json"
    )
  fi
  local command=(
    "$PYTHON" "${entrypoint[@]}"
    audio_path="$AUDIO"
    output_path="$run_dir/output"
    device=cuda
    precision="$precision"
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
    profile_pass_kind="$pass_kind"
  )
  printf '%q ' "${command[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  local start finish exit_code
  start=$(date +%s%N)
  set +e
  (cd "$repo" && "${command[@]}") > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  exit_code=$?
  set -e
  finish=$(date +%s%N)
  printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
    "$start" "$finish" "$((finish - start))" "$exit_code" > "$run_dir/process-wall.json"
  [[ $exit_code -eq 0 ]] || die "$precision/$side/$name failed"
  capture_paths "$run_dir"
}

for precision in "$PRECISION"; do
  for index in 01 02 03; do
    if (( 10#$index % 2 == 1 )); then
      order=(baseline candidate)
    else
      order=(candidate baseline)
    fi
    for side in "${order[@]}"; do
      run_process "$REPO" "$precision" "$side" "run-$index" untraced_control
    done
  done
  run_process "$REPO" "$precision" baseline exactness-audit exactness_audit
  run_process "$REPO" "$precision" candidate exactness-audit exactness_audit
done

cd "$REPO"
"$PYTHON" utils/analyze_shared_rope_full_song.py \
  --run-root "$RUN_ROOT" \
  --precision "$PRECISION" \
  --baseline-commit "$COMMIT" \
  --baseline-branch "$BRANCH" \
  --candidate-commit "$COMMIT" \
  --candidate-branch "$BRANCH" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "shared_rope_full_song_$PRECISION=PASS"
