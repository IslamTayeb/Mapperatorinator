#!/usr/bin/env bash
#SBATCH --job-name=lazy-start-full-song
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
  echo "lazy-start-full-song: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "missing or empty artifact: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside Slurm}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
BASELINE_REPO=${MAPPERATORINATOR_BASELINE_REPO:?Set the clean baseline worktree}
BASELINE_COMMIT=${MAPPERATORINATOR_BASELINE_COMMIT:?Set the baseline commit}
BASELINE_BRANCH=${MAPPERATORINATOR_BASELINE_BRANCH:?Set the baseline branch}
CANDIDATE_REPO=${MAPPERATORINATOR_CANDIDATE_REPO:?Set the clean candidate worktree}
CANDIDATE_COMMIT=${MAPPERATORINATOR_CANDIDATE_COMMIT:?Set the candidate commit}
CANDIDATE_BRANCH=${MAPPERATORINATOR_CANDIDATE_BRANCH:?Set the candidate branch}
CANDIDATE_REMOTE_REF=${MAPPERATORINATOR_CANDIDATE_REMOTE_REF:?Set the pushed candidate ref}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
RUN_LABEL=${RUN_LABEL:-lazy-start-full-song}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ -f "$AUDIO" ]] || die "missing audio: $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
for repo in "$BASELINE_REPO" "$CANDIDATE_REPO"; do
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "missing worktree: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || die "dirty worktree: $repo"
done
[[ "$(git -C "$BASELINE_REPO" rev-parse HEAD)" == "$BASELINE_COMMIT" ]] \
  || die "baseline commit mismatch"
[[ "$(git -C "$BASELINE_REPO" branch --show-current)" == "$BASELINE_BRANCH" ]] \
  || die "baseline branch mismatch"
[[ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" == "$CANDIDATE_COMMIT" ]] \
  || die "candidate commit mismatch"
[[ "$(git -C "$CANDIDATE_REPO" branch --show-current)" == "$CANDIDATE_BRANCH" ]] \
  || die "candidate branch mismatch"
git -C "$CANDIDATE_REPO" show-ref --verify --quiet "$CANDIDATE_REMOTE_REF" \
  || die "missing pushed candidate ref"
[[ "$(git -C "$CANDIDATE_REPO" rev-parse "$CANDIDATE_REMOTE_REF")" == "$CANDIDATE_COMMIT" ]] \
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
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 ]] || {
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another user GPU job exists"
}

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "baseline_commit=$BASELINE_COMMIT"
  echo "baseline_branch=$BASELINE_BRANCH"
  echo "candidate_commit=$CANDIDATE_COMMIT"
  echo "candidate_branch=$CANDIDATE_BRANCH"
  echo "candidate_remote_ref=$CANDIDATE_REMOTE_REF"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "run_root=$RUN_ROOT"
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
  local command=(
    "$PYTHON" inference.py --config-name profile_salvalai
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

for precision in fp32 fp16; do
  for index in 01 02 03; do
    if (( 10#$index % 2 == 1 )); then
      order=(baseline candidate)
    else
      order=(candidate baseline)
    fi
    for side in "${order[@]}"; do
      if [[ "$side" == baseline ]]; then
        repo=$BASELINE_REPO
      else
        repo=$CANDIDATE_REPO
      fi
      run_process "$repo" "$precision" "$side" "run-$index" untraced_control
    done
  done
  run_process "$BASELINE_REPO" "$precision" baseline exactness-audit exactness_audit
  run_process "$CANDIDATE_REPO" "$precision" candidate exactness-audit exactness_audit
done

cd "$CANDIDATE_REPO"
"$PYTHON" utils/analyze_lazy_startup_full_song.py \
  --run-root "$RUN_ROOT" \
  --baseline-commit "$BASELINE_COMMIT" \
  --baseline-branch "$BASELINE_BRANCH" \
  --candidate-commit "$CANDIDATE_COMMIT" \
  --candidate-branch "$CANDIDATE_BRANCH" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "lazy_start_full_song=PASS"
