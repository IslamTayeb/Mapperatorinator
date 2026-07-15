#!/usr/bin/env bash
#SBATCH --job-name=combined-start-song
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
  echo "combined-start-song: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "missing or empty artifact: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside Slurm}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
LAZY_REPO=${MAPPERATORINATOR_LAZY_REPO:?Set the clean lazy-parent worktree}
LAZY_COMMIT=${MAPPERATORINATOR_LAZY_COMMIT:?Set the lazy-parent commit}
LAZY_BRANCH=${MAPPERATORINATOR_LAZY_BRANCH:?Set the lazy-parent branch}
AOT_REPO=${MAPPERATORINATOR_AOT_REPO:?Set the clean AOT-parent worktree}
AOT_COMMIT=${MAPPERATORINATOR_AOT_COMMIT:?Set the AOT-parent commit}
AOT_BRANCH=${MAPPERATORINATOR_AOT_BRANCH:?Set the AOT-parent branch}
COMBINED_REPO=${MAPPERATORINATOR_COMBINED_REPO:?Set the clean combined worktree}
COMBINED_COMMIT=${MAPPERATORINATOR_COMBINED_COMMIT:?Set the combined commit}
COMBINED_BRANCH=${MAPPERATORINATOR_COMBINED_BRANCH:?Set the combined branch}
COMBINED_REMOTE_REF=${MAPPERATORINATOR_COMBINED_REMOTE_REF:?Set the pushed combined ref}
MANIFEST=${MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST:?Set the validated AOT manifest}
NATIVE_CACHE=${MAPPERATORINATOR_NATIVE_EXTENSION_CACHE:?Set the populated native-extension cache}
RUN_LABEL=${RUN_LABEL:-combined-start-full-song}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"

export NVIDIA_TF32_OVERRIDE=0

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ -f "$AUDIO" ]] || die "missing audio: $AUDIO"
[[ -s "$MANIFEST" ]] || die "missing native-extension manifest: $MANIFEST"
[[ -d "$NATIVE_CACHE" ]] || die "missing native-extension cache: $NATIVE_CACHE"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"

verify_repo() {
  local repo=$1
  local commit=$2
  local branch=$3
  [[ -d "$repo/.git" || -f "$repo/.git" ]] || die "missing worktree: $repo"
  [[ -z "$(git -C "$repo" status --porcelain)" ]] || die "dirty worktree: $repo"
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] \
    || die "commit mismatch: $repo"
  [[ "$(git -C "$repo" branch --show-current)" == "$branch" ]] \
    || die "branch mismatch: $repo"
}

verify_repo "$LAZY_REPO" "$LAZY_COMMIT" "$LAZY_BRANCH"
verify_repo "$AOT_REPO" "$AOT_COMMIT" "$AOT_BRANCH"
verify_repo "$COMBINED_REPO" "$COMBINED_COMMIT" "$COMBINED_BRANCH"
git -C "$COMBINED_REPO" show-ref --verify --quiet "$COMBINED_REMOTE_REF" \
  || die "missing pushed combined ref"
[[ "$(git -C "$COMBINED_REPO" rev-parse "$COMBINED_REMOTE_REF")" == "$COMBINED_COMMIT" ]] \
  || die "pushed combined ref mismatch"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$NATIVE_CACHE"
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
  echo "lazy_commit=$LAZY_COMMIT"
  echo "lazy_branch=$LAZY_BRANCH"
  echo "aot_commit=$AOT_COMMIT"
  echo "aot_branch=$AOT_BRANCH"
  echo "combined_commit=$COMBINED_COMMIT"
  echo "combined_branch=$COMBINED_BRANCH"
  echo "combined_remote_ref=$COMBINED_REMOTE_REF"
  echo "manifest=$MANIFEST"
  echo "manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')"
  echo "native_cache=$NATIVE_CACHE"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  echo "run_root=$RUN_ROOT"
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
  local variant=$3
  local name=$4
  local pass_kind=$5
  local manifest_mode=$6
  local run_dir="$RUN_ROOT/$precision/$variant/$name"
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
  printf '%s\n' "$manifest_mode" > "$run_dir/manifest-mode.txt"
  local start finish exit_code
  start=$(date +%s%N)
  set +e
  (
    cd "$repo"
    if [[ "$manifest_mode" == direct ]]; then
      export MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST="$MANIFEST"
    elif [[ "$manifest_mode" == fallback ]]; then
      unset MAPPERATORINATOR_NATIVE_EXTENSION_MANIFEST
    else
      die "unknown manifest mode: $manifest_mode"
    fi
    "${command[@]}"
  ) > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  exit_code=$?
  set -e
  finish=$(date +%s%N)
  printf '{"started_unix_ns":%s,"finished_unix_ns":%s,"elapsed_ns":%s,"exit_code":%s}\n' \
    "$start" "$finish" "$((finish - start))" "$exit_code" > "$run_dir/process-wall.json"
  [[ $exit_code -eq 0 ]] || die "$precision/$variant/$name failed"
  capture_paths "$run_dir"
}

repo_for() {
  case "$1" in
    lazy_parent) printf '%s\n' "$LAZY_REPO" ;;
    aot_parent) printf '%s\n' "$AOT_REPO" ;;
    combined|combined_fallback) printf '%s\n' "$COMBINED_REPO" ;;
    *) die "unknown variant: $1" ;;
  esac
}

manifest_mode_for() {
  case "$1" in
    lazy_parent|combined_fallback) printf '%s\n' fallback ;;
    aot_parent|combined) printf '%s\n' direct ;;
    *) die "unknown variant: $1" ;;
  esac
}

for precision in fp32 fp16; do
  for index in 01 02 03; do
    case "$index" in
      01) order=(lazy_parent aot_parent combined) ;;
      02) order=(combined aot_parent lazy_parent) ;;
      03) order=(aot_parent lazy_parent combined) ;;
    esac
    for variant in "${order[@]}"; do
      run_process \
        "$(repo_for "$variant")" "$precision" "$variant" "run-$index" \
        untraced_control "$(manifest_mode_for "$variant")"
    done
  done
  for variant in lazy_parent aot_parent combined combined_fallback; do
    run_process \
      "$(repo_for "$variant")" "$precision" "$variant" exactness-audit \
      exactness_audit "$(manifest_mode_for "$variant")"
  done
done

cd "$COMBINED_REPO"
"$PYTHON" utils/analyze_combined_startup_full_song.py \
  --run-root "$RUN_ROOT" \
  --lazy-commit "$LAZY_COMMIT" \
  --lazy-branch "$LAZY_BRANCH" \
  --aot-commit "$AOT_COMMIT" \
  --aot-branch "$AOT_BRANCH" \
  --combined-commit "$COMBINED_COMMIT" \
  --combined-branch "$COMBINED_BRANCH" \
  --output "$RUN_ROOT/analysis.json" \
  --text-output "$RUN_ROOT/analysis.txt"

require_nonempty "$RUN_ROOT/analysis.json"
require_nonempty "$RUN_ROOT/analysis.txt"
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "COMBINED_STARTUP_FULL_SONG_PASS"
