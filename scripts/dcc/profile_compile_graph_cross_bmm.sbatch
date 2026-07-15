#!/usr/bin/env bash
#SBATCH --job-name=compile-graph-cross
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

die() { echo "compile-graph-cross: $*" >&2; exit 2; }

SLURM_JOB_ID=${SLURM_JOB_ID:?Run under Slurm}
REPO=${MAPPERATORINATOR_REPO:?Set the clean scout worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set the exact pushed scout commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set the scout worktree branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set the pushed scout remote ref}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
ALLOW_PARALLEL=${MAPPERATORINATOR_ALLOW_PARALLEL:-0}
RUN_ROOT="$WORK/runs/compile-graph-cross-${SLURM_JOB_ID}"
EXTENSION_ROOT="$WORK/torch_extensions/compile-graph-cross-${COMMIT}"
JOB_TMP="$WORK/tmp/compile-graph-cross-${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
[[ -f "$AUDIO" ]] || die "missing audio: $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree: $REPO"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD mismatch"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "branch mismatch"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || die "missing remote ref"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "remote mismatch"

mkdir -p "$WORK/logs" "$WORK/cache" "$WORK/cache/torch_inductor" \
  "$WORK/cache/triton" "$WORK/cache/cuda" "$EXTENSION_ROOT" "$JOB_TMP" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor/compile-graph-cross-${COMMIT}"
export TRITON_CACHE_DIR="$WORK/cache/triton/compile-graph-cross-${COMMIT}"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$EXTENSION_ROOT"
export TMPDIR="$JOB_TMP"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "missing executable: $tool"
done
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "$GPU_NAME" == *"$EXPECTED_GPU"* ]] || die "expected $EXPECTED_GPU, got $GPU_NAME"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j|%N' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 || "$ALLOW_PARALLEL" == 1 ]] || {
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job exists without parallel opt-in"
}

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "run_root=$RUN_ROOT"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "tmpdir=$TMPDIR"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  "$PYTHON" -c 'import torch; print(torch.__version__); print(torch._dynamo.list_backends())'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$PYTHON" utils/profile_compile_graph_cross_bmm.py \
  --config-name profile_salvalai \
  --output-path "$RUN_ROOT/baseline" \
  --json-path "$RUN_ROOT/analysis.json" \
  --warmup 50 \
  --iterations 1000 \
  audio_path="$AUDIO" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 > "$RUN_ROOT/stdout.txt" 2> "$RUN_ROOT/stderr.txt"

[[ -s "$RUN_ROOT/analysis.json" ]] || die "analysis.json is missing"
"$PYTHON" - "$RUN_ROOT/analysis.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["schema_version"] == 1
assert report["summary"]["measured_buckets"] == [128, 576, 640]
assert any(row.get("pass") for row in report["summary"]["modes"].values())
PY
find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
echo "compile_graph_cross_bmm_scout=PASS"
