#!/usr/bin/env bash
#SBATCH --job-name=while-toy-r1
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §32 rung 1: standalone CUDA conditional WHILE toy graph on Turing 2080 Ti.
# No Inductor. Unique TMPDIR / TORCH_EXTENSIONS per job.

set -euo pipefail

die() { echo "while-toy-r1: $*" >&2; exit 2; }

SLURM_JOB_ID=${SLURM_JOB_ID:?}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?}
COMMIT=${MAPPERATORINATOR_COMMIT:?}
BRANCH=${MAPPERATORINATOR_BRANCH:?}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
NATIVE_BUILD_TOOLS=${MAPPERATORINATOR_NATIVE_BUILD_TOOLS:-$WORK/cache/native-build-tools}
RUN_LABEL=${RUN_LABEL:-while-toy-r1}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/imt11-while-toy-${SLURM_JOB_ID}"
JOB_EXT="$WORK/tmp/torch_extensions-while-toy-${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "missing python"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "dirty worktree"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || die "HEAD mismatch"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] || die "branch mismatch"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" || die "missing remote ref"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] || die "remote mismatch"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$NATIVE_BUILD_TOOLS/bin:$PATH"
export PYTHONPATH="$NATIVE_BUILD_TOOLS${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export TMPDIR="$JOB_TMP"
export TORCH_CUDA_ARCH_LIST=7.5
export NVIDIA_TF32_OVERRIDE=0
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

command -v ninja >/dev/null 2>&1 || die "missing ninja in $NATIVE_BUILD_TOOLS"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "$GPU_COUNT" -eq 1 ]] || die "expected exactly one GPU, got $GPU_COUNT"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "$GPU_NAME" == *"2080 Ti"* ]] || die "expected RTX 2080 Ti, got $GPU_NAME"

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "repo=$REPO"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "gpu=$GPU_NAME"
  echo "python=$PYTHON"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "tmpdir=$TMPDIR"
  echo "run_root=$RUN_ROOT"
  echo "rung=1_toy_while"
  echo "ledger=section_32"
  "$PYTHON" - <<'PY'
import torch
from cuda.bindings import runtime
required = (
    "cudaGraphConditionalHandleCreate",
    "cudaGraphAddNode",
    "cudaGraphNodeParams",
)
missing = [name for name in required if not hasattr(runtime, name)]
if missing:
    raise SystemExit(f"cuda.bindings missing conditional APIs: {missing}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
PY
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | grep -E '^(JobId|JobName|Partition|NodeList|TresPerNode|JobState)=' || true
  fi
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$PYTHON" utils/probe_cuda_conditional_while.py \
  --output-dir "$RUN_ROOT/capability" \
  --limits 1,3,10 \
  --replays 3 \
  --required-gpu-substring "2080 Ti" \
  | tee "$RUN_ROOT/stdout.txt"

[[ -f "$RUN_ROOT/capability/probe.json" ]] || die "missing probe.json"
"$PYTHON" - "$RUN_ROOT/capability/probe.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["decision"]["pass"] is True
assert report["metadata"]["conditional_type"] == "WHILE"
assert "2080 Ti" in report["metadata"]["gpu"]
assert int(report["metadata"]["runtime_version"]) >= 12030
for row in report["probes"]:
    assert int(row["observed_counter"]) == int(row["limit"])
    assert bool(row["memory_stable"])
print("while_toy_rung1=PASS")
print(f"gpu={report['metadata']['gpu']}")
print(f"runtime_version={report['metadata']['runtime_version']}")
print(f"driver_version={report['metadata']['driver_version']}")
for row in report["probes"]:
    print(
        f"limit={row['limit']} median_cuda_ms={row['cuda_event_median_ms']:.6f} "
        f"host_wall_median_s={row['host_wall_median_seconds']:.6e}"
    )
PY

echo "while_toy_rung1=PASS" | tee "$RUN_ROOT/decision.txt"
echo "artifacts=$RUN_ROOT"
