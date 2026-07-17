#!/usr/bin/env bash
#SBATCH --job-name=while-wrap-r2
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §32 rung 2: wrap tip decode step + greedy EOS tail in CUDA WHILE.
# No Inductor. Unique TMPDIR / TORCH_EXTENSIONS per job.

set -euo pipefail

die() { echo "while-wrap-r2: $*" >&2; exit 2; }

SLURM_JOB_ID=${SLURM_JOB_ID:?}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?}
COMMIT=${MAPPERATORINATOR_COMMIT:?}
BRANCH=${MAPPERATORINATOR_BRANCH:?}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
NATIVE_BUILD_TOOLS=${MAPPERATORINATOR_NATIVE_BUILD_TOOLS:-$WORK/cache/native-build-tools}
RUN_LABEL=${RUN_LABEL:-while-wrap-r2}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/imt11-while-wrap-${SLURM_JOB_ID}"
JOB_EXT="$WORK/tmp/torch_extensions-while-wrap-${SLURM_JOB_ID}"

[[ -x "$PYTHON" ]] || die "missing python"
[[ -f "$AUDIO" ]] || die "missing audio $AUDIO"
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
export TOKENIZERS_PARALLELISM=false

command -v ninja >/dev/null 2>&1 || die "missing ninja"
command -v ffmpeg >/dev/null 2>&1 || die "missing ffmpeg"
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
  echo "audio=$AUDIO"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "tmpdir=$TMPDIR"
  echo "run_root=$RUN_ROOT"
  echo "rung=2_step_wrap"
  echo "ledger=section_32"
  "$PYTHON" - <<'PY'
import torch
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected PyTorch CUDA 12.8, got {torch.version.cuda}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
PY
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
"$PYTHON" utils/profile_cuda_graph_while_step_wrap.py \
  --config-name profile_salvalai \
  --output-dir "$RUN_ROOT/step-wrap" \
  --capture-output-path "$RUN_ROOT/step-wrap/baseline-capture" \
  --source-prefix 576 \
  --active-prefix 640 \
  --fixed-steps 8 \
  --forced-stops 1,3,7 \
  audio_path="$AUDIO" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=optimized \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  profile_inference=true \
  profile_detail_ranges=false \
  profile_cuda_capture=false \
  profile_pass_kind=untraced_control \
  | tee "$RUN_ROOT/stdout.txt"

find "$RUN_ROOT" -type f -name '*.osu' -delete || true
[[ -f "$RUN_ROOT/step-wrap/real-prefix.json" ]] || die "missing real-prefix.json"

"$PYTHON" - "$RUN_ROOT/step-wrap/real-prefix.json" <<'PY'
import json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert report["decision"]["pass"] is True
assert report["metadata"]["ledger_section"] == 32
assert report["metadata"]["rung"] == 2
assert "2080 Ti" in report["metadata"]["gpu"]
assert report["fixed_work"]["while_no_post_stop_waste"] is True
# Budget probe: host parent-launch amortization vs k1 for early-stop cases.
# Window-level claim uses avg ~90 tok/window (7809/87); need ≥0.15 ms/token
# equivalent before smoke/reciprocal.
forced = report["forced_stop"]
host_ms = []
for stop, summary in forced.items():
    k1 = float(summary["reciprocal_cuda_ms"]["k1"])
    wh = float(summary["reciprocal_cuda_ms"]["while"])
    logical = int(summary["expected_logical_steps"])
    # Approximate host-amortization headroom as k1-while when WHILE avoids
    # post-stop waste relative to padded blocks; per-token vs k1 equal-work.
    delta_ms = k1 - wh
    host_ms.append((int(stop), logical, k1, wh, delta_ms, delta_ms / max(logical, 1)))
best = max(host_ms, key=lambda row: row[5])
window_avg_tokens = 7809 / 87.0
budget_ms_per_token = 0.15
clears = best[5] >= budget_ms_per_token
window_level_ms = best[5] * window_avg_tokens
print("while_step_wrap_rung2=PASS")
print(f"best_forced_stop={best[0]}")
print(f"best_delta_ms_per_token_vs_k1={best[5]:.6f}")
print(f"budget_ms_per_token={budget_ms_per_token}")
print(f"clears_token_budget={clears}")
print(f"window_level_ms_at_avg_tokens={window_level_ms:.3f}")
print(f"window_budget_ms={budget_ms_per_token * window_avg_tokens:.3f}")
# Do not fail the wrap exactness gate on budget; budget gates smoke/reciprocal.
pathlib.Path(sys.argv[1]).with_name("budget.json").write_text(
    json.dumps(
        {
            "forced_stop_rows": [
                {
                    "stop": s,
                    "logical": log,
                    "k1_ms": k1,
                    "while_ms": wh,
                    "delta_ms": d,
                    "delta_ms_per_token": dpt,
                }
                for s, log, k1, wh, d, dpt in host_ms
            ],
            "best_delta_ms_per_token_vs_k1": best[5],
            "budget_ms_per_token": budget_ms_per_token,
            "clears_token_budget": clears,
            "window_avg_tokens": window_avg_tokens,
            "window_level_ms_at_avg_tokens": window_level_ms,
            "note": "Budget uses WHILE vs equal-work k1 on forced-stop cases; "
            "negative means WHILE overhead > launch savings on this short loop.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
if not clears:
    print("budget_gate=MISS — do not open smoke/reciprocal yet; measure larger window or host-launch probe")
else:
    print("budget_gate=PASS — eligible for rung 3 smoke")
PY

echo "while_step_wrap_rung2=PASS" | tee "$RUN_ROOT/decision.txt"
echo "artifacts=$RUN_ROOT"
