#!/usr/bin/env bash
#SBATCH --job-name=batched-super-graph
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=08:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean graph worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed graph commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the graph branch}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
RUN_LABEL=${RUN_LABEL:-batched-super-timing-graph}
SMOKE_ITERATIONS=${SMOKE_ITERATIONS:-1}
FULL_ITERATIONS=${FULL_ITERATIONS:-20}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID:-manual}"
REPORT_ROOT="$RUN_ROOT/reports"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing executable Python at $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "missing graph worktree at $REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "graph worktree is dirty at $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$COMMIT" ]]; then
  echo "graph worktree does not match MAPPERATORINATOR_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$BRANCH" ]]; then
  echo "graph worktree is not on MAPPERATORINATOR_BRANCH=$BRANCH" >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing SALVALAI audio at $AUDIO" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT" "$REPORT_ROOT"
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
export PATH="$(dirname "$PYTHON"):$PATH"

for tool in ffmpeg ffprobe; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required audio tool: $tool" >&2
    exit 2
  fi
done

{
  echo "host=$(hostname)"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "repo=$REPO"
  echo "python=$PYTHON"
  echo "audio=$AUDIO"
  echo "smoke_iterations=$SMOKE_ITERATIONS"
  echo "full_iterations=$FULL_ITERATIONS"
  echo "run_root=$RUN_ROOT"
  git -C "$REPO" status --short --branch
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"torch_cuda={torch.version.cuda}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  command -v nsys || echo "nsys=missing"
  command -v ncu || echo "ncu=missing"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job -o "$SLURM_JOB_ID"
  fi
} | tee "$RUN_ROOT/preflight.txt"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ "$GPU_NAME" != *"2080 Ti"* ]]; then
  echo "expected RTX 2080 Ti, got $GPU_NAME" >&2
  exit 2
fi

nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
finalize() {
  kill "$SMI_PID" 2>/dev/null || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -type f ! -name sha256sums.txt -print0 \
      | sort -z \
      | xargs -0 sha256sum > "$RUN_ROOT/sha256sums.txt"
  fi
}
trap finalize EXIT

cd "$REPO"

run_profile() {
  local phase=$1
  local precision=$2
  local batch=$3
  local mode=$4
  local repetition=$5
  local order=$6
  local iterations=$7
  local reference=${8:-}
  local report="$REPORT_ROOT/${phase}-${precision}-b${batch}-${mode}-r${repetition}.json"
  local args=(
    utils/profile_batched_super_timing_graph.py
    --config-name profile_salvalai
    --mode "$mode"
    --batch-size "$batch"
    --seed 12345
    --phase "$phase"
    --repetition "$repetition"
    --order "$order"
    --report-path "$report"
  )
  if [[ -n "$reference" ]]; then
    args+=(--reference-report "$reference")
  fi
  args+=(
    "audio_path=$AUDIO"
    "output_path=$RUN_ROOT/unused-output"
    device=cuda
    "precision=$precision"
    attn_implementation=sdpa
    inference_engine=v32
    use_server=false
    parallel=false
    super_timing=false
    timer_num_beams=1
    timer_cfg_scale=1.0
    "timer_iterations=$iterations"
    max_batch_size=16
    seed=12345
  )
  set +e
  "$PYTHON" "${args[@]}" 2>&1 | tee "$RUN_ROOT/${phase}-${precision}-b${batch}-${mode}-r${repetition}.log"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ ! -f "$report" ]]; then
    echo "profile did not produce $report (status=$status)" >&2
    return "$status"
  fi
  if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
    echo "profile failed before a declared gate (status=$status)" >&2
    return "$status"
  fi
  return "$status"
}

compare_reports() {
  local eager=$1
  local graph=$2
  "$PYTHON" - "$eager" "$graph" <<'PY'
import json
import sys
from pathlib import Path
from utils.profile_batched_super_timing_graph import compare_exact

eager = json.loads(Path(sys.argv[1]).read_text())
graph = json.loads(Path(sys.argv[2]).read_text())
comparison = compare_exact(eager, graph)
comparison["eager_gates_pass"] = bool(eager.get("gates", {}).get("pass"))
comparison["graph_gates_pass"] = bool(graph.get("gates", {}).get("pass"))
comparison["pass"] = all((
    comparison["pass"],
    comparison["eager_gates_pass"],
    comparison["graph_gates_pass"],
))
print(json.dumps(comparison, indent=2))
raise SystemExit(0 if comparison["pass"] else 3)
PY
}

for batch in 1 4 8; do
  eager="$REPORT_ROOT/smoke-fp32-b${batch}-eager-r1.json"
  graph="$REPORT_ROOT/smoke-fp32-b${batch}-graph-r1.json"
  run_profile smoke fp32 "$batch" eager 1 1 "$SMOKE_ITERATIONS"
  run_profile smoke fp32 "$batch" graph 1 2 "$SMOKE_ITERATIONS" "$eager" || true
  compare_reports "$eager" "$graph"
done

"$PYTHON" utils/summarize_batched_super_timing_graph.py \
  --report-root "$REPORT_ROOT" \
  --precision fp32 \
  --phase smoke \
  --required-repetitions 1 \
  --output-path "$RUN_ROOT/fp32-smoke-summary.json"

for batch in 1 2 4 8 16; do
  eager1="$REPORT_ROOT/full-fp32-b${batch}-eager-r1.json"
  graph1="$REPORT_ROOT/full-fp32-b${batch}-graph-r1.json"
  run_profile full fp32 "$batch" eager 1 1 "$FULL_ITERATIONS"
  run_profile full fp32 "$batch" graph 1 2 "$FULL_ITERATIONS" "$eager1" || true
  if ! compare_reports "$eager1" "$graph1"; then
    echo "stopping FP32 B${batch} after repetition 1 exactness failure" | tee -a "$RUN_ROOT/stopped-candidates.txt"
    continue
  fi

  eager2="$REPORT_ROOT/full-fp32-b${batch}-eager-r2.json"
  graph2="$REPORT_ROOT/full-fp32-b${batch}-graph-r2.json"
  run_profile full fp32 "$batch" graph 2 1 "$FULL_ITERATIONS" || true
  run_profile full fp32 "$batch" eager 2 2 "$FULL_ITERATIONS"
  if ! compare_reports "$eager2" "$graph2"; then
    echo "stopping FP32 B${batch} after repetition 2 nondeterminism" | tee -a "$RUN_ROOT/stopped-candidates.txt"
    continue
  fi

  eager3="$REPORT_ROOT/full-fp32-b${batch}-eager-r3.json"
  run_profile full fp32 "$batch" eager 3 1 "$FULL_ITERATIONS"
  run_profile full fp32 "$batch" graph 3 2 "$FULL_ITERATIONS" "$eager3" || true
done

set +e
"$PYTHON" utils/summarize_batched_super_timing_graph.py \
  --report-root "$REPORT_ROOT" \
  --precision fp32 \
  --phase full \
  --required-repetitions 3 \
  --output-path "$RUN_ROOT/fp32-full-summary.json"
FP32_SUMMARY_STATUS=$?
set -e
if [[ "$FP32_SUMMARY_STATUS" -ne 0 && "$FP32_SUMMARY_STATUS" -ne 4 ]]; then
  exit "$FP32_SUMMARY_STATUS"
fi

FP32_PROMOTION=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["promotion_pass"]).lower())' "$RUN_ROOT/fp32-full-summary.json")
FINAL_STATUS=0
FP16_PROMOTION=not_run
if [[ "$FP32_PROMOTION" == "true" ]]; then
  WINNING_BATCH=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winning_batch_size"])' "$RUN_ROOT/fp32-full-summary.json")
  FP16_FAILED=0
  for repetition in 1 2 3; do
    eager="$REPORT_ROOT/full-fp16-b${WINNING_BATCH}-eager-r${repetition}.json"
    graph="$REPORT_ROOT/full-fp16-b${WINNING_BATCH}-graph-r${repetition}.json"
    if [[ "$repetition" -eq 2 ]]; then
      run_profile full fp16 "$WINNING_BATCH" graph "$repetition" 1 "$FULL_ITERATIONS" || true
      run_profile full fp16 "$WINNING_BATCH" eager "$repetition" 2 "$FULL_ITERATIONS"
      if ! compare_reports "$eager" "$graph"; then
        FP16_FAILED=1
        break
      fi
    else
      run_profile full fp16 "$WINNING_BATCH" eager "$repetition" 1 "$FULL_ITERATIONS"
      run_profile full fp16 "$WINNING_BATCH" graph "$repetition" 2 "$FULL_ITERATIONS" "$eager" || true
      if ! compare_reports "$eager" "$graph"; then
        FP16_FAILED=1
        break
      fi
    fi
  done
  set +e
  "$PYTHON" utils/summarize_batched_super_timing_graph.py \
    --report-root "$REPORT_ROOT" \
    --precision fp16 \
    --phase full \
    --required-repetitions 3 \
    --output-path "$RUN_ROOT/fp16-full-summary.json"
  FP16_SUMMARY_STATUS=$?
  set -e
  if [[ "$FP16_SUMMARY_STATUS" -ne 0 && "$FP16_SUMMARY_STATUS" -ne 4 ]]; then
    exit "$FP16_SUMMARY_STATUS"
  fi
  FP16_PROMOTION=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["promotion_pass"]).lower())' "$RUN_ROOT/fp16-full-summary.json")
  if [[ "$FP16_FAILED" -ne 0 || "$FP16_PROMOTION" != "true" ]]; then
    FINAL_STATUS=5
  fi
else
  FINAL_STATUS=4
fi

"$PYTHON" - "$RUN_ROOT/decision.json" "$FP32_PROMOTION" "$FP16_PROMOTION" "$FINAL_STATUS" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "fp32_promotion_pass": sys.argv[2] == "true",
    "fp16_promotion": sys.argv[3],
    "exit_status": int(sys.argv[4]),
}, indent=2) + "\n")
PY
cat "$RUN_ROOT/fp32-full-summary.json"
if [[ -f "$RUN_ROOT/fp16-full-summary.json" ]]; then
  cat "$RUN_ROOT/fp16-full-summary.json"
fi
exit "$FINAL_STATUS"
