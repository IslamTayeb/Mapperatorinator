#!/usr/bin/env bash
#SBATCH --job-name=weighted-h8
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:30:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
SHARED_REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated weighted-bucket DCC worktree}
WORK=/work/imt11/Mapperatorinator
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed weighted-H8 verifier commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-weighted-bucket-verifier}
SOURCE_REPORT="$WORK/runs/weighted-prefix-source-49559386-1896b19/weighted-prefix-source.json"
SOURCE_PROFILE="$WORK/runs/inference-denominator-five_smoke-49543717-a709b86/serial_multi_song-five_smoke-49543717-a709b86/lambada/repeat01/run05.profile.json"
SOURCE_AUDIO="$WORK/data/five-song-profile/lambada.mp3"
SOURCE_REPORT_SHA=8757f97874fd7249420feb1d2e03de43a646da8187cdebb34d3ee306baf7de62
SOURCE_PROFILE_SHA=05e49f8799fb9611c2d5974877b83fc57940edafd45940ef7ed7b504ab12145f
SOURCE_AUDIO_SHA=1e692d9ffbc2a3051219ac19e96f2c32b428fa0e4fc36278022866e19e6e6183
CEILING_REPORT="$REPO/notes/inference-weighted-bucket-ceiling-report.json"
CEILING_REPORT_SHA=44a680ab29867e3aea8dde713127bdb154ef42a316fd2834bc75afbbc0927fc9
MODEL_SNAPSHOT=74f22583400d259bf424819e11027c17933efe54
MODEL_CONFIG_SHA=ff96b8c059179978c93b6f938e39fac945d5682a3495789c00cc159d571a1a22
MODEL_CACHE="$WORK/cache/huggingface/models--OliBomby--Mapperatorinator-v32"
MODEL_REF="$MODEL_CACHE/refs/main"
MODEL_CONFIG="$MODEL_CACHE/snapshots/$MODEL_SNAPSHOT/gamemode=0/config.json"

if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  echo "isolated weighted-bucket checkout is missing at $REPO" >&2
  exit 2
fi
if [[ "$(realpath "$REPO")" == "$(realpath "$SHARED_REPO")" ]]; then
  echo "refusing weighted H8 from shared DCC checkout" >&2
  exit 2
fi
for path in "$SOURCE_REPORT" "$SOURCE_PROFILE" "$SOURCE_AUDIO" "$CEILING_REPORT" "$MODEL_REF" "$MODEL_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "weighted H8 prerequisite is missing at $path" >&2
    exit 2
  fi
done
for pair in \
  "$SOURCE_REPORT:$SOURCE_REPORT_SHA" \
  "$SOURCE_PROFILE:$SOURCE_PROFILE_SHA" \
  "$SOURCE_AUDIO:$SOURCE_AUDIO_SHA" \
  "$CEILING_REPORT:$CEILING_REPORT_SHA" \
  "$MODEL_CONFIG:$MODEL_CONFIG_SHA"; do
  path=${pair%%:*}
  expected=${pair##*:}
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    echo "weighted H8 SHA-256 mismatch at $path" >&2
    exit 2
  fi
done
if [[ "$(cat "$MODEL_REF")" != "$MODEL_SNAPSHOT" ]]; then
  echo "weighted H8 model ref differs from pinned snapshot" >&2
  exit 2
fi

COMMIT=$(git -C "$REPO" rev-parse HEAD)
BRANCH=$(git -C "$REPO" branch --show-current)
SHORT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD)
if [[ "$COMMIT" != "$EXPECTED_COMMIT" || "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "checkout is $BRANCH@$COMMIT, expected $EXPECTED_BRANCH@$EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "weighted H8 checkout is dirty; refusing verifier" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

RUN_ROOT="$WORK/runs/weighted-h8-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/weighted-h8.json"
TELEMETRY="$RUN_ROOT/nvidia-smi.csv"
mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" "$WORK/cache/torch_inductor" \
  "$WORK/cache/triton" "$WORK/cache/cuda" "$WORK/torch_extensions" "$RUN_ROOT"
export PATH="$ENV/bin:$PATH"
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

cd "$REPO"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}"
echo "commit=$COMMIT"
echo "branch=$BRANCH"
echo "source_contract_sha256=de55e2d16c4b085c5cce62e83be8cd4f83bb23f2a40e87f80be23a88f216b99a"
echo "run_root=$RUN_ROOT"

python -m py_compile \
  osuT5/osuT5/inference/optimized/batch/weighted_h8.py \
  utils/verify_optimized_weighted_h8.py
python - "$SOURCE_REPORT" <<'PY'
import sys
from pathlib import Path
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    load_reviewed_source_capture_report,
)

load_reviewed_source_capture_report(Path(sys.argv[1]))
print("reviewed_source_capture=PASS")
PY

nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

set +e
python utils/verify_optimized_weighted_h8.py \
  --config-name profile_salvalai_smoke15 \
  --source-report "$SOURCE_REPORT" \
  --source-profile "$SOURCE_PROFILE" \
  --source-audio "$SOURCE_AUDIO" \
  --weighted-ceiling-report "$CEILING_REPORT" \
  --report-path "$REPORT" \
  --capture-warmup-repeats 1 \
  --comparison-top-k 20 \
  --q1-bmm-cross-attention \
  --native-q1-self-attention \
  --native-q1-rope-cache-self-attention \
  audio_path="$SOURCE_AUDIO" \
  output_path="$RUN_ROOT/output" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  inference_engine=v32 \
  optimized_inference_mode=single \
  use_server=false \
  parallel=false \
  cfg_scale=1.0 \
  num_beams=1 \
  seed=12345 \
  inference_generation_compile=false \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=64 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_warmup=0 \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1 \
  inference_stateful_monotonic_logits_processor=true \
  inference_q1_bmm_cross_attention=true \
  inference_decode_session_runtime=true \
  inference_decode_session_cuda_graph=true \
  inference_native_decode_kernels=true \
  inference_native_q1_self_attention=true \
  inference_native_q1_rope_cache_self_attention=true
VERIFY_STATUS=$?
set -e

if [[ ! -f "$REPORT" ]]; then
  echo "weighted H8 verifier exited without a structured report" >&2
  exit 2
fi

if (( VERIFY_STATUS != 0 )); then
  python - "$REPORT" "$COMMIT" "${SLURM_JOB_ID:-manual}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("pass") is not False or report.get("exactness_pass") is not False:
    raise SystemExit("weighted H8 failure report did not preserve rejection status")
runtime = report.get("runtime_metadata", {})
if runtime.get("git_commit") != sys.argv[2]:
    raise SystemExit("weighted H8 failure report commit differs from wrapper")
if runtime.get("slurm_job_id") != sys.argv[3]:
    raise SystemExit("weighted H8 failure report job differs from wrapper")
failure = report.get("failure", {})
if not isinstance(failure.get("type"), str) or not isinstance(
    failure.get("message"), str
):
    raise SystemExit("weighted H8 failure report is malformed")
print("weighted_h8_failure_report_validation=PASS")
print(f"failure_type={failure['type']}")
print(f"failure_message={failure['message']}")
PY
  sha256sum "$REPORT"
  exit "$VERIFY_STATUS"
fi

python - "$REPORT" "$COMMIT" "${SLURM_JOB_ID:-manual}" <<'PY'
import json
import sys
from osuT5.osuT5.inference.optimized.batch.weighted_h8 import (
    validate_weighted_h8_report,
)

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
validate_weighted_h8_report(report)
if report["runtime_metadata"]["git_commit"] != sys.argv[2]:
    raise SystemExit("weighted H8 report commit differs from wrapper")
if report["runtime_metadata"]["slurm_job_id"] != sys.argv[3]:
    raise SystemExit("weighted H8 report job differs from wrapper")
print("weighted_h8_report_validation=PASS")
print(f"exactness_pass={str(report['exactness_pass']).lower()}")
print(f"worst_candidate_tps={report['performance']['worst_candidate_wall_tokens_per_second']}")
print(f"minimum_same_order_gain={report['performance']['minimum_same_order_gain_fraction']}")
print(f"required_tps={report['performance']['required_weighted_complete_tokens_per_second']}")
print("phase_b_authorized=false")
print("scheduler_or_runtime_wiring_authorized=false")
print(
    "performance_gate_status="
    + ("PASS" if report["pass"] is True else "REJECTED")
)
PY
sha256sum "$REPORT"
python - "$REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report["pass"] is not True:
    raise SystemExit("weighted H8 exact report failed the absolute or 5% keep bar")
PY
