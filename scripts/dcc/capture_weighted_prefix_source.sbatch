#!/usr/bin/env bash
#SBATCH --job-name=weighted-prefix-source
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
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed capture-only verifier commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-weighted-bucket-verifier}
SOURCE_PROFILE="$WORK/runs/inference-denominator-five_smoke-49543717-a709b86/serial_multi_song-five_smoke-49543717-a709b86/lambada/repeat01/run05.profile.json"
SOURCE_AUDIO="$WORK/data/five-song-profile/lambada.mp3"
SOURCE_PROFILE_SHA=05e49f8799fb9611c2d5974877b83fc57940edafd45940ef7ed7b504ab12145f
SOURCE_AUDIO_SHA=1e692d9ffbc2a3051219ac19e96f2c32b428fa0e4fc36278022866e19e6e6183
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
  echo "refusing to run weighted-prefix capture from shared DCC checkout" >&2
  exit 2
fi
for path in "$SOURCE_PROFILE" "$SOURCE_AUDIO"; do
  if [[ ! -f "$path" ]]; then
    echo "weighted-prefix source is missing at $path" >&2
    exit 2
  fi
done
for path in "$MODEL_REF" "$MODEL_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "weighted-prefix pinned model source is missing at $path" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$SOURCE_PROFILE" | awk '{print $1}')" != "$SOURCE_PROFILE_SHA" ]]; then
  echo "weighted-prefix source profile SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(sha256sum "$SOURCE_AUDIO" | awk '{print $1}')" != "$SOURCE_AUDIO_SHA" ]]; then
  echo "weighted-prefix source audio SHA-256 mismatch" >&2
  exit 2
fi
if [[ "$(cat "$MODEL_REF")" != "$MODEL_SNAPSHOT" ]]; then
  echo "weighted-prefix model cache ref does not match pinned snapshot" >&2
  exit 2
fi
if [[ "$(sha256sum "$MODEL_CONFIG" | awk '{print $1}')" != "$MODEL_CONFIG_SHA" ]]; then
  echo "weighted-prefix gamemode-0 model config SHA-256 mismatch" >&2
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
  echo "weighted-bucket checkout is dirty; refusing capture" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

RUN_ROOT="$WORK/runs/weighted-prefix-source-${SLURM_JOB_ID:-manual}-${SHORT_COMMIT}"
REPORT="$RUN_ROOT/weighted-prefix-source.json"
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
echo "repo=$REPO"
echo "source_profile=$SOURCE_PROFILE"
echo "source_audio=$SOURCE_AUDIO"
echo "run_root=$RUN_ROOT"
python -m py_compile \
  osuT5/osuT5/inference/optimized/batch/weighted_bucket.py \
  utils/verify_optimized_weighted_prefix_source.py
python - <<'PY'
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    validate_reviewed_context_replay_pin,
)

validate_reviewed_context_replay_pin()
PY
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$TELEMETRY" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

python utils/verify_optimized_weighted_prefix_source.py \
  --config-name profile_salvalai_smoke15 \
  --source-profile "$SOURCE_PROFILE" \
  --source-audio "$SOURCE_AUDIO" \
  --report-path "$REPORT" \
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

python - "$REPORT" <<'PY'
import json
import sys
from osuT5.osuT5.inference.optimized.batch.weighted_bucket import (
    canonical_json_sha256,
    validate_source_contract,
)

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
contract = {
    key: value
    for key, value in report.items()
    if key not in {"rng_advance", "runtime_metadata", "source_contract_sha256"}
}
validate_source_contract(contract)
digest = canonical_json_sha256(contract)
if digest != report["source_contract_sha256"]:
    raise ValueError("weighted source contract digest does not self-validate")
if contract["maintainer_boundary"]["phase_a_authorized"] is not False:
    raise ValueError("capture-only report cannot authorize Phase A")
runtime = report.get("runtime_metadata", {})
if runtime.get("capture_only") is not True or runtime.get("h8_executed") is not False:
    raise ValueError("weighted source report crossed the capture-only H8 boundary")
if runtime.get("scheduler_or_runtime_wiring_authorized") is not False:
    raise ValueError("weighted source report cannot authorize scheduler/runtime wiring")
if (
        runtime.get("fresh_context_replay_match") is not True
        or runtime.get("live_context_evidence_match") is not True
):
    raise ValueError("weighted source report lost a reviewed-context gate")
print("source_contract_validation=PASS")
print(f"source_contract_sha256={digest}")
print("phase_a_authorized=false")
print("h8_executed=false")
PY
sha256sum "$REPORT"
