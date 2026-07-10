#!/usr/bin/env bash
#SBATCH --job-name=opt-single-engine
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the isolated migration worktree}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed migration commit}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/optimized-single-architecture-migration}
PROFILE_CONFIG=${PROFILE_CONFIG:-profile_salvalai_smoke15}
RUN_LABEL=${RUN_LABEL:-smoke}
WORK=/work/imt11/Mapperatorinator
AUDIO="$WORK/data/salvalai.mp3"
RUN_ROOT="$WORK/runs/optimized-single-engine-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "migration worktree commit differs from EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  echo "migration worktree branch differs from EXPECTED_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "migration worktree is dirty" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "missing SALVALAI audio at $AUDIO" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$WORK/torch_extensions" "$RUN_ROOT"
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

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "commit=$EXPECTED_COMMIT"
echo "profile_config=$PROFILE_CONFIG"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  optimized_inference_mode=single
  use_server=false
  parallel=false
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_sync_cuda=true
  profile_record_token_ids=true
)

LEGACY_OVERRIDES=(
  inference_engine=v32
  inference_generation_compile=true
  inference_active_prefix_decode_loop=true
  inference_active_prefix_decode_bucket_size=64
  inference_active_prefix_decode_cuda_graph=true
  inference_active_prefix_decode_cuda_graph_warmup=0
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
  inference_stateful_monotonic_logits_processor=true
  inference_q1_bmm_cross_attention=true
  inference_decode_session_runtime=true
  inference_decode_session_cuda_graph=true
  inference_native_decode_kernels=true
  inference_native_q1_self_attention=true
  inference_native_q1_rope_cache_self_attention=true
)

OPTIMIZED_OVERRIDES=(
  inference_engine=optimized
  inference_generation_compile=false
  inference_active_prefix_decode_loop=false
  inference_active_prefix_decode_bucket_size=128
  inference_active_prefix_decode_cuda_graph=false
  inference_active_prefix_decode_cuda_graph_warmup=0
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
  inference_stateful_monotonic_logits_processor=false
  inference_q1_bmm_cross_attention=false
  inference_decode_session_runtime=false
  inference_decode_session_cuda_graph=false
  inference_native_decode_kernels=false
  inference_native_q1_self_attention=false
  inference_native_q1_rope_cache_self_attention=false
)

run_profile() {
  local mode=$1
  local label=$2
  local output="$RUN_ROOT/$label"
  local -a mode_overrides
  if [[ "$mode" == legacy ]]; then
    mode_overrides=("${LEGACY_OVERRIDES[@]}")
  elif [[ "$mode" == optimized ]]; then
    mode_overrides=("${OPTIMIZED_OVERRIDES[@]}")
  else
    echo "unknown engine mode $mode" >&2
    exit 2
  fi
  mkdir -p "$output"
  (
    cd "$REPO"
    python inference.py --config-name "$PROFILE_CONFIG" \
      output_path="$output" \
      "${COMMON_OVERRIDES[@]}" \
      "${mode_overrides[@]}"
  )
  local profile
  profile=$(find "$output" -name '*.profile.json' -type f | sort | head -n 1)
  if [[ -z "$profile" ]]; then
    echo "missing profile for $label" >&2
    exit 2
  fi
  echo "$profile" > "$RUN_ROOT/$label.profile-path.txt"
}

run_profile legacy legacy_first
run_profile optimized optimized_second
run_profile optimized optimized_first
run_profile legacy legacy_second

LEGACY_FIRST=$(<"$RUN_ROOT/legacy_first.profile-path.txt")
OPTIMIZED_SECOND=$(<"$RUN_ROOT/optimized_second.profile-path.txt")
OPTIMIZED_FIRST=$(<"$RUN_ROOT/optimized_first.profile-path.txt")
LEGACY_SECOND=$(<"$RUN_ROOT/legacy_second.profile-path.txt")

compare_pair() {
  local baseline=$1
  local candidate=$2
  local report=$3
  (
    cd "$REPO"
    python utils/summarize_inference_profile.py \
      --compare "$baseline" "$candidate" \
      --labels main_generation,timing_context \
      --require-token-equivalence \
      --require-output-equivalence \
      --json-output "$report"
  )
}

compare_pair "$LEGACY_FIRST" "$OPTIMIZED_SECOND" "$RUN_ROOT/legacy-first-compare.json"
compare_pair "$LEGACY_SECOND" "$OPTIMIZED_FIRST" "$RUN_ROOT/optimized-first-compare.json"

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("legacy-first-compare.json", "optimized-first-compare.json"):
    report = json.loads((root / name).read_text(encoding="utf-8"))
    if report.get("token_equivalence_pass") is not True:
        raise SystemExit(f"{name}: token equivalence failed")
    if report.get("output_artifact_pass") is not True:
        raise SystemExit(f"{name}: output equivalence failed")
    for label, label_report in report["reports"].items():
        metrics = label_report["performance"]["metrics"]
        if metrics["tokens_per_second"]["pct"] < -5.0:
            raise SystemExit(f"{name}: meaningful {label} throughput regression")
        if metrics["model_elapsed_seconds"]["pct"] > 5.0:
            raise SystemExit(f"{name}: meaningful {label} model-time regression")
        if metrics["total_stage_wall_seconds"]["pct"] > 5.0:
            raise SystemExit(f"{name}: meaningful stage-wall regression")

for profile_path_file in root.glob("optimized_*.profile-path.txt"):
    profile = json.loads(Path(profile_path_file.read_text().strip()).read_text())
    metadata = profile.get("metadata", {})
    if metadata.get("inference_engine") != "optimized":
        raise SystemExit("optimized selector metadata missing")
    if metadata.get("inference_generation_compile") is not False:
        raise SystemExit("requested compile metadata changed")
    if metadata.get("optimized_effective_config_version") != "accepted-fp32-270.475-v1":
        raise SystemExit("effective optimized config metadata missing")
    effective = metadata.get("optimized_effective_config", {})
    required = {
        "generation_compile": True,
        "active_prefix_decode_bucket_size": 64,
        "stateful_monotonic_logits_processor": True,
        "q1_bmm_cross_attention": True,
        "decode_session_runtime": True,
        "native_q1_rope_cache_self_attention": True,
    }
    for key, expected in required.items():
        if effective.get(key) != expected:
            raise SystemExit(f"effective config mismatch for {key}")
print("optimized_single_engine_smoke_gate=PASS")
PY

find "$RUN_ROOT" -type f \
  \( -name '*.profile.json' -o -name '*-compare.json' \) -print0 \
  | sort -z | xargs -0 sha256sum
