#!/usr/bin/env bash
#SBATCH --job-name=static-server-regression
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=02:00:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

set -euo pipefail

ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator
CONTROL_REPO=${CONTROL_REPO:?Set CONTROL_REPO to the clean pre-migration worktree}
CANDIDATE_REPO=${CANDIDATE_REPO:?Set CANDIDATE_REPO to the clean migration worktree}
CONTROL_COMMIT=${CONTROL_COMMIT:?Set CONTROL_COMMIT}
CANDIDATE_COMMIT=${CANDIDATE_COMMIT:?Set CANDIDATE_COMMIT}
SONG_LIST=${SONG_LIST:-$WORK/data/five-song-profile/songs.yaml}
RUN_LABEL=${RUN_LABEL:-smoke15}
RUN_ORDER=${RUN_ORDER:-control_first}
RUN_ROOT="$WORK/runs/static-server-regression-${RUN_LABEL}-${SLURM_JOB_ID:-manual}"

for repo in "$CONTROL_REPO" "$CANDIDATE_REPO"; do
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "missing regression worktree at $repo" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
    echo "regression worktree is dirty at $repo" >&2
    git -C "$repo" status --short >&2
    exit 2
  fi
done
if [[ "$(git -C "$CONTROL_REPO" rev-parse HEAD)" != "$CONTROL_COMMIT" ]]; then
  echo "control commit changed" >&2
  exit 2
fi
if [[ "$(git -C "$CANDIDATE_REPO" rev-parse HEAD)" != "$CANDIDATE_COMMIT" ]]; then
  echo "candidate commit changed" >&2
  exit 2
fi
if [[ ! -f "$SONG_LIST" ]]; then
  echo "missing five-song list at $SONG_LIST" >&2
  exit 2
fi

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$RUN_ROOT/extensions" "$RUN_ROOT/import-audits"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-manual}"
echo "control_commit=$CONTROL_COMMIT"
echo "candidate_commit=$CANDIDATE_COMMIT"
echo "song_list=$SONG_LIST"
echo "run_order=$RUN_ORDER"
echo "run_root=$RUN_ROOT"
nvidia-smi -L
nvidia-smi \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu \
  --format=csv -l 1 > "$RUN_ROOT/nvidia-smi.csv" &
SMI_PID=$!
trap 'kill "$SMI_PID" 2>/dev/null || true' EXIT

COMMON_OVERRIDES=(
  device=cuda
  precision=fp32
  attn_implementation=sdpa
  inference_engine=v32
  optimized_inference_mode=single
  use_server=true
  parallel=false
  max_batch_size=5
  server_batch_timeout=0.2
  cfg_scale=1.0
  num_beams=1
  seed=12345
  profile_inference=true
  profile_sync_cuda=true
  profile_record_token_ids=true
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

audit_import_state() {
  local repo=$1
  local label=$2
  local cold_requirement=$3
  local extension_dir="$RUN_ROOT/extensions/$label"
  local audit_path="$RUN_ROOT/import-audits/$label.json"
  mkdir -p "$extension_dir"
  if find "$extension_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "$label extension directory was not empty before the import audit" >&2
    exit 2
  fi
  (
    cd "$repo"
    TORCH_EXTENSIONS_DIR="$extension_dir" python - "$audit_path" "$cold_requirement" <<'PY'
import importlib
import json
from pathlib import Path
import sys

audit_path = Path(sys.argv[1])
cold_requirement = sys.argv[2]
audits = []
for module_name in (
    "inference",
    "osuT5.osuT5.inference.processor",
    "osuT5.osuT5.inference.server",
    "osuT5.osuT5.model.custom_transformers.modeling_varwhisper",
):
    importlib.import_module(module_name)
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("osuT5.osuT5.inference.optimized")
        or name.startswith("mapperatorinator_q1_")
        or name
        in {
            "osuT5.osuT5.inference.native_decoder_layer",
            "osuT5.osuT5.inference.native_linear",
            "osuT5.osuT5.inference.native_q1_attention",
            "torch.utils.cpp_extension",
        }
    )
    audits.append({"import": module_name, "forbidden_modules": forbidden_modules})
    if cold_requirement == "require_cold" and forbidden_modules:
        raise SystemExit(
            f"{module_name} eagerly imported optimized/native modules: {forbidden_modules}"
        )
audit_path.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
PY
  )
  if find "$extension_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "$label default V32 import populated its isolated extension directory" >&2
    find "$extension_dir" -mindepth 1 -maxdepth 2 -print >&2
    exit 1
  fi
}

run_suite() {
  local repo=$1
  local label=$2
  local output="$RUN_ROOT/$label"
  local extension_dir="$RUN_ROOT/extensions/$label"
  mkdir -p "$output"
  (
    cd "$repo"
    export TORCH_EXTENSIONS_DIR="$extension_dir"
    python utils/profile_static_server_batch.py \
      --config-name profile_salvalai_smoke15 \
      --song-list "$SONG_LIST" \
      --suite-id "${label}-${SLURM_JOB_ID:-manual}" \
      --output-root "$output" \
      --repeats 1 \
      --max-workers 5 \
      "${COMMON_OVERRIDES[@]}"
  )
  local manifest
  manifest=$(find "$output" -name static_server_batch_manifest.json -type f | sort | head -n 1)
  if [[ -z "$manifest" ]]; then
    echo "missing static-server manifest for $label" >&2
    exit 2
  fi
  echo "$manifest" > "$RUN_ROOT/$label.manifest-path.txt"
}

# The pinned pre-migration control is known to import the legacy native q1
# module and torch cpp-extension machinery eagerly. Record that frozen baseline;
# require the migrated candidate to remove those imports.
audit_import_state "$CONTROL_REPO" control record_only
audit_import_state "$CANDIDATE_REPO" candidate require_cold
case "$RUN_ORDER" in
  control_first)
    run_suite "$CONTROL_REPO" control
    run_suite "$CANDIDATE_REPO" candidate
    ;;
  candidate_first)
    run_suite "$CANDIDATE_REPO" candidate
    run_suite "$CONTROL_REPO" control
    ;;
  *)
    echo "RUN_ORDER must be control_first or candidate_first" >&2
    exit 2
    ;;
esac

CONTROL_MANIFEST=$(<"$RUN_ROOT/control.manifest-path.txt")
CANDIDATE_MANIFEST=$(<"$RUN_ROOT/candidate.manifest-path.txt")
set +e
(
  cd "$CANDIDATE_REPO"
  python utils/summarize_inference_profile.py \
    --compare-static-server "$CONTROL_MANIFEST" "$CANDIDATE_MANIFEST" \
    --strict \
    --json-output "$RUN_ROOT/compare-static-server.json"
)
STRICT_COMPARE_STATUS=$?
set -e

python - "$CONTROL_MANIFEST" "$CANDIDATE_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

expected_metadata = {
    "inference_engine": "v32",
    "optimized_inference_mode": "single",
    "use_server": True,
    "server_rng_policy": "shared_global",
    "token_equivalence_status": "not_checked_shared_server_rng",
    "rng_reset_policy": "server_global_rng_shared_across_concurrent_requests",
    "server_seed_applied": False,
}
expected_generation = {
    "use_server": True,
    "server_rng_policy": "shared_global",
    "token_equivalence_status": "not_checked_shared_server_rng",
    "optimized_effective_config_version": None,
    "optimized_runtime_owner": None,
    "optimized_result_class": None,
    "generation_compile_enabled": False,
    "stateful_monotonic_logits_processor": False,
    "q1_bmm_cross_attention_enabled": False,
    "native_q1_self_attention_requested": False,
    "native_q1_self_attention_enabled": False,
    "native_q1_rope_cache_self_attention_requested": False,
    "native_q1_rope_cache_self_attention_enabled": False,
    "decode_session_runtime_enabled": False,
    "decode_session_cuda_graph_enabled": False,
    "active_prefix_decode_loop_enabled": False,
    "active_prefix_decode_cuda_graph_enabled": False,
}

for name, path in zip(("control", "candidate"), sys.argv[1:]):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    aggregate = manifest.get("aggregate", {})
    if aggregate.get("result_class") != "static_server_batch":
        raise SystemExit(f"{name}: real static batching was not observed")
    if aggregate.get("server_batch_observed") is not True:
        raise SystemExit(f"{name}: server_batch_observed changed")
    if aggregate.get("token_equivalence_status") != "not_checked_shared_server_rng":
        raise SystemExit(f"{name}: shared server RNG label changed")
    if manifest.get("rng_reset_policy") != "server_global_rng_shared_across_concurrent_requests":
        raise SystemExit(f"{name}: manifest RNG reset policy changed")
    for run_index, run in enumerate(manifest.get("runs", [])):
        if run.get("server_seed_applied") is not False:
            raise SystemExit(f"{name} run {run_index}: server seed was unexpectedly applied")
        if run.get("token_equivalence_status") != "not_checked_shared_server_rng":
            raise SystemExit(f"{name} run {run_index}: run token-equivalence label changed")
        profile_path = Path(run.get("profile_path", ""))
        if not profile_path.is_file():
            raise SystemExit(f"{name} run {run_index}: missing profile {profile_path}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        metadata = profile.get("metadata", {})
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise SystemExit(
                    f"{name} run {run_index}: metadata {field}="
                    f"{metadata.get(field)!r}, expected {expected!r}"
                )
        generation = profile.get("generation", [])
        if not generation:
            raise SystemExit(f"{name} run {run_index}: profile has no generation records")
        for record_index, record in enumerate(generation):
            for field, expected in expected_generation.items():
                if record.get(field) != expected:
                    raise SystemExit(
                        f"{name} run {run_index} generation {record_index}: {field}="
                        f"{record.get(field)!r}, expected {expected!r}"
                    )
print("static_server_runtime_contract_gate=PASS")
PY

python - "$CONTROL_MANIFEST" "$CANDIDATE_MANIFEST" "$RUN_ROOT/work-normalized-static-server.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def main_batch_work(manifest_path: str) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    batches: dict[str, dict] = {}
    for run in manifest.get("runs", []):
        profile = json.loads(Path(run["profile_path"]).read_text(encoding="utf-8"))
        for record in profile.get("generation", []):
            if record.get("profile_label") != "main_generation":
                continue
            ids = record.get("server_batch_ids") or []
            elapsed = record.get("server_batch_elapsed_seconds") or []
            sizes = record.get("server_batch_sizes") or []
            request_counts = record.get("server_batch_request_counts") or []
            work_items = record.get("server_batch_work_items") or []
            per_sample_tokens = record.get("generated_tokens_per_sample") or []
            if not (
                len(ids)
                == len(elapsed)
                == len(sizes)
                == len(request_counts)
                == len(work_items)
                == len(per_sample_tokens)
                == 1
            ):
                raise SystemExit(
                    "work-normalized diagnostic requires one B1 request row and one server batch "
                    "per main record"
                )
            if int(sizes[0]) != 5 or int(request_counts[0]) != 5 or int(work_items[0]) != 1:
                raise SystemExit(
                    "work-normalized diagnostic requires a merged B5 server batch made from five "
                    "independent B1 request rows"
                )
            batch_id = str(ids[0])
            batch = batches.setdefault(
                batch_id,
                {
                    "batch_id": ids[0],
                    "batch_size": int(sizes[0]),
                    "request_count": int(request_counts[0]),
                    "work_items_per_request": int(work_items[0]),
                    "elapsed_seconds": float(elapsed[0]),
                    "request_generated_tokens": [],
                },
            )
            if (
                batch["batch_size"] != int(sizes[0])
                or batch["request_count"] != int(request_counts[0])
                or batch["work_items_per_request"] != int(work_items[0])
                or batch["elapsed_seconds"] != float(elapsed[0])
            ):
                raise SystemExit(f"inconsistent duplicated server batch ledger for batch {batch_id}")
            batch["request_generated_tokens"].append(int(per_sample_tokens[0]))

    ordered = sorted(batches.values(), key=lambda batch: int(batch["batch_id"]))
    for batch in ordered:
        if len(batch["request_generated_tokens"]) != batch["batch_size"]:
            raise SystemExit(
                f"batch {batch['batch_id']} has {len(batch['request_generated_tokens'])} request records, "
                f"expected {batch['batch_size']}"
            )
        batch["active_decode_steps"] = max(batch["request_generated_tokens"])
        batch["generated_tokens_sum"] = sum(batch["request_generated_tokens"])
    active_steps = sum(batch["active_decode_steps"] for batch in ordered)
    elapsed_seconds = sum(batch["elapsed_seconds"] for batch in ordered)
    if active_steps <= 0 or elapsed_seconds <= 0:
        raise SystemExit("work-normalized diagnostic requires positive active steps and elapsed time")
    return {
        "git_commit": manifest.get("git_commit"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "unique_main_batches": len(ordered),
        "active_decode_steps": active_steps,
        "unique_main_batch_elapsed_seconds": elapsed_seconds,
        "active_decode_steps_per_second": active_steps / elapsed_seconds,
        "seconds_per_active_decode_step": elapsed_seconds / active_steps,
        "batches": ordered,
    }


control = main_batch_work(sys.argv[1])
candidate = main_batch_work(sys.argv[2])
baseline = control["active_decode_steps_per_second"]
current = candidate["active_decode_steps_per_second"]
report = {
    "schema_version": 1,
    "result_class": "static_server_shared_rng_work_normalized_diagnostic",
    "claim_scope": (
        "diagnostic only: normalizes unique B5 main-batch elapsed time by the longest generated "
        "row in each batch; it does not replace scheduler-wall throughput or prove token equivalence"
    ),
    "control": control,
    "candidate": candidate,
    "active_decode_steps_per_second_delta_pct": ((current / baseline) - 1.0) * 100.0,
}
Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(
    "work_normalized_active_decode_steps_per_second: "
    f"baseline={baseline:.3f}, candidate={current:.3f}, "
    f"delta={report['active_decode_steps_per_second_delta_pct']:+.1f}%"
)
PY

for extension_dir in "$RUN_ROOT/extensions/control" "$RUN_ROOT/extensions/candidate"; do
  if find "$extension_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "default V32 static server populated isolated extension directory $extension_dir" >&2
    find "$extension_dir" -mindepth 1 -maxdepth 2 -print >&2
    exit 1
  fi
done
find "$RUN_ROOT" -type f \
  \( -name static_server_batch_manifest.json -o -name 'compare-static-server.json' \
     -o -name 'work-normalized-static-server.json' -o -path '*/import-audits/*.json' \) \
  -print0 | sort -z | xargs -0 sha256sum
echo "strict_compare_status=$STRICT_COMPARE_STATUS"
if (( STRICT_COMPARE_STATUS != 0 )); then
  echo "static server strict scheduler-wall gate failed" >&2
  exit "$STRICT_COMPARE_STATUS"
fi
