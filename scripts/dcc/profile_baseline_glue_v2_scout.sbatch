#!/usr/bin/env bash
#SBATCH --job-name=s46-glue-scout
#SBATCH --partition=gpu-common
#SBATCH --account=romerolab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2080:1
#SBATCH --time=00:45:00
#SBATCH --output=/work/imt11/Mapperatorinator/logs/%x-%j.out
#SBATCH --error=/work/imt11/Mapperatorinator/logs/%x-%j.err

# §46 W-BASE FP16 e2e scout on tip composition.
# Gates: exact tokens+RNG; sampling-tail hits>0; saved_ms/tok≥0.15; candidate TPS≥450.

set -euo pipefail

die() {
  echo "baseline-glue-v2-scout: $*" >&2
  exit 2
}

require_nonempty() {
  [[ -s "$1" ]] || die "required artifact is missing or empty: $1"
}

SLURM_JOB_ID=${SLURM_JOB_ID:?Run this wrapper inside a Slurm allocation}
PYTHON=${MAPPERATORINATOR_PYTHON:-/hpc/group/romerolab/imt11/envs/mapperatorinator/bin/python}
REPO=${MAPPERATORINATOR_REPO:?Set MAPPERATORINATOR_REPO to the clean candidate worktree}
COMMIT=${MAPPERATORINATOR_COMMIT:?Set MAPPERATORINATOR_COMMIT to the pushed candidate commit}
BRANCH=${MAPPERATORINATOR_BRANCH:?Set MAPPERATORINATOR_BRANCH to the candidate branch}
REMOTE_REF=${MAPPERATORINATOR_REMOTE_REF:?Set MAPPERATORINATOR_REMOTE_REF to the pushed candidate remote ref}
PRECISION=${MAPPERATORINATOR_PRECISION:-fp16}
WORK=${MAPPERATORINATOR_WORK:-/work/imt11/Mapperatorinator}
AUDIO=${MAPPERATORINATOR_AUDIO:-$WORK/data/salvalai.mp3}
EXPECTED_GPU=${MAPPERATORINATOR_EXPECTED_GPU:-2080 Ti}
ALLOW_PARALLEL_RECIPROCAL=${MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL:-0}
PROFILE_CONFIG=${MAPPERATORINATOR_PROFILE_CONFIG:-profile_salvalai}
MIN_SAVED_MS_PER_TOK=${MAPPERATORINATOR_MIN_SAVED_MS_PER_TOK:-0.15}
MIN_CANDIDATE_TPS=${MAPPERATORINATOR_MIN_CANDIDATE_TPS:-450}
RUN_LABEL=${RUN_LABEL:-baseline-glue-v2-scout-$PRECISION}
RUN_ROOT="$WORK/runs/${RUN_LABEL}-${SLURM_JOB_ID}"
JOB_TMP="${SLURM_TMPDIR:-/tmp}/imt11-baseline-glue-v2-scout-${SLURM_JOB_ID}"
JOB_EXT="$WORK/tmp/torch_extensions-baseline-glue-v2-scout-${SLURM_JOB_ID}"

case "$PRECISION" in
  fp32|fp16) ;;
  *) die "MAPPERATORINATOR_PRECISION must be fp32 or fp16" ;;
esac

[[ -x "$PYTHON" ]] || die "missing executable Python at $PYTHON"
[[ "$ALLOW_PARALLEL_RECIPROCAL" == 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] \
  || die "MAPPERATORINATOR_ALLOW_PARALLEL_RECIPROCAL must be 0 or 1"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || die "missing worktree at $REPO"
[[ -f "$AUDIO" ]] || die "missing input audio at $AUDIO"
[[ ! -e "$RUN_ROOT" ]] || die "refusing to overwrite $RUN_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || die "candidate worktree is dirty"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] \
  || die "worktree HEAD differs from MAPPERATORINATOR_COMMIT"
[[ "$(git -C "$REPO" branch --show-current)" == "$BRANCH" ]] \
  || die "worktree branch differs from MAPPERATORINATOR_BRANCH"
git -C "$REPO" show-ref --verify --quiet "$REMOTE_REF" \
  || die "missing pushed remote ref $REMOTE_REF"
[[ "$(git -C "$REPO" rev-parse "$REMOTE_REF")" == "$COMMIT" ]] \
  || die "pushed remote ref differs from MAPPERATORINATOR_COMMIT"

mkdir -p "$WORK/logs" "$WORK/tmp" "$WORK/cache" \
  "$WORK/cache/torch_inductor" "$WORK/cache/triton" "$WORK/cache/cuda" \
  "$JOB_TMP" "$JOB_EXT" "$RUN_ROOT"
export PATH="$(dirname "$PYTHON"):$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$WORK/cache/torch_inductor"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/cuda"
export TORCH_EXTENSIONS_DIR="$JOB_EXT"
export TMPDIR="$JOB_TMP"
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0

for tool in ffmpeg ffprobe nvidia-smi scontrol squeue sha256sum date df; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required executable: $tool"
done
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#GPU_NAMES[@]} -eq 1 ]] || die "expected exactly one visible GPU"
[[ "${GPU_NAMES[0]}" == *"$EXPECTED_GPU"* ]] \
  || die "expected $EXPECTED_GPU, got ${GPU_NAMES[0]}"
mapfile -t OTHER_GPU_JOBS < <(
  squeue -h -u "$USER" -t RUNNING,PENDING -o '%A|%b|%j' \
    | awk -F'|' -v current="$SLURM_JOB_ID" \
        '$1 != current && tolower($2) ~ /gpu/ { print }'
)
[[ ${#OTHER_GPU_JOBS[@]} -eq 0 || "$ALLOW_PARALLEL_RECIPROCAL" == 1 ]] || {
  printf '%s\n' "${OTHER_GPU_JOBS[@]}" >&2
  die "another GPU job is running or pending for $USER"
}

{
  echo "host=$(hostname)"
  echo "job_id=$SLURM_JOB_ID"
  echo "commit=$COMMIT"
  echo "branch=$BRANCH"
  echo "remote_ref=$REMOTE_REF"
  echo "precision=$PRECISION"
  echo "profile_config=$PROFILE_CONFIG"
  echo "min_saved_ms_per_tok=$MIN_SAVED_MS_PER_TOK"
  echo "min_candidate_tps=$MIN_CANDIDATE_TPS"
  echo "tmpdir=$TMPDIR"
  echo "torch_extensions_dir=$TORCH_EXTENSIONS_DIR"
  echo "run_root=$RUN_ROOT"
  echo "audio_sha256=$(sha256sum "$AUDIO" | awk '{print $1}')"
  "$PYTHON" -c 'import torch, transformers; print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}")'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader
  scontrol show job -o "$SLURM_JOB_ID"
} | tee "$RUN_ROOT/preflight.txt"

cd "$REPO"
COMMON_OVERRIDES=(
  audio_path="$AUDIO"
  device=cuda
  precision="$PRECISION"
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
  profile_pass_kind=untraced_control
)

run_role() {
  local role=$1
  local kind=$2
  local run_dir="$RUN_ROOT/$role"
  mkdir -p "$run_dir/output"
  local evidence="$run_dir/evidence.json"
  local command=(
    "$PYTHON" utils/run_exact_baseline_glue_v2.py
    --mode "$kind"
    --config-name "$PROFILE_CONFIG"
    --rope-evidence-path "$run_dir/rope-evidence.json"
    --evidence-manifest "$evidence"
    "${COMMON_OVERRIDES[@]}"
    output_path="$run_dir/output"
  )
  printf '%q ' "${command[@]}" > "$run_dir/command.txt"
  printf '\n' >> "$run_dir/command.txt"
  "${command[@]}" > "$run_dir/stdout.txt" 2> "$run_dir/stderr.txt"
  mapfile -t profiles < <(find "$run_dir/output" -type f -name '*.profile.json' | sort)
  [[ ${#profiles[@]} -eq 1 ]] || die "$role produced ${#profiles[@]} profiles"
  local profile=${profiles[0]}
  local osu=${profile%.profile.json}
  require_nonempty "$profile"
  require_nonempty "$osu"
  require_nonempty "$evidence"
  require_nonempty "$run_dir/rope-evidence.json"
  printf '%s\n' "$profile" > "$run_dir/profile-path.txt"
  printf '%s\n' "$osu" > "$run_dir/osu-path.txt"
  printf '%s\n' "$evidence" > "$run_dir/evidence-path.txt"
}

run_role baseline baseline
run_role candidate candidate

"$PYTHON" - \
  "$RUN_ROOT/baseline/profile-path.txt" \
  "$RUN_ROOT/candidate/profile-path.txt" \
  "$RUN_ROOT/baseline/osu-path.txt" \
  "$RUN_ROOT/candidate/osu-path.txt" \
  "$RUN_ROOT/baseline/evidence-path.txt" \
  "$RUN_ROOT/candidate/evidence-path.txt" \
  "$MIN_SAVED_MS_PER_TOK" \
  "$MIN_CANDIDATE_TPS" \
  "$RUN_ROOT/scout_gate.json" <<'PY'
import hashlib
import json
import pathlib
import sys

def load_json(path: pathlib.Path):
    return json.loads(path.read_text())

def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def flatten_token_ids(profile: dict) -> list[int]:
    tokens: list[int] = []
    records = profile.get("generation") or []
    if not isinstance(records, list) or not records:
        raise RuntimeError("profile missing generation records")
    for record in records:
        per_sample = record.get("generated_token_ids_per_sample")
        if isinstance(per_sample, list) and per_sample:
            if isinstance(per_sample[0], list):
                for row in per_sample:
                    tokens.extend(int(t) for t in row)
            else:
                tokens.extend(int(t) for t in per_sample)
            continue
        ids = record.get("generated_token_ids")
        if isinstance(ids, list):
            tokens.extend(int(t) for t in ids)
    if not tokens:
        raise RuntimeError("profile missing generated token ids")
    return tokens

def main_model_seconds(profile: dict) -> float:
    records = profile.get("generation") or []
    if not isinstance(records, list) or not records:
        raise RuntimeError("profile missing generation records for timing")
    total = 0.0
    for record in records:
        value = record.get("model_elapsed_seconds")
        if value is None:
            raise RuntimeError("generation record missing model_elapsed_seconds")
        total += float(value)
    return total

baseline_profile = load_json(pathlib.Path(pathlib.Path(sys.argv[1]).read_text().strip()))
candidate_profile = load_json(pathlib.Path(pathlib.Path(sys.argv[2]).read_text().strip()))
baseline_osu = pathlib.Path(pathlib.Path(sys.argv[3]).read_text().strip())
candidate_osu = pathlib.Path(pathlib.Path(sys.argv[4]).read_text().strip())
baseline_evidence = load_json(pathlib.Path(pathlib.Path(sys.argv[5]).read_text().strip()))
candidate_evidence = load_json(pathlib.Path(pathlib.Path(sys.argv[6]).read_text().strip()))
min_saved = float(sys.argv[7])
min_tps = float(sys.argv[8])
out_path = pathlib.Path(sys.argv[9])

baseline_tokens = flatten_token_ids(baseline_profile)
candidate_tokens = flatten_token_ids(candidate_profile)
token_equal = baseline_tokens == candidate_tokens
osu_equal = sha256_file(baseline_osu) == sha256_file(candidate_osu)
hits = int(candidate_evidence.get("sampling_tail_cuda_graph_hits", 0))
rng_equal = all(
    baseline_evidence[key] == candidate_evidence[key]
    for key in ("rng_after_seed", "rng_after_inference")
)
n_tok = max(len(baseline_tokens), 1)
base_s = main_model_seconds(baseline_profile)
cand_s = main_model_seconds(candidate_profile)
saved_ms_per_tok = (base_s - cand_s) * 1000.0 / n_tok
base_tps = n_tok / base_s if base_s > 0 else 0.0
cand_tps = n_tok / cand_s if cand_s > 0 else 0.0
exact_pass = bool(token_equal and osu_equal and rng_equal and hits > 0)
budget_pass = saved_ms_per_tok >= min_saved
tps_pass = cand_tps >= min_tps
gate = {
    "schema_version": 1,
    "token_count_baseline": len(baseline_tokens),
    "token_count_candidate": len(candidate_tokens),
    "token_stream_equal": token_equal,
    "osu_sha_equal": osu_equal,
    "rng_equal": rng_equal,
    "sampling_tail_cuda_graph_hits": hits,
    "baseline_main_model_s": base_s,
    "candidate_main_model_s": cand_s,
    "baseline_tps": base_tps,
    "candidate_tps": cand_tps,
    "saved_ms_per_token": saved_ms_per_tok,
    "min_saved_ms_per_token": min_saved,
    "min_candidate_tps": min_tps,
    "exact_pass": exact_pass,
    "budget_pass": budget_pass,
    "tps_pass": tps_pass,
    "path_hits": {
        "sampling_tail_cuda_graph_hits": hits,
        "decode_loop_calls": int(candidate_evidence.get("decode_loop_calls", 0)),
        "pinned_eos_flag_enabled": bool(
            candidate_evidence.get("pinned_eos_flag_enabled")
        ),
    },
    "pass": bool(exact_pass and budget_pass and tps_pass),
}
out_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(gate, indent=2, sort_keys=True))
if not exact_pass:
    raise SystemExit("baseline_glue_v2_scout_gate=FAIL exactness")
if not budget_pass:
    raise SystemExit(
        f"baseline_glue_v2_scout_gate=FAIL budget "
        f"saved_ms_per_token={saved_ms_per_tok:.4f} < {min_saved}"
    )
if not tps_pass:
    raise SystemExit(
        f"baseline_glue_v2_scout_gate=FAIL tps "
        f"candidate_tps={cand_tps:.2f} < {min_tps}"
    )
print("baseline_glue_v2_scout_gate=PASS")
PY

echo "baseline_glue_v2_scout=PASS"
