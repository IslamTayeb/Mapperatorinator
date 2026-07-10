# Inference Profiling Runbook

This is the operational runbook for exact FP32 inference experiments on one RTX 2080/2080 Ti. Current results and historical decisions live in:

- [single-song frontier](../notes/inference-single-frontier.md)
- [batch/offline frontier](../notes/inference-batch-frontier.md)
- [experiment ledger](../notes/inference-experiment-ledger.md)

Profiling is opt-in. Normal V32 inference must not write profile artifacts, import optimized/native modules, or change behavior.

The canonical accepted fast path is selected with
`inference_engine=optimized optimized_inference_mode=single`. `inference.py`
loads the default-cold optimized adapter, `Processor` delegates each window to
`OptimizedSingleRuntime`, and `server.model_generate()` remains the ordinary
V32 path. The complete historical micro-flag bundle may reproduce the same
immutable runtime through `legacy_single_adapter.py`; partial bundles fail
loudly, and the public optimized selector may not be combined with legacy
micro-flags. Detailed ownership and migration evidence lives in the
[optimized-single migration map](../notes/2026-07-10-optimized-single-architecture-migration-map.md).

## Result Classes And Metrics

Use one explicit result class:

- `bitwise-calculation-exact`: required intermediate values and caches are bitwise identical, and all output gates pass.
- `exact-output`: intermediate FP32 values may be allclose, but token IDs/counts, stop behavior, final RNG, timing/main semantics, and `.osu` bytes match.
- `documented-drift`: any of those observable gates differs. Keep it in a separate table; never present it as an exact speedup.

Use synchronized, untraced `model_elapsed_seconds` for single-song main-generation TPS. Use scheduler wall from first main-generation start to last main-generation finish for offline/batch aggregate TPS. Torch/Nsight traced time and per-request attributed server time are diagnostic only.

Keep these modes separate:

1. cold single song;
2. same-process warm repeat;
3. serial multi-song;
4. static IPC server batch;
5. static window batch;
6. optimized offline batch/continuous scheduling;
7. online optimized server.

## DCC Setup

Verify live Slurm account, partition, GPU constraint, and profiling-tool permissions before submission. Do not reuse a stale scheduler directive from an old note.

```bash
REPO=/hpc/group/romerolab/imt11/projects/Mapperatorinator
ENV=/hpc/group/romerolab/imt11/envs/mapperatorinator
WORK=/work/imt11/Mapperatorinator

cd "$REPO"
export PATH="$ENV/bin:$PATH"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TRANSFORMERS_CACHE="$WORK/cache/huggingface"
export TMPDIR="$WORK/tmp"
export TOKENIZERS_PARALLELISM=false
```

Keep model/Hugging Face cache state identical across a comparison. Set and
record persistent `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and
`CUDA_CACHE_PATH` explicitly; an unset compiler cache can silently become
node-local and make the first timing window look like a runtime regression. If
a claim includes compiler/native cold start, isolate and record
`TORCH_EXTENSIONS_DIR`; otherwise use the same persistent extension cache for
baseline and candidate and report whether it was prebuilt.

Every run note must record:

- commit and branch;
- Slurm job/status and node/GPU;
- Hydra config and all changed flags;
- requested engine/mode, seed, precision, attention backend, server/parallel mode;
- effective optimized config version, runtime owner, and legacy-delegation status;
- model/HF, compiler, native-extension, and graph-cache state;
- run, profile/manifest, comparison, telemetry, and trace paths.

Use `scripts/dcc/profile_inference_denominators.sbatch` for the current
single-full, five-song smoke, and five-song full denominator jobs. Submit it
only after pushing and fast-forwarding the DCC checkout, and pass both the
requested run kind and the full expected commit through `sbatch --export`.

## Current Single-Song Baseline

The exact opt-in SALVALAI baseline is job `49230082`, commit `d7b8684`: `7,639` main tokens in `28.243s`, `270.475 tok/s`, main/timing token identity, and byte-identical output.

```bash
python inference.py --config-name profile_salvalai \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/single-full-${SLURM_JOB_ID}" \
  device=cuda \
  precision=fp32 \
  attn_implementation=sdpa \
  use_server=false \
  parallel=false \
  inference_engine=optimized \
  optimized_inference_mode=single
```

The selector activates immutable effective config
`accepted-fp32-270.475-v1`; it is default-off and supports only the validated
FP32/SDPA, batch-1, non-server, non-parallel, no-CFG, no-beam path. Record
requested and effective fields separately; timing contexts intentionally do
not use the native self-attention path. The complete historical legacy bundle
delegates to the same runtime only for reproduction. Do not use subsets of its
micro-flags as experimental switches.

Architecture-only changes must preserve this numerical frontier rather than
claim a new speed result. Gate them against the public selector, the delegated
legacy bundle, and pre-migration V32 as applicable; require one-token and
256-step exactness, reciprocal smoke and full-song output equivalence,
fresh-process default/compile-only import checks, and the throughput-only
static-server regression. Keep the per-boundary job history in the migration
map and place only the final current-commit decision in the experiment ledger.

## Smoke-To-Full Ladder

Before writing runtime code, record the current-stack profile, hypothesis, avoidable target, physical/fantasy floor, projected end-to-end ceiling, and falsification condition. Then advance only one evidence level at a time:

1. component microbenchmark or roofline;
2. captured real-tensor/logit/cache verifier;
3. short and then 256-step exact token/logit/RNG loop;
4. 15-second smoke;
5. reciprocal-order full-song main/timing/output comparison;
6. five-song or batch suite.

A microbenchmark or verifier may authorize the next gate, never production integration. If the signal falls below `5%`, exactness fails, or a broader metric regresses, stop immediately, remove candidate runtime wiring, retain only reusable measurement/verifier code, and add the result plus revisit condition to the experiment ledger.

Start with the 15-second middle slice:

```bash
python inference.py --config-name profile_salvalai_smoke15 \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/smoke15-${SLURM_JOB_ID}" \
  device=cuda precision=fp32 attn_implementation=sdpa \
  use_server=false parallel=false
```

For a decoder/cache/kernel candidate, run the correctness gates before speed profiling:

```bash
python utils/verify_one_token_decode.py \
  --config-name profile_salvalai_smoke15 \
  --sequence-index 9 \
  --report-path "$WORK/runs/one-token-${SLURM_JOB_ID}.json" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/one-token-output-${SLURM_JOB_ID}" \
  device=cuda precision=fp32 attn_implementation=sdpa \
  use_server=false parallel=false cfg_scale=1.0 num_beams=1

python utils/verify_direct_decode_loop.py \
  --config-name profile_salvalai_smoke15 \
  --max-new-tokens 256 \
  --report-path "$WORK/runs/direct-loop-${SLURM_JOB_ID}.json" \
  audio_path="$WORK/data/salvalai.mp3" \
  output_path="$WORK/runs/direct-loop-output-${SLURM_JOB_ID}" \
  device=cuda precision=fp32 attn_implementation=sdpa \
  use_server=false parallel=false cfg_scale=1.0 num_beams=1
```

Add candidate-specific verifier switches shown by `--help`; do not copy switches from an old experiment without checking the current CLI. The direct-loop gate must compare tokens, raw logits/top-k, and final RNG state. Cache-writing whole-layer candidates must also use `profile_decode_decoder_layer_island.py --verify-cache-write-candidates` and the current ABI/cache validator.

Compare smoke profiles:

```bash
python utils/summarize_inference_profile.py \
  --compare "$BASELINE_PROFILE" "$CANDIDATE_PROFILE" \
  --label main_generation \
  --strict \
  --json-output "$WORK/runs/smoke-compare-${SLURM_JOB_ID}.json"
```

Only promote a token-equivalent candidate with a credible current-stack projection above `5%`. Full-song promotion uses both main and timing plus output hashes:

```bash
python utils/summarize_inference_profile.py \
  --compare "$BASELINE_PROFILE" "$CANDIDATE_PROFILE" \
  --strict-full-song \
  --json-output "$WORK/runs/full-compare-${SLURM_JOB_ID}.json"
```

Reciprocal run order is required when compile, graph, thermal, or cache order could bias the result. Report cold run 0 separately from warmed repeats.

## Five-Song And Serial Suites

Use `utils/profile_inference_suite.py` for warm-repeat or serial multi-song evidence. Pass an explicit YAML/JSON/text song list for `serial_multi_song`; the established DCC list is under `$WORK/data/five-song-profile/songs.yaml`.

Suite manifests must preserve cold-run, warmed/all aggregate, timing, per-song, output hashes, and runtime cache/environment metadata. Compare warmed claims with:

```bash
python utils/summarize_inference_profile.py \
  --compare-suite "$BASE_MANIFEST" "$CANDIDATE_MANIFEST" \
  --suite-scope warmed_runs \
  --strict \
  --require-output-equivalence \
  --json-output "$WORK/runs/suite-compare-${SLURM_JOB_ID}.json"
```

Use `--suite-scope all_runs` for an all-run claim and `--gate-cold-run0` only when cold run 0 is part of the claim. Five-song serial results are not concurrent batch results.

## Static IPC Server Profiles

Use `utils/profile_static_server_batch.py` for the existing V32 server. Generation compile remains unsupported with `use_server=true`, and `use_server=true` must not be combined with `parallel=true`.

The server socket path must be derived from the explicit runtime key, including device, precision, attention backend, generation compile, `max_batch_size`, and `server_batch_timeout`, and hash-shortened when necessary for AF_UNIX limits. Refuse stale/existing sockets by default unless the run explicitly documents compatible reuse.

The primary metric is scheduler-wall aggregate throughput. Per-request server/model time is attributed because a merged server batch is replicated into multiple request records; never sum attributed times as independent GPU work.

Static server RNG is currently shared-global. Required manifest labels are:

```text
same_calculation=false
server_rng_policy=shared_global
token_equivalence_status=not_checked_shared_server_rng
throughput_claim_scope=static_ipc_concurrent_full_song_requests
```

Compare two manifests with the knob-specific allowance only when that knob is the experiment:

```bash
python utils/summarize_inference_profile.py \
  --compare-static-server "$BASE_MANIFEST" "$CANDIDATE_MANIFEST" \
  --strict \
  --json-output "$WORK/runs/static-server-compare-${SLURM_JOB_ID}.json"

# Add exactly one when applicable:
# --allow-server-batch-timeout-change
# --allow-server-max-batch-size-change
```

The comparator self-validates aggregate fields and the unique server-batch ledger. A malformed manifest is invalid evidence, not a performance regression. Current capacity evidence rejects lowering `server_batch_timeout` below `0.2s` and increasing `max_batch_size` beyond `10`; see the batch frontier.

For architecture-only server regressions, use
`scripts/dcc/verify_static_server_regression.sbatch` in both launch orders. It
reads every underlying profile to require V32/shared-RNG metadata and absent
optimized runtime state, audits fresh imports, and keeps per-side extension
directories empty. Mixed-song request ordering can assign the shared RNG stream
to different rows, changing stop lengths and scheduler work. Keep the strict
scheduler-wall comparison red when generated work differs. The script's
B5-only active-step normalization is diagnostic-only: report it to explain
shape variance, but never use it to replace scheduler throughput, token/output
equivalence, or a real promotion gate.

## Continuous-Scheduler Dry Runs

`osuT5/osuT5/inference/continuous_batching.py` is CPU-only and model-free. Use it to validate request lifecycle, planned arrivals, state hashes, slot acquire/release, and slot generations—not to claim TPS.

```bash
python utils/profile_continuous_scheduler.py \
  --output-root /tmp/mapperatorinator-continuous-scheduler \
  --suite-id local-state-gate

python utils/summarize_inference_profile.py \
  --compare-continuous-scheduler \
  /tmp/mapperatorinator-continuous-scheduler/continuous_scheduler_manifest.json \
  /tmp/mapperatorinator-continuous-scheduler/continuous_scheduler_manifest.json \
  --strict \
  --json-output /tmp/mapperatorinator-continuous-scheduler/compare-self.json
```

Strict dry-run evidence must include RNG, logits-processor, and cache hashes. `--allow-missing-state-hashes` is planning-only and cannot pass the strict promotion gate.

## Diagnostic Profiling

Throughput claims must come from untraced profiles. Use diagnostics only to prove a current target and its avoidable ceiling.

- `profile_generation_detail_ranges=true`: decoder attention/MLP/final-projection NVTX and torch ranges.
- `profile_active_prefix_decode_diagnostics=true`: active-prefix CPU/CUDA-event counters and graph/bucket behavior.
- `profile_model_generate_cuda_ledger=true`: production `model.generate()` CUDA-event and host-gap accounting.
- `utils/summarize_active_prefix_diagnostics.py`: active-prefix aggregates and non-additive fantasy ceilings.
- `utils/summarize_torch_trace_kernels.py`: kernel-family shares from torch Chrome traces.
- `utils/profile_decode_linear_kernels.py`: captured one-token linear signatures and replay.
- `utils/profile_decode_attention_components.py`: real q_len=1 self/cross attention components.
- `utils/profile_decode_full_forward_island.py` and `profile_decode_decoder_stack_island.py`: broad forward/stack ceilings.
- `utils/profile_decode_replay_gap.py`: production graph shell and input-copy gap.
- roofline/pressure summarizers under `utils/summarize_*roofline.py` and `summarize_decoder_layer_segment_pressure.py`.

Do not add overlapping profiler ranges together. Distinguish required model math from avoidable launch/layout/cache/control/movement cost, and compute fantasy-free and fantasy-floor TPS before implementation.

For useful-utilization diagnosis, prefer Nsight Systems timelines plus CUDA events and power/clock/memory telemetry. Probe Nsight Compute/DCGM permission; if unavailable, record that fact. A high `nvidia-smi` utilization percentage can include padding, rejected speculative work, or inefficient batches and is not a promotion criterion.

## Acceptance Checklist

Before merging an optimization:

1. Current accepted baseline and same-config candidate were run on the target GPU.
2. The candidate clears the `<5%` reject policy or has explicit approval for a smaller infrastructure-only change.
3. Main/timing tokens, counts, stops, RNG, output SHA/size, and touched caches pass the relevant exactness class; requested engine fields and effective runtime owner/config metadata are internally consistent.
4. Main model time/TPS, timing context, total stage wall, per-window behavior, and cold/warm costs are reported.
5. Batch claims additionally pass per-request order, arrival, EOS/max-token, slot reuse, and state isolation gates.
6. The result note contains commit, job, GPU, config, artifacts, decision, why it worked, and a concrete revisit condition.
7. Rejected runtime code is removed; accepted code remains default-off unless separately approved.
8. The canonical frontier and experiment ledger are updated.

Never commit generated outputs, profiles, manifests, traces, model weights, audio, or native build artifacts.
