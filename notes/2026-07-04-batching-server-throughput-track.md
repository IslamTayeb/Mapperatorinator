# Batching / Server Throughput Track

## Scope

This is the mergeable batching/server throughput track. It is separate from the
single-song `500 tok/s` campaign and from
`experiment/decoder-layer-runtime-island-do-not-merge`.

Do not mix these results into single-song TPS claims. Static server batching,
static window batching, warm-repeat serial multi-song, and future continuous
batching need separate result tables and acceptance gates.

## Audit Findings

Two read-only audits of the current control plane found that the existing
`use_server=true` path is static IPC request batching, not continuous batching,
and `parallel=true` is static window batching inside `InferenceProcessor`.
The accepted DecodeSession/native fast path is batch-1, sequential, non-server
only and should keep failing loudly for `use_server=true` or `parallel=true`.

Main merge risks before benchmarking static server throughput:

- stale IPC sockets could reuse a server loaded with different runtime flags;
- worker clients could auto-start unowned replacement servers;
- server startup and request waits could hang indefinitely;
- shared server-batch elapsed time could be counted once per request;
- shared global server RNG makes concurrent token hashes throughput diagnostics,
  not exactness evidence against cold single-song runs.

## Implemented Guardrails

Branch `codex/batching-server-throughput` now has:

- static server batch metadata in `server.py`/`processor.py`;
- suite aggregation of batching histograms and queue wait metadata;
- `utils/profile_static_server_batch.py` for concurrent full-song requests
  through the existing `InferenceServer`;
- stale-socket refusal by default, with `--allow-existing-server` only for
  explicitly documented reuse runs;
- connect-only worker clients and owner-only server startup;
- server start, request, and suite timeouts;
- server socket paths and server-loading configuration fingerprint in the
  manifest;
- scheduler-wall aggregate TPS as the primary static-server throughput metric;
- attributed per-request model TPS labeled separately;
- deduped server batch IDs/histograms to detect whether real multi-request
  batching occurred;
- `server_seed_applied=false` and
  `token_equivalence_status=not_checked_shared_server_rng` metadata for
  concurrent server runs.

## DCC Smoke

First smoke attempt:

- Job: `49267521`
- Commit: `a19b87c`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-141620-a19b87c`
- Result: failed before GPU inference. The harness compiled the base
  `profile_salvalai_smoke15` config before applying the song list, so it tried
  to validate the Mac-local SALVALAI audio path.
- Fix: commit `69c2504` sets the top-level compile-validation audio path to the
  first song-list entry before `compile_args()`.

Corrected smoke:

- Job: `49267597`
- Commit: `69c2504`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-141822-69c2504`
- Config: `profile_salvalai_smoke15`, five-song list, 15s middle slice,
  `max_batch_size=5`, `use_server=true`, `parallel=false`, fp32, SDPA,
  `profile_record_token_ids=true`, `generate_positions=false`.
- Result: failed after model load. The owner server started but shut down before
  worker clients reached `model_generate()` because the default server
  `idle_timeout` was only `20s`. Commit `2d0d6d7` exposes
  `server_idle_timeout` through `load_model_with_server()` and sets a long
  harness idle timeout.

Compile-enabled server smoke:

- Job: `49267683`
- Commit: `2d0d6d7`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-142136-2d0d6d7`
- Result: reached generation and failed in TorchInductor cudagraph-tree setup
  with `AssertionError` from `torch/_inductor/cudagraph_trees.py`.
- Slurm cleanup: the job stayed alive after the Python failure because the
  background server thread did not exit cleanly; it was cancelled after the
  failure was diagnosed.
- Decision: `use_server=true` plus `inference_generation_compile=true` is
  rejected for the current static IPC server because generation runs in a
  background batch thread. Commit `1475062` adds an `inference.py` validation
  error for this combination. Server batching profiles must use
  `inference_generation_compile=false` until a server-specific compile path is
  proven.

Compile-disabled server smoke:

- Job: `49267768`
- Commit: `1475062`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-142507-1475062`
- Config: same five-song 15s static server smoke, but
  `inference_generation_compile=false`.
- Result: completed.
- Manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-142507-1475062/static-server-batch-static-smoke-49267768/static_server_batch_manifest.json`
- GPU telemetry:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-smoke-20260704-142507-1475062/nvidia-smi.csv`
- Aggregate main tokens: `7,234`
- Scheduler wall: `59.9996s`
- Primary static-server throughput:
  `120.568 main tok/s` by scheduler wall
- Attributed request-model throughput:
  `40.447 main tok/s`; this is not the primary batching metric because merged
  server batch elapsed time is replicated across request records.
- Result class: `static_server_batch`
- Real multi-request batching observed:
  - main unique server batches: `8x size 5`, `2x size 4`, `2x size 1`
  - timing unique server batches: `9x size 5`, `1x size 4`, `1x size 1`
- Per-request token equivalence status:
  `not_checked_shared_server_rng`; this is throughput evidence only under the
  current shared-global server RNG policy.
- Telemetry summary: all samples averaged `43.55%` GPU util, max `79%`,
  average memory `2008.6 MiB`, max memory `3750 MiB`, average power `113.32 W`,
  max power `248.39 W`; active-memory samples averaged `58.06%` GPU util and
  `147.2 W`.

This validates the static server harness and metadata path on DCC. It does not
prove a same-calculation speedup over single-song inference because concurrent
server RNG reseeding is not implemented.

## Serial And Static-Window Comparison

Serial multi-song smoke:

- Job: `49267816`
- Commit: `1475062`
- Manifest:
  `/work/imt11/Mapperatorinator/runs/batching-compare-smoke-20260704-142810-1475062/serial/serial_multi_song-serial-smoke-49267816/suite_manifest.json`
- Config: five-song 15s serial multi-song, `use_server=false`,
  `parallel=false`, `inference_generation_compile=false`.
- Aggregate main tokens: `5,955`
- Main model time: `85.405s`
- Main throughput: `69.727 tok/s`
- Main wall throughput: `69.658 tok/s`
- Telemetry active-memory samples: `55.35%` average GPU util, `65%` max,
  `2168.7 MiB` average memory, `2208 MiB` max, `131.32 W` average power,
  `209.45 W` max.

Static-window `parallel=true` smoke:

- Job: `49267817`
- Commit: `1475062`
- Manifest:
  `/work/imt11/Mapperatorinator/runs/batching-compare-smoke-20260704-142810-1475062/parallel/serial_multi_song-parallel-smoke-49267817/suite_manifest.json`
- Config: same five-song 15s setup, but `parallel=true`.
- Aggregate main tokens: `3,524`
- Main model time: `59.984s`
- Main throughput: `58.749 tok/s`
- Main wall throughput: `58.739 tok/s`
- Static-window profile rows had `mode=parallel`, but batch-size histogram was
  only `1` for all five timing and main rows; this run did not demonstrate
  useful multi-window batching.
- Telemetry active-memory samples: `46.8%` average GPU util, `61%` max,
  `2142.4 MiB` average memory, `2172 MiB` max, `104.33 W` average power,
  `155.24 W` max.

Strict serial-vs-parallel comparison:

- Report:
  `/work/imt11/Mapperatorinator/runs/batching-compare-smoke-20260704-142810-1475062/serial-vs-parallel-strict-compare.json`
- Result: FAIL.
- Shape: serial `sequence_count=10`, parallel `sequence_count=1` for all five
  songs.
- Token equivalence: FAIL for all five songs.
- Output artifact equivalence: FAIL for all five songs.
- No-regression: FAIL; aggregate main throughput `69.727 -> 58.749 tok/s`
  (`-15.7%`), and every per-song main-generation row regressed or changed
  generated-token count.

Decision: static-window `parallel=true` is rejected as an exact optimization
for this setup. It should remain a separately reported mode, not a promoted
single-song or same-calculation batching speedup. Static server batching is the
only mode in this smoke that improved aggregate scheduler-wall throughput, but
it remains throughput-mode evidence under shared server RNG.

## Continuous Batching Design Constraints

Continuous batching is not a quick flag on top of `model.generate()`. Equivalent
claims require per-request/generated-window state for generated tokens, stopping
state, cache slots, logits-processor state, RNG behavior, output assembly, and
profile metadata. If exact per-request token/output/RNG behavior cannot be
proved, report the mode only as `documented-drift`.

The control plane now reserves the batch-specific flags without changing runtime
behavior:

- `inference_continuous_batching`
- `inference_continuous_batching_mode`
- `continuous_batch_max_active_sequences`
- `continuous_batch_max_wait_ms`
- `continuous_batch_prefill_policy`
- `continuous_batch_decode_order_policy`
- `continuous_batch_rng_policy`
- `inference_batch_decode_session_runtime`
- `inference_batch_native_decode_kernels`

These are declared in `config.py` and `configs/inference/default.yaml`, included
in inference profile metadata, and rejected in `inference.py` until an explicit
continuous scheduler exists. Local validation with the repo `.venv` passed:
default settings are accepted, changing scheduler options without
`inference_continuous_batching=true` raises `ValueError`, batch-specific native
flags without the master flag raise `ValueError`, and enabling continuous
batching with `use_server=true` raises the reserved `NotImplementedError`.

The next mergeable infrastructure step refactors the static IPC queue to use
explicit state:

- `generation_compatibility_key()` replaces `frozenset(generate_kwargs.items())`
  for request grouping, supports nested hashable/list/dict values, and rejects
  mutable runtime objects such as tensors with a clear `TypeError`;
- `StaticServerRequest` stores per-request progress, token counts, queue waits,
  and static-server batch metadata;
- `StaticServerRequestGroup` stores the original generation kwargs plus pending
  request records.

This is not a throughput optimization or continuous scheduler yet. It is meant
to preserve current static batching behavior while creating a reviewable state
boundary for future continuous batching. Local CPU validation ran
`py_compile` on `server.py` and direct helper tests for deterministic grouping,
ordered-value distinction, tensor rejection, isolated metadata lists, remaining
work accounting, and `_cut_model_kwargs()` row slicing.

Paired DCC regression smoke:

- Control job: `49268405`, `main@46ccc50`, node
  `dcc-core-ferc-s-z25-21`, RTX 2080 Ti UUID
  `GPU-8de8e67b-289d-a1ae-3745-25f40db81f55`.
- Branch job: `49268272`, `codex/server-batch-request-state@41de52c`, same
  node/GPU.
- Control manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-control-20260704-46ccc50/static-server-batch-static-batch-ctrl-49268405/static_server_batch_manifest.json`
- Branch manifest:
  `/work/imt11/Mapperatorinator/runs/static-server-batch-request-state-20260704-41de52c/static-server-batch-static-batch-state-49268272/static_server_batch_manifest.json`
- Both jobs completed with Slurm state `COMPLETED` and observed real
  `static_server_batch` batches.
- Control: `7,172` main tokens, `62.368s` scheduler wall,
  `114.995` generated-token/s by scheduler wall, main unique batches
  `8x size 5`, `2x size 4`, `2x size 1`.
- Branch: `6,681` main tokens, `59.428s` scheduler wall,
  `112.422` generated-token/s by scheduler wall, main unique batches
  `8x size 5`, `2x size 3`, `2x size 2`.
- Interpretation: shared global server RNG and concurrent scheduling changed
  generated-token counts and batch-tail shape, so this pair is not exact
  token-equivalent throughput evidence. It is a functional/no-wall-regression
  smoke for the request-state refactor: real batching still happens, scheduler
  wall did not regress, and no server batching error occurred. Both jobs emitted
  an NFS temp-dir cleanup warning after successful completion; Slurm exit codes
  were `0:0`.

## Continuous Scheduler Harness

Branch `codex/continuous-batch-scheduler-harness` adds a CPU-only scripted
scheduler harness in `osuT5/osuT5/inference/continuous_batching.py`. It is not
wired into `InferenceServer`, does not change runtime behavior, and does not
claim throughput.

The harness models:

- one compatibility-key group per scheduler instance;
- FIFO and round-robin decode order policies;
- active slot acquire/release events with slot generations;
- scripted token emission;
- `eos`, `max_new_tokens`, and `script_exhausted` stop reasons;
- no post-stop decode;
- active batch-size histograms;
- deterministic report dictionaries with reserved RNG/logits state hash fields.

It intentionally does not model real RNG, logits processors, cache tensors,
CUDA graphs, `torch.multinomial`, or output assembly. Those are still required
equivalence gates before any real continuous batching runtime can be considered
same-calculation.

Local validation used the repo `.venv`:

- `python -m py_compile osuT5/osuT5/inference/continuous_batching.py tests/test_continuous_batching_scheduler.py tests/test_server_batch_state.py`
- direct calls to `tests/test_continuous_batching_scheduler.py` and
  `tests/test_server_batch_state.py`

## Batching Summary Tests

Branch `codex/batching-summary-tests` adds
`tests/test_batching_summary_helpers.py`, covering synthetic static-server
metadata without GPU/model/audio dependencies. The tests lock down three
important reporting rules:

- `_profile_batch_summary()` preserves per-record static server metadata,
  including batch IDs, sizes, request counts, work items, batching mode,
  elapsed-time attribution, and queue waits.
- `_aggregate_batch_summaries()` keeps attributed per-request batch counts
  separate from deduped unique server-batch counts, preventing replicated
  merged-batch elapsed metadata from inflating batch totals.
- `utils/profile_static_server_batch.py` classifies runs with only batch size
  `1` as `static_server_no_batch_observed`, and reports
  `static_server_batch` only when at least one server batch size is greater
  than `1`.

Local validation used the repo `.venv`:

- `python -m py_compile tests/test_batching_summary_helpers.py utils/profile_inference_suite.py utils/profile_static_server_batch.py`
- direct calls to `tests/test_batching_summary_helpers.py`,
  `tests/test_continuous_batching_scheduler.py`, and
  `tests/test_server_batch_state.py`

## Static Server Comparator And Batch Timeout Knob

Branch `codex/batching-continuous-next` adds mergeable batching gate
infrastructure, not a runtime speed claim.

Changes:

- `server_batch_timeout` is now a public `InferenceConfig`/Hydra field and is
  recorded in profile metadata plus static-server manifests.
- `load_model_with_server()` forwards the configured timeout into
  `InferenceClient(batch_timeout=...)`.
- `utils/profile_static_server_batch.py` includes `server_batch_timeout` in the
  server config fingerprint and manifest.
- `utils/summarize_inference_profile.py --compare-static-server` compares two
  static server manifests.

The static-server comparator is deliberately narrower than exact suite
comparison. Under the current shared global server RNG, static server token
hashes are not exactness evidence. The strict static-server gate checks:

- same run/request/server contract;
- real `static_server_batch` observed on both sides;
- all runs remain labelled `not_checked_shared_server_rng`;
- scheduler-wall throughput and scheduler wall do not regress;
- aggregate generated main-token count does not shrink.

When profiling only the coalescing wait knob, use
`--allow-server-batch-timeout-change` so the comparator ignores only
`server_batch_timeout` in the server fingerprint. This does not make the result
exact-equivalent; it only makes the intended scheduler-policy change explicit.

Local validation used the repo `.venv`:

- `.venv/bin/python -m py_compile config.py inference.py utils/profile_inference_suite.py utils/profile_static_server_batch.py utils/summarize_inference_profile.py tests/test_summarize_inference_profile.py`
- YAML load smoke for `configs/inference/default.yaml`
- `git diff --check`
- custom in-process runner for `tests/test_summarize_inference_profile.py`,
  `tests/test_batching_summary_helpers.py`,
  `tests/test_continuous_batching_scheduler.py`, and
  `tests/test_server_batch_state.py`, which executed `27` test functions. The
  only warning was the existing pydub ffmpeg discovery warning.

Recommended DCC next step: run paired five-song static server smoke jobs from
the same commit, one with `server_batch_timeout=0.2` and one with a lower value
such as `0.02`, then compare with:

```bash
python utils/summarize_inference_profile.py \
  --compare-static-server "$BASE/static_server_batch_manifest.json" "$CAND/static_server_batch_manifest.json" \
  --allow-server-batch-timeout-change \
  --strict \
  --json-output "$RUN/compare-static-server.json"
```

Result:

- Job `49268949` failed before inference because the ad hoc Slurm script had a
  metadata-print quoting bug. No Mapperatorinator profile was produced.
- Job `49268950` completed on `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti UUID
  `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`, commit `68b63d3`.
- Run root:
  `/work/imt11/Mapperatorinator/runs/static-server-timeout-20260704-200509-68b63d3-rerun`
- Summary:
  `/work/imt11/Mapperatorinator/runs/static-server-timeout-20260704-200509-68b63d3-rerun/summary.json`
- Telemetry:
  `/work/imt11/Mapperatorinator/runs/static-server-timeout-20260704-200509-68b63d3-rerun/nvidia-smi.csv`

| timeout | main tokens | scheduler wall | scheduler-wall tok/s | result |
| ---: | ---: | ---: | ---: | --- |
| `0.2` | `7,474` | `65.541s` | `114.035` | baseline |
| `0.05` | `6,392` | `62.639s` | `102.046` | comparator FAIL |
| `0.02` | `7,163` | `91.372s` | `78.394` | comparator FAIL |

Both lower-timeout runs still observed real `static_server_batch` batches and
preserved the shared-RNG token-status label. They failed because scheduler-wall
throughput dropped and generated main-token counts shrank. Lowering the static
IPC coalescing wait is rejected for this workload; keep the default `0.2s`
unless new profiling gives a concrete reason to retest.

Telemetry for the whole three-pass job: `151` samples, average GPU util
`45.46%`, max `85%`, average memory `2071.45 MiB`, max `3726 MiB`, average
power `112.92 W`, max `249.45 W`. Active-memory samples averaged `59.57%` GPU
util and `2700.70 MiB`.

## Static Server Max Batch Size Sweep

Job `49268989` tested whether static server throughput improves when the server
can coalesce ten concurrent requests into larger batches. It ran from
`main@195dfd7` on `dcc-core-gpu-ferc-s-h36-5` with the same five-song 15s
smoke, `repeats=2`, `max_workers=10`, fp32/SDPA, compile disabled,
`server_batch_timeout=0.2`, and `generate_positions=false`.

Run root:
`/work/imt11/Mapperatorinator/runs/static-server-maxbatch-20260704-201718-195dfd7`

| max batch | main tokens | scheduler wall | scheduler-wall tok/s | unique main batches |
| ---: | ---: | ---: | ---: | --- |
| `5` | `13,436` | `99.687s` | `134.781` | `20x size 5` |
| `10` | `14,068` | `93.159s` | `151.011` | `4x size 10`, `3x size 9`, `2x size 7`, `1x size 5`, `1x size 4`, `2x size 2`, `6x size 1` |

Result: `max_batch_size=10` improved scheduler-wall main throughput by
`+12.0%` and reduced scheduler wall by `-6.55%`. This is an accepted
operational static-server batching recommendation for ten-request workloads,
not an exact token-equivalent or single-song claim. Shared server RNG remains
`not_checked_shared_server_rng`.

Future max-batch comparisons should use:

```bash
python utils/summarize_inference_profile.py \
  --compare-static-server "$BASE/static_server_batch_manifest.json" "$CAND/static_server_batch_manifest.json" \
  --allow-server-max-batch-size-change \
  --strict \
  --json-output "$RUN/compare-static-server-maxbatch.json"
```

## Continuous Batching Scheduler Ledger

The next mergeable continuous-batching step is still verifier infrastructure,
not runtime. The CPU scheduler now reports the state surfaces that a future
server-side model runtime must preserve before it can claim equivalence:

- scheduler config: max active sequences, wait budget, prefill policy, decode
  order, and RNG policy;
- per-request enqueue, activation, and finish steps;
- queue-wait, decode, and end-to-end latency step counts;
- cache slot id/generation acquire and release events;
- stop reasons and generated-token counts;
- placeholder per-request RNG, logits-processor, and cache state hashes.

This lets future continuous batching work fail loudly if it cannot account for
request-local RNG, logits-processor state, cache slot reuse/reorder, or stopping
behavior. It does not run the model, does not touch `InferenceServer`, and does
not create a throughput claim.

Local validation:

```bash
.venv/bin/python -m py_compile osuT5/osuT5/inference/continuous_batching.py tests/test_continuous_batching_scheduler.py
.venv/bin/python - <<'PY'
# in-process run of tests/test_continuous_batching_scheduler.py
PY
# ran 7 continuous batching tests
```

## Continuous Scheduler Dry-Run Manifest Gate

Added `utils/profile_continuous_scheduler.py` as a CPU-only dry-run harness for
the scheduler. It uses `generation_compatibility_key()` to enforce the same
compatibility-key discipline as the static server, feeds scripted requests into
`ContinuousBatchScheduler`, and writes `continuous_scheduler_manifest.json` with
`model_generation_executed=false`, `result_class=continuous_scheduler_dry_run`,
request token hashes, stop reasons, active-batch histogram, cache slot
acquire/release events, synthetic RNG/logits/cache state hashes, and scheduler
CPU wall diagnostics.

Added `utils/summarize_inference_profile.py --compare-continuous-scheduler` for
manifest comparisons. `--strict` checks:

- dry-run/model-free contract;
- `continuous_scheduler_dry_run` result class;
- scripted generated-token hashes/counts and stop reasons;
- scheduling shape: active batch histogram, stop-reason counts, and cache slot
  events.

CPU scheduler wall time is recorded but not part of `--strict`; use
`--require-no-regression` only when intentionally measuring CPU harness cost.
This avoids turning a synthetic scheduler microbenchmark into a model-throughput
claim.

Local smoke:

```bash
rm -rf /tmp/mapperatorinator-continuous-scheduler-smoke
.venv/bin/python utils/profile_continuous_scheduler.py \
  --output-root /tmp/mapperatorinator-continuous-scheduler-smoke \
  --suite-id local-smoke
.venv/bin/python utils/summarize_inference_profile.py \
  --compare-continuous-scheduler \
  /tmp/mapperatorinator-continuous-scheduler-smoke/continuous_scheduler_manifest.json \
  /tmp/mapperatorinator-continuous-scheduler-smoke/continuous_scheduler_manifest.json \
  --strict \
  --json-output /tmp/mapperatorinator-continuous-scheduler-smoke/compare-self.json
```

The smoke produced `3` requests, `9` scripted tokens, active batch histogram
`{'1': 1, '2': 4}`, and strict self-compare PASS. Local in-process tests now run
`32` batching/profile comparison tests.

Recommended sequence:

1. Keep static batching instrumentation mergeable and non-regressing.
2. Add RNG/logits-processor/cache-slot equivalence gates against the harness.
3. Only then wire a disabled real continuous server mode for DCC profiling.
