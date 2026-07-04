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

Recommended sequence:

1. Keep static batching instrumentation mergeable and non-regressing.
2. Build a dummy-model scheduler test before touching real generation.
3. Add RNG/logits-processor/cache-slot equivalence gates.
4. Only then profile real continuous batching on DCC.
