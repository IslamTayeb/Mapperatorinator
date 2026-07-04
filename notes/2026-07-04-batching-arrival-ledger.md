# Batching Arrival Ledger And Static-Server Metadata

## Scope

This is mergeable batching/server infrastructure on
`codex/batching-continuous-throughput`. It does not change single-song decode
math, does not enable real continuous GPU batching, and does not create a
single-song TPS claim.

## Changes

- Continuous scheduler dry runs now support per-request `arrival_step` /
  `planned_arrival_step`. Requests are enqueued when the dry-run scheduler reaches
  their planned arrival step, so the CPU harness can model idle gaps, queued
  requests, slot reuse, and cache-slot generations before any model-backed
  continuous runtime exists.
- `ContinuousBatchScheduler.step()` now advances `current_step` for no-op idle
  steps, making planned arrivals and idle service periods visible in manifests.
- Continuous dry-run manifests now include `scheduler_step_count`,
  `idle_step_count`, and `planned_arrival_step_histogram`.
- `utils/summarize_inference_profile.py --compare-continuous-scheduler --strict`
  now also gates lifecycle/state ledger fields: RNG hashes, logits-processor
  state hash, cache state hash, enqueue/activation/finish steps, queue wait,
  decode/latency steps, cache slot id, and slot generation.
- Static server profiles now label raw `use_server=true` runs as
  `server_rng_policy=shared_global` and
  `token_equivalence_status=not_checked_shared_server_rng`.
- Static server batching now records first queue wait separately from per-slice
  queue wait after partial-request requeue. Server batch summaries also preserve
  per-batch elapsed time and per-slice queue wait so deduped unique-batch timing
  can be inspected.
- `use_server=true` plus `parallel=true` now fails loudly. Those are separate
  batching modes until a dedicated mixed-mode harness exists.
- Static-server manifests/comparisons now explicitly mark the aggregate as
  `same_calculation=false` with
  `throughput_claim_scope=static_ipc_concurrent_full_song_requests`, and the
  comparator checks timing-token non-shrink plus request p95/max latency
  non-regression when relevant.

## Validation

Local validation used the repo `.venv`:

```bash
.venv/bin/python -m py_compile inference.py osuT5/osuT5/inference/server.py osuT5/osuT5/inference/processor.py osuT5/osuT5/inference/continuous_batching.py utils/profile_continuous_scheduler.py utils/profile_inference_suite.py utils/profile_static_server_batch.py utils/summarize_inference_profile.py tests/test_continuous_batching_scheduler.py tests/test_batching_summary_helpers.py tests/test_summarize_inference_profile.py tests/test_server_batch_state.py
git diff --check
.venv/bin/python utils/profile_continuous_scheduler.py --output-root /tmp/mapperatorinator-continuous-scheduler-arrival-smoke --suite-id local-arrival-smoke
.venv/bin/python utils/summarize_inference_profile.py --compare-continuous-scheduler /tmp/mapperatorinator-continuous-scheduler-arrival-smoke/continuous_scheduler_manifest.json /tmp/mapperatorinator-continuous-scheduler-arrival-smoke/continuous_scheduler_manifest.json --strict --json-output /tmp/mapperatorinator-continuous-scheduler-arrival-smoke/compare-self.json
```

Because `pytest` is not installed in the repo `.venv`, focused tests were run
with an in-process runner. It executed 34 tests across:

- `tests/test_continuous_batching_scheduler.py`
- `tests/test_batching_summary_helpers.py`
- `tests/test_summarize_inference_profile.py`
- `tests/test_server_batch_state.py`

The only warning was the existing pydub ffmpeg discovery warning. Continuous
scheduler dry-run strict self-compare passed.

## Decision

Keep this as mergeable verifier/profile infrastructure. It is the right next
step before any real continuous server runtime because it makes request arrivals,
queueing, cache slot lifecycle, and exactness ledgers explicit without touching
model generation.

Next DCC validation should rerun the existing five-song 15s static server smoke
from this branch and compare against the latest accepted static-server manifest
as operational throughput/no-regression evidence only.
