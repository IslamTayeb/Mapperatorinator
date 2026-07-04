# Continuous Scheduler State Validation

## Scope

Mergeable batching/server verifier infrastructure on
`codex/batching-continuous-throughput`. This does not run the model, does not
wire continuous batching into `InferenceServer`, and does not claim throughput.

## Why

The CPU-only continuous scheduler harness was useful for lifecycle planning, but
strict comparison could still pass two identically malformed manifests. Before
any model-backed continuous batching work, the manifest gate needs to prove the
request/state ledger is internally consistent.

## Changes

- Direct `ContinuousBatchScheduler.enqueue()` now fails if
  `planned_arrival_step` is greater than the scheduler's current step. The
  dry-run harness owns staggered arrival release.
- `utils/profile_continuous_scheduler.py` requires per-request RNG/logits/cache
  state hashes by default:
  `initial_rng_state_hash`, `final_rng_state_hash`,
  `logits_processor_state_hash`, and `cache_state_hash`.
- `--allow-missing-state-hashes` is explicit planning-only mode and records
  `state_hash_policy=allow_missing_planning_only`.
- `utils/summarize_inference_profile.py --compare-continuous-scheduler
  --strict` now rejects planning-only missing-hash manifests and self-validates
  both manifests.

Self-validation checks:

- generated-token hash and count recomputation;
- missing state-hash field consistency;
- enqueue, activation, finish, queue-wait, decode, and latency step math;
- no activation before planned arrival and no decode outside request lifecycle;
- active-batch histogram consistency;
- aggregate request/token/stop/cache-slot counts;
- one acquire and one release per completed request;
- cache-slot generation monotonicity.

## Validation

`pytest` is not installed in the repo venv, so validation used the existing
dependency-light in-process test runner style.

```bash
.venv/bin/python -m py_compile \
  osuT5/osuT5/inference/continuous_batching.py \
  utils/profile_continuous_scheduler.py \
  utils/summarize_inference_profile.py \
  tests/test_continuous_batching_scheduler.py \
  tests/test_summarize_inference_profile.py

.venv/bin/python - <<'PY'
# in-process test-function runner over:
# tests/test_continuous_batching_scheduler.py
# tests/test_summarize_inference_profile.py
# tests/test_batching_summary_helpers.py
# tests/test_server_batch_state.py
PY
# ran 36 test functions

rm -rf /tmp/mapperatorinator-continuous-scheduler-state-gate
.venv/bin/python utils/profile_continuous_scheduler.py \
  --output-root /tmp/mapperatorinator-continuous-scheduler-state-gate \
  --suite-id local-state-gate
.venv/bin/python utils/summarize_inference_profile.py \
  --compare-continuous-scheduler \
  /tmp/mapperatorinator-continuous-scheduler-state-gate/continuous_scheduler_manifest.json \
  /tmp/mapperatorinator-continuous-scheduler-state-gate/continuous_scheduler_manifest.json \
  --strict \
  --json-output /tmp/mapperatorinator-continuous-scheduler-state-gate/compare-self.json
```

CLI dry-run result:

- `3` requests
- `9` scripted tokens
- active batch histogram `{'1': 1, '2': 4}`
- strict self-compare PASS
- manifest self-validation PASS for baseline and candidate

## Decision

Keep this as mergeable verifier infrastructure. It makes future continuous
batching harder to accidentally over-claim, but it is not a model-backed
continuous scheduler and not a TPS result.
