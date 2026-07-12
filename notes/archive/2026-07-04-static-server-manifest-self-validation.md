# Static Server Manifest Self-Validation

## Purpose

Harden Track B static-server throughput comparisons before tuning server
runtime behavior. This is mergeable validation infrastructure, not a throughput
claim and not single-song TPS evidence.

## Branch

```text
codex/batching-server-throughput-track
```

Base:

```text
main@fea2864
```

## Change

`utils/summarize_inference_profile.py --compare-static-server` now validates
each static-server manifest before comparing baseline and candidate. The new
self-validation recomputes:

- run count;
- main and timing generated-token totals;
- request wall-time sum, max, and p95;
- request-attributed model-time totals;
- scheduler-wall tok/s;
- attributed model tok/s;
- `same_calculation=false`;
- `throughput_claim_scope=static_ipc_concurrent_full_song_requests`;
- `token_equivalence_status=not_checked_shared_server_rng`;
- `result_class` from observed unique server batch sizes;
- aggregate batching summaries from per-run `generation_batch_summary` ledgers.

It also validates static server batch ledger shape:

- `server_batches` length must match `server_batch_count`;
- `batch_id`, `batch_size`, `request_count`, and `work_items` must be ints;
- batch elapsed/queue wait entries must be numeric or null;
- `static_server_batch` requires observed unique batch size greater than `1`;
- no-batch manifests remain classified separately as
  `static_server_no_batch_observed`.

`--strict` / `--require-mode-contract` now fails if either manifest fails
self-validation.

## Tests

Local validation:

```text
.venv/bin/python -m py_compile utils/summarize_inference_profile.py tests/test_summarize_inference_profile.py tests/test_batching_summary_helpers.py tests/test_server_batch_state.py
git diff --check
targeted in-process runner: 29 test functions passed
```

The targeted runner printed the existing pydub ffmpeg warning. No GPU or DCC
profiling was run for this checkpoint.

## Next

Use this stronger gate before testing static-server runtime knobs such as
`torch.cuda.empty_cache()` policy, `max_batch_size`, and `server_batch_timeout`.
Those future runs must remain labeled throughput-only unless a per-request
RNG/token/output exactness protocol is added.
