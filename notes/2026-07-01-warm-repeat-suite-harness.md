# Warm Repeat Suite Harness

## Summary

Added `utils/profile_inference_suite.py` to measure same-process inference behavior without changing default inference.

The first use case is `warm_repeat`: load the model once, run the same config multiple times, reset RNG before each run, write one profile JSON per run, and write a `suite_manifest.json` that separates first-run cold/specialization cost from warmed subsequent runs.

## Why

Active-prefix validation showed strong order and warm-state sensitivity. That means a candidate can be unattractive as a cold single-song baseline but still matter for future long-lived batch or multi-song serving. The old single-profile schema did not make that distinction explicit enough.

## Guardrails

- Requires `profile_inference=true`.
- Requires `profile_record_token_ids=true`.
- Requires `use_server=false` until server reseeding is explicit.
- Resets RNG before each run with `accelerate.set_seed`.
- Labels suite metadata with `suite_id`, `run_kind`, `suite_run_index`, `run_index`, `song_index`, `suite_repeat_count`, `rng_reset_policy`, and `warmup_excluded`.
- Reports token equivalence against run 0 in the suite manifest.
- Fails loudly for `serial_multi_song` until the harness accepts an explicit multi-song config/list.

## Non-Claim

Warmed suite throughput is not a cold single-song speedup. It can guide future runtime, batching, and serving work, but retained cold single-song changes still need the normal full-song SALVALAI token-equivalence and no-regression gates.

## First Intended DCC Runs

Run paired three-repeat suites on RTX 2080 Ti:

```bash
python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 3 \
  --run-kind warm_repeat \
  --output-root "$RUN_DIR/compile" \
  inference_active_prefix_decode_loop=false

python utils/profile_inference_suite.py \
  --config-name profile_salvalai \
  --repeats 3 \
  --run-kind warm_repeat \
  --output-root "$RUN_DIR/active512" \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512
```

Compare run 0 separately from runs 1..2. If active-prefix only wins warmed runs, document it as `warm_repeat` evidence and keep it default-off for cold single-song profiling.
