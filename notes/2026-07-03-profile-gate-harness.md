# Profile Gate Harness Tightening

## Context

Full-song inference promotion requires checking both map/main generation and timing-context generation. The existing `utils/summarize_inference_profile.py --compare ... --strict` gate was label-scoped, so a caller could check `main_generation` and then manually inspect timing-context regressions later. That was easy to forget during fast profiling loops.

## Change

Added a multi-label compare path and CLI shortcut:

```bash
python utils/summarize_inference_profile.py \
  --compare BASELINE.profile.json CANDIDATE.profile.json \
  --strict-full-song \
  --json-output compare-full-song.json
```

`--strict-full-song` runs strict profile comparison for both `main_generation` and `timing_context` and exits nonzero if either label fails same-calculation metadata, generated-token equivalence, aggregate no-regression, total-stage wall, generated-token/record counts, or per-window no-regression.

Also changed `utils/verify_one_token_decode.py` to default to `--sequence-index 9`, matching the documented post-warmup one-token gate used by the 500 tok/s runtime probes.

## Verification

Local focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_summarize_inference_profile.py \
  tests/test_inference_profile_metrics.py
```

Result: `10 passed`.

## Decision

Keep. This is verifier/harness infrastructure only; it does not touch model runtime and does not claim a speedup.
