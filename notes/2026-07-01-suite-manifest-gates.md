# Suite Manifest Gates

## Summary

Added dependency-light suite metrics and comparison gates for warm-repeat, future serial multi-song, and active-prefix cold-tax attribution. This is testing/profiling infrastructure only; it does not change generation behavior or claim an inference speedup.

## Changes

- Added `utils/inference_profile_metrics.py` for profile-record summaries without importing Hydra/Accelerate.
- Extended `utils/profile_inference_suite.py` manifest schema to v3:
  - per-run `main_first_record` and `main_remaining_records`;
  - aggregate first-record and remaining-record segments for all/warmed/by-song runs;
  - timing-context generated tokens, model time, wall time, and tok/s;
  - runtime cache/env metadata including TorchInductor/Triton/CUDA/HF cache paths and `TORCH_LOGS`.
- Added `utils/summarize_inference_profile.py --compare-suite BASE CANDIDATE --suite-scope warmed_runs|all_runs` so warmed and multi-song suites can be gated by paired token hashes and selected-scope performance.
- Added lightweight tests for the profile metric helper and suite manifest comparison.

## Why

Active-prefix is strategically useful for warmed/batch inference, but prior runs showed the risk directly: warmed aggregate throughput can be strong while the first generation window pays graph/capture/specialization cost. The suite manifest now records both sides explicitly so future batch or serving claims report first-song cold cost and warmed steady-state behavior separately.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_inference_profile_metrics.py tests/test_summarize_inference_profile.py`
- `python -m compileall utils/inference_profile_metrics.py utils/profile_inference_suite.py utils/summarize_inference_profile.py tests/test_inference_profile_metrics.py tests/test_summarize_inference_profile.py`

Both passed locally.

## DCC Smoke Validation

Job `49164135` on `dcc-core-ferc-s-z25-20` (RTX 2080 Ti, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`, commit `006f828`) ran smoke15 warm-repeat suites with isolated TorchInductor/Triton/CUDA cache dirs:

- Run dir: `/work/imt11/Mapperatorinator/runs/suite-gates-smoke15-49164135-006f828`
- Compile-only manifest: `warm_repeat-compile-only/suite_manifest.json`
- Active512 manifest: `warm_repeat-active512/suite_manifest.json`
- Warmed compare: `compare-suite-warmed.json`
- All-run compare: `compare-suite-all-runs.json`

The Slurm job is marked `FAILED` because a final ad-hoc Python summary block had a quoting typo after all suite and compare artifacts had been written. The inference runs and suite comparisons completed.

Results:

| suite | run0 main | run1 warmed main | token equivalence |
| --- | ---: | ---: | --- |
| compile-only | `47.394 tok/s` | `100.597 tok/s` | run1 PASS vs run0 |
| active512 | `36.240 tok/s` | `137.748 tok/s` | run1 PASS vs run0 |

`--compare-suite --suite-scope warmed_runs --strict` passed:

- token hashes: PASS for both paired runs;
- warmed main throughput: `100.597 -> 137.748 tok/s`, `+36.9%`;
- warmed model time: `10.776s -> 7.869s`, `-27.0%`.

`--compare-suite --suite-scope all_runs` reported the expected cold-tax regression:

- all-run throughput: `64.432 -> 57.383 tok/s`, `-10.9%`;
- all-run model time: `33.648s -> 37.781s`, `+12.3%`.

This validates the intended reporting split: active-prefix remains a warmed/batch candidate, not a cold single-song baseline replacement, and the suite manifest now exposes that distinction directly.
