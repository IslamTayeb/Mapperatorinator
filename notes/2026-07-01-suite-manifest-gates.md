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
