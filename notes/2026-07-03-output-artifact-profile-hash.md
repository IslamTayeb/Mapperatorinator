# Output Artifact Profile Hash Gate

## Summary

Added a profiling guardrail so accepted inference speedups can mechanically prove the generated output artifact stayed byte-identical, not just token-identical.

This is not a speed optimization and does not change normal inference behavior. Hashing only runs when `profile_inference=true`, after the output file is written.

## Change

- `inference.py` now records `result_file_path`, `result_file_size_bytes`, and `result_file_sha256` in profile metadata.
- `utils/profile_inference_suite.py` carries `result_file_sha256` and `result_file_size_bytes` into each suite run record.
- `utils/summarize_inference_profile.py` reports output artifact equivalence for profile and suite comparisons.
- `--strict-full-song` now requires output artifact hash equivalence in addition to strict main/timing token and no-regression gates.
- Suite comparisons can require output equivalence with `--require-output-equivalence`.

## Why

The 500 tok/s campaign requires same settings and seed to produce the same generated map. Token identity is necessary but not sufficient for a retained full-song claim because postprocessing or output-writing changes could still alter the final `.osu` bytes. Making output hashes part of profile artifacts keeps that check in the normal promotion path.

## Verification

Local checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_summarize_inference_profile.py
python -m py_compile inference.py utils/profile_inference_suite.py utils/summarize_inference_profile.py tests/test_summarize_inference_profile.py
git diff --check
```

Result: `tests/test_summarize_inference_profile.py` passed `10 / 10`.

## Decision

Keep. This is low-risk guardrail infrastructure on `main`, not an inference-mode change and not a throughput claim.
