# Profile Comparison Gates

## Change

Added stricter testing-suite support around profile comparison:

- `utils/summarize_inference_profile.py --compare ... --strict` now exits nonzero on same-calculation metadata mismatch, missing generated-token IDs, token mismatch, aggregate performance regression, generated-token/record-count mismatch, total stage-wall regression, or selected-label per-window regression.
- `--json-output` writes the same gate result as structured JSON for Slurm logs or later note-taking.
- `tests/test_summarize_inference_profile.py` covers equivalent non-regression and token/performance failure cases with synthetic profiles.
- `utils/verify_one_token_decode.py` now makes slow no-cache diagnostic forwards opt-in via `--include-no-cache-diagnostics`; the pass condition remains HF cached raw logits versus direct static-cache q_len=1 logits/top-k.
- `configs/inference/profile_salvalai_smoke15.yaml` now explicitly pins compile on and active-prefix off, rather than relying only on parent config inheritance.

## Why

Recent active-prefix experiments repeatedly had token-equivalent subcases with hidden regressions in first-window time, timing generation, or total stage wall time. Manual reading was too easy to get wrong. The strict comparison gate makes the no-regression policy executable before a candidate is promoted.

## Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_summarize_inference_profile.py` passed locally.
- `python -m compileall utils/summarize_inference_profile.py utils/verify_one_token_decode.py utils/verify_direct_decode_loop.py tests/test_summarize_inference_profile.py` passed locally.
- Plain `python -m pytest ...` failed before collection in the local Mac environment because the global pytest/anyio plugin stack imports missing `_pytest.scope`; rerunning with plugin autoload disabled isolates the repo test.

## Status

Keep. This is a testing harness improvement only; it changes no default inference behavior and claims no inference speedup.
