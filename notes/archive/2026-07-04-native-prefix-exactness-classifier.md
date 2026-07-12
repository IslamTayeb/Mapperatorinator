# Native Self+Cross Prefix Exactness Classifier

## Purpose

Classify why the approximate native self+cross prefix verifier failed strict
cache SHA equality. The path had a large prefix640 projected speed signal in
job `49258712`, but it was not same-calculation because cache-slot SHA checks
failed.

This pass answers whether the mismatch is a harmless verifier artifact such as
signed-zero bits, or real numeric fp32 drift from native projection/reduction
order.

## Artifacts

- Branch: `codex/native-prefix-exactness-classifier`
- Commits:
  - `e678e42`: add cache bit-diff diagnostics
  - `b71c275`: compare bit diffs against the expected cache slot
- Corrected DCC job: `49266140`
- GPU node: `dcc-core-ferc-s-z25-20`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/native-prefix-exactness2-49266140-b71c275`
- Report:
  `/work/imt11/Mapperatorinator/runs/native-prefix-exactness2-49266140-b71c275/native_self_cross_prefix_exactness.json`

The Slurm wrapper is `FAILED` because the strict verifier intentionally exits
nonzero when candidate cache-write checks fail. The report JSON is valid.

An earlier classifier job, `49266071`, used the first diagnostic implementation.
It revealed that bit-diff stats could be order-sensitive after benchmark
variants mutated the shared cache slot. Commit `b71c275` fixed this by storing
the expected q_len=1 K/V slot in each capture, restoring it before every
cache-write check, comparing candidates against that expected clone, and
restoring it afterward.

## Result

Corrected job `49266140`:

- report pass: `False`
- logits replay: PASS, `max_abs=0.0`
- candidate cache-write checks: FAIL
- layer output allclose: PASS, but `max_abs=7.62939453125e-06`

Native self+cross prefix cache-slot mismatch:

| variant | key numeric mismatches | value numeric mismatches | signed-zero mismatches | key max_abs | value max_abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_self_cross_prefix_warp2` | `456` | `595` | `0` | `7.152557e-07` | `4.768372e-07` |
| `native_self_cross_prefix_warp4` | `456` | `595` | `0` | `7.152557e-07` | `4.768372e-07` |
| `native_self_cross_prefix_warp8` | `456` | `595` | `0` | `7.152557e-07` | `4.768372e-07` |

Examples show ordinary one-ULP-ish numeric differences, not bit-only signed-zero
noise:

```text
key flat_index=1 expected=-1.225412368774414 actual=-1.2254124879837036
expected_bits=0xbf9cda50 actual_bits=0xbf9cda51

value flat_index=0 expected=-1.7810331583023071 actual=-1.7810332775115967
expected_bits=0xbfe3f8e5 actual_bits=0xbfe3f8e6
```

Short-run graph replay timings are diagnostic-only because this job used fewer
iterations than the original speed-sizing run:

| variant | graph replay ms/layer |
| --- | ---: |
| `repo_decoder_layer` | `0.219004` |
| `native_self_cross_prefix_warp2` | `0.195207` |
| `native_self_cross_prefix_warp4` | `0.194075` |
| `native_self_cross_prefix_warp8` | `0.182528` |

## Interpretation

The exactness failure is real numeric drift from native fp32 math, not a
fingerprint artifact. The candidate replaces PyTorch/HF projection reductions
with native `RMSNorm+Linear` and `Linear+Residual` reductions, so exact
byte-level cache identity is not expected.

This rules out an easy exactness fix. Making the native self+cross prefix
same-calculation would likely require matching PyTorch/cuBLAS reduction order
byte-for-byte or writing reference K/V/projection values, which defeats most of
the intended speedup.

The path can still be useful only under an explicitly labeled
`documented-drift` policy: it would need repeated baseline-vs-candidate runs,
direct-loop generated-token/logit/RNG checks, full-song output checks, and a
large operational speedup before asking to keep it. It must not be mixed into
same-calculation results.

## Decision

Keep the bit-pattern classifier as verifier infrastructure.

Do not pursue native self+cross prefix as a same-calculation production path
from the current evidence. The next exact work should stay focused on a broader
decoder-layer/native math-memory verifier that either preserves exact PyTorch
outputs or proves a new exact calculation boundary before production wiring.
