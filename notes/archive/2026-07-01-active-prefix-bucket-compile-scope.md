# Active-Prefix Bucket-Scoped Compile Scope

## Hypothesis

Active-prefix is strong in warmed runs but weak cold because one shared HF `model.get_compiled_call()` may mix graph state across active-prefix bucket lengths. A default-off experiment compiled a separate wrapper per active-prefix bucket so the active-prefix length was captured inside the compiled callable.

## Implementation

Commit `b4f3f11` added an experimental `inference_active_prefix_decode_compile_scope=bucket` path. The wrapper was cached on the model by bucket length and compile config. It kept prefill, logits processors, sampling, RNG, EOS, and token append behavior unchanged.

The code was reverted in `ea1c422` after measurement.

## Evidence

DCC job `49163585`, node `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, torch `2.10.0+cu128`, Transformers `4.57.3`.

Run dir: `/work/imt11/Mapperatorinator/runs/ap-bucket-compile-49163585-b4f3f11`

Direct-loop gate:

- Path: `/work/imt11/Mapperatorinator/runs/ap-bucket-compile-49163585-b4f3f11/gate_bucket_compile.json`
- PASS: token match, logits pass, top-k match, RNG match.
- `8` generated steps, `max_abs=0.0`, matching generated tokens `[12, 1648, 2242, 2717, 3915, 4012, 4053, 32]`.

15s smoke:

| variant | main tokens | main model time | tok/s | stage wall sum |
| --- | ---: | ---: | ---: | ---: |
| compile-only | `1,084` | `21.773s` | `49.786` | `61.378s` |
| active512 shared compile | `1,084` | `29.741s` | `36.448` | `68.700s` |
| active512 bucket compile | `1,084` | `31.946s` | `33.933` | `72.195s` |

Strict comparisons:

- Active shared vs bucket main: token equivalence PASS, no-regression FAIL, `-6.9%` tok/s, every main window failed per-window non-regression.
- Compile-only vs bucket main: token equivalence PASS, no-regression FAIL, `-31.8%` tok/s, `+17.6%` stage wall.

## Conclusion

Reject and keep reverted. Bucket-scoped wrappers preserved exact calculation but increased cold compile/graph overhead enough to make active-prefix worse. This narrows the active-prefix path: graph-state isolation probably needs a direct runtime/capture plan with explicit setup accounting, not one `torch.compile` wrapper per bucket around the existing HF custom loop.
