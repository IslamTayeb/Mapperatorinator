# Full-Song Active Diagnostics After Native q1 Self-Attention

## Context

After the CUDA graph replay island probes, the biggest open accounting question was the non-layer remainder: accepted full-song SALVALAI model time was `32.217s`, while the captured one-token decoder-layer island accounted for about `15.637s`. That left roughly `16.580s` outside the measured steady q_len=1 decoder-layer replay island.

This pass is diagnostic-only. It enables active-prefix decode-loop counters, which are CPU-side wall spans and can include synchronization/control effects. They are not exclusive synchronized GPU-kernel timings and should not be used as a throughput claim.

## Run

DCC job `49228590`, commit `225dd04`, on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`, torch `2.10.0+cu128`, Transformers `4.57.3`.

- Run dir: `/work/imt11/Mapperatorinator/runs/active-diagnostics-native-full-49228590-225dd04`
- Profile: `/work/imt11/Mapperatorinator/runs/active-diagnostics-native-full-49228590-225dd04/output/beatmapb0180c4d555f41c4937534546336177c.osu.profile.json`
- Accepted comparison baseline: `/work/imt11/Mapperatorinator/runs/native-q1-self-full-49225493-c563af0/candidate_native_on/profile.json`

Flags matched the current fastest exact opt-in stack, plus:

```text
profile_active_prefix_decode_diagnostics=true
```

## Equivalence

Against accepted job `49225493`:

- Main tokens: PASS, `7,639 / 7,639` generated token IDs matched.
- Timing-context tokens: PASS, `821 / 821` generated token IDs matched.
- Same-calculation metadata contract: PASS.

Strict no-regression was not used for promotion because this was diagnostic-only:

- Main aggregate improved in this run (`237.111 -> 244.666 tok/s`), but strict per-window no-regression failed on two one-token windows (`seq84`, `seq85`) totaling only `0.721ms` model-time overhead.
- Timing aggregate improved (`100.822 -> 101.726 tok/s`), but strict timing per-window no-regression failed on many tiny timing windows totaling about `80.9ms` model-time overhead.

## Main-Generation Counters

Diagnostic profile aggregate:

| metric | value |
| --- | ---: |
| records | `87` |
| main tokens | `7,639` |
| model time | `31.222s` |
| tok/s | `244.666` |
| outer wall | `33.529s` |
| seq0 model/wall | `2.161s / 4.350s` |
| seq1+ model/wall | `29.061s / 29.179s` |

Largest active-prefix CPU-side diagnostic spans:

| span | wall |
| --- | ---: |
| token append + stopping | `15.216s` |
| stopping criteria | `14.752s` |
| logits processors | `4.577s` |
| prepare inputs | `2.780s` |
| decode forward span | `2.011s` |
| sampling | `1.200s` |
| cache position setup | `1.198s` |
| prefill forward span | `1.166s` |
| update kwargs | `0.662s` |

Logits processor split:

| processor | wall |
| --- | ---: |
| `MonotonicTimeShiftLogitsProcessor` | `3.035s` |
| `TopPLogitsWarper` | `1.282s` |
| `TemperatureLogitsWarper` | `0.160s` |

CUDA graph duplicate-capture ceiling is no longer target-sized:

| metric | value |
| --- | ---: |
| graph captures | `198` |
| normalized graph shapes | `11` |
| duplicate capture seconds | `0.154s` |
| duplicate capture percent of model time | `0.494%` |
| projected tok/s without duplicate capture | `245.880` |

## Interpretation

This run does not reveal a new easy speedup.

The full-song counters rule out more graph-cache/capture cleanup as a major path. Duplicate capture is under `0.5%` of diagnostic model time. Prefill-forward CPU span is only about `1.166s`, so the full `~16.6s` outside the one-token decoder-layer island is not simply repeated prefill forward cost. Prepare-inputs, logits processors, sampling, and stop/control spans are visible, but several of those spans include asynchronous CUDA launch/control or synchronization effects and should not be treated as additive GPU work.

The strongest remaining conclusion is unchanged: a `500 tok/s` attempt needs a broader decoder/runtime backend, not another narrow graph-cache or per-linear tweak. The next serious measurement should isolate the full decoder-step tail around decoder final norm, output projection, logits copy/processor/sampling, and stop/control synchronization, or prototype a whole decoder-layer/multi-layer native island with an explicit `>1.6s` projected full-song saving before production integration.

## Decision

No optimization graduated. Keep the accepted baseline at job `49225493` (`237.111 tok/s`) and treat job `49228590` as diagnostic attribution only.
