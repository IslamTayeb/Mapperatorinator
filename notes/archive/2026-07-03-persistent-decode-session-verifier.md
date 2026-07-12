# Persistent DecodeSession Verifier

## Purpose

Test whether CUDA graph captures can be reused across generation windows without changing generated tokens, raw logits/top-k order, or RNG state. This is verifier infrastructure only. It is not wired into production inference and does not claim a throughput win.

The motivation is the current q1 opt-in full-song diagnostic: job `49223017` measured `198` graph captures but only `11` normalized graph shapes. Duplicate captures cost `2.866s`, or `7.610%` of main model time, with a no-duplicate projection of `219.537 tok/s`.

## Utility

Added `utils/verify_persistent_decode_session.py`.

The verifier:

- loads one model process;
- builds reference windows with fresh HF `generate()` and fresh static cache per window;
- runs candidate windows with one shared `MapperatorinatorCache`, explicit cache reset, stable `BaseModelOutput` encoder-output buffer, and one shared CUDA graph cache;
- compares generated token IDs, captured raw logits/top-k order, stop reason, and final CPU/CUDA RNG state;
- reports graph captures by active-prefix bucket.

It intentionally leaves reserved production flags disabled. This proves a runtime direction before touching normal inference.

## Jobs

| job | commit | state | note |
| --- | --- | --- | --- |
| `49223099` | `40c411f` | `FAILED` | Initial `sequence_indices=9,10`; smoke15 has only indices `0..9`, so it failed before model comparison. |
| `49223121` | `40c411f` | `COMPLETED` | Two-window, 64-token gate. |
| `49223151` | `40c411f` | `COMPLETED` | Four-window, 256-token extended gate. |

Run roots:

- `/work/imt11/Mapperatorinator/runs/persistent-decode-session-verify-20260703-031951-40c411f`
- `/work/imt11/Mapperatorinator/runs/persistent-decode-session-verify-20260703-032118-40c411f-seq8-9`
- `/work/imt11/Mapperatorinator/runs/persistent-decode-session-verify-20260703-032337-40c411f-seq6-9-256`

Environment for passing jobs:

- Node: `dcc-core-gpu-ferc-s-h36-5`
- GPU: RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`
- Flags: active-prefix bucket64, CUDA graph warmup0, stateful monotonic logits processor, q1 BMM cross-attention.

## Current q1 Diagnostic Baseline

Before this verifier, job `49223017` reran the accepted q1 opt-in stack with active-prefix diagnostics on full-song SALVALAI at commit `976b0eb`:

| metric | value |
| --- | ---: |
| main generated tokens | `7,639` |
| main model time | `37.662s` |
| main tok/s | `202.830` |
| timing model time | `10.024s` |
| timing tok/s | `81.9` |
| CUDA graph captures | `198` |
| normalized graph shapes | `11` |
| decode replays | `7,552` |
| duplicate capture ceiling | `2.866s` |
| duplicate capture share | `7.610%` |
| projected tok/s without duplicate capture | `219.537` |

Strict compare against accepted q1 job `49213490` passed token equivalence (`7,639 / 7,639`) and had slightly better main model time (`201.125 -> 202.830 tok/s`), but failed strict no-regression because total stage wall was `+2.2%` and three per-window main records regressed. Treat it as diagnostics only.

## Verifier Results

Two-window gate, job `49223121`:

| sequence | generated tokens | token match | logits pass | max logit abs diff | stop reason |
| ---: | ---: | --- | --- | ---: | --- |
| `8` | `64` | PASS | PASS | `9.155e-05` | `max_new_tokens` |
| `9` | `64` | PASS | PASS | `1.068e-04` | `max_new_tokens` |

Final RNG state matched. The candidate captured `2` graphs for `2` unique prefixes and replayed them `126` times:

| prefix | graphs | replays |
| ---: | ---: | ---: |
| `128` | `1` | `88` |
| `192` | `1` | `38` |

Extended gate, job `49223151`:

| sequence | generated tokens | token match | logits pass | max logit abs diff | stop reason |
| ---: | ---: | --- | --- | ---: | --- |
| `6` | `256` | PASS | PASS | `1.221e-04` | `max_new_tokens` |
| `7` | `256` | PASS | PASS | `1.068e-04` | `max_new_tokens` |
| `8` | `256` | PASS | PASS | `1.221e-04` | `max_new_tokens` |
| `9` | `256` | PASS | PASS | `1.678e-04` | `max_new_tokens` |

Final RNG state matched. The candidate captured `5` graphs for `5` unique prefixes and replayed them `1,020` times:

| prefix | graphs | replays |
| ---: | ---: | ---: |
| `128` | `1` | `160` |
| `192` | `1` | `256` |
| `256` | `1` | `256` |
| `320` | `1` | `256` |
| `384` | `1` | `92` |

The stable encoder output shape was `[1, 1024, 768]` in both passing jobs.

## Decision

Keep the verifier utility. Persistent graph reuse is correctness-plausible and strategically useful, but it is not an accepted speedup yet.

Do not wire this into production inference directly from the verifier. The full-song q1 duplicate-capture ceiling is real but bounded: about `2.9s` or `7.6%` projected on SALVALAI. That sits in the “keep if simple/strategic” band, and the runtime structure also matters for future batch/serving and native-kernel work. However, production integration must still pass normal 15s smoke and full-song profile gates with total-stage/per-window regression checks.

Next implementation step, if pursuing this path:

1. Move the verifier-owned stable cache, stable encoder buffer, and shared graph cache into `DecodeSession`.
2. Add a production-off integration path that can persist one session across sequential generation windows.
3. Run 15s smoke with exact token IDs and compare against the accepted q1 opt-in stack.
4. Promote to full-song only if it shows a meaningful speed signal and no unacceptable timing/total-stage regressions.

