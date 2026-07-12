# Persistent DecodeSession Runtime

## Purpose

Move the persistent graph/cache idea from verifier-only infrastructure into a default-off production inference path, then measure whether it removes enough repeated CUDA graph capture/setup work to matter for normal single-song inference.

This is still same-calculation work: same model, fp32, SDPA, same seed, same sampling/output behavior, same active-prefix bucket64 + CUDA graph + stateful monotonic + q1 BMM stack. The candidate must match fixed-seed generated token IDs.

## Change

Commit `768b50f` adds an opt-in runtime path:

- `inference_decode_session_runtime=true`
- `inference_decode_session_cuda_graph=true`

It is accepted only with:

- `use_server=false`
- `parallel=false`
- `cfg_scale=1.0`
- `num_beams=1`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_cuda_graph=true`
- active-prefix bucket64, warmup0, min decode steps 1
- stateful monotonic logits processor
- q_len=1 BMM cross-attention

The runtime reuses:

- one `MapperatorinatorCache` object, explicitly reset between generation windows;
- a stable `BaseModelOutput` encoder-output buffer;
- one shared active-prefix CUDA graph cache per sequential context.

This keeps graph/cache state stable across generation windows while leaving token sampling, stopping, static-cache semantics, and output behavior unchanged.

## Prior Evidence

Full-song q1 diagnostic job `49223017` measured:

| metric | value |
| --- | ---: |
| main generated tokens | `7,639` |
| main model time | `37.662s` |
| main tok/s | `202.830` |
| CUDA graph captures | `198` |
| normalized graph shapes | `11` |
| duplicate capture ceiling | `2.866s` |
| duplicate capture share | `7.610%` |
| projected tok/s without duplicate capture | `219.537` |

Verifier jobs `49223121` and `49223151` proved a shared cache, stable encoder buffer, and shared graph cache can preserve generated tokens, raw logits/top-k order, and final RNG across multiple smoke windows.

## 15s Smoke

Job `49223253`, node `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`, commit `768b50f`.

Run root:

`/work/imt11/Mapperatorinator/runs/decode-session-runtime-smoke-49223253-768b50f`

Environment:

- Torch `2.10.0+cu128`
- Transformers `4.57.3`
- CUDA device: RTX 2080 Ti
- Config: `profile_salvalai_smoke15`
- Precision: `fp32`
- Attention: `sdpa`

| run | main tokens | main model time | main tok/s | timing tokens | timing model time | timing tok/s | total stage wall | equivalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q1 control | `1,084` | `4.766s` | `227.445` | `164` | `3.470s` | `47.261` | `29.882s` | baseline |
| DecodeSession candidate | `1,084` | `4.603s` | `235.504` | `164` | `2.787s` | `58.839` | `11.201s` | PASS main/timing |

Main improved `+3.5%`; timing improved `+24.5%`. Strict main per-window no-regression failed on two early records:

| sequence | candidate overhead |
| ---: | ---: |
| `seq0` | `+1.019ms` model time |
| `seq2` | `+1.757ms` model time |

Decision after smoke: promote to full-song validation because the change targets duplicate graph capture, which should matter more over the full 87-window song than over the 10-window smoke.

## Full-Song Validation

Job `49223294`, node `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`, commit `768b50f`.

Run root:

`/work/imt11/Mapperatorinator/runs/decode-session-runtime-full-49223294-768b50f`

| run | main tokens | main model time | main tok/s | timing tokens | timing model time | timing tok/s | total stage wall | equivalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| q1 control | `7,639` | `37.631s` | `203.000` | `821` | `9.864s` | `83.231` | `52.374s` | baseline |
| DecodeSession candidate | `7,639` | `35.337s` | `216.173` | `821` | `8.165s` | `100.553` | `48.493s` | PASS main/timing |

Full-song deltas:

| metric | delta |
| --- | ---: |
| main throughput | `+6.5%` |
| main model time | `-2.293s` |
| timing throughput | `+20.8%` |
| timing model time | `-1.699s` |
| total timing+map stage wall | `-3.881s` (`-7.4%`) |

Strict timing-context no-regression passed.

Strict main per-window no-regression failed on three tiny records:

| sequence | candidate overhead |
| ---: | ---: |
| `seq0` | `+2.173ms` model time |
| `seq45` | `+0.005ms` model time |
| `seq85` | `+0.298ms` model time |

The failed-window overhead totals about `2.477ms`, compared with `2.293s` aggregate main-generation savings.

## Decision

Keep as an accepted default-off opt-in win.

Why:

- fixed-seed generated token IDs matched for main and timing;
- main-generation full-song throughput improved `203.000 -> 216.173 tok/s`;
- timing-context throughput improved `83.231 -> 100.553 tok/s`;
- total timing+map stage wall improved `52.374s -> 48.493s`;
- the strict main failures are tiny scoped micro-regressions;
- the runtime establishes stable graph/cache state needed for future native-kernel and fused decoder-step work.

This is in the `5-10%` keep band: not enough to be a broad kernel breakthrough, but simple/strategic enough to keep because it removes repeated graph captures and unlocks the next runtime layer.

## Next Work

1. Historical next step: re-profile the new `216.173 tok/s` opt-in baseline with Nsight/torch diagnostic ranges to see the post-DecodeSession kernel split. This was done and later superseded by native q_len=1 self-attention job `49225493` at `237.111 tok/s`.
2. Do not chase graph-cache reuse further unless a fresh diagnostic shows remaining capture/setup overhead above `5%`.
3. Shift attention back to real decoder compute: one-token linear/MLP launch reduction, q_len=1 active-prefix self-attention/cache layout, or narrow C++/CUDA/CUTLASS kernels for measured hotspots.
4. Preserve the same verifier ladder for future runtime changes: one-token logits, direct-loop token/logit/RNG, 15s smoke token equivalence, then full-song token equivalence and non-regression accounting.
