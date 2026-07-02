# Active-Prefix CUDA Graph Loop

## Summary

Added a default-off production path that captures the active-prefix one-token decode forward with manual CUDA graphs. This is the first current-branch exact full-song result to clear the original `100 tok/s` main-generation target, but it is not the cold default because strict zero-regression still fails on a small set of windows.

Use:

```bash
python inference.py --config-name profile_salvalai \
  inference_generation_compile=true \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512 \
  inference_active_prefix_decode_cuda_graph=true \
  inference_active_prefix_decode_cuda_graph_min_decode_steps=1
```

## Implementation

Commits:

- `e82bd96` added the default-off graph loop flag.
- `8e8757b` restored immediate graph capture as the graph path default after delayed capture regressed.

The graph path is intentionally narrow:

- `use_server=false`
- batch size 1
- `parallel=false`
- `cfg_scale=1`
- `num_beams=1`
- static cache
- decode-only active-prefix

It captures only the one-token model forward. HF setup, logits processors, sampling, RNG consumption, EOS handling, stopping, token append, and generated-token accounting stay outside the graph so generated-token identity remains testable.

## Validation

Retained baseline:

- Job: `49113713`
- Profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- Config: SDPA, `inference_generation_compile=true`, active-prefix disabled
- Main generation: `7,639` tokens, `82.615s`, `92.465 tok/s`
- Token equivalence: PASS against compile-disabled baseline

15s smoke, current commit:

- Job: `49166771`
- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-21`, RTX 2080 Ti
- Commit: `8e8757b`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-smoke-49166771-8e8757b`
- Candidate: active-prefix bucket512, CUDA graph, `min_decode_steps=1`
- Same-calculation metadata: PASS
- Token equivalence: PASS, `1,084 / 1,084`
- Strict per-window no-regression: PASS, `10 / 10`
- Main generation: `29.556 -> 110.451 tok/s` in the paired isolated-cache smoke

Full-song validation, current commit:

- Job: `49167356`
- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-22`, RTX 2080 Ti, driver `595.71.05`
- Stack: torch `2.10.0+cu128`, Transformers `4.57.3`
- Commit: `8e8757b`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-full-49167356-8e8757b`
- Profile: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-full-49167356-8e8757b/active512_graph/beatmapefa1873a66ee4fdf83106f9960f8c818.osu.profile.json`
- Compare JSON: `/work/imt11/Mapperatorinator/runs/active-graph-immediate-full-49167356-8e8757b/compare_retained_vs_active512_graph.json`

| metric | retained compile-only | active512 graph | delta |
| --- | ---: | ---: | ---: |
| main tokens | `7,639` | `7,639` | unchanged |
| main model time | `82.615s` | `71.981s` | `-10.634s`, `-12.9%` |
| main throughput | `92.465 tok/s` | `106.125 tok/s` | `+14.8%` |
| total timing+map stage | `113.928s` | `101.481s` | `-12.447s`, `-10.9%` |
| token equivalence | baseline | PASS, `7,639 / 7,639` | exact |

The compare metadata contract failed only because the old retained baseline profile is missing newer metadata keys (`temperature`, `top_p`, `cfg_scale`, `lookback`, and similar fields). There were no mismatched metadata values.

## Scoped Regression

Strict per-window zero-regression failed on 11/87 map windows:

| seq | baseline tok/s | candidate tok/s | model delta |
| ---: | ---: | ---: | ---: |
| 27 | `93.582` | `89.140` | `+29.82ms` |
| 28 | `93.894` | `92.613` | `+8.25ms` |
| 43 | `84.729` | `81.955` | `+11.99ms` |
| 44 | `17.273` | `16.315` | `+3.40ms` |
| 45 | `17.888` | `17.636` | `+0.80ms` |
| 46 | `58.767` | `48.307` | `+33.16ms` |
| 51 | `95.259` | `93.966` | `+10.97ms` |
| 72 | `92.234` | `90.959` | `+8.51ms` |
| 84 | `16.519` | `16.428` | `+0.33ms` |
| 85 | `18.224` | `17.610` | `+1.91ms` |
| 86 | `72.468` | `65.877` | `+19.33ms` |

Total failing-window overhead: `128ms` model time and `135ms` outer wall time. This is small relative to the `10.634s` main-generation model-time savings, but it means the result should be reported as aggregate-accepted with a scoped micro-regression, not as a strict per-window PASS.

## Rejected Delayed Capture

Job `49166715` tested delayed graph capture with `inference_active_prefix_decode_cuda_graph_min_decode_steps=16` on the 15s smoke slice.

- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-21`, RTX 2080 Ti
- Commit: `f712b59`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-graph-delay-smoke-49166715-f712b59`
- Token equivalence: PASS, `1,084 / 1,084`
- Main generation: `36.823 -> 97.146 tok/s`
- Strict per-window no-regression: FAIL, 5/10 windows

Delayed capture avoided some tiny capture costs but threw away too much graph benefit on medium windows. Keep `min_decode_steps=1` as the default for the graph path.

## Decision

Keep the graph path as a default-off opt-in performance mode because it is exact and improves full-song main generation by more than 10% on RTX 2080 Ti. Do not make it the default cold single-song path yet because strict zero-regression still fails on a small number of windows and the feature is limited to the simple non-server generation path.

The current retained conservative default baseline remains compile-only SDPA with active-prefix disabled. For future 200 tok/s work, use active512 graph as the fastest exact opt-in starting point, and focus next on reducing the remaining one-token decoder cost rather than wrapper cleanup.

