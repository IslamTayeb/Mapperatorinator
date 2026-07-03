# Active-Prefix Cold-Overhead Attribution

## Summary

Bucketed active-prefix decode remains default-off for cold single-song inference, but the warmed/batch signal is real. The targeted profiling pass shows the blocker is first-window graph/capture/specialization cost, not steady-state decode speed or token mismatch.

## Runs

- Commit: `daa182842c320596fca70590e17595357c5b96fc`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Stack: Python `3.10.12`, torch `2.10.0+cu128`, Transformers `4.57.3`
- Full-song matrix job: `49156700`, `COMPLETED`, exit `0:0`, elapsed `00:10:01`
- Diagnostics job: `49156749`, `COMPLETED`, exit `0:0`, elapsed `00:11:08`
- Nsight stats/export job: `49157290`, `COMPLETED`, exit `0:0`, elapsed `00:00:23`
- Full-song matrix dir: `/work/imt11/Mapperatorinator/runs/active-prefix-cold-matrix-49156700-daa1828`
- Diagnostics dir: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828`

The full-song matrix used separate TorchInductor and CUDA cache dirs per condition. Slurm allocated different GPU indices for the concurrent matrix and diagnostics jobs (`IDX:1` and `IDX:3`), so they did not share one physical GPU.

## Full-Song Matrix

| run | main tokens | main model time | main tok/s | timing tok/s | seq0 | seq1+ tok/s | equivalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cold compile-only | 7,639 | 90.287s | 84.608 | 19.425 | 16.413s, 35.703 tok/s | 95.473 | baseline |
| cold active512 | 7,639 | 79.369s | 96.247 | 20.862 | 25.531s, 22.953 tok/s | 131.003 | PASS vs cold compile |
| warm active512 run 0 | 7,639 | 80.409s | 95.002 | 20.771 | 26.434s, 22.168 tok/s | 130.672 | baseline |
| warm active512 run 1 | 7,639 | 59.737s | 127.877 | 98.275 | 3.809s, 153.829 tok/s | 126.109 | PASS |
| warm active512 run 2 | 7,639 | 60.335s | 126.610 | 97.655 | 3.871s, 151.393 tok/s | 124.911 | PASS |

Cold active512 beats the slower same-job cold compile-only run, but it still does not replace the retained clean cold baseline because prior cold validation was order-sensitive and the retained baseline is `92.465 tok/s` with cleaner timing/total-stage history. This matrix is attribution evidence, not a promotion.

The important shape is `seq0` versus steady state:

- Cold active512 `seq0`: `25.531s`.
- Warm active512 `seq0`: `3.809-3.871s`.
- Cold active512 after `seq0`: `131.003 tok/s`.
- Cold compile-only after `seq0`: `95.473 tok/s`.

That means active-prefix is already fast after the first long window. The cold weakness is front-loaded.

## Diagnostics

`TORCH_LOGS=recompiles,perf_hints,cudagraphs` on smoke15 produced:

- `244` recompile-line matches.
- `3,441` graph-recording lines.
- `8,665` CUDA graph/cudagraph log matches.
- `8` perf-hint matches.

The logs show `transformers/cache_utils.py:update` recompiling by `layer_idx` and hitting `torch._dynamo` `recompile_limit`. This points at cache-update / decoder-layer specialization and CUDA graph recording as concrete contributors to first-window cost.

Focused torch-profiler trace:

- Profile: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/torchprof_active512_seq9/beatmapd89f63c8bbb94ab2b338561bd23e258a.osu.profile.json`
- Trace: `torch_profiles/000_generation_main_generation_seq9.trace.json`
- Diagnostic overhead: traced range `71.170s` wall, so no throughput claim.
- Event mix: `20,504` TorchDynamo cache lookups, `16,310` `cudaGraphLaunch` calls, `131` `CUDAGraphNode.record` events, and `5,628` FMHA kernels with `1.063s` self CUDA.

Nsight Systems:

- Report: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/nsys_active512_smoke15/ap512.nsys-rep`
- SQLite: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/nsys_active512_smoke15/ap512.sqlite`
- Default report: `/work/imt11/Mapperatorinator/runs/active-prefix-diagnostics-49156749-daa1828/nsys_active512_smoke15/ap512_nsys_stats_default.txt`

The smoke15 Nsight run generated one-token early map windows and put the first long map window at `seq3`, so it validates top-level NVTX capture but should not be used for full-song `seq0` timing.

## Decision

Keep active-prefix as:

```text
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=512
```

but only as a default-off opt-in for warm-repeat, batch, serving, and custom-runtime experiments.

Next implementation work should target graph/runtime stabilization, not kernel work:

- Prime or pre-record active-prefix graph/cache variants, while counting setup cost in any cold single-song claim.
- Reduce or avoid `StaticCache.update` specialization by decoder `layer_idx`.
- Move toward a bufferized/direct decode loop that reuses stable call shapes and graph state.

Do not claim a cold single-song win by moving the cold cost outside the measured `main_generation` stage. Cold claims must include setup, timing generation, main generation, total stage time, token equivalence, and per-window non-regression. Warm/batch claims should report first-song cold cost and warmed amortized throughput separately.
