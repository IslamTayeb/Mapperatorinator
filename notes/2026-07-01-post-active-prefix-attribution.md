# Post Active-Prefix Attribution

## Summary

Job `49150687` traced the active-prefix bucket-256 path after the first full-song `121.926 tok/s` candidate run. Follow-up jobs `49151748` and `49152465` showed that bucket-256 was over-promoted and that active-prefix should remain default-off for cold single-song profiling. This trace is diagnostic only: torch profiler plus `TORCH_LOGS=recompiles,cudagraphs` massively distorted wall time and model time.

The useful signal is the event mix and graph-churn evidence, not throughput.

## Run

- Commit: `1f20478`
- Job: `49150687`
- Node/GPU: `dcc-core-ferc-s-z25-20`, NVIDIA GeForce RTX 2080 Ti, driver `595.71.05`
- Status: `COMPLETED`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-prefix256-detail2-smoke15-49150687-1f20478`
- Profile: `/work/imt11/Mapperatorinator/runs/active-prefix256-detail2-smoke15-49150687-1f20478/beatmapb552be5e78c74287b4c4205939d54f2d.osu.profile.json`
- Trace path in profile: `torch_profiles/000_generation_main_generation_seq9.trace.json`
- Flags: `profile_salvalai_smoke15`, `inference_generation_compile=true`, `inference_active_prefix_decode_loop=true`, `inference_active_prefix_decode_bucket_size=256`, `profile_generation_detail_ranges=true`, `profile_torch_event_limit=500`, `TORCH_LOGS=recompiles,cudagraphs`

The first attempted detail job, `49150613`, failed before inference because the inline Python version-print command lost string quoting. No profile was produced from that job.

## True Runtime vs Profiler Overhead

Do not use this run for throughput claims:

- Traced `main_generation.seq9`: `234` tokens, `10.999s` synchronized model time, `107.836s` outer wall.
- The profiler record itself reports `68.736s` trace wall time.
- Untraced full-song active-prefix candidates have post-warmup windows around `130-145 tok/s`, while this traced smoke is around `21-28 tok/s`.

This confirms torch profiler plus Dynamo/CUDAGraph logging is heavy enough to invalidate wall-time speed claims.

## Event Evidence

Grouped event totals for traced `main_generation.seq9`:

| group | count | CUDA total | self CUDA | self CPU |
| --- | ---: | ---: | ---: | ---: |
| self-attention SDPA NVTX | 2,820 | 411.197ms | 0.000ms | 96.647ms |
| cross-attention SDPA NVTX | 2,808 | 573.945ms | 0.000ms | 91.157ms |
| self-attention total NVTX | 2,808 | 540.642ms | 0.000ms | 441.515ms |
| cross-attention total NVTX | 2,808 | 627.512ms | 0.000ms | 354.462ms |
| MLP fc1 NVTX | 5,616 | 201.387ms | 135.634ms | 132.215ms |
| MLP fc2 NVTX | 5,616 | 196.059ms | 134.045ms | 133.158ms |
| fmha cutlass kernel | 5,628 | 977.634ms | 977.634ms | 0.000ms |
| GEMV kernels | 17,009 | 220.391ms | 220.391ms | 0.000ms |
| `aten::addmm` | 16,908 | 274.740ms | 274.723ms | 678.284ms |
| copy/memcpy group | 79,915 | 111.659ms | 106.095ms | 617.129ms |
| `aten::cat` | 542 | 1.947ms | 1.947ms | 6.851ms |
| softmax group | 1,404 | 7.609ms | 5.073ms | 5.609ms |
| sort | 234 | 10.504ms | 9.143ms | 6.057ms |
| multinomial/exponential | 468 | 11.191ms | 0.496ms | 12.267ms |
| `cudaGraphLaunch` | 16,310 | 0.000ms | 0.000ms | 191.849ms |
| `cudaLaunchKernel` CPU | 70,753 | 0.006ms | 0.000ms | 612.654ms |
| TorchDynamo cache lookup | 20,504 | 0.000ms | 0.000ms | 446.738ms |
| CUDAGraph record | 131 | 0.361ms | 0.000ms | 151.996ms |

Log counts:

- `__recompiles`: `71` lines.
- `Recording function`: `1,058` lines.
- `Checkpointing cuda caching allocator state`: `17` lines.

The first visible recompile confirmed active-prefix shape specialization:

```text
Recompiling function sdpa_attention_forward ... key size mismatch at index 2. expected 2560, actual 1024
```

## Interpretation

- Sampling/logits processors are still not target-sized. Sort, softmax, and multinomial/exponential are tens of milliseconds under a heavily profiled 234-token trace, not a plausible `>=10%` full-song lever.
- Attention is still meaningful, especially cross-attention after self-attention was shortened, but the biggest actionable trace smell is runtime/graph discipline: many CUDAGraph launches, many compiled graph calls, many Dynamo cache lookups, and many graph recordings.
- Active-prefix post-warmup windows improved despite graph churn, but full-song validation showed cold/order sensitivity. The next major exact-calculation target is to make that path more stable before any cold-baseline claim: fewer graph variants/recordings, fewer per-token compiled graph calls, and less repeated cache/mask/update plumbing.
- The first full-song active-prefix window regressed in multiple validations, including bucket-256 (`8.138s -> 11.515s`) and bucket-512 cold-first (`15.971s` compile-only after-active vs `24.739s` active512 in job `49152465`). Removing first-window specialization cost is a concrete next target.

## Decision

Do not start with fused sampling kernels. The trace does not support them as target-sized.

Next experiments should be:

1. Treat job `49151748` and `49152465` as the completed bucket sweep/validation: bucket `512` is the only active-prefix shape worth further runtime work, but it is not retained for cold single-song inference.
2. Graph-backed or bufferized direct decode loop if it can remove the first-window/specialization cost while preserving active-prefix exactness.
3. Cross-attention q_len=1 microprofile only if the next trace under a more stable runtime still shows cross-attention as target-sized.
