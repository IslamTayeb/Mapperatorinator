# Post-q1 500 tok/s Profiling Pass

## Purpose

Freshly re-profile the accepted exact opt-in stack before starting deeper 500 tok/s work. This pass is measurement only. It does not introduce or claim a speedup.

Current campaign baseline:

- `inference_generation_compile=true`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_bucket_size=64`
- `inference_active_prefix_decode_cuda_graph=true`
- `inference_active_prefix_decode_cuda_graph_warmup=0`
- `inference_active_prefix_decode_cuda_graph_min_decode_steps=1`
- `inference_stateful_monotonic_logits_processor=true`
- `inference_q1_bmm_cross_attention=true`

Reference accepted full-song result remains DCC job `49213490`, commit `3af8d69`: `7,639` main tokens, `37.981s` model time, `201.125 tok/s`, main/timing token equivalence PASS.

## Jobs

Run root: `/work/imt11/Mapperatorinator/runs/500tps-postq1-profile-20260703-015239-b0ec059`

| job | purpose | state | node / GPU | artifact |
| --- | --- | --- | --- | --- |
| `49221975` | full-song untraced `profile_inference` throughput | Slurm `FAILED` because strict compare exited nonzero after valid inference | `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1` | `throughput/throughput.profile.json` |
| `49221976` | smoke15 Nsight/NVTX diagnostic | `COMPLETED` | `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-58946b9b-e585-f3ba-8d50-5522c8d1c391` | `nsys/postq1_smoke15.nsys-rep` |
| `49221977` | smoke15 torch-profiler seq9 diagnostic | `COMPLETED` | `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-61cfde6c-eed9-9e99-6c89-393c6b53fa8e` | `torch/torch_smoke15.profile.json` |
| `49222076` | Nsight text stats export | `COMPLETED` | `dcc-core-ferc-s-z25-20` | `nsys/nsys_stats_summary.txt` |

Environment for the jobs:

- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Commit: `b0ec059`
- Precision: `fp32`
- Attention: `sdpa`
- `use_server=false`, `parallel=false`, `seed=12345`, `profile_record_token_ids=true`

## Full-song throughput result

Official TPS source: untraced profile `throughput/throughput.profile.json`.

| metric | q1 baseline `49213490` | post-q1 profile `49221975` | result |
| --- | ---: | ---: | --- |
| main tokens | `7,639` | `7,639` | same |
| main model time | `37.981s` | `38.072s` | `+0.091s`, `+0.2%` worse |
| main tok/s | `201.125` | `200.646` | `-0.479 tok/s`, `-0.2%` |
| main outer wall | `38.125s` | `38.228s` | `+0.3%` worse |
| total timing+map stage | `52.638s` | `70.621s` | not comparable as a speed claim because this run used isolated caches and paid much larger model-load/setup cost |
| token equivalence | PASS | PASS | `7,639 / 7,639` matched |

Strict compare path: `throughput/compare-main-vs-q1bmm.txt`.

Interpretation:

- No speed win. Keep `201.125 tok/s` from job `49213490` as the campaign baseline.
- The current commit still reproduces the optimized path in the `~200 tok/s` range with exact main-token identity.
- The Slurm job failure is from the strict comparison command returning nonzero after the valid profile was written. It is not an inference failure.
- Future total-stage comparison jobs should not isolate Hugging Face/model caches unless the claim explicitly includes first-download/model-cache cold cost. Isolate compiler/CUDA caches for cold compile checks, but avoid invalidating model-download/load caches when comparing normal inference stage wall time.

## Diagnostic attribution

Torch-profiler diagnostic: `torch/torch_smoke15.profile.json`, traced label `generation.main_generation.seq9`.

The trace is diagnostic only. The profiled `seq9` outer wall was inflated to `25.545s`, while the smoke main-generation model aggregate was `5.562s` for `1,084` tokens (`194.902 tok/s`). Do not use traced wall time as throughput.

Actual kernel self-CUDA classification for `main_generation.seq9`, excluding NVTX/record-function ranges and CUDA API calls:

| bucket | self CUDA | share | count |
| --- | ---: | ---: | ---: |
| one-token GEMV/GEMM linear work | `289.975ms` | `36.0%` | `25,532` |
| FMHA attention | `265.009ms` | `32.9%` | `2,832` |
| elementwise | `111.762ms` | `13.9%` | `59,894` |
| copy/index/cat | `81.302ms` | `10.1%` | `40,299` |
| layernorm | `25.329ms` | `3.1%` | `8,683` |
| other kernels | `19.465ms` | `2.4%` | `3,990` |
| softmax | `13.756ms` | `1.7%` | `3,264` |

Top actual kernels:

| kernel class | self CUDA | count | note |
| --- | ---: | ---: | --- |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm75` | `265.009ms` | `2,832` | dominant attention kernel |
| cuBLAS GEMV family | `108.464ms` | `11,184` | repeated one-token linear work |
| cuBLAS GEMV family | `63.477ms` | `3,029` | repeated one-token linear work |
| cuBLAS GEMV family | `41.968ms` | `2,796` | repeated one-token linear work |
| DtoD copies | `29.700ms` | `20,517` | device movement still visible |
| layernorm kernel | `25.329ms` | `8,683` | small but many calls |
| `volta_sgemm_128x64_tn` | `23.536ms` | `121` | larger linear/GEMM work |
| index/cat kernels | `32.969ms` combined | `11,280` | cache/cat/index movement |

Top profiler CPU self-time events:

| event | self CPU | count | interpretation |
| --- | ---: | ---: | --- |
| `cudaGraphLaunch` | `197.904ms` | `233` | launch/control overhead remains visible even with graph replay |
| `cudaLaunchKernel` | `114.955ms` | `19,562` | many small launches still exist outside/around graph work |
| active-prefix `prepare_inputs` | `57.970ms` | `234` | runtime plumbing, not enough alone for 500 tok/s |
| `cudaMemcpyAsync` | `44.350ms` | `5,050` | memory movement/API overhead |
| active-prefix graph range | `44.268ms` | `233` | graph replay wrapper CPU |
| stateful monotonic logits processor | `34.090ms` | `234` | not the main CUDA target |

Active-prefix diagnostic counters over smoke15 main generation:

| run | graph captures | capture time | graph replays | bucket shapes |
| --- | ---: | ---: | ---: | --- |
| Nsight smoke | `24` | `0.791s` | `1,074` | `128..704` by `64` |
| Torch smoke | `24` | `0.680s` | `1,074` | `128..704` by `64` |

`compile_lookup_wall_cpu_s` remained negligible for main generation (`~0.006s`). Sampling/logits work is visible but not target-sized relative to attention plus linear kernels.

Nsight text stats from `49222076` are available at `nsys/nsys_stats_summary.txt`. Treat them as whole-smoke/timeline context rather than clean decode-only attribution because they include timing generation, model setup, prefill, audio/model-cache effects, and Nsight overhead. The top whole-smoke CUDA kernel totals were:

| kernel family | Nsight GPU kernel time share | note |
| --- | ---: | --- |
| `volta_sgemm_128x64_tn` | `34.8%` | linear/GEMM work across the profiled smoke |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm75` | `24.9%` | attention |
| radix sort kernels | about `5.7%` combined for top listed sort kernels | top-p/sampling visible but still smaller than linear plus attention |
| `volta_sgemm_128x128_tn` | `2.7%` | additional GEMM |

Nsight MemOps were dominated by Host-to-Device time over the whole smoke; do not interpret that as decode-only memory traffic without a narrower trace.

## Decision

No optimization graduated from this pass. Keep the current exact opt-in baseline at `201.125 tok/s`.

The next plausible 500 tok/s work should target:

1. One-token linear/GEMV/GEMM launch reduction or grouping.
2. q_len=1 active-prefix self-attention/cache-layout work.
3. Device movement/index/cat cleanup only when it falls out of the DecodeSession/runtime work.

Do not prioritize fused sampling/logits processors from this evidence. They are not large enough to plausibly produce the required `~60%` model-time reduction.

Before native kernels, add verifier/runtime structure that can preserve exact tokens/RNG while making stable buffers and per-step work explicit. If graph replay hides too much component detail, run a separate no-throughput component trace with `inference_active_prefix_decode_cuda_graph=false` and/or `inference_generation_compile=false`, but do not compare that speed to the accepted stack.
