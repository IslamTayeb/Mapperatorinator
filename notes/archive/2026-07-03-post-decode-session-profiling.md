# Post-DecodeSession Profiling

## Purpose

Re-profile the accepted `216.173 tok/s` DecodeSession opt-in baseline before choosing the next 500 tok/s target. This was diagnostic only and was later superseded by native q_len=1 self-attention job `49225493` at `237.111 tok/s`. Throughput claims still come from untraced `profile_inference`, not torch profiler or Nsight wall time.

Current accepted opt-in stack:

- `inference_generation_compile=true`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_bucket_size=64`
- `inference_active_prefix_decode_cuda_graph=true`
- `inference_active_prefix_decode_cuda_graph_warmup=0`
- `inference_active_prefix_decode_cuda_graph_min_decode_steps=1`
- `inference_stateful_monotonic_logits_processor=true`
- `inference_q1_bmm_cross_attention=true`
- `inference_decode_session_runtime=true`
- `inference_decode_session_cuda_graph=true`

Accepted full-song baseline from job `49223294`:

| metric | value |
| --- | ---: |
| main tokens | `7,639` |
| main model time | `35.337s` |
| main tok/s | `216.173` |
| timing tok/s | `100.553` |
| total timing+map stage | `48.493s` |
| token equivalence | PASS main/timing |

## Jobs

Job `49223379`, node `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`, commit `80adf86`.

Run root:

`/work/imt11/Mapperatorinator/runs/post-decode-session-profile-49223379-80adf86`

Artifacts:

- Torch profile JSON: `torch/torch_smoke15.profile.json`
- Torch Chrome trace: `torch/torch_profiles/000_generation_main_generation_seq9.trace.json`
- Torch active-prefix diagnostics: `torch/active_diagnostics.json`
- Nsight report: `nsys/decode_session_smoke15.nsys-rep`
- Nsight sqlite: `nsys/decode_session_smoke15.sqlite`
- Nsight kernel stats: `nsys/nsys_stats_cuda_gpu_kern_sum.csv`
- Nsight API stats: `nsys/nsys_stats_cuda_api_sum.csv`
- Nsight NVTX stats: `nsys/nsys_stats_nvtx_sum.csv`
- Nsight profile JSON: `nsys/nsys_smoke15.profile.json`

Environment:

- Torch `2.10.0+cu128`
- Transformers `4.57.3`
- Precision `fp32`
- Attention `sdpa`
- Config `profile_salvalai_smoke15`
- `use_server=false`, `parallel=false`, `seed=12345`

## Active-Prefix Diagnostics

Torch diagnostic main aggregate:

| metric | value |
| --- | ---: |
| main tokens | `1,084` |
| model time | `5.099s` |
| tok/s | `212.602` |
| graph captures | `24` |
| normalized graph shapes | `10` |
| duplicate graph captures | `14` |
| duplicate capture ceiling | `0.102866s` |
| duplicate capture model share | `2.017%` |
| projected tok/s without duplicate capture | `216.979` |

Nsight diagnostic main aggregate:

| metric | value |
| --- | ---: |
| main tokens | `1,084` |
| model time | `5.529s` |
| tok/s | `196.064` |
| graph captures | `24` |
| normalized graph shapes | `10` |
| duplicate graph captures | `14` |
| duplicate capture ceiling | `0.107769s` |
| duplicate capture model share | `1.949%` |
| projected tok/s without duplicate capture | `199.962` |

Interpretation: persistent DecodeSession graph/cache reuse largely exhausted the duplicate-capture target. Further graph-cache dictionary cleanup is not a plausible `>5%` path by itself.

## Torch Profiler Buckets

The traced window was `generation.main_generation.seq9`. Its wall time was inflated by profiler overhead, so use this only for attribution.

After excluding profiler ranges and CUDA API events, self-CUDA classified as:

| bucket | self CUDA | share | count |
| --- | ---: | ---: | ---: |
| GEMV/GEMM linear work | `289.754ms` | `32.4%` | `25,532` |
| FMHA attention | `264.564ms` | `29.6%` | `2,832` |
| elementwise/GELU/fill | `115.584ms` | `12.9%` | `60,852` |
| copy/index/cat | `86.268ms` | `9.6%` | `43,853` |
| other kernels | `83.914ms` | `9.4%` | `18,054` |
| sampling sort/softmax | `29.109ms` | `3.3%` | `4,200` |
| layernorm | `25.997ms` | `2.9%` | `8,683` |

Top individual events still match the earlier post-q1 profile:

- `fmha_cutlassF_f32_aligned_64x64_rf_sm75`: `264.564ms`, `2,832` calls.
- cuBLAS GEMV families: `108.530ms`, `63.491ms`, and `41.972ms` top groups.
- Layernorm kernel: `25.997ms`, `8,683` calls.
- DtoD/copy/index/cat kernels remain visible but not the top single target.

Top CPU/API overhead in the trace still includes:

- `cudaGraphLaunch`: `~177.9ms` self CPU, `233` calls.
- `cudaLaunchKernel`: `~103.9ms` self CPU, `17,744` calls.
- `cudaMemcpyAsync`: `~41.4ms` self CPU, `4,764` calls.

These are diagnostic profiler/API costs. The end-to-end accepted speedup already removed the larger repeated graph-capture cost.

## Nsight Buckets

Whole-smoke Nsight kernel buckets:

| bucket | GPU kernel time | share | count |
| --- | ---: | ---: | ---: |
| GEMV/GEMM linear work | `449.994ms` | `39.8%` | `3,200` |
| FMHA attention | `279.507ms` | `24.7%` | `720` |
| elementwise/GELU/fill/reduce | `194.746ms` | `17.2%` | `75,802` |
| sampling sort/softmax | `122.409ms` | `10.8%` | `9,429` |
| copy/index/cat | `55.249ms` | `4.9%` | `16,006` |
| other kernels | `22.779ms` | `2.0%` | `13,629` |
| layernorm | `6.682ms` | `0.6%` | `1,240` |

Top whole-smoke kernels:

- `volta_sgemm_128x64_tn`: `34.7%` of kernel time.
- `fmha_cutlassF_f32_aligned_64x64_rf_sm75`: `24.7%`.
- radix sort kernels: `~5.7%` across the two largest rows.
- `volta_sgemm_128x128_tn`: `2.7%`.

Nsight API stats still show many launches and syncs over the whole smoke:

- `cudaLaunchKernel`: `125,434` calls.
- `cudaStreamSynchronize`: `8,909` calls.
- `cudaMemcpyAsync`: `33,265` calls.
- `cudaGraphLaunch`: `1,228` calls.

These whole-smoke counts include timing generation, setup, and diagnostic overhead. They should guide investigation, not become throughput claims.

## Decision

No optimization graduated from this diagnostic pass.

At the time of this diagnostic, use the new `216.173 tok/s` DecodeSession opt-in path as the exact single-song baseline. That baseline is now superseded by native q_len=1 self-attention job `49225493` at `237.111 tok/s`. Do not chase more graph-cache reuse unless a fresh full-song diagnostic shows graph/capture/setup has grown again above `5%`.

Next plausible `>5%` exact work:

1. One-token linear/MLP/projection launch reduction or fused native C++/CUDA/CUTLASS/cuBLASLt decoder-block work.
2. q_len=1 active-prefix self-attention/cache-layout work that improves common buckets, not only the `L>=640` tail.
3. Device movement/index/cat cleanup only if it falls out of stable-buffer DecodeSession work.

Do not prioritize sampling/top-p fusion from this evidence. Sampling sort/softmax is visible, especially in whole-smoke Nsight, but it is not large enough to explain the remaining gap to `500 tok/s`.
