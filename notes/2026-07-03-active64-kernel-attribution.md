# Active64 Kernel Attribution

## Purpose

After duplicate graph-capture cleanup proved too small to reach `200 tok/s`, this pass asked what the accepted active-prefix path is actually spending CUDA time on. The goal was to separate plausible kernel/backend targets from misleading Python/profiler counters.

## Runs

### Accepted Runtime Trace

- Source job: `49207288`
- Run dir: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e`
- Profile: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e/active64_seq66_trace/profile.json`
- Trace: `/work/imt11/Mapperatorinator/runs/active64-nextdiag-49207288-cf4f87e/active64_seq66_trace/torch_profiles/000_generation_main_generation_seq66.trace.json`
- Runtime: active-prefix bucket64, manual CUDA graph enabled
- Status: diagnostic only; torch profiler wall time is not throughput

The CUDA graph replay range hides detailed decoder ranges, but kernel names remain visible.

### No-Graph Component Trace

- Job: `49210611`
- Status: `COMPLETED`, exit `0:0`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Commit: `f9ba99e`
- Run dir: `/work/imt11/Mapperatorinator/runs/active64-nograph-trace-49210611-f9ba99e`
- Profile: `/work/imt11/Mapperatorinator/runs/active64-nograph-trace-49210611-f9ba99e/profile.json`
- Trace: `/work/imt11/Mapperatorinator/runs/active64-nograph-trace-49210611-f9ba99e/torch_profiles/000_generation_main_generation_seq9.trace.json`
- Runtime: active-prefix bucket64, manual CUDA graph disabled, generation compile disabled, `main_generation.seq9` traced
- Status: diagnostic only; this intentionally does not represent accepted throughput

This uncaptured trace exposes detailed decoder ranges, but `record_function`/NVTX and profiler shape grouping double-count inclusive ranges. Treat the kernel-level split as the reliable signal.

## Kernel Split

Accepted active64 graph trace, classified by self CUDA time after excluding the top-level CUDA graph replay range:

| category | self CUDA | share | count |
| --- | ---: | ---: | ---: |
| FMHA attention | `379.462ms` | `57.4%` | `2,964` |
| GEMV linear | `113.577ms` | `17.2%` | `10,394` |
| elementwise | `65.195ms` | `9.9%` | `31,865` |
| device copies | `44.147ms` | `6.7%` | `21,808` |
| SGEMM linear | `28.505ms` | `4.3%` | `136` |
| layernorm | `13.509ms` | `2.0%` | `4,576` |
| top-p sort | `12.555ms` | `1.9%` | `492` |
| softmax | `0.939ms` | `0.1%` | `246` |

No-graph component trace, classified by kernel self CUDA:

| category | self CUDA | share | count |
| --- | ---: | ---: | ---: |
| FMHA attention | `918.677ms` | `63.6%` | `5,628` |
| GEMV linear | `226.151ms` | `15.6%` | `19,829` |
| elementwise | `134.311ms` | `9.3%` | `60,157` |
| device copies | `86.616ms` | `6.0%` | `38,417` |
| SGEMM linear | `34.245ms` | `2.4%` | `136` |
| layernorm | `29.264ms` | `2.0%` | `8,683` |
| top-p sort | `9.090ms` | `0.6%` | `234` |
| softmax | `2.512ms` | `0.2%` | `468` |

The two traces are directionally consistent despite different runtime modes. Attention dominates, one-token linear/GEMV work is second, and sampling/top-p is not large enough to carry the next milestone alone.

## Decoder Range Hints

In the no-graph trace, one representative layer showed:

| range | calls | CUDA total |
| --- | ---: | ---: |
| `decoder.layer0.self_attn` | `234` | `45.481ms` |
| `decoder.layer0.cross_attn` | `234` | `51.565ms` |
| `attention.layer0.self.sdpa` | `235` | `29.559ms` |
| `attention.layer0.cross.sdpa` | `234` | `47.081ms` |
| `decoder.layer0.mlp.fc1` | `234` | `11.087ms` self CUDA |
| `decoder.layer0.mlp.fc2` | `234` | `10.768ms` self CUDA |
| `decoder.output_projection` | `234` | `6.262ms` self CUDA |

These are diagnostic hints only because the no-graph trace disables the accepted CUDA graph path and profiler ranges can double-count. The qualitative ranking still matches the raw kernel split.

## Interpretation

- There is no remaining small Python/logits/sampling cleanup likely to produce `>=10%` full-song gains by itself.
- The next exact-calculation path toward `200 tok/s` has to reduce the replayed decoder compute, especially q_len=1 self-attention/cross-attention and one-token linear/GEMV work.
- Persistent graph/cache reuse is useful cleanup but has a measured ceiling well below `200 tok/s`.
- A fused sampling/top-p path is low priority unless a future trace changes the cost mix; the current kernel-level top-p/softmax cost is only about `2%` in the accepted graph trace.

## Decision

Prioritize measured attention/cache-layout/backend work before fused sampling. Any native CUDA/CUTLASS or external backend spike should start with q_len=1 self-attention or cross-attention microbenchmarks on SM75, then pass the one-token logits gate and 15s generated-token equivalence before any throughput claim.
