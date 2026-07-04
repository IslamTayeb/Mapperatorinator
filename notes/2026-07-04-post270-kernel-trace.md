# Post-270 Current-Stack Kernel Trace

## Purpose

Refresh kernel-level attribution on the current fastest exact opt-in stack before
starting any new runtime or kernel implementation work. This was diagnostic
only; torch profiler wall time is not a throughput claim.

## Run

- Branch: `experiment/post270-kernel-trace`
- Job: `49250089`
- Node/GPU: `dcc-core-ferc-s-z25-20`, NVIDIA GeForce RTX 2080 Ti, SM75
- Driver/CUDA shown by `nvidia-smi`: `595.71.05` / `13.2`
- Torch/CUDA: `2.10.0+cu128` / `12.8`
- Transformers: `4.57.3`
- Job commit: `376898e`
- Config: `configs/inference/profile_salvalai_smoke15.yaml`
- Trace label: `main_generation.seq9`
- Precision/backend: `fp32`, `attn_implementation=sdpa`
- Fast stack:
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
  - `inference_native_decode_kernels=true`
  - `inference_native_q1_self_attention=true`
  - `inference_native_q1_rope_cache_self_attention=true`

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/post270-kernel-trace-49250089-376898e/beatmap1137438dcd894a2c838a231701aff4d4.osu.profile.json
/work/imt11/Mapperatorinator/runs/post270-kernel-trace-49250089-376898e/torch_profiles/000_generation_main_generation_seq9.trace.json
/work/imt11/Mapperatorinator/runs/post270-kernel-trace-49250089-376898e/kernel_summary_v2.json
/work/imt11/Mapperatorinator/runs/post270-kernel-trace-49250089-376898e/compare-current-gap-main.json
```

## Equivalence

Compared against:

```text
/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/profile.json
```

Result:

- Same-calculation metadata: PASS
- Output artifact hash: PASS, `ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`
- Main generated tokens: PASS, `1,084 / 1,084`
- Performance gates: intentionally not used because the candidate traced
  `main_generation.seq9` under torch profiler. Outer wall and per-window checks
  regress as expected from profiler overhead.

The diagnostic profile still reported stable synchronized map model time:
`1,084` map tokens in `3.997s`, `271.218 tok/s`, with
`model_generate_host_gap_seconds` about `0.001s`.

## Kernel Summary

`utils/summarize_torch_trace_kernels.py` now parses torch-profiler Chrome
traces and buckets GPU kernel/memcpy/memset events. On the seq9 trace it found
`593.186ms` of GPU work across `111,075` GPU work events:

| Category | Time | Share | Count |
| --- | ---: | ---: | ---: |
| `gemv_gemm_linear` | `289.522ms` | `48.8%` | `25,557` |
| `attention_native_q1` | `109.898ms` | `18.5%` | `2,796` |
| `elementwise_reduce_gelu_fill` | `77.507ms` | `13.1%` | `39,973` |
| `copy_index_cat_memcpy` | `50.173ms` | `8.5%` | `29,122` |
| `layernorm` | `26.251ms` | `4.4%` | `8,683` |
| `sampling_sort_softmax` | `20.786ms` | `3.5%` | `3,498` |
| `attention_fmha_sdpa` | `16.404ms` | `2.8%` | `36` |
| `other_kernel` | `2.645ms` | `0.4%` | `1,410` |

Top named families:

- `q1_rope_cache_attention_kernel`: `109.898ms`, `18.5%`
- cuBLAS GEMV template kernels: about `173ms`, `29.2%`
- `gemv2T_kernel_val`: `86.113ms`, `14.5%`
- `vectorized_elementwise_kernel`: `57.231ms`, `9.6%`
- `vectorized_layer_norm_kernel`: `26.251ms`, `4.4%`
- `memcpy32_post`: `24.552ms`, `4.1%`
- `volta_sgemm`: `24.139ms`, `4.1%`
- residual FMHA/SDPA: `16.404ms`, `2.8%`

## Interpretation

This confirms the current stack's remaining work is not dominated by a small
standalone sampling, copy, or SDPA issue. The largest bucket is one-token
linear/GEMV/GEMM work, followed by the accepted fused native q1 self-attention
kernel and broad elementwise/reduction glue.

For SALVALAI, the accepted full-song baseline is `28.243s` for `7,639` main
tokens. `500 tok/s` requires about `15.278s`, so the remaining required saving
is about `12.965s`. The trace says the right question is still broad decoder
compute/runtime reduction, not another narrow kernel:

- linears/GEMV are target-sized, but prior per-linear native and cuBLAS SGEMV
  probes were exact and too small end to end;
- sampling/sort/softmax is visible but only `3.5%` in this trace and already
  failed to translate into production-sized model-time savings;
- copy/index/memcpy and layernorm are not enough alone;
- residual SDPA is now only `2.8%` because q1 BMM cross-attention and native q1
  self-attention already replaced the target-sized attention paths.

## Decision

Keep `utils/summarize_torch_trace_kernels.py` as diagnostic infrastructure and
use it before opening large Chrome traces. Do not start production code from
this trace alone.

The next useful bottleneck proof is an exclusive DecodeSession replay-gap probe:
compare production-style CUDA graph replay, isolated full-forward graph replay,
decoder-stack-plus-projection replay, static input copy, and a no-op graph shell
on the same prepared seq9 inputs. If the exclusive production-vs-replay delta is
below the 5% full-song bar, stop and do not implement another runtime path. If
it is target-sized, repeat weighted across active-prefix buckets before any
native/runtime branch.
