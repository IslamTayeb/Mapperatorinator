# Post-Native q1 Self-Attention Profiling

## Purpose

Re-profile the accepted native q_len=1 self-attention opt-in stack before starting another kernel project. This pass is diagnostic only. Throughput claims still come from untraced `profile_inference`, not torch profiler or Nsight wall time.

Current accepted full-song baseline remains DCC job `49225493`: full-song SALVALAI main generation `237.111 tok/s`, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu` output.

## Runs

An initial diagnostic job `49226943` produced useful artifacts, but it is not the official diagnostic because the DCC checkout was fast-forwarded while the job was still running. That mixed the untraced/torch commit metadata (`9461990`) with the later Nsight metadata (`06eb637`). The guard-only code change does not alter the valid runtime path, but the clean result below supersedes it.

Clean diagnostic job:

- DCC job: `49227916`
- Node/GPU: `dcc-core-gpu-ferc-s-h36-5`, RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`
- Commit: `06eb637`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir: `/work/imt11/Mapperatorinator/runs/post-native-clean-49227916-06eb637`

Flags were the accepted native opt-in stack:

```text
inference_generation_compile=true
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=64
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
inference_q1_bmm_cross_attention=true
inference_decode_session_runtime=true
inference_decode_session_cuda_graph=true
inference_native_decode_kernels=true
inference_native_q1_self_attention=true
```

## Untraced Smoke

Official smoke throughput comes from the untraced profile:

| metric | value |
| --- | ---: |
| main tokens | `1,084` |
| main model time | `4.156s` |
| main throughput | `260.840 tok/s` |
| timing tokens | `164` |
| timing model time | `3.094s` |
| timing throughput | `53.004 tok/s` |
| seq9 main throughput | `295.260 tok/s` |

The first map record still has native-extension/setup wall overhead: `seq0` generated one token, had `0.0556s` model time, but `2.508s` outer wall.

## Instrumentation Overhead

Torch profiler and Nsight preserved fixed-seed generated-token IDs, but they slowed the run and must not be used for TPS claims:

- Torch profiler main token equivalence: PASS (`1,084 / 1,084`).
- Torch profiler timing token equivalence: PASS (`164 / 164`).
- Nsight main token equivalence: PASS (`1,084 / 1,084`).
- Nsight timing token equivalence: PASS (`164 / 164`).
- Torch profiler main model time regressed `4.156s -> 6.144s`, and total stage wall regressed `13.696s -> 137.744s`.
- Nsight main model time regressed `4.156s -> 5.828s`, and total stage wall regressed `13.696s -> 16.580s`.

## Torch Profiler: seq9

Torch profiler on `main_generation.seq9` classified self-CUDA after excluding top-level ranges:

| bucket | self-CUDA | share |
| --- | ---: | ---: |
| GEMV/GEMM/linear | `333.528ms` | `40.0%` |
| elementwise/reduce/GELU/fill | `147.284ms` | `17.7%` |
| attention | `137.548ms` | `16.5%` |
| copy/index/cat/memcpy | `92.433ms` | `11.1%` |
| other | `66.128ms` | `7.9%` |
| sampling/sort/softmax | `30.168ms` | `3.6%` |
| layernorm | `26.998ms` | `3.2%` |

Top events:

- native self-attention kernel: `114.926ms`, `2,796` calls.
- GEMV/GEMM cublas kernels: `109.784ms`, `63.850ms`, `42.323ms`, `30.674ms`, `23.080ms`, `22.756ms`.
- residual FMHA: `22.621ms`, `36` calls.
- device-to-device copies: `30.771ms`, `20,518` events.
- layernorm kernel: `26.998ms`, `8,683` calls.
- index/cat copies: `17.953ms` and `16.672ms`.

Active-prefix diagnostic totals across the 15s smoke:

| diagnostic bucket | wall CPU |
| --- | ---: |
| decode forward | `1.532s` |
| steady decode forward | `1.481s` |
| prepare inputs | `0.749s` |
| logits processors | `1.060s` |
| sampling | `0.415s` |
| token append + stop | `0.864s` |
| stopping criteria | `0.686s` |

These CPU-side buckets include required synchronization/control and are diagnostic only; they do not by themselves prove a Python-only target.

## Nsight Kernel Split

Nsight whole-smoke kernel attribution:

| bucket | kernel time | share |
| --- | ---: | ---: |
| GEMV/GEMM/linear | `482.452ms` | `40.7%` |
| attention | `297.303ms` | `25.1%` |
| elementwise/reduce/GELU/fill | `208.724ms` | `17.6%` |
| sampling/sort/softmax | `118.935ms` | `10.0%` |
| copy/index/cat | `58.116ms` | `4.9%` |
| other | `12.945ms` | `1.1%` |
| layernorm | `7.155ms` | `0.6%` |

Top kernels:

- `volta_sgemm_128x64_tn`: `419.873ms`, `2,200` instances.
- residual FMHA: `297.303ms`, `720` instances.
- sampling/sort radix kernels: `32.996ms`, `31.694ms`, `20.335ms`, `18.581ms`.
- `volta_sgemm_128x128_tn`: `32.865ms`, `243` instances.
- GELU/elementwise/fill kernels are individually small but numerous.

GPU telemetry over the whole diagnostic job is mostly polluted by model load, profiler export, and idle phases. It is useful only as a sanity check: max GPU utilization `85%`, max power `199.9W`, max memory used `2.622GB`. Average utilization is not meaningful for throughput.

## Interpretation

The next plausible `>5%` exact single-song target is still one-token linear/MLP/projection work. The biggest remaining kernel bucket is GEMV/GEMM/linear in both torch profiler and Nsight (`~40%`). The prior `F.linear` call-form probe already rejected one-by-one Python/PyTorch rewrites, so any next attempt must be a real fused/native island, cuBLASLt/CUTLASS scheduling experiment, or decoder-block-level runtime change with measured ceiling above `5%` full-song.

Attention is still target-sized but the mix changed:

- Native q1 self-attention is now visible directly in torch profiler and costs `114.926ms` in seq9.
- Residual FMHA remains visible in Nsight, likely from prefill and other non-native attention paths.
- Do not assume another attention kernel will win until the exact residual attention source is separated into native self q1, cross q1 BMM, prefill, and any remaining SDPA/FMHA paths.

Sampling/sort/softmax is visible in Nsight at `10.0%`, but torch seq9 puts it at only `3.6%` filtered self-CUDA, and exact RNG/top-p semantics make it high-risk. It is not the first target unless fresh focused profiling proves it is consistently above the `>5%` full-song ceiling and exact RNG identity can be preserved.

## Decision

No new optimization graduated from this diagnostic pass.

Next implementation should not be a quick-tweak loop. Recommended order:

1. Build an isolated captured-tensor probe for one-token linear/MLP/projection islands under the current native opt-in path. Require a projected `>1.6s` full-song saving before production code.
2. If linear fusion fails, split the residual attention bucket by source before writing another attention kernel.
3. Treat sampling/logits fusion as lower priority and exact-RNG risky unless focused profiles make it target-sized.
4. Keep native-extension setup visible in cold wall comparisons; do not move it out of measured cold paths unless the setup path is still honestly accounted for.
