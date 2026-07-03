# Decode Linear Kernel Probe

## Purpose

Measure the actual one-token decoder `nn.Linear` shapes after the accepted q1 BMM opt-in stack, before starting native kernel work. This is a diagnostic microbenchmark only. It does not claim an inference speedup.

Campaign baseline at the time was DCC job `49213490`: active-prefix bucket64, CUDA graph warmup0, stateful monotonic logits processor, q_len=1 BMM cross-attention, `7,639` SALVALAI main tokens, `37.981s` model time, `201.125 tok/s`, token equivalence PASS. This was later superseded by persistent DecodeSession runtime job `49223294`: `35.337s`, `216.173 tok/s`, main/timing token equivalence PASS.

## Utility

Added `utils/profile_decode_linear_kernels.py`.

The utility:

- loads the real Mapperatorinator model and SALVALAI 15s smoke prompt;
- builds the direct one-token cached decode state through the existing verifier helpers;
- captures actual decoder linear inputs/weights/outputs with forward hooks;
- benchmarks equivalent PyTorch call forms with CUDA events;
- records logits replay equality so the captured step is tied to the real decode path.

It is not a throughput benchmark. Microbenchmarks can justify or reject implementation directions only.

## Jobs

| job | commit | state | note |
| --- | --- | --- | --- |
| `49222548` | `5194102` | `FAILED` | Utility failed before model work because the DCC run missed `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`. |
| `49222569` | `5194102` | `FAILED` | Utility reached code but failed on raw-model path assumption. |
| `49222601` | `d8f7eb8` | `FAILED` after valid JSON | Produced valid linears JSON, then failed only in a shell summary step with literal `$RUN_ROOT`. |
| `49222623` | `41a1af0` | `COMPLETED` | Final corrected probe. |

Final run root:

`/work/imt11/Mapperatorinator/runs/decode-linear-kernels-20260703-023503-41a1af0`

Artifacts:

- JSON: `decode_linear_kernels.json`
- stdout copy: `decode_linear_kernels.stdout.json`
- preflight: `preflight.txt`
- telemetry: `gpu_telemetry.csv`

Environment:

- Node: `dcc-core-ferc-s-z25-20`
- GPU: RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Driver/CUDA from `nvidia-smi`: `595.71.05` / `13.2`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `seed=12345`, `q1_bmm_cross_attention=true`

## Result

The final probe passed:

- logits replay: PASS, `max_abs=0.0`
- prompt tokens: `84`
- full prefix tokens: `85`
- active-prefix length: `128`
- captured decoder linears: `73`

Captured per-token linear call counts:

| operation | calls/token | shape |
| --- | ---: | --- |
| self-attn `Wqkv` | `12` | `[1,1,768] -> [1,1,2304]` |
| self-attn `Wo` | `12` | `[1,1,768] -> [1,1,768]` |
| cross-attn `Wq` | `12` | `[1,1,768] -> [1,1,768]` |
| cross-attn `Wo` | `12` | `[1,1,768] -> [1,1,768]` |
| MLP `fc1` | `12` | `[1,1,768] -> [1,1,3072]` |
| MLP `fc2` | `12` | `[1,1,3072] -> [1,1,768]` |
| output projection | `1` | `[1,1,768] -> [1,1,4069]` |

Representative CUDA-event timings on RTX 2080 Ti:

| signature | count/token | `F.linear` | best simple variant | isolated speedup |
| --- | ---: | ---: | ---: | ---: |
| `768 -> 768`, bias | `36` | `0.024588ms` | `matmul`, `0.024453ms` | `1.006x` |
| `768 -> 2304`, bias | `12` | `0.024951ms` | `matmul`, `0.024406ms` | `1.022x` |
| `768 -> 3072`, bias | `12` | `0.025122ms` | `matmul`, `0.024056ms` | `1.044x` |
| `3072 -> 768`, bias | `12` | `0.025245ms` | `matmul`, `0.024840ms` | `1.016x` |
| `768 -> 4069`, no bias | `1` | `0.025012ms` | flat | `1.000x` |

The simple PyTorch call-form alternatives (`matmul`, `addmm`, `mv`) are effectively flat. Even picking the best simple variant per signature saves only about `0.029ms/token`, roughly `0.22s` over `7,639` tokens, before integration overhead. That is far below the acceptance threshold.

Synthetic one-layer MLP block:

| variant | time |
| --- | ---: |
| `F.linear -> GELU -> F.linear` | `0.062998ms` |
| `addmm -> GELU -> addmm` | `0.060078ms` |
| `mv -> GELU -> mv` | `0.058672ms` |

The best synthetic MLP call-form variant is `1.074x` for the block. If this applied perfectly across all 12 layers, it would still save only about `0.40s` full-song model time, around `1%` of the accepted q1 path. That is useful as a ceiling check, not an implementation target.

Approximate captured-linear launch cost from the `F.linear` microbench is `~1.81ms/token`, or `~13.9s` over `7,639` tokens. This is close to the post-q1 profile's large GEMV/GEMM bucket, but the profile bucket also includes q1 cross-attention BMM and other GEMM-like kernels. Do not treat the full `36%` GEMV/GEMM bucket as pure `nn.Linear`.

## Decision

Do not pursue Python/PyTorch-level rewrites of decoder linears from `F.linear` to `matmul`, `addmm`, or `mv`. They are exact but too small.

Do not start broad native CUDA just to replace individual one-token linears one by one. The per-call floor is dominated by tiny repeated launches and memory movement, so isolated individual kernels would need to be part of a larger fused decoder-block/runtime plan to matter.

If linear work remains the target, the next plausible experiments are:

1. A verifier-only fused MLP/native block prototype for one decoder layer, because MLP is the cleanest dependency island and carries to a future autoregressive encoder-decoder.
2. cuBLASLt or native C++/CUDA only if it can fuse or schedule several operations under `DecodeSession`, not just swap one `F.linear` call form.
3. Separate q1 BMM and self-attention measurement before attributing remaining GEMM/FMA time to linears.

The bigger 500 tok/s path still requires multiple wins. This probe rejects simple linear call-form cleanup and strengthens the case for either q_len=1 self-attention/cache-layout work or a deeper fused decoder-step runtime.

## Follow-up Attempt: Eager Component Trace

After the linear probe, job `49222680` attempted a diagnostic-only torch profile of `main_generation.seq9` with both `inference_generation_compile=false` and `inference_active_prefix_decode_cuda_graph=false`, while keeping active-prefix bucket64, stateful monotonic, and q1 BMM enabled. The goal was to expose attention/MLP child ranges that CUDA graph replay hides.

Run root:

`/work/imt11/Mapperatorinator/runs/postq1-eager-component-trace-20260703-024100-202d3d4`

The job was canceled after `2m18s` with no profile artifact. It reached the selected traced seq9 window but torch-profiler overhead was too high for this broad eager path. Do not use this as timing evidence. If component attribution is still needed, build a lighter direct-step microprobe that captures real self-attention and q1 cross-attention tensors, instead of profiling a whole eager generate window.
