# 500 tok/s Single-Song Campaign

## Purpose

Define the next inference campaign after the exact `200 tok/s` milestone. This is a single-song campaign: it is about faster normal inference for one song on RTX 2080/2080 Ti, not batching, multi-process execution, parallel multi-song throughput, or aggregate TPS.

Same-calculation remains mandatory. For the same settings and seed, the generated token IDs and generated map behavior should stay the same. Precision changes, quantization, sampling/RNG changes, output-policy changes, windowing/overlap changes, generated-token behavior changes, and output-length changes are non-equivalent unless explicitly labeled.

## Baseline

Use the current fastest exact opt-in path as the campaign baseline unless a new full-song exact-equivalent single-song run beats it:

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

DCC job `49223294` on RTX 2080 Ti validated the current fastest stack on full-song SALVALAI:

| metric | value |
| --- | ---: |
| main tokens | `7,639` |
| synchronized model time | `35.337s` |
| throughput | `216.173 tok/s` |
| timing throughput | `100.553 tok/s` |
| total timing+map stage | `48.493s` |
| equivalence | PASS main and timing tokens |

The `500 tok/s` target for the same `7,639` tokens implies `15.278s` main-generation model time, a further `56.8%` reduction from the accepted opt-in path.

## Campaign Direction

The next work should be a decoder-runtime and measured-kernel campaign, not a flag sweep.

1. Re-profile the accepted DecodeSession path first. Use untraced `profile_inference` for speed claims and Nsight/torch profiler only for diagnosis.
2. Keep future `DecodeSession` expansions verifier-first before broad kernel work. It should own stable encoder buffers, static input buffers, static cache state, active-prefix metadata, CUDA graph state, generated-token buffers, sampler/logits-processor state, EOS/stopping state, and an RNG ledger.
3. Keep `DecodeSession` default-off, and require generated-token identity, raw-logit/top-k equality, final RNG-state identity, and output behavior before broadening beyond the accepted simple single-song path.
4. Prefer stable PyTorch built-ins first, then isolated C++/CUDA/CUTLASS/cuBLASLt kernels if profiling predicts a meaningful `>5%` full-song single-song gain.
5. Avoid Triton-first work. Use Triton only if no stable alternative exists and the path is default-off, isolated, and strongly measured.

## Kernel Priorities

Use fresh post-q1 profiling to choose the target. Current evidence suggests the likely order is:

1. One-token linear/MLP/projection work if GEMM/BMM/linear remains dominant.
2. q_len=1 active-prefix self-attention if FMHA remains target-sized.
3. Fused q1 cross-attention only if accepted q1 BMM remains target-sized.

Do not retry simple BMM for active-prefix self-attention. The SM75 probe job `49221520`, run dir `/work/imt11/Mapperatorinator/runs/500tps-single-song-campaign-20260703-012406`, showed BMM is slower than SDPA at bucket-like self-attention lengths:

| length | SDPA | BMM range | interpretation |
| ---: | ---: | ---: | --- |
| `64` | `0.01686ms` | `0.05157-0.05970ms` | BMM slower |
| `96` | `0.02688ms` | `0.05072-0.05933ms` | BMM slower |
| `128` | `0.02858ms` | `0.05077-0.05896ms` | BMM slower |
| `512` | `0.06993ms` | `0.05177-0.05969ms` | BMM starts winning |
| `1024` | `0.14977ms` | `0.05136-0.05946ms` | BMM wins |

This supports the current split: q1 BMM is useful for long unmasked cross-attention, while active-prefix self-attention needs SDPA or a real custom kernel.

## Gates

Correctness gates, in order:

1. `utils/verify_one_token_decode.py --sequence-index 9`
2. `utils/verify_direct_decode_loop.py` with candidate flags
3. 15s middle-song SALVALAI smoke with fixed seed and recorded token IDs
4. full-song SALVALAI profile with recorded token IDs
5. generated map/output sanity check under the same settings

Acceptance gates:

- Exact generated main-token IDs must match.
- Timing-context token IDs must match when touched.
- Final RNG state must match for direct-loop/runtime changes.
- Full-song untraced `profile_inference` must improve main-generation throughput.
- Total timing+map stage wall time must not meaningfully regress.
- Per-window regressions must be documented; major regressions block promotion.
- Keep `>=10%` full-song wins.
- Keep `5-10%` only if simple, stable, or strategically useful.
- Remove `<5%` complexity unless it is verifier-only infrastructure.

## Initial Implementation Checkpoint

This original checkpoint added default-off campaign flags and a verifier-first `DecodeSession` wrapper around the existing exact one-token prefill/decode helpers. It was not a speed win by itself.

Follow-up job `49223294` accepted `inference_decode_session_runtime=true` plus `inference_decode_session_cuda_graph=true` as a default-off production opt-in for the simple active-prefix graph + stateful + q1-BMM path. Native kernels and chunked DecodeSession remain reserved.

## Codex `/goal` Prompt

```text
/goal Optimize Mapperatorinator single-song inference speed toward 500 tok/s main-generation throughput on RTX 2080/2080 Ti, same-calculation only. This is not a batching, multi-process, or multi-song aggregate-throughput goal. The objective is faster normal single-song inference while preserving the same generated map for the same settings and seed.

Current fastest exact opt-in baseline is active-prefix bucket64 CUDA graph + warmup0 + stateful monotonic logits processor + q_len=1 BMM cross-attention + persistent DecodeSession graph/cache reuse: full-song SALVALAI 7,639 main tokens, 35.337s synchronized model time, 216.173 tok/s, fixed-seed token equivalence PASS for main and timing. Keep this as the campaign baseline unless a new full-song exact-equivalent single-song run beats it.

Do not claim wins from batching, multiple processes, multiple songs in parallel, changed precision, quantization, sampling policy, RNG behavior, output policy, model quality, windowing/overlap, generated-token behavior, output length, or non-equivalent generated maps unless explicitly labeled non-equivalent. Same settings and same seed should generate the same token IDs and same output behavior.

Use a profiling-first loop. Start with fresh post-q1 profiling of the accepted opt-in stack. Use untraced profile_inference for throughput claims; use Nsight/torch profiler only for diagnosis. Start candidates with configs/inference/profile_salvalai_smoke15.yaml, fixed seed, use_server=false, attn_implementation=sdpa, and profile_record_token_ids=true. Promote only after one-token logits gate PASS, direct-loop token/logit/RNG gate PASS, 15s generated-token equivalence PASS, and full-song SALVALAI equivalence PASS.

Prioritize stable runtime and kernel work over framework churn. Prefer PyTorch built-ins first, then isolated C++/CUDA/CUTLASS/cuBLASLt kernels if profiling predicts a meaningful >5% full-song single-song gain. Avoid Triton-first work because maintainability/longevity matters; use Triton only if no stable alternative exists and the path is default-off, isolated, and strongly measured. Do not prioritize FlashInfer fp32 SM75, FA2 reruns, TensorRT-RTX FP32 Turing, sampling fusion, or bucket sweeps unless new profiling overturns prior evidence.

Build future DecodeSession runtime expansions before broad kernel work: stable encoder buffers, static input buffers, static cache state, active-prefix metadata, CUDA graph state, generated-token buffers, sampler/logits processor state, EOS/stopping state, and RNG ledger. Keep expansions verifier-first until exact generated-token, raw-logit/top-k, final RNG-state, and output behavior are proven across multiple sampled decode steps and windows.

After fresh profiling, pursue isolated native kernel work only for measured hotspots: one-token linear/MLP/projection work, q_len=1 active-prefix self-attention, or fused q1 cross-attention if q1 BMM remains target-sized. Do not retry simple BMM for active-prefix self-attention: the SM75 probe showed BMM is slower than SDPA at bucket64/96/128.

For every candidate, record DCC job id, node/GPU UUID, driver/CUDA, torch/transformers versions, commit, exact flags, profile paths, generated tokens, synchronized model time, tok/s, total timing+map stage wall, token equivalence, output/map equivalence status, timing-generation impact, and regressions. Keep >=10% full-song exact single-song wins; keep 5-10% only if simple, stable, or strategically unlocks DecodeSession/native-kernel work; remove <5% complexity by default. Commit and push clean accepted checkpoints, document accepted and rejected experiments in docs/inference_profiling.md and notes/, update AGENTS.md with durable conventions, and stop only when 500 tok/s exact single-song throughput is reached or profiling shows no remaining plausible exact-calculation path to a major gain.
```
