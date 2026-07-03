# Weighted Decoder Stack Diagnostic

## Summary

Ran `utils/profile_decode_decoder_stack_island.py` across the active-prefix bucket lengths from the full-song active64 replay distribution. This is diagnostic only and not an inference throughput claim.

Result: the decoder stack remains a huge target-sized surface, but the direct stack/model-wrapper/output-projection boundary is not where the win is. A simple Python stack bypass or final-projection cleanup should stay rejected.

## Run

- DCC job: `49233687`
- Status: `COMPLETED`, `00:08:59`, exit `0:0`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Commit: `1e4e1d5`
- Run dir: `/work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687`
- Summary JSON: `/work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687/weighted_decoder_stack_summary.json`
- Preflight failures: jobs `49233674` and `49233679` failed after `9s` due Slurm script quoting/preflight issues before useful profiling.

Flags matched the current fastest exact opt-in stack: generation compile, active-prefix bucket64 CUDA graph warmup0/min1, stateful monotonic, q1 BMM cross-attention, DecodeSession graph, native decode kernels, native q1 self-attention, and fused RoPE/cache self-attention.

## Weighted Result

Weighted by replay counts from the full-song active-prefix diagnostic distribution, total decode replays `7,552`:

| boundary | weighted seconds | share of `28.243s` model time | ideal free-boundary tps |
| --- | ---: | ---: | ---: |
| full model forward logits | `17.120s` | `60.6%` | `686.7` |
| model core decoder hidden | `16.860s` | `59.7%` | `671.1` |
| decoder stack hidden | `16.666s` | `59.0%` | `659.9` |
| output projection logits | `0.192s` | `0.7%` | `272.3` |
| model core + projection logits | `17.148s` | `60.7%` | `688.5` |
| decoder stack + projection logits | `17.063s` | `60.4%` | `683.3` |

All per-prefix reports passed allclose/logit replay checks.

## Interpretation

The output projection is not target-sized. Even deleting it would barely move the retained `270.475 tok/s` baseline.

The direct decoder stack plus projection is essentially the same size as the full model forward under graph replay: `17.063s` versus `17.120s`. The weighted gap is only `56ms`, so a Python-level wrapper bypass is not worth pursuing.

The stack itself is target-sized at `16.666s`, but that means the next kernel/runtime project has to reduce real decoder-layer compute and per-token runtime/control together. It cannot be another narrow wrapper around `Wqkv`, `Wo`, output projection, graph-cache lookup, or top-p sampling.

The accepted production model time remains `28.243s`, so the forward-vs-production gap is still about `11.1s`. Prior tail diagnostics put exact production-like tail CUDA at only about `1.6s`, so the remaining gap is likely distributed across CUDA graph replay synchronization/control, prefill/first-token effects, input preparation, and measurement boundaries rather than one obvious small kernel.

## Decision

Keep as diagnostic evidence. No production code changed. Next implementation-class work should be a broad decoder-layer/native runtime island with a stable C++/CUDA/CUTLASS/cuBLASLt strategy and verifier gates, not another narrow Python boundary rewrite.
