# Self-Attention Island Probe

## Context

Current accepted exact opt-in baseline remains DCC job `49225493`: full-song SALVALAI main generation `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu` output.

Post-native profiling showed the remaining kernel mix is dominated by linear/GEMV plus elementwise/copy/attention work. The next question was whether a larger self-attention island around the accepted native q_len=1 kernel could plausibly clear the `>5%` full-song bar.

## Utility

Added `utils/profile_decode_self_attention_island.py`.

The utility captures real one-token decoder self-attention module inputs from the direct decode verifier path, then benchmarks:

- repo module forward: `Wqkv -> RoPE -> static-cache update/slice -> native q1 attention -> Wo`
- manual native island with the same math
- pre-attention setup only: `Wqkv -> RoPE -> static-cache update/slice`
- precomputed native q1 attention only
- output projection only
- optional CUDA graph replay timings for the same variants

This is diagnostic-only. It is not an inference throughput claim.

## DCC Runs

Initial DCC job `49228322` on commit `1bc6050` failed before benchmarking because the utility passed `native_q1_self_attention` into `decode_one_token_raw_logits()`, which does not own that flag. Fixed in commit `9d31a07` by keeping the native flag in `generation_profile_context`.

DCC job `49228341` on commit `9d31a07` passed the logits replay gate but exposed a bad component attribution: the first `native_attention_only` helper recomputed q/k/v/cache setup. Fixed in commit `0bb3ce3` by precomputing q/k/v/mask before timing the native attention kernel.

DCC job `49228358` on commit `0bb3ce3` completed on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`.

- Run dir: `/work/imt11/Mapperatorinator/runs/self-attn-island-49228358-0bb3ce3`
- Logits replay: PASS, `max_abs=0.0`
- Active prefix length: `128`
- Captured self-attention modules: `12`
- Ungraphed timings per layer:
  - repo module: `0.563861ms`
  - manual island: `0.530670ms`
  - pre-attention setup: `0.437198ms`
  - native q1 attention only: `0.017640ms`
  - output projection only: `0.042195ms`

The ungraphed module projection was invalid because it exceeded real full-song model time: `0.563861ms * 12 * 7552 = 51.099s`, while the accepted full-song model time is `32.217s`. That is launch/Python/runtime overhead not representative of the production CUDA graph path.

DCC job `49228373` on commit `9381b25` repeated the probe with `--cuda-graph-replay` on the same node/GPU family.

- Run dir: `/work/imt11/Mapperatorinator/runs/self-attn-graph-49228373-9381b25`
- Logits replay: PASS, `max_abs=0.0`
- Active prefix length: `128`
- Captured self-attention modules: `12`
- CUDA graph replay timings per layer:
  - repo module: `0.067031ms`
  - manual island: `0.067020ms`
  - pre-attention setup: `0.049950ms`
  - native q1 attention only: `0.009563ms`
  - output projection only: `0.007050ms`
  - component sum: `0.066563ms`

Projected over `12` decoder layers and `7,552` one-token decode replays, the full self-attention island is about `6.075s`, or `18.9%` of the accepted `32.217s` model time. If the entire self-attention island became free, the full-song main-generation ceiling would be about `292 tok/s`, not `500 tok/s`.

## Decision

No optimization graduated.

The simple manual island did not improve CUDA graph replay time: `0.067031ms -> 0.067020ms` per layer, only `~0.001s` projected over full-song SALVALAI. Do not implement a Python/manual self-attention rewrite.

The useful signal is target sizing:

- The native q1 attention reduction itself is already small under graph replay: `~0.866s` full-song ceiling.
- Output projection in this island is also small: `~0.639s` full-song ceiling.
- The dominant self-attention subpiece is pre-attention setup, roughly `4.526s` full-song ceiling: qkv projection, RoPE, cache update/slice, and related layout work.
- Even a perfect self-attention island cannot reach `500 tok/s`; further large gains need broader decoder-block work across MLP/linear/projection/cache/layout, likely native/CUTLASS/cuBLASLt or a larger DecodeSession-owned runtime island.

Next work should use CUDA graph replay diagnostics before production kernel work. Ungraphed microbenchmarks are still useful for debugging, but they overstate launch/Python savings after the current DecodeSession graph path.
