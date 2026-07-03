# Cross-Attention Island Probe

## Context

Current accepted exact opt-in baseline remains DCC job `49225493`: full-song SALVALAI `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu`.

After the self-attention and linear/MLP graph replay probes, the next target to size was decoder cross-attention: q projection, cached encoder K/V reuse, q_len=1 attention over `L=1024`, and output projection.

## Utility

Added `utils/profile_decode_cross_attention_island.py`.

The utility captures the real one-token decoder cross-attention module boundary from the direct decode verifier path and benchmarks:

- repo cross-attention module forward under `q1_bmm_cross_attention=true`
- manual q1 BMM cross-attention island
- q projection only
- precomputed q1 BMM attention only
- output projection only
- optional CUDA graph replay timings for each variant

This is diagnostic-only and not an inference throughput claim.

## DCC Result

DCC job `49228434` on commit `bc2a356` completed on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`.

- Run dir: `/work/imt11/Mapperatorinator/runs/cross-graph-49228434-bc2a356`
- Logits replay: PASS, `max_abs=0.0`
- Active prefix length: `128`
- Captured cross-attention modules: `12`
- K/V shape: `[1, 12, 1024, 64]`

CUDA graph replay per-layer timings:

| component | graph ms |
| --- | ---: |
| repo cross-attention module | `0.037054ms` |
| manual q1 BMM cross island | `0.036868ms` |
| q projection only | `0.007521ms` |
| q1 BMM attention only | `0.020390ms` |
| output projection only | `0.007431ms` |
| component sum | `0.035341ms` |

Full-song projection over `12` decoder layers and `7,552` one-token decode replays:

- Full cross-attention island: `3.358s`, `10.4%` of the accepted `32.217s` model time.
- q1 BMM attention reduction alone: about `1.848s`.
- q projection alone: about `0.682s`.
- output projection alone: about `0.674s`.
- If the entire cross-attention island were free, throughput would rise only to about `265 tok/s`.

## Decision

No optimization graduated.

The simple manual cross-attention rewrite is not useful under CUDA graph replay: `0.037054ms -> 0.036868ms` per layer, only about `0.017s` projected full-song.

The useful signal is target sizing. Cross-attention is still meaningful, but smaller than the self-attention island (`~6.075s`) and MLP island (`~3.724s`). Even making self-attention, cross-attention, and MLP islands free would not reach `500 tok/s`; the rough ceiling is about `401 tok/s` before overlap and remaining layernorm/sampling/output/runtime costs.

Next useful diagnostic is a whole decoder-layer graph replay island to size residual/layernorm/control overhead and understand how much of the remaining `32.217s` is inside decoder layers versus outside them. Do not implement a narrow cross-attention rewrite from this evidence.
