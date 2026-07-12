# Decode Replay Gap Probe

## Purpose

Add a verifier-only stop/go probe for the current `270.475 tok/s` exact
single-song stack. The question was whether the gap between production-style
DecodeSession CUDA graph replay and isolated same-input model/decoder graph
replay is large enough to justify another runtime-shell implementation branch.

This is diagnostic only. It does not run full generation and cannot claim a TPS
speedup.

## Utility

```text
utils/profile_decode_replay_gap.py
```

The utility:

- builds real `profile_salvalai_smoke15` one-token seq9 inputs;
- runs the same static-cache prefill and direct one-token raw-logit path used by
  existing verifiers;
- captures production-style graph replay with `_capture_decode_cuda_graph()`;
- captures isolated same-input full-forward graph replay;
- captures isolated same-input decoder-stack-plus-projection graph replay;
- measures static input copy and an optional empty graph shell floor;
- compares all graph outputs against the same expected raw logits.

## Run

- Branch: `experiment/decode-replay-gap-probe`
- Commit: `3257f57`
- DCC job: `49250100`
- Node/GPU: `dcc-core-ferc-s-z25-20`, NVIDIA GeForce RTX 2080 Ti
- Slurm status: `COMPLETED`, exit `0:0`
- Driver/CUDA shown by `nvidia-smi`: `595.71.05` / `13.2`
- Torch/CUDA: `2.10.0+cu128` / `12.8`
- Transformers: `4.57.3`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`
- Precision/backend: `fp32`, `attn_implementation=sdpa`
- Fast stack: active-prefix bucket64 CUDA graph, stateful monotonic logits,
  q1-BMM cross-attention, DecodeSession graph/cache reuse, native q1
  self-attention, fused RoPE/cache self-attention.

Artifact:

```text
/work/imt11/Mapperatorinator/runs/decode-replay-gap-49250100-3257f57/decode_replay_gap_seq9.json
```

## Correctness

The report passed all graph-output correctness gates:

| Gate | Result |
| --- | ---: |
| `production_graph_max_abs` | `0.0` |
| `full_forward_graph_max_abs` | `0.0` |
| `decoder_stack_plus_projection_graph_max_abs` | `0.0` |
| active prefix length | `128` |
| cache position | `[84]` |
| probe token | `12` |

## Results

Projected over the accepted SALVALAI decode-step count (`7,552` one-token
decode replays):

| Measurement | ms/call | Projected full-song seconds | Share of accepted `28.243s` model time |
| --- | ---: | ---: | ---: |
| production graph replay | `1.811697` | `13.681937s` | `48.44%` |
| isolated full-forward graph replay | `1.800818` | `13.599778s` | `48.15%` |
| decoder-stack + projection graph replay | `1.801175` | `13.602474s` | `48.16%` |
| static input copy | `0.029143` | `0.220091s` | `0.78%` |
| empty graph shell | `0.002089` | `0.015777s` | `0.06%` |

Deltas:

| Delta | ms/call | Projected full-song delta | If removed |
| --- | ---: | ---: | ---: |
| production - full forward | `0.010879` | `0.082158s` | `271.263 tok/s` |
| production - decoder stack + projection | `0.010522` | `0.079463s` | `271.237 tok/s` |
| full forward - decoder stack + projection | `-0.000357` | `-0.002696s` | noise/tie |
| production - full forward - static copy | `-0.018264` | `-0.137932s` | no positive exclusive gap |

The 5% full-song keep bar is `1.41215s`; the 10% bar is `2.8243s`; reaching
`500 tok/s` requires about `12.965s` saved from the accepted `28.243s` baseline.

## Interpretation

The production CUDA graph replay shell is not target-sized. It is only about
`0.08s` slower than isolated full-forward replay over the full SALVALAI decode
step count, and static input copy is only about `0.22s`. This is far below even
the remove-complexity band, let alone the `5%` keep threshold.

That means the large remaining gap is not an obvious removable wrapper around
`_capture_decode_cuda_graph()`. Combined with the post-270 kernel trace from job
`49250089`, the bottleneck remains broad decoder work: one-token linear/GEMV
work, native q1 attention, elementwise/reduce glue, layernorm/copy, and
surrounding per-token generation semantics. A production branch aimed only at
graph-shell cleanup, static input copy, or module-boundary rewrites would be
looking in the wrong place.

## Decision

Keep `utils/profile_decode_replay_gap.py` as diagnostic infrastructure. Do not
promote or start production runtime-shell work from this result.

Next implementation-class work should require a new target-sized proof from one
of these areas:

- broad decoder-layer/native runtime island that reduces real graph-replayed
  decoder compute;
- fused operation classes that combine linear/GEMV, elementwise/reduce, residual
  and cache/layout work rather than replacing individual linears;
- a new full-song/profile auditor result showing a different exclusive
  removable gap above the `5%` bar.
