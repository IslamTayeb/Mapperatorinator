# Decoder Layer Candidate Prefix640

## Purpose

Run one bounded current-main diagnostic before any broad decoder-layer runtime
work: does the existing verifier-only manual whole-layer runtime island show
target-sized CUDA-graph replay headroom at the dominant long active-prefix
bucket?

This is diagnostic only. It is not an inference throughput claim.

## Run

- DCC job: `49250163`
- Branch/commit: `main`, `d051721`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
  `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Driver/CUDA shown by `nvidia-smi`: `595.71.05` / `13.2`
- Torch/CUDA: `2.10.0+cu128` / `12.8`
- Transformers: `4.57.3`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`
- Active prefix override: `640`
- Precision/backend: `fp32`, `attn_implementation=sdpa`
- Current fast stack: active-prefix bucket64 CUDA graph, stateful monotonic
  logits, q1-BMM cross-attention, DecodeSession graph/cache reuse, native q1
  self-attention, fused RoPE/cache self-attention.
- Warmup/iters: `100` / `2000`

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-prefix640-49250163-d051721/decoder_layer_candidate_prefix640.json
/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-prefix640-49250163-d051721/summary.txt
/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-prefix640-49250163-d051721/preflight.txt
```

## Correctness

The verifier gate passed:

- `pass=True`
- logits replay max abs: `0.0`
- active prefix: `640`
- `repo_decoder_layer` CUDA graph replay allclose: `True`
- `manual_decoder_runtime_island` CUDA graph replay allclose: `True`
- self-attention, cross-attention, and MLP residual segment graph replay
  allclose: `True`

## Result

Projected over `12` layers and `7,552` full-song decode steps:

| Boundary | Graph replay ms/layer | Projected full-song seconds |
| --- | ---: | ---: |
| Repo decoder layer | `0.1757168ms` | `15.924s` |
| Manual decoder runtime island | `0.1819684ms` | `16.491s` |
| Self-attention residual segment | `0.0772741ms` | part of segment sum |
| Cross-attention residual segment | `0.0362048ms` | part of segment sum |
| MLP residual segment | `0.0468790ms` | part of segment sum |

Manual island effect:

```text
saved_s = -0.5665507
speedup = 0.965644x
projected_tps = 265.155
```

The residual segment sum was `14.532s`; the layer-minus-segments remainder was
`1.392s`.

## Decision

Rejected as an optimization direction.

The manual whole-layer runtime island is exact but slower under the CUDA graph
replay path that matters for the accepted single-song runtime. This confirms
the prior instability/flat results at prefix128 and kills the idea that a
Python/module-boundary rewrite can unlock the next target-sized speedup.

Future broad decoder-layer work must improve real graph-replayed math or memory
behavior, likely through native/CUDA/CUTLASS/cuBLASLt work that reduces multiple
adjacent operation classes together. Do not productionize or repeat the manual
whole-layer island unless new profiling shows the current repo layer itself has
changed.
