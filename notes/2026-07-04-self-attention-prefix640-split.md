# Self-Attention Prefix640 Split

## Purpose

Refresh the self-attention component split on current `main` after the accepted
fused RoPE/cache self-attention path and the rejected manual decoder-layer
runtime island. This checks whether the largest single decoder-layer segment
still has standalone target-sized headroom before starting broader native work.

This is diagnostic only. It is not an inference throughput claim.

## Run

- DCC job: `49250169`
- Branch/commit: `main`, `439f135`
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
/work/imt11/Mapperatorinator/runs/self-attn-split-prefix640-49250169-439f135/self_attn_split_prefix640.json
/work/imt11/Mapperatorinator/runs/self-attn-split-prefix640-49250169-439f135/summary.txt
/work/imt11/Mapperatorinator/runs/self-attn-split-prefix640-49250169-439f135/preflight.txt
```

## Correctness

The verifier gate passed:

- `pass=True`
- logits replay max abs: `1.9073486328125e-05`
- active prefix: `640`
- repo module, fused RoPE/cache island, native attention, post-`Wqkv`
  attention, and output projection CUDA graph replay allclose: `True`

## Result

Projected over `12` layers and `7,552` full-song decode steps:

| Boundary | Graph replay ms/layer | Projected full-song seconds |
| --- | ---: | ---: |
| Repo self-attention island | `0.0732743ms` | `6.640s` |
| Fused RoPE/cache self-attention island | `0.0734329ms` | `6.655s` |
| Pre-attention setup only | `0.0475738ms` | `4.311s` |
| Native attention only | `0.0355599ms` | `3.223s` |
| Fused post-`Wqkv` attention only | `0.0358595ms` | `3.250s` |
| Output projection only | `0.0067894ms` | `0.615s` |

The accepted fused path is already effectively the repo path under the current
runtime:

```text
repo -> fused island delta = -0.014s
manual native island -> fused island delta = 2.447s
native attention only -> fused post-Wqkv attention only = -0.027s
```

The full self-attention island ideal-free ceiling is only `353.615 tok/s`.
Component timings are not additive because the component graph sum (`8.149s`)
exceeds the measured repo island (`6.640s`), so use them for target ordering,
not exact savings.

## Utility Update

`utils/profile_decode_self_attention_island.py` now emits graph-replay component
projections under `projected_full_song[*].cuda_graph_component_seconds`,
`cuda_graph_component_sum_s`, and
`cuda_graph_component_unexplained_vs_repo_s`. This avoids relying on the older
ungraphed component ceiling fields, which can be misleading after DecodeSession
CUDA graph replay removes launch overhead.

## Decision

No optimization graduated.

Standalone self-attention is no longer a 500-class path. The full island is
large enough to matter, but deleting it entirely would still only reach about
`354 tok/s`, and the accepted fused RoPE/cache path already captured the clear
setup/cache win. The remaining attention math and projection pieces are
meaningful only as part of a broader decoder-layer native math/memory island.

Do not start another standalone self-attention wrapper or post-`Wqkv` attention
kernel from this evidence. Future work should combine self-attention, MLP,
cross-attention, residual/glue, and possibly layer-level buffer ownership in one
verifier-only broad native candidate before any production wiring.
