# Linear/MLP CUDA Graph Replay Probe

## Context

After native q_len=1 self-attention, current accepted exact opt-in baseline remains DCC job `49225493`: full-song SALVALAI `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu`.

The self-attention island graph probe showed a perfect self-attention-island removal would only reach about `292 tok/s`. The next question was how much graph-replayed linear/MLP work remains and whether simple call-form changes look better under CUDA graph replay.

## Utility Change

Extended `utils/profile_decode_linear_kernels.py` with `--cuda-graph-replay`.

The utility still captures real one-token decoder linear inputs through the direct decode verifier path and checks logits replay. The new flag also captures each diagnostic variant in a CUDA graph and reports replay time. This is diagnostic-only and is not an inference throughput claim.

## DCC Result

DCC job `49228407` on commit `4aa5ab9` completed on `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`, driver `595.71.05`, torch `2.10.0+cu128`, Transformers `4.57.3`.

- Run dir: `/work/imt11/Mapperatorinator/runs/linear-graph-49228407-4aa5ab9`
- Logits replay: PASS, `max_abs=0.0`
- Active prefix length: `128`
- Captured linears: `73`
- Operation counts:
  - self-attn qkv: `12`
  - self-attn out: `12`
  - cross-attn q: `12`
  - cross-attn out: `12`
  - MLP fc1: `12`
  - MLP fc2: `12`
  - output projection: `1`

CUDA graph replay per-call costs and full-song projections:

| shape / group | members | graph ms | projected full-song |
| --- | ---: | ---: | ---: |
| self-attn qkv `768 -> 2304` | `12` | `0.015424ms` | `1.398s` |
| MLP fc1 `768 -> 3072` | `12` | `0.020594ms` | `1.866s` |
| MLP fc2 `3072 -> 768` | `12` | `0.020608ms` | `1.868s` |
| `768 -> 768` self/cross projections | `36` | `0.006847ms` | `1.861s` |
| final output projection `768 -> 4069` | `1` | `0.024437ms` | `0.185s` |

Captured one-layer MLP island under graph replay:

| variant | graph ms | projected full-song, 12 layers |
| --- | ---: | ---: |
| functional MLP | `0.041098ms` | `3.724s` |
| addmm MLP | `0.041041ms` | `3.719s` |
| mv MLP | `0.041347ms` | `3.747s` |

## Decision

No optimization graduated.

Simple linear/MLP call-form changes remain too small under CUDA graph replay. The best MLP call-form delta is only about `0.005s` projected over full-song SALVALAI. The best individual linear variant deltas are also below the keep threshold.

Target sizing:

- A perfect MLP removal is only about `3.72s`.
- All captured graph-replayed linear calls total about `7.18s`, but that overlaps with the self-attention island because self-attn qkv/out linears are part of both counts.
- Self-attention plus MLP cannot reach `500 tok/s`; even if both were free, the rough ceiling is about `341 tok/s` before accounting for overlap and remaining runtime.
- The next target to size is graph-replayed cross-attention: q projection, q_len=1 attention reduction over encoder cache, out projection, and cache-reuse/layout overhead.

Do not implement per-linear or compiled-MLP rewrites from this evidence. Future native/CUTLASS/cuBLASLt work should target a broader decoder-block island and must first show a `>5%` full-song ceiling under CUDA graph replay.
