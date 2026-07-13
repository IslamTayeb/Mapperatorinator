# Specialized-kernel restoration and PR #120 follow-up — 2026-07-13

## Questions

1. Did the shared FP16/FP32 scout regress because it lost the accepted specialized
   decoder kernels?
2. Can the shared scout keep the original decoder layer and recover accepted FP32
   performance?
3. Does any remaining decoder region justify another fusion?
4. Does PR #120's batched all-window encoder precompute clear the current `5%`
   single-song gate?

Production never lost its kernels. The regression was confined to the old scout,
which replaced `VarWhisperDecoderLayer.forward` with a reference layer and thereby
bypassed native q1 self attention, fused RoPE/cache q1 self attention, and q1 BMM
cross attention. It also combined the decoder with a custom native cross kernel
that was not part of the accepted path and was already slower.

## Implementation

- Branch: `codex/specialized-kernel-restoration`
- Worktree: `/work/projects/Mapperatorinator-worktrees/specialized-kernel-restoration`
- Specialized dispatch commits: `a1ae3e2`, `4f36710`
- Shared parity/profiler commit: `9de2438`
- Encoder ceiling commits: `5454f06`, `3d670d2`
- Region profiler commits: `d13a41a`, `948793e`

The shared policy now keeps the original decoder layer. FP32 selects the existing
production q1 kernels; FP16 selects their dtype-generic scout equivalents. Both
use accepted-style q1 BMM cross attention. FP16 BMM scores and value reduction
accumulate in FP32. Production defaults remain FP32 and unchanged.

OliBomby's terminology comment also applied: optimized metadata now reports
`decoder_loop_backend=active_prefix_cuda_graph` and
`torch_compile_enabled=false` instead of calling direct CUDA-graph replay
`generation_compile`.

## DCC environment

- GPU: NVIDIA GeForce RTX 2080 Ti, 11,264 MiB
- PyTorch: `2.10.0+cu128`
- CUDA: `12.8`
- Transformers: `4.57.3`
- Accepted workload: SALVALAI, FP32/SDPA, seed `12345`, batch 1, CFG 1, one beam
- Accepted inventory: 87 windows, 7,684 main tokens, 12 active-prefix buckets

## Specialized FP32 parity and FP16 gate

Job `49718293` measured real prefix buckets `128`, `576`, and `640` with 100
warmups and 1,000 reciprocal CUDA-graph replays.

| Variant | Weighted measured replay | Projected main | Projected throughput |
| --- | ---: | ---: | ---: |
| Cached accepted FP32 | `9.075806s` | `28.243s` anchor | `272.067 tok/s` anchor |
| Recaptured shared-specialized FP32 | `9.063594s` | `28.247s` including capture delta | `272.028 tok/s` |
| FP16 framework self + BMM cross | `13.381216s` replay delta vs anchor | `32.612s` | `235.617 tok/s` |
| FP16 native self + BMM cross | `11.656390s` replay delta vs anchor | `30.845s` | `249.119 tok/s` |

The recaptured FP32 graph had zero layer, cache-key, cache-value, and logit drift
at all measured buckets. All cache ownership, active-write, future/cross-cache,
determinism, finiteness, and memory gates passed. Its weighted replay was
`0.1346%` faster than the cached accepted graph, so the declared exact/within-1%
parity gate passed.

FP16 native self was `5.80%` faster than FP16 framework self at the matched
prefix boundary, but both remained slower than accepted FP32. FP16 native worst
measured max-absolute drift was approximately `0.0583` at the layer, `0.6717` in
cache, and `0.6256` in logits. No FP16 production promotion was performed.

Slurm records job `49718293` as failed because the component summarizer correctly
returned `STOP_COMPONENT_SCOUT` after FP32 parity passed but neither FP16 variant
met the speed target.

## Calibrated decoder-region ceiling

The first region job, `49719079`, demonstrated why eager profiler time cannot be
used directly: eager/detail-range execution was roughly an order of magnitude
slower than production CUDA-graph replay. Job `49719218` calibrated each eager
region share to a matched production graph measurement before applying the gate.

| Region | Calibrated weighted seconds | Clears `1.412s`? |
| --- | ---: | :---: |
| Fused self attention | `0.971606` | No |
| MLP | `0.522099` | No |
| q1 BMM cross attention | `0.424123` | No |
| Self norm + QKV | `0.235600` | No |
| Cross norm + query | `0.170100` | No |
| Self output projection | `0.153345` | No |
| Cross output projection | `0.144497` | No |
| Final norm + logits | `0.016890` | No |

The mapped regions sum to an optimistic `2.638260s`, but that would require making
every mapped decoder operation free. The largest single concrete region is only
`0.971606s`, so no additional kernel or fusion is authorized. Residual adds,
dropout, and other unmarked work remain explicitly unattributed.

## PR #120 encoder-precompute ceiling

Job `49718525` was a setup-only failure because the GPU wrapper redundantly
invoked unavailable `pytest`; no model measurement occurred. The corrected job
`49718592` compared B1/B2/B4/B8/B16 over identical accepted FP32 tensors. It used
the existing `max_batch_size` configuration as requested by OliBomby.

| Encoder batch | Complete precompute | Saved vs B1 | Max drift | Incremental peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `2.429968s` | — | exact | `318,774,784` B |
| 2 | `2.207607s` | `0.222360s` | `5.329e-4` | `362,831,872` B |
| 4 | `2.152227s` | `0.277741s` | `1.895e-4` | `450,928,640` B |
| 8 | `2.087643s` | `0.342325s` | `3.639e-4` | `627,109,888` B |
| 16 | `2.081407s` | `0.348560s` | `3.639e-4` | `979,501,056` B |

B16 saved only `0.348560s`, about `1.23%` of the `28.243s` main baseline, while
requiring a 273.7 MB output store and roughly 980 MB incremental peak VRAM. It
does not clear the `1.412s` gate. No encoder store or production wiring was added.

## OliBomby comment decisions

- `max_batch_size`: applied to the encoder ceiling; no hard-coded runtime batch.
- Avoid duplicate fast methods: already satisfied by runtime binding and shared
  Processor preparation/assembly.
- Direct graph versus `torch.compile` terminology: applied to metadata.
- Server/Web UI exposure: not applied; the approved boundary keeps `server.py`
  V32-only, and prior optimized server batching was slower than accepted single.
- Default BF16: not applied on the RTX 2080 Ti/Turing target.
- Sampling inside the graph: not repeated; the exact production tail graph had
  already produced only `+0.375%` synchronized model throughput and was reverted.
- `torch.compile` atop the graph: not repeated; the PR path had a `176.8s` cold
  compile and the accepted pybind q1 operation is not Dynamo-traceable.

## Decision

Keep the restored shared dispatch and reusable verifiers. The shared framework
itself does not regress when it retains the accepted topology. Keep production
FP32 unchanged, reject the tested FP16 variants, reject PR #120 encoder
precompute on current Turing hardware, and do not implement another decoder
fusion from the current profile.

Revisit only if the GPU/framework changes or a materially different concrete
region first demonstrates at least `1.412s` calibrated weighted headroom.
