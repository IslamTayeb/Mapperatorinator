# Bottleneck-First Stop/Go Gate

## Purpose

Record the current stop/go rule before any more RTX 2080/2080 Ti inference
optimization work. The goal is to avoid spending implementation time on large
but necessary decoder math, non-exclusive profiler buckets, or already rejected
sub-5% cleanup paths.

This note is diagnostic only. It is not a throughput claim.

## Current Baseline And Target

- Accepted fastest exact single-song baseline: full-song SALVALAI
  `7,639` main tokens in `28.243s`, or `270.475 tok/s`.
- `500 tok/s` requires `15.278s` model time.
- Required saving: `12.965s`.
- 5% model-time bar: `1.412s`.
- 10% model-time bar: `2.824s`.

## Refreshed Bottleneck Ledger

The refreshed `utils/summarize_decode_runtime_gap.py` ledger from the latest
current-stack artifacts still points at broad graph-replayed decoder compute:

| Boundary or target | Current projection | Idealized ceiling | Decision |
| --- | ---: | ---: | --- |
| `graph.replay` active-loop event | `15.585s` | `603.510 tok/s` if removed | Aggregate decoder compute; split before coding |
| decoder layers replay | `13.018s` | `586.784 tok/s` | Only measured 500-capable broad boundary |
| weighted full-forward replay | `17.120s` | `446.214 tok/s` | Useful ceiling, not enough for 500 |
| decoder stack hidden | `16.666s` | `458.350 tok/s` | Useful ceiling, not enough for 500 |
| captured linears above bandwidth floor | `2.129s` removable | `292.526 tok/s` at floor | Not enough alone; much is necessary math/data movement |
| graph replay shell gap | `0.082s` | `271.263 tok/s` if removed | Rejected standalone |
| static input copy | `0.220s` | `272.598 tok/s` if removed | Rejected standalone |

Standalone logits processor, sampling, graph input copy, duplicate capture,
per-linear call-form/native/cuBLAS work, self-attention-only work, and manual
Python/module recomposition are not current bottlenecks under the accepted
stack.

## Stop/Go Rule

Do not start production runtime or kernel code unless a current-stack proof
shows:

- the target is exclusive or overlap-aware enough to be meaningful;
- the avoidable portion is not mathematically required model computation;
- any graph-replay, bandwidth, or hardware floor is included;
- fantasy-free and fantasy-floor TPS are computed;
- projected full-song synchronized model-time saving clears `~1.35-1.4s`
  before production wiring, with `>=2.6s` as the stronger target.

Verifier ceilings and microbenchmarks can justify one bounded diagnostic, but
not a retained speed claim.

## Next Allowed Diagnostic

The smallest useful next diagnostic is a verifier-only whole decoder-layer
math/memory island in `utils/profile_decode_decoder_layer_island.py`. It should
replace the captured one-token `VarWhisperDecoderLayer` boundary, not compose
the already rejected per-linear/native-q1 pieces.

The diagnostic must:

- match the final hidden output and any self-cache K/V writes;
- keep the existing one-token logits replay gate passing;
- use CUDA-graph replay for the stop/go timing;
- project full-song saving with:

```text
saved_s =
  (repo_decoder_layer.cuda_graph_replay_ms_per_call
   - native_decoder_layer_math_memory_island.cuda_graph_replay_ms_per_call)
  * member_count * full_song_decode_steps / 1000
```

Stop if the saving is below the 5% bar, appears only in eager/ungraphed timing,
or comes from recombining already rejected narrow pieces rather than reducing
broad math/memory traffic.
