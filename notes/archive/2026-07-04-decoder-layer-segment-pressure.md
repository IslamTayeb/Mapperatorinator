# Decoder Layer Segment Pressure Auditor

## Purpose

Split the current whole-layer replay bucket into segment-level stop/go evidence
before writing native decoder-layer code. The broad decoder layer is the only
current 500-capable idealized boundary, but that does not mean every large
sub-bucket is removable or worth production wiring.

This is diagnostic-only. It is not an inference throughput claim.

## Utility

Added:

```text
utils/summarize_decoder_layer_segment_pressure.py
```

The utility reads a `utils/profile_decode_decoder_layer_island.py` report,
uses the captured ABI dimensions, and compares graph-replayed segment time
against optimistic fp32 compute and memory-bandwidth floors. It reports:

- measured segment seconds;
- roofline floor seconds;
- above-floor headroom;
- TPS if only that segment reached its floor;
- whether the above-floor headroom clears the 5% and 10% full-song bars;
- whether any single segment would reach the `500 tok/s` target at its floor;
- recommended next scope.

## Validation

DCC run:

```text
/work/imt11/Mapperatorinator/runs/decoder-layer-segment-pressure-20260704141703-0a28a3e
```

Command:

```text
python3 utils/summarize_decoder_layer_segment_pressure.py \
  /work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_report.json \
  --json-output /work/imt11/Mapperatorinator/runs/decoder-layer-segment-pressure-20260704141703-0a28a3e/decoder_layer_segment_pressure.json
```

Source report:

```text
/work/imt11/Mapperatorinator/runs/decoder-layer-candidate-cache-write-49253972-7044278/decoder_layer_candidate_cache_check_report.json
```

Source context:

| Field | Value |
| --- | --- |
| baseline | `7,639` main tokens, `28.243s`, `270.474 tok/s` |
| target | `500 tok/s`, `15.278s`, `12.965s` required saving |
| 5% bar | `1.412s` |
| config | `profile_salvalai_smoke15`, sequence `9`, active prefix `128`, bucket `64`, fp32, SDPA |
| branch/commit | `codex/decoder-layer-segment-pressure`, `0a28a3e` |

## Result

| Segment | Measured | Roofline floor | Above floor | TPS if at floor | Clears 5% | Reaches 500 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| self-attention residual | `5.430s` | `1.513s` | `3.917s` | `314.028` | yes | no |
| cross-attention residual | `4.024s` | `1.626s` | `2.397s` | `295.563` | yes | no |
| MLP residual | `4.692s` | `2.789s` | `1.903s` | `290.010` | yes | no |
| whole decoder layer | `15.739s` | `5.928s` | `9.811s` | `414.435` | yes | no |
| MLP fc2+residual tail | `2.002s` | `1.392s` | `0.610s` | `276.443` | no | no |

The manual decoder runtime island in the same source report saved only:

```text
0.680159s
```

That remains below the 5% production keep threshold.

## Interpretation

The stop/go result is stricter than “the layer is large”:

1. Self-attention residual, cross-attention residual, and MLP residual each have
   enough above-floor headroom to justify verifier-only native/CUDA/CUTLASS
   exploration.
2. No single segment reaches `500 tok/s` at its optimistic floor.
3. Even this representative whole-layer replay at its optimistic floor reaches
   only about `414 tok/s`; the weighted stack roofline separately projects
   about `426 tok/s`.
4. The next implementation-worthy scope is therefore a multi-segment or
   whole-layer native math/memory verifier, not another narrow production path.

Any native experiment should first live inside
`utils/profile_decode_decoder_layer_island.py` as a verifier candidate, pass
the ABI/cache-write/logits gates, and stop unless CUDA-graph replay projects at
least `~1.35-1.4s` full-song saving, preferably `>=2.6s`.
