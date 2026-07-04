# Combined Bottleneck Ledger

## Purpose

Make the pre-implementation bottleneck proof mechanical before another broad
runtime or native-kernel branch. This combines the current active-loop gap,
weighted decoder-stack replay, decoder-layer replay, linear roofline, and graph
shell replay-gap evidence into one report.

This is diagnostic only. It is not an inference throughput claim.

## Utility Update

`utils/summarize_decode_runtime_gap.py` now additionally supports:

- weighted decoder-stack summaries through `--decoder-stack-report`;
- linear roofline summaries through `--linear-roofline-report`;
- graph replay shell gap reports through `--decode-replay-gap-report`.

## Inputs

Source artifacts:

```text
/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/output/beatmap7423e0d696194cd0a49e6924ee574fdd.osu.profile.json
/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/main_active_summary.json
/work/imt11/Mapperatorinator/runs/decoder-stack-weighted-49233687/weighted_decoder_stack_summary.json
/work/imt11/Mapperatorinator/runs/post270-gap-diag2-49232331-840dbc4/decoder_layer_island_seq9.json
/work/imt11/Mapperatorinator/runs/current-linear-roofline-49250117-407edbf/linear_roofline.json
/work/imt11/Mapperatorinator/runs/decode-replay-gap-49250100-3257f57/decode_replay_gap_seq9.json
```

Local validation command:

```text
python3 utils/summarize_decode_runtime_gap.py \
  /tmp/mapperatorinator_runtime_gap_profile.json \
  --active-summary /tmp/mapperatorinator_main_active_summary.json \
  --decoder-stack-report /tmp/mapperatorinator_weighted_decoder_stack.json \
  --decoder-layer-report /tmp/mapperatorinator_decoder_layer_seq9.json \
  --linear-roofline-report /tmp/mapperatorinator_linear_roofline.json \
  --decode-replay-gap-report /tmp/mapperatorinator_decode_replay_gap.json \
  --event-limit 8 \
  --json-output /tmp/mapperatorinator_runtime_gap_full_ledger.json
```

## Result

Accepted baseline remains `7,639` full-song SALVALAI main tokens in `28.243s`,
or `270.474 tok/s` by the auditor's rounded baseline inputs. Reaching
`500 tok/s` requires `15.278s` model time, or `12.965s` saved.

| Target | Projected seconds | Ceiling if idealized | Decision |
| --- | ---: | ---: | --- |
| `graph.replay` active-loop event | `15.585s` | `603.510 tok/s` if removed | Aggregate decoder compute target; split before coding |
| weighted full-forward replay | `17.120s` | `446.214 tok/s` | Not enough for 500 by itself |
| weighted decoder stack hidden | `16.666s` | `458.350 tok/s` | Not enough for 500 by itself |
| decoder layers replay | `13.018s` | `586.784 tok/s` | Only 500-capable broad compute ceiling |
| captured linears | `7.156s` | `362.259 tok/s` if free | Required model math/data movement |
| captured linears above bandwidth floor | `2.129s` | `292.526 tok/s` at floor | Below 10% bar as a standalone path |
| production graph replay shell gap | `0.082s` | `271.263 tok/s` if removed | Rejected as standalone target |
| static input copy | `0.220s` | `272.598 tok/s` if removed | Rejected as standalone target |

The active-loop ledger also showed host gap below `1%`, and standalone
`logits_processor`, `graph.input_copy`, and `sampling.multinomial` ceilings
below the 5% keep bar.

## Interpretation

The current theoretical floor is still far from `500 tok/s` unless a candidate
attacks broad decoder-layer compute. Matching the current weighted full-forward
replay boundary would be a useful improvement, but it would still stop around
`446 tok/s`, so a pure wrapper/shell/runtime-boundary cleanup is not enough.

The only currently measured 500-capable ceiling is the decoder-layer replay
boundary, and even that is an idealized diagnostic boundary. The next
implementation-class branch should therefore start from a broad decoder-layer
or multi-layer decoder runtime/kernel design that can reduce multiple adjacent
costs together: self-attention setup/cache/layout, MLP, cross-attention,
elementwise/residual/glue, and the per-token control needed to feed that
runtime.

Do not start another standalone graph-shell, static-copy, final-projection,
sampling, logits-processor, per-linear, or narrow MLP/self-attention tweak
without fresh evidence that its current-stack removable ceiling clears the
normal keep threshold.
