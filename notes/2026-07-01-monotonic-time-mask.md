# Monotonic Time-Shift Mask Experiment

## Idea

Replace the per-token full-prefix scan in `MonotonicTimeShiftLogitsProcessor` with a stateful batch-size-1 fast path. The target was the trace-visible logits processor overhead: repeated `aten::_unique`, `aten::sort`, and mask construction from `torch.isin`/prefix scans.

## Result

Rejected and reverted.

Superseded context on 2026-07-02: this rejection applies to the old normal-generation/pre-graph experiment. After active-prefix CUDA graph replay changed the cost mix, diagnostics showed monotonic masking became a dominant CPU-side cost. Commit `a980c8d` reintroduced a default-off `inference_stateful_monotonic_logits_processor=true` flag scoped to active512 graph/simple batch-1 generation, and full-song job `49168188` accepted it with token equivalence PASS and `92.465 -> 134.873 tok/s` against the retained compile-only baseline. See `notes/2026-07-02-stateful-monotonic-graph.md`.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `01c18d6`
- Baseline job: `49109301`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json`
- Candidate commit: `9d7e5b7`
- Candidate job: `49109743`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-cand-49109743-9d7e5b7/beatmapd81b370ad0ac422cb1b5a01b3d3a093d.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=69.389, candidate=66.549, delta=-2.840 (-4.1%, worse)
model_elapsed_seconds: baseline=41.707, candidate=43.487, delta=+1.780 (+4.3%, worse)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

## Interpretation

The candidate preserved fixed-seed output tokens, but it made main generation slower. The likely explanation is that removing `torch.isin` and full-prefix scans added per-token state maintenance plus new mask/slice/`masked_fill`/`torch.where` work. This is the wrong tradeoff unless a future profiler shows a much cheaper replacement.

Do not keep this class of logits-processor complexity for a sub-5% or negative result.
