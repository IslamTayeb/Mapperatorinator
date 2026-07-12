# Active Diagnostic Single-Range Ceilings

## Purpose

Make active-prefix diagnostic profiles easier to use as reject gates before
writing native/runtime code. The new summary section estimates each CUDA-event
range's single-range ceiling:

> If this one measured range disappeared, what fantasy throughput would the
> diagnostic profile imply?

This is diagnostic-only. CUDA-event ranges are nested/non-exclusive and include
queued stream work, so the ceilings are not additive and are not throughput
claims.

## Code Change

Updated `utils/summarize_active_prefix_diagnostics.py` to emit
`cuda_event_single_range_ceilings` in JSON and print a ranked section with:

- event seconds and milliseconds;
- share of diagnostic `model_elapsed_seconds`;
- event calls;
- per-decode-step microseconds;
- estimated tok/s if that single range were removed;
- whether the event clears the local `5%` model-time keep bar.

Updated `tests/test_summarize_active_prefix_diagnostics.py` to cover the new
ceiling math. Updated `AGENTS.md` and `docs/inference_profiling.md` so future
runs use the summary instead of ad hoc parsers.

## Existing Artifact Check

Applied the new summarizer to the existing CUDA-event diagnostic profile from
DCC job `49231413`:

- Profile:
  `/work/imt11/Mapperatorinator/runs/active-diag-cuda-events-49231413-6f126d3/diagnostic/beatmapa7062cbcf44f4aa8a7c13f5fb51b6b6f.osu.profile.json`
- Local JSON summaries:
  - `/tmp/mapperatorinator-active-diag-49231413/main_active_summary.json`
  - `/tmp/mapperatorinator-active-diag-49231413/timing_active_summary.json`

Main-generation diagnostic slice:

| range | event seconds | model-time share | estimated tok/s if removed |
| --- | ---: | ---: | ---: |
| `decode_forward.cuda_graph` | `2.464s` | `58.1%` | `609.2` |
| `prepare_inputs` | `0.542s` | `12.8%` | `292.9` |
| `stopping_criteria` | `0.298s` | `7.0%` | `274.8` |
| `prefill_forward` | `0.237s` | `5.6%` | `270.6` |
| `logits_processor` | `0.108s` | `2.5%` | `262.1` |
| `sampling.multinomial` | `0.047s` | `1.1%` | `258.3` |

Timing-context diagnostic slice:

| range | event seconds | model-time share | estimated tok/s if removed |
| --- | ---: | ---: | ---: |
| `setup.compile_lookup` | `1.473s` | `50.3%` | `112.8` |
| `decode_forward.cuda_graph` | `0.430s` | `14.7%` | `65.7` |
| `prefill_forward` | `0.201s` | `6.9%` | `60.2` |

## Interpretation

This reinforces the current direction:

- `decode_forward.cuda_graph` is the only clearly large main-generation bucket.
  Future large wins need to reduce real decoder compute or move more exact work
  under a tighter decoder/runtime boundary.
- `prepare_inputs`, `stopping_criteria`, and `prefill_forward` clear `5%` in
  this diagnostic slice, but prior fast-prepare and stopping-specialization
  attempts regressed or were too small. Treat these as candidates only inside a
  broader exact `DecodeSession` runtime, not standalone quick patches.
- `logits_processor`, sampling, append, update, and duplicate graph capture are
  below the keep bar for this path. Do not start fused sampling/logits work from
  this evidence.

## Decision

Accepted as profiling infrastructure. No inference speed claim and no runtime
behavior change.
