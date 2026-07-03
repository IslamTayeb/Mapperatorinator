# Batch And Multi-Song Inference Reporting Policy

## Summary

Future batch inference and multiple-song throughput improvements matter, but they must not be mixed with cold single-song inference claims.

## Result Classes

Use separate result classes:

| class | meaning | required reporting |
| --- | --- | --- |
| `cold_single_song` | fresh process/job, one song | fixed seed, token equivalence, main/timing/total/per-window non-regression |
| `warm_repeat` | same loaded model process, same song repeated | first run separately from runs 2..N, per-run token equivalence, RNG reset policy |
| `serial_multi_song` | multiple songs in one long-lived process | per-song token equivalence, first-song cold cost, warmed subsequent-song cost, aggregate throughput |
| `batched_multi_request` | real or simulated concurrent batching | batch-size distribution, queue/wait/service timing, per-request token equivalence if claimed equivalent |

## Why This Matters

Active-prefix validation showed the risk directly: bucketed active-prefix had strong post-warmup windows and order-dependent full-song results, but it did not graduate as the retained cold single-song baseline. A future batch-serving setup may still benefit from those warmed steady-state properties.

## Profiling Additions To Consider

- Add suite-level metadata such as `suite_id`, `song_index`, `run_kind`, `rng_reset_policy`, and `warmup_excluded`.
- Add summaries for generation by song, mode, batch size, and first-window versus post-warmup windows.
- If server batching is measured, add server-side queue wait, actual cross-request batch composition, and per-batch service time.

## Rule

Report warmed or multi-song improvements as warmed or multi-song improvements. Do not use them to replace the retained cold single-song baseline unless they pass the normal cold single-song graduation gates.

For realistic batch or multi-song scouting, prefer suites of at least 5 songs when feasible. If `batch_mode` and `normal_mode` diverge materially, report paired mode-specific results instead of averaging them into one headline number.
