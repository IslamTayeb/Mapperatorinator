# Inference Status

## Production boundary

- V32 remains the default compatibility surface and server implementation.
- The accepted fast path is opt-in optimized single. Its runtime, state, logits logic, and kernels live under `osuT5/osuT5/inference/optimized/` and are loaded lazily.
- No optimized batch scheduler or optimized server is production-authorized.

## Current conclusions

- The opt-in optimized-single path is the only graduated non-V32 inference mode; its established SALVALAI frontier is approximately `270.475 tok/s` on one RTX 2080/2080 Ti in FP32 with SDPA.
- Minimal-runtime cleanup smoke `49708641` preserved performance within `5%`. After merging OliBomby `f345d7d`, reciprocal V32/optimized smoke `49709859` at `9a8b8fc` preserved exact main/timing tokens and final `.osu` bytes with new SHA `dc690dcc23483f3a4468c55d8e1ea722bd12a67e1ad9ff8e63986f0cbc8568a2`.
- Current exact batching approaches do not beat optimized serial on real songs. Fast fixed-prefix/model-only islands did not survive complete-step control, setup, dependency, compatibility, and tail costs.
- A bounded two-song Hybrid-B2 main-loop wall scout produced exact files but only paired `91` steps while `2,625` used a slow private fallback. It ran near `75 tok/s` and took `2.6-3.8x` optimized serial main wall, rejecting the candidate before the complete queue-wall suite. No crossover was measured for this candidate; N=3-5 were intentionally not tested.
- Current speculative and broad decoder/kernel replacements also fail the end-to-end `5%` bar.
- Relaxing exactness does not rescue the tested FP16/native-prefix shapes on an RTX 2080 Ti. Current-main sentinel job `49715303` passed candidate self-repeat, cache ownership, finiteness, and RNG gates, but projected FP16 framework/native main times were `31.699s`/`32.213s` against the `25.419s` target. An auxiliary fused-linear variant (`49715718`) improved the local FP16 native boundary by `13.73%` yet still projected `29.956s`, slower than accepted. The current accepted SALVALAI work inventory is `7,684` main tokens, `28.243949s`, and 12 active-prefix buckets including `832`.
- Shared specialized dispatch now recaptures the original decoder exactly: job `49718293` had zero measured FP32 layer/cache/logit drift and replayed `0.135%` faster than the cached accepted graph. FP16 native-self+BMM-cross still projected only `249.119 tok/s`, so production remains accepted FP32.
- No current decoder region justifies another fusion. After calibrating eager detail-range shares to production graphs, job `49719218` placed fused self attention at `0.972s`, MLP at `0.522s`, and q1 BMM cross at `0.424s`, all below the `1.412s` gate.
- PR #120 encoder precompute is rejected on the current 2080 Ti path. Job `49718592` saved only `0.349s` at B16 across all 87 windows, with `3.64e-4` max encoder drift and about 980 MB incremental peak VRAM.

## What is worth doing next

Prioritize exact single-song bottlenecks because the same path handles every
serial song and the singleton/tail work that defeated batching. Reopen batching
only for a materially different complete-step shape with a near-zero-overhead
B1 fallback and much higher real-song compatibility. Require reciprocal N=2
scheduler-main and complete request-wall wins before larger queues or
production policy work.

Operational procedure is in [`docs/inference_profiling.md`](../docs/inference_profiling.md).
Family decisions and revisit conditions are in the
[`experiment ledger`](inference-experiment-ledger.md).
