# Inference Status

## Production boundary

- V32 remains the default compatibility surface and server implementation.
- The accepted fast path is opt-in optimized single. Its runtime, state, logits logic, and kernels live under `osuT5/osuT5/inference/optimized/` and are loaded lazily.
- No optimized batch scheduler or optimized server is production-authorized.

## Exactness

Production-equivalent work preserves token IDs/counts, stopping, RNG, timing
and main semantics, request-local mutable state, and final `.osu` bytes.
Internal FP32 differences may be classified as exact-output only when every
observable gate still matches. Everything else is documented drift.

## Current conclusions

- Exact optimized single is the only graduated inference optimization; its established SALVALAI frontier is approximately `270.475 tok/s` on one RTX 2080/2080 Ti in FP32 with SDPA.
- Current exact batching approaches do not beat optimized serial on real songs. Fast fixed-prefix/model-only islands did not survive complete-step control, setup, dependency, compatibility, and tail costs.
- The latest two-song Hybrid-B2 queue produced exact files but only paired `91` steps while `2,625` used a slow private fallback. It ran near `75 tok/s` and took `2.6-3.8x` optimized serial wall. There is no measured batching crossover.
- Current speculative and broad decoder/kernel replacements also fail the end-to-end `5%` bar.

## What is worth doing next

Prioritize exact single-song bottlenecks because the same path handles every
serial song and the singleton/tail work that defeated batching. Reopen batching
only for a materially different complete-step shape with a near-zero-overhead
B1 fallback and much higher real-song compatibility. Require a reciprocal
direct N=2 wall win before larger queues or production policy work.

Operational procedure is in [`docs/inference_profiling.md`](../docs/inference_profiling.md).
Family decisions and revisit conditions are in the
[`experiment ledger`](inference-experiment-ledger.md).
