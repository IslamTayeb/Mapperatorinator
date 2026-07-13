# Inference Status

## Production boundary

- V32 remains the default compatibility surface and server implementation.
- The accepted fast path is opt-in optimized single. Its runtime, state, logits logic, and kernels live under `osuT5/osuT5/inference/optimized/` and are loaded lazily.
- No optimized batch scheduler or optimized server is production-authorized.

## Current conclusions

- The opt-in optimized-single path is the only graduated non-V32 inference mode; its established SALVALAI frontier is approximately `270.475 tok/s` on one RTX 2080/2080 Ti in FP32 with SDPA.
- Minimal-runtime cleanup smoke `49708641` preserved performance within `5%`. After merging OliBomby `f345d7d`, reciprocal V32/optimized smoke `49709859` at `9a8b8fc` preserved exact main/timing tokens and final `.osu` bytes with new SHA `dc690dcc23483f3a4468c55d8e1ea722bd12a67e1ad9ff8e63986f0cbc8568a2`.
- Dead Hugging Face generation-compile selection was removed from optimized single at `6466b43`; the accepted hot path remains eager prefill plus manual CUDA graphs and native kernels. Reciprocal FP32/FP16 smoke preserved token IDs, stopping, dispatch/calculation metadata, and final `.osu` bytes. FP16 had one `6.194%` synchronized-main timing outlier despite materially better total stage wall, so repeat profiling should watch steady FP16 model time.
- Current exact batching approaches do not beat optimized serial on real songs. Fast fixed-prefix/model-only islands did not survive complete-step control, setup, dependency, compatibility, and tail costs.
- A bounded two-song Hybrid-B2 main-loop wall scout produced exact files but only paired `91` steps while `2,625` used a slow private fallback. It ran near `75 tok/s` and took `2.6-3.8x` optimized serial main wall, rejecting the candidate before the complete queue-wall suite. No crossover was measured for this candidate; N=3-5 were intentionally not tested.
- Current speculative and broad decoder/kernel replacements also fail the end-to-end `5%` bar.

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
