# Mapperatorinator Agent Guide

## Safety

- Fail loudly, preserve unrelated changes, and avoid compatibility work outside declared compatibility surfaces unless requested.
- Never commit generated beatmaps, audio, weights, profiles, traces, caches, native builds, or other run artifacts.
- Keep profiling and optimized/native imports opt-in. Default inference must remain quiet and cold.
- Use short-lived branches and persistent worktrees for experiments. Commit and push reproducible checkpoints before remote jobs.

## Inference boundary

- V32 is the default compatibility surface. Preserve its output, APIs, metadata, performance, server behavior, and cold imports.
- Put optimized runtimes, schedulers, exactness logic, kernels, batching, and speculative work under `osuT5/osuT5/inference/optimized/`.
- Outside that package, allow only lazy selectors, validation, metadata, shared preparation/assembly, and narrow dispatch hooks.
- `inference.py` is the selector. `Processor` owns shared window preparation and output assembly. `server.py` remains V32-only until a separately approved optimized-server plan exists.

## Evidence and promotion

- Exact claims preserve token IDs/counts, stopping, RNG, timing/main semantics, request-local mutable state, and final `.osu` bytes. Any relaxation is documented drift, not exactness.
- Compare like with like. Single-song claims use synchronized untraced model time; batch promotion requires both first-main-to-last-main scheduler wall and complete request-to-output wall. Keep single, serial queue, offline batch, and server modes separate.
- Start from a current profile and a falsifiable end-to-end hypothesis. Prove at least `5%` realistic headroom before production work.
- Promote one gate at a time: component -> real tensors -> short loop -> smoke -> full song -> queue.
- Stop on the first exactness, ownership, memory, negative-wall, or insufficient-gain failure. Remove candidate runtime wiring, keep reusable verifier infrastructure, and record the lesson and revisit condition.
- Never present a projection, trace, synthetic prompt, model-free schedule, or isolated kernel result as production throughput.

## DCC

- Verify live account, partition, GPU, environment, cache paths, queue state, and profiler availability; do not reuse stale Slurm values.
- Use reproducible configs, one GPU experiment at a time, and an explicit branch worktree in every wrapper.
- Record commit, job, hardware, flags, cache state, artifacts, exactness, wall time, memory, decision, and revisit condition.

## Sources of truth

- Operations: `docs/inference_profiling.md`
- Current state: `notes/inference-status.md`
- Decisions: `notes/inference-experiment-ledger.md`
