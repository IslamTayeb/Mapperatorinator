# Mapperatorinator Agent Guide

## Safety And Repository Hygiene

- Fail loudly. Backward compatibility is unnecessary unless a task explicitly requires it.
- Keep inference profiling opt-in through `profile_inference`; default inference must not emit profile artifacts.
- Never commit generated beatmaps, audio, model weights, profiles, traces, native build artifacts, or CUDA caches.
- Use `main` only for accepted work. Put risky runtime/kernel experiments on short-lived branches, commit and push reproducible checkpoints before DCC jobs, and remove rejected runtime code after recording the result.
- Preserve unrelated user changes. Do not delete dated experiment notes until their evidence is represented in the canonical ledger and the deletion is a separately reviewed change.

## Maintainer Boundary

- The maintainer's V32 inference/server behavior is the compatibility surface. The default engine remains V32 and must not change output, imports, metadata, performance, or server behavior.
- Put new inference-engine work under `osuT5/osuT5/inference/optimized/`. Keep scheduler, exactness, metrics, speculative, single-song, batch, benchmark, and kernel implementations there.
- Existing files may receive only default-off selectors, validation, metadata, narrow lazy adapters, or small abstracted kernel dispatch hooks. Do not place optimized scheduler/runtime state machines or fused-kernel implementations in `inference.py`, `server.py`, or model files.
- `inference.py` and `osuT5/osuT5/inference/server.py:model_generate()` remain the public runtime control plane. Public flags must be declared in `config.py`/Hydra defaults, validated in `inference.py`, forwarded through loader/server entry points, and surfaced in profile metadata.
- Native extensions must not import or compile unless an explicitly selected optimized mode requests them.

## Campaign Contract

- Target hardware is one RTX 2080/2080 Ti in FP32 with SDPA as the current measurement baseline, not a permanent backend commitment.
- Primary goal: exceed `500` scheduler-wall main-generation tok/s for an exact-output offline queue of at least five songs. In parallel, improve the exact single-song frontier beyond `270.475 tok/s`; `500` single-song remains a stretch goal.
- Preserve two result classes:
  - `bitwise-calculation-exact`: required intermediate/cache behavior is bitwise identical in addition to output equality.
  - `exact-output`: internal FP32 values may be allclose, but generated token IDs/counts, stop behavior, final RNG state, timing/main behavior, and final `.osu` bytes match.
- Precision, sampling, RNG, output policy, windowing, overlap, model quality, or generated-token changes are non-equivalent unless explicitly approved and reported in a separate `documented-drift` table.
- Keep cold single-song, warm repeat, serial multi-song, static IPC batching, static window batching, offline continuous batching, and online server throughput as separate modes with separate baselines and gates.

## Measurement And Promotion

- Measure before and after every optimization. Use synchronized untraced model time for single-song TPS and scheduler wall from first main start to last main finish for offline/batch TPS.
- Every optimization path must begin with a current-stack profile and a written hypothesis: identify the measured bottleneck, the avoidable fraction, the fantasy/physical floor, the projected end-to-end ceiling, and the observation that would falsify the idea.
- Run the cheapest incremental falsification first and advance exactly one gate at a time: component microbenchmark -> real-tensor verifier -> short exact decode loop -> 15-second smoke -> full-song -> five-song/batch suite. Never graduate directly from a microbenchmark, roofline, or verifier to production runtime code.
- Stop at the first broader gate that removes the target-sized signal, breaks exactness, or introduces a material regression. Remove experimental runtime wiring immediately, keep only generally useful verifier/measurement infrastructure, and record what failed and what new evidence would justify revisiting it.
- Native whole-decoder-layer gate job `49550902`, commit `ce82dda`, preserved all 12 layer outputs and full-cache semantics at `1e-4` but projected only `1.989s` worst-order full-song saving, below its predeclared `2.8243s` strong bar. Do not run its weighted-bucket/short-loop/runtime follow-ups or merge the verifier branch; revisit only for a materially different candidate that first clears the same reciprocal prefix gate.
- Refresh the current bottleneck and fantasy/roofline ceiling before production work. Prove the target has at least `5%` avoidable end-to-end headroom on the current accepted stack.
- Reject and remove wins below `5%`; keep `5-10%` only when simple and isolated; treat `>=10%` as the normal strong promotion threshold.
- A full-song single-song promotion must check main generation, timing context, total profiled stage wall, token/record counts, fixed-seed token IDs, output SHA/size, and per-window regressions. Report scoped regressions explicitly.
- Use `utils/summarize_inference_profile.py --compare BASE CANDIDATE --strict-full-song --json-output REPORT.json` as the full-song gate.
- Start with `configs/inference/profile_salvalai_smoke15.yaml`; promote only exact, target-sized candidates to full-song SALVALAI and then the five-song suite.
- New decoder/runtime paths must pass, in order: one-token logits/top-k/cache, short and 256-step token/logit/RNG loops, 15-second smoke, reciprocal-order full-song comparison, and output-byte equality.
- Batch-equivalent claims additionally require per-request seed/generator, token IDs/counts, stop reason, final RNG hash, logits-processor state, cache state/slot generation, request-order invariance, staggered arrivals, slot reuse, and identical final output. Never sample stopped or dummy rows.
- Static server requests currently share global RNG. Label those manifests `server_rng_policy=shared_global`, `token_equivalence_status=not_checked_shared_server_rng`, and `same_calculation=false`; they are throughput diagnostics only.

## Runtime And Batching Rules

- Build and prove the offline optimized engine before adding an optimized server adapter. Keep encoder/prefill and token-decode scheduling separate; prefill remains serial until measured stalls exceed `5%` of scheduler wall.
- Treat each production song's windows as one dependency chain: a later window cannot become active before the preceding window's generated output exists. Do not use synthetic probe inputs to claim mixed-queue compatibility; derive model-free schedules from accepted exact-token production profiles, and label tensor-shape compatibility unproven when those profiles do not record encoder/frame/condition shapes.
- Hybrid L2 fixed-shape job `49552768`, commit `682abdc`, passed exact processed-score/token/RNG/resource gates at `670.026 tok/s` worst reciprocal order. Treat it only as a prefix-128 repeated-step ceiling: it authorizes one changing-prefix verifier, not scheduler/runtime wiring. The accepted five-song schedule projects `566.904` decode-only but `459.985` with current setup charges; setup would need about `42.43%` reduction if decode holds. Setup-only jobs `49552299`/`49552651` are not performance evidence.
- Legacy reciprocal jobs `49559982` (smoke) and `49560037` (full song) proved the two proposed graph-safety calculations token/output exact, but the full-song aggregate straddled noise (`-0.7%` and `+0.2%` main TPS by reciprocal pairing) and strict per-window gates failed. Do not change V32 for this verifier need. Keep the graph-capturable monotonic processor subclass and replacement helper under `optimized/` only; `osuT5/osuT5/inference/logit_processors.py` and its legacy tests must remain byte-identical to `main`. Final isolated-subclass regression job `49560227`, commit `2186d5a`, passed the H8 bitwise/capture/resource gates and again failed only the absolute performance bar (`514.276 < 623.657 tok/s`); report SHA `bb52add44510a677b28c417c3f54bdb52dc07fa032ba6b77600c4223373eb01d`.
- Weighted real-prefix H8 job `49559747`, commit `511e0ba`, passed bitwise raw-logit/cache/token/RNG/state gates but reached only `507.198-532.528 tok/s`, below the required `>623.657 tok/s`. Dependency-aware K3 then projected only `518.230 tok/s`, below its strict `>525` CPU gate. Do not build an offline scheduler/runtime/server from these results. The currently measured batch candidates are exhausted; reopen only after materially better accepted setup or complete-step evidence first restores more than `5%` projected headroom.
- Compare merged fixed-slot decode at `B=1/2/5/8` against `1-4` independent B1 CUDA-graph lanes before choosing a scheduler execution shape. Do not assume larger batches improve Turing throughput.
- Maintain explicit per-request generator, logits processors, stopping state, encoder outputs, self/cross caches, token buffer, cache position, slot generation, and graph state.
- Add paging only after memory fragmentation or fixed-slot capacity is measured as the limiting factor. Add operation-level overlap only after merged-batch and lane-pool experiments fail or leave target-sized idle gaps.
- Do not combine `use_server=true` with `parallel=true`. Do not enable server generation compile until a dedicated server-thread compile path passes exactness and performance gates.
- Static IPC socket identity must include the explicit runtime key for scheduling/backend knobs and be hash-shortened under AF_UNIX limits; never attach a normal client, web owner, or benchmark harness to a stale socket created with different runtime settings.
- The existing CPU continuous-scheduler harness is model-free verifier infrastructure. Do not call it a runtime throughput optimization or wire it into `InferenceServer` before RNG/logits/cache/output gates exist.
- Preserve `generation_compatibility_key()` and explicit request/group state for static server grouping; mutable runtime objects in generation kwargs must fail loudly or move into explicit request state.

## DCC Operations

- Run expensive profiling through Slurm on a GPU host. Verify live account, partition, GPU constraint, and tool availability; do not copy stale scheduler values.
- Give each concurrent experiment branch its own persistent DCC Git worktree and pass that checkout explicitly to its Slurm wrapper. Never move the shared source checkout while another experiment may submit or run from it; wrappers for branch-specific scouts should fail loudly when their explicit checkout is missing.
- Reproducible Hydra configs are mandatory. Keep the environment `bin` directory on `PATH`, and keep Hugging Face/model cache variables consistent between baseline and candidate.
- Project source is `/hpc/group/romerolab/imt11/projects/Mapperatorinator`; the environment is `/hpc/group/romerolab/imt11/envs/mapperatorinator`; data, caches, runs, and logs belong under `/work/imt11/Mapperatorinator`.
- Record job ID, commit, node/GPU, Slurm status, exact config/flags, run root, profile/manifest/compare paths, cache state, and telemetry/profiler availability in every experiment entry.
- Use Nsight Systems, CUDA events, torch traces, and available GPU telemetry for utilization diagnosis. Coarse `nvidia-smi` utilization alone is not proof of useful GPU work. Throughput claims must come from untraced profiles.

## Canonical Inference Documentation

- Current runbook: `docs/inference_profiling.md`
- Single-song frontier: `notes/inference-single-frontier.md`
- Batch/offline frontier: `notes/inference-batch-frontier.md`
- Accepted/rejected evidence ledger: `notes/inference-experiment-ledger.md`
- Historical dated notes remain evidence sources until a reviewed cleanup verifies that the canonical ledger preserves their job IDs, commits, artifacts, decisions, and revisit conditions.

## Protected Audit Trails

- Never merge `codex/batched-fast-decode-session`; it is abandoned at `a74537a`.
- Never merge `experiment/decoder-layer-runtime-island-do-not-merge` wholesale. Cherry-pick generally useful verifier/docs/tooling only after explicit review.
- Do not reintroduce a rejected experiment unless new current-stack profiling shows why its previous ceiling, exactness failure, or regression is stale. Consult `notes/inference-experiment-ledger.md` first.
