# Inference Batch And Offline Frontier

Last consolidated: 2026-07-10. This document is the current multi-song source of truth. Static IPC results describe the legacy V32 server. The optimized package now has exact one-token merged-batch physics evidence, but the exact offline engine has not yet produced a queue-level GPU throughput result.

## Objective

Exceed `500` scheduler-wall main-generation tok/s for a queue of at least five songs on one RTX 2080/2080 Ti while every request matches running that song alone with the same seed: generated token IDs/counts, stop behavior, final RNG state, and `.osu` bytes.

The primary metric is aggregate main tokens divided by wall time from the first main-generation start to the final main-generation finish. Also report full timing+main wall, cold setup, p50/p95 request latency, queue wait, per-request tokens/stops, active batch-size histogram, VRAM, graph/cache events, and useful GPU timeline/telemetry.

## Mode Separation

| Mode | Current status | Exactness status | Use |
| --- | --- | --- | --- |
| Serial multi-song | implemented | exact per song when strict suite passes | denominator and operational fallback |
| Static IPC server batching | implemented in V32 | shared-global RNG; throughput-only | legacy serving characterization |
| Static window `parallel=true` | implemented | rejected for current exact claim | separate non-equivalent mode |
| CPU continuous scheduler | implemented, model-free | lifecycle/state ledger only | verifier/planning infrastructure |
| Optimized offline engine | planned under `inference/optimized/` | must be per-request exact | primary `500+` target |
| Optimized online server | deferred | unproven | only after offline win |

Never combine or average results across these modes.

## Current Exact Serial Denominator

Jobs `49543717` (15-second) and `49543718` (full-song), commit `a709b86`,
ran two same-process passes over Lambada, PEGASUS, Ela ke Leitada, SALVALAI,
and Nube Negra with the full accepted `270.475` stack. Every repeat matched
its per-song main token hash and byte-identical `.osu` hash.

| Workload/scope | Main tokens | Active main wall | Active main tok/s | First-main to last-main wall | Scheduler-wall main tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15s cold/all, 10 runs | `11,910` | `43.772s` | `272.091` | `60.983s` | `195.299` |
| 15s warmed, 5 songs | `5,955` | `20.892s` | `285.044` | `28.326s` | `210.230` |
| Full cold/all, 10 runs | `85,268` | `328.361s` | `259.677` | `402.796s` | `211.690` |
| Full warmed, 5 songs | `42,634` | `163.702s` | `260.437` | `196.060s` | `217.454` |

For warmed full songs, the complete timing+main interval was `204.510s`:
`208.470` main tok/s and `228.782` total timing+main tok/s. The corresponding
warmed 15-second values were `203.457` main tok/s and `231.131` total tok/s.
The schema-v4 strict self-compare for the 15-second manifest passed shape,
token/output exactness, active-model, scheduler-wall, timing, segment, and
per-song gates.

This changes the target sizing. Perfectly removing serial gaps would raise the
15-second warmed denominator only from `210.230` to the observed active-wall
ceiling `285.044 tok/s` (`+35.6%`); reaching `500` still needs `1.75x` more
active-generation throughput. For full songs, `500` is `2.30x` current
scheduler-wall throughput and `1.92x` current active-wall throughput. A
scheduler-only rewrite therefore cannot hit the objective: merged/lane
execution or broader math/memory amortization must also win.

Coarse whole-job telemetry is diagnostic only. The full job sampled nonzero
GPU utilization `92.6%` of seconds, averaged `73.4%` utilization and `149.6W`,
and peaked near `2,702 MiB`; this includes timing generation and does not prove
that main decode kernels use the GPU efficiently. The 15-second job was nonzero
only `66.3%` of samples, confirming more scheduling/setup headroom for short
queues.

```text
/work/imt11/Mapperatorinator/runs/inference-denominator-five_smoke-49543717-a709b86/serial_multi_song-five_smoke-49543717-a709b86/suite_manifest.json
/work/imt11/Mapperatorinator/runs/inference-denominator-five_smoke-49543717-a709b86/serial_multi_song-five_smoke-49543717-a709b86/strict-self-compare.json
/work/imt11/Mapperatorinator/runs/inference-denominator-five_full-49543718-a709b86/serial_multi_song-five_full-49543718-a709b86/suite_manifest.json
```

## Best Existing Measurements

### Static IPC server

Static IPC requests currently share global server RNG. All results below are `same_calculation=false`, `server_rng_policy=shared_global`, and `token_equivalence_status=not_checked_shared_server_rng`.

| Workload | Evidence | Result | Decision |
| --- | --- | --- | --- |
| Five concurrent 15s requests, max batch 5 | job `49267768`, commit `1475062` | `7,234` tokens / `59.9996s` = `120.568` scheduler-wall tok/s; real B5 batches | harness validated, throughput-only |
| Ten concurrent requests, max batch 5 vs 10 | job `49268989`, commit `195dfd7` | `134.781 -> 151.011 tok/s` (`+12.0%`), wall `99.687 -> 93.159s` | max 10 is better for this workload, throughput-only |
| Ten-request metadata/arrival-ledger repeat | job `49269123`, commit `0e6346d` | max 10 reached `163.201 tok/s`, but generated-token non-shrink gate failed | infrastructure validation, not promoted |
| Twenty concurrent requests, max batch 10 vs 20 | job `49269905`, commit `b4039b0` | `158.123 -> 149.374 tok/s` (`-5.5%`); max memory `9,334 MiB` | reject max 20; stop larger static sweeps |

Capacity-20 artifacts:

```text
/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/maxbatch10/static-server-batch-maxbatch10-49269905/static_server_batch_manifest.json
/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/maxbatch20/static-server-batch-maxbatch20-49269905/static_server_batch_manifest.json
/work/imt11/Mapperatorinator/runs/static-server-capacity20-20260704-194727-b4039b0/compare-rerun.json
```

The 20-request max-10 point is the best tested capacity configuration, not an exact-output engine. Larger batches reduced queue wait but slowed model work, changed token totals, approached the 2080 Ti memory ceiling, and left many tail/singleton batches.

### Serial and static window denominator

Jobs `49267816`/`49267817`, commit `1475062`, compared compile-disabled five-song 15-second modes:

- serial: `69.727 tok/s`, exact repeat denominator;
- `parallel=true`: `58.749 tok/s`, token/output mismatch for all five songs and sequence-count mismatch.

Static window batching is rejected as an exact optimization from this evidence. `use_server=true` and `parallel=true` must fail loudly until a dedicated mixed-mode harness exists.

### Abandoned compiled server fast path

Branch `codex/batched-fast-decode-session` is abandoned at `a74537a` and must not be merged:

- best exact lockstep B5: `263.544` unique main tok/s;
- paired optimized serial 15-second reference: `288.703 tok/s`;
- B10: `199.721` unique main tok/s, token mismatch, fragmentation, and near-capacity VRAM.

This proves that pushing the batch-1 fast stack through the existing server/lockstep abstraction is not enough. Preserve the branch only as an audit trail. Revisit its ideas only if a lower-level active-prefix graph step or batched decoder runtime beats optimized serial with private per-request state.

### Exact merged one-token physics

Jobs `49546220`, `49546893`, `49546977`, and `49547025` measured the eager,
batch-compatible active-prefix/q1-cross path at B1/B2/B5/B8. Each merged row
matched an independently prefetched B1 reference for FP32 raw-logit allclose,
top-k, anchor/next sampled token, and private final generator state. This is a
fixed-shape one-token verifier and ceiling probe, not queue or full-output
evidence.

| Batch | Complete sampled-step tok/s | Model-only tok/s | Step wall | Peak allocated | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | `75.402` | `79.680` | `13.262 ms` | `1.069 GiB` | eager denominator |
| B2 | `149.341` | `171.313` | `13.392 ms` | `1.339 GiB` | exact; advance |
| B5 | `318.233` | `427.464` | `15.712 ms` | `2.100 GiB` | exact; advance |
| B8 | `439.636` | `669.064` | `18.197 ms` | `2.824 GiB` | exact; stop shape expansion |

B8 improves complete throughput `38.15%` over B5 and has ample VRAM, but it is
`13.66%` below ideal B5-scaled capacity and still below the `500 tok/s` queue
target. Model execution itself clears `500`; rowwise logits processing,
sampling, and host/launch control add `6.240 ms/step`, `34.29%` of complete
wall. Therefore do not expand merged batch size again or build a scheduler from
this microbenchmark. First run a bounded active-row sampling/control or graph
overhead scout at B8, then prove a multi-step mixed-song loop. Lane-pool
comparison remains open.

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b1-49546220-5dd463e/merged-b1.json
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b2-49546893-e1a51b5/merged-b2.json
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b5-49546977-437c7f5/merged-b5.json
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b8-49547025-b949018/merged-b8.json
```

Job `49545900` failed in config validation before model execution because the
stateful processor's required public active-prefix selector was missing. It is
a harness setup failure and carries no physics evidence.

B8 component profile job `49547132`, commit `17fe895`, decomposed the prior
`6.240 ms/step` complete-minus-model gap. Reaching `500 tok/s` at B8 requires
step wall below `16.000 ms`: remove `2.196871 ms`, or `35.207%` of the measured
gap.

| Isolated non-additive component | Wall / step | Fantasy-free B8 TPS | Clears required saving? |
| --- | ---: | ---: | --- |
| logits clone | `0.104 ms` | `442.160` | no |
| clone + logits processors | `3.318 ms` | `537.678` | **yes** |
| clone + top-p/top-k warpers | `1.442 ms` | `477.467` | no |
| softmax | `0.055 ms` | `440.976` | no |
| private-generator multinomial | `0.753 ms` | `458.606` | no |
| empty eight-row Python loop | `0.0003 ms` | `439.643` | no |
| idle CUDA synchronize | `0.005 ms` | `439.766` | no |

Only the logits-processor family has an individual target-sized ceiling. After
subtracting the separately measured clone, its approximate incremental cost is
`3.214 ms`, with a `533.950 tok/s` fantasy-free ceiling. The active processor
list contains monotonic time-shift masking and conditional temperature; source
inspection shows conditional temperature performs a device-to-host `.cpu()`
lookback per row, but this job did not isolate those two processors from each
other. The next cheapest gate is therefore a processor-only subcomponent probe,
not an implementation. Component times overlap and must never be summed; CUDA
event intervals include device idle between host submissions and are not
kernel-active time.

Component report:

```text
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b8-49547132-17fe895/merged-b8.json
```

### CPU continuous-scheduler harness

`osuT5/osuT5/inference/continuous_batching.py` models arrivals, activation, round-robin/FIFO decode, stop reasons, cache-slot acquire/release, and slot generations. Strict manifests validate token/count recomputation, lifecycle arithmetic, state hashes, active-batch histograms, and cache-slot balance.

The harness is model-free, server-unwired, and has no TPS claim. `--allow-missing-state-hashes` is planning-only. Keep it as the state-ledger oracle for a future GPU scheduler.

## Why The Existing Server Is Inspiration, Not The New Runtime

The V32 server already demonstrates request collection and compatibility grouping, but it has the wrong exactness and execution boundaries for the new objective:

- shared global RNG prevents per-request equivalence claims;
- server batch elapsed time is attributed to every request;
- compile in the background batch thread hit a TorchInductor cudagraph TLS assertion in job `49267683`;
- fixed lockstep batching wastes work at divergent stops and performed worse than optimized serial;
- the maintainer wants V32 behavior preserved.

New scheduler/runtime code belongs under `osuT5/osuT5/inference/optimized/`. The legacy server may later receive only a lazy adapter after the offline engine wins.

## Batch Physics Gate

Do not build a production scheduler before comparing the two plausible exact execution shapes on current `main`:

1. merged one-token decode at `B=1/2/5/8` with fixed slots;
2. `1-4` independent B1 CUDA-graph lanes sharing immutable weights but owning private streams, graph instances, input/output buffers, generators, caches, cuBLAS workspaces, and request state.

Use distinct songs as well as identical-song controls. Cover request-order permutations, staggered arrivals, timing/main contexts, different prefix buckets, EOS/max-token stops, cache-slot release/reuse, and output assembly.

For each shape report:

- model-only replay and complete sampled-step aggregate tok/s;
- per-request token/RNG/cache/output equality against running alone;
- VRAM per lane/slot and maximum safe active count;
- graph capture/replay count, active-batch histogram, queue wait, and tail behavior;
- GPU timeline, clocks/power, memory traffic indicators, and useful idle gaps.

Choose a lane pool if concurrent B1 graphs are the best exact result, fixed-slot merged batching if small B scales, and reject both if neither improves optimized serial by at least `5%`.

If both fail, profile B1/2/4/8 component scaling before writing more runtime code. Only a measured target-sized ceiling can justify request-major multi-vector linear work, broader fusion, or operation-level/nano-batch overlap.

## Optimized Offline Engine Contract

Start with a known offline queue. Online arrivals and IPC are deferred.

Each request owns:

- request/song/window/context IDs and explicit seed;
- private generator and final RNG hash;
- logits processors and stopping state;
- encoder output and self/cross cache ownership;
- generated token buffer, cache position, cache slot, and slot generation;
- graph/lane state and output artifact metadata.

Scheduler requirements:

- separate encoder/prefill and token-decode queues;
- serial prefill initially, batched only if measured stalls exceed `5%` of scheduler wall;
- iteration-level scheduling, one token position per active request;
- group only compatible context, encoder shape, prefix bucket, sampling contract, and kernel mode;
- stable request slots; release immediately at stop; reset cache and increment slot generation before reuse;
- never sample stopped/dummy rows or advance their RNG;
- page caches only after measured capacity/fragmentation pressure;
- preserve exact request behavior independent of queue order.

Promotion ladder:

1. B1 parity with current exact single-song output;
2. identical-song B2/B5;
3. mixed five-song 15-second queue;
4. 10 and 20 queued requests;
5. five full songs;
6. larger full-song queue only if memory permits.

The five-song, three-seed (`12345`, `23456`, `34567`) gate must include reciprocal request orders, staggered arrivals, identical requests, mixed lengths, EOS/max-token stops, slot reuse, timing contexts, and byte-identical final maps.

## Server Integration Gate

Do not add the optimized server mode until the offline engine is exact and wins materially.

The eventual adapter must:

- keep V32 as the default and unchanged;
- lazily import the optimized engine;
- use one dedicated CUDA owner thread for model creation, compile, graph capture, and replay;
- keep IPC threads CPU-only and enqueue requests;
- achieve B1 optimized-server main model time within `5%` of non-server optimized single before B>1 testing;
- separately test queue latency, cancellation, failure propagation, stale sockets, and exact request isolation.

Until those gates pass, call results offline-engine throughput, not server optimization.

## Immediate Next Decision Points

- Finish the bounded B8 processor/short-loop/packed-prefill evidence before any merged scheduler wiring; retain the independent-lane comparison as the alternate physics shape.
- Run the independent B1 CUDA-graph lane comparison in its isolated experiment worktree.
- Build only the execution shape that clears `5%` exact-output aggregate improvement.
- Feed accepted single-song components, including any exact speculative verifier win, back into the offline engine and measure combined scheduler-wall throughput.
- Stop for user input before reduced precision, output/RNG relaxation, or maintainer-facing changes outside the optimized package/adapter boundary.
