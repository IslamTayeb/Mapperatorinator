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
`3.214 ms`, with a `533.950 tok/s` fantasy-free ceiling. Component times overlap
and must never be summed; CUDA event intervals include device idle between host
submissions and are not kernel-active time.

Component report:

```text
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b8-49547132-17fe895/merged-b8.json
```

Follow-up job `49547244`, commit `538bee7`, split the actual base processor list:

- `MonotonicTimeShiftLogitsProcessor`: `2.424 ms/step` including clone; clears
  the `2.197 ms` saving requirement by itself.
- `TemperatureLogitsWarper`: `0.203 ms/step` including clone; below threshold.

On the identical-prompt control only, applying the base processor list once to
the full B8 score tensor and then retaining rowwise warpers, softmax, private
generators, and multinomial draws preserved processed scores bitwise
(`max_abs=0`), top-k, sampled tokens, and final RNG for every row. Same-run
complete throughput improved `450.449 -> 544.383 tok/s` (`+20.85%`), reducing
step wall `17.760 -> 14.696 ms` and clearing the `500 tok/s` control target.

This is not a valid request-state design. It uses one shared processor object
and has not covered mixed prompts/songs, staggered arrival, EOS/max-token stops,
slot release/reuse, or processor-state reset. Do not wire it into a scheduler or
production runtime. The next gate must give the batched monotonic operation
explicit per-request state and pass those lifecycle cases.

Candidate report:

```text
/work/imt11/Mapperatorinator/runs/merged-batch-physics-b8-49547244-538bee7/merged-b8.json
```

The next verifier-only promotion is implemented at commit `aec60c6`: a fixed
B8, 16-step changing-prefix loop compares every active row against an
independent B1 session at every step for raw logits/top-k, processed scores,
sampled tokens, private RNG, and active self/cross-cache prefixes. Model-free
tests first prove forced-EOS/max-token stopping and reject inactive or dummy
draws. Its separate timing control compares eight private B1 processor calls
against one shared B8 processor call with one batched host transfer per step;
the job passes only with exact timed transcripts/RNG, `>=500 tok/s`, and a
`>=5%` same-run gain.

The first Slurm attempt, job `49547750`, reached a real RTX 2080 Ti with the
exact clean checkout and passed the prior-report guard, but exited before model
load because the DCC environment does not install `pytest` and the script
redundantly invoked it. This is harness-only evidence: it says nothing about
16-step correctness or throughput. The in-job pytest call is removed for a
corrected run; local targeted tests pass.

Corrected job `49547779`, commit `82b7d32`, completed the real verifier but
failed the configured strict cache gate. Across all 16 steps and all eight
rows, raw logits/top-k, processed scores/top-k, sampled token transcripts,
private final RNG, stops, and self-cache prefixes matched; maximum raw-logit,
processed-score, and self-cache absolute differences were `2.594e-4`,
`2.899e-4`, and `3.815e-5`. The cross-attention cache created by independent
B1 encoder/prefill passes versus merged B8 encoder/prefill missed
`atol=rtol=1e-4` from step 0 onward, with maximum absolute difference
`1.369e-3`. It was constant across decode steps.

The separate changing-prefix timing control preserved all 128 tokens and final
RNG while measuring `488.324 -> 547.760 tok/s` (`+12.17%`) and therefore still
shows a target-sized processor-sharing ceiling. Do not call this a promoted
exact result: top-level `pass=false`, and the job correctly exited nonzero.
Before any 256-step, mixed-song, lifecycle, or scheduler work, classify the
cross-cache B1-vs-B8 numeric drift under the campaign exactness policy or find a
stricter merged encoder/cross-cache construction. Do not loosen the gate after
seeing this result without explicit approval.

Corrected report:

```text
/work/imt11/Mapperatorinator/runs/merged-batch-loop16-49547779-82b7d32/merged-b8-loop16.json
```

Job `49548273`, commit `8a75179`, tested the engine-shaped alternative without
relaxing `atol=rtol=1e-4`: prefill eight private B1 requests serially, allocate
a fresh B8 static cache/session, and pack encoder output, self/cross K/V rows,
cross-cache update flags, prefill logits/positions, prompt/mask/frames, and
condition state into stable slots before merged decode.

The pack and all 16 decode steps passed. Packed cache rows, encoder outputs,
prefill logits, and input state were bitwise-equal before step 0. Across all
eight rows, raw logits/top-k, processed scores/top-k, 128 sampled tokens, final
private RNG, stop behavior, and self/cross caches passed at every step. Maximum
absolute differences were `2.899e-4` raw, `3.204e-4` processed,
`2.289e-5` self cache, and exactly `0` cross cache. This confirms the earlier
cross-cache failure came from separately reordered B8 encoder/prefill math, not
merged decode.

Setup was measured separately and excluded from decode TPS: the correctness
run spent `0.343913s` wall (`0.343808s` CUDA) on eight serial B1 prefills and
`0.046751s` wall (`0.046739s` CUDA) on allocation/pack/bitwise verification.
The changing-prefix decode control preserved timed transcripts/RNG and measured
`438.481 -> 497.328 tok/s` (`+13.42%`). This is a strong exact physics signal
but missed the explicit `500 tok/s` job gate by `0.53%`, so top-level
`pass=false`. Do not rerun for noise, call it a target hit, or production-wire
it. Retain the verifier and exact serial-prefill/pack design; wait for review
before any 256-step, mixed-song, lifecycle, or scheduler gate.

Packed-prefill report:

```text
/work/imt11/Mapperatorinator/runs/packed-prefill-batch-loop16-49548273-8a75179/merged-b8-loop16.json
```

### Accepted five-song compatibility ceiling

Commit `ef358de` added a CPU-only analyzer for the five accepted warmed
repeat01 smoke profiles from job `49543717`. It consumes the recorded exact
main token IDs rather than synthesizing model inputs, keeps only one active
window dependency per song, excludes the prefill-produced first token, and
regroups compatible decode rows after every selected group. The primary
`largest_compatible_first` policy breaks ties by group size, lowest current
64-token prefix bucket, then stable compatibility-key hash; reciprocal request
order preserved the complete schedule hash. Synchronous round-robin is
reported separately as policy sensitivity, not as an optimal schedule.

The five profiles contain `5,955` main tokens: `50` prefill-produced tokens and
`5,905` modeled decode tokens. The primary policy produced `3,489` group events:
`1,458` B1, `1,646` B2, and `385` B3. Thus `75.31%` of decode tokens are in
groups larger than one, but no group can exceed five because only five song
dependency chains exist. The synchronous sensitivity produced B1/B2/B3/B4/B5
event counts of `2,427/703/194/155/174`.

Using the fastest reviewed exact point for every partition selects the
normalized L1 graph lane (`414.905 tok/s`) over the old eager B1/B2/B5 points.
This optimistic bucket-extrapolated model gives `418.418` decode-only main
tok/s and `357.146` after linearly charging the measured B8 serial-prefill and
pack setup to all 50 windows. The latter is `+69.88%` over the accepted
`210.230` serial scheduler-wall denominator, so it clears the relative `5%`
gate, but it is separately below the absolute `500` objective. Only `5.17%` of
decode rows use the measured reference bucket 128; longer-bucket reuse is an
optimistic extrapolation.

Even the physically impossible all-B8 fantasy is only `501.539` decode-only
main tok/s and falls to `485.563` with just the initial measured setup, or
`415.994` with full linearized setup. Stop before a mixed B8 decode, 256-step
promotion, lifecycle scheduler, or runtime wiring. Accepted profiles prove
prompt/token counts, bucket schedules, runtime/sampling contracts, and exact
token transcripts; they do not record encoder/frame/condition tensor shapes.

The same report authorizes one narrower verifier scout. Exact reciprocal L2
lanes measured `2.568743 ms` model-only (`778.591 tok/s`) but `5.026351 ms`
complete (`397.903 tok/s`), leaving a `2.457608 ms` control gap. Reaching 500
requires a `<4.0 ms` step, or removal of `1.026351 ms` (`41.76%`) of that gap.
The prior B8 shared-processor control removed `3.064 ms`, evidence that the
required saving is plausible but not an additive L2 measurement. B2 accounts
for `55.75%` of accepted decode tokens under the primary schedule. A bounded
hybrid should therefore test two concurrent private B1 graph replays, join,
then one coordinator batched full-scan monotonic processor while preserving
private row warpers and generators. Capping its optimistic ceiling at the
measured L2 model-only interval projects `616.517` decode-only queue tok/s but
only `492.118` with full linearized setup. This authorizes only that verifier;
it is not a 500-TPS queue claim and does not authorize scheduler/runtime code.

Canonical report:

```text
notes/inference-mixed-queue-compatibility-report.json
SHA-256 9dae2b72b96556bc6df2e4e7fd04faa0d7d992fa09ff4d903470af27dd56ef9c
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

- Treat the exact packed B8 result as a stopped verifier signal: `497.328 tok/s`
  cleared the relative keep bar but missed the absolute queue microgate, so do
  not advance it to 256 steps, mixed songs, lifecycle, or scheduler wiring.
- Finish the reciprocal independent-B1 CUDA-graph lane comparison in its
  isolated experiment worktree; advance only if exact private state and the
  measured keep gate both pass.
- Build only the execution shape that clears `5%` exact-output aggregate improvement.
- Feed accepted single-song components, including any exact speculative verifier win, back into the offline engine and measure combined scheduler-wall throughput.
- Stop for user input before reduced precision, output/RNG relaxation, or maintainer-facing changes outside the optimized package/adapter boundary.
