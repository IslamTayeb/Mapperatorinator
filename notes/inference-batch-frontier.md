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
overhead scout at B8, then prove a multi-step mixed-song loop. The lane-pool
comparison below is now closed after its L=2 complete-step rejection.

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
### Independent B1 lane physics: rejected after L=2

The lane family was tested as a hypothesis, not assumed to be the favored
execution shape. L=2 proved that private B1 streams and graphs can overlap the
model-only interval, but complete exact sampling/control erased that gain. The
family therefore stops at L=2 under the campaign's `5%` keep rule.

The smallest clean gate is feasible without changing `DecodeSession`, model
code, V32, or a scheduler. `utils/verify_optimized_b1_lane_capture.py` builds an
independent eager B1 reference and a second B1 session, warms one full prepared
one-token call on a persistent private stream, then captures it with
`torch.cuda.graph(..., stream=lane)` and no shared pool token. L=1 must pass:

- raw FP32 logit allclose and top-k identity for prefill, eager setup, and graph
  replay;
- anchor/next sampled-token and final private-generator-state identity, plus an
  eager-versus-graph transcript/final-RNG comparison for every token counted in
  the fixed-shape complete-step timing observation;
- a zero sentinel at the target self-cache slot before first replay, followed by
  cache-position/shape-contract equality, self/cross-cache allclose, and
  disjoint reference/lane cache storage;
- recorded private stream, graph, graph-pool, session, cache, encoder, static
  buffer, generator, processor, and self/cross-cache ownership;
- warmed graph-only and graph-plus-sampling fixed-shape timing plus capture
  memory. This is a component scout, not a runtime throughput claim.

DCC job `49547823`, commit `08a59f5`, completed the normalized L=1 gate on
RTX 2080 Ti:

| Evidence | Result |
| --- | ---: |
| one-step, timed-transcript/RNG, resource, and capture gates | PASS |
| model-only graph replay | `537.426 tok/s` (`1.861 ms/step`) |
| complete graph-plus-sampling wall | `414.905 tok/s` (`2.410 ms/step`) |
| complete CUDA interval | `415.099 tok/s` |
| native context entry | `2.174s` |
| graph warmup / capture | `11.129 ms` / `11.220 ms` |
| graph capture allocated/reserved delta | `17,408` / `0` bytes |
| complete-interval verifier allocation | `1,433,664,000` bytes (`1.335 GiB`) |

The target self-cache position `84` was zeroed before replay, every captured
K/V view became nonzero, cache position/shape contracts matched, and the final
self/cross cache matched the eager reference. All 50 timed sampled tokens and
the final generator hash matched the eager fixed-shape loop. The normalized
observation counts only the 50 output-discarding replays inside the measured
complete interval; setup/parity/model-only and untimed transcript replays remain
separate in the outer graph ledger. The allocation includes the eager reference
state retained by the verifier and is not a lane-only capacity measurement.

Earlier job `49547755`, commit `718a996`, passed the same exactness/resource
gates, but retained 50 sampled GPU output tensors inside the measured loop. Its
`414.565 tok/s` and peak-memory fields are superseded as normalized evidence.
The fix moved transcript capture to an untimed replay ledger, discarded sample
outputs in the timed loop like the merged verifier, and separately checked the
timed generator's final state. The nearly unchanged complete TPS across the two
jobs is reassuring; the model-only microtiming varied materially and should not
be used alone to choose the execution shape.

The native extension cache was prebuilt (`10` entries before and after); the
Inductor cache remained empty and Triton/CUDA cache entries were unchanged.
`nsys`, `ncu`, and `dcgmi` were available inside the allocation. Coarse
whole-job telemetry is not a utilization claim because the 46-second job was
dominated by model load and validation.

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/b1-lane-capture-49547823-08a59f5/b1-lane-capture.json
/work/imt11/Mapperatorinator/runs/b1-lane-capture-49547823-08a59f5/cache-state.txt
/work/imt11/Mapperatorinator/runs/b1-lane-capture-49547823-08a59f5/nvidia-smi.csv
/work/imt11/Mapperatorinator/logs/b1-lane-capture-49547823.out
/work/imt11/Mapperatorinator/logs/b1-lane-capture-49547823.err
```

The report SHA-256 is
`b91383f063a48be95a1174197a8c7cc389c75615dfff9d369c6ad42447dd7cc8`.
Accept L=1 as exact verifier/denominator evidence, not a runtime win. Its
complete result is `5.6%` below merged B8, while model-only replay is above the
merged complete result. That justified exactly one L=2 concurrency
falsification with two private streams/graphs/pools/caches/encoders/processors/
generators, reciprocal launch orders, and a required `435.650 tok/s` (`+5%`)
complete-step result.

DCC job `49548733`, commit `aa662c90`, ran that L=2 gate on the same RTX 2080
Ti. Both reciprocal orders passed one-step parity, zero-sentinel cache rewrite,
exact 50-token-per-row transcripts and final private RNG hashes, resource
ownership, and graph capture. The Slurm job intentionally exited `1` because
the performance gate failed:

| Evidence | Result |
| --- | ---: |
| exactness / private-resource ownership / capture | PASS / PASS / PASS |
| reviewed L=1 complete denominator / required `+5%` bar | `414.905` / `435.650 tok/s` |
| same-job serial model-only / complete | `556.778` / `405.962 tok/s` |
| concurrent model-only, order `0,1` / `1,0` | `778.591` / `786.853 tok/s` |
| concurrent complete, order `0,1` / `1,0` | `397.903` / `398.291 tok/s` |
| worst gain versus reviewed L=1 / same-job serial | `-4.098%` / `-1.985%` |
| reciprocal complete-throughput spread | `0.097%` |
| verifier peak allocated / reserved | `1,981,775,360` / `2,183,135,232` bytes |

The model-only interval improved by `39.8-41.3%` against same-job serial, so
CUDA stream/graph overlap is real. The complete sampled step was nevertheless
slower in both orders. This is the decision metric, and the result is also
`37.36-37.75 tok/s` short of the fixed keep threshold. Peak memory is
verifier-wide (`1.846 GiB` allocated, `2.033 GiB` reserved), including retained
reference and lane state; it is not a two-lane production capacity estimate.

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/l2-lane-capture-49548733-aa662c9/l2-lane-capture.json
/work/imt11/Mapperatorinator/runs/l2-lane-capture-49548733-aa662c9/cache-state.txt
/work/imt11/Mapperatorinator/runs/l2-lane-capture-49548733-aa662c9/nvidia-smi.csv
/work/imt11/Mapperatorinator/logs/l2-lane-capture-49548733.out
/work/imt11/Mapperatorinator/logs/l2-lane-capture-49548733.err
```

The report SHA-256 is
`2533d7be39a065f62fd0fa5af6cf8fbe8173c01bbfc2eb1dc43bb4ccb296332e`.
Extension caches were unchanged before/after. Retain the L=1/L=2 verifier as
exact physics evidence, but do not build L=3/L=4 or lane-pool runtime wiring.
Revisit only after measured complete sampling/control changes, or a materially
different hardware/runtime stack, gives the L=2 complete path a projected
ceiling above `5%`.

Lane observations use the merged verifier's `row-N` request IDs and logical
workload-contract keys. The L=2 comparison replayed the same two-row workload
serially and concurrently; it did not compare different request counts merely
because both reports were normalized observations.

The retained verifier requires every lane to own a distinct persistent CUDA
stream, graph, default
graph-private allocator pool, session/cache/encoder state, static buffers,
generator, and logits processor while sharing only immutable model weights.
Never pass the same graph-pool token to concurrently replayed graphs. PyTorch
documents cuBLAS workspaces as allocated per handle/stream combination but does
not expose their pointers, so evidence records the unique warmed stream as the
workspace owner plus `CUBLAS_WORKSPACE_CONFIG`; it must not fabricate a pointer.
NVIDIA's multi-stream cuBLAS reproducibility caveat makes reciprocal L=2 launch
order and exact token/RNG checks mandatory.

Primary references: [PyTorch CUDA graph memory management](https://docs.pytorch.org/docs/main/notes/cuda.html),
[PyTorch cuBLAS workspaces](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.rst),
[PyTorch graph stream/pool API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html),
[NVIDIA cuBLAS multi-stream reproducibility](https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html),
and [NVIDIA graph-pool concurrency warning](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/numerical-errors.html).

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

## Batch Physics Result And Remaining Gate

The first one-token physics comparison is complete:

1. merged one-token decode reached `439.636` complete tok/s at B8 and retained
   exact row behavior, but rowwise sampling/control consumed `34.29%` of the
   complete step;
2. two independent B1 CUDA-graph lanes retained exact transcripts/private state
   and overlapped model work, but regressed complete throughput to
   `397.903-398.291 tok/s` and are rejected.

Do not build a production scheduler from either one-token component result. The
remaining merged-family gate is to profile the measured B8 rowwise
sampling/control gap, then prove a multi-step mixed-song loop with distinct
songs, request-order permutations, timing/main contexts, prefix buckets,
EOS/max-token stops, cache-slot release/reuse, and output assembly.

For each shape report:

- model-only replay and complete sampled-step aggregate tok/s;
- per-request token/RNG/cache/output equality against running alone;
- VRAM per lane/slot and maximum safe active count;
- graph capture/replay count, active-batch histogram, queue wait, and tail behavior;
- GPU timeline, clocks/power, memory traffic indicators, and useful idle gaps.

Only a measured target-sized complete-step ceiling can justify merged scheduler
wiring, request-major multi-vector linear work, broader fusion, or
operation-level/nano-batch overlap. The model-only L=2 lane signal alone is not
such a ceiling.

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

- Fixed-slot packed B8 is the selected execution shape after its exact `+13.42%`
  complete-step gain and the lane rejection. Run exactly one mixed eight-row,
  at-least-five-song 16-step gate before any longer loop or scheduler wiring;
  require exact private behavior, `>=5%`, and `>=500 tok/s`.
- Keep the independent-lane verifier as rejected physics evidence; do not build
  L=3/L=4 or lane runtime wiring.
- Feed accepted single-song components, including any exact speculative verifier win, back into the offline engine and measure combined scheduler-wall throughput.
- Stop for user input before reduced precision, output/RNG relaxation, or maintainer-facing changes outside the optimized package/adapter boundary.
