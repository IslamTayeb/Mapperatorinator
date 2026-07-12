# Hybrid L2 Fixed-Shape Verifier Plan

## Scope

This is one bounded verifier-only follow-up to the rejected independent-lane
L2 result. It does not add a scheduler, server adapter, production flag,
changing-prefix loop, L3/L4 lanes, or runtime wiring.

The fixed workload is SALVALAI smoke15 `seq9`, FP32/SDPA, active-prefix bucket
`128`, two private row seeds (`12345`, `23456`), and `50` repeats in both
reciprocal graph/sampling orders. The source denominator is DCC job `49548733`
at commit `aa662c90`, report SHA
`2533d7be39a065f62fd0fa5af6cf8fbe8173c01bbfc2eb1dc43bb4ccb296332e`.

## Bottleneck Proof

The reviewed normalized L1 complete denominator is `414.904697 tok/s`, making
the strict `+5%` keep bar `435.649932 tok/s`. The rejected L2 scout preserved
tokens, final RNG, cache parity, capture, and private ownership, and its two
model-only orders reached `778.724` and `787.030 tok/s`. Complete throughput,
however, was only `397.903` and `398.291 tok/s`.

For the slower order, two tokens took `5.026346 ms` complete versus about
`2.568742 ms` of overlapped model work, leaving about `2.457604 ms` of control,
logits processing, and sampling. Clearing the L1 `+5%` bar requires saving
about `0.435504 ms`, or `17.72%` of that gap. Reaching `500 tok/s` requires a
step below `4.000 ms`, saving `1.026346 ms`, or `41.76%` of the gap.

The B8 identical-prompt processor control saved about `49.89%` of its analogous
complete-minus-model gap. That is not transferable evidence, but it gives this
single hybrid L2 test an optimistic ceiling near `526 tok/s`, high enough for
one cheapest fixed-shape falsification and not enough to authorize runtime
work.

## Hypothesis And Event Design

Two B1 model graphs retain private streams, graph pools, caches, encoder
outputs, static buffers, logits processors, and generators while sharing only
immutable model weights. Their fixed `[1, 4069]` logits feed a preallocated
`[2, 4069]` bridge on a dedicated control stream. One captured B2 base
processor pass executes the reviewed full-scan family:

1. `MonotonicTimeShiftLogitsProcessor`;
2. `TemperatureLogitsWarper`.

The processed rows are copied into private static sampling inputs. Each lane
then replays a private captured `TopPLogitsWarper -> softmax -> multinomial`
tail with its explicit CUDA generator registered on that graph. The event
chain is model replay, lane-ready, control wait, raw-logit copy, source release,
shared processor replay, private processed-row copy, processor-done, private
sampling replay, sample-done, and one interval-end control wait. The next model
replay cannot overwrite a source logit before its bridge copy completes.

The timed loop allocates no Python tensors, performs no `stack`/`cat`, retains
no sampled-output archive, and never calls a per-step global synchronize. CUDA
events and all tensor/graph storage are created before the interval. A separate
exact transcript run uses a preallocated device token archive and compares
both reciprocal orders against independent private B1 references.

## Gates And Stop Conditions

The report must self-validate all of the following:

- reviewed L2 commit, SHA, shape, seeds, and denominator;
- exact base/warper class families;
- private model/cache/encoder/graph/stream/generator and sampling-tail
  ownership, with only the explicit B2 bridge shared;
- bitwise processed rows and top-k versus private B1 stateful references;
- exact 50-token transcript and final RNG per row in both reciprocal
  graph/sampling orders;
- exact final RNG for the separate complete output-discard timing interval;
- worst-order complete wall throughput above `435.649932 tok/s`;
- a separately reported strict worst-order target above `500 tok/s`.

Stop immediately on exactness, capture, ownership, or event-contract failure.
At or below the `+5%` bar, reject the family. Above the keep bar but at or below
`500`, retain verifier evidence only and do not change prefix or build runtime.
Only a worst reciprocal order above `500` is eligible for one separately
reviewed changing-prefix exactness gate; even that is not production or
scheduler authorization.

## DCC Operations

`scripts/dcc/verify_hybrid_l2.sbatch` requires an explicit isolated DCC
worktree, pushed commit, and exact branch. It refuses the shared checkout and
pins the reviewed L2 report by SHA. The wrapper records cache state, CUDA/tool
versions, telemetry, report paths, and both performance bars. On 2026-07-10,
live `sinfo` still exposed `gpu-common` RTX 2080 resources; account and GRES in
the wrapper match the current campaign allocation. The first submitted job is
recorded below and did not reach measurement.

## Setup-Only Graph-Capture Failure

DCC job `49552299` ran commit `640b824` on
`dcc-core-gpu-ferc-s-h36-6` with one RTX 2080 Ti and exited `1` after
`00:01:05`. The run root is
`/work/imt11/Mapperatorinator/runs/hybrid-l2-49552299-640b824`; stdout/stderr
are `/work/imt11/Mapperatorinator/logs/hybrid-l2-49552299.{out,err}`.
The reviewed source contract passed, the exact model loaded, and the job then
failed while capturing the shared B2 processor graph. It produced no hybrid
report and no performance or exactness evidence.

The failure was a device-placement bug in the existing full-scan monotonic
processor: advanced indexing created `torch.arange(batch_size)` on CPU while
`input_ids` was on CUDA. CUDA graph capture rejected that operation and then
reported an invalidated capture. Commit `79c644f` makes only that batch index
device-local. CPU tests prove the full-scan output remains bitwise identical
to the pre-fix calculation, every `arange` in the full-scan path receives the
input device, and shared B2 processing remains bitwise-equal to two private
stateful B1 processors.

This is graph-safety setup, not an inference optimization or throughput win.
The job had warm native-extension cache (`10` entries), empty Inductor cache,
and recorded Nsight Systems, Nsight Compute, DCGM, telemetry, and cache-state
availability. That setup-only job stopped there; its valid successor is recorded
separately below.

### Second setup-only capture failure

DCC job `49552651` ran commit `e13f5ee` on the same
`dcc-core-gpu-ferc-s-h36-6` RTX 2080 Ti node and exited `1` after `00:00:58`.
The run root is
`/work/imt11/Mapperatorinator/runs/hybrid-l2-49552651-e13f5ee`; stdout/stderr
are `/work/imt11/Mapperatorinator/logs/hybrid-l2-49552651.{out,err}`. The
reviewed workload contract and exact model load again passed, but the shared
processor capture then rejected boolean advanced row indexing in
`scores[apply_mask] = ...`. No hybrid report was produced, so this job contains
no performance or exactness evidence.

Commit `3c7a16f` replaces that row-selection assignment with equivalent
full-shape tensor algebra:
`scores.masked_fill_(batch_mask & apply_mask.unsqueeze(1), -torch.inf)`.
Randomized CPU tests cover B=1/2/5, no-timeshift, no-SOS, SOS-reset, and mixed
prefixes and preserve bitwise output against the pre-rewrite calculation.
Separate randomized B2 tests preserve bitwise equality against two private
stateful B1 processors.

The remaining full-scan operations are fixed-shape device-local comparisons,
`torch.where`, reductions, three explicitly device-local `arange` calls,
device-local integer advanced indexing, a basic contiguous slice assignment
into a boolean mask, and in-place full-shape `masked_fill_`. The method has no
host transfer, `.item()`, CPU-created index tensor, or boolean advanced row
selection left. This makes the corrected assignment the last obvious capture
hazard in this small method. This setup-only job did not prove the complete
path; the subsequent valid job did. The correction itself remains graph-safety
setup, not a measured inference win.

## Valid Fixed-Shape Result

DCC job `49552768` ran commit `682abdc` on one RTX 2080 Ti and completed the
fixed-shape gate. Both reciprocal orders passed report self-validation, source
and workload hashes, resource ownership, CUDA graph capture, bitwise processed
scores, top-k identity, exact 50-token transcripts, and exact final private RNG
state for both the transcript and complete output-discard timing intervals.

| Evidence | Result |
| --- | ---: |
| exactness / hashes / ownership / capture | PASS / PASS / PASS / PASS |
| reviewed normalized L1 denominator | `414.904697 tok/s` |
| rejected independent L2 worst order | `397.903378 tok/s` |
| hybrid complete wall, order `0,1` | `670.026 tok/s` |
| hybrid complete wall, order `1,0` | `723.633 tok/s` |
| worst-order gain versus reviewed L1 | `+61.49%` |
| worst-order gain versus rejected independent L2 | `+68.39%` |
| verifier peak allocated / reserved | approximately `1.982 / 2.147 GB` |

The valid artifacts are:

```text
/work/imt11/Mapperatorinator/runs/hybrid-l2-49552768-682abdc/hybrid-l2.json
/work/imt11/Mapperatorinator/runs/hybrid-l2-49552768-682abdc/cache-state.txt
/work/imt11/Mapperatorinator/runs/hybrid-l2-49552768-682abdc/nvidia-smi.csv
/work/imt11/Mapperatorinator/logs/hybrid-l2-49552768.out
/work/imt11/Mapperatorinator/logs/hybrid-l2-49552768.err
```

The report SHA-256 is
`d7f47e5c174a8f3504401ec0d428b0070f4724e95971efc150c3d13007f343ef`.
Jobs `49552299` and `49552651` remain the separate setup-only capture failures
documented above. They produced no report and contribute no performance or
exactness evidence to this result.

This is a complete output-discard fixed-shape microbenchmark for SALVALAI
smoke15 `seq9`: two B1 graph lanes, two private row seeds, `50` repeats, one
identical base prompt and prefix shape, a shared B2 full-scan processor bridge,
and private sampling graph tails. It is not evidence for changing prefixes,
different songs, queue or scheduler-wall throughput, EOS/stopping, staggered
arrival, slot reuse, a runtime, a server, or production integration.

The result authorizes only one separately designed and reviewed changing-prefix
verifier. It does not authorize that verifier's implementation as part of this
documentation change. The two one-line core graph-safety changes also remain
unmergeable until legacy exact-output and no-regression validation covers the
V32 B1 stateful and B2 full-scan processor paths plus representative legacy
output, imports, metadata, and performance:

- commit `79c644f`: create the batch index on the input device;
- commit `3c7a16f`: replace boolean row advanced assignment with full-shape
  `masked_fill_`.

## Changing-Prefix Follow-Up Result

The authorized bounded follow-up is complete. DCC job `49555060`, commit
`2b6e690`, completed in `00:05:52` with exit `0:0` on one RTX 2080 Ti. Phase A
passed at `619.208 tok/s` worst reciprocal order. Phase B passed 16 closed-loop
steps and five reset-separated trials per order at `621.893` worst and `628.868
tok/s` best; its five-trial ranges were `620.430-622.791` and
`626.726-630.113 tok/s`.

Every token/RNG/static-input/cache/padding/processor/reset/reciprocal gate
passed. Raw logits and active/current self-cache comparisons had `max_abs=0`
and matching hashes; cross-cache hashes matched and remained unchanged. Timed
allocation delta was zero. The worst Phase B order improved `49.26%` over its
same-job private-B1 control and was `7.18%` below the fixed `670.026 tok/s`
result. Peak allocated/reserved VRAM was approximately `2.512/3.326 GB`.

The extension cache was warm and unchanged (`10` entries, `112` files); native
context entry was `4.299s`. Setup and exact-reference costs are excluded from
decode TPS. Report:

```text
/work/imt11/Mapperatorinator/runs/hybrid-changing-prefix-49555060-2b6e690/hybrid-changing-prefix.json
```

SHA-256:
`365c9ffa99878ee68d45d8152634398b254ab731f2d75bb01d3d86dfdc8f6a6b`.
Telemetry is beside the report; logs are
`/work/imt11/Mapperatorinator/logs/hybrid-prefix-49555060.{out,err}`.

This result covers only one identical-input, two-seed SALVALAI smoke15 `seq9`,
bucket-128 shape (`305 / 5,905` production decode tokens), and no EOS occurred.
Applying it to all buckets would give an invalid optimistic extrapolation of
`542.663` decode-only / `443.896` full-setup tok/s and require `61.65%` setup
removal for `500`. It authorizes production-weighted bucket/queue physics only,
not setup optimization or scheduler/runtime wiring. The exact-five B2-only
ceiling analysis is pending and should decide the next experiment.
