# Decode Runtime Gap Auditor

## Purpose

Add a lightweight JSON auditor for the 500 tok/s campaign. The goal is to make
the current bottleneck/theoretical-ceiling check repeatable before starting
another runtime or kernel branch.

The utility is diagnostic-only:

```text
utils/summarize_decode_runtime_gap.py
```

It combines:

- a `profile_inference` JSON;
- optional `utils/summarize_active_prefix_diagnostics.py` JSON;
- optional full-forward, decoder-stack, and decoder-layer graph-replay island
  reports.

It does not run inference, import model code, or claim throughput. It only
aggregates existing evidence.

## Validation

Local validation used real DCC artifacts copied from:

```text
/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/profile.json
/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/main_active_summary.json
/work/imt11/Mapperatorinator/runs/post270-gap-diag2-49232331-840dbc4/full_forward_island_seq9.json
/work/imt11/Mapperatorinator/runs/post270-gap-diag2-49232331-840dbc4/decoder_layer_island_seq9.json
/work/imt11/Mapperatorinator/runs/decoder-stack-island-20260703-140615-75fc8da/decoder_stack_island.json
```

Command:

```bash
python3 utils/summarize_decode_runtime_gap.py \
  /tmp/mapper-runtime-gap/profile.json \
  --active-summary /tmp/mapper-runtime-gap/main_active_summary.json \
  --full-forward-report /tmp/mapper-runtime-gap/full_forward_island_seq9.json \
  --decoder-stack-report /tmp/mapper-runtime-gap/decoder_stack_island.json \
  --decoder-layer-report /tmp/mapper-runtime-gap/decoder_layer_island_seq9.json \
  --json-output /tmp/mapper-runtime-gap/runtime_gap_summary.json
```

Local checks:

```text
python3 -m py_compile utils/summarize_decode_runtime_gap.py
git diff --check
```

## Result

The auditor reproduced the current bottleneck ordering:

| Item | Projected full-song seconds |
| --- | ---: |
| `graph.replay` | `15.585s` |
| `prepare_inputs` | `2.993s` |
| `stopping_criteria` | `1.767s` |
| `prefill_forward` | `1.605s` |
| `logits_processor` | `0.735s` |
| `graph.input_copy` | `0.442s` |
| `sampling.multinomial` | `0.320s` |

Aggregate/composite ranges such as `loop_total` and
`decode_forward.cuda_graph` are labeled so they are not mistaken for standalone
optimization targets.

Graph-replay island projections from the supplied reports:

| Island | Projected full-song seconds |
| --- | ---: |
| full forward | `14.095s` |
| decoder stack hidden | `13.807s` |
| decoder stack plus projection | `14.096s` |
| decoder layers | `13.018s` |

The ledger again showed negligible host gap in the diagnostic profile:

```text
cuda=4.093284s
host_gap=0.000946s
```

## Interpretation

This utility makes the bottleneck discipline explicit:

- do not prioritize Python outside `model.generate`;
- do not restart standalone graph-input-copy, logits-processor, or sampling
  cleanup from the current evidence;
- treat `prepare_inputs`, `stopping_criteria`, and `prefill_forward` as
  diagnostic/control ranges that need a low-perturb exclusive proof before
  coding;
- keep the next implementation-class path focused on broad decoder-layer or
  decoder-runtime work unless a fresh auditor report changes the ordering.

## Avoidability Ledger Update

The auditor now emits per-row decision fields instead of requiring manual
interpretation:

- `target_class`, distinguishing aggregate/composite ranges,
  graph-replayed decoder compute, control/setup, and unknown events;
- `contains_required_model_math` and `standalone_target_candidate`;
- idealized remaining model time, target-TPS reachability, and missing seconds
  to target after removing a bucket;
- 5%/10% model-time and throughput bars;
- `exclusive_proof_required` for control/setup and idealized replay rows;
- `candidate_ledger`, which combines active-loop events and graph-replay island
  ceilings into one sorted table with caveats.

Validation on the same DCC artifacts produced the intended ordering. The rows
that can close the `500 tok/s` gap are aggregate ranges, graph-replayed decoder
compute, or idealized graph-replay island boundaries:

| Source | Name | Idealized saving | Target class | Reaches target | Standalone target | Needs exclusive proof |
| --- | --- | ---: | --- | --- | --- | --- |
| active event | `loop_total` | `26.589s` | aggregate/composite | yes | no | yes |
| active event | `decode_forward.cuda_graph` | `17.737s` | aggregate/composite | yes | no | yes |
| active event | `graph.replay` | `15.585s` | graph-replayed decoder compute | yes | no | yes |
| island replay | decoder layers | `15.225s` | idealized graph replay boundary | yes | no | yes |
| island replay | decoder stack hidden | `14.436s` | idealized graph replay boundary | yes | no | yes |
| island replay | full forward | `14.148s` | idealized graph replay boundary | yes | no | yes |
| active event | `prepare_inputs` | `2.993s` | control/setup | no | no | yes |

That is the useful constraint: standalone control/setup cleanup can clear a
small keep bar but cannot reach the goal by itself, while the only 500-class
rows include required decoder math or idealized replay boundaries. Future
runtime/kernel work should therefore start by reducing broad decoder compute or
by proving an exclusive production-vs-replay gap, not by retrying rejected
prepare, sampling, graph-copy, or tail-only paths.

Checks:

```text
python3 -m py_compile utils/summarize_decode_runtime_gap.py
python3 utils/summarize_decode_runtime_gap.py /tmp/mapper-runtime-gap/profile.json \
  --active-summary /tmp/mapper-runtime-gap/main_active_summary.json \
  --full-forward-report /tmp/mapper-runtime-gap/full_forward_island_seq9.json \
  --decoder-stack-report /tmp/mapper-runtime-gap/decoder_stack_island.json \
  --decoder-layer-report /tmp/mapper-runtime-gap/decoder_layer_island_seq9.json \
  --json-output /tmp/mapper-runtime-gap/runtime_gap_summary.updated.json
```

## Decision

Keep the utility as verifier/profiling infrastructure. No optimization
graduated and the accepted baseline remains job `49230082` at `270.475 tok/s`.
