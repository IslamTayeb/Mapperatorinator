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

## Decision

Keep the utility as verifier/profiling infrastructure. No optimization
graduated and the accepted baseline remains job `49230082` at `270.475 tok/s`.
