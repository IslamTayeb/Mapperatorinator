# Current-Stack Runtime Gap Smoke Diagnostic

## Purpose

Re-check the current fastest exact opt-in stack after the rejected tail-graph
runtime experiment. The specific question was whether the remaining
production-vs-replay gap points to host/runtime cleanup or to real queued
decoder CUDA work.

This is diagnostic-only. It enabled CUDA-event and active-prefix loop ledgers,
so the timing is perturbed and must not be used as a throughput claim.

## Run

- Job: `49250056`
- Commit: `9374133`
- Branch: `main`
- Node: `dcc-core-ferc-s-z25-20`
- GPU: RTX 2080 Ti, UUID `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Driver: `595.71.05`
- Torch/CUDA: `2.10.0+cu128`, CUDA `12.8`
- Transformers: `4.57.3`
- Run dir:
  `/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133`
- Profile:
  `/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/output/beatmap7423e0d696194cd0a49e6924ee574fdd.osu.profile.json`
- Main summary:
  `/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/main_active_summary.json`
- Timing summary:
  `/work/imt11/Mapperatorinator/runs/current-gap-smoke-49250056-9374133/timing_active_summary.json`

Two earlier submissions did not produce inference evidence:

- `49250053` failed before inference because an inline Python version-print
  heredoc lost quotes.
- `49250054` failed before generation because the Slurm PATH did not include
  the env `ffprobe`.

The successful job prepended
`/hpc/group/romerolab/imt11/envs/mapperatorinator/bin` to `PATH`.

## Flags

The inference flags matched the current accepted opt-in stack:

```text
profile_salvalai_smoke15
seed=12345
precision=fp32
attn_implementation=sdpa
use_server=false
parallel=false
profile_record_token_ids=true
inference_generation_compile=true
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=64
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
inference_q1_bmm_cross_attention=true
inference_decode_session_runtime=true
inference_decode_session_cuda_graph=true
inference_native_decode_kernels=true
inference_native_q1_self_attention=true
inference_native_q1_rope_cache_self_attention=true
```

Diagnostic-only flags:

```text
profile_model_generate_cuda_ledger=true
profile_active_prefix_decode_diagnostics=true
```

## Equivalence And Overhead

Comparison baseline:
`/work/imt11/Mapperatorinator/runs/fused-rope-cache-smoke15-49230035-d7b8684/candidate.profile.json`

- Same-calculation metadata: PASS
- Main generated-token identity: PASS, `1,084 / 1,084`
- Output artifact equivalence: not checked because the older smoke baseline
  lacks `result_file_sha256` and `result_file_size_bytes`

The diagnostic path regressed model time, as expected from extra CUDA events:

| Metric | Baseline | Diagnostic |
| --- | ---: | ---: |
| Main model time | `3.774s` | `4.094s` |
| Main tok/s | `287.257` | `264.763` |
| Total stage wall | `13.329s` | `13.683s` |

This is not a candidate speed result.

## Model-Generate CUDA Ledger

The host gap around production `model.generate()` was effectively zero:

| Label | Records | Tokens | Model time | CUDA-event time | Host gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `main_generation` | `10` | `1,084` | `4.094230s` | `4.093284s` | `0.000946s` |
| `timing_context` | `10` | `164` | `3.165255s` | `3.164337s` | `0.000918s` |

This agrees with the full-song ledger jobs: the current stack is not waiting on
Python outside `model.generate()`.

## Main Active-Prefix CUDA Events

Aggregated `main_generation` diagnostic CUDA-event totals:

| Range | CUDA event time | Calls | Share of diagnostic model time |
| --- | ---: | ---: | ---: |
| `loop_total` | `3.773018s` | `10` | `92.2%` |
| `decode_forward.cuda_graph` | `2.516996s` | `1,074` | `61.5%` |
| `graph.replay` | `2.211618s` | `1,064` | `54.0%` |
| `prepare_inputs` | `0.424733s` | `1,084` | `10.4%` |
| `stopping_criteria` | `0.250799s` | `1,084` | `6.1%` |
| `prefill_forward` | `0.227785s` | `10` | `5.6%` |
| `graph.capture` | `0.129418s` | `10` | `3.2%` |
| `logits_processor` | `0.104347s` | `1,084` | `2.5%` |
| `graph.input_copy` | `0.062701s` | `1,064` | `1.5%` |
| `sampling.multinomial` | `0.045403s` | `1,084` | `1.1%` |

Duplicate graph capture stayed small:

| Metric | Value |
| --- | ---: |
| normalized graph shapes | `10` |
| graph captures | `24` |
| duplicate captures | `14` |
| duplicate capture time | `0.052020s` |
| duplicate capture share | `1.271%` |

## Interpretation

The large current-stack bucket is still graph-replayed decoder forward work.
The smoke graph-replay CUDA event time, naively scaled by
`7,639 / 1,084`, is about `15.6s`, which lines up with the existing
decoder-stack/layer replay diagnostics. That makes broad decoder compute and a
larger decoder runtime island the only currently plausible major path.

The visible `prepare_inputs` and `stopping_criteria` ranges are not enough to
restart old quick paths by themselves:

- they are diagnostic CUDA-event ranges, nested/non-exclusive, and the run is
  slowed by diagnostic finalization;
- fast prepare and tail/control graph attempts already failed production
  promotion;
- graph input copying, logits processors, sampling, and duplicate graph capture
  are all below the standalone keep threshold in this current-stack smoke.

## Decision

No optimization graduated and no runtime code should be changed from this pass.
Keep the accepted full-song baseline at `270.475 tok/s` from job `49230082`.

Next implementation work should be gated by a broad decoder-layer/runtime
island or a more exclusive gap auditor that predicts at least a `>5%`
full-song synchronized model-time saving. Do not spend more time on standalone
tail graphing, graph-cache cleanup, graph input copying, sampling fusion, or
prepare/stopping cleanup without new evidence that those buckets are exclusive
and target-sized.
