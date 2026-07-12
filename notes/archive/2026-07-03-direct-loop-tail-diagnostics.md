# Direct-Loop Tail Diagnostics

## Purpose

Measure the logits-processor, sampling, append, and stopping tail after the accepted native q1 self-attention path, without changing generation semantics. This is diagnostic only; it is not an inference throughput claim.

Current accepted full-song single-song baseline remains DCC job `49225493`: `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu` output.

## Harness Change

Commit `4f1420f` added default-off `--tail-diagnostics` to `utils/verify_direct_decode_loop.py`. The verifier still compares the candidate direct loop against HF `generate()` for:

- generated-token identity;
- raw-logit allclose/top-k equality;
- final CPU/CUDA RNG-state identity;
- stop reason and output shape.

The new diagnostics record CPU wall spans and CUDA events around the candidate loop's real tail operations: logits extraction, each logits processor, softmax/multinomial, EOS masking, token append, stopping criteria, and finished check. This keeps the measurement inline with the real `custom_generate` processor list, including HF-injected `TopPLogitsWarper`, and avoids extra `torch.multinomial()` calls before equivalence checks.

The shared `_build_generation_logits_processors()` helper now passes `inference_stateful_monotonic_logits_processor` through to `build_logits_processor_list()` so direct-loop probes match the accepted active-prefix stack when that flag is enabled.

## Run

- DCC job: `49229277`
- Node/GPU: RTX 2080 Ti, see Slurm stdout at `/work/imt11/Mapperatorinator/logs/mm-taildiag3-49229277.out`
- Commit: `4f1420f`
- Report: `/work/imt11/Mapperatorinator/runs/direct-tail-diagnostics-20260703-070925-4f1420f/direct_tail_diagnostics.json`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `max_new_tokens=256`

Candidate flags:

```text
--candidate-decode-session
--candidate-active-prefix-decode
--candidate-active-prefix-decode-bucket-size 64
--candidate-cuda-graph-forward
--candidate-cuda-graph-warmup 0
--candidate-q1-bmm-cross-attention
--candidate-native-q1-self-attention
--tail-diagnostics
inference_generation_compile=true
inference_stateful_monotonic_logits_processor=true
attn_implementation=sdpa
precision=fp32
use_server=false
parallel=false
cfg_scale=1.0
num_beams=1
seed=12345
```

## Correctness

| check | result |
| --- | --- |
| verifier pass | PASS |
| generated-token identity | PASS |
| raw-logit/top-k gate | PASS |
| final RNG-state identity | PASS |
| stop reason | `max_new_tokens` for reference and candidate |
| sampled steps | `256` |
| vocab size | `4,069` |
| input length range | `84..339` |

Processor list observed by the candidate verifier:

```text
0:_CaptureRawLogitsProcessor
1:MonotonicTimeShiftLogitsProcessor
2:TemperatureLogitsWarper
3:TopPLogitsWarper
```

Processor `0` is the verifier's raw-logit capture hook, not production inference, and should be excluded from production-tail projections.

## Tail Timing

Top per-token CUDA-event timings:

| bucket | CPU wall us/token | CUDA event us/token |
| --- | ---: | ---: |
| logits_processor.total | `628.436` | `116.750` |
| TopPLogitsWarper | `196.826` | `64.106` |
| sampling.multinomial | `163.386` | `44.344` |
| MonotonicTimeShiftLogitsProcessor | `259.883` | `36.538` |
| stopping_criteria | `162.991` | `23.116` |
| eos_mask | `69.327` | `9.625` |
| finished_check | `47.176` | `8.476` |
| sampling.softmax | `28.506` | `6.511` |
| append_token.cat | `41.875` | `4.628` |
| logits_extract | `50.089` | `3.952` |
| TemperatureLogitsWarper | `34.229` | `3.946` |

Excluding the verifier capture hook and avoiding the non-exclusive `logits_processor.total` bucket, the summed production-like tail CUDA-event time was `205.242us/token`. Projected over `7,639` full-song main tokens, that is `1.568s`, or just under the `~1.61s` 5% full-song threshold for the accepted `32.217s` baseline.

The equivalent CPU wall sum was `1,054.288us/token`, projected to `8.054s`, but this is not exclusive model time. It includes Python timing overhead, launches, asynchronous control, and synchronization effects around dependent GPU operations. Treat it as a reason to keep the tail in mind for a broader DecodeSession runtime, not as proof that a standalone Python cleanup can save `8s`.

## Decision

Do not start standalone fused sampling/top-p work as the next 500 tok/s path. The exact device-time ceiling is too small for the risk and complexity:

- exact `torch.multinomial` RNG behavior must be preserved;
- top-p threshold behavior depends on sort/cumsum/mask order and `-inf` placement;
- the production-like CUDA-event projection is only `~1.568s`, below the normal `>5%` full-song threshold;
- prior broad profiling still shows decoder layer/linear/MLP/cache/layout work as the larger target.

Tail work may still be useful later if it is part of a broader DecodeSession-owned runtime that removes host synchronization/control around multiple tail operations without changing RNG or generated tokens. It should not displace broader decoder-layer or native/CUTLASS/cuBLASLt island work.

## Verification

- Local syntax check passed:

```bash
python -m py_compile utils/verify_direct_decode_loop.py utils/verify_one_token_decode.py
```

- DCC verifier job `49229277` passed as above.
- DCC lightweight pytest was not run because `/hpc/group/romerolab/imt11/envs/mapperatorinator` does not currently include `pytest`.
