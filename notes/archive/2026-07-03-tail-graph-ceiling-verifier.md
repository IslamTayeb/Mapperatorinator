# Tail Graph Ceiling Verifier

## Summary

Keep the verifier-only utility from branch `experiment/tail-graph-ceiling`.
It does not claim a production inference speedup, but it found a target-sized
runtime ceiling for graphing the exact one-token tail around logits processors,
sampling, EOS masking, append, and stopping.

This result should steer the next work toward a DecodeSession-owned tail/runtime
island with stable buffers and exact RNG semantics, not standalone sampling
fusion or fixed-K graphing.

## Evidence

- Local branch: `experiment/tail-graph-ceiling`
- Commits: `00f7862`, `d22fcf4`
- DCC job: `49236164`
- DCC report path: `/work/imt11/Mapperatorinator/runs/tail-graph-ceiling-49236164-d22fcf4/tail_graph_ceiling.json`
- Node/GPU from observed Slurm output: `dcc-chsi-gpu-ferc-s-i11-1`, RTX 2080 Ti, GPU UUID `GPU-5b9229da-8433-e112-04c8-086f173136fb`
- Driver/CUDA stack from observed Slurm output: driver `595.71.05`, torch `2.10.0+cu128`, transformers `4.57.3`, CUDA `12.8`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `max_new_tokens=256`, seed `12345`, fp32, SDPA, accepted opt-in stack with DecodeSession, active-prefix bucket64 CUDA graph, stateful monotonic, q1 BMM cross-attention, native q1 self-attention, and fused RoPE/cache self-attention

The report could not be re-read from resembool/Azure during migration:
`/work/imt11/...` was not mounted locally and `ssh dcc` failed with
`Permission denied`. The numbers below are from the prior observed DCC stdout
and pause summary in this thread.

## Result

The verifier captured one generated tail step at `generated_so_far=64`,
`cur_len=148`.

Correctness passed:

- generated next token matched: `2143`
- output IDs matched
- unfinished/finished state matched
- final RNG state matched
- subsequent eager CUDA samples matched
- verifier-only `_CaptureRawLogitsProcessor` was excluded from timing

Measured ceiling:

| Path | Time |
| --- | ---: |
| Eager exact tail | `1.0499565 ms/token` |
| CUDA graph replay tail | `0.1872870 ms/token` |
| Speedup | `5.606x` |
| Saved | `862.6696 us/token` |
| Projected SALVALAI main-token ceiling | `6.5899s` over `7,639` tokens |

Processors in the captured production-like tail:

- `MonotonicTimeShiftLogitsProcessor`
- `TemperatureLogitsWarper`
- `TopPLogitsWarper`

Stopping criteria:

- `MaxLengthCriteria`
- `EosTokenCriteria`

## Interpretation

This explains why earlier subrange tail diagnostics understated the runtime
target. The individual CUDA-event pieces projected only about `1.6s`, but the
whole eager tail contains launch/control gaps between exact tail operations. A
single CUDA graph replay removes much of that gap in the verifier.

This is still not production-ready:

- the capture is a single fixed `cur_len` shape;
- production generation changes input length every token;
- EOS stopping and final RNG state must stay exact;
- `torch.multinomial` RNG behavior must remain identical across many generated
steps, not just one replay;
- graphing a fixed-K block remains non-equivalent when EOS occurs early unless
there is exact rollback or true device-side early exit.

## Decision

Keep the verifier-only utility and merge it after docs. Do not claim any TPS
win from it.

Next exact path:

1. Build a multi-step verifier for a stable-shape DecodeSession tail graph that
   uses preallocated buffers instead of variable-length `input_ids`.
2. Prove generated-token identity, raw-logit/top-k equality, final RNG-state
   identity, EOS/stopping behavior, and output behavior across multiple sampled
   steps.
3. Only then wire a default-off production candidate through
   `inference.py`/`server.py:model_generate()`.
4. Promote with the normal 15s smoke, full-song SALVALAI, output hash, and
   no-regression gates.
