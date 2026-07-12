# CUDA Graph Multinomial RNG Probe

## Summary

Test whether CUDA graph replay can preserve exact `torch.multinomial(probs, 1)` behavior on the DCC RTX 2080 Ti stack. This is a prerequisite for any broader runtime idea that tries to graph logits/sampling/tail work while preserving fixed-seed token identity and final RNG state.

This was diagnostic only. It did not change Mapperatorinator inference throughput.

## Jobs

### Token Sequence Probe

- Job: `49231654`
- Run dir: `/work/imt11/Mapperatorinator/runs/cudagraph-rng-multinomial/49231654`
- Report: `/work/imt11/Mapperatorinator/runs/cudagraph-rng-multinomial/49231654/report.json`
- Torch: `2.10.0+cu128`

Result:

- Default CUDA generator graph replay matched eager for 32 multinomial samples.
- Explicit CUDA generator graph replay failed unless registered with `CUDAGraph.register_generator_state`.
- Explicit registered generator graph replay matched eager for 32 multinomial samples.

### RNG-State Probe

- Job: `49231667`
- Run dir: `/work/imt11/Mapperatorinator/runs/cudagraph-rng-multinomial-state/49231667`
- Report: `/work/imt11/Mapperatorinator/runs/cudagraph-rng-multinomial-state/49231667/report.json`
- Torch: `2.10.0+cu128`

Result:

| Case | Sequence match | Final CUDA RNG state match | Next eager sample match | Notes |
| --- | --- | --- | --- | --- |
| default generator, no reset after capture | PASS | PASS | PASS | graph replay matched 64 eager draws |
| default generator, reset after capture | PASS | PASS | PASS | graph replay matched 64 eager draws |
| explicit generator, not registered | FAIL | n/a | n/a | `RuntimeError: Attempt to increase offset for a CUDA generator not in capture mode.` |
| explicit generator, registered | PASS | PASS | PASS | graph replay matched 64 eager draws |

## Interpretation

One-token graph replay can include PyTorch CUDA `torch.multinomial(probs, 1)` without changing the sampled token stream or final CUDA RNG state on this stack. If future runtime work uses an explicit CUDA generator, it must register that generator with the `CUDAGraph` before capture.

This does not make fixed multi-token graph blocks automatically safe. A fixed `K`-token graph that samples after an EOS/stopping point would still over-advance RNG compared with HF eager generation unless it has exact rollback or true device-side early exit. The next diagnostic for multi-token graph work should be a forced-EOS block audit: no-EOS-in-block should match tokens/RNG, while forced EOS before `K` should demonstrate the oversampling problem unless a rollback design is present.

## Decision

Keep this as enabling evidence for one-token/tail graph experiments. Do not claim a speedup. Do not productionize multi-token sampling graphs until forced-EOS and final-RNG gates pass.
