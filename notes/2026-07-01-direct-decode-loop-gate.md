# Direct Decode Loop Gate

## Hypothesis

Custom runtime work needs a faster correctness gate than a full 15s smoke run. A multi-token gate should catch direct-loop mistakes in HF generation semantics, cache handling, active-prefix decode, logits processors, EOS stopping, and RNG consumption before spending Slurm time on end-to-end smoke profiles.

This is a testing-suite improvement, not an inference optimization.

## Implementation

Added `utils/verify_direct_decode_loop.py`, which runs the same probe inputs through:

- normal HF `model.generate()`;
- the candidate loop through HF `custom_generate`.

The gate resets CPU/CUDA RNG state before each path, lets HF construct the final processors/stopping criteria for both paths, captures raw logits before in-place logits processors, and checks:

- generated-token identity;
- final RNG-state identity;
- raw-logit `allclose`/top-k agreement for every sampled step;
- prompt prefix identity and stop reason.

The candidate loop is deliberately narrow: batch size 1, non-server, no CFG, no beams, no streamer, no synced GPUs, no auxiliary generation outputs, and no prefill chunking.

## Evidence

DCC job: `49161597`
Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti
Driver: `595.71.05`
Torch: `2.10.0+cu128`
Transformers: `4.57.3`
Commit: dirty test patch on `cb874ee`
Run dir: `/work/imt11/Mapperatorinator/runs/direct-loop-gate-49161597-cb874ee`

All variants generated the same eight `seq9` tokens:

```text
[12, 1648, 2242, 2717, 3915, 4012, 4053, 32]
```

| gate | token match | RNG match | raw-logit steps | max_abs | wall |
| --- | --- | --- | ---: | ---: | ---: |
| compile false, plain loop | PASS | PASS | 8 | `0.0` | `8.110s` |
| compile true, plain loop | PASS | PASS | 8 | `0.0` | `40.794s` |
| compile false, buffered active512 | PASS | PASS | 8 | `0.0` | `7.713s` |
| compile true, buffered active512 | PASS | PASS | 8 | `0.0` | `49.090s` |

## Interpretation

The direct loop can preserve HF generation semantics for a short sampled run, including RNG state, when routed through `custom_generate`. That makes the gate useful for future direct runtime work before 15s smoke promotion.

The compile-enabled variants pay noticeable setup/compile time, so this gate should not be treated as a throughput benchmark. It is most useful for custom/direct decode loop correctness and for catching mistakes before expensive smoke/full-song jobs.

## Decision

Keep the utility and documentation. It changes no default inference behavior and provides a stricter pre-smoke gate for future direct decode/runtime experiments.
