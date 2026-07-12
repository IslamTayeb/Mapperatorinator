# Post-Fused Direct-Loop Tail Diagnostics

## Purpose

Re-measure exact logits-processor, sampling, append, and stopping tail cost after the fused RoPE/cache self-attention win. The accepted baseline is now job `49230082`: `28.243s` full-song main model time and `270.475 tok/s`, so the `5%` threshold is about `1.41s`.

This is verifier-only. It is not a throughput claim.

## Runs

Invalid first pass:

- DCC job: `49230274`
- Commit: `e63e022`
- Result: verifier PASS, but not used for current-stack conclusions.
- Reason: the launcher omitted `inference_stateful_monotonic_logits_processor=true`, so it measured the older expensive monotonic processor rather than the accepted stack.

Corrected current-stack pass:

- DCC job: `49230279`
- Commit: `e63e022`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Run dir: `/work/imt11/Mapperatorinator/runs/fused-tail-stateful-49230279-e63e022`
- JSON: `/work/imt11/Mapperatorinator/runs/fused-tail-stateful-49230279-e63e022/tail_diag.json`
- Summary: `/work/imt11/Mapperatorinator/runs/fused-tail-stateful-49230279-e63e022/tail_summary.json`

Flags:

```text
profile_salvalai_smoke15
sequence_index=9
max_new_tokens=256
candidate-active-prefix-decode
candidate-active-prefix-decode-bucket-size=64
candidate-cuda-graph-forward
candidate-cuda-graph-warmup=0
candidate-q1-bmm-cross-attention
candidate-native-q1-self-attention
candidate-native-q1-rope-cache-self-attention
candidate-decode-session
tail-diagnostics
inference_generation_compile=true
inference_stateful_monotonic_logits_processor=true
```

## Correctness

The corrected verifier passed:

- `pass=true`
- `token_match=true`
- `rng_match=true`
- `logits_pass=true`
- `logits_allclose=true`
- `logits_topk_match=true`
- generated sampled steps: `256`

## Projected Tail Cost

Projection uses `7,639` full-song main tokens.

| tail component | CUDA event us/token | projected CUDA seconds | CPU wall us/token | projected CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| logits extract | `3.974` | `0.030s` | `50.164` | `0.383s` |
| stateful monotonic processor | `37.285` | `0.285s` | `265.839` | `2.031s` |
| temperature warper | `3.958` | `0.030s` | `34.471` | `0.263s` |
| top-p warper | `65.946` | `0.504s` | `200.502` | `1.532s` |
| softmax | `6.661` | `0.051s` | `29.213` | `0.223s` |
| multinomial | `45.317` | `0.346s` | `167.006` | `1.276s` |
| EOS mask | `9.830` | `0.075s` | `68.762` | `0.525s` |
| append token | `4.672` | `0.036s` | `41.487` | `0.317s` |
| stopping criteria | `23.783` | `0.182s` | `167.219` | `1.277s` |
| finished check | `9.258` | `0.071s` | `48.998` | `0.374s` |

Excluding the verifier-only `_CaptureRawLogitsProcessor`, the approximate CUDA-event tail sum is about `1.61s`. That barely clears the new `5%` bar, but it is distributed across many exact semantics:

- stateful monotonic mask;
- top-p sorting/masking;
- exact `torch.multinomial` RNG behavior;
- EOS masking and stopping;
- token append.

## Interpretation

Tail work is now barely target-sized in aggregate because the model-forward path got faster. It is still not a clean standalone optimization:

- Fusing only logits processors would target about `0.82s` CUDA-event work, below the `5%` full-song bar.
- Fusing only top-p/softmax would target about `0.55s`.
- Replacing `torch.multinomial` risks changing RNG behavior and generated tokens.
- EOS/stopping/append work is small by CUDA-event timing, while CPU wall includes synchronization and host-control effects.

The active diagnostics and this verifier agree directionally: per-token control and tail work matter, but a safe implementation needs to be part of a broader DecodeSession/runtime island. It cannot be a casual fused-sampling patch unless exact multinomial RNG and EOS stopping are preserved.

## Decision

No optimization graduated.

Do not start standalone sampling/top-p fusion from this evidence. Revisit tail work only as part of a broader exact runtime path that can preserve:

- fixed-seed generated-token identity;
- final RNG state;
- EOS stopping and generated-token counts;
- output/map equivalence.

The best next implementation-class target remains a broader decoder/runtime island, not a narrow tail or single-kernel cleanup.
