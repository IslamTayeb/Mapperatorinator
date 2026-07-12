# q_len=1 BMM Cross-Attention

## Hypothesis

After active-prefix bucket64 plus CUDA graph replay reached `155.578 tok/s`, the remaining CUDA profile showed real decoder compute, especially attention, not Python sampling. The no-graph component trace showed many unmasked decoder cross-attention calls with Q shape `[1, 12, 1, 64]` and K/V shape `[1, 12, 1024, 64]`. For this single-query, unmasked cross-attention case, PyTorch SDPA's general fused attention path looked overbuilt for SM75.

The candidate replaces only q_len=1 decoder cross-attention with explicit fp32 `bmm -> softmax -> bmm` when all of these are true:

- batch size is `1`
- eval mode
- dtype is `torch.float32`
- query length is `1`
- the module is decoder cross-attention
- no cross-attention mask is present
- `inference_q1_bmm_cross_attention=true`

Self-attention, masked attention, non-fp32 paths, server/batch paths, and normal defaults stay on the existing attention path.

## Evidence

Microbench job `49212482` on RTX 2080 Ti, run dir `/work/imt11/Mapperatorinator/runs/attn-bmm-sm75-49212482`, compared SDPA with manual BMM for `H=12`, `D=64`, fp32:

| case | SDPA | BMM | speedup | max abs |
| --- | ---: | ---: | ---: | ---: |
| `L=1024`, no mask | `0.14856ms` | `0.05477ms` | `2.71x` | `1.64e-7` |
| `L=1024`, half masked | `0.15549ms` | `0.06804ms` | `2.29x` | not promoted |
| `L=2560`, no mask | `0.36425ms` | `0.05653ms` | `6.44x` | small |
| `L=512`, no mask | `0.06886ms` | `0.05520ms` | `1.25x` | small |

Small lengths such as `64`, `96`, and `128` favored SDPA, so the implementation did not generalize this to active-prefix self-attention.

## Correctness Gates

One-token logits gate, job `49212884`, run dir `/work/imt11/Mapperatorinator/runs/q1bmm-logit-gate-49212884-3af8d69-len128`:

- Command used `--candidate-active-prefix-decode --candidate-active-prefix-decode-length 128 --candidate-q1-bmm-cross-attention`.
- PASS: raw logits allclose, top-k match, `max_abs=4.196e-05`.

Direct-loop gate, job `49213018`, run dir `/work/imt11/Mapperatorinator/runs/q1bmm-direct-gate-49213018-3af8d69`:

- Command used active-prefix bucket64, CUDA graph replay, warmup0, and q1 BMM over `64` sampled decode steps.
- PASS: generated-token identity, raw-logit allclose, top-k equality, and final RNG-state identity.

15s smoke, job `49213078`, run dir `/work/imt11/Mapperatorinator/runs/q1bmm-smoke15-49213078-3af8d69`:

| run | main tokens | main model time | main tok/s | timing tok/s | token equivalence |
| --- | ---: | ---: | ---: | ---: | --- |
| active64 graph + stateful control | `1,084` | `6.420s` | `168.853` | `44.143` | baseline |
| q1 BMM candidate | `1,084` | `4.820s` | `224.910` | `48.686` | PASS main and timing |

Strict main per-window no-regression failed on one one-token window by `0.600ms`; timing per-window passed. The smoke signal was large enough to promote.

## Full-Song Result

Full-song job `49213490` on RTX 2080 Ti (`dcc-core-ferc-s-z25-20`), commit `3af8d69`, run dir `/work/imt11/Mapperatorinator/runs/q1bmm-full-49213490-3af8d69`:

| run | main tokens | main model time | main tok/s | timing tokens | timing tok/s | total timing+map stage | equivalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| active64 graph + stateful control | `7,639` | `49.279s` | `155.014` | `821` | `75.652` | `65.200s` | baseline |
| q1 BMM candidate | `7,639` | `37.981s` | `201.125` | `821` | `84.160` | `52.638s` | PASS |

Generated token IDs matched exactly for both contexts:

- main generation: `7,639 / 7,639`
- timing context: `821 / 821`

Aggregate performance improved across the checked metrics:

- main generation: `+29.7%` tok/s, `-11.298s` model time
- timing context: `+11.2%` tok/s, `-1.097s` model time
- total timing+map stage wall: `-12.562s`, `-19.3%`

Strict zero-tolerance per-window gates did not pass:

- main generation: `4 / 87` late one-token windows regressed, totaling `4.4ms` failed-window model overhead
- timing context: `1 / 87` late one-token window regressed, totaling `1.0ms` failed-window model overhead

The aggregate savings dominate those scoped micro-regressions by orders of magnitude, and generated-token identity passed.

## Decision

Keep `inference_q1_bmm_cross_attention=true` as a default-off accepted opt-in for the simple fp32 active-prefix graph path. This is the first exact full-song opt-in path to cross `200 tok/s` main-generation throughput on RTX 2080 Ti.

Do not make it the cold default. The retained conservative baseline remains compile-only SDPA with active-prefix disabled. Do not generalize the BMM path to self-attention, server/batch generation, non-fp32, or masked attention without fresh logits/token gates and full-song non-regression checks.

## Why It Worked

The target operation is decoder cross-attention with a single query, fixed encoder K/V length, no attention mask, and fp32. On SM75, explicit batched matrix multiplies avoid enough SDPA/fused-attention overhead to reduce real decoder compute substantially. The change is strategically useful for future encoder-decoder work because q_len=1 decoder cross-attention is likely to remain present in an autoregressive encoder-decoder architecture.

## Next Questions

- Re-profile the accepted q1 BMM path to see the new split between self-attention, cross-attention, MLP, output projection, and graph/runtime overhead.
- Do not pursue sampling fusion unless the refreshed trace shows sampling has grown past the `10%` threshold.
- The next plausible exact win is likely one-token linear/MLP launch reduction or a narrow self-attention/cache-layout improvement, not a broader SDPA replacement.
