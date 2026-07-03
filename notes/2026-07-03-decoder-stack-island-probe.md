# Decoder Stack Island Probe

## Purpose

Size whether the remaining post-270 gap is caused by the Mapperatorinator/VarWhisper model-forward wrapper around the decoder stack, before writing broader native decoder-layer or `DecodeSession` runtime code.

This is diagnostic infrastructure only. It is not an inference throughput claim.

## Utility

Added:

```text
utils/profile_decode_decoder_stack_island.py
```

The utility prepares the exact one-token active-prefix seq9 input from `profile_salvalai_smoke15`, then benchmarks these equivalent boundaries:

- full `Mapperatorinator.forward()` logits;
- `VarWhisperModel` core decoder hidden state;
- direct `VarWhisperDecoder` stack hidden state;
- output projection only;
- model core plus projection;
- decoder stack plus projection.

It explicitly reproduces Mapperatorinator's outer decoder embedding step when `embed_decoder_input=true`; earlier attempts showed this is required for exact logits.

## DCC Runs

Failed preflights:

| job | commit | result |
| --- | --- | --- |
| `49232655` | `0a0f4df` | failed before measuring; utility did not unwrap `Mapperatorinator.transformer` |
| `49232680` | `3a270da` | wrote useful measurements but failed pass; direct decoder stack bypassed Mapperatorinator's outer decoder embedding, causing logits mismatch |

Passing run:

- Job: `49232726`
- Commit: `75fc8da`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir:

```text
/work/imt11/Mapperatorinator/runs/decoder-stack-island-20260703-140615-75fc8da
```

Report:

```text
/work/imt11/Mapperatorinator/runs/decoder-stack-island-20260703-140615-75fc8da/decoder_stack_island.json
```

Config:

- `profile_salvalai_smoke15`
- seq9, seed `12345`
- fp32, SDPA
- accepted opt-in stack: generation compile, active-prefix bucket64 CUDA graph, stateful monotonic logits processor, q1 BMM cross-attention, DecodeSession CUDA graph, native q1 self-attention, fused RoPE/cache self-attention
- `--cuda-graph-replay --warmup 50 --iters 500`

## Result

Overall pass: `true`.

Correctness:

| check | result |
| --- | --- |
| decoder hidden allclose to model core | PASS, `max_abs=0.0` |
| projected logits allclose to expected full logits | PASS, `max_abs=0.0` |
| graph replay allclose for every benchmarked boundary | PASS |

CUDA graph replay projections over `7,552` full-song one-token decode steps:

| boundary | ms/call | projected full-song s |
| --- | ---: | ---: |
| full model forward logits | `1.848340` | `13.958667` |
| VarWhisper model core hidden | `1.818931` | `13.736568` |
| direct decoder stack hidden | `1.828233` | `13.806819` |
| output projection logits | `0.025425` | `0.192009` |
| model core plus projection logits | `1.883153` | `14.221568` |
| decoder stack plus projection logits | `1.866474` | `14.095608` |

Derived graph replay gaps:

```text
full_minus_model_core_plus_projection_s = -0.263s
full_minus_decoder_stack_plus_projection_s = -0.137s
decoder_stack_plus_projection_minus_component_sum_s = +0.097s
```

Negative full-minus-direct gaps mean the direct stack-plus-projection boundary was slightly slower than the existing full forward under this graph-replay measurement.

## Decision

Keep the utility as diagnostic infrastructure.

Do not pursue a simple Mapperatorinator/VarWhisper wrapper bypass or direct decoder-stack Python replacement as an optimization. The exact direct decoder stack plus output projection did not beat full model forward under CUDA graph replay.

The accepted full-song model time is `28.243s`; the graph-replayed full forward boundary is about `13.959s`, leaving roughly `14.284s` outside this isolated replay boundary. This probe says that gap is not explained by the model-core wrapper itself. It is more likely in production generation-loop/runtime-control structure, repeated one-token graph/call orchestration, or real decoder compute that only a broader native/runtime design would reduce.

## Next

Use this result to narrow the next work:

- Do not implement a direct decoder-stack wrapper bypass.
- Do not restart fast-prepare, graph-copy cleanup, or standalone tail fusion from this evidence.
- If pursuing native work, target real decoder compute such as a broad MLP residual or larger decoder-layer native island, and require a verifier-only projected saving comfortably above the `~1.4s` full-song keep threshold before production wiring.
- If pursuing runtime-control work, first collect lower-level Nsight Compute/Nsight Systems evidence for the accepted graph replay and loop orchestration rather than relying on Python wall spans.
