# Decoder Layer Segment Probe

## Summary

Break down the current fastest exact single-song stack after fused RoPE/cache self-attention. This was a diagnostic-only CUDA graph replay probe, not a throughput claim and not an optimization.

Current accepted exact opt-in baseline remains DCC job `49230082`: full-song SALVALAI `7,639` main tokens, `28.243s` synchronized model time, `270.475 tok/s`, fixed-seed token equivalence PASS for main/timing, and byte-identical `.osu`.

The question was whether the whole one-token decoder layer still has enough surface area for a broad native/runtime island, and which segments dominate after the accepted native attention work.

## Job

- Job: `49231614`
- Commit: `ccfebec`
- Node: `dcc-core-ferc-s-z25-20`
- GPU: RTX 2080 Ti, UUID `GPU-ca8caefc-47d0-69fb-2718-9f51f0cb67c1`
- Driver/CUDA: NVIDIA driver `595.71.05`, CUDA `13.2` from `nvidia-smi`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Run dir: `/work/imt11/Mapperatorinator/runs/decoder-layer-segments-49231614-ccfebec`
- Report: `/work/imt11/Mapperatorinator/runs/decoder-layer-segments-49231614-ccfebec/decoder_layer_segments.json`
- Config: `profile_salvalai_smoke15`, sequence `9`, fp32, SDPA
- Probe shape: one-token decoder step, active-prefix length override `640`, 12 captured decoder layers with one common signature

## Correctness

- Logits replay: PASS, `max_abs=0.0`
- Per-segment CUDA graph replay allclose: PASS for the whole decoder layer, self-attention residual segment, cross-attention residual segment, MLP residual segment, and norm-only diagnostics.

## CUDA Graph Replay Split

Full-song projections use `7,552` one-token decode steps, `12` decoder layers, and the accepted `28.243s` model-time baseline.

| Segment | Graph replay ms/layer | Projected full-song seconds | Model-time share | Layer-share |
| --- | ---: | ---: | ---: | ---: |
| whole decoder layer | `0.176124` | `15.961s` | `56.5%` | `100.0%` |
| self-attn norm + self-attn + residual | `0.077250` | `7.001s` | `24.8%` | `43.9%` |
| cross-attn norm + cross-attn + residual | `0.036630` | `3.320s` | `11.8%` | `20.8%` |
| MLP norm + fc1/GELU/fc2 + residual | `0.046879` | `4.248s` | `15.0%` | `26.6%` |
| residual segment sum | n/a | `14.569s` | `51.6%` | `91.3%` |
| unexplained vs whole layer | n/a | `1.392s` | `4.9%` | `8.7%` |

Norm-only diagnostic ranges were each about `0.52-0.54s` projected full-song. They are useful for target sizing, but they are not additive with the residual segments above.

## Ceiling Math

`500 tok/s` on the accepted `7,639` tokens requires about `15.278s` model time, or `12.965s` saved from the `28.243s` baseline. A `5%` full-song gain is about `1.412s`.

Fantasy free-island ceilings:

| Removed island | Remaining model time | Ceiling tok/s |
| --- | ---: | ---: |
| self-attention residual only | `21.242s` | `359.6` |
| cross-attention residual only | `24.923s` | `306.5` |
| MLP residual only | `23.995s` | `318.4` |
| all three residual segments | `13.674s` | `558.6` |
| whole decoder layer | `12.282s` | `622.0` |

This confirms that self-attention, cross-attention, and MLP are each target-sized, but no narrow single segment can reach `500 tok/s` alone. Reaching `500 tok/s` would require broad decoder-layer savings plus some reduction in the surrounding per-token runtime/control gap.

## Interpretation

The result supports a broad `DecodeSession`/decoder-layer runtime island, not another isolated micro-kernel first. A narrow self-attention, cross-attention, MLP, or per-linear rewrite can still be useful only if its measured projection clears the `>5%` full-song bar and it composes with the existing exact path.

The current target order is:

1. Explain and reduce per-token runtime/control idle around the CUDA graph replay without changing `torch.multinomial` RNG, logits processors, EOS behavior, generated-token counts, or output bytes.
2. If that gate looks safe, prototype a broader decoder-layer/native island that can attack multiple residual segments together.
3. Keep cuBLASLt/CUTLASS/C++/CUDA experiments diagnostic-first and default-off until raw logits, direct-loop token/logit/RNG, smoke token equivalence, full-song token equivalence, and output sanity gates pass.

## Decision

No production optimization graduated. Keep `270.475 tok/s` from job `49230082` as the current exact single-song opt-in baseline.

Use this probe as evidence that the next useful work is broad runtime/kernel design, not standalone sampling fusion, graph-cache cleanup, simple linears, or another narrow attention call-form rewrite.
