# Post-Fused Weighted Decoder Accounting

## Purpose

Re-profile the accepted fused RoPE/cache self-attention stack before starting any broader native runtime or kernel work. The accepted full-song single-song baseline is DCC job `49230082`: `7,639` SALVALAI main tokens, `28.243s` synchronized model time, `270.475 tok/s`, fixed-seed main/timing token equivalence PASS, and byte-identical `.osu` output.

The question for this pass was whether a remaining island is still large enough to justify C++/CUDA/CUTLASS work toward `500 tok/s`. For SALVALAI, `500 tok/s` means about `15.278s` model time, so the remaining required saving is about `12.965s`.

## Run

- DCC job: `49230173`
- Commit: `e63e022`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti `GPU-de50b870-b708-690d-74a4-6e855163a133`
- Run dir: `/work/imt11/Mapperatorinator/runs/post-fused-accounting-49230173`
- Summary: `/work/imt11/Mapperatorinator/runs/post-fused-accounting-49230173/summary.json`
- Config: `profile_salvalai_smoke15`, `seq9`, `precision=fp32`, `attn_implementation=sdpa`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`
- Active stack: generation compile, active-prefix bucket64, CUDA graph warmup0/min-decode1, stateful monotonic processor, q1 BMM cross-attention, DecodeSession runtime + CUDA graph, native q1 self-attention, fused RoPE/cache self-attention

Two preflight jobs produced no profiling evidence:

- `49230162` failed because the launcher mangled a Python heredoc in the environment print block.
- `49230169` failed because the Hydra config resolved the local Mac `audio_path`; job `49230173` fixed this with `audio_path=/work/imt11/Mapperatorinator/data/salvalai.mp3`.

## Weighted Results

Replay counts used the full-song active64 prefix distribution:

```text
128:22, 192:64, 256:126, 320:227, 384:136, 448:332,
512:682, 576:1727, 640:2907, 704:1221, 768:108
```

All completed probes passed their logits/allclose gates.

| island | projected full-song seconds | share of `28.243s` | ideal tok/s if free | interpretation |
| --- | ---: | ---: | ---: | --- |
| full model forward graph replay | `17.000s` | `60.2%` | `679.4` | decoder step compute is still large, but production synchronized model time has a big non-forward/control gap |
| decoder layers graph replay | `16.319s` | `57.8%` | `640.6` | only broad decoder-layer/runtime work is large enough for the remaining `500 tok/s` gap |
| self-attention modules | `6.403s` | `22.7%` | `349.8` | still large, but not enough alone; fused RoPE/cache removed the clean setup win |
| q1 cross-attention modules at `L=640` | `2.956s` | `10.5%` | `302.1` | target-sized for small wins, not enough for `500 tok/s` |
| linear calls at `L=640` | `7.321s` | `25.9%` | `365.1` | broad launch/linear reduction matters, but individual `F.linear` rewrites were already rejected |
| MLP module at `L=640` | `3.780s` | `13.4%` | `312.3` | standalone MLP fusion might clear `5%` only with a large speedup |

Representative `L=640` CUDA graph replay timings:

| component | ms per module/layer | projected seconds |
| --- | ---: | ---: |
| decoder layer | `0.185454` | `16.807s` across 12 layers and 7,552 decode steps |
| self-attention module | `0.073844` | `6.692s` across 12 layers and 7,552 decode steps |
| cross-attention module | `0.032618` | `2.956s` across 12 layers and 7,552 decode steps |
| MLP module | `0.041715` | `3.780s` across 12 layers and 7,552 decode steps |

Standalone linear projections at `L=640`:

| operation group | count per token | graph ms/call | projected seconds |
| --- | ---: | ---: | ---: |
| `decoder.self_attn.qkv` | `12` | `0.016904` | `1.532s` |
| `decoder.cross_attn.out`, `decoder.cross_attn.q`, `decoder.self_attn.out` | `36` | `0.006877` | `1.870s` |
| `decoder.mlp.fc1` | `12` | `0.020617` | `1.868s` |
| `decoder.mlp.fc2` | `12` | `0.020608` | `1.868s` |
| `decoder.output_projection` | `1` | `0.024361` | `0.184s` |

## Interpretation

The fused self-attention cache kernel did its job: the old self-attention setup/cache/layout bucket is no longer the obvious next isolated target. The current self-attention probe shows the fused path is essentially tied with the repo module under graph replay, so more self-attention-only kernel work is unlikely to produce another large win unless it changes the broader layer/runtime structure.

The large remaining target is not one embarrassing kernel. The path to `500 tok/s` would need to remove or compress a broad part of the decoder step: many one-token linears, layernorm/residual/MLP work, cross-attention, self-attention, and the CPU/control gap around graph replay, sampling, and stop checks.

The most important accounting wrinkle is the gap between isolated graph replay and production synchronized model time:

- accepted production model time: `28.243s`
- weighted full-forward graph replay: `17.000s`
- implied non-forward/control/synchronization gap: about `11.243s`

Do not start a broad native decoder-layer rewrite until this gap is explained on the current fused stack. A whole decoder-layer island can theoretically reach the target if it removes a large fraction of both the `16.3s` layer replay and the `11.2s` surrounding overhead, but a narrow C++ MLP/cross/self kernel cannot reach `500 tok/s` by itself.

## Decision

No optimization graduated from this diagnostic. Keep `270.475 tok/s` from job `49230082` as the current exact opt-in single-song baseline.

Next diagnostic: rerun active-prefix decode-loop counters on the current fused stack to explain the `~11.2s` gap before choosing between broad DecodeSession/runtime work and a focused native island.
