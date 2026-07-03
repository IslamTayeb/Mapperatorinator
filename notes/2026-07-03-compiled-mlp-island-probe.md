# Compiled MLP Island Probe

## Purpose

Check whether a PyTorch-compiled one-layer decoder MLP island can beat the current `fc1 -> GELU -> fc2` path enough to justify deeper MLP fusion work under the accepted native opt-in stack.

This is diagnostic only. It does not claim an inference throughput win.

## Job

| job | commit | state |
| --- | --- | --- |
| `49228228` | `356fb8a` | `COMPLETED` |

Artifacts:

- Run root: `/work/imt11/Mapperatorinator/runs/compiled-mlp-probe-49228228-356fb8a`
- JSON: `/work/imt11/Mapperatorinator/runs/compiled-mlp-probe-49228228-356fb8a/compiled_mlp_probe.json`
- Summary: `/work/imt11/Mapperatorinator/runs/compiled-mlp-probe-49228228-356fb8a/summary.json`
- Slurm logs: `/work/imt11/Mapperatorinator/logs/compiled-mlp-probe-49228228.out` and `.err`

Environment:

- Node: `dcc-core-gpu-ferc-s-h36-5`
- GPU: RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`

Flags:

```text
inference_generation_compile=true
inference_q1_bmm_cross_attention=true
inference_native_q1_self_attention=true
--compile-mlp-variant
```

## Correctness

The direct replay gate passed:

- `pass=true`
- `logits_replay_allclose=true`
- `logits_replay_max_abs=0.0`
- captured decoder linears: `73`
- active-prefix length: `128`

The compiled MLP block output was allclose to the eager MLP reference, with `max_abs=1.907e-06`.

## Result

One-layer decoder MLP timings:

| variant | time | isolated speedup |
| --- | ---: | ---: |
| `functional_mlp` | `0.060294ms` | `1.000x` |
| `addmm_mlp` | `0.057353ms` | `1.051x` |
| `mv_mlp` | `0.055374ms` | `1.089x` |
| `compiled_functional_mlp` | `0.118316ms` | `0.510x` |

The compiled PyTorch MLP island is slower than eager. It is not a production candidate.

The best simple variant in this run (`mv_mlp`) saves `0.004920ms/layer/token`. If that unrealistically applied cleanly to all 12 decoder layers over `7,639` full-song main tokens, it would save about `0.451s`, projecting the accepted `237.111 tok/s` path to only about `240.48 tok/s`. That remains well below the `>1.61s` `5%` full-song threshold.

## Decision

Do not pursue `torch.compile` around the MLP island as an optimization.

Do not implement `mv_mlp` or other simple PyTorch MLP call-form changes in production; the ceiling is still too small.

MLP remains a plausible area only if a real native/CUTLASS/cuBLASLt fused block can deliver roughly `1.4x` for the full MLP island or otherwise save at least `0.018ms/layer/token`. Anything smaller is below the current campaign threshold.

The next larger exact target should be measured, not assumed:

1. a real native/CUTLASS MLP block probe, if we are willing to write a narrow C++/CUDA diagnostic kernel; or
2. the self-attention projection/RoPE/cache/layout island, which combines `Wqkv`, RoPE, cache update, q1 attention, and `Wo`.
