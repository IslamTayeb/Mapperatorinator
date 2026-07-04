# Current-Stack Linear Roofline

## Purpose

Refresh linear/GEMV target sizing on the current fastest exact opt-in stack after
the post-270 kernel trace showed `gemv_gemm_linear` as the largest kernel bucket.

This is diagnostic only. It does not claim an inference speedup.

## Utility

Added:

```text
utils/summarize_decode_linear_roofline.py
```

It reads `utils/profile_decode_linear_kernels.py` JSON and computes, per
captured one-token linear signature:

- graph-replayed time;
- FLOPs;
- a lower-bound byte count for fp32 weight/input/output/bias traffic;
- achieved minimum-byte bandwidth;
- projected full-song seconds;
- a nominal peak-memory-bandwidth floor.

The default `--peak-memory-bandwidth-gb-s 616.0` is the nominal RTX 2080 Ti
memory-bandwidth figure and should be treated as a fantasy floor, not a
reachable performance claim.

## Run

- Branch: `experiment/current-linear-roofline`
- Commit: `407edbf`
- DCC job: `49250117`
- Node/GPU: `dcc-core-ferc-s-z25-20`, NVIDIA GeForce RTX 2080 Ti
- Slurm status: `COMPLETED`, exit `0:0`
- Driver/CUDA shown by `nvidia-smi`: `595.71.05` / `13.2`
- Torch/CUDA: `2.10.0+cu128` / `12.8`
- Transformers: `4.57.3`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`
- Precision/backend: `fp32`, `attn_implementation=sdpa`
- Current fast stack: active-prefix bucket64 CUDA graph, stateful monotonic
  logits, q1-BMM cross-attention, DecodeSession graph/cache reuse, native q1
  self-attention, fused RoPE/cache self-attention.

Artifacts:

```text
/work/imt11/Mapperatorinator/runs/current-linear-roofline-49250117-407edbf/linear_graph_probe.json
/work/imt11/Mapperatorinator/runs/current-linear-roofline-49250117-407edbf/linear_roofline.json
```

## Correctness

The underlying linear capture passed:

- `pass=true`
- `logits_replay_allclose=true`
- `logits_replay_max_abs=0.0`
- active prefix length: `128`
- captured decoder linears: `73`

## Result

Projected over `7,552` full-song one-token decode replays:

| Metric | Value |
| --- | ---: |
| Captured linear projected time | `7.156s` |
| Fraction of accepted `28.243s` model time | `25.34%` |
| Nominal peak-bandwidth floor | `5.027s` |
| Removable above that fantasy floor | `2.129s` |
| Achieved minimum-byte bandwidth | `432.7 GB/s` |
| If all captured linears were free | `362.3 tok/s` |
| If captured linears ran at nominal bandwidth floor | `292.5 tok/s` |
| 5% keep bar | `1.412s` |
| 10% keep bar | `2.824s` |
| Required saving for `500 tok/s` | `12.965s` |

Per-signature projections:

| Signature | Operations | Projected | Peak floor | Achieved min-byte bandwidth |
| --- | --- | ---: | ---: | ---: |
| `in3072_out768_bias1` | MLP `fc2` x12 | `1.868s` | `1.391s` | `458.8 GB/s` |
| `in768_out3072_bias1` | MLP `fc1` x12 | `1.858s` | `1.392s` | `461.7 GB/s` |
| `in768_out768_bias1` | self/cross q/out x36 | `1.852s` | `1.045s` | `347.7 GB/s` |
| `in768_out2304_bias1` | self-attn `qkv` x12 | `1.395s` | `1.044s` | `461.2 GB/s` |
| `in768_out4069_bias0` | output projection x1 | `0.183s` | `0.153s` | `516.1 GB/s` |

## Interpretation

The linear/GEMV bucket is real and target-sized as model math, but it is not an
easy per-linear optimization target:

- zeroing every captured one-token linear would still only reach about
  `362 tok/s`, well short of `500 tok/s`;
- the fantasy memory-bandwidth floor leaves only `2.129s` above nominal RTX
  2080 Ti bandwidth, which is above the 5% bar but below the 10% bar and far
  short of the `12.965s` required for `500 tok/s`;
- the achieved minimum-byte bandwidth is already about `433 GB/s`, so simple
  call-form or per-linear wrapper work is unlikely to recover the whole
  `2.129s`;
- this matches the previous production native-linear attempt: exact, but only
  `+2.99%` full-song.

## Decision

Keep `utils/summarize_decode_linear_roofline.py` as diagnostic infrastructure.
Do not start standalone per-linear CUDA/cuBLAS/CUTLASS production work from this
result.

Future linear work must be paired with broader decoder-layer/runtime fusion that
also reduces adjacent elementwise, residual, layout, cache, or attention work.
Before production code, require a current-stack verifier or microprobe showing a
combined projected saving above the normal keep threshold, not just a large
linear category share.
