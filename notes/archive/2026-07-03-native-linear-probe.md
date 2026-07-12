# Native-Stack Linear/MLP Probe

## Purpose

Re-run the captured one-token decoder linear/MLP probe after the accepted native q_len=1 self-attention stack. Post-native profiling showed GEMV/GEMM/linear work was again the largest bucket, so this checked whether simple PyTorch call-form changes still had enough ceiling before starting native MLP/projection work.

This is diagnostic only. It is not an inference throughput claim.

## Baseline Context

Current fastest exact opt-in baseline:

- active-prefix bucket64
- CUDA graph warmup0/min decode step 1
- stateful monotonic logits processor
- q_len=1 BMM cross-attention
- persistent DecodeSession graph/cache reuse
- native q_len=1 self-attention for map generation

Full-song accepted result from DCC job `49225493`: `7,639` SALVALAI main tokens, `32.217s` synchronized model time, `237.111 tok/s`, fixed-seed main/timing token equivalence PASS, and byte-identical generated `.osu` output.

For this campaign, `500 tok/s` requires about `15.278s` full-song main-generation model time, another `52.6%` reduction from that accepted opt-in path. A production candidate should project to at least `>5%` full-song saving, roughly `1.61s`, before it becomes worth implementation work.

## Job

| job | commit | state | note |
| --- | --- | --- | --- |
| `49228126` | `dd62e84` | `FAILED` after valid JSON | Probe produced `linear_probe.json`; the wrapper failed only while reading a literal `$RUN/linear_probe.json` path in its summary step. |

Artifacts:

- Run root: `/work/imt11/Mapperatorinator/runs/native-linear-probe-49228126-dd62e84`
- JSON: `/work/imt11/Mapperatorinator/runs/native-linear-probe-49228126-dd62e84/linear_probe.json`
- Slurm logs: `/work/imt11/Mapperatorinator/logs/native-linear-probe-49228126.out` and `.err`

Environment:

- Node: `dcc-core-gpu-ferc-s-h36-5`
- GPU: RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `seed=12345`
- Candidate flags: q_len=1 BMM cross-attention enabled, native q_len=1 self-attention enabled

## Correctness

The probe passed its replay gate:

- `pass=true`
- `logits_replay_allclose=true`
- `logits_replay_max_abs=0.0`
- captured decoder linear calls: `73`
- active-prefix length for the sampled step: `128`

Captured per-token linear call counts stayed the same as the earlier q1 stack:

| operation | calls/token |
| --- | ---: |
| decoder self-attn `Wqkv` | `12` |
| decoder self-attn `Wo` | `12` |
| decoder cross-attn `Wq` | `12` |
| decoder cross-attn `Wo` | `12` |
| decoder MLP `fc1` | `12` |
| decoder MLP `fc2` | `12` |
| decoder output projection | `1` |

## Results

Representative CUDA-event timings on RTX 2080 Ti:

| signature | representative calls/token | `F.linear` | best simple variant | isolated speedup |
| --- | ---: | ---: | ---: | ---: |
| `3072 -> 768`, bias | `12` | `0.023706ms` | `matmul`, `0.023519ms` | `1.008x` |
| `768 -> 2304`, bias | `12` | `0.024369ms` | `matmul`, `0.023555ms` | `1.035x` |
| `768 -> 3072`, bias | `12` | `0.023497ms` | `matmul`, `0.022953ms` | `1.024x` |
| `768 -> 4069`, no bias | `1` | `0.025025ms` | `mv`, `0.024972ms` | `1.002x` |
| `768 -> 768`, bias | `36` | `0.023606ms` | `matmul`, `0.023269ms` | `1.014x` |

The synthetic one-layer MLP block also stayed small:

| variant | time | speedup |
| --- | ---: | ---: |
| `F.linear -> GELU -> F.linear` | `0.060290ms` | `1.000x` |
| `addmm -> GELU -> addmm` | `0.058566ms` | `1.029x` |
| `mv -> GELU -> mv` | `0.057507ms` | `1.048x` |

## Projection

Picking the best simple variant per captured linear signature saves only about `0.03ms/token`. Across `7,639` full-song main tokens, that is roughly `0.2-0.3s` before integration overhead, far below the `>1.61s` `5%` campaign threshold.

Applying the best synthetic MLP call-form speedup across all 12 layers would save roughly `0.25s` full-song main-generation model time. That is also far below threshold.

## Decision

Do not implement production changes that replace decoder `F.linear` calls with PyTorch `matmul`, `addmm`, or `mv`. They are exact but too small.

Do not treat the full post-native GEMV/GEMM bucket as a cheap call-form target. The remaining linear work is real, but a meaningful win likely requires a fused/native decoder island such as MLP fusion, projection plus adjacent layout work, or attention projection/cache fusion. Those should start from captured-tensor microprobes and project to `>5%` full-song saving before production code.

Next measured question: tune or specialize the existing native q_len=1 attention kernel, because it is already integrated, exact, default-off, and still target-sized in the post-native trace.
