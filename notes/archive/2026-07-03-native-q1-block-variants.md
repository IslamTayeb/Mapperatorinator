# Native q1 Attention Block-Size Variants

## Purpose

After accepting native q_len=1 self-attention, post-native profiling still showed attention as a target-sized bucket. This probe checked whether the existing narrow native kernel had an easy SM75 tuning win from changing the CUDA block size before attempting broader attention/cache fusion.

This is diagnostic only. It does not claim an inference throughput win.

## Utility Change

`utils/profile_decode_attention_components.py` now supports:

```text
--native-q1-block-sizes=64,128,256
```

The flag benchmarks diagnostic native q1 self-attention block-size variants against the same captured tensors used by the existing attention component probe. It does not change production inference.

## Job

| job | commit | state | note |
| --- | --- | --- | --- |
| `49228161` | `b2cd395` | `FAILED` after valid JSON | All per-length JSONs were written; the shell reducer failed because it globbed both primary JSON and `.stdout.json`. |

Artifacts:

- Run root: `/work/imt11/Mapperatorinator/runs/native-q1-variants-49228161-b2cd395`
- Primary per-length JSONs: `native_q1_variants_len{128..1024}.json`
- Parsed primary-only summary: `summary_primary_only.json`
- Slurm logs: `/work/imt11/Mapperatorinator/logs/native-q1-variants-49228161.out` and `.err`

Environment:

- Node: `dcc-core-gpu-ferc-s-h36-5`
- GPU: RTX 2080 Ti `GPU-cbcd7909-6bd2-1b73-d336-a63ead10aa5d`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision: `fp32`
- Attention: `sdpa`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`

## Results

All native block-size variants matched the captured self-attention reference:

- allclose: PASS for all tested lengths and block sizes
- max abs: `3.576e-07`

Representative self-attention-only timings:

| active prefix | current block128 | block64 | block256 | best |
| ---: | ---: | ---: | ---: | --- |
| `128` | `0.018754ms` | `0.018762ms` | `0.018974ms` | block128 |
| `192` | `0.017573ms` | `0.017443ms` | `0.017625ms` | block64 |
| `256` | `0.017288ms` | `0.017499ms` | `0.017347ms` | block128 |
| `320` | `0.019555ms` | `0.021357ms` | `0.019590ms` | block128 |
| `384` | `0.022660ms` | `0.025255ms` | `0.022647ms` | block256 |
| `448` | `0.026158ms` | `0.029108ms` | `0.025808ms` | block256 |
| `512` | `0.029190ms` | `0.032908ms` | `0.028848ms` | block256 |
| `576` | `0.032691ms` | `0.036723ms` | `0.032610ms` | block256 |
| `640` | `0.035850ms` | `0.040679ms` | `0.035676ms` | block256 |
| `704` | `0.039318ms` | `0.044491ms` | `0.038730ms` | block256 |
| `768` | `0.042470ms` | `0.048288ms` | `0.041780ms` | block256 |
| `1024` | `0.067433ms` | `0.079741ms` | `0.064499ms` | block256 |

Simple average over lengths:

- block64 vs block128: slower on average (`0.914x` speed ratio vs block128)
- block256 vs block128: `1.008x` average speedup
- best observed block256 case: `1.045x` at length `1024`

## Full-Song Projection

Using active64 full-song replay counts from diagnostic job `49207288`:

| active prefix | decode replays | best block | projected saving |
| ---: | ---: | --- | ---: |
| `128` | `22` | block128 | `0.0000s` |
| `192` | `64` | block64 | `0.0001s` |
| `256` | `126` | block128 | `0.0000s` |
| `320` | `227` | block128 | `0.0000s` |
| `384` | `136` | block256 | `0.0000s` |
| `448` | `332` | block256 | `0.0014s` |
| `512` | `682` | block256 | `0.0028s` |
| `576` | `1,727` | block256 | `0.0017s` |
| `640` | `2,907` | block256 | `0.0061s` |
| `704` | `1,221` | block256 | `0.0086s` |
| `768` | `108` | block256 | `0.0009s` |

Total projected saving from selecting the best tested block size per prefix is only `0.0216s` over full-song SALVALAI main generation. Against the accepted native path's `32.217s` model time, this is about `0.067%`, projecting `237.111 -> 237.270 tok/s`.

Forcing block256 everywhere projects similarly: `0.0212s`, about `0.066%`.

## Decision

Do not change the production native q1 attention block size. Block256 helps the long-prefix tail but the full-song weighted ceiling is far below the `5%` campaign threshold and even below the `1-3%` remove-complexity band.

The useful takeaway is negative: further self-attention-only kernel tuning is unlikely to move full-song throughput unless it changes more than the block schedule. Future attention work should target a larger fused island, such as projection + RoPE/cache update + native q1 attention, and must project to `>5%` before production integration.
