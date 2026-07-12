# Active-Prefix Bucket Size Sweep

## Hypothesis

The active-prefix CUDA graph path uses bucketed graph shapes. Smaller buckets should reduce padded q_len=1 self-attention/cache work, but may increase graph-shape churn and first-use capture overhead. After `warmup=0` and the stateful monotonic processor, bucket size became worth rechecking under the current fastest opt-in path.

Current starting point:

```bash
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=512
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
```

## 15s Smoke Sweep

- Job: `49205143`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Commit: `39e85e4`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-bucket-sweep-49205143-39e85e4`
- Config: `profile_salvalai_smoke15`, fixed seed, SDPA, `profile_record_token_ids=true`

Each case used isolated TorchInductor, Triton, and CUDA cache dirs. All active buckets matched bucket512 token IDs for both main generation (`1,084 / 1,084`) and timing context (`164 / 164`).

| bucket | main tok/s | main model time | timing tok/s | total stage wall | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| `512` first | `162.357` | `6.677s` | `46.579` | `14.052s` | baseline |
| `256` | `168.553` | `6.431s` | `50.188` | `13.405s` | exact, `+3.8%` main |
| `384` | `165.783` | `6.539s` | `46.921` | `14.038s` | exact, small |
| `768` | `161.035` | `6.731s` | `47.693` | `13.974s` | exact, slower main |
| `1024` | `150.273` | `7.214s` | `44.148` | `15.079s` | exact, slower |
| `512` repeat | `162.122` | `6.686s` | `48.112` | `14.274s` | repeat/noise check |

Because the best bucket was at the lower bound, a second low-bucket smoke sweep tested smaller graph buckets.

- Job: `49205622`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-low-bucket-sweep-49205622-39e85e4`

All tested buckets again matched bucket512 token IDs for both main and timing.

| bucket | main tok/s | main model time | timing tok/s | total stage wall | main delta vs 512 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `512` first | `162.807` | `6.658s` | `45.269` | `14.023s` | baseline |
| `64` | `171.091` | `6.336s` | `50.229` | `13.676s` | `+5.1%` |
| `128` | `170.302` | `6.365s` | `46.924` | `14.267s` | `+4.6%` |
| `192` | `171.509` | `6.320s` | `50.407` | `13.430s` | `+5.3%` |
| `256` | `168.689` | `6.426s` | `46.025` | `13.708s` | `+3.6%` |
| `320` | `169.397` | `6.399s` | `50.024` | `13.778s` | `+4.0%` |
| `512` repeat | `162.362` | `6.676s` | `45.912` | `14.264s` | `-0.3%` |

Bucket64 and bucket192 were promoted to full-song validation.

## Full Song

- Job: `49206207`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
- Commit: `39e85e4`
- Run dir: `/work/imt11/Mapperatorinator/runs/active-bucket-full-49206207-39e85e4`
- Config: `profile_salvalai`, fixed seed, SDPA, `profile_record_token_ids=true`

| bucket | main tokens | main model time | main tok/s | timing tokens | timing tok/s | total stage wall | token equivalence vs 512 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `512` | `7,639` | `51.887s` | `147.223` | `821` | `75.016` | `67.773s` | baseline |
| `192` | `7,639` | `49.369s` | `154.733` | `821` | `77.070` | `65.225s` | PASS main and timing |
| `64` | `7,639` | `49.101s` | `155.578` | `821` | `76.524` | `64.946s` | PASS main and timing |

Against same-job bucket512:

| bucket | main delta | timing delta | total stage delta | strict notes |
| ---: | ---: | ---: | ---: | --- |
| `192` | `+5.1%`, `-2.518s` model | `+2.7%`, `-0.292s` model | `-3.8%`, `-2.549s` | main per-window failed `12 / 87`, total failed main overhead `71ms`; timing per-window failed `3 / 87`, total failed timing overhead `25ms` |
| `64` | `+5.7%`, `-2.786s` model | `+2.0%`, `-0.216s` model | `-4.2%`, `-2.828s` | main per-window failed `12 / 87`, total failed main overhead `67ms`; timing per-window failed `29 / 87`, total failed timing overhead `185ms` |

Against the retained compile-only baseline from job `49113713`, both bucket64 and bucket192 matched generated token IDs (`7,639 / 7,639` main and `821 / 821` timing). The old retained baseline profile is missing newer metadata keys, so `--strict` reports a metadata-contract failure for that historical comparison even though token identity passes.

## Decision

Keep bucket64 as the current fastest exact opt-in active-prefix graph setting for full-song main-generation throughput:

```bash
inference_active_prefix_decode_bucket_size=64
```

This is a config-only full-song win over the previous bucket512 active graph path: `147.223 -> 155.578 tok/s` in the same job, with exact generated-token identity, improved aggregate timing throughput, and lower total timing+map stage wall time. It is a 5-10% win, so it is worth keeping under the existing rule for simple, well-contained optimizations.

Document the caveat: bucket64 has broader timing-context per-window regressions than bucket192, even though aggregate timing and total stage wall improve. Use bucket192 as the safer fallback when timing-context per-window stability matters more than maximum main-generation throughput.

The retained cold single-song default baseline is still compile-only SDPA with active-prefix disabled. Active-prefix remains opt-in.
