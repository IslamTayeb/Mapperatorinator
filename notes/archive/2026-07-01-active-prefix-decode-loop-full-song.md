# Active-Prefix Decode Loop Full-Song Validation

## Summary

Bucketed active-prefix decode is exact in the tested simple generation path and remains useful runtime evidence, but it is not the retained cold single-song baseline.

The retained cold single-song baseline is still SDPA plus `inference_generation_compile=true`, active-prefix disabled: job `49113713`, `7,639` SALVALAI main tokens, `82.615s` synchronized model time, `92.465 tok/s`, token equivalence PASS against compile-disabled.

## Hypothesis

Full static-cache self-attention forces most one-token decode steps to attend over the full static target length. Prior SDPA microprofiles showed `q_len=1` attention cost scales strongly with KV length. Decode-only active-prefix self-attention passed one-token logits gates, while active-prefix prefill failed. The candidate therefore leaves prefill unchanged, then uses bucketed active-prefix lengths only during one-token decode.

Bucketing keeps graph/cache shapes reusable while cutting the effective self-attention length for most decode steps.

## Validation Evidence

| job | run | main tokens | main model time | tok/s | token equivalence | status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `49150185` | compile-only baseline | 7,639 | 82.142s | 92.998 | baseline | clean same-job compile comparison |
| `49150185` | active-prefix bucket 256 | 7,639 | 62.653s | 121.926 | PASS, 7,639 / 7,639 | over-promoted; not reproduced as cold baseline |
| `49151748` | bucket 256, first in sweep | 7,639 | 86.122s | 88.699 | PASS vs bucket 128 | rejected for cold baseline |
| `49151748` | bucket 128, second in sweep | 7,639 | 93.850s | 81.396 | PASS vs bucket 256 | rejected |
| `49151748` | bucket 512, third in sweep | 7,639 | 70.527s | 108.313 | PASS vs bucket 256 | warm/order-sensitive candidate |
| `49152465` | bucket 512, cold-first validation | 7,639 | 78.108s | 97.800 | PASS vs same-job compile-only | strategic opt-in candidate |
| `49152465` | compile-only after active512 | 7,639 | 89.888s | 84.984 | baseline for same-job compare | anomalously slow versus retained baseline |

Important paths:

- `49150185` run dir: `/work/imt11/Mapperatorinator/runs/active-prefix256-full-49150185-821fb41`
- `49151748` run dir: `/work/imt11/Mapperatorinator/runs/active-prefix-bucket-full-49151748-1f20478`
- `49152465` run dir: `/work/imt11/Mapperatorinator/runs/active-prefix512-validate-49152465-1f20478`
- Active512 cold-first profile: `/work/imt11/Mapperatorinator/runs/active-prefix512-validate-49152465-1f20478/active512-cold/beatmap88846e053e3d447bb4eac47696c387e2.osu.profile.json`

## Regression Check

The user explicitly wants accepted changes to prove performance has not degraded. Active-prefix does not pass that bar as a retained cold single-song baseline.

Job `49152465` active512 cold-first versus the retained compile-only baseline:

| metric | retained compile-only | active512 cold-first | result |
| --- | ---: | ---: | --- |
| main generation | 82.615s, 92.465 tok/s | 78.108s, 97.800 tok/s | small `~5.8%` improvement |
| first main-generation window | compile-only clean runs around `8.138-15.971s` | 24.739s, 23.7 tok/s | regressed |
| timing generation | clean compile-only job `49150185`: 24.060s, 34.1 tok/s | 39.610s, 20.7 tok/s | regressed |
| total profiled stage time | clean compile-only job `49150185`: 111.531s | active512 stage sum about 123s | regressed |
| generated main tokens | 7,639 | 7,639 | unchanged |
| token IDs | baseline | candidate | PASS in same-job compare |

The same-job compile-only run in `49152465` was unusually slow (`84.984 tok/s`), so do not claim the `+15.1%` paired delta as the retained cold single-song improvement.

## Decision

Keep the opt-in active-prefix code and notes because the idea is exact in the tested decode-only path and strategically useful for a future custom runtime or warmed multi-song process. Do not enable it in `configs/inference/profile_salvalai.yaml`.

Use the retained cold baseline:

```yaml
inference_generation_compile: true
inference_active_prefix_decode_loop: false
```

Use active-prefix only for explicitly labeled experiments, for example:

```bash
python inference.py --config-name profile_salvalai \
  inference_active_prefix_decode_loop=true \
  inference_active_prefix_decode_bucket_size=512
```

## Lessons

- Active-prefix during prefill is non-equivalent; keep it decode-only.
- Bucket size matters. Bucket `512` appears less harmful cold than bucket `128`/`256`, but it still has a first-window tax.
- Warm/post-specialization windows can reach about `130-145 tok/s`, which is valuable for future warm-repeat or multi-song work but cannot be reported as cold single-song throughput.
- Future work should target graph/runtime discipline: stable bucket capture, fewer graph variants, fewer per-token compiled graph calls, and less repeated cache/mask/update plumbing.

## Next

- Add a warm-repeat SALVALAI suite before making batch or multi-song claims: same model process, repeated song, per-run seed reset, first run reported separately from runs 2..N.
- Prototype graph-backed or bufferized direct decode only if it can preserve active-prefix exactness and reduce the first-window/specialization tax.
- Keep the no-degradation rule: a candidate must not regress timing generation, total profiled stage time, token counts, exactness, or per-window performance unless explicitly labeled as a scoped regression and approved.
