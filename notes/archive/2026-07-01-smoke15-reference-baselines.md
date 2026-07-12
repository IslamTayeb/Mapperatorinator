# 15s Smoke Reference Baselines

## Purpose

The 200 tok/s phase uses a middle-15-second SALVALAI smoke slice for fast scouting before longer smoke or full-song runs. This note records the first paired reference profiles on the retained branch baseline.

## Runs

Both jobs ran on DCC `gpu-common` with `--gres=gpu:2080:1` on `dcc-core-ferc-s-z25-21` at commit `ce92ebb`.

| run | job | profile | main tokens | main model time | tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| compile-disabled | `49132862` | `/work/imt11/Mapperatorinator/runs/smoke15-nocompile-49132862-ce92ebb/beatmap3ccc5112a05940db8fdd747994c4bef2.osu.profile.json` | 1,084 | 22.068s | 49.1 |
| retained compile baseline | `49132861` | `/work/imt11/Mapperatorinator/runs/smoke15-compile-49132861-ce92ebb/beatmapfb8360ef206b41f4a6cc1fe7a6a735ed.osu.profile.json` | 1,084 | 13.335s | 81.3 |

`utils/summarize_inference_profile.py --compare` reported token equivalence PASS for all `1,084` generated main-generation token IDs.

## Interpretation

The retained compile baseline is `+65.5%` faster than compile-disabled on this 15s slice by total synchronized main-generation model time. The slice is still warmup-sensitive: the compile-enabled total is `81.3 tok/s`, while post-warmup map windows are mostly around `94-105 tok/s`, matching the accepted full-song compile behavior more closely.

Use these profiles as reference points for future 15s smoke candidates:

1. Compare compile-disabled custom-runtime prototypes against the compile-disabled reference first.
2. Compare compile-enabled candidates against the retained compile reference next.
3. Promote only token-equivalent candidates with a plausible meaningful speed signal to a longer smoke or full-song run.
