# Active-Prefix Primer Rejection

## Summary

Tested a default-off measured active-prefix primer prototype and rejected it. The primer preserved generated-token identity and improved the first long active-prefix map window, but the setup cost was not recovered in the 15s cold smoke aggregate.

The code was reverted.

## Hypothesis

The active-prefix cold weakness is front-loaded graph/capture/specialization cost. A throwaway primer might pay part of that cost before the first long map window, while still counting the setup inside profiled generation time.

Prototype shape:

- Add default-off flags for a measured active-prefix primer.
- Run a one-time throwaway `model.generate` using a scratch `StaticCache`.
- Use fresh logits processors and `torch.random.fork_rng` so real generation RNG and state are preserved.
- Keep active-prefix default-off and keep the primer active-prefix-only.

## Smoke Result

- Job: `49159121`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Config: `profile_salvalai_smoke15`, `attn_implementation=sdpa`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`
- Run dir: `/work/imt11/Mapperatorinator/runs/ap-primer-smoke15-49159121-a88f7b3`
- Compile-only profile: `/work/imt11/Mapperatorinator/runs/ap-primer-smoke15-49159121-a88f7b3/cold_compile/beatmap93ae628f2add47ab9d8422cd841a8319.osu.profile.json`
- Active512 profile: `/work/imt11/Mapperatorinator/runs/ap-primer-smoke15-49159121-a88f7b3/active512/beatmapd90c2deec8c14a68a43ddbdbdb63a63f.osu.profile.json`
- Primer profile: `/work/imt11/Mapperatorinator/runs/ap-primer-smoke15-49159121-a88f7b3/active512_primer/beatmap9d7ca078f4b141ed81eea9441fc5658d.osu.profile.json`

| run | main tokens | main model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| cold compile-only | `1,084` | `22.599s` | `47.967` | baseline |
| active512 | `1,084` | `30.266s` | `35.816` | PASS vs compile-only |
| active512 + measured primer | `1,084` | `31.477s` | `34.438` | PASS vs compile-only and active512 |

Comparison:

- Primer vs active512: `35.816 -> 34.438 tok/s`, `-3.8%`.
- Primer vs compile-only: `47.967 -> 34.438 tok/s`, `-28.2%`.
- `seq3` improved: active512 `25.909s`, primer `14.923s`.
- Early records paid setup: primer `seq0=11.705s`, `seq1=0.135s`, `seq2=0.111s`.

## Decision

Rejected and reverted. The result supports the cold-overhead diagnosis because explicit priming did move cost away from the first long map window, but this simple primer merely relocated setup cost and added overhead. It is not worth keeping as code.

Future priming work should not be a blind throwaway `generate` wrapper. It needs better bucket/shape coverage, explicit setup profile fields, and a way to amortize or reduce setup rather than moving it into earlier records.
