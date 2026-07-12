# Active-Prefix Mask Fast Path Rejection

## Summary

Tested a narrow active-prefix mask construction fast path and rejected it. The patch was exact, but it made cold 15s main generation much slower. It is reverted.

## Hypothesis

Active-prefix decode currently builds the normal full static-cache 4D causal mask in `prepare_inputs_for_generation`, then `sdpa_attention_forward` slices the mask to the active-prefix bucket. The candidate moved decode-step input preparation under the active-prefix context and capped static-cache mask construction to the active bucket length, with the 2D decoder mask sliced to the same key length.

Expected mechanism: avoid repeated full `max_cache_len` mask construction before active-prefix SDPA slices to bucket512.

## Exactness Gate

- Job: `49158276`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, `attn_implementation=sdpa`, `use_server=false`, `parallel=false`, `cfg_scale=1.0`, `num_beams=1`
- Candidate: active-prefix decode only, decode length `512`
- Run dir: `/work/imt11/Mapperatorinator/runs/ap-mask-gate9b-49158276-dbfd235`

Results:

| gate | pass | max_abs | top-k |
| --- | --- | ---: | --- |
| compile disabled | PASS | `0.0` | match |
| compile enabled | PASS | `2.2888e-05` | match |

The first attempted gate job `49158113` failed before testing the patch because the default `sequence_index=0` generated only one captured HF raw-logit step. The real gate used `sequence_index=9`.

## Smoke Result

- Job: `49158365`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti, driver `595.71.05`
- Run dir: `/work/imt11/Mapperatorinator/runs/ap-mask-smoke15-49158365-dbfd235`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/ap-mask-smoke15-49158365-dbfd235/cold_compile/beatmap0e617ba986c74d0fad7347ba99105a93.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/ap-mask-smoke15-49158365-dbfd235/active512/beatmap9716bbce853d43cf94e23ea7a118c341.osu.profile.json`

| run | main tokens | main model time | tok/s | token equivalence |
| --- | ---: | ---: | ---: | --- |
| cold compile-only | `1,084` | `22.369s` | `48.461` | baseline |
| active512 mask fast path | `1,084` | `30.544s` | `35.490` | PASS |

The same-calculation metadata contract passed and all `1,084` generated main-token IDs matched.

Per-window shape:

| window | compile-only | candidate |
| --- | ---: | ---: |
| `seq3` | `16.259s`, `30.690 tok/s` | `26.186s`, `19.056 tok/s` |
| `seq9` | `2.272s`, `103.007 tok/s` | `1.560s`, `150.046 tok/s` |

## Decision

Rejected and reverted. The candidate improved warmed/post-specialization windows but worsened the first long map window enough to regress cold 15s main generation by `-26.8%`. This confirms the active-prefix problem is still graph/capture/specialization discipline, not a simple mask-construction slice.

Do not retry this exact idea unless a future trace proves mask construction has become target-sized and the replacement does not add graph variants or first-window compile/capture cost.
