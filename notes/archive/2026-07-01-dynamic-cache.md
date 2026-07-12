# Dynamic Cache Experiment

## Idea

The static-cache path can force SDPA to operate over max-target cache shapes. The candidate added an opt-in `inference_static_cache=false` path so generation would skip the local preallocated `StaticCache` and use the default/dynamic cache behavior while keeping compiled generation enabled.

## Result

Rejected and reverted as non-equivalent.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113275`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json`
- Candidate commit: `52b8871`
- Candidate job: `49114897`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-dyncache-49114897-52b8871/beatmap36600487b37a414b9f239d9a8f6e9586.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=70.259, candidate=69.424, delta=-0.835 (-1.2%, worse)
model_elapsed_seconds: baseline=41.190, candidate=23.522, delta=-17.668 (-42.9%, better)
generated_tokens: baseline=2894, candidate=1633
Token equivalence: FAIL (baseline_len=2894, candidate_len=1633, first_mismatch=0).
```

## Interpretation

The lower model elapsed time is not usable because the generated-token behavior changed immediately and the candidate produced far fewer tokens. This is a non-equivalent cache/generation behavior change, not a same-calculation inference speedup.

Do not retest dynamic/default-cache generation as an optimization unless the implementation first proves fixed-seed token equivalence against the static-cache baseline.
