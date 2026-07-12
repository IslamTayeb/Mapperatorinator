# Inference Mode Experiment

## Idea

Replace `@torch.no_grad()` with `@torch.inference_mode()` for `model_generate` and `model_forward`. This should be exact for pure inference and can reduce autograd/version-counter overhead.

## Smoke Result

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113275`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json`
- Candidate commit: `02b2437`
- Candidate job: `49115644`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-infermode-49115644-02b2437/beatmap9e37e6e4b10f4b8f9a813989da932130.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=70.259, candidate=92.692, delta=+22.433 (+31.9%, better)
model_elapsed_seconds: baseline=41.190, candidate=31.222, delta=-9.969 (-24.2%, better)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

## Full-Song Result

Rejected and reverted.

RTX 2080 Ti full SALVALAI run on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113713`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`
- Candidate commit: `02b2437`
- Candidate job: `49115936`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/full-infermode-49115936-02b2437/beatmapd3e6077d36ea49e795275451d96d09d8.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=92.465, candidate=94.565, delta=+2.100 (+2.3%, better)
model_elapsed_seconds: baseline=82.615, candidate=80.780, delta=-1.835 (-2.2%, better)
generated_tokens: baseline=7639, candidate=7639
Token equivalence: PASS (7639 generated token IDs match).
```

## Interpretation

The smoke result overpredicted the full-song effect. The full-song accepted-result scale was only `+2.3%`, below the keep threshold, so this two-line change was reverted.
