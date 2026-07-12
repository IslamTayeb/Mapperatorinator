# Generation Compile Accepted Win

## Idea

The profiler showed Mapperatorinator is dominated by repeated single-token autoregressive decode. The code had explicitly disabled the Transformers generation compile path for Hugging Face-loaded models. The candidate added an opt-in `inference_generation_compile` flag and, when enabled, sets `model.generation_config.disable_compile = False`.

This is architecture-stable because a future Mapperatorinator-like encoder-decoder core should still have a repeated autoregressive decode loop even if latent diffusion/compression owns later stages.

## Smoke Result

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `01c18d6`
- Baseline job: `49109301`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-base-49109301-01c18d6/beatmap6f980906005d441fb87edde94f269b83.osu.profile.json`
- Candidate commit: `3e9033c`
- Candidate job: `49113275`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=69.389, candidate=70.259, delta=+0.871 (+1.3%, better)
model_elapsed_seconds: baseline=41.707, candidate=41.190, delta=-0.517 (-1.2%, better)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

The total smoke result was small because the first compiled main-generation window paid compile overhead. Excluding the first main window, the candidate ran `94.5 tok/s` versus baseline `68.4 tok/s`.

## Full-Song Result

RTX 2080 Ti full SALVALAI run on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113712`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/full-base-49113712-3e9033c/beatmap9024531ea69844218e3c15e53ad2972c.osu.profile.json`
- Candidate commit: `3e9033c`
- Candidate job: `49113713`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/full-compile-49113713-3e9033c/beatmapcfb70d0020da473c90f6c1acb32d6bbf.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=62.919, candidate=92.465, delta=+29.546 (+47.0%, better)
model_elapsed_seconds: baseline=121.410, candidate=82.615, delta=-38.795 (-32.0%, better)
generated_tokens: baseline=7639, candidate=7639
Token equivalence: PASS (7639 generated token IDs match).
```

## Interpretation

This is an accepted exact-calculation win. It does not change precision, SDPA baseline, sampling policy, output policy, generated-token behavior, windowing, or model quality. The gain comes from reducing repeated per-token generation-loop overhead after the one-time compile cost.

Keep the global default disabled for short one-off inference, but keep `inference_generation_compile=true` in the SALVALAI profiling configs and use it as the new full-song profiling baseline.
