# Static Compile Config Experiment

## Idea

The accepted generation-compile win leaves the Transformers default compile configuration in place. This scout forced `CompileConfig(dynamic=False, mode="reduce-overhead")` while keeping `inference_generation_compile=true`, testing whether static-shape specialization would make the repeated decode loop faster.

This would have been architecture-stable if it worked, because future Mapperatorinator-like autoregressive encoder-decoder cores should still have a repeated generation loop.

## Result

Rejected and reverted.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113275`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json`
- Candidate commit: `d6ce772`
- Candidate job: `49116934`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-compilecfg-49116934-d6ce772/beatmap9c4f9667e17a4de8a00b5e3a86400669.osu.profile.json`
- Slurm stderr: `/work/imt11/Mapperatorinator/logs/smoke-compilecfg-49116934.err`

Comparator result:

```text
tokens_per_second: baseline=70.259, candidate=53.755, delta=-16.504 (-23.5%, worse)
model_elapsed_seconds: baseline=41.190, candidate=53.836, delta=+12.646 (+30.7%, worse)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

The job completed successfully with `sacct` state `COMPLETED`, exit code `0:0`, elapsed `00:02:57`.

## Interpretation

This is an exact-calculation loss. The generated token IDs matched, but model time regressed badly.

The likely reason is compile specialization churn. Slurm stderr reported:

```text
torch._dynamo hit config.recompile_limit (8)
function: 'forward' (.../modeling_mapperatorinator.py:139)
last reason: tensor 'decoder_input_ids' stride mismatch at index 0. expected 24, actual 25
```

Forcing `dynamic=False` made the generation compile path brittle to per-step decoder input shape or stride changes. Leave the accepted generation-compile path on its default compile config unless a future PyTorch/Transformers update changes this behavior and fresh smoke profiling proves a win.
