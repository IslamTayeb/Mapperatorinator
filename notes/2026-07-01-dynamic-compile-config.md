# Dynamic Compile Config Experiment

## Idea

The accepted generation-compile win leaves the Transformers default compile configuration in place. After `CompileConfig(dynamic=False)` regressed badly, this scout tested the opposite explicit setting: `CompileConfig(dynamic=True, mode="reduce-overhead")` while keeping `inference_generation_compile=true`.

This was worth one smoke run because the installed Transformers documentation describes dynamic compile with static cache, and the idea would have carried to future autoregressive encoder-decoder inference loops if it worked.

## Result

Rejected and reverted.

RTX 2080 Ti smoke slice on DCC `gpu-common`, node `dcc-core-ferc-s-z25-21`:

- Baseline commit: `3e9033c`
- Baseline job: `49113275`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-compile-49113275-3e9033c/beatmapa00e6b0a1f394a05a1bc95ab28a1dac4.osu.profile.json`
- Candidate commit: `96aecec`
- Candidate job: `49118947`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-dyncfg-49118947-96aecec/beatmapff0b384d836c47d0b502e73f79a16e26.osu.profile.json`
- Slurm stderr: `/work/imt11/Mapperatorinator/logs/smoke-dyncfg-49118947.err`

Comparator result:

```text
tokens_per_second: baseline=70.259, candidate=63.226, delta=-7.033 (-10.0%, worse)
model_elapsed_seconds: baseline=41.190, candidate=45.772, delta=+4.582 (+11.1%, worse)
generated_tokens: baseline=2894, candidate=2894
Token equivalence: PASS (2894 generated token IDs match).
```

The job completed successfully with `sacct` state `COMPLETED`, exit code `0:0`, elapsed `00:02:16`.

## Interpretation

This is an exact-calculation loss. The generated token IDs matched, but model time regressed.

Unlike `dynamic=False`, this run did not show the decoder stride recompile-limit warning in the observed log tail. It did show repeated cudagraph partition messages and a large first-window setup cost. Post-warmup main-generation windows were around `89-95 tok/s`, not above the default compile-config baseline enough to justify keeping an explicit compile config.

Leave `generation_config.compile_config` unset for the accepted `inference_generation_compile` path.
