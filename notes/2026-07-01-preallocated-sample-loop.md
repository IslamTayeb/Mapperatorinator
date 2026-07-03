# Preallocated Sample Loop Experiment

## Idea

Replace Hugging Face's per-token `torch.cat` updates in `_sample` with an opt-in Mapperatorinator sampling loop that preallocates `input_ids` and `decoder_attention_mask` buffers. The prototype was restricted to the profiled baseline case: batch size 1, sampling, `num_beams=1`, no CFG, static encoder-decoder cache, no streamer, and no return-dict outputs.

This was the main remaining migration-stable candidate because a future autoregressive encoder-decoder core would still need an efficient cached decode loop.

## Result

Rejected and reverted.

First smoke run:

- Candidate commit: `6d6ba7c`
- Job: `49121236`
- Profile: `/work/imt11/Mapperatorinator/runs/smoke-prealloc-49121236-6d6ba7c/beatmap91cce3dd6f0f406687a580823cd2723c.osu.profile.json`
- Main generation: `2,894` tokens, `39.876s`, `72.6 tok/s`

This was only directional because the old smoke baseline had been cleaned from DCC.

Paired same-commit smoke after adding actual-dispatch instrumentation:

- Baseline commit: `f7b5222`
- Baseline job: `49121863`
- Baseline node: `dcc-core-ferc-s-z25-21`
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke-pairbase-49121863-f7b5222/beatmap93c5d8f5a9504f92af53f1698a535fa8.osu.profile.json`
- Candidate commit: `f7b5222`
- Candidate job: `49121864`
- Candidate node: `dcc-core-gpu-ferc-s-h36-6`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke-prealloc2-49121864-f7b5222/beatmape999b779433b4f8395818898648fd657.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=92.259, candidate=92.686, delta=+0.427 (+0.5%, better)
model_elapsed_seconds: baseline=31.368, candidate=32.367, delta=+0.999 (+3.2%, worse)
generated_tokens: baseline=2894, candidate=3000
Token equivalence: FAIL (baseline_len=2894, candidate_len=3000, first_mismatch=1350).
```

Instrumentation showed `preallocated_sample_enabled=true` for all `20` candidate main-generation windows, so this was a real test of the custom path rather than a fallback.

## Interpretation

This is not an acceptable same-calculation speedup. Token identity changed, output length changed, and the token-normalized speed delta was far below the keep threshold.

The divergence happened inside main generation, not just through timing-context contamination. Main sequence token counts matched through sequence 5, then diverged during sequence 6; the flattened first mismatch was at token `1,350`.

The candidate stderr also showed a TorchDynamo recompile due to timing/main model output projection shape mismatch:

```text
Recompiling function forward in .../modeling_mapperatorinator.py:150
tensor '...proj_out...weight' size mismatch at index 0. expected 4097, actual 4069
```

That recompile was model-shape related rather than clear per-token stride churn, but the token mismatch alone rejects the experiment.

Do not retry a mutable preallocated `_sample` loop as a quick optimization. A future custom decode loop would need to prove token identity first with compile disabled, then with compile enabled, before any throughput claim.
