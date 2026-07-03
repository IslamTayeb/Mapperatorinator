# Inference Optimization Goal

## Objective

Optimize Mapperatorinator inference on RTX 2080/2080 Ti for same-calculation speedups only.

Current baseline is roughly `65-78 tok/s` main-generation throughput. The first accepted target is `100 tok/s`; `120 tok/s` is a strong first milestone and `150+ tok/s` is stretch. Treat `200 tok/s` as a north star, not the initial stop condition.

## Equivalence Rule

Accepted speedups must preserve fixed-seed generated token IDs for the same audio/config slice. Do not count changed precision, sampling policy, output policy, model quality, windowing/overlap, or generated-token behavior as equivalent speedups.

## Workflow

Start with `configs/inference/profile_salvalai_smoke.yaml`, which profiles the middle 30s of SALVALAI with `seed=12345`, `attn_implementation=sdpa`, `use_server=false`, and `profile_record_token_ids=true`.

Promote only stable, meaningful smoke wins to full-song SALVALAI profiling on RTX 2080/2080 Ti. Compare baseline and candidate profiles with:

```bash
python utils/summarize_inference_profile.py --compare baseline.profile.json candidate.profile.json
```

## Keep Or Remove

Keep `>=10%` full-song main-generation throughput wins. Keep `5-10%` only when the code is very simple. Remove `1-3%` complexity by default.

Document accepted wins, failed ideas, reversions, and DCC job/profile paths in this directory.
