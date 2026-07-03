# Fullgraph Generation Compile Config

## Idea

Test whether forcing fullgraph compilation improves the retained generation-compile baseline:

```python
CompileConfig(fullgraph=True, mode="reduce-overhead")
```

This is distinct from the rejected `dynamic=False`, `dynamic=True`, and `mode="max-autotune"` scouts.

## Result

Rejected and reverted.

- Candidate commit: `149fe88`
- Candidate job: `49137638`
- Baseline profile reused: `/work/imt11/Mapperatorinator/runs/smoke15-mask-base-2d807c1/beatmap4438946a6a864548b8d968d4a3a24682.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke15-fullgraph-149fe88/beatmapa2f8469fd07e475aaa4d7f68477dd5ca.osu.profile.json`
- Node: `dcc-core-ferc-s-z25-21`
- GPU: RTX 2080 Ti

Comparator result:

```text
tokens_per_second: baseline=51.392, candidate=50.992, delta=-0.399 (-0.8%, worse)
model_elapsed_seconds: baseline=21.093, candidate=21.258, delta=+0.165 (+0.8%, worse)
generated_tokens: baseline=1084, candidate=1084
Token equivalence: PASS (1084 generated token IDs match).
```

Post-warmup windows were mixed. Seq9 improved slightly (`104.6 -> 105.3 tok/s`), but the total 15s slice regressed and the effect size is far below the keep threshold.

## Decision

Do not keep forced fullgraph compile. It is not a meaningful path toward 200 tok/s on the current RTX 2080 Ti baseline. Leave the retained generation compile config unset so Transformers uses its default `CompileConfig(fullgraph=False, dynamic=None, backend="inductor", mode="reduce-overhead", options=None)`.
