# Max-Autotune Generation Compile Config

## Idea

Test whether PyTorch Inductor `mode="max-autotune"` improves the retained SDPA plus generation-compile baseline. This is distinct from the rejected `dynamic=False` and `dynamic=True` compile-config experiments because it keeps `dynamic=None` and changes only the compile mode.

The scout added a temporary default-off `inference_generation_compile_mode` flag, then tested:

```python
CompileConfig(mode="max-autotune")
```

## Result

Rejected and reverted.

- Candidate commit: `0cccf36`
- Baseline job: `49136379`
- Candidate job: `49136380`
- Node: `dcc-core-ferc-s-z25-21`
- GPU: RTX 2080 Ti
- Baseline profile: `/work/imt11/Mapperatorinator/runs/smoke15-compile-default-0cccf36/beatmap1572085922b24c95b60ecf1f5abe5ec4.osu.profile.json`
- Candidate profile: `/work/imt11/Mapperatorinator/runs/smoke15-compile-maxautotune-0cccf36/beatmapf92dca264acf4af3bd5a08b78880c75a.osu.profile.json`

Comparator result:

```text
tokens_per_second: baseline=50.602, candidate=31.765, delta=-18.837 (-37.2%, worse)
model_elapsed_seconds: baseline=21.422, candidate=34.126, delta=+12.703 (+59.3%, worse)
generated_tokens: baseline=1084, candidate=1084
Token equivalence: PASS (1084 generated token IDs match).
```

Post-warmup windows were worse too:

- Baseline seq9: `104.1 tok/s`
- Candidate seq9: `88.9 tok/s`

Stderr showed max-autotune selected some Triton kernels, but those choices did not help the RTX 2080 Ti single-token decode workload. The candidate also paid a larger first-window compile/autotune cost.

## Decision

Do not keep the compile-mode flag just for this failed scout. Leave the accepted baseline with Transformers' default generation compile config: `CompileConfig(fullgraph=False, dynamic=None, backend="inductor", mode="reduce-overhead", options=None)`.
