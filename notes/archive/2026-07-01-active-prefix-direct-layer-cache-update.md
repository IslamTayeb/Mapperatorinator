# Active-Prefix Direct-Layer Cache Update Rejection

## Summary

Tested a default-off active-prefix decode experiment that bypassed `Cache.update(..., layer_idx, ...)` dispatch and called the underlying per-layer static-cache update directly for VarWhisper SDPA self-attention during decode.

The one-token logits gates passed with compile disabled and enabled, and generated-token equivalence passed on the 15s smoke. The speed result regressed, so the code was reverted.

## Hypothesis

Active-prefix cold overhead logs showed `transformers/cache_utils.py:update` recompiling by `layer_idx` and hitting the TorchDynamo recompile limit. Calling the already-selected cache layer directly might avoid part of the Python/container dispatch guard while preserving Hugging Face `StaticLayer.update` tensor mutation semantics.

Prototype shape:

- Add a default-off `inference_active_prefix_direct_layer_cache_update` flag.
- Enable it only inside the active-prefix decode-token context.
- Leave prefill, cross-attention, non-active-prefix generation, server defaults, and the retained compile-only baseline unchanged.
- Route one-token logits gates through the same context so the exactness gate exercises the candidate path.

## Gates

- Job: `49160690`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti, driver `595.71.05`
- Stack: Python `3.10.12`, torch `2.10.0+cu128`, Transformers `4.57.3`
- Base commit: `fc735f5` plus dirty candidate patch
- Run dir: `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5`
- Slurm status: `COMPLETED`, exit `0:0`, elapsed `00:06:02`

One-token logits gates:

| gate | report | result |
| --- | --- | --- |
| compile disabled | `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5/one_token_compile_false.json` | PASS, `max_abs=0.0`, top-k match |
| compile enabled | `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5/one_token_compile_true.json` | PASS, `max_abs=2.2888e-05`, top-k match |

15s smoke profiles:

| run | profile | main tokens | main model time | main tok/s | token equivalence |
| --- | --- | ---: | ---: | ---: | --- |
| cold compile-only | `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5/cold_compile/beatmap1112762b2ad943d484e8c30310096e40.osu.profile.json` | `1,084` | `22.406s` | `48.380` | baseline |
| active512 old path | `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5/active512_old/beatmapbf98ae9b69634f79b038345db1d18c1d.osu.profile.json` | `1,084` | `30.906s` | `35.074` | PASS vs compile-only |
| active512 direct-layer | `/work/imt11/Mapperatorinator/runs/direct-layer-cache-49160690-fc735f5/active512_direct_layer/beatmap0590740f79d84f97a48ab1dc47ec4e0a.osu.profile.json` | `1,084` | `31.330s` | `34.599` | PASS vs compile-only and old active512 |

Comparison:

- Direct-layer vs old active512: `35.074 -> 34.599 tok/s`, `-1.4%`.
- Direct-layer vs compile-only: `48.380 -> 34.599 tok/s`, `-28.5%`.
- First long map window did not improve: old active512 `seq3=26.539s`, direct-layer `seq3=26.926s`.
- Post-warm window did not improve: old active512 `seq9=1.574s`, direct-layer `seq9=1.599s`.

## Decision

Rejected and reverted. Bypassing `Cache.update` dispatch by calling the existing layer update directly did not reduce the measured cold first-window active-prefix overhead. It likely moved the relevant guards to the layer update frame or left the dominant compile/capture specialization elsewhere.

Do not retry this exact per-layer cache-update dispatch bypass without new TorchDynamo/TORCH_LOGS evidence showing `Cache.update` dispatch itself, rather than the layer update or broader graph capture, is still target-sized.
