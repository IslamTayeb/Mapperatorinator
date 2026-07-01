# Post-Warmup Main-Generation Range Trace on RTX 2080 Ti

## Summary

This note records a diagnostic torch-profiler trace for a post-warmup 15s SALVALAI map window after the profiler JSON event sorting was changed to keep high-total CUDA semantic ranges. This is not a throughput claim. The profiled window was heavily inflated by `torch.profiler`; use the event mix and call counts only.

## Inputs

- Commit: `6b56484`
- Job: `49139453`
- Node/GPU: `dcc-core-ferc-s-z25-21`, NVIDIA GeForce RTX 2080 Ti, driver `595.71.05`, capability `7.5`
- Config: `configs/inference/profile_salvalai_smoke15.yaml`
- Key flags: `precision=fp32`, `attn_implementation=sdpa`, `use_server=false`, `inference_generation_compile=true`, `profile_torch_generation=true`, `profile_torch_generation_label_filter=main_generation.seq9`, `profile_generation_detail_ranges=true`, `profile_torch_event_limit=300`
- Profile: `/work/imt11/Mapperatorinator/runs/smoke15-detail-sort-seq9-49139453-6b56484/beatmapb9132188521242dcafc4ac392113007f.osu.profile.json`
- Trace: `/work/imt11/Mapperatorinator/runs/smoke15-detail-sort-seq9-49139453-6b56484/torch_profiles/000_generation_main_generation_seq9.trace.json`

## Profiler Overhead

The selected record was `main_generation.seq9`, `234` generated tokens. The traced outer wall time was `109.597s` and the torch profile wall was `70.941s`, so this run must not be used for throughput. Earlier untraced seq9 runs on the same smoke slice were about `2.25s` for the same token count.

## Event Mix

Important JSON event rows:

| Event group | Calls | CUDA total | CPU self | Notes |
| --- | ---: | ---: | ---: | --- |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm75` | 5,628 | `2032.006ms` | `0ms` | Dominant true CUDA kernel family |
| `mapperatorinator.attention.layer*.self.sdpa` | 2,820 | `1464.362ms` | `97.320ms` | Self-attention SDPA dominates semantic decoder ranges |
| `mapperatorinator.attention.layer*.cross.sdpa` | 2,808 | `574.713ms` | `91.346ms` | Cross-attention is smaller but still material |
| `mapperatorinator.decoder.layer*.self_attn` | 2,808 | `1593.725ms` | `450.651ms` | Top-level self-attn range; nested with SDPA/projections |
| `mapperatorinator.decoder.layer*.cross_attn` | 2,808 | `628.165ms` | `361.519ms` | Top-level cross-attn range |
| `mapperatorinator.decoder.layer*.mlp.fc1/fc2` | 11,232 | `410.140ms` | `271.630ms` | Material, but smaller than attention |
| GEMV/GEMM/addmm kernel group | 33,906 | `495.198ms` | `704.811ms` | Many small one-token linear/GEMV launches |
| `aten::_foreach_copy_` / `copy_` / `index_copy_` / `cudaMemcpyAsync` / `transpose` | 57,897 | `46.230ms` | `679.258ms` | CPU/launch bookkeeping visible; CUDA copy time is small |
| `aten::multinomial` + `aten::sort` | 468 | `20.784ms` | `17.172ms` | Sampling still not target-sized |

The JSON also showed `TorchDynamo Cache Lookup` at `20,504` calls and `483.616ms` self CPU, `cudaLaunchKernel` at `70,588` calls and `642.307ms` self CPU, and `cudaGraphLaunch` at `16,310` calls and `215.210ms` self CPU. Those CPU numbers are profiler-inflated, but they support the hypothesis that graph fragmentation and many tiny compiled regions are still worth auditing.

## Compile Health Clue

The stderr log included:

```text
torch._dynamo hit config.recompile_limit (8)
function: 'update' (.../transformers/cache_utils.py:742)
last reason: layer_idx == 7
```

This points at HF `StaticCache.update(..., layer_idx, ...)` specialization by decoder layer. That does not prove a speed bug by itself, but it makes the next `TORCH_LOGS=recompiles,cudagraphs` compile-health audit higher priority than another backend toggle.

The follow-up compile-health smoke job was `49139485` on `dcc-core-ferc-s-z25-20`, commit `6b56484`, no torch profiler. It reproduced the steady-state shape:

- Profile: `/work/imt11/Mapperatorinator/runs/smoke15-compile-health-49139485-6b56484/beatmap59842e3c729249abaefcd7f52ce2b8a7.osu.profile.json`
- Main generation: `1,084` tokens, `21.541s`, `50.323 tok/s`.
- Post-warmup `seq9`: `234` tokens, `2.249s`, `104.053 tok/s`.
- Stderr showed CUDA graph recordings for the timing path and map windows plus two `modeling_mapperatorinator.py:139` recompiles caused by `decoder_input_ids` stride changes and the timing-model to map-model `proj_out.weight` size change.

That means automatic compile/CUDA-graph handling still records multiple graph variants, but the recurring evidence is not yet a clean main-generation bottleneck.

## Dynamo Cache-Limit Scout

PyTorch's recompilation docs say `TORCH_LOGS=recompiles` exposes recompilation reasons, and that hitting the bounded Dynamo cache limit can skip future compilation attempts; if the number of variants has a reasonable constant bound, raising the limit is a valid scout. The DCC PyTorch build reported:

- `torch 2.10.0+cu128`
- `torch._dynamo.config.recompile_limit = 8`
- `torch._dynamo.config.cache_size_limit = 8`
- `torch._dynamo.config.accumulated_cache_size_limit = 256`

Scout job `49139564` set `recompile_limit=32` and `cache_size_limit=32` before importing `inference.py`:

- Profile: `/work/imt11/Mapperatorinator/runs/smoke15-dynamo-cache32-49139564-6b56484/beatmap731abf43b14e48d8b8bfb5c266d9f125.osu.profile.json`
- Main generation: `1,084` tokens, `21.310s`, `50.867 tok/s`.
- Token equivalence: PASS, `1,084 / 1,084`.
- Delta vs smoke reference job `49139323`: `50.850 -> 50.867 tok/s`, `+0.0%`.

Decision: reject Dynamo cache-limit tuning as a main-generation optimization. It may reduce timing-context compile overhead in some runs, but it does not move the retained main-generation target and should not be promoted.

## Interpretation

- Attention is target-sized. Self + cross SDPA account for about `2.04s` of CUDA total in the selected traced window, matching the previous raw FMHA kernel total and dwarfing sampling/logits overhead.
- Attention alone is probably not enough unless a replacement is substantially better for SM75 `q_len=1` decode. A narrow attention/cache-layout kernel could be meaningful, but a small sampler or copy cleanup cannot move full-song throughput toward `200 tok/s`.
- The next exact-runtime work should build on the one-token gate: extract a reusable logits-only direct-step ABI, then test manual CUDA graph or narrow `torch.compile(mode="reduce-overhead")` around that stable step.
- Native CUTLASS or other kernel work should wait until the direct-step ABI and compile-health audit identify one isolated operation with enough ceiling.

## Direct-Step ABI Extraction

Commit `8cb1160` extracted the static-cache one-token path into `osuT5.osuT5.inference.direct_decode` and rewired `utils/verify_one_token_decode.py` to use it. The helper is intentionally logits-only and does not replace HF sampling, EOS handling, RNG, or production `model.generate`.

DCC validation job `49139917`, node `dcc-core-ferc-s-z25-20`, commit `8cb1160`:

| Mode | Report | Result | Notes |
| --- | --- | --- | --- |
| `inference_generation_compile=false` | `/work/imt11/Mapperatorinator/runs/one-token-abi-gate-49139917-8cb1160/one_token_decode_seq9_compile_false.json` | PASS, `max_abs=0.0`, top-k match | Candidate prepared shape `[1, 1]`, prefill shape `[1, 84]` |
| `inference_generation_compile=true` | `/work/imt11/Mapperatorinator/runs/one-token-abi-gate-49139917-8cb1160/one_token_decode_seq9_compile_true.json` | PASS, `max_abs=2.2888e-05`, top-k match | Candidate prepared shape `[1, 1]`, prefill shape `[1, 84]` |

Decision: keep the ABI extraction as infrastructure. It is not a speed claim, but it is the required launch point for manual CUDA graph and direct-step compile experiments.

## Next

- Run a no-profiler compile/CUDA-graph health audit with `TORCH_LOGS=recompiles,cudagraphs` on the 15s smoke slice.
- Start the direct one-token ABI extraction from `utils/verify_one_token_decode.py` and then prototype manual CUDA graph replay around the exact candidate step.
- Keep static-cache update rewrites as a later experiment only if direct-step graphing shows `StaticCache.update` remains a measured blocker. Raising Dynamo limits alone was not enough.
