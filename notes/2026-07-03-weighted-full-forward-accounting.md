# Weighted Full-Forward Accounting

## Purpose

Close the main remaining accounting gap after the accepted native q1 self-attention path. Previous probes measured individual islands at one active-prefix length, but full-song active64 generation uses many bucket lengths. This pass weights the one-token CUDA-graph replay cost by the actual full-song main-generation replay counts.

Current accepted full-song single-song baseline remains DCC job `49225493`: `237.111 tok/s`, `7,639` main tokens, `32.217s` synchronized model time, token equivalence PASS for main/timing, and byte-identical `.osu` output.

## Runs

Whole one-token model-forward bucket probe:

- DCC job: `49229391`
- Commit: `e70b871`
- Run dir: `/work/imt11/Mapperatorinator/runs/full-forward-buckets-20260703-072500-e70b871`
- Utility: `utils/profile_decode_full_forward_island.py`
- Result: every bucket PASS, `cuda_graph_replay_max_abs=0.0`

Decoder-layer bucket probe:

- DCC job: `49229433`
- Commit: `e70b871`
- Run dir: `/work/imt11/Mapperatorinator/runs/decoder-layer-buckets-20260703-073511-e70b871`
- Utility: `utils/profile_decode_decoder_layer_island.py`
- Result: every bucket PASS, `logits_replay_max_abs=0.0`

Both probes used the accepted opt-in stack:

```text
inference_generation_compile=true
inference_active_prefix_decode_loop=true
inference_active_prefix_decode_bucket_size=64
inference_active_prefix_decode_cuda_graph=true
inference_active_prefix_decode_cuda_graph_warmup=0
inference_active_prefix_decode_cuda_graph_min_decode_steps=1
inference_stateful_monotonic_logits_processor=true
inference_q1_bmm_cross_attention=true
inference_decode_session_runtime=true
inference_decode_session_cuda_graph=true
inference_native_decode_kernels=true
inference_native_q1_self_attention=true
attn_implementation=sdpa
precision=fp32
use_server=false
parallel=false
cfg_scale=1.0
num_beams=1
seed=12345
```

Replay counts came from full-song diagnostic job `49228590`:

| prefix | decode replays |
| ---: | ---: |
| 128 | `22` |
| 192 | `64` |
| 256 | `126` |
| 320 | `227` |
| 384 | `136` |
| 448 | `332` |
| 512 | `682` |
| 576 | `1,727` |
| 640 | `2,907` |
| 704 | `1,221` |
| 768 | `108` |

Total weighted decode replays: `7,552`.

## Whole Model Forward

Weighted one-token `model(**prepared_inputs)` CUDA-graph replay:

| prefix | graph ms/step | weighted full-song seconds |
| ---: | ---: | ---: |
| 128 | `2.258145` | `0.049679` |
| 192 | `2.386698` | `0.152749` |
| 256 | `2.396282` | `0.301931` |
| 320 | `2.407583` | `0.546521` |
| 384 | `2.518241` | `0.342481` |
| 448 | `2.618655` | `0.869393` |
| 512 | `2.587013` | `1.764343` |
| 576 | `2.641059` | `4.561109` |
| 640 | `2.682011` | `7.796607` |
| 704 | `2.780457` | `3.394939` |
| 768 | `2.873576` | `0.310346` |

Weighted total: `20.090s`, or `62.4%` of accepted full-song model time. If the entire one-token model forward were free, the idealized ceiling would be about `630 tok/s`.

## Decoder Layers

Weighted full decoder-layer CUDA-graph replay:

| prefix | graph ms/layer | weighted full-song seconds |
| ---: | ---: | ---: |
| 128 | `0.198112` | `0.052302` |
| 192 | `0.176416` | `0.135487` |
| 256 | `0.179865` | `0.271956` |
| 320 | `0.227666` | `0.620162` |
| 384 | `0.188077` | `0.306942` |
| 448 | `0.192877` | `0.768421` |
| 512 | `0.209595` | `1.715328` |
| 576 | `0.202291` | `4.192283` |
| 640 | `0.205892` | `7.182354` |
| 704 | `0.231755` | `3.395679` |
| 768 | `0.213642` | `0.276880` |

Weighted total: `18.918s`, or `58.7%` of accepted full-song model time. If every one-token decoder layer were free, the idealized ceiling would be about `574 tok/s`.

The model-forward minus decoder-layer residual is only `1.172s`, about `3.6%` of model time. That bucket includes embedding, mask/model wrapper work, decoder final norm, output projection, and other non-layer work inside the graph-replayed model call. It is below the normal standalone implementation threshold.

## Interpretation

This pass removes several tempting but weak targets:

- Decoder final norm/output projection/model wrapper work inside the graph is not target-sized by itself (`~1.17s` combined residual).
- Output projection alone was already measured at roughly `0.185s` full-song in the linear graph replay probe.
- Standalone sampling/top-p tail CUDA time projected to only `1.568s`, just below the 5% threshold and high-risk for exact RNG.
- Duplicate graph capture and prefill-forward diagnostics are below threshold.

The remaining path toward `500 tok/s` is therefore not a cleanup pass. With `32.217s` current model time and `15.278s` needed for `500 tok/s`, the model needs to save about `16.94s`. If all non-layer time stayed fixed, decoder-layer time would need to drop from `18.918s` to about `1.979s`, an `~89.5%` decoder-layer reduction. Even if some outside time is also improved, the main lever has to be broad decoder-layer/native runtime work, not isolated linears, final projection, sampling, or graph-cache bookkeeping.

## Weighted Self-Attention Split

Follow-up DCC job `49229477` on commit `5ceffa3` ran `utils/profile_decode_self_attention_island.py --cuda-graph-replay` across the same active-prefix buckets:

- Run dir: `/work/imt11/Mapperatorinator/runs/self-attn-buckets-20260703-074301-5ceffa3`
- Result: every bucket PASS, `logits_replay_max_abs=0.0`

Weighted graph-replay totals:

| self-attention component | weighted full-song seconds | model-time share |
| --- | ---: | ---: |
| repo self-attention module | `8.806s` | `27.3%` |
| manual native island | `8.826s` | `27.4%` |
| pre-attention setup only | `4.378s` | `13.6%` |
| native q1 attention only | `3.075s` | `9.5%` |
| output projection only | `0.630s` | `2.0%` |

The repo module and manual native island are essentially tied, so the Python/manual decomposition is not itself an optimization. The important split is that qkv/RoPE/cache/setup remains larger than the native attention kernel. Freeing the entire weighted self-attention module would raise the accepted baseline only to about `326 tok/s`; it is a good target, but not sufficient for `500 tok/s` alone.

## Decision

Use `utils/profile_decode_full_forward_island.py` plus bucket weighting before any future broad runtime/kernel rewrite. The next implementation-class project should target a large decoder-layer island or multi-layer decoder stack with stable C++/CUDA/CUTLASS/cuBLASLt work, and it should show an exclusive projected saving of at least `1.6s`, preferably several seconds, before production integration.

Do not start standalone work on:

- decoder final norm/projection;
- graph wrapper/model-forward residual;
- top-p/sampling fusion;
- duplicate graph capture;
- simple PyTorch call-form linears.

## Verification

- Local syntax check passed for `utils/profile_decode_full_forward_island.py`.
- DCC jobs `49229391` and `49229433` completed successfully.
- DCC job `49229477` completed successfully.
- All captured bucket probes had exact replay logits (`max_abs=0.0`).
