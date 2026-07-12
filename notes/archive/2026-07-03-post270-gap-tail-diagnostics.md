# Post-270 Gap And Tail Diagnostics

## Purpose

Re-profile the accepted `270.475 tok/s` exact single-song opt-in stack before
starting another implementation branch. The goal was to explain the remaining
gap to `500 tok/s`, not to claim a speedup.

Accepted baseline entering this pass:

- Job: `49230082`
- Main tokens: `7,639`
- Full-song synchronized main model time: `28.243s`
- Throughput: `270.475 tok/s`
- Equivalence: main/timing token identity PASS and byte-identical `.osu`

For SALVALAI, `500 tok/s` means about `15.278s`, so the remaining required
reduction is about `12.965s`.

## Environment

- DCC jobs: `49232331`, `49232344`
- Commit: `840dbc4915daf6df4d24f589ff3388af571c350c`
- Branch: `main`
- Node/GPU: `dcc-core-ferc-s-z25-21`, RTX 2080 Ti
  `GPU-825b182c-b59e-7d16-c8ec-6084dc8199b8`
- Driver: `595.71.05`
- Torch: `2.10.0+cu128`
- Transformers: `4.57.3`
- Precision/backend: `fp32`, `attn_implementation=sdpa`

Accepted opt-in stack:

- `inference_generation_compile=true`
- `inference_active_prefix_decode_loop=true`
- `inference_active_prefix_decode_bucket_size=64`
- `inference_active_prefix_decode_cuda_graph=true`
- `inference_active_prefix_decode_cuda_graph_warmup=0`
- `inference_active_prefix_decode_cuda_graph_min_decode_steps=1`
- `inference_stateful_monotonic_logits_processor=true`
- `inference_q1_bmm_cross_attention=true`
- `inference_decode_session_runtime=true`
- `inference_decode_session_cuda_graph=true`
- `inference_native_decode_kernels=true`
- `inference_native_q1_self_attention=true`
- `inference_native_q1_rope_cache_self_attention=true`

## Job 49232331: Smoke Gap Diagnostic

Run dir:

```text
/work/imt11/Mapperatorinator/runs/post270-gap-diag2-49232331-840dbc4
```

Artifacts:

- `control.profile.json`
- `active_diag.profile.json`
- `control_vs_active_diag.main.json`
- `control_vs_active_diag.timing.json`
- `full_forward_island_seq9.json`
- `decoder_layer_island_seq9.json`

### Smoke Control Versus Diagnostic

The diagnostic run enables `profile_active_prefix_decode_diagnostics=true` and
`profile_nvtx_generation_ranges=true`, so timing is diagnostic-only.

| run | main tokens | main model s | main tok/s | timing tokens | timing model s | timing tok/s | stage wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | `1,084` | `3.726830` | `290.864` | `164` | `2.964689` | `55.318` | `13.177954` |
| diagnostic | `1,084` | `3.923744` | `276.267` | `164` | `2.798995` | `58.592` | `11.492774` |

Token equivalence passed for both main (`1,084 / 1,084`) and timing
(`164 / 164`). Do not treat the diagnostic timing changes as a throughput win or
loss; the ranges add overhead and change synchronization.

### Active-Prefix Diagnostic Counters

Main generation over the 15s smoke:

| counter | value |
| --- | ---: |
| decode steps | `1,074` |
| CUDA graph captures | `24` |
| CUDA graph replays | `1,074` |
| buckets seen | `128, 192, 256, 320, 384, 448, 512, 576, 640, 704` |
| `decode_forward.cuda_graph` event sum | `2.395082s` |
| `prepare_inputs` event sum | `0.424856s` |
| `stopping_criteria` event sum | `0.250258s` |
| `prefill_forward` event sum | `0.220507s` |
| `logits_processor` event sum | `0.102653s` |
| `sampling.multinomial` event sum | `0.044228s` |

The diagnostic counters still show real cost both inside the one-token forward
and around the generation loop. `prepare_inputs` looks target-sized in this
instrumented profile, but the fixed production fast-prepare retry already failed
full-song promotion, so this is not a reason to re-add that flag.

### Full-Forward Island

`utils/profile_decode_full_forward_island.py` at seq9/prefix128:

| metric | value |
| --- | ---: |
| pass | `true` |
| active prefix length | `128` |
| CUDA graph replay | `1.8664055ms/call` |
| projected full-song replay time | `14.095094s` |
| fraction of accepted model time | `49.9%` |
| outside isolated full-forward boundary | `14.147906s` |
| ideal TPS if isolated full-forward were free | `539.939 tok/s` |

This is the clearest result from this pass: after the fused RoPE/cache win, the
accepted model time is roughly half one-token model-forward replay and half
outside that isolated replay boundary.

### Decoder-Layer Island

`utils/profile_decode_decoder_layer_island.py --candidate-decoder-runtime-island`
at seq9/prefix128:

| component | projected full-song CUDA graph replay |
| --- | ---: |
| whole decoder layer stack | `13.018426s` |
| self-attention residual segment | `4.167457s` |
| cross-attention residual segment | `3.272446s` |
| MLP residual segment | `4.222353s` |
| residual segment unexplained/glue | `1.356170s` |
| manual decoder-runtime island | `12.980547s` |
| manual island projected saving | `0.037879s` |

The manual Python-layer island is an exact calibration boundary, not an
optimization. It does not materially reduce runtime.

## Job 49232344: Direct-Loop Tail Diagnostic

Run dir:

```text
/work/imt11/Mapperatorinator/runs/post270-tail-diag-49232344-840dbc4
```

Artifact:

```text
/work/imt11/Mapperatorinator/runs/post270-tail-diag-49232344-840dbc4/tail_diagnostics.json
```

Gate result:

- `pass=true`
- generated-token identity PASS
- raw-logit/top-k gate PASS
- final CPU/CUDA RNG state PASS
- stop reason matched: `max_new_tokens`
- sampled steps: `256`

CUDA-event tail timings, excluding the verifier raw-logit capture hook and the
non-exclusive `logits_processor.total` bucket:

| component | us/token |
| --- | ---: |
| `TopPLogitsWarper` | `64.596` |
| `torch.multinomial` | `44.182` |
| `MonotonicTimeShiftLogitsProcessor` | `37.604` |
| `stopping_criteria` | `23.646` |
| `eos_mask` | `9.635` |
| `finished_check` | `8.496` |
| `softmax` | `6.655` |
| `append_token.cat` | `4.829` |
| `logits_extract` | `4.191` |
| `TemperatureLogitsWarper` | `3.994` |

Production-like CUDA tail sum:

```text
~207.8 us/token * 7,639 tokens ~= 1.59s
```

Fantasy zero-tail ceiling:

```text
28.243s - 1.59s ~= 26.65s
7,639 / 26.65s ~= 286 tok/s
```

CPU wall summed to about `~1.07ms/token`, but these spans are not exclusive
synchronized model time. They include Python timing, launches, asynchronous work,
and synchronization effects.

## Decision

No optimization graduated.

These diagnostics reject three tempting shortcuts:

1. Manual Python-layer decoder island rewrites. The exact manual island saves
   only `0.038s` projected full-song time.
2. Standalone tail/logits/sampling fusion as the main path. Even a fantasy
   zero-tail result only reaches about `286 tok/s`.
3. More fast-prepare cleanup. `prepare_inputs` remains visible in diagnostics,
   but the full-song fixed fast-prepare production retry regressed.

The next plausible major path is broader and harder: a `DecodeSession`/native
decoder-layer runtime that reduces both one-token decoder compute and the
outside-forward runtime/control boundary while preserving exact token IDs, logits
top-k, RNG state, EOS/stopping behavior, and byte-identical output.

Ranked next probes:

1. A fused/native self-attention residual island that includes norm, `Wqkv`,
   fused RoPE/cache attention, output projection, and residual. It needs to beat
   the existing `4.167s` self-attention segment by enough to clear the `1.41s`
   full-song keep threshold.
2. A fused/native MLP residual island. The current MLP segment is `4.222s`; a
   candidate needs a broad win, not another narrow `fc1+GELU` kernel.
3. A fused/native cross-attention residual island. It is useful for future
   encoder-decoder work but smaller (`3.272s`) and already has the q1 BMM win.
4. Tail CUDA graph work only as part of a broader runtime island. A one-token
   tail graph is likely exact but too small; fixed multi-token graphs still need
   an EOS/RNG audit before they can be considered equivalent.
