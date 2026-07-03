# Compiled Decoder-Layer Probe

## Purpose

Test a cheap PyTorch-first idea before writing broader native decoder-runtime code: compile the captured one-token decoder-layer island with `torch.compile(mode="reduce-overhead")` and compare it against the current graph-replayed decoder layer.

This is diagnostic-only. It is not an inference throughput claim.

## Run

- DCC job: `49231529`
- Commit: `25d18e2`
- Node/GPU: see preflight under the run dir
- Run dir: `/work/imt11/Mapperatorinator/runs/compiled-decoder-layer-probe-49231529-25d18e2`
- Report: `/work/imt11/Mapperatorinator/runs/compiled-decoder-layer-probe-49231529-25d18e2/compiled_decoder_layer_probe.json`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`, active prefix forced to `640`
- Stack: current exact fused opt-in path, including DecodeSession, native q1 self-attention, and fused RoPE/cache self-attention

The Slurm job is marked `FAILED` because the final shell summary used an unexported `RUN_DIR`. The JSON report was written successfully and is the relevant artifact.

## Result

Correctness for the diagnostic replay passed:

- probe pass: `true`
- logits replay max abs: `0.0`
- compiled decoder-layer output allclose: `true`
- compiled max abs: `1.43e-06`

Timings at active prefix `640`:

| variant | eager/event ms per layer | CUDA graph replay ms per layer | note |
| --- | ---: | ---: | --- |
| repo decoder layer | `0.805842` | `0.201039` | graph replay is the relevant runtime mode |
| compiled decoder layer | `0.644417` | n/a | graph replay failed |

The compiled layer could not be captured for CUDA graph replay:

```text
RuntimeError: Cannot prepare for replay during capturing stage. during CUDA graph capture.
```

Torch also warned that Dynamo cannot trace the current native pybind attention op:

```text
Dynamo does not know how to trace ... q1_rope_cache_attention
```

## Interpretation

This kills regional `torch.compile` as the next production path for the current fused native stack. The eager compiled layer was numerically close and somewhat faster before graph replay, but the accepted runtime depends on CUDA graph replay. A path that cannot capture cleanly is not useful unless the native attention op is first converted into a traceable PyTorch custom op or the whole decoder island moves under a stable native runtime.

The result is consistent with the broader evidence:

- current `decode_forward.cuda_graph` is the dominant CUDA-event bucket;
- per-linear and compiled MLP rewrites are too small;
- standalone sampling/tail work is not target-sized;
- the next plausible `>5%` path needs a broader `DecodeSession`/decoder-layer native island.

## Decision

Do not pursue `torch.compile(mode="reduce-overhead")` around the decoder layer in the current native-pybind stack.

Keep the diagnostic flag in `utils/profile_decode_decoder_layer_island.py` because it is useful for future traceability checks if native ops are registered as PyTorch custom ops. Do not promote it to production.

Next implementation-class work should start with a diagnostic backend sweep for a broad decoder-layer/runtime island:

- capture real one-token decoder-layer tensors through the existing verifier path;
- test cuBLASLt or CUTLASS/CUDA variants for fp32 one-token linears/MLP under CUDA graph replay;
- keep plain CUDA for RMSNorm/residual/cache/layout glue;
- require a weighted full-song projected saving comfortably above `1.4s` before any production flag.
