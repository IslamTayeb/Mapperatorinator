# Direct CUDA Graph Decode Gate

## Summary

Added and validated a verifier-only CUDA graph path inside `utils/verify_direct_decode_loop.py`. This is not an inference speed win and should not be reported as one. It is a correctness gate proving that a manual CUDA graph replay can advance through multiple sampled one-token decode steps under active-prefix bucket512 without changing generated tokens, raw-logit agreement, top-k order, or final RNG state.

## Why This Matters

The earlier fixed-step graph profile proved that one prepared decoder step could be captured and replayed, but that did not answer whether a graph-backed decode loop could handle changing generated tokens, changing `cache_position`, updated masks, sampling, logits processors, EOS checks, and RNG accounting.

This gate is narrower than production inference, but it closes the next correctness question: copy the current prepared one-token inputs into static graph buffers, replay the captured model forward, then leave sampling and stopping outside the graph so HF generation semantics remain comparable.

## Implementation

Initial graph gate commit: `0602d32`
Bucket-recapture verifier commit: `13335e5`

New verifier flags:

```bash
python utils/verify_direct_decode_loop.py \
  --candidate-active-prefix-decode \
  --candidate-active-prefix-decode-bucket-size 512 \
  --candidate-cuda-graph-forward \
  --candidate-cuda-graph-warmup 3
```

The graph path:

- Captures only the one-token model forward after the candidate decode step prepares static-shape inputs.
- Copies later prepared inputs into captured static buffers before each replay.
- Caches captured graphs by active-prefix length and model-input signature, so longer gates can recapture when bucket512 becomes bucket1024.
- Keeps logits processors, sampling, EOS handling, and RNG state checks outside the graph.
- Records graph diagnostics in the JSON report.

## DCC Validation

- Job: `49165810`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti allocation
- Commit: `0602d32`
- Config: `profile_salvalai_smoke15`, sequence index `9`, `max_new_tokens=8`, `seed=12345`, `attn_implementation=sdpa`, `use_server=false`, active-prefix bucket `512`
- Run dir: `/work/imt11/Mapperatorinator/runs/direct-graph-gate-49165810-0602d32`
- Logs: `/work/imt11/Mapperatorinator/logs/direct-graph-gate-49165810.out` and `.err`
- Slurm state: `COMPLETED`, exit code `0:0`

All four gates generated the same token IDs:

```text
[12, 1648, 2242, 2717, 3915, 4012, 4053, 32]
```

| Gate | Token match | RNG match | Logits allclose | Top-k match | Max abs | Graph capture | Graph replays | Wall |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| direct active512, compile false | PASS | PASS | PASS | PASS | `0.0` | n/a | n/a | `17.989s` |
| direct active512, compile true | PASS | PASS | PASS | PASS | `0.0` | n/a | n/a | `47.984s` |
| graph active512, compile false | PASS | PASS | PASS | PASS | `0.0` | `0.0541s` | `7` | `7.415s` |
| graph active512, compile true | PASS | PASS | PASS | PASS | `1.068e-4` | `0.0537s` | `7` | `18.623s` |

The compile-enabled graph path had small fp32-level logit drift within the configured tolerance. Top-k ordering and generated tokens still matched.

## Interpretation

This is a useful direct-runtime correctness milestone. It shows the active-prefix graph idea is not immediately invalidated by token progression, cache-position updates, logits processors, or RNG accounting in the sampled direct-loop gate.

It is not a throughput claim:

- The gate is only 8 decode tokens.
- It uses one sequence/window and does not cross active-prefix bucket changes.
- The graph path still calls `prepare_inputs_for_generation()` and copies tensors into static buffers each step.
- The wall time includes short-run verifier overhead, model setup, compile/setup effects, and report generation.
- It has not run 15s smoke token equivalence or full-song non-regression.

## Decision

Keep the verifier infrastructure and use it before any production CUDA graph decode-loop attempt.

## Longer Gate Follow-Up

Follow-up job `49165980` repeated the graph replay gate at 64 generated tokens:

- Partition: `common`
- Node/GPU: `dcc-dhvimdcore-gpu-ferc-s-p15-10`, NVIDIA GeForce RTX 2080 Ti, compute capability `7.5`, driver `595.71.05`
- Commit: `26ec057`
- Config: `profile_salvalai_smoke15`, sequence index `9`, `max_new_tokens=64`, active-prefix bucket `512`
- Run dir: `/work/imt11/Mapperatorinator/runs/direct-graph-gate64-49165980-26ec057`
- Logs: `/work/imt11/Mapperatorinator/logs/direct-graph-gate64-49165980.out` and `.err`
- Slurm state: `COMPLETED`, exit code `0:0`

| Gate | Token match | RNG match | Logits allclose | Top-k match | Steps | Max abs | Graph capture | Graph replays | Wall |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| graph active512, compile false | PASS | PASS | PASS | PASS | `64` | `0.0` | `0.0770s` | `63` | `28.448s` |
| graph active512, compile true | PASS | PASS | PASS | PASS | `64` | `1.068e-4` | `0.0764s` | `63` | `70.252s` |

The retained generated-token sequence length was `64` in both cases, and both reference and candidate stopped because of `max_new_tokens`.

This strengthened the correctness signal but stayed inside one active-prefix bucket and still measured verifier overhead, not production throughput.

## Bucket Transition Follow-Up

Job `49166100` first attempted a 448-token transition gate on `profile_salvalai_smoke15`, sequence index `9`, but the generated sequence hit EOS at 353 tokens before crossing bucket512. It still passed both compile-disabled and compile-enabled graph gates with one captured graph:

| Gate | Token match | RNG match | Steps | Stop | Max abs | Capture count | Graphs | Wall |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| graph active512, compile false | PASS | PASS | `353` | EOS | `0.0` | `1` | `512:352` | `33.201s` |
| graph active512, compile true | PASS | PASS | `353` | EOS | `1.221e-4` | `1` | `512:352` | `59.609s` |

Run dir: `/work/imt11/Mapperatorinator/runs/direct-graph-transition-49166100-13335e5`.

The actual bucket-transition gate used full-song `profile_salvalai`, sequence index `0`, `max_new_tokens=512`:

- Compile-disabled job: `49166191`, node `dcc-dhvimdcore-gpu-ferc-s-p15-12`, commit `13335e5`, run dir `/work/imt11/Mapperatorinator/runs/direct-graph-transition-seq0-49166191-13335e5`
- Compile-enabled job: `49166213`, node `dcc-dhvimdcore-gpu-ferc-s-p15-12`, commit `13335e5`, run dir `/work/imt11/Mapperatorinator/runs/direct-graph-transition-seq0-compile-49166213-13335e5`

| Gate | Token match | RNG match | Logits allclose | Top-k match | Steps | Stop | Max abs | Capture count | Graphs | Wall |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| graph active512, compile false | PASS | PASS | PASS | PASS | `512` | max_new_tokens | `0.0` | `2` | `512:414`, `1024:97` | `66.538s` |
| graph active512, compile true | PASS | PASS | PASS | PASS | `512` | max_new_tokens | `1.373e-4` | `2` | `512:414`, `1024:97` | `82.194s` |

This proves the verifier can recapture a second CUDA graph when the active-prefix bucket changes from `512` to `1024`, while preserving generated-token identity, final RNG state, logits allclose, and top-k order in both compile modes.

It is still not a speed claim. The verifier continues to run `prepare_inputs_for_generation()` and tensor copies outside the graph; the wall times are gate overhead and should not be compared to `profile_inference` throughput.

Next useful checks:

1. Prototype an opt-in smoke-only direct graph loop that reports untraced `profile_inference` model time.
2. Keep graph caching keyed by active-prefix bucket/signature and count capture cost honestly in cold runs.
3. Keep active-prefix default-off and do not claim speed until 15s smoke and then full-song SALVALAI runs pass token equivalence and no-regression gates.
