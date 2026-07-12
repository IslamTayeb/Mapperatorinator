# Multistep Tail Graph Verifier

## Summary

Keep the verifier-only multistep tail graph utility from
`experiment/multistep-tail-graph-verifier`.

This is not a production inference speedup and does not change the retained
`270.475 tok/s` full-song single-song baseline. It proves that a fixed-start
four-step tail CUDA graph can preserve the exact sampled tokens, output buffer,
continue/stopping state, final CUDA RNG state, and next eager CUDA samples for
the current fastest opt-in stack.

The result strengthens the case for a future DecodeSession-owned tail/runtime
island, but production work still needs exact EOS behavior across variable
generation lengths and full-song throughput validation.

## Evidence

- Branch: `experiment/multistep-tail-graph-verifier`
- Commit: `c35dbe7`
- DCC job: `49246243`
- DCC report path:
  `/work/imt11/Mapperatorinator/runs/tail-graph-multistep-20260704-101951-c35dbe7/tail_graph_multistep.json`
- Node/GPU: `dcc-core-ferc-s-z25-20`, RTX 2080 Ti,
  GPU UUID `GPU-ba708720-cba9-538c-6e0f-ecaea3486d09`
- Driver/CUDA stack: driver `595.71.05`, torch `2.10.0+cu128`,
  transformers `4.57.3`, CUDA device capability `(7, 5)`
- Config: `profile_salvalai_smoke15`, `sequence_index=9`,
  `max_new_tokens=256`, seed `12345`, fp32, SDPA, `use_server=false`,
  accepted opt-in stack with active-prefix bucket64 CUDA graph,
  DecodeSession graph/cache reuse, stateful monotonic, q1 BMM cross-attention,
  native q1 self-attention, and fused RoPE/cache self-attention

Command-specific verifier flags:

- `--buffered-input-ids`
- `--tail-graph-ceiling`
- `--tail-graph-multistep-ceiling`
- `--tail-graph-step-index 64`
- `--tail-graph-multistep-steps 4`

## Result

Top-level gates passed:

- generated-token identity: PASS
- final RNG-state identity: PASS
- raw-logit/top-k gate: PASS
- stop reason: `max_new_tokens` for reference and candidate
- captured raw-logit steps: `256 / 256`

One-step tail graph:

| Metric | Value |
| --- | ---: |
| Pass | `true` |
| Eager exact tail | `1.0488226 ms/token` |
| CUDA graph replay tail | `0.1875217 ms/token` |
| Speedup | `5.593x` |
| Saved | `861.301 us/token` |
| Projected SALVALAI main-token ceiling | `6.579s` over `7,639` tokens |

Four-step tail graph:

| Metric | Value |
| --- | ---: |
| Pass | `true` |
| Steps | `4` |
| Start length | `148` |
| Eager exact tail | `3.8633123 ms/block`, `0.9658281 ms/token` |
| CUDA graph replay tail | `0.6514345 ms/block`, `0.1628586 ms/token` |
| Speedup | `5.930x` |
| Saved | `802.969 us/token` |
| Projected SALVALAI main-token ceiling | `6.134s` over `7,639` tokens |

Four-step correctness details:

- eager, graph, and actual generated tokens matched:
  `[2143, 2631, 3915, 3992]`
- output buffer matched
- unfinished state matched
- continue values matched: `[1, 1, 1, 1]`
- no stop occurred before the fixed block end
- final RNG state matched actual eager generation
- next eager CUDA samples matched after graph replay:
  `[3813, 1413, 2894, 1306, 3910, 3117, 3923, 683]`

## Interpretation

The one-step and four-step reports agree on target size: the exact tail around
logits processors, top-p/temperature sampling, EOS masking, append, and stopping
has a roughly `6s` full-song ceiling if graph replay could replace the current
production-like eager tail.

This is still target sizing only:

- the verifier uses captured raw logits, not a production forward path;
- fixed-start graph replay does not yet solve variable-length generation;
- fixed-K graph replay is non-equivalent if EOS happens before the block ends
  unless the runtime implements exact rollback or true device-side early exit;
- no untraced `profile_inference` full-song throughput was measured;
- no generated `.osu` output claim is made from this verifier.

## Cold-Start Note

This verifier measures post-capture replay potential and should not be used to
hide cold-start costs. Future tail/runtime candidates must report cold
single-song first-window behavior separately from warmed graph replay. Useful
cold-start fields include first-window compile/capture time, graph-cache setup,
model/load cache state, full-song all-window throughput, and output hash
equivalence.

## Decision

Keep the verifier-only infrastructure and merge it after documentation. Do not
claim a TPS win.

Next exact path:

1. Use this verifier before any production tail CUDA graph or sampler/runtime
   island.
2. Add an exact production candidate only if it preserves generated-token
   identity, raw-logit/top-k equality, final RNG-state identity,
   EOS/stopping behavior, generated-token counts, and output bytes.
3. Validate cold 15s smoke, cold full-song SALVALAI, and any warmed-repeat mode
   separately before promotion.
