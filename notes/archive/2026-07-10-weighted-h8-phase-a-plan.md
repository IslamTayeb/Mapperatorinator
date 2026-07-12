# Weighted Real-Prefix Bucket-576 H8 Phase A

## Scope and prerequisite

This is a verifier-only physics scout. It consumes the fixed reviewed source
artifact from job `49559386`, commit `1896b19`, canonical source-contract SHA
`de55e2d16c4b085c5cce62e83be8cd4f83bb23f2a40e87f80be23a88f216b99a`.
It does not use the old fixed Hybrid-L2 report as a state prerequisite and it
does not authorize Phase B, a scheduler, runtime wiring, or server work.

The two rows are independently reachable seed-`12345` states from the accepted
Lambada transcript. Row 0 begins at prefix `520`, cache position `519`, and
samples accepted output offsets `42..49`. Row 1 advances one exact accepted
step and begins at prefix `521`, cache position `520`, sampling offsets
`43..50`. A second seed is intentionally not fabricated: the reviewed source
contract authorizes only the production seed-`12345` history. The staggered
states still have distinct prefixes, RNG states, cache positions, and tokens.

## Hypothesis and falsification

The changing-prefix bucket-128 verifier reached `621.893 tok/s` worst-order,
but the source-derived duplicated-ten queue requires a weighted complete B2
rate above `623.656675984333 tok/s`. Bucket `576` accounts for `1,099 / 5,905`
accepted decode pair steps and is the first production-weighted shape.

The hypothesis is that the existing five-graph Hybrid-B2 chain—two private
model graphs, one shared B2 processor graph, and two private sampling/stop
graphs—retains enough complete-step throughput at the real bucket-576 state to
clear the weighted requirement. A same-job serial private-B1 control uses two
additional stateful incremental processor graphs so its timed loop also avoids
per-step allocation, synchronization, and graph recapture.

Those private B1 graphs expose distinct captured monotonic-state input and
output buffers. Every replay copies output back to input inside the timed
interval; a separate untimed H8 ledger must match independent B1 processed
scores, probabilities, sampled tokens, RNG, monotonic state, and feedback state
at every step before either timing loop is allowed to run.

Stop before any broader bucket sweep if either reciprocal order fails accepted
tokens, stop behavior, per-step/final RNG, raw-logit allclose/top-k, processed
score bitwise/top-k, probability bitwise, static/prefix equality, active cache,
future suffix, cross cache, or source handoff. If exactness passes, run five
reset-separated H8 trials per reciprocal order. Continue only if:

1. worst reciprocal candidate complete-wall throughput is greater than
   `623.656675984333 tok/s`; and
2. both same-order gains over serial private B1 are greater than `5%`.

Raw-logit, active-self-cache, and cross-cache bitwise hashes are reported
separately. Only a pass for all three may be labeled
`bitwise-calculation-exact`; otherwise a clean scout is
`exact-sampled-prefix`, never full exact-output or `.osu` equivalence.

## First launch result

DCC job `49559647` at commit `432bdd0` failed before graph capture or timing.
The structured rejection report is
`/work/imt11/Mapperatorinator/runs/weighted-h8-49559647-432bdd0/weighted-h8.json`
(SHA-256 `e5b31c47b4e8587bc9300cce51da5dfb467ebb3f44e3194e3fc8a26a0ba74641`).
The H8 driver correctly retained only four mutable tensor inputs for per-step
copies, but incorrectly passed that reduced mapping to the generic B1 lane
capture helper. Without `encoder_outputs` and `past_key_values`, the full model
entered the encoder with `input_features=None` and raised in `conv1d`.

This is setup-only evidence: it says nothing about H8 exactness or throughput.
The correction keeps the complete prepared model-call mapping for capture,
keeps the four mutable tensors as a separate per-step view, and fails loudly
before CUDA capture if encoder outputs, cache ownership, or decoder IDs are
missing. A local regression test proves the complete mapping reaches the lane
helper while the mutable view stays narrow.

## Corrected Phase A result

DCC job `49559747` at commit `511e0ba` completed the verifier and then exited
`1:0` intentionally because the performance keep bar failed. The strict report
validator passed and the report is
`/work/imt11/Mapperatorinator/runs/weighted-h8-49559747-511e0ba/weighted-h8.json`
(SHA-256 `d96024181d8bf0a68f2fd74a58cb7266f4f0bddb73cec3c3ed579c72c2ab30a1`).

Exactness passed in both reciprocal orders. Raw logits, active self cache, and
cross cache matched bitwise, so this bounded sampled-prefix result is
`bitwise-calculation-exact`; it is still not full-song, final `.osu`, or server
evidence. The private B1 recurrence ledger also matched processed scores,
probabilities, tokens, RNG, monotonic state, and feedback state at every step.

| Order | Hybrid B2 complete wall | Private B1 control | Same-order gain |
| --- | ---: | ---: | ---: |
| `0,1` | `532.528 tok/s` | `352.473 tok/s` | `+51.08%` |
| `1,0` | `507.198 tok/s` | `352.452 tok/s` | `+43.91%` |

The absolute Phase A requirement was `>623.656675984333 tok/s`; the worst
reciprocal result was `18.67%` below it and would need another `22.96%` speedup.
Substituting the observed worst-bucket rate uniformly across all `5,905` pair
steps is only a diagnostic, not a weighted claim: it yields `28.168s` including
the pinned `4.883s` setup and `422.819` aggregate tok/s for `11,910` tokens.

Seven private/shared graphs passed ownership and no-timed-allocation gates.
Current/peak allocated memory was about `2.161/2.163 GB`, reserved memory was
`3.536 GB`, and the prebuilt extension cache stayed unchanged. Reject Phase B
and the B2 weighted bucket sweep under this gate. No scheduler, runtime, offline
engine, or server work is authorized from this result.
