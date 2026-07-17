# §52 Integrator path-hit scout (§47+§49 decision A)

**Status:** **HARVESTED — STOP/DIAGNOSE** (A falsified: E_runtime≪1.7; main_tps≪tip)  
**Branch / WT:** `codex/turbo-integrator` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Tip:** `44ab1f3e`  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no 500 claim**; **no tip graduate**; **no merge to main**; **no §44**.

## Inputs absorbed

| Worker | Tip | Role in §52 |
| --- | --- | --- |
| §47 W-KV | `d3cd6939` | keep-accepted-KV O(1) rewind; teacher_fwd/cycle=1 |
| §49 W-DG | `d2f09578` (files) | γ=3 graphed draft chain; persistent session cache |
| §48 W-VG | verify path only | graph-native K verify; **STOP_KILL stands — no grind** |

## Scout harvest (`50150615`, z25-20, 2080 Ti, FP16 SALVALAI)

| Metric | Value |
| --- | ---: |
| Slurm | **COMPLETED** 0:0 · 00:04:20 |
| commit | `44ab1f3e` |
| main_tps | **38.61** (≪ tip 366.11; ≪ 384) |
| main_wall_s / tokens | **49.26 s** / **1881** |
| median E[acc] | **1.00** (mean **1.06**; max 1.78) |
| median ms/cycle | **31.18** |
| median phase ms | draft **2.25** / verify **14.65** / rebuild **15.89** / host **0.55** |
| teacher_forwards/cycle | **1.0** |
| teacher_extra_q1/cycle | **1.0** (residual/bonus every cycle) |
| kv_mode | `keep_accepted_o1_rewind` |
| recommend §44 | **NO** |
| Artifact | `/work/imt11/Mapperatorinator/runs/s52-turbo-fp16-scout-50150615/` |

### Path-hit counters (last window; session-cumulative where noted)

| Counter | Value | Gate |
| --- | ---: | --- |
| `draft_chain_graph_replay` | **1510** | HIT |
| `draft_chain_graph_capture` | 11 | persistent cache alive |
| `draft_chain_eager_fallback` | **0** | HIT |
| `verify_graph_native_replays` | **1259** | HIT |
| `verify_graph_captures` | 251 | high zoo / bucket churn |
| `verify_prepare_inputs_calls` | **1510** | NOT 0 (residual Q1 / legacy prep) |
| `keep_accepted_o1_rewind` | on | HIT |
| `crop_to_L_full_rebuild` | 0 | canary separated |

## Ceiling recompute

Authorizing projection assumed E∈{2.0, 2.4} → 461–554 TPS.  
**Runtime E≈1.06 &lt; 1.7 kill** → **decision (A) falsified** even though DG + graph-native path hits landed.

| Scenario | step / E | TPS |
| --- | ---: | ---: |
| Authorizing (pre-scout) | 4.332 ms · E=2.0/2.4 | 461 / 554 |
| Runtime observed | ~31.2 ms · E≈1.06 | **~38.6** (measured) |
| If E=1.06 @ ideal 4.332 ms | — | ~245 (still &lt;384) |

## Diagnosis (binding)

1. **Acceptance collapse** — held-out draft E~2.0–2.4 did not appear in-loop (E≈1.06). Chain samples temp-softmax **without** structural processors / top-p → q mismatched to teacher; residual every cycle (`extra_q1/cycle=1.0`).
2. **Rebuild still ~16 ms/cycle** — residual Q1 teacher + draft `decode_token` every cycle (not crop-rebuild; keep-KV mode correct).
3. **Verify ~14.7 ms** vs §48 microbench 3.075 ms — full-song bucket/capture zoo (251 captures) + residual Q1 prep calls; not a VG grind reopen.
4. **Draft chain worked** — 2.25 ms/cycle vs §47 eager draft 5.84 ms; 1510 replays / 0 eager fallback.

## Ruling

**STOP/DIAGNOSE.** Do **not** §44. Do **not** grind §48. Tip stays `55949274` / **366.11**.  
Per post-rung decision: A falsifies on E_runtime&lt;1.7 → open **(B) §51 EAGLE** (parked → now eligible). Optional follow-ups before EAGLE: restore processor/top-p-consistent draft proposal, or measure whether non-chain draft recovers E with keep-KV+native verify only.

## Not this lever

Merge to main; 500 claim; §48 VG grind; tip graduation; claiming 38.61 as production TPS.

## Addendum — E-collapse root cause (2026-07-17 diagnose)

**Verdict:** not an easy ≥1.7 config/wiring fix → **no §52b** re-scout. Open **§51 EAGLE**.

### Checks

| Hypothesis | Result |
| --- | --- |
| Wrong draft depth (2-layer vs 1-layer) | **REJECT** — scout used `draft_1layer.pt`; profile `turbo_draft_init_layers=[0]`; payload `draft_decoder_layers=1` |
| Wrong γ / temp | **REJECT** — runtime γ=3, temp=0.9, top_p=0.9 (matches §43 perf preset) |
| Graphed-draft logits ≠ TF probe | **PARTIAL** — chain samples `softmax(logits/T)` **without** top-p / structural processors (`draft_chain_graph._sample_into`); verify/rejection uses processor+temp+top_p. Real bug, but **not primary** |
| keep-KV drift poisoning drafts | **REJECT as unique cause** — §47 eager keep-KV (processors+top_p on) already E_map median **1.095** / weighted ~same band |
| Path miss / wrong ckpt path | **REJECT** — chain replay 1510 / eager_fallback 0; ckpt path as above |

### Primary cause

**Teacher-forced tip-dump α (§43) ≠ in-loop speculative E.**

- §43: Leviathan closed-form on **tip-authored map dumps**, `use_cache=False` full teacher-force, **no** structural processors; 1-layer γ=3 temp0.9 → held-out E≈**1.97** (α≈0.53–0.59).
- §52/§47 runtime: AR draft on **model-committed prefixes**, processors on, StaticCache; map windows median E≈**1.08**, verify-weighted ≈**1.25**; timing windows ≈**1.00** (draft never trained for timing).
- First-token accepts stay rare → residual Q1 every cycle (`extra_q1/cycle=1.0`) even though path hits land.

### Secondary (fix later, do not claim §52b)

Graphed draft chain proposal distribution mismatches rejection `q` (no top-p/processors in-graph). Eager path already proves fixing that alone cannot reach E≥1.7 on this draft.

### Ruling

**STOP §52.** Do not grind. **§51 EAGLE** scaffold on `codex/turbo-eagle-draft-head`. Tip `55949274` / **366.11**. No §44.
