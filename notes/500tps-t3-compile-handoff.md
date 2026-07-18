# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **PROMOTE Y** under relaxed gates (full-step reseal sealed)  
**This agent:** REPLACEMENT (2026-07-18 relaunch) — complete  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` @ **`88d4ddf8`**  

**Local WT:** `/work/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**Frozen tip:** `55949274` / FP16 **366.11** — **no merge**; **no PR #120 push**  
**T4 turbo:** **PARKED** — §34 unchanged  
**Do not abandon torch.compile.**

## Promote decision — **Y**

| Gate | Result |
| --- | --- |
| A5000 +≥10% main_tps | **PASS +23.0%** (348.41 → 428.51) |
| 2080 Ti no-regression | **PASS +22.1%** (243.90 → 297.88) |
| Coherent maps | **Y** — base HO 583 / compile HO 569; both parse |
| T5 KS | **PASS** (Bonferroni α=0.002; fail_metrics=[]) |
| Greedy bit-identical | **FAIL expected** — documented Inductor fp16 drift; not required |

## Jobs

| Cell | Job | State |
| --- | ---: | --- |
| A5000 baseline | **50230514** | COMPLETED |
| A5000 compile | **50230515** | COMPLETED |
| 2080 baseline | **50232052** | COMPLETED |
| 2080 compile | **50232053** | COMPLETED |
| H4 cancelled | 50228030/031/096 + 50230336/337/339 | CANCELLED |

## Metrics

### A5000 @ `fb02adc0` full_step / mode=default

| Variant | main_tps | ms/map-token | cold_start_s | map_tok |
| --- | ---: | ---: | ---: | ---: |
| baseline | **348.41** | **4.845** | 164.0 | 7526 |
| compile | **428.51** | **3.981** | 377.0 | 7363 |
| Δ | **+23.0%** | **−17.8%** | | |

### 2080 Ti

| Variant | main_tps | ms/map-token | cold_start_s | map_tok |
| --- | ---: | ---: | ---: | ---: |
| baseline | **243.90** | **5.050** | 51.6 | 7704 |
| compile | **297.88** | **4.313** | 255.6 | 7612 |
| Δ | **+22.1%** | **−14.6%** | | |

## KS (A5000 sampled maps)

| Metric | KS stat | p | n_a / n_b |
| --- | ---: | ---: | --- |
| ho_count | 1.0 | 1.0 | 1 / 1 (degenerate single-map) |
| density | 0.027 | ≈1.0 | 155 / 153 |
| ho_type | 0.002 | 1.0 | 583 / 569 |
| timeshift | 0.033 | 0.900 | 582 / 568 |
| slider_length | 0.123 | 0.123 | 178 / 175 |

Artifacts: `notes/t3-artifacts/summary-5023051{4,5}.json`, `summary-5023205{2,3}.json`, `ks-a5000-50230514-50230515.json`, `T5_GATES-a5000-reseal.json`, `coherent-a5000.json`

## Binding pattern (promoted)

| Knob | Value |
| --- | --- |
| Outer step | Inductor `forward_only` (full decode-step) |
| `_tail` | **eager** |
| mode | **default** |
| Warm | every bucket |
| Exactness | coherent + T5 KS — **not** bit-identical |

## Restore history

- Tip had harvest-4 sub-ops @ `d0488151`; restored from `3e0aacb7` in `946bff1e`
- H4 STOPPED / fallback-not-required
- T5 track rule: T3 `required_pass=ks_parity` (greedy FAIL OK)

## Do-not

- Push to Tiger14n / PR #120  
- Merge / tip `55949274` grind / claim 500 / T4 wire  
- Fold T3 relaxed exactness into §34 turbo  
- `reduce-overhead` near manual capture; Inductor `_tail`

## Ruling

**PROMOTE Y** — full decode-step compile-then-capture (eager `_tail`, `mode=default`) under relaxed exactness. Tip frozen. No PR #120 push.
