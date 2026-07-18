# T3 TORCH.COMPILE — handoff (PIVOT EXECUTION PACKAGE)

**Status:** **FULL-STEP RESTORE IN PROGRESS** · H4 cancelled · A5000 reseal submitting  
**This agent:** REPLACEMENT (2026-07-18 relaunch) — restore harvest-3 full-step + reseal under relaxed gates  
**Package:** Pivot **T3** compile-then-capture  
**Branch / WT:** `codex/t3-compile-then-capture` (local tip updating)  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/t3-compile-then-capture`  
**Base:** `codex/turbo-on-tiger-pr120` @ `b96c3e38` (tiger PR #120 `d01cdd27` + §58/§59 rails)  
**Frozen tip:** `55949274` / FP16 **366.11** — **regression reference only**; **no merge**; **no push to PR #120**  
**T4 turbo:** **PARKED** — §34 turbo unchanged; do not wire speculative  
**Do not abandon torch.compile.**

## Early progress (2026-07-18 relaunch)

| Step | State |
| --- | --- |
| Scancel H4 | **50228030/031/096** already CANCELLED; live relaunch **50230336/337/339** CANCELLED |
| Restore full-step | `compiled_decode.py` + cells restored from `3e0aacb7` (eager `_tail`, `mode=default`); sub-op default removed |
| T5 track rule | T3 `required_pass` = **ks_parity** (greedy FAIL = documented drift) |
| A5000 reseal | submitting via `scripts/dcc/t3_submit_reseal.sh a5000` |
| 2080 reseal | after A5000 pair (≤2 concurrent GPU) |
| Promote | pending harvest |

## T3 EXACTNESS RELAXATION — BINDING

**Scope: T3 only.** Does **not** change §34 turbo, T4 PARK, tip freeze, or other tracks.

| Field | Ruling |
| --- | --- |
| Exactness bar | **NOT** bit-identical `.osu` / greedy token-match |
| Quality bar | **“Mostly good” / coherent maps** + **T5 KS pack** |
| Promote candidate | **Full decode-step** compile-then-capture — eager `_tail`, `mode=default` |
| Harvest 2/3 speed | **STAND** — A5000 **+28.8%** (h2) / **+22.7%** (h3) |
| Harvest 4 (sub-op) | **STOPPED / fallback-not-required** |
| Tip / upstream | Tip `55949274` **FROZEN**; **no merge**; **no PR #120 push** |

### Promote gates (post-relaxation)

| Gate | Criterion |
| --- | --- |
| A5000 | main-gen **+≥10%** vs like-with-like uncompiled fast path |
| 2080 Ti | **no-regression** |
| Exactness | coherent + **T5 KS** — not greedy byte-match |
| Forbidden | `reduce-overhead` near manual capture; Inductor `_tail`; tip grind; PR #120 push |

## Binding pattern

| Knob | Value |
| --- | --- |
| Outer step | **Inductor** `forward_only` (full decode-step) |
| Sampling `_tail` | **eager** |
| `fullgraph` / `dynamic` / `mode` | `True` / `False` / **`default`** |
| Warm | **EVERY** bucket before capture |
| Opt-in | `MAPPERATORINATOR_COMPILE_DECODE=1` |

## Reference speed seals (STAND)

| Harvest | Commit | A5000 Δ main_tps |
| --- | ---: | ---: |
| 2 | `eb85f4b3` | **+28.8%** (343.30 → 442.20) |
| 3 | `3e0aacb7` | **+22.7%** (348.61 → 427.71) |

## H4 CANCELLED jobs

| Job | State |
| ---: | --- |
| 50228030 / 50228031 / 50228096 | CANCELLED |
| 50230336 / 50230337 / 50230339 | CANCELLED (relaunch) |

## Do-not

- Push to Tiger14n / PR #120  
- Wire T4 / modify tip `55949274` / claim 500 / merge  
- Grind harvest-4 sub-op as promote path  
- Require bit-identical greedy for T3 promote  

## Ruling

**Promote candidate = full decode-step** (eager `_tail`, `mode=default`).  
**Exactness = relaxed** (mostly good + T5 KS). Tip frozen. No PR #120 push.
