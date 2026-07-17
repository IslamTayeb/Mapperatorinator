# §43 Draft quality (W4) — held-out E, multi-song distill, K/γ/temp sweeps

**Status:** DONE — recommend config below for turbo perf build  
**Branch / WT:** `codex/turbo-draft-quality` @ `5ae2dd7d` (+ nan-filter follow-up)  
**Base tip draft:** train `50146289` E[acc]=**2.921** on SALVALAI (in-domain).  
**Job:** `50147299` COMPLETED 0:0 **00:09:40** z25-20  
**Constraint:** offline suite — **no** `generate_window` / canary / §39 wiring.

## Goal

Acceptance table + recommended `(K, draft config)` for the turbo perf build.

## Protocol

| Phase | What |
| --- | --- |
| (a) | Tip SALVALAI-only draft → E on held-out **ela-ke-leitada**, **nube-negra** (+ SALVALAI ref) |
| (b) | Multi-song distill shards (salvalai+pegasus+lambada) + CE/KL **4000** steps from tip ckpt |
| (c) | Temp ∈ {0.7,0.9,1.1}, γ∈3…8, K∈{1,2,4} from logit dumps → acceptance-vs-cost |
| (d) | Tree (K>1 MC) + 1-layer train (1500 steps) + half-width **cost sim** (α proxy = 1-layer) |

**Cost model:** `cost = K·γ·(n_draft_layers/12) + 1` verify unit; half-width multiplies draft term by 0.5.  
**Recommend:** among held-out aggregate, prefer E≥1.8 then max E/cost.

**NaN note:** ela × tip teacher produced NaN teacher logits on **3314/7858** positions (old five-song FP32 dumps). v2 tables use **finite-only** rows. nube + SALVALAI: 0 NaNs.

## Results — E[accepted/step] @ temp/top-p 0.9, γ=5

| Draft | SALVALAI | ela (filt.) | nube | held-out mean |
| --- | ---: | ---: | ---: | ---: |
| Tip 2-layer (`50146289`) | **2.922** | 1.980 | 2.087 | **2.034** |
| Multi 2-layer (`50147299`) | 2.966 | 2.339 | 2.537 | **2.438** |
| 1-layer cheap | 2.343 | 2.077 | 2.121 | **2.099** |

All clear E≥1.8 on held-out (promote bar for simple K-draft).

## Acceptance-vs-cost (held-out agg, nan-filtered)

Top production drafts by E/cost:

| Rank | Draft | T | γ | K | held-out E | E/cost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **one_layer** | 0.7 | 3 | 1 | 1.996 | **1.597** |
| 2 | one_layer | 0.9 | 3 | 1 | 1.974 | 1.579 |
| 3 | one_layer | 1.1 | 3 | 1 | 1.966 | 1.573 |
| 4 | one_layer | 0.7 | 4 | 1 | 2.083 | 1.562 |
| — | multi_2layer | 0.7 | 3 | 1 | 2.224 | 1.483 |
| — | multi_2layer | 0.9 | 3 | 1 | 2.212 | 1.475 |
| — | tip_2layer | 1.1 | 3 | 1 | 1.941 | 1.294 |

- **Tree K∈{2,4}** never beats K=1 on E/cost once draft is cheap (extra draft tokens dominate).  
- **Half-width sim** ties 1-layer (same α proxy, same relative cost under this model).

## Recommended config for perf build

**Primary (throughput / E·cost):**  
`draft=one_layer` · **K=1** · **γ=3** · draft-temp **0.9** (0.7 slightly better E/cost; keep 0.9 to match tip sampling) · ckpt `draft_1layer.pt`

**Quality / safer ship (higher held-out E):**  
`draft=multi_2layer` · **K=1** · **γ=5** · temp **0.9** · ckpt `draft_multsong.pt` (held-out mean E **2.44**)

Do **not** enable tree verify for v1 perf — K>1 loses on the cost model here.

## Artifacts

| Item | Path |
| --- | --- |
| Run root | `/work/imt11/Mapperatorinator/runs/s43-draft-quality-50147299/` |
| Table (authoritative) | `…/acceptance_table_v2.json` |
| Sweeps | `…/phase_c_sweeps_v2.json` |
| Logits | `…/logit_dumps/*.pt` |
| Multi-song ckpt | `…/draft_multsong.pt` |
| 1-layer ckpt | `…/draft_1layer.pt` |
| Shards | `…/distill_shards_*.json` |
| Logs | `/work/imt11/Mapperatorinator/logs/s43-draft-quality-50147299.{out,err}` |

## Scripts

- `utils/s43_draft_quality_suite.py`
- `utils/s43_offline_sweep_from_logits.py`
- `jobs/s43-draft-quality.sbatch`

## Decision

**GO for perf wiring with K=1.** Prefer **1-layer γ=3** for throughput; keep **multi 2-layer γ=5** as quality preset. Tip SALVALAI-only draft already generalizes (held-out E≈2.03) — multi-song still worth shipping if verify budget allows the wider ckpt. No turbo runtime changes in this §. TIER1 still required before any 500 claim.
