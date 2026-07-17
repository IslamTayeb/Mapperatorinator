# §55 Track1 bank — STOP / hand to §53–§54

**Status:** **STOP_TRACK1_BANK** — full-song in-loop E still ≪1.7  
**Branch / WT:** `codex/turbo-s55-bank` @ `3e751815`  
**Base:** integrator `44ab1f3e` (§47 `d3cd6939` + §49 chain)  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no merge**; **no §44**; **no turbo tip graduate**; **no 500 claim**.

## Reality check

§52 sealed at **38.61 TPS / E≈1.06**. Do not blind re-scout 1-layer tip.  
§55 tried the cheap E fix (multsong 2-layer + offline-aligned sampling + chain top-p).

## Jobs

| Phase | Job | Result |
| --- | --- | --- |
| In-loop E probe (1 map window, 512 tok) | **`50151167`** | **E=1.977** (median cycle 2.0) — **false green** for full song |
| FP16 path-hit scout (SALVALAI full) | **`50152542`** | main_tps **37.99**; median E **1.0** / mean **1.074**; path_ok **true** |

### Scout `50152542` (z25? h36-5, 2080 Ti, FP16)

| Metric | Value |
| --- | ---: |
| commit | `3e751815` |
| draft | `draft_multsong.pt` γ=5, structural_processors=0 |
| main_tps | **37.99** (≪ 384; ≪ tip 366.11) |
| median / mean E | **1.0** / **1.074** |
| cycle-glue median | **0.454 ms** (gate ≤0.4 **MISS**) |
| draft / verify / rebuild ms | 3.90 / 15.78 / 14.99 |
| path hits | chain replay **1190**, native verify **829**, eager_fallback **0**, keep-KV on |
| recommend §44 / graduate | **NO** / **NO** |

Artifact: `/work/imt11/Mapperatorinator/runs/s55-turbo-fp16-scout-50152542/`

## Diagnosis

1. **Single-window probe ≠ full-song in-loop E.** Probe window was map-only AR draft on a tip-like prefix → E≈1.98. Full song mixes timing windows (E≈1) + many model-committed map windows → median E collapses to **1.0** (same band as §52/§47).
2. Multsong 2-layer + offline-aligned temp/top-p + chain top-p **does not** restore full-song E≥1.7 cheaply.
3. Path hits land; wall still ~38 TPS — acceptance, not wiring, remains binding.

## Ruling

**STOP Track1 bank attempt.** Do **not** §44. Do **not** graduate turbo tip. Tip stays `55949274` / **366.11**.  
Hand off to **§54** (verify kernels) and **§53** (EAGLE; already STOP_KILL on held-out/in-loop — do not re-open without new architecture).

## Return block

| Field | Value |
| --- | --- |
| In-loop E (probe / scout) | **1.977** (1-window) / **1.074 mean · 1.0 median** (full song) |
| Scout job / numbers | **`50152542`** / **37.99 TPS**, path_ok, E unhealthy |
| Graduate Y/N | **N** |

## Not this lever

Merge to main; 500 claim; bare §52b; claiming probe E as production acceptance.
