# §55 Track1 bank — in-loop E raise → scout → graduate?

**Status:** OPEN — cheap in-loop E probe first  
**Branch / WT:** `codex/turbo-s55-bank` @ `/work/projects/Mapperatorinator-worktrees/turbo-integrator`  
**Base:** integrator `44ab1f3e` (§47 `d3cd6939` + §49 chain)  
**Campaign tip unchanged:** `55949274` / FP16 **366.11** — **no merge to main**; **no 500 claim**.

## Reality check (§52)

Scout `50150615` @ `44ab1f3e`: main_tps **38.61**, path hits LAND, **E≈1.06 ≪ 1.7**.  
Root cause: TF tip-dump E ≠ in-loop map E. **Do not** blind re-scout same tip/1-layer.

## §55 plan

1. **Raise in-loop E** toward ≥1.7 cheaply:
   - Prefer §43 **multi-song 2-layer** `draft_multsong.pt`, γ=5, temp/top_p 0.9
   - Align sampling with offline probe: **structural processors OFF**; chain uses **temp+top-p** for sample + q
   - Log **E per window** (`turbo_E_*`, `per_window_E` in scout summary)
2. **Only if E≥1.7:** one instrumented FP16 SALVALAI scout (`jobs/s55-turbo-fp16-pathhit-scout.sbatch`)
   - Counters: accepted/verify, NVTX draft/verify/accept/glue, graph hits, **cycle-glue ≤0.4 ms**
   - If E healthy and TPS >8% under 417–436 → STOP diagnose wall
3. **If scout ≥384.4 sustained AND E healthy:** fire §44 TIER1 → GRADUATE turbo tip as separate line (bit-exact tip unchanged). Merge candidate per §34.
4. **If cannot get in-loop E≥1.7 cheaply:** STOP Track1 bank; hand to §53/§54.

## Jobs

| Phase | Script | Notes |
| --- | --- | --- |
| In-loop E probe | `jobs/s55-inloop-e-probe.sbatch` | 512 tok map window; gate E≥1.7 |
| Scout (gated) | `jobs/s55-turbo-fp16-pathhit-scout.sbatch` | unique TMPDIR/TORCH_EXTENSIONS |

## Env (probe + scout)

```
MAPPERATORINATOR_TURBO_DRAFT_CKPT=.../draft_multsong.pt
MAPPERATORINATOR_TURBO_GAMMA=5
MAPPERATORINATOR_TURBO_STRUCTURAL_PROCESSORS=0
MAPPERATORINATOR_TURBO_DRAFT_CHAIN_GRAPH=1
MAPPERATORINATOR_TURBO_STEP_PROFILE=1
```

## Ruling (fill after jobs)

| Field | Value |
| --- | --- |
| In-loop E | TBD |
| Scout job / TPS | TBD / TBD |
| Graduate Y/N | TBD |

Tip stays `55949274` / **366.11** until graduate line is authorized.
