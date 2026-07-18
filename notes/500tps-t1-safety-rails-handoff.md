# §59 T1 SAFETY RAILS — handoff

**Status:** **SMOKE PASS** (2026-07-18)  
**Branch / WT:** `codex/turbo-on-tiger-pr120` @ **`72d6ed72`** (rails `26e465b3`; import-path fix)  
**Local WT:** `/work/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**DCC WT:** `/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-on-tiger-pr120`  
**Base:** tiger PR #120 `d01cdd27` (local turbo scaffold on top)  
**Optimized tip FROZEN:** `55949274` / FP16 **366.11** — **do not modify**  
**Remote:** `islamtayeb` only — **no push to Tiger14n / origin / PR #120**  
**No merge.**

## What fixed

| Rail | Detail |
| --- | --- |
| Force SDPA | `inference.py` forces `attn_implementation='sdpa'` when `fast_decoder_loop` / `super_timing_fast_loop`; `require_sdpa_for_fast_path` gates capture |
| Loud capture | CaptureError → stderr + traceback + `warnings.warn`; latch refuses silent stock fallback unless `MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK=1` |
| Graph-cache VRAM | `_decoder_cache` / `_k_decoder_cache` are `OrderedDict` of `(decoder, cache, nbytes)`; shared budget default **2048 MiB** (`MAPPERATORINATOR_GRAPH_CACHE_BUDGET_MB`); LRU + `close()` + `empty_cache` |
| CFG left-pad KV | Pad-aware graph keeps full-length **2D** mask; replay sets new pos=1 + RoPE via `(mask.cumsum-1).clamp(0)`; CFG mask merge uncond∥cond; CFG token `repeat(2,1)` into (2B,1) graph buffer |

Preserved: `neutralize_dynamic_rope`, `_IncrementalMonotonicMask`, `CUDAGraphDecoder` q_len=K, buckets, turbo `get_k_decoder`.

## CFG bug (local doc)

Capture previously passed `decoder_attention_mask=None`, so decode attended pad K/V written at prefill. Stock HF extends the 2D pad mask each step. Unequal CFG left-pad → pad leak. Fix is local-only vs PR #120 until an upstream stacked PR is authorized.

FA2 `kv[:, :cache_position[-1]+1]` dynamic slice (modeling_varwhisper) is why the fast path forces SDPA.

## Smoke

| Item | Value |
| --- | --- |
| Scout | `utils/s59_t1_safety_rails_scout.py` |
| Job | `jobs/s59-t1-safety-rails.sbatch` → **`50212775`** (**COMPLETED**, ExitCode `0:0`, Elapsed `00:00:34`, Node `dcc-core-ferc-s-z25-20`) |
| Asserts | require_sdpa / CFG mask+cumsum / eviction@1MiB / loud latch / force-SDPA — **all PASS** |
| Env | `unset MAPPERATORINATOR_ALLOW_CAPTURE_FALLBACK` |
| DCC sync | job commit `72d6ed72` (PYTHONPATH=repo root; preflight `import_ok CaptureError`) |
| Decision | **PASS** |

### Prior smoke (superseded)

| Item | Value |
| --- | --- |
| Job | **`50194265`** (**FAILED**, ExitCode `1:0`) |
| Reason | `ModuleNotFoundError: No module named 'osuT5.osuT5'` (scout inserted `REPO/osuT5` on `sys.path`) |
| Fix | `72d6ed72` — insert repo root only; sbatch exports `PYTHONPATH=$REPO` + preflight import |

## Harvest seal

- **Sealed:** 2026-07-18
- **Job:** `50212775` State=`COMPLETED` ExitCode=`0:0` Elapsed=`00:00:34` Node=`dcc-core-ferc-s-z25-20`
- **Smoke verdict:** **PASS** (`OK=true`, `DECISION=PASS`, `RC=0`)
- **Commit (job):** `72d6ed72987f3ea8f1ca28edacadfd3852bb935c`
- **Campaign tip frozen:** `55949274` — untouched
- **Per-rail asserts:**
  - SDPA (`require_sdpa` / `force_sdpa_setup`): **PASS** / **PASS**
  - Loud capture (`loud_capture_failure`): **PASS**
  - Cache (`graph_cache_eviction`): **PASS** (1 eviction under 1 MiB budget)
  - CFG (`cfg_left_pad_mask`): **PASS**
- **Preflight:** `import_ok CaptureError`; `pythonpath=` DCC WT root
- **Remote run dir:** `/work/imt11/Mapperatorinator/runs/s59-t1-rails-50212775/`
- **Remote logs:** `/work/imt11/Mapperatorinator/logs/s59-t1-rails-50212775.{out,err}`
- **Local artifacts:** `/home/islam/projects/Mapperatorinator/notes/s59-artifacts/` (`verdict-50212775.txt`, `summary-50212775.json`, `asserts-50212775.json`, `preflight-50212775.txt`, `stdout-50212775.txt`, `stderr-50212775.txt`, `s59-t1-rails-50212775.{out,err}`)
- **No 500 claim.** No PR #120 push.

## Do not

- Push to `origin` / `tiger14n` / PR #120
- Merge
- Touch tip `55949274`
- Claim 500 TPS
