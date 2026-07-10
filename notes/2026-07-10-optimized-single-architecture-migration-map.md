# Optimized Single Architecture Migration Map

## Scope

This is a refactor-only migration from `main@3f4b088`. It must not change
precision, model math, sampling, RNG order, stopping, output assembly,
windowing, metadata semantics, or accepted FP32 single-song performance.

The endpoint is one implementation of the accepted stack owned by
`osuT5/osuT5/inference/optimized/`. It is selected by
`inference_engine=optimized optimized_inference_mode=single`; the accepted
legacy micro-flag bundle may delegate to the same implementation. Default V32
must remain cold with respect to optimized and native-extension modules.

Batching, new speculative behavior, new kernels, reduced precision, schedulers,
and performance optimization are out of scope. Mechanical ownership cleanup
for existing batch/speculative verifier code is allowed when it removes a
dependency on a legacy shim without changing its calculations or claims.

## Frozen Pre-Migration Evidence

The accepted full-song stack is generation compile, active-prefix bucket `64`,
CUDA graph warmup `0`/minimum step `1`, stateful batch-1 monotonic processing,
FP32 q_len=1 BMM cross-attention, persistent per-context cache/graph/encoder
state, native q1 self-attention, and fused native RoPE/cache self-attention.

Accepted DCC job `49230082` measured `7,639` SALVALAI main tokens in
`28.243s` synchronized model time, or `270.475 tok/s`, with exact main/timing
tokens and byte-identical `.osu` output.

At pre-migration `main@3f4b088`, the default import baseline was not compliant.
A fresh process importing
`inference`, `server`, and `modeling_varwhisper` loads:

```text
osuT5.osuT5.inference.decode_loop
osuT5.osuT5.inference.native_q1_attention
torch.utils.cpp_extension
```

Compilation was call-lazy, but ordinary V32 model import already imported
the native implementation and extension machinery.

| File | Pre-migration SHA-256 |
| --- | --- |
| `inference.py` | `01bffc470997a026dc5020e79632fccdc015a6c07c3bfc9b6a6c981ed5a38502` |
| `inference/server.py` | `04652b0a9b49b0d67788dfa81e667bf8021c315c43a6e6566208722f93603782` |
| `inference/processor.py` | `cb77f7cb90c81067dcd22b7e8a19b8204e1133ebbcf692469a7f66d17854829e` |
| `inference/decode_loop.py` | `da0b50bb1130c642b2270481b068a7c4ef468016d6b9aeb96f0e2f3f36605358` |
| `inference/direct_decode.py` | `cbaddf37772773a725598d7f970465d52efebe485a568697efe95b5f1a72798c` |
| `inference/logit_processors.py` | `0c04fb70cc1a56c7415bc024db134495d08cd41fcf84c7c422277a132fffcb3b` |
| `inference/native_q1_attention.py` | `c65438acc24d03f13fc0df694e29ddc2b21aff425cd01fef9d14814544febcd9` |
| `runtime_profiling.py` | `1bd09e211aa6092929f288c12a8858b8fd3b07ac123ec95ca15095325668f37d` |
| `modeling_varwhisper.py` | `ed4ccb7bd18dc43caf4e6f60ae477c1a04f8b754643b80e484f18453aa2af8c2` |

## Pre-Migration Accepted Runtime Dependency Graph

```text
config.py / Hydra defaults
  -> inference.py validation, loader selection, requested metadata
  -> timing Processor, then main Processor
  -> Processor.generate_sequential, one call per song window
  -> Processor.model_generate
  -> server.model_generate
     -> logits processors and persistent cache/session dictionary
     -> generation runtime context
     -> model.generate(custom_generate=active_prefix_decode_generate)
  -> active-prefix prefill/decode/CUDA-graph loop
  -> neutral VarWhisper dispatch points
     -> q1 BMM cross, native q1 self, fused RoPE/cache native q1 self
  -> generation stats -> Processor record -> final profile/output
```

The pre-migration tree had two different state objects associated with
`DecodeSession`:

1. `inference/direct_decode.py:DecodeSession` is a verifier-first dataclass used
   by diagnostic utilities and optimized batch/speculative scouts. Production
   single-song inference does not instantiate it.
2. The accepted runtime used `Processor.decode_session_state`, a raw dictionary
   filled by `server._session_cache()` and the active-prefix loop.

The migration first moves the verifier ABI mechanically, then introduces a
typed production session matching the accepted dictionary lifecycle. Moving
the verifier class alone does not migrate the 270 TPS path.

## State And Calculation Contract

- Timing and main use separate `Processor` instances.
- Session state clears at top-level `Processor.generate()`.
- A fresh session is created for each unfinished output context and persists
  across that context's windows.
- Cache object identity persists, but self/cross contents and `is_updated`
  reset before each window.
- Graph entries/static buffers and stable encoder object identity persist;
  each window copies new encoder data into the stable object.
- The stateful monotonic processor is fresh for each generation window.
- Sampling uses the global PyTorch generator and serial `torch.multinomial`.
  No private generator or extra draw may be introduced.
- Native q1 self and fused RoPE/cache are requested for all contexts but are
  intentionally disabled for timing. q1 BMM cross remains enabled for timing
  when its tensor contract matches.
- The fused path preserves Wqkv, rotary cos/sin, initialized `StaticCache`
  aliases, cache position, mask trim, kernel call, and
  transpose/contiguous/view order.
- Preserve extension name `mapperatorinator_q1_attention`, source, signatures,
  singleton loading, mask flatten/fp32/contiguous conversion, and preload before
  compilation/capture.

## Ownership Classification

### Optimized implementation moved

| Pre-migration owner | Implementation | Current owner |
| --- | --- | --- |
| `inference/native_q1_attention.py` | accepted extension source, loader, q1/fused wrappers | `optimized/kernels/q1_attention.py` |
| `modeling_varwhisper.py` | q1 BMM math and native/fused orchestration/eligibility | `optimized/kernels/dispatch.py`, reached through neutral `runtime_dispatch.py` hooks |
| `runtime_profiling.py` | optimized feature state and native preload | `optimized/single/runtime_context.py`; legacy utility contexts remain lazy adapters |
| `inference/direct_decode.py` | verifier state ABI, prefill, one-token helpers | `optimized/single/session.py` |
| `inference/decode_loop.py` | active-prefix loop, stable encoder, graph capture/replay, diagnostics | `optimized/single/decode_loop.py` |
| `inference/logit_processors.py` | stateful batch-1 monotonic specialization | `optimized/single/logits.py` |
| `processor.py` and `server.py` | production cache/graph/encoder session lifecycle | `optimized/single/state.py`, orchestrated by `optimized/single/engine.py` |
| optimized branches of `server.model_generate` | runtime validation, context fallback, loop selection, effective stats | `optimized/single/engine.py` |
| `inference/native_linear.py` | rejected verifier-only linear scout | `optimized/kernels/linear.py` |
| `inference/native_decoder_layer.py` | rejected verifier-only layer scout | `optimized/kernels/decoder_layer.py` |

### Generic shared infrastructure to retain

- `inference/cache_utils.py` and normal cache construction.
- Generic Processor prompt/context/window orchestration, event assembly, and
  profile-record projection.
- Generic server IPC, grouping, static batching, ordinary V32 generation,
  forward, result splitting, and shared-RNG labels.
- Full-scan monotonic processing and other ordinary logits processors.
- Normal V32 SDPA, FA2, and eager attention paths.
- Model/tokenizer loading and compile configuration.
- NVTX/detail ranges, SDPA backend selection, profile comparison tools, EOS
  calculation, pre/postprocessing, and `.osu` assembly.

### Allowed adapters outside optimized

- `config.py` and Hydra: public selectors and reproduction flags.
- `inference.py`: validation, requested metadata, one lazy loader call.
- `processor.py`: one default-cold engine delegation point and generic metadata
  copying.
- `legacy_single_adapter.py`: complete-bundle validation and one default-cold
  compatibility delegation to `OptimizedSingleRuntime`; partial bundles fail.
- `server.py`: ordinary V32 behavior only; optimized options or session state
  fail with an instruction to use the public selector or complete legacy bundle.
- `runtime_dispatch.py`: neutral callable hook registry with no optimized or
  native implementation.
- `runtime_profiling.py`: lazy compatibility contexts for explicit diagnostic
  callers; no optimized state or calculation ownership.
- `modeling_varwhisper.py`: neutral preinstalled callable hooks only; no
  optimized/native imports or kernel/eligibility implementation in forward.
- Legacy loop/session/native files: small lazy compatibility shims only.

### Legacy V32 behavior to leave untouched

- Default `model.generate()` math and RNG order.
- Default cache construction and full-scan monotonic calculation.
- Compile-only V32 behavior.
- Static IPC server batching, grouping, splitting, queue attribution, and
  shared-global RNG policy.
- Timing/main context ordering and final result assembly.

## Target Package Ownership

```text
osuT5/osuT5/inference/optimized/
  adapter.py
  contracts.py
  single/
    __init__.py
    config.py
    engine.py
    decode_loop.py
    session.py
    state.py
    logits.py
    runtime_context.py
  kernels/
    __init__.py
    dispatch.py
    q1_attention.py
    linear.py
    decoder_layer.py
  batch/
  speculative/
  benchmark/
```

Names may change only for a cleaner dependency direction. Ownership may not
move back outside this package.

The final external adapter boundary is deliberately small:

```text
inference.py -> optimized/adapter.py -> InferenceEngineBinding
             -> Processor -> OptimizedSingleRuntime.generate_window()

complete legacy bundle -> legacy_single_adapter.py
                       -> the same OptimizedSingleRuntime

runtime_dispatch.py <-> neutral model hook calls
server.model_generate() -> ordinary V32 only
```

## Public Selector And Compatibility Contract

Default and compile-only V32 must not import optimized single, kernels, legacy
native implementations, or `torch.utils.cpp_extension`.

Optimized single internally selects one immutable effective configuration:

```text
precision=fp32
attn_implementation=sdpa
batch_size=1
cfg_scale=1.0
num_beams=1
use_server=false
parallel=false
generation_compile=true
active_prefix_bucket_size=64
active_prefix_cuda_graph=true
active_prefix_cuda_graph_warmup=0
active_prefix_cuda_graph_min_decode_steps=1
stateful_monotonic=true
q1_bmm_cross_attention=true
session_runtime=true
session_cuda_graph=true
native_q1_self_attention=true
native_q1_rope_cache_self_attention=true
```

The canonical accepted legacy bundle resolves to the same effective object.
Compile-only V32 stays generic. Unsupported partial bundles fail loudly with an
explicit migration instruction and may not silently change calculations.

`offline_batch` and `server` fail before model loading, optimized single import,
kernel import, CUDA initialization, or extension-cache mutation. The optimized
adapter receives an injected generic loader callable and never imports
`inference.py` back into the package.

## Metadata Contract

Keep existing requested fields unchanged, including engine/mode and every
legacy micro-flag. Add separate effective configuration/version fields rather
than rewriting requested values.

Preserve per-window generation compile, stateful monotonic, q1 BMM, native
requested/enabled/disabled reason, session/graph/count, active-prefix settings,
CUDA ledger/diagnostics, token IDs/counts, output SHA/size, and server
RNG/batch-attribution fields. For the legacy bundle, top-level engine remains
`v32` even though implementation ownership delegates.

## Incremental Migration Boundaries

Each boundary gets a clean commit and push. A failed exactness/performance gate
is fixed or reverted before the next boundary.

### K1: Native module ownership

Move q1 extension verbatim, replace old native module with a lazy shim, remove
model top-level native import, install neutral callable hooks before
compile/capture, and move unused native linear/layer scouts mechanically.

Gates: fresh V32 import cold; optimized kernel import does not compile; shim
import cold; fake loader exactly once; wrapper ABI; one real native one-step.

### S1: Verifier session/state ownership

Move `direct_decode.py` mechanically into optimized single state/session,
update optimized internal imports, and leave lazy compatibility re-exports.

Gates: API/import identity, one-token logits/top-k/cache, no V32 import change.

### S2: Production session ownership

Replace the raw dictionary with a typed optimized session matching per-context
and per-window lifetime exactly.

Gates: two-window persistent parity, pointers, reset, capture count,
token/logit/RNG equality.

### L1: Active-prefix loop ownership

Move loop and graph helpers byte-for-byte, update optimized internal imports,
and replace the legacy file with a lazy shim.

Gates: bucket/signature/static-input unit tests; 8-step then 256-step exact
loop; forced EOS; graph capture/replay ledger.

### P1: Optimized logits ownership

Keep V32 full scan unchanged; move stateful B1 specialization to optimized
single; optimized builder selects the subclass without warming V32 imports.

Gates: randomized full-scan/stateful parity, SOS reset, sequence jump, graph
capture, tokens.

### E1: Optimized single engine

Move optimized generation orchestration/effective stats from server and context
session reset/delegation from Processor. Make adapter load a real single engine
with compile enabled and install hooks before compile/capture.

Gates: selector matrix, lazy load, engine-vs-legacy one-step/short-loop parity,
hook restoration after success/exception.

### C1: Legacy delegation and cleanup

Route canonical legacy bundle to E1; reject partial bundles; remove duplicate
optimized implementation from server, Processor, model, and legacy modules.

Gates: AST ownership audit, complete local suite, fresh-process V32 import and
metadata snapshot, unchanged server/static-server tests.

### K2: Optimized attention dispatch ownership

Move q1 BMM/native/fused attention eligibility and orchestration out of the
model into `optimized/kernels/dispatch.py`; leave only neutral hook calls in
VarWhisper. Mechanically detach existing speculative scouts from public legacy
micro-flags without changing their calculation or authorization.

Gates: source/AST ownership, hook restoration, ordinary SDPA/FA2/eager parity,
one-token logits/top-k/cache, and fresh-process optimized-cold V32 imports.

### K3: Active-prefix attention ownership

Move active-prefix length state and self-attention input trimming into
optimized runtime context/dispatch. The optimized loop installs the hook;
generic model and profiling code do not own optimized state.

Gates: source/AST ownership, nested context restoration, one-token and 8/256
step exactness, reciprocal smoke/full-song, and unchanged V32/server paths.

## Mandatory Local Gates

### Fresh-process imports

- Import `inference`, Processor, server, and VarWhisper on V32.
- Assert no optimized, legacy/optimized native kernel, or
  `torch.utils.cpp_extension` module is loaded.
- Default and compile-only V32 leave extension cache unchanged.
- Importing `optimized.adapter` imports neither optimized single nor kernels.
- Unsupported optimized modes fail before loader/import/CUDA side effects.

### Control plane

- Dataclass and Hydra V32/single defaults agree.
- Optimized single supports only FP32/SDPA/B1/sequential/no-CFG/no-beam.
- Optimized plus legacy flags remains ambiguous and rejected.
- V32 non-single optimized modes fail.
- Legacy bundle normalization equals optimized effective configuration.

### One-step exactness

Use real SALVALAI seq9 tensors and compare prefill/q1 logits, finite/nonfinite
layout, top-20, sampled token, cache position, all active self-cache prefixes,
cross cache/`is_updated`, timing fallback, main native path, graph signature,
and capture count.

### Short loop and session

- First 8 then 256 tokens, every logits/top-k step, transcript, stop reason,
  final CPU/CUDA RNG, and forced EOS without an extra draw.
- Active self/cross cache prefixes, bucket sequence, capture/replay counts.
- Two consecutive windows with cache identity/reset, stable encoder identity,
  and graph reuse.

### Server and ownership

- Existing server control-plane, batch-state, batching-summary, and static
  comparator tests remain unchanged.
- Ordinary `server.model_generate()` never delegates.
- Default stats keys and false/`None` values remain exact.
- AST/source audit rejects CUDA source, `load_inline`, q1 BMM math, fused-cache
  implementation, graph/session state machines, or optimized module-scope
  imports outside optimized owners and enumerated lazy shims.
- Optimized batch/speculative code imports optimized sources, not shims.

## DCC Promotion Gates

Use only a pushed commit in an isolated DCC worktree after live `sinfo`, account,
GPU, checkout, cache, and environment checks.

1. Exact one-step logits/top-k/self-cache/cross-cache.
2. Exact short and 256-step token/RNG/forced-EOS.
3. Reciprocal 15-second legacy bundle versus optimized single.
4. Reciprocal full-song legacy bundle versus optimized single.
5. Separate pre-migration versus candidate default-V32 reciprocal regression.
6. Main `7,639` and timing `821` tokens, final RNG hashes, `.osu` SHA/size.
7. Same-order main throughput has no meaningful regression versus its control;
   preserve the historical `270.475 tok/s` frontier and report absolute ambient
   drift rather than promoting it as a new result. Require no meaningful timing,
   stage-wall, cold setup, or per-window regression.
8. Cold import, extension cache/build, graph cache/capture, first-window setup,
   clocks/power/memory, and warm replay reporting.
9. Static-server regression under its throughput-only shared-RNG contract.

The recent full-song consistency artifact is SHA
`483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631`,
size `31,709`, with `7,639` main and `821` timing tokens. Final migration
comparisons use same-job reciprocal controls.

K2/K3 current-runtime evidence through `f68cf2b` now completes the one-step,
8/256-step, reciprocal smoke, public-optimized-versus-delegated-legacy
full-song, and pre-migration-accepted-versus-final-delegated-legacy full-song
gates below. The numerical frontier remains `270.475 tok/s`; these are
architecture-neutral regressions, not a speed promotion. Final default V32,
compile-only V32, and reciprocal shared-RNG static-server characterization are
also recorded below.

## Stop And Rollback Policy

- Import cleanliness alone never graduates a boundary.
- Any token, stop, RNG, cache, or output mismatch stops the boundary.
- Any meaningful main, timing, stage-wall, cold setup, or per-window regression
  stops the boundary until fixed or reverted.
- No performance improvement is claimed or pursued.
- Every boundary commit records its predecessor for isolated rollback.
- Protected batching and decoder-layer audit branches are never merged.

## Boundary Evidence

### K1 accepted native q1 ownership move

Commit `b4e9926` moved the accepted q1 CUDA/C++ source byte-for-byte from the
legacy module to `optimized/kernels/q1_attention.py`. The source SHA remained
`c65438acc24d03f13fc0df694e29ddc2b21aff425cd01fef9d14814544febcd9`.
The legacy module is now a call-lazy compatibility shim. A neutral hook registry
is installed only by the enabled native runtime context, before model compile
or graph capture; the VarWhisper hot path performs no import.

Local gates passed `8` new import/loader/ABI/hook tests and `24` existing
control-plane, server, and logits tests. A fresh default V32 import loaded no
optimized, legacy-native, optimized-kernel, or `torch.utils.cpp_extension`
module.

DCC job `49560678` on RTX 2080 Ti, wrapper commit `31073f0`, passed the real
SALVALAI seq9 q_len=1 gate with active-prefix length `128`, q1 BMM cross,
native q1 self, fused RoPE/cache, and verifier DecodeSession enabled. Raw logits
were allclose (`max_abs=1.9073486328125e-05`), finite/nonfinite layout matched,
and top-20 IDs matched exactly. The report is
`/work/imt11/Mapperatorinator/runs/optimized-single-one-token-49560678-31073f0/one-token.json`
with SHA
`9ea09f187bc0ae1f388029dd8bf880ce07cbda1de72dee3df7f6b2d1d54b2057`.
The pre/post extension-cache inventories were identical with SHA
`5c4e9fab18def3140bf8b9e447f8b5267ea613cdfd4135c4f0b4e0931e73878e`;
the existing `mapperatorinator_q1_attention` cache entry was reused.

### S1 accepted verifier session ownership move

Commit `caa5c00` moved the verifier session and one-token helper source
byte-for-byte to `optimized/single/session.py`; SHA remained
`cbaddf37772773a725598d7f970465d52efebe485a568697efe95b5f1a72798c`.
The legacy `direct_decode.py` is now a lazy compatibility export. Optimized
batch/speculative modules import the source owner directly. This boundary did
not replace or modify the production session dictionary.

Local session/import/batch/speculative gates passed. DCC job `49560747` on
RTX 2080 Ti reran the same real SALVALAI seq9 path through the compatibility
export and optimized source. Raw logits were allclose with the same
`max_abs=1.9073486328125e-05`, finite/nonfinite layout and top-20 matched, and
the gate completed successfully. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-one-token-49560747-caa5c00/one-token.json`,
SHA `942a90b9bfb0f669524a2aa11bddf2fdeacf6f6f966e6e0122050451ed65af9f`.

### L1 accepted active-prefix loop ownership move

Commit `bb8d8a0` moved the active-prefix decode loop, bucket selection, stable
encoder holder, graph signatures, capture/replay, static input copies, and
diagnostics to `optimized/single/decode_loop.py`. The legacy module is a lazy
compatibility export. `server.py` imports the source owner only inside the
explicit active-prefix branch; a fresh ordinary server import remains
optimized-cold.

Local bucket, static-buffer, graph-signature, stable-encoder, compatibility,
and fresh-import gates passed. DCC job `49560776`, wrapper commit `cb1bc50`,
passed the 8-step direct-loop gate with exact generated tokens, exact final RNG,
all raw-logit/top-k steps matching, `max_new_tokens` stopping, one graph capture,
and seven replays. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-short-loop-8-49560776-cb1bc50/direct-loop.json`,
SHA `258fafe28bfaab710c6f49918a20db4edb2a7ab523302dd311bdf6c40af176bc`.

Reciprocal production smoke job `49560783` compared the accumulated migration
through L1 against pre-migration `3f4b088`. Both launch orders matched all
`1,084` main and `164` timing tokens and produced the same `.osu` SHA
`ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`. Candidate aggregate main model throughput was `+1.2%` and
`+0.3%`; timing was `+9.9%` and `+7.0%`; total stage wall improved in both
orders. The zero-tolerance strict comparator returned nonzero only for ordinary
sub-window timing jitter, so this is accepted as no meaningful regression, not
as a performance win.

### P1 accepted optimized stateful logits ownership move

Commit `f7d6c46` left the V32 full-scan monotonic processor in the legacy module
and moved the incremental batch-1 state, SOS reset, sequence-jump
reinitialization, and time-shift offset cache to
`optimized/single/logits.py`. Direct legacy construction with
`stateful_batch1=true` now fails with an explicit migration instruction;
`server.py` lazily selects the optimized owner only for the accepted opt-in
flag. The graph-safe verifier now replaces pristine legacy or optimized
processors without assuming optimized state lives on the V32 class.

Local full-scan, randomized B=1/2/5, growing-prefix, SOS/sequence-jump,
graph-safe, server-routing, import-cold, and broad `tests/` gates passed. DCC
job `49560829` on RTX 2080 Ti passed the real one-token gate with exact top-20,
raw logits allclose at `max_abs=1.9073486328125e-05`, and unchanged extension
cache inventory. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-one-token-49560829-f7d6c46/one-token.json`,
SHA `16d6cd77ce14b2585a305ccfd0590d1c0da41b96a59937cccaf79ea3fd307f0c`.
DCC job `49560834` passed the 8-step token/RNG/raw-logit/top-k/stop gate with
one graph capture and seven replays. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-short-loop-8-49560834-f7d6c46/direct-loop.json`,
SHA `5896d61fee4f0c23d8be37f42f65d5f15ce7d3094ab5197e469e6d559c5fa34d`.

Reciprocal production smoke job `49560849` compared `f7d6c46` against
pre-migration `3f4b088`. Both orders matched all `1,084` main tokens, all `164`
timing tokens, same-calculation metadata, and `.osu` SHA
`ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`. Candidate aggregate main throughput was `+1.3%` and `+1.2%`;
timing was `+10.0%` and `+9.8%`; stage wall improved `15.5%` and `15.2%`.
The strict job status was nonzero solely because its zero-tolerance per-window
gate flagged jitter: the largest main model-time increase was `2.195ms` and the
largest timing increase was `2.005ms`, while every aggregate metric improved.
This boundary is accepted as exact and without meaningful regression; no speed
claim is made. Comparison reports:
`/work/imt11/Mapperatorinator/runs/v32-integration-migration-p1-smoke-49560849/control-first-compare.json`,
SHA `bed422041409ce4d0e162e4bc7459c58779cd330e08d8938798b8e51ab7f77c9`,
and `candidate-first-compare.json`, SHA
`4a379632c40dd0818ffda397229ad6c6bd9f35c73fb2358cfaa6082043c11a39`.

### S2 accepted production session ownership move

Commit `29bff1f` replaced the production raw state dictionary with
`optimized/single/state.py:ProductionDecodeSession`. The typed owner contains
only the persistent cache, graph cache, and stable encoder holder. It allocates
the cache once per unfinished output context, resets self-attention then
cross-attention then `is_updated` before subsequent windows, and preserves
object identity. RNG, logits processors, stopping state, cache positions, and
tokens remain outside the persistent session exactly as before. Processor owns
only a branch-local lazy factory and the existing context reset points;
`server.py` now uses a narrow typed-state protocol and no longer implements
cache reset or graph/encoder dictionary ownership.

The full local `tests/` tree passed, with the isolated legacy-stub logits test
run in its own process. DCC job `49560908` passed the real one-token regression
gate at the same raw-logit tolerance and exact top-20. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-one-token-49560908-29bff1f/one-token.json`,
SHA `b044083d86e298bb18dc755cac63b45a891fce913386d25bb2555a618320b801`.

DCC job `49560943` exercised four consecutive SALVALAI windows for 64 tokens
each through the production state owner. All tokens, logits/top-k steps, stop
reasons, and final RNG matched; one cache identity was reused; the stable
encoder shape remained `[1, 1024, 768]`; and two unique prefix buckets produced
exactly two graph captures with `252` replays. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-session-64-49560943-29bff1f/persistent-session.json`,
SHA `65e2683de32519afbe69c59b0001c17fe69d06f7a40fd7a119e9196af094eea8`.
The promoted 256-token job `49560951` also passed all four windows (`1,024`
candidate tokens total), exact final RNG, stable identities/shape, and five
captures for five unique prefix buckets with `1,020` replays. Report:
`/work/imt11/Mapperatorinator/runs/optimized-single-session-256-49560951-29bff1f/persistent-session.json`,
SHA `5a0e11f48a1d8c825df6666d222654d9da7d07980ecd2ecdd984a74574204367`.

Reciprocal production smoke job `49560956` compared the accumulated migration
through S2 against pre-migration `3f4b088`. Both orders matched all `1,084`
main tokens, all `164` timing tokens, same-calculation metadata, and `.osu` SHA
`ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`. Candidate aggregate main throughput was `+1.3%` and `+0.8%`;
timing was `+8.6%` and `+3.6%`; stage wall improved `13.8%` and `11.9%`.
The zero-tolerance strict status was nonzero only for window jitter. The largest
isolated main model-time increase was `7.649ms` and timing increase was
`15.417ms`, while every aggregate metric improved. No speed claim is made.
Comparison reports:
`/work/imt11/Mapperatorinator/runs/v32-integration-migration-s2-smoke-49560956/control-first-compare.json`,
SHA `35148f9cf59880ea1099c09fbc98e25fbc0c77a0e4bf24c9ed422971c6e8afa2`,
and `candidate-first-compare.json`, SHA
`6b8d3b554712478b6a5a9e27412593560b74e9fc8ca4dd1e2ea8e2f2f8299769`.

### E1 accepted real optimized single engine

Commit `52bfeaf` made `inference_engine=optimized
optimized_inference_mode=single` a working runtime. A neutral
`InferenceEngineBinding` unwraps once in Processor so the raw Hugging Face model
retains its identity on model, cache, compile, and graph paths. The lazy adapter
validates unsupported modes before importing `optimized.single` or invoking the
injected raw loader. Single mode forces the frozen
`accepted-fp32-270.475-v1` effective config and generation compile without
rewriting requested legacy flags. Processor creates one production session per
unfinished context and delegates each window to optimized-owned generation
orchestration. Requested metadata remains unchanged; effective config/version,
runtime owner, and result class are reported separately.

The full local `tests/` tree passed. Adapter/selector tests prove that
`offline_batch` and `server` fail before single-engine import/model loading,
the raw loader is called once with FP32/SDPA/compile enabled, the effective
configuration is immutable and equals the accepted legacy stack, sampling
settings remain request-owned, and binding unwrap preserves raw-model identity.
Default V32 import and server tests remain optimized/native cold.

DCC pair job `49561071` was the first end-to-end execution through the public
optimized selector. It matched all `1,084` main and `164` timing tokens and
the `.osu` SHA
`ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`, against the canonical legacy bundle. Aggregate optimized main
throughput differed by `-0.5%`, timing by `-0.1%`, and stage wall by `+0.7%`,
all below the meaningful-regression threshold. Comparison report:
`/work/imt11/Mapperatorinator/runs/optimized-single-engine-e1-pair-49561071/legacy-first-compare.json`,
SHA `ff24425b8911d142b84cbb771f77c064cd3df110b0fe71b8c9d9255bd1c365ff`.

Reciprocal job `49561104` confirmed the result in both launch orders. All
main/timing tokens and `.osu` bytes matched. Optimized main throughput was
`-0.2%` with legacy launched first and `+0.9%` with optimized launched first;
timing was `+5.0%` and `+4.3%`; stage wall improved `15.9%` and `12.7%`.
No performance win is claimed. Effective metadata reported compile enabled and
bucket `64` while requested `inference_generation_compile` remained false.
Reports:
`/work/imt11/Mapperatorinator/runs/optimized-single-engine-e1-reciprocal-49561104/legacy-first-compare.json`,
SHA `e416b81cd5907d6683573abbe0fc9846712c1a75307c1e99185e04c480e6c9d5`,
and `optimized-first-compare.json`, SHA
`fcc11af93d7a3e527632eb88c760d87e9ff52852514e2f8dc2e56e870642e992`.

### C1 accepted legacy delegation and server cleanup

Commit `efaa4a6` made the canonical legacy micro-flag bundle construct the same
`OptimizedSingleRuntime` and frozen config as the public optimized selector.
The compatibility adapter is default-cold; compile-only V32 does not trigger
it. Any partial optimized legacy bundle fails with an explicit instruction to
use the complete accepted bundle or `inference_engine=optimized`.

The same boundary removed active-prefix loop selection, production session
reset/graph/encoder ownership, stateful optimized logits construction, q1
activation, native eligibility, and optimized stats calculation from
`server.py`. Ordinary V32 still reports the same false/`None` optimized
metadata keys and calls ordinary `model.generate()`. Optimized batch and
speculative modules now obtain stateful processor implementations from
`optimized/single/logits.py`, not the server. The complete local `tests/` tree
passed, including source ownership, default-cold compile-only, server/static
state, selector, and partial-bundle failure gates.

DCC reciprocal job `49561276` compared the canonical legacy selector with the
public optimized selector after delegation. All main/timing tokens and `.osu`
bytes matched in both orders. Optimized main throughput was `+1.9%` and
`+0.0%`; timing was `+10.2%` and `+1.7%`; stage wall improved `15.7%` and
`12.3%`. Reports:
`/work/imt11/Mapperatorinator/runs/optimized-single-engine-c1-reciprocal-49561276/legacy-first-compare.json`,
SHA `f89a16b06753a75e8bb5e1fb4847e63ede473d944b5ed62bb747d9bde6c71203`,
and `optimized-first-compare.json`, SHA
`0d5e99fddd23dddf8c17c0c0b38de93b54c086ce7447e2f73fbbf1b42e027a37`.

Default-V32 reciprocal job `49561279` compared candidate C1 against
pre-migration `3f4b088` with generation compile and every optimized flag off.
Both orders matched all `1,084` main tokens, all `164` timing tokens,
same-calculation metadata, and `.osu` SHA
`ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`. Candidate default V32 aggregate main improved `1.6%` and `1.9%`,
timing improved `0.7%` and `1.6%`, and stage wall improved `1.7%` in both
orders. Zero-tolerance strict status was nonzero only for isolated window
jitter. Reports:
`/work/imt11/Mapperatorinator/runs/v32-integration-default-v32-c1-49561279/control-first-compare.json`,
SHA `70776976015f1fa81544326fd707970a234e8c9845428583fea762a6b47735e3`,
and `candidate-first-compare.json`, SHA
`8dc3fcfde4bde9becc6fdd641b184604361a0db1c005e7777bd4b58aae236e66`.

Compile-only V32 reciprocal job `49561361` also matched every main/timing token
and `.osu` byte. The candidate-first reciprocal pair was performance-neutral
(`-0.03%` main, `-0.16%` timing, `-0.11%` stage); the other order exposed only
compile/cache-order benefit. Compile-only therefore remains generic and does
not dispatch to optimized single. Reports:
`/work/imt11/Mapperatorinator/runs/v32-integration-compile-only-v32-c1-49561361/control-first-compare.json`,
SHA `9ea7d3cd1f8464cee86a55991017d93d0a3d8b390d54e1e2c930c78e604eb465`,
and `candidate-first-compare.json`, SHA
`57e1dc12321d0ba5b02e50a16bdb53fa77a8cd70a56e55f8e15a672adf7485b2`.

### K2 accepted optimized attention dispatch ownership

Commit `2ed6620` moved q_len=1 BMM cross-attention, native q1
self-attention, fused RoPE/cache orchestration, eligibility, and exact operation
order out of `modeling_varwhisper.py` and into
`optimized/kernels/dispatch.py`. `optimized/single/runtime_context.py` owns
lazy kernel preload and installs the selected optimized callables through the
neutral `inference/runtime_dispatch.py` registry. VarWhisper now performs only
neutral hook calls and contains no optimized eligibility or native-kernel
implementation.

Commit `a633124` is a mechanical ownership cleanup within the existing rejected
v32-mini scout. The scout now owns its approved stateful-monotonic choice under
`optimized/speculative/mini_draft_gpu.py` instead of requesting a partial
legacy runtime bundle. It introduces no new speculative behavior, evidence, or
revisit authorization.

### K3 accepted active-prefix attention ownership

Commit `f68cf2b` moved active-prefix length state and self-attention input
trimming from `runtime_profiling.py` and VarWhisper into
`optimized/single/runtime_context.py` and
`optimized/kernels/dispatch.py`. The optimized decode loop/session installs the
context explicitly; the model sees only the neutral input hook. Default V32,
compile-only V32, and direct server generation neither own nor select this
state. Commits `05e512f` through `7326a62` change no runtime calculation; they
add and harden the final static-server/import regression wrapper only.

Current-runtime DCC gates on an RTX 2080 Ti passed through `f68cf2b`:

- Job `49562049`: real one-token gate PASS; raw-logit maximum absolute
  difference `1.907e-05` and exact top-20 IDs.
- Job `49562055`: 8-step token/logit/final-RNG gate PASS; one graph capture and
  seven replays.
- Job `49562091`: 256-step exact gate PASS; five graph captures and `255`
  replays.
- Reciprocal 15-second job `49562101`: both launch orders matched all `1,084`
  main and `164` timing tokens plus `.osu` SHA
  `ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
  size `4,144`. Optimized main throughput was `+0.7%` and `+0.4%`, timing was
  `+9.7%` and `+4.2%`, and stage wall was `-15.5%` and `-11.3%`. This is
  no-regression evidence, not a speed claim.

Reciprocal full-song job `49562130` compared the public optimized selector with
the complete delegated legacy bundle on RTX 2080 Ti node `z25-20`. Both orders
matched all `7,639` main and `821` timing tokens plus `.osu` SHA
`483483a1c29ef8a44c4a8d3a82fe0778ae306470ec3157e98969eeabd92c2631`,
size `31,709`. With legacy launched first, main was `268.707 -> 269.120 tok/s`
(`+0.2%`), timing `+0.5%`, and stage wall `-0.2%`. With optimized launched
first, main was `268.882 -> 268.700 tok/s` (`-0.1%`), timing `-0.3%`, and stage
wall `-0.1%`. The strict job exited `1` only because zero-tolerance per-window
jitter failed after the exact aggregate/output gates passed. Run root:

```text
/work/imt11/Mapperatorinator/runs/optimized-single-engine-full-song-final-49562130
legacy-first-strict-full.json SHA 51a1b763faea816947f64712ab44065c6cbb369761ff0a60de5eb39fbd88ea7e
optimized-first-strict-full.json SHA 55b395cdf4316b9496b6eb77991946315935f50810645a303d735f32dad4ffe2
```

Reciprocal full-song job `49562226` then compared pre-migration accepted legacy
at `3f4b088` with the final delegated legacy runtime. Both orders again matched
all `7,639` main and `821` timing tokens and the same output SHA/size. In the
control-first order, main was `268.850 -> 268.919 tok/s` (`+0.0%`), timing was
`-0.7%`, outer wall `+0.8%`, and stage wall `+0.6%`. In the candidate-first
order, main was `267.813 -> 267.621 tok/s` (`-0.1%`), timing `-0.4%`, outer wall
`+0.7%`, and stage wall `+0.6%`. Slurm status `FAILED` reflects only the strict
zero-tolerance comparator after valid exact profiles, not a generation or
output failure. Run root and comparison reports:

```text
/work/imt11/Mapperatorinator/runs/v32-integration-full-migration-49562226
control-first-compare.json SHA 2761eb6fb34874d31adb42aa9cbd56a7d9ea06a18c4f20ddc1098d07cfdf569a
candidate-first-compare.json SHA f3c4ff1a8df34a7a08ce694dbe6ee3c86197b13e65aa265fae06013e92cbf954
```

Absolute main throughput in both full-song jobs drifted below the historical
`270.475 tok/s` frontier equally for controls and candidates. The accepted
frontier therefore remains `270.475 tok/s`; K2/K3 are exact-output,
architecture-neutral ownership migrations.

### Final default and compile-only V32 regression gates

Reciprocal default-V32 smoke job `49562311` compared final runtime commit
`f68cf2b` with pre-migration `3f4b088` on RTX 2080 Ti node `z25-20`, with
generation compile and every optimized feature disabled. Both orders matched
all `1,084` main and `164` timing tokens, same-calculation metadata, and `.osu`
SHA `ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
size `4,144`. Control-first main/timing changed `-0.8%/-1.6%`, while
candidate-first changed `+0.4%/+2.0%`; stage wall improved `0.1%` and `0.2%`.
The reciprocal sign change and sub-2% magnitude are architecture-neutral.
Slurm status `FAILED` reflects only the strict zero-tolerance per-window gate.

```text
/work/imt11/Mapperatorinator/runs/v32-integration-final-default-v32-49562311
control-first-compare.json SHA a2a348c7c67e5479a7d953b82f7a146942698169bbf912176bd5c617556f9373
candidate-first-compare.json SHA bc97902c43a73c0fba1dd200211ed594b12a2f3b19040a7a9124db2cae3de4fa
```

Reciprocal compile-only V32 smoke job `49562444` repeated the gate on RTX 2080
Ti node `h36-5`. Both orders again matched all main/timing tokens, metadata,
and output bytes. Main throughput changed `+0.4%` and `+0.5%`. The first
control paid a compiler-cold timing specialization cost; the warm reciprocal
pair changed timing `+2.1%` and stage wall `-1.8%`. This proves compile-only
remains token/output-equivalent and aggregate-neutral, not a speed result. No
final RNG hash was recorded, so this is not labeled full exact-output. Both
comparison statuses were nonzero only for zero-tolerance per-window jitter.

```text
/work/imt11/Mapperatorinator/runs/v32-integration-final-compile-v32-49562444
control-first-compare.json SHA 00c593fcde6b04bc1b606cb36c65134748f7aa71e1fc2c8011aafea02607f5b8
candidate-first-compare.json SHA 430badec03553a2c51f93c12707c73177820d849814da03b6423ea19524d4164
```

### Final static-server regression characterization

Mixed five-song static-server jobs `49562635` and `49562710` ran the pinned
pre-migration control and migrated V32 server in reciprocal launch order on RTX
2080 Ti. Both jobs ended `FAILED 1:0`: the intended strict comparator remained
red after valid profiles, rather than inference failing. Both sides
self-validated real static batching and retained
`same_calculation=false`, `server_rng_policy=shared_global`, and
`token_equivalence_status=not_checked_shared_server_rng`. Every underlying
profile reported `inference_engine=v32`, no optimized runtime owner/config/result
class, and all optimized feature states false. The candidate fresh-import audit
loaded no optimized/native/cpp-extension modules and both isolated extension
directories remained empty; the control audit reproduced its frozen eager
legacy q1/`torch.utils.cpp_extension` imports without compiling an extension.

Scheduler-wall results changed sign with launch order. Control-first job
`49562635` measured `152.558 -> 120.859 tok/s` (`-20.8%`) with `6,815` versus
`6,292` main tokens. Candidate-first job `49562710` measured
`106.933 -> 115.261 tok/s` (`+7.8%`) with `7,173` versus `7,057` main tokens.
Both strict comparisons correctly remained red because shared global RNG and
concurrent request ordering produced different stop lengths/work; neither run
is exactness or same-work throughput evidence.

The post-hoc B5-only diagnostic explains the apparent regression without
replacing the scheduler metric. In job `49562635`, control and candidate
executed `1,961` versus `2,478` longest-row active decode steps across ten B5
batches; normalized B5 throughput was `67.098 -> 67.984` active steps/s
(`+1.3%`). In reverse-order job `49562710`, after reporting and excluding B1-B4
tails, normalized B5 throughput was `56.946 -> 70.402` (`+23.6%`). The migrated
server had no negative normalized B5 result, but active-step/prefix work still
differed. Server performance therefore remains non-comparable under the
existing shared-RNG workload; ownership/import/metadata compatibility is the
accepted boundary. No server throughput win is claimed.

```text
/work/imt11/Mapperatorinator/runs/static-server-regression-final-smoke15-r2-49562635
compare-static-server.json SHA fd83cde92fc89bef8a6fea49113f3ae7129a3c16a8399c642cc4d9696cff3ad6
work-normalized-static-server.json SHA 8369d4a26ad40b4f767824712c70ffab07f4c7b56c5f8ea43ae21e5e06f55cb8

/work/imt11/Mapperatorinator/runs/static-server-regression-final-smoke15-candidate-first-49562710
compare-static-server.json SHA 1c8f6a79903df8047de89f0dcb7904c08294ce6e226026185b6d2486434adaa0
work-normalized-static-server.json SHA 490d762830b6472aed90f9bce673e0bd8a2c13386c315019c38eeef0282a931a

control import audit SHA 2accf09898b894ee642ca00ec09bb6c96402706773b73456f89e565f3e55d014
candidate import audit SHA cddfa51b70ec3fd9f483fceb4f48a3b45640d2cffe5e40e4148a430a955638d7
```

Together with token/output-equivalent default/compile-only runs and the exact
optimized/delegated-legacy full-song gates, this accepts the migration as an
architecture-neutral ownership change. Static server performance remains
unproven/non-comparable; its ownership/import/metadata contract is preserved.
The historical `270.475 tok/s` single-song frontier remains unchanged.

Final local verification passed `289` general tests plus the separate `2`
legacy-logits tests. A final focused server/profile run passed `32` tests, and
the static Slurm wrapper passed `bash -n` and `git diff --check` after its
post-processing-only changes.
