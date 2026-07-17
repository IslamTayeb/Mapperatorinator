# Inference Decision Ledger

This is a family-level decision index, not a chronological lab notebook.
Deleted experiment narratives remain available through Git history; large
reports and traces remain in their recorded DCC locations while retained.

| Family | Decision | Durable lesson | Revisit only when |
| --- | --- | --- | --- |
| Optimized single | Accepted, opt-in. SALVALAI FP32/SDPA reached `270.475 tok/s` with exact tokens and output (`49230082`). Cleanup smoke `49708641` preserved performance; upstream-merge smoke `49709859` / `9a8b8fc` preserved reciprocal V32/optimized tokens and bytes. | Persistent session state, active-prefix graphs, stateful monotonic processing, and q1 kernels form the useful production stack. | A current exact full-song comparison demonstrates a further end-to-end gain. |
| Optimized architecture | Accepted. Runtime and kernel ownership moved under `inference/optimized/`; V32 stays cold and `server.py` stays V32-only. | Isolation and lazy selection are compatibility requirements, not cleanup preferences. | A separately reviewed public boundary requires change. |
| Measurement/verifiers | Retained only as a compact synchronized profile, reciprocal comparator, output hash, and focused runtime exactness tests. | Keep the smallest gate that directly supports a live production claim. | Expand only for a concrete measured hypothesis, then delete candidate-specific machinery after the decision. |
| Additional configuration and Python-loop variants | Cut outside the accepted optimized-single bundle. Dead Hugging Face generation-compile selection was removed at `6466b43`; the returned compiled callable was unreachable under the accepted step-one manual CUDA-graph path. Reciprocal FP32/FP16 smoke stayed exact and improved total stage wall, although one FP16 order showed a `6.194%` synchronized-main timing outlier. Other compile modes, cache variants, preparation shortcuts, and stopping specializations were flat, unstable, or below `5%`. | Apparently enabled compile controls are not production compile regions; distinguish wrapper construction from executed compiled forwards and measure cold wall separately from steady model time. | A current profile shows an eager region is target-sized, or repeated FP16 measurements establish a steady regression from the cleanup. |
| Additional narrow kernels and fusions | Mostly cut outside the accepted q1 paths. Isolated linear, MLP, attention, replay-shell, and cache-copy wins lacked enough weighted end-to-end headroom. | Weight traffic, launch overhead, and prefix distribution matter more than a best-case microbenchmark. | A broader real-prefix ceiling clears the end-to-end bar before runtime work. |
| Whole decoder/layer replacements | Cut. Exact or near-exact bounded gates did not clear their predeclared full-song savings bar. Never merge `experiment/decoder-layer-runtime-island-do-not-merge` wholesale. | Broad rewrites need a strong weighted ceiling because integration risk is high. | A materially different implementation first clears the reciprocal real-prefix gate. |
| Speculative decoding | Cut. N-gram and mini-draft proposals preserved target semantics in bounded tests but proposal/verification cost exceeded accepted q1 decoding. | Acceptance rate is irrelevant without a cheaper complete accepted-token path. | A draft plus verifier demonstrates a current-stack end-to-end ceiling above `5%`. |
| Static/server batching | Cut as an exact optimization. Shared RNG, changed work, coalescing delay, and poor Turing batch scaling kept it below optimized serial. Never merge `codex/batched-fast-decode-session`. | Server throughput diagnostics are not an exact offline queue. | A private-state offline engine wins before any server adapter is considered. |
| Independent lanes and merged batches | Cut on current Turing FP32 stack. Model-only overlap and fixed-prefix B2 islands looked fast, but complete sampling/control and real mixed prefixes erased the win. | Complete-step cost and compatibility coverage dominate model-only physics. | A complete exact step beats optimized B1 and real profiles show enough compatible work. |
| Practical two-song queue | Rejected by a bounded main-generation wall scout on `codex/phase-pipeline-scout` (`49627068`) before the complete queue-wall suite. Outputs were exact, but only `91` steps paired versus `2,625` private steps; candidate was about `75 tok/s` and took `2.6-3.8x` optimized serial main wall. | Sparse compatibility plus an expensive fallback makes this queue shape fundamentally negative. | A materially different shape first wins reciprocal N=2 scheduler-main and complete request wall above the optimized-single frontier. |
| Phase overlap | Cut. Reciprocal model-versus-control-tail ordering produced a negative worst-order same-work result (`49619389`). | An overlap fantasy must include contention, startup, and drain in both orders. | A different phase partition wins both reciprocal same-work controls. |

## Current boundary

No batching, speculative, broad-kernel, scheduler, or optimized-server candidate
is authorized. Improve exact optimized single or produce new current-stack
evidence that invalidates a family-level revisit condition. Do not rerun an old
idea merely because its isolated number was promising.

Historical artifact hashes, intermediate failures, and per-job commands are
evidence, not standing instructions. Use Git history only when auditing a
specific old decision.

**§48 W-VG graph-native K=γ verify — OPEN** (2026-07-17): branch `codex/turbo-graph-native-verify`. Lifted k>1 graph gate; static `{ids[1,γ], cache_position[γ]}`; mask in-graph; no HF prepare_inputs on hot path; production side-stream warmup; persistent verify_fp across windows. In-loop probe `utils/s48_graph_native_verify_inloop.py`. Gate ≤1.2× / kill >1.35×. Handoff: `notes/500tps-section48-handoff.md`. Tip still `55949274` / **366.11**. No 500 claim.
