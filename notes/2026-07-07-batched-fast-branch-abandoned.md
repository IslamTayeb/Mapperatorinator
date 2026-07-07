# Batched Fast DecodeSession Branch Abandoned

The unmerged branch `codex/batched-fast-decode-session` is abandoned at
commit `a74537a`. It should remain available as an audit trail, but should not
be merged into `main`.

The branch tried to push the accepted single-song fast path through server
batching and continuous/static decode experiments. The best exact compiled
server lockstep result was B=5 identical SALVALAI at `263.544` unique main
tok/s, below both the accepted full-song single-song baseline (`270.475` tok/s)
and the paired 15s non-server fast reference (`288.703` tok/s). The B=10 check
was slower (`199.721` unique main tok/s), fragmented, approached the RTX 2080 Ti
VRAM ceiling, and was non-equivalent.

Decision: for 5+ songs on one RTX 2080 Ti, prefer the optimized single-song path
serially in one long-lived process. Do not promote the post-main server fast
batching branch unless a future lower-level active-prefix graph-step or batched
decoder runtime proves exact per-request output behavior and beats optimized
serial throughput.

Rollback smoke validation ran after returning the working checkout to `main`.
DCC job `49325496` on `dcc-core-ferc-s-z25-20`, commit `f6add76`, used
`profile_salvalai_smoke15` with the accepted fast single-song opt-in flags,
fp32, SDPA, `use_server=false`, and `parallel=false`. Main generation produced
`1,084` tokens in `3.867s` synchronized model time, or `280.3` tok/s. The output
SHA was `ff63c232115906483a592c940e6f0fccbb8639775378d39fd86237f4191ed4ba`,
matching the prior 15s fast-path reference. Profile path:
`/work/imt11/Mapperatorinator/runs/rollback-fastpath-smoke-49325496-f6add76/profile/beatmap13bc54a39d704a799e211e79b1f60d88.osu.profile.json`.
