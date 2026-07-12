# Dependency-Aware Five-Song K3 Ceiling

This CPU-only gate consumes the pinned weighted parent report
`notes/inference-weighted-bucket-ceiling-report.json`, file SHA-256
`44a680ab29867e3aea8dde713127bdb154ef42a316fd2834bc75afbbc0927fc9`.
It does not run a model or GPU and authorizes no scheduler, runtime, or server
work.

The accepted five-song workload has five songs and ten main windows per song.
Only each song's first window may be prepared before first-main. The other
`45` windows depend on their predecessor's generated output and therefore keep
the accepted per-window linear setup charge inside first-main-to-last-main:

```text
per-window setup = (0.34391318541020155 + 0.04675073456019163) / 8
                 = 0.04883298999629915 s
```

The self-validating transition ledger contains exactly `45` ordered
song/sequence transitions, totals `2.1974845498334616s`, and has canonical
SHA-256 `cf60a51f1581f7d477453bd359269f5b4228c44d1c79d6ff3680f2d7513994ba`.

| Scenario | Charged setup | Wall | Main tok/s | 525 bar |
| --- | ---: | ---: | ---: | --- |
| setup-free fantasy | `0s` | `9.293543s` | `640.767` | pass |
| all 50 setups charged | `2.441649s` | `11.735193s` | `507.448` | fail |
| five initial excluded, 45 dependent charged | `2.197485s` | `11.491028s` | `518.230` | fail |

The strict `>525 tok/s` bar requires less than `11.342857s`. The dependency-aware
fantasy exceeds it by `0.148171s` and has only `3.65%` headroom over `500`,
below the campaign's `5%` keep bar. Therefore a K3 GPU scout is not authorized.
Revisit only after accepted evidence moves or removes enough in-wall setup, or
a materially different current-stack ceiling changes the parent K3 wall.

The local CLI-generated report SHA-256 is
`b50fb9f7d75f54b3c597e00cadad076ed50110c51626ab62410eb43cce426c01`.
This exhausts currently authorized candidates under the current serial-setup
policy; it is not a physical claim that no FP32 design can work.
