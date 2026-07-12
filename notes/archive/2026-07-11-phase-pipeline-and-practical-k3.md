# Phase And Practical Queue Evidence

Phase-overlap job `49619389` at `8830a5f` was exact and resource-private, but
its reciprocal model-versus-control-tail fantasy ranged from `-9.998%` to
`+4.218%` against same-job serial. The worst-order result rejected the overlap
DAG. Report SHA:
`b5591c8a16c5042b8da3e960781c8601bca785b2cc4c03d362660c039cdcc242`.

The practical objective then became an exact queue that beats optimized serial
main wall and exceeds the approximately `270.475 tok/s` single-song frontier.
Incremental Hybrid-B2 evidence passed:

- real-prefix H8: `49622288` / `ceb062f`, exact/private, worst `512.959 tok/s`;
- changing-prefix multi-bucket: `49624973` / `8475908`, exact across buckets `576..832`;
- complete-window handoff: `49625905` / `37fa210`, exact through EOS and successor preparation;
- complete two-song output: `49626469` / `e420006`, all main windows and final `.osu` bytes exact.

The bounded reciprocal main-generation wall scout `49627068` at `22e31fc`
then rejected the execution shape before the complete queue-wall suite. For the
same `2,827` main tokens, Hybrid-B2 took `37.669528-37.747532s`
(`74.892-75.047 tok/s`) while optimized serial took
`10.018026-14.366688s` (`196.775-282.191 tok/s`). Only `91` steps paired;
`2,625` used the slow private fallback. Both final Lambada and Pegasus `.osu`
files matched accepted SHA/bytes.

Report:
`/work/imt11/Mapperatorinator/runs/two-song-wall-scout-49627068-22e31fc/two-song-wall-scout.json`,
SHA `f8b328853573a1479b0ca455023a9d9869e16671194fffe7ad076b98f89ea292`.

Stop this candidate at N=2; N=3-5 and production policy/runtime work were not
authorized. Revisit only with a materially different complete-step shape that
first wins reciprocal N=2 scheduler-main and complete request wall.
