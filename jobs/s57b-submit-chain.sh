#!/usr/bin/env bash
# Submit §57b dump→shards→train chain (1 GPU at a time; afterok deps).
# Usage: bash jobs/s57b-submit-chain.sh
set -euo pipefail
REPO=${PROBE_REPO:-/hpc/group/romerolab/imt11/projects/Mapperatorinator-worktrees/turbo-continuation-distill}
WORK=/work/imt11/Mapperatorinator
DUMP_SBATCH="$REPO/jobs/s57b-turbo-short-rest-dump.sbatch"
SHARDS_SBATCH="$REPO/jobs/s57b-continuation-shards.sbatch"
TRAIN_SBATCH="$REPO/jobs/s57b-continuation-distill-train.sbatch"

SONGS=(pegasus lambada)
for extra in ela-ke-leitada nube-negra; do
  if [[ -f "$WORK/data/five-song-profile/${extra}.mp3" ]]; then
    SONGS+=("$extra")
  fi
done

echo "songs=${SONGS[*]}"
echo "repo=$REPO"
echo "commit=$(git -C "$REPO" rev-parse --short HEAD)"

prev=""
dump_jobs=()
for song in "${SONGS[@]}"; do
  export S57B_SONG="$song"
  export MAPPERATORINATOR_AUDIO="$WORK/data/five-song-profile/${song}.mp3"
  if [[ -n "$prev" ]]; then
    jid=$(sbatch --parsable --dependency=afterok:"$prev" --export=ALL "$DUMP_SBATCH")
  else
    jid=$(sbatch --parsable --export=ALL "$DUMP_SBATCH")
  fi
  echo "dump_$song=$jid"
  dump_jobs+=("$jid")
  prev="$jid"
done

shards_jid=$(sbatch --parsable --dependency=afterok:"$prev" --export=ALL "$SHARDS_SBATCH")
echo "shards=$shards_jid"

export S57_RESUME_CKPT="$WORK/runs/s57-continuation-distill-50182001/draft_continuation.pt"
export S57_TRAIN_STEPS=4000
export S57_SONG_FILTER=""
export S57_SHARDS_DIR="$WORK/runs/s57b-continuation-shards"
train_jid=$(sbatch --parsable --dependency=afterok:"$shards_jid" --export=ALL "$TRAIN_SBATCH")
echo "train=$train_jid"

echo "CHAIN dumps=${dump_jobs[*]} shards=$shards_jid train=$train_jid"
