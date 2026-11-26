#!/bin/bash
# Real-model full-logit checks. Claim/release remain in the copied kit driver.
set -u
N=${VADE_NIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; K=$N/kit
sha=${1:?commit}; tag=${2:?tag}; pair=${3:-any}
cat "$N/inbox/GATE2.md"; cat "$N/inbox/ONEROW.md"
while :; do
  p=$(bash "$K/queue.sh" take "$pair" GATE2 12 "model word proof $tag"); rc=$?
  [ "$rc" = 9 ] && continue
  [ "$rc" = 0 ] || exit "$rc"
  if [ "$p" = four ]; then bash "$K/queue.sh" done GATE2; pair=two; continue; fi
  bash "$K/GATE2-pair-go.sh" pair "$p" "$sha" "$tag" \
    'decode 2 max=32 --decode-rows-from=2 --device-wire' \
    'decode 4 max=32 --verify-rows=2,4,8 --device-wire' \
    'decode 1 max=8 --batch-steps=8 --batch-prefix-sets --device-wire'; rc=$?
  while [ "$rc" = 9 ]; do bash "$K/wait.sh" "$tag" 45; rc=$?; done
  bash "$K/queue.sh" done GATE2
  exit "$rc"
done
