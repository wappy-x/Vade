#!/bin/bash
# Usage: bash TCGATE-run.sh COMMIT two|four TAG "PARTS" [CONTROL]
# One waiting-line turn. TCGATE_SEAT defaults to TCGATE. No claim logic is changed.
set -u
N=${VADE_NIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; K=$N/kit
sha=${1:?commit}; pair=${2:?pair}; tag=${3:?tag}; parts=${4:?parts}; control=${5:-}; seat=${TCGATE_SEAT:-GATE2}
case "$pair" in one|two|three|any) ;; *) echo 'Use one, two, three, or any; served runs never use four.' >&2; exit 2;; esac
[ ! -e "$N/runs/$tag" ] && [ ! -e "$N/runs-$tag.out" ] || { echo 'Tag already used' >&2; exit 2; }
cat "$N/inbox/ONEROW.md"
[ "$seat" = TCGATE ] || cat "$N/inbox/$seat.md"
while :; do
  p=$(/bin/bash "$K/queue.sh" take "$pair" "$seat" 12 "gate $tag: $parts"); rc=$?
  [ "$rc" = 9 ] && continue
  [ "$rc" = 0 ] || exit "$rc"
  if [ "$p" = four ]; then
    /bin/bash "$K/queue.sh" done "$seat"
    pair=two
    continue
  fi
  /bin/bash "$K/GATE2-go.sh" gate "$p" "$sha" "$tag" "$parts" "$control"; rc=$?
  while [ "$rc" = 9 ]; do /bin/bash "$K/wait.sh" "$tag" 45; rc=$?; done
  /bin/bash "$K/queue.sh" done "$seat"
  exit "$rc"
done
