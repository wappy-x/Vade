#!/bin/bash
# Usage: bash TCGATE-run.sh COMMIT two|four TAG "PARTS" [CONTROL]
# One waiting-line turn. TCGATE_SEAT defaults to TCGATE. No claim logic is changed.
set -u
N=${VADE_NIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; K=$N/kit
sha=${1:?commit}; pair=${2:?pair}; tag=${3:?tag}; parts=${4:?parts}; control=${5:-}; seat=${TCGATE_SEAT:-TCGATE}
case "$pair" in two|four) ;; *) echo 'Use pair two or four; pair one belongs to the train.' >&2; exit 2;; esac
[ ! -e "$N/runs/$tag" ] && [ ! -e "$N/runs-$tag.out" ] || { echo 'Tag already used' >&2; exit 2; }
cat "$N/inbox/TCGATE.md"
[ "$seat" = TCGATE ] || cat "$N/inbox/$seat.md"
while :; do
  p=$(/bin/bash "$K/queue.sh" take "$pair" "$seat" 12 "gate $tag: $parts"); rc=$?
  [ "$rc" = 9 ] && continue
  [ "$rc" = 0 ] || exit "$rc"
  /bin/bash "$K/TCGATE-go.sh" gate "$p" "$sha" "$tag" "$parts" "$control"; rc=$?
  while [ "$rc" = 9 ]; do /bin/bash "$K/wait.sh" "$tag" 45; rc=$?; done
  /bin/bash "$K/queue.sh" done "$seat"
  exit "$rc"
done
