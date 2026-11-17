#!/bin/bash
# One final served comparison: current main, candidate, candidate, current main.
# Call only after quality passes and the candidate word proof is complete.
set -eu
N=${VADE_NIGHT_ROOT:-$HOME/Desktop/Vade/benchmarks/2026-11-14/night-limit}
K=$N/kit
base=${1:?main commit}; head=${2:?candidate commit}; tag=${3:?new tag}
pair=${4:-any}
for cell in m1 n1 n2 m2; do
  [ ! -e "$N/runs/$tag-$cell" ] && [ ! -e "$N/runs-$tag-$cell.out" ] || { echo "Existing speed cell $cell; inspect it before continuing"; exit 2; }
done
while :; do
  rc=0; p=$(bash "$K/queue.sh" take "$pair" GATE2 38 "final main/new/new/main read plain chats draft: $tag") || rc=$?
  [ "$rc" != 9 ] || continue
  [ "$rc" = 0 ] || exit "$rc"
  case "$p" in one|two|three) break;; *) bash "$K/queue.sh" done GATE2; pair=two;; esac
done
trap 'bash "$K/queue.sh" done GATE2' EXIT
for cell in m1 n1 n2 m2; do
  case "$cell" in m*) sha=$base;; *) sha=$head;; esac
  cat "$N/inbox/GATE2.md" "$N/inbox/ONEROW.md"
  date; uptime
  rc=0; bash "$K/KITF-go.sh" measure "$p" "$sha" "$tag-$cell" 'one chats read draft' || rc=$?
  while [ "$rc" = 9 ]; do rc=0; bash "$K/wait.sh" "$tag-$cell" 45 || rc=$?; done
  [ "$rc" = 0 ] || exit "$rc"
  r=$N/runs/$tag-$cell
  [ -f "$r/LOADS-DONE" ] || exit 3
  grep -E 'CACHE_(HIT|MISS)|flash-serve$|flash.ptx$' "$r/build-T.out"
  date; uptime
done
echo "Bracket complete on $p. Engine timers, same-word checks, ruling D and the base spread still require grading."
