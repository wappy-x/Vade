#!/bin/bash
# Run one prepared cell inside the lead's dedicated pair-three gate block.
# The normal kit driver retains every claim, stop, release and cleanup action.
set -eu
N=${VADE_NIGHT_ROOT:-$HOME/Desktop/Vade/benchmarks/2026-11-14/night-limit}
K=$N/kit
kind=${1:?measure or gate}; sha=${2:?commit}; tag=${3:?new tag}; loads=${4:?loads}; control=${5:-}
awk -F '\t' '$1=="three" && $3=="GATE2" && $4==100 && ($5=="G" || $5=="H") {ok=1} END{exit !ok}' "$N/queue/line.tsv" || { echo 'No dedicated GATE2 block'; exit 2; }
cat "$N/inbox/GATE2.md" "$N/inbox/ONEROW.md" > "$N/logs/GATE2-inbox-$tag.txt"
date; uptime
rc=0
case "$kind" in
  measure) [ -z "$control" ] || exit 2; bash "$K/KITF-go.sh" measure three "$sha" "$tag" "$loads" || rc=$?;;
  gate) bash "$K/GATE2-go-KB.sh" gate three "$sha" "$tag" "$loads" "$control" || rc=$?;;
  *) exit 2;;
esac
while [ "$rc" = 9 ]; do rc=0; bash "$K/wait.sh" "$tag" 45 || rc=$?; done
[ "$rc" = 0 ] || exit "$rc"
[ -f "$N/runs/$tag/LOADS-DONE" ] || exit 3
date; uptime
# The dedicated block continues. The caller calls queue.sh done after its last cell.
