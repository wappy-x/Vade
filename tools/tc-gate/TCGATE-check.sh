#!/bin/bash
# Usage: TCGATE_SEAT=TCBF bash TCGATE-check.sh FULL_COMMIT two|four NEW_TAG
# Runs two fresh copies of every served gate request, row tools and quality, then prints the table.
# Each run uses the waiting line; allow about an hour, plus queue time (ESTIMATE).
# Format-specific device/twin and subtle sum-order controls remain mandatory separate receipts.
set -u
N=${VADE_NIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; K=$N/kit
sha=${1:?full commit}; pair=${2:?two or four}; tag=${3:?new tag}
for round in a b; do
  for part in 1 2 3; do
    case "$part" in 1) loads='one chats read';; 2) loads='draft deep';; 3) loads='turns tool';; esac
    /bin/bash "$K/TCGATE-run.sh" "$sha" "$pair" "$tag-$round$part" "$loads" || exit $?
  done
done
QUALITY_SECONDS=420 /bin/bash "$K/TCGATE-run.sh" "$sha" "$pair" "$tag-quality" quality || exit $?
set -- "$N/runs/$tag-a1+$N/runs/$tag-a2+$N/runs/$tag-a3" "$N/runs/$tag-b1+$N/runs/$tag-b2+$N/runs/$tag-b3" --quality "$N/runs/$tag-quality"
[ -z "${TCGATE_TILE_CONTROL:-}" ] || set -- "$@" --tile "$TCGATE_TILE_CONTROL"
[ -z "${TCGATE_SMALL_CONTROL:-}" ] || set -- "$@" --small "$TCGATE_SMALL_CONTROL"
python3 "$K/TCGATE-table.py" "$@" | tee "$N/runs/$tag-table.md"
exit "${PIPESTATUS[0]}"
