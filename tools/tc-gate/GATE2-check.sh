#!/bin/bash
# GATE2-check.sh COMMIT NUMBERED_BASELINE TAG [PAIR]
# Two fresh plain starts create the immutable baseline first; the other gates
# follow. Existing evidence is read only and must name this exact commit.
set -eu
N=${VADE_NIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; K=$N/kit
sha=${1:?commit}; baseline=${2:?numbered baseline path}; tag=${3:?tag}; pair=${4:-any}
export TCGATE_SEAT=GATE2
case "$baseline" in "$N"/baseline-tc/[0-9][0-9]) ;; *) echo 'Expected numbered baseline folder'; exit 2;; esac
if [ -e "$baseline" ]; then
  [ "$(cat "$baseline/COMMIT")" = "$sha" ] || { echo 'Existing baseline belongs to another commit'; exit 3; }
fi
quality_gate(){
  main_sha=${GATE2_MAIN_COMMIT:-8f615ab806b56d1b9da92fec1e370e8b7edc9b20}
  quality_sha=${GATE2_QUALITY_COMMIT:-$sha}
  if [ -z "${GATE2_MAIN_QUALITY:-}" ]; then
    main_run="$N/runs/$tag-main-quality"
    if [ ! -e "$main_run" ]; then QUALITY_SECONDS=420 bash "$K/GATE2-run.sh" "$main_sha" "$pair" "$tag-main-quality" quality; fi
    [ -f "$main_run/LOADS-DONE" ] && grep -q "A: build of $main_sha started" "$main_run/drive.log" || { echo 'Invalid main quality evidence'; exit 3; }
    GATE2_MAIN_QUALITY="$main_run"
  fi
  "$K/GATE2-quality" "$N/quality-tc/histories.json" "$N/quality-tc/maker-reference.json" "$GATE2_MAIN_QUALITY" > "$N/runs/$tag-main-quality.txt"
  quality_run="$N/runs/$tag-quality"
  if [ ! -e "$quality_run" ]; then QUALITY_SECONDS=420 bash "$K/GATE2-run.sh" "$quality_sha" "$pair" "$tag-quality" quality; fi
  [ -f "$quality_run/LOADS-DONE" ] && grep -q "A: build of $quality_sha started" "$quality_run/drive.log" || { echo 'Invalid candidate quality evidence'; exit 3; }
  "$K/GATE2-quality" "$N/quality-tc/histories.json" "$N/quality-tc/maker-reference.json" "$GATE2_MAIN_QUALITY" "$quality_run" > "$N/runs/$tag-quality.txt"
}
for cell in a1 b1 a2 b2 a3 b3; do
  case "$cell" in
    a1|b1) loads='one chats read'; check=2;;
    a2|b2) loads='draft deep'; check=4;;
    a3|b3) loads='turns tool golden'; check=3;;
  esac
  r="$N/runs/$tag-$cell"
  if [ -e "$r" ]; then
    [ -f "$r/LOADS-DONE" ] && grep -q "A: build of $sha started" "$r/drive.log" || { echo "Refuse incomplete or wrong-commit evidence $r"; exit 3; }
  else
    bash "$K/GATE2-run.sh" "$sha" "$pair" "$tag-$cell" "$loads"
  fi
  rc=0; python3 "$K/GATE2-verdict.py" "$r" --only="$check" > "$r/CHECK.txt" || rc=$?
  # Exit3 means omitted parts; the requested part must explicitly pass.
  if { [ "$rc" != 0 ] && [ "$rc" != 3 ]; } || ! grep -q "^GATE PART $check .*: PASS$" "$r/CHECK.txt"; then cat "$r/CHECK.txt"; exit 1; fi
  case "$cell" in a2|b2) "$K/GATE2-paths" "$r" > "$r/PATHS.txt";;
    *) "$K/GATE2-paths" --plain "$r" > "$r/PATHS.txt";; esac
  if [ "$cell" = b1 ]; then
    for round in a b; do
      f="$N/runs/$tag-${round}1/PL/rank0-PL.log"
      # Print only when exactly one fresh server start exists.
      awk '/^TCGATE SERVER START rank=0 / {n++; s=""; for(i=1;i<=NF;i++)if($i~/^(epoch|pid|serve_sha|ptx_sha)=/)s=s $i " "} END{if(n==1)print s;else exit 1}' "$f" > "$N/runs/$tag-start-$round.txt"
      python3 "$K/ids.py" collect "$N/runs/$tag-${round}1" > "$N/runs/$tag-ids-$round.json"
    done
    cmp "$N/runs/$tag-ids-a.json" "$N/runs/$tag-ids-b.json"
    read -r ea pa sa ka < "$N/runs/$tag-start-a.txt"
    read -r eb pb sb kb < "$N/runs/$tag-start-b.txt"
    [ "$ea $pa" != "$eb $pb" ] && [ "$sa $ka" = "$sb $kb" ] || { echo 'Fresh start or image identity failed'; exit 1; }
    if [ -e "$baseline" ]; then cmp "$baseline/ids.json" "$N/runs/$tag-ids-a.json";
    else bash "$K/TCGATE-baseline.sh" "${baseline##*/}" "$sha" MXFP8 "$N/runs/$tag-a1" "$N/runs/$tag-b1"; fi
    echo 'Two fresh plain baselines match. Other exactness, quality and control gates remain pending.'
  fi
  [ "$cell" != a2 ] || quality_gate
done
a="$N/runs/$tag-a1+$N/runs/$tag-a2+$N/runs/$tag-a3"
b="$N/runs/$tag-b1+$N/runs/$tag-b2+$N/runs/$tag-b3"
python3 "$K/GATE2-verdict.py" "$a" "$b" > "$N/runs/$tag-exactness.txt"
"$K/GATE2-paths" "$N/runs/$tag-a1" "$N/runs/$tag-a2" "$N/runs/$tag-a3" > "$N/runs/$tag-paths.txt"
"$K/GATE2-paths" "$N/runs/$tag-b1" "$N/runs/$tag-b2" "$N/runs/$tag-b3" > "$N/runs/$tag-paths-repeat.txt"
bash "$K/GATE2-model.sh" "$sha" "$tag-model" "$pair"
# Controls are separate builds. They never enter the integration branch.
if [ -n "${GATE2_CONTROL_COMMIT:-}" ]; then
  bash "$K/GATE2-run.sh" "$GATE2_CONTROL_COMMIT" "$pair" "$tag-control-tile" 'one turns' tc-tile
  "$K/GATE2-paths" --plain "$N/runs/$tag-control-tile" > "$N/runs/$tag-control-tile/PATHS.txt"
  GATE2_TILE_CONTROL="$N/runs/$tag-control-tile"
  bash "$K/GATE2-run.sh" "$GATE2_CONTROL_COMMIT" "$pair" "$tag-control-rows" 'chats deep' tc-rows
  "$K/GATE2-paths" "$N/runs/$tag-control-rows" > "$N/runs/$tag-control-rows/PATHS.txt"
  GATE2_SMALL_CONTROL="$N/runs/$tag-control-rows"
  bash "$K/GATE2-run.sh" "$GATE2_CONTROL_COMMIT" "$pair" "$tag-control-one" 'one turns' tc-one
  "$K/GATE2-paths" --plain "$N/runs/$tag-control-one" > "$N/runs/$tag-control-one/PATHS.txt"
  python3 "$K/GATE2-verdict.py" "$a" "$N/runs/$tag-control-one" --expect-fail=1,3 --only=1,3 > "$N/runs/$tag-control-one.txt"
fi
: "${GATE2_TILE_CONTROL:?provide wrong-word tile served receipt}"
: "${GATE2_SMALL_CONTROL:?provide wrong-word small-row served receipt}"
python3 "$K/GATE2-verdict.py" "$a" "$GATE2_TILE_CONTROL" --expect-fail=1,3 --only=1,3 > "$N/runs/$tag-control-tile.txt"
python3 "$K/GATE2-verdict.py" "$GATE2_SMALL_CONTROL" --expect-fail=2,4 --only=2,4 > "$N/runs/$tag-control-small.txt"
echo 'Served evidence collected. READY still requires format twin/order receipts, speed bracket, source checks and new-golden handover.'
