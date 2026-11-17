#!/bin/bash
# D2 rank launcher (seat D2, 2026-11-15). RUNS ON A BOX, from the kit folder of a D2-pair-check.sh run.
# One rank of a two-box run of the device check tool `resident-decode-check`. The ring ends are read from the cable
# port, as serve.sh does. Usage: GATE2-pair-run.sh RANK PORT KIND ARG...
#   KIND decode: ARG... = REPEATS FLAG...  -> the tool's `pair` mode: the prompt of request.json, then its decode steps,
#                REPEATS times. The tool saves EXCHANGES (exchanges a decode step), TIMING, OUTPUT_IDS and counters
#                in out-decode/run<N>/.
#   KIND chain:  ARG... = ARM EXCHANGES REPEATS -> the tool's `pair-split-chain` mode (the sum exchange on the real wire).
# Output: rank<RANK>.log and pair<RANK>.done (the exit code) in the kit folder.
set -uo pipefail
rank=${1:?rank}; port=${2:?port}; kind=${3:?kind}; shift 3
cd "$(dirname "$0")"
export SEQPAR_CONTROL_SECRET
export TP_TOPOLOGY=path
own=$(ip -4 -o addr show dev enp1s0f1np1 | awk '{print $4}' | cut -d/ -f1 | head -n 1)
case ${own##*.} in 1) peer=${own%.*}.2;; 2) peer=${own%.*}.1;; *) echo BAD_PORT_ADDRESS; echo 1 > "pair$rank.done"; exit 1;; esac
M="$HOME/models/nvidia/GLM-5.3-Flash-FP4-Local"
rc=0
case "$kind" in
  decode)
    repeats=${1:?repeats}; shift
    rm -rf prompt.ids out-decode
    ./resident-decode-check prepare "$M" request.json prompt.ids > "prepare$rank.log" 2>&1 || { echo PREPARE_FAIL; cat "prepare$rank.log"; echo 1 > "pair$rank.done"; exit 1; }
    case " $* " in *" --batch-steps="*)
      : > jobs.txt
      for i in 1 2 3 4 5 6 7; do
        sed "s/Say hello\./Count to $i./" request.json > "request-$i.json"
        ./resident-decode-check prepare "$M" "request-$i.json" "prompt-$i.ids" >> "prepare$rank.log" 2>&1 || { echo 1 > "pair$rank.done"; exit 1; }
        printf 'prompt-%s.ids request-%s.json out-batch-%s 1\n' "$i" "$i" "$i" >> jobs.txt
      done
      set -- "$@" --jobs=jobs.txt
    ;; esac
    timeout --kill-after=15 900 ./resident-decode-check "$rank" 2 "$own:$port" "$peer:$port" pair "$M" prompt.ids request.json out-decode "$repeats" "$@" > "rank$rank.log" 2>&1; rc=$?;;
  chain)
    arm=${1:?arm}; ex=${2:?exchanges}; rep=${3:?repeats}
    rm -rf out-chain
    timeout --kill-after=15 300 ./resident-decode-check pair-split-chain "$rank" "$own:$port" "$peer:$port" out-chain "$arm" "$ex" "$rep" > "rank$rank.log" 2>&1; rc=$?;;
  *) echo "KIND: decode or chain"; rc=2;;
esac
unset SEQPAR_CONTROL_SECRET
sed -i "s/$own/CABLE_PORT/g; s/$peer/CABLE_PEER/g" ./*.log 2>/dev/null
echo "$rc" > "pair$rank.done"
exit "$rc"
