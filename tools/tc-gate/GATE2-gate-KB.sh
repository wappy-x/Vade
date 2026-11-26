#!/bin/bash
# TCGATE COPY of lane-measure.sh (seat TCGATE, 2026-11-15, tensor-core wave). The claim, claim race, release, stop,
# bail, signal and clean-up lines are lane-measure.sh's, unchanged. Changed: the build (TCGATE-build.sh also builds the
# check tool), the staged files, the server start (TCGATE-serve.sh), the parts below, and the summary.
#
# THE GATE: after a tensor-core rebase "the same token IDs as main" is false by design. This script collects what
# TCGATE-verdict.py needs to prove the six exactness rules on the NEW token IDs, on the served model, on a pair.
#
# Usage: /bin/bash TCGATE-gate.sh PAIR COMMIT TAG "PARTS" [CONTROL]
#   PAIR     one | two | three | four
#   COMMIT   full 40-character commit
#   TAG      new short name, lower case letters, digits, "-"; results go to night-limit/runs/TAG/
#   PARTS    a quoted list of:
#     one chats read draft   the four loads of lane-measure.sh, the same requests. "chats" also runs each of the 8
#                            chats ALONE, the copy chat alone, and other groups of 4 and 2 (rule 2 and rule 5).
#     deep     server with --mtp 7: plain, strict depth 1, 3 and 7 for structured, code, prose (rule 4)
#     turns    server at context 32768, --mtp 3 (the k32 form): TCGATE-turns.py (rule 3: a kept second turn against
#              the same turn cold; a continuation read cold against the decode steps that made it)
#     tool     the check tool's one-box row sweeps on rank 0: dense-tc-rows, nv-tc-groups, prompt-rows (rule 2, rule 6).
#              TCGATE_TOOL_MODES names other modes (a format seat's own sweep), each run as `MODE OUT`.
#     quality  same-history top-1 of the served build (TCGATE-quality.py) on the histories of quality-tc/histories.json:
#              main's answers to the four loads, and the four gate cells the maker's engine scored on 2026-11-10.
#              QUALITY_SECONDS (default 1500) bounds it. Read it with TCGATE-quality-report.py.
#   CONTROL  optional: a word for the wrong-on-purpose control commit (exported to its kernel build as VADE_GATE_CONTROL).
# Result: runs/TAG/SUMMARY.txt; the verdict of one run or of two runs: python3 TCGATE-verdict.py runs/TAG [runs/TAG2].
# It refuses to start when a box of the pair is claimed, runs a GPU program, or runs a build.
set -u
N=$HOME/Desktop/Vade/benchmarks/2026-11-14/night-limit; K=$N/kit
PAIR=${1:?pair}; SHA=${2:?full commit}; TAG=${3:?tag}; LOADS=${4:?parts}; CONTROL=${5:-}; TRACE=""
case "$TAG" in *[!a-z0-9-]*|"") echo "TAG: lower case letters, digits and - only"; exit 2;; esac
case "$SHA" in ????????????????????????????????????????) ;; *) echo "COMMIT must be the full 40-character commit"; exit 2;; esac
for l in $LOADS; do case "$l" in one|chats|read|draft|deep|turns|tool|quality|golden) ;; *) echo "PARTS: one chats read draft deep turns tool quality"; exit 2;; esac; done
case "$CONTROL" in *[!a-z0-9-]*) echo "CONTROL: lower case letters, digits and - only"; exit 2;; esac
R=$N/runs/$TAG; [ -e "$R" ] && { echo "runs/$TAG exists: pick a new TAG"; exit 2; }; mkdir -p "$R"
FOLDER=night-$TAG; MODE=u; LOADS_C=${LOADS// /,}
S=$HOME/Desktop/Vade/system/scripts/spark-ssh.sh
REPO=$HOME/Desktop/Vade/code/vade
# The names of the lab boxes are not kept in this repository. The lab keeps them in a file of its own:
# one line a pair, "PAIR RANK0 RANK1" (default: pair-hosts.tsv in the lab kit; GATE2_PAIR_HOSTS names another file).
case "$PAIR" in one|two|three|four) ;; *) echo "PAIR: one two three four"; exit 2;; esac
HOSTS=${GATE2_PAIR_HOSTS:-$K/pair-hosts.tsv}
read -r R0 R1 <<< "$(awk -v p="$PAIR" '$1==p{print $2, $3; exit}' "$HOSTS" 2>/dev/null)"
[ -n "${R0:-}" ] && [ -n "${R1:-}" ] || { echo "no box names for pair $PAIR in $HOSTS"; exit 2; }
ID=night-$TAG-$(date +%H%M%S); D="~/$FOLDER"
NSFLAGS='--sample=none --cpuctxsw=process-tree'
on(){ local h=$1; shift; "$S" "$h" "$*"; }
mask(){ sed -E 's/([0-9]{1,3}\.){2,3}[0-9]{1,3}/<addr>/g'; }
log(){ echo "$(date +%H:%M:%S) $*" | mask | tee -a $R/drive.log; }
for h in $R0 $R1; do
  n=$(on $h 'echo $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) $(pgrep -x flash-serve | wc -l) $(pgrep -x cargo | wc -l)'); [ "$n" = "0 0 0" ] || { log "ABORT: $h is not idle ($n)"; touch $R/DRIVE-END; exit 3; }
  open=$(on $h "awk '/SPARKS CLAIMED/{o[\$4]=\$0} /SPARKS RELEASED/{delete o[\$4]} END{for(k in o) print o[k]}' ~/gpu.lock | cut -c1-120"); [ -z "$open" ] || { log "ABORT: open claim on $h: $open"; touch $R/DRIVE-END; exit 4; }
done
for h in $R0 $R1; do on $h "printf 'SPARKS CLAIMED %s $ID $R0,$R1 NIGHT-LIMIT measure $TAG commit=${SHA:0:10} loads=$LOADS_C limit=1800s\n' \"\$(date -Is)\" >> ~/gpu.lock; mkdir -p $D/kit && chmod 700 $D $D/kit"; done
log "claimed pair $PAIR: $R0 (rank 0), $R1 (rank 1) ($ID)"
# Added 02:20 by the lead: two runs that start within the same seconds could both pass the checks above (seen on pair two at 02:11:58).
# Wait, read the open claims again: the smaller claim id keeps the pair, the other releases and exits 4.
sleep 6
rival=$(on $R0 "awk '/SPARKS CLAIMED/{o[\$4]=1} /SPARKS RELEASED/{delete o[\$4]} END{for(k in o) print k}' ~/gpu.lock" | grep -v -x -- "$ID" | sort | head -1)
if [ -n "$rival" ] && [[ "$rival" < "$ID" ]]; then
  for h in $R0 $R1; do on $h "printf 'SPARKS RELEASED %s $ID $R0,$R1 exit=4 lost a claim race to $rival\n' \"\$(date -Is)\" >> ~/gpu.lock"; done
  log "ABORT: claim race lost to $rival; claim released"; touch $R/DRIVE-END 2>/dev/null; exit 4
fi
release(){ local h; for h in $R0 $R1; do on $h "a=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); printf 'SPARKS RELEASED %s $ID $R0,$R1 exit=$1 compute_apps=%s\n' \"\$(date -Is)\" \"\$a\" >> ~/gpu.lock"; done; log "released pair $PAIR ($ID) exit=$1"; }
# Stop only what this run started: every process whose working directory is the box folder or below it. Never by name.
stopall(){ local a b i h; for h in $R0 $R1; do on $h "nsys shutdown --session=ptctrace --kill none >/dev/null 2>&1; bash $D/stop-mine.sh $FOLDER" >/dev/null 2>&1; done
  for i in $(seq 1 40); do a=$(on $R0 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'); b=$(on $R1 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'); [ "$a $b" = "0 0" ] && { sleep 4; log "gpus clear"; return 0; }; sleep 2; done; log "GPU NOT CLEAR $R0=$a $R1=$b"; return 1; }
bail(){ log "BAIL: $*"; stopall; release 1; for h in $R0 $R1; do on $h "rm -rf $D"; done; touch $R/DRIVE-END; exit 4; }
trap 'log "driver signal"; stopall; release 6; for h in $R0 $R1; do on $h "rm -rf $D"; done; touch $R/DRIVE-END; exit 6' TERM INT HUP
# ---- A: stage and build on rank 0 (TCGATE-build.sh: the server and the check tool) ----
for h in $R0 $R1; do on $h "cat > $D/stop-mine.sh" < $K/stop-mine.sh; done
if [ -z "$CONTROL" ]; then
  on $R0 "cat > $D/build.sh" < "$K/GATE2-build-cache.sh"
else
  on $R0 "cat > $D/build.sh" < "$K/TCGATE-build.sh"
fi
( git -C $REPO archive --prefix=src-T/ $SHA | on $R0 "cat > $D/src-T.tar" ) || bail "archive"
on $R0 "cd $D && rm -f build-T.out && setsid -f bash build.sh T $SHA $FOLDER $CONTROL > build-T.out 2>&1 < /dev/null"; log "A: build of $SHA started on $R0 (control: ${CONTROL:-none})"
for h in $R0 $R1; do
  for f in TCGATE-serve.sh TCGATE-turns.py TCGATE-quality.py GATE2-goldens.tsv TCGATE-many.rs dr11f.py prompts.json gate.py g8192.tsv g8192d.tsv many.rs json.rs; do on $h "cat > $D/kit/$f" < $K/$f; done
  on $h "chmod +x $D/kit/TCGATE-serve.sh"
done
for i in $(seq 1 260); do grep -q "BUILD_COMPLETE T\|_FAIL\|MISMATCH" <<< "$(on $R0 "cat $D/build-T.out 2>/dev/null")" && break; sleep 3; done
on $R0 "cat $D/build-T.out" > $R/build-T.out
grep -q "BUILD_COMPLETE T" $R/build-T.out || { on $R0 "cat $D/build-T-kernels.log" > $R/build-T-kernels.log 2>/dev/null; on $R0 "cat $D/build-T-bins.log" > $R/build-T-bins.log 2>/dev/null; bail "build failed (the build denies warnings; full logs in runs/$TAG/build-T-kernels.log and build-T-bins.log): $(tail -3 $R/build-T.out | tr '\n' ' ' | cut -c1-300)"; }
log "A: kit built: serve=$(grep flash-serve $R/build-T.out | cut -c1-12) ptx=$(grep flash.ptx $R/build-T.out | cut -c1-12) $(grep -o 'secs=[0-9]*' $R/build-T.out) $(grep '^CONTROL=' $R/build-T.out)"
on $R0 "cp -p $D/kit-T/flash-serve $D/kit-T/flash.ptx $D/kit-T/resident-decode-check $D/kit/" || bail "kit copy"
on $R0 "tar cf - -C $D/kit-T flash-serve flash.ptx" | on $R1 "tar xf - -C $D/kit" || bail "kit copy to rank 1"
on $R0 "cd $D/kit && ~/.cargo/bin/rustc +1.99.0 --edition=2024 -O many.rs -o many 2>&1 | tail -5; ~/.cargo/bin/rustc +1.99.0 --edition=2024 -O TCGATE-many.rs -o manyx 2>&1 | tail -5"
on $R0 "test -x $D/kit/many && test -x $D/kit/manyx" || bail "client compile"
on $R0 "cat > $D/kit/histories.json" < ${TCGATE_HISTORIES:-$N/quality-tc/histories.json} || bail "histories"
SECRET=$(openssl rand -hex 32)
# ---- load helpers (lane-measure.sh's; the server line is TCGATE-serve.sh with the context in KEEP_CTX) ----
CTX=16384
start(){ local L=$1 mtp=$2 pids=() h p i t
  for h in "$R0:0" "$R1:1"; do
    printf '%s\n' "$SECRET" | on "${h%:*}" "IFS= read -r SEQPAR_CONTROL_SECRET; export SEQPAR_CONTROL_SECRET; cd $D/kit && rm -f rank${h#*:}-$L.log && KEEP_CTX=$CTX setsid -f bash ./TCGATE-serve.sh ${h#*:} $mtp $MODE > rank${h#*:}-$L.log 2>&1 < /dev/null" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  for i in $(seq 1 150); do
    t=$(on "$R0" "tail -n 60 $D/kit/rank0-$L.log")
    case "$t" in *'ready bind='*) return 0;; *'FAIL '*|*'timed out'*|*panic*|*BAD_PORT*|*'invalid socket'*) echo "$t" | tail -15 | mask >> "$R/drive.log"; return 2;; esac
    sleep 2
  done; return 2; }
run(){ local tl=$1; shift; on "$R0" "cd $D/kit && timeout $tl $*" 2>&1 | mask | cut -c1-260 | tee -a "$R/drive.log"; return ${PIPESTATUS[0]}; }
pull(){ local L=$1; mkdir -p $R/$L; on $R0 "cd $D/kit && tar cf - rank0-$L.log out-$L* 2>/dev/null" | tar xf - -C $R/$L; on $R1 "cat $D/kit/rank1-$L.log" | mask > $R/$L/rank1-$L.log
  find "$R/$L" -type f \( -name '*.log' -o -name '*.json' -o -name '*.jsonl' \) -exec sed -i '' -E 's/([0-9]{1,3}\.){2,3}[0-9]{1,3}/<addr>/g' {} + ; }
CUR=none; T0=0; WINDOWS=""
begin(){ CUR=$1; local mtp=$2; T0=$(date +%s); log "$CUR start (mtp $mtp)"
  start $CUR $mtp || { log "$CUR first start failed; one retry"; stopall; start $CUR $mtp || { pull $CUR; bail "$CUR did not start"; }; }
  log "$CUR ready after $(( $(date +%s) - T0 )) s"; }
endload(){ pull $CUR; log "$CUR DONE $(( $(date +%s) - T0 )) s"; stopall || bail "gpu not clear after $CUR"; }
one(){ run 120 python3 dr11f.py out-$CUR prompts.json rank0-$CUR.log $CUR-$1 $2 $3 $4 $5; }
chats(){ run 240 ./many out-$CUR prompts.json $CUR-$1 $2 $3 different 0; }
alone(){ run 240 ./manyx out-$CUR prompts.json $CUR-$1 $2 128 $3 0 $4; }
# ---- the parts ----
has(){ case " $LOADS " in *" $1 "*) return 0;; esac; return 1; }
if has one || has chats || has read; then
  CTX=16384; begin PL 0; one warm prose 0 P 64
  if has one; then
    run 120 python3 dr11f.py out-PL prompts.json rank0-PL.log PL-one prose 1 P 256
    one one code 1 P 256; one one structured 1 P 256
  fi
  if has chats; then
    chats warm 8 48
    run 240 ./many out-PL prompts.json PL-c8 8 128 different 0
    run 240 ./many out-PL prompts.json PL-c4 4 128 different 0
    run 240 ./many out-PL prompts.json PL-c2 2 128 different 0
    run 240 ./many out-PL prompts.json PL-s8 8 128 copies 0
    run 240 ./many out-PL prompts.json PL-s2 2 128 copies 0
    # TCGATE: each of the 8 chats alone, the copy chat alone, chats 4..7 as a group of 4, and three more pairs.
    for i in 0 1 2 3 4 5 6 7; do alone a$i 1 different $i; done
    alone s1 1 copies 0
    alone c4b 4 different 4
    alone c2b 2 different 2; alone c2c 2 different 4; alone c2d 2 different 6
  fi
  if has read; then run 400 python3 gate.py out-PL-read g8192d.tsv PL-read; fi
  endload
fi
if has draft; then
  CTX=16384; begin DR 3; one warm structured 0 P 64; one warmS structured 10 S3 64
  for wl in structured code prose; do one d $wl 1 P 256; done
  run 120 python3 dr11f.py out-DR prompts.json rank0-DR.log DR-d structured 3 S3 256
  run 120 python3 dr11f.py out-DR prompts.json rank0-DR.log DR-d prose 3 S3 256
  one d code 3 S3 256
  for wl in structured code prose; do one d $wl 2 S2 256; done
  endload
fi
# deep: plain, then strict drafting at depth 1, 3 and 7, each for structured, code and prose. The answers must equal plain.
if has deep; then
  CTX=16384; begin DP 7; one warm structured 0 P 64; one warmS structured 10 S7 64
  for wl in structured code prose; do one d $wl 0 P 256; done
  for dp in 1 3 7; do for wl in structured code prose; do one d $wl $dp S$dp 256; done; done
  endload
fi
# turns: the server of the k32 load (context 32768, --mtp 3, prompt keep as the build has it).
if has turns; then
  CTX=32768; begin K32 3
  run 900 python3 TCGATE-turns.py out-K32 rank0-K32.log K32
  endload
fi
# The original four-cell golden fixture, on a fresh default-context server.
if has golden; then
  CTX=32768; begin GD 0
  run 400 python3 gate.py out-GD-goldens GATE2-goldens.tsv GD
  endload
fi
# quality: same-history top-1 at every answer position, on a server with the prompt cache as the build has it.
if has quality; then
  CTX=16384; begin QL 0
  run $(( ${QUALITY_SECONDS:-1500} + 600 )) python3 TCGATE-quality.py out-QL histories.json '$HOME/models/nvidia/GLM-5.3-Flash-FP4-Local' QL ${QUALITY_SECONDS:-1500}
  endload
fi
# tool: the check tool's one-box row sweeps, on rank 0, from the box folder (stop-mine.sh stops them with the rest).
if has tool; then
  CUR=TOOL; T0=$(date +%s); log "TOOL start"
  for m in ${TCGATE_TOOL_MODES:-prompt-rows nv-tc-groups dense-tc-rows}; do
    on $R0 "cd $D/kit && rm -rf out-tool-$m && CUDA_VISIBLE_DEVICES=0 timeout -k 10 600 ./resident-decode-check $m $D/kit/out-tool-$m > tool-$m.log 2>&1; echo \"TOOL $m exit=\$?\"" 2>&1 | mask | cut -c1-260 | tee -a "$R/drive.log"
  done
  mkdir -p $R/TOOL; on $R0 "cd $D/kit && tar cf - tool-*.log 2>/dev/null" | tar xf - -C $R/TOOL
  find "$R/TOOL" -type f -name '*.log' -exec sed -i '' -E 's/([0-9]{1,3}\.){2,3}[0-9]{1,3}/<addr>/g' {} +
  log "TOOL DONE $(( $(date +%s) - T0 )) s"; stopall || bail "gpu not clear after TOOL"
fi
touch $R/LOADS-DONE
stopall; release 0
for h in $R0 $R1; do on $h "rm -rf $D"; done; log "box folders removed"
# ---- E: the summary: speeds, server step times, the answers against a baseline (information), and this run's verdict ----
BASELINE=${TCGATE_BASELINE:-$N/baseline/ids.json}
{ echo "RUN $TAG  pair $PAIR  commit $SHA  parts: $LOADS  control: ${CONTROL:-none}"
  echo; echo "SPEEDS (client side)"; grep -E "^MEASURED|^(PL|DR|DP|K32)-| TURNS | CHECK |^TOOL | QUALITY " $R/drive.log | grep -v -- "-warm" | cut -c1-230
  echo; echo "SERVER STEP TIMES"; for f in $R/*/rank0-*.log; do [ -f "$f" ] && python3 $K/ct.py walls "$f" | grep -v "GROUP   prefill  rows=1[01][0-9] \|GROUP   prefill  rows=9[0-9] \|GROUP   prefill  rows=8[0-9] "; done
  echo; echo "ANSWERS AGAINST THE BASELINE $BASELINE (information only: after a rebase a difference from main is expected)"; python3 $K/ids.py check $BASELINE $R
  echo; echo "GATE VERDICT OF THIS RUN ALONE (rule 1 needs a second run: TCGATE-verdict.py RUN_A RUN_B)"; python3 $K/TCGATE-verdict.py $R; } > $R/SUMMARY.txt 2>&1
cat $R/SUMMARY.txt
log "NIGHT-MEASURE-END"; touch $R/DRIVE-END
