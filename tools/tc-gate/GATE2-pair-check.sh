#!/bin/bash
# D2 copy of lane-measure.sh (seat D2, 2026-11-15). One command: claim a Spark pair, build ONE commit on rank 0
# (the server and the device check tool `resident-decode-check`, with TCGATE-build.sh), copy the kit to rank 1, run the
# check tool on BOTH ranks over the real wire, pull the result files, release the pair, remove the box folders.
# The claim, race, release, stop and clean-up parts are the lines of lane-measure.sh, unchanged.
# Cost: one pair for about 4 to 8 minutes (build about 110 s). Gain: the exchange census of a decode step
# (the tool's EXCHANGES file), and a two-rank device check, run through the kit.
#
# Usage: /bin/bash GATE2-pair-check.sh PAIR COMMIT TAG RUN [RUN...]
#   RUN   one quoted word list a two-rank run, in order: "decode REPEATS FLAG..." or "chain ARM EXCHANGES REPEATS"
#         (the words of GATE2-pair-run.sh after RANK and PORT). A word of the form max=N in a decode run sets the
#         request's max_tokens (default 64).
# Result: runs/TAG/<n>-<kind>/rank0, rank1 (logs and the tool's saved files), runs/TAG/SUMMARY.txt, runs/TAG/drive.log.
set -u
N=$HOME/Desktop/Vade/benchmarks/2026-11-14/night-limit; K=$N/kit
PAIR=${1:?pair}; SHA=${2:?full commit}; TAG=${3:?tag}; shift 3
[ $# -ge 1 ] || { echo "give one run or more"; exit 2; }
case "$TAG" in *[!a-z0-9-]*|"") echo "TAG: lower case letters, digits and - only"; exit 2;; esac
case "$SHA" in ????????????????????????????????????????) ;; *) echo "COMMIT must be the full 40-character commit"; exit 2;; esac
for r in "$@"; do case "${r%% *}" in decode|chain) ;; *) echo "RUN starts with decode or chain"; exit 2;; esac; done
R=$N/runs/$TAG; [ -e "$R" ] && { echo "runs/$TAG exists: pick a new TAG"; exit 2; }; mkdir -p "$R"
FOLDER=night-$TAG
S=$HOME/Desktop/Vade/system/scripts/spark-ssh.sh
REPO=$HOME/Desktop/Vade/code/vade
# The names of the lab boxes are not kept in this repository. The lab keeps them in a file of its own:
# one line a pair, "PAIR RANK0 RANK1" (default: pair-hosts.tsv in the lab kit; GATE2_PAIR_HOSTS names another file).
case "$PAIR" in one|two|three|four) ;; *) echo "PAIR: one two three four"; exit 2;; esac
HOSTS=${GATE2_PAIR_HOSTS:-$K/pair-hosts.tsv}
read -r R0 R1 <<< "$(awk -v p="$PAIR" '$1==p{print $2, $3; exit}' "$HOSTS" 2>/dev/null)"
[ -n "${R0:-}" ] && [ -n "${R1:-}" ] || { echo "no box names for pair $PAIR in $HOSTS"; exit 2; }
ID=night-$TAG-$(date +%H%M%S); D="~/$FOLDER"
on(){ local h=$1; shift; "$S" "$h" "$*"; }
mask(){ sed -E 's/([0-9]{1,3}\.){2,3}[0-9]{1,3}/<addr>/g'; }
log(){ echo "$(date +%H:%M:%S) $*" | mask | tee -a $R/drive.log; }
for h in $R0 $R1; do
  n=$(on $h 'echo $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) $(pgrep -x flash-serve | wc -l) $(pgrep -x cargo | wc -l)'); [ "$n" = "0 0 0" ] || { log "ABORT: $h is not idle ($n)"; touch $R/DRIVE-END; exit 3; }
  open=$(on $h "awk '/SPARKS CLAIMED/{o[\$4]=\$0} /SPARKS RELEASED/{delete o[\$4]} END{for(k in o) print o[k]}' ~/gpu.lock | cut -c1-120"); [ -z "$open" ] || { log "ABORT: open claim on $h: $open"; touch $R/DRIVE-END; exit 4; }
done
for h in $R0 $R1; do on $h "printf 'SPARKS CLAIMED %s $ID $R0,$R1 NIGHT-LIMIT measure $TAG commit=${SHA:0:10} loads=d2-pair-check limit=1800s\n' \"\$(date -Is)\" >> ~/gpu.lock; mkdir -p $D/kit && chmod 700 $D $D/kit"; done
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
stopall(){ local a b i h; for h in $R0 $R1; do on $h "bash $D/stop-mine.sh $FOLDER" >/dev/null 2>&1; done
  for i in $(seq 1 40); do a=$(on $R0 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'); b=$(on $R1 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l'); [ "$a $b" = "0 0" ] && { sleep 4; log "gpus clear"; return 0; }; sleep 2; done; log "GPU NOT CLEAR $R0=$a $R1=$b"; return 1; }
bail(){ log "BAIL: $*"; stopall; release 1; for h in $R0 $R1; do on $h "rm -rf $D"; done; touch $R/DRIVE-END; exit 4; }
trap 'log "driver signal"; stopall; release 6; for h in $R0 $R1; do on $h "rm -rf $D"; done; touch $R/DRIVE-END; exit 6' TERM INT HUP
# ---- A: stage and build on rank 0 ----
for h in $R0 $R1; do on $h "cat > $D/stop-mine.sh" < $K/stop-mine.sh; done
on $R0 "cat > $D/build.sh" < $K/TCGATE-build.sh
( git -C $REPO archive --prefix=src-T/ $SHA | on $R0 "cat > $D/src-T.tar" ) || bail "archive"
on $R0 "cd $D && rm -f build-T.out && setsid -f bash build.sh T $SHA $FOLDER > build-T.out 2>&1 < /dev/null"; log "A: build of $SHA started on $R0"
for h in $R0 $R1; do on $h "cat > $D/kit/GATE2-pair-run.sh" < $K/GATE2-pair-run.sh; on $h "chmod +x $D/kit/GATE2-pair-run.sh"; done
for i in $(seq 1 260); do grep -q "BUILD_COMPLETE T\|_FAIL\|MISMATCH" <<< "$(on $R0 "cat $D/build-T.out 2>/dev/null")" && break; sleep 3; done
on $R0 "cat $D/build-T.out" > $R/build-T.out
grep -q "BUILD_COMPLETE T" $R/build-T.out || { on $R0 "cat $D/build-T-kernels.log" > $R/build-T-kernels.log 2>/dev/null; on $R0 "cat $D/build-T-bins.log" > $R/build-T-bins.log 2>/dev/null; bail "build failed (the build denies warnings; full logs in runs/$TAG/build-T-kernels.log and build-T-bins.log): $(tail -3 $R/build-T.out | tr '\n' ' ' | cut -c1-300)"; }
log "A: kit built: check=$(grep resident-decode-check $R/build-T.out | cut -c1-12) ptx=$(grep flash.ptx $R/build-T.out | cut -c1-12) $(grep -o 'secs=[0-9]*' $R/build-T.out)"
on $R0 "cp -p $D/kit-T/resident-decode-check $D/kit-T/flash.ptx $D/kit/" || bail "kit copy"
on $R0 "tar cf - -C $D/kit-T resident-decode-check flash.ptx" | on $R1 "tar xf - -C $D/kit" || bail "kit copy to rank 1"
SECRET=$(openssl rand -hex 32)
PORT=$((36000 + RANDOM % 900))
# ---- B: the two-rank runs ----
n=0; worst=0
for spec in "$@"; do
  n=$((n+1)); kind=${spec%% *}; words=""; max=64
  for w in ${spec#* }; do case "$w" in max=*) max=${w#max=};; *) words="$words $w";; esac; done
  port=$((PORT + 4*n)); L=$n-$kind
  req='{"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0,"frequency_penalty":0,"messages":[{"role":"user","content":"Say hello."}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":'$max',"cache":false,"ignore_eos":true,"draft":"plain","mtp_depth":0}'
  pids=()
  for h in "$R0:0" "$R1:1"; do
    printf '%s\n' "$req" | on "${h%:*}" "cat > $D/kit/request.json; rm -f $D/kit/pair${h#*:}.done $D/kit/rank${h#*:}.log"
    printf '%s\n' "$SECRET" | on "${h%:*}" "IFS= read -r SEQPAR_CONTROL_SECRET; export SEQPAR_CONTROL_SECRET; cd $D/kit && setsid -f bash ./GATE2-pair-run.sh ${h#*:} $port $kind $words > launch${h#*:}.log 2>&1 < /dev/null" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  log "B: run $L started on both ranks ($kind$words max_tokens=$max)"
  a=; b=
  for i in $(seq 1 330); do a=$(on $R0 "cat $D/kit/pair0.done 2>/dev/null"); b=$(on $R1 "cat $D/kit/pair1.done 2>/dev/null"); [ -n "$a" ] && [ -n "$b" ] && break; sleep 3; done
  log "B: run $L exit rank0=${a:-none} rank1=${b:-none}"
  [ "${a:-9}" = 0 ] && [ "${b:-9}" = 0 ] || worst=1
  i=0; for h in $R0 $R1; do mkdir -p $R/$L/rank$i
    on $h "cd $D/kit && tar cf - --exclude='*.f32' rank$i.log launch$i.log prepare$i.log out-decode out-chain out-batch-* 2>/dev/null" | tar xf - -C $R/$L/rank$i 2>/dev/null
    find "$R/$L/rank$i" -type f -exec sed -i '' -E 's/([0-9]{1,3}\.){2,3}[0-9]{1,3}/<addr>/g' {} + 2>/dev/null
    on $h "cd $D/kit && rm -rf out-decode out-chain out-batch-*"; i=$((i+1)); done
  stopall || bail "gpu not clear after $L"
done
release $worst
for h in $R0 $R1; do on $h "rm -rf $D"; done; log "box folders removed"
# ---- C: the summary ----
{ echo "RUN $TAG  pair $PAIR  commit $SHA  runs: $*"
  echo "IMAGE $(grep -E 'flash.ptx|resident-decode-check' $R/build-T.out | cut -c1-90 | tr '\n' ' ')"
  for d in $R/*-decode $R/*-chain; do [ -d "$d" ] || continue; echo; echo "== $(basename $d)"
    for f in $d/rank*/out-decode/run*/EXCHANGES $d/rank*/out-decode/run*/TIMING $d/rank*/out-decode/run*/EXCHANGE_STAGE; do [ -f "$f" ] && echo "${f#$d/}: $(cat $f)"; done
    for f in $d/rank*/out-decode/run*/OUTPUT_IDS; do [ -f "$f" ] && echo "${f#$d/}: sha256=$(shasum -a 256 $f | cut -c1-16) tokens=$(wc -l < $f | tr -d ' ')"; done
    grep -h -a "CHAIN_RESULT\|RESIDENT JOB PASS\|RESIDENT DECODE PASS" $d/rank*/rank*.log 2>/dev/null | cut -c1-200
    grep -h -a -i "error\|panic\|FAIL" $d/rank*/rank*.log $d/rank*/launch*.log 2>/dev/null | cut -c1-300 | head -8
  done; } > $R/SUMMARY.txt 2>&1
cat $R/SUMMARY.txt
log "NIGHT-MEASURE-END"; touch $R/DRIVE-END
exit $worst
