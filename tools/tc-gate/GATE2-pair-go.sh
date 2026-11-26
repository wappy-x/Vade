#!/bin/bash
# D2 copy of go.sh: adds the mode `pair` (GATE2-pair-check.sh PAIR COMMIT TAG RUN...).
# NIGHT-LIMIT starter (2026-11-14 night). Starts one run of lane-measure.sh or lane-bench.sh fully detached (its own
# session, so it ends by itself and releases the pair even if the caller is stopped), then waits for it in the foreground.
# Cost: one small script. Gain: a writer cannot leave a pair claimed by ending a turn, and cannot start a run wrong.
#
#   go.sh measure PAIR COMMIT TAG "LOADS" [trace]      -> lane-measure.sh with the same words
#   go.sh bench   PAIR BASE COMMIT TAG ARG...          -> lane-bench.sh with the same words
# The run's console output goes to night-limit/runs-TAG.out; its last line is `driver exit=<code>`.
# go.sh waits up to 540 seconds and exits with the run's own code (0 done, 3 a box is busy, 4 the pair is claimed). If the
# run is not over by then it says so and exits 9: call `wait.sh TAG` (again and again) until it does not exit 9.
# Do not start the same work twice.
set -u
N=$HOME/Desktop/Vade/benchmarks/2026-11-14/night-limit; K=$N/kit
mode=${1:?measure or bench}; shift
case "$mode" in
  measure) script=$K/lane-measure.sh; tag=${3:?usage: go.sh measure PAIR COMMIT TAG \"LOADS\" [trace]};;
  bench)   script=$K/lane-bench.sh;   tag=${4:?usage: go.sh bench PAIR BASE COMMIT TAG ARG...};;
  pair)    script=$K/GATE2-pair-check.sh; tag=${3:?usage: GATE2-pair-go.sh pair PAIR COMMIT TAG RUN...};;
  *) echo "go.sh: first word is measure, bench or pair"; exit 2;;
esac
case "$tag" in *[!a-z0-9-]*|"") echo "go.sh: TAG is lower-case letters, digits and dash"; exit 2;; esac
out=$N/runs-$tag.out
[ -e "$out" ] || [ -e "$N/runs/$tag" ] && { echo "go.sh: the tag $tag was used already; pick a new one"; exit 2; }
python3 - "$out" "$script" "$@" <<'PY'
import os, sys
out, cmd = sys.argv[1], sys.argv[2:]
if os.fork(): sys.exit(0)
os.setsid()
if os.fork(): os._exit(0)
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2); os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execv('/bin/bash', ['/bin/bash', '-c', '/bin/bash "$@"; echo "driver exit=$?"', 'go'] + cmd)
PY
echo "go.sh: started $mode $tag at $(date +%H:%M:%S); console output in $out"
exec /bin/bash $K/wait.sh "$tag" 540
