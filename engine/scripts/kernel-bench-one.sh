#!/bin/bash
set -euo pipefail
if [ "$#" -lt 4 ]; then
  echo 'usage: kernel-bench-one.sh RUNNER CHECKPOINT NEW_OUT COMMAND_FILE [bench options...]' >&2
  exit 2
fi
runner=$1
checkpoint=$2
out=$3
commands=$4
shift 4
test -x "$runner"
test -f "$commands"
test ! -e "$out"
owner="kernel-bench-$$"
exec 9>> "$HOME/gpu.lock"
flock -n 9
tail -1 "$HOME/gpu.lock" | grep -q 'SPARKS RELEASED'
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
printf 'SPARKS CLAIMED %s %s single-device kernel bench\n' "$(date -Iseconds)" "$owner" >&9
printf 'CLAIM %s owner=%s\n' "$(date -Iseconds)" "$owner"
release() {
  result=$?
  if test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && tail -1 "$HOME/gpu.lock" | grep -q " $owner single-device kernel bench"; then
    printf 'SPARKS RELEASED %s %s exit=%s GPU-empty\n' "$(date -Iseconds)" "$owner" "$result" >&9
    printf 'RELEASE %s owner=%s exit=%s\n' "$(date -Iseconds)" "$owner" "$result"
  else
    printf 'RELEASE REFUSED owner=%s: GPU is occupied or ownership changed\n' "$owner" >&2
  fi
}
trap release EXIT
"$runner" serve "$checkpoint" "$out" "$@" < "$commands"
