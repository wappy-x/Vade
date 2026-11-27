#!/bin/bash
set -euo pipefail
driver="$(cd "$(dirname "$0")" && pwd)/flywheel4"   # built next to this script; see README.md
receipt=${1:?receipt folder required}
child=
finish() {
  status=$?
  trap - EXIT TERM INT HUP
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  "$driver" cleanup "$receipt" || status=1
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
"$driver" run "$receipt" &
child=$!
wait "$child"
child=
