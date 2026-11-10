#!/bin/sh
# Internal curl guard. Rust passes its locked file as standard input.
# Keep exclusion until curl exits. Stop curl if the Rust owner exits.
set -u
PARENT=$1; shift
OWNER=$$; CHILD=; WATCHER=
cleanup() {
  trap '' HUP INT TERM
  if [ -n "$WATCHER" ]; then kill "$WATCHER" 2>/dev/null || :; wait "$WATCHER" 2>/dev/null || :; fi
  if [ -n "$CHILD" ]; then kill "$CHILD" 2>/dev/null || :; wait "$CHILD" 2>/dev/null || :; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM
curl "$@" <&0 &
CHILD=$!
(
  TICK=
  stop() { if [ -n "$TICK" ]; then kill "$TICK" 2>/dev/null || :; wait "$TICK" 2>/dev/null || :; fi; }
  trap 'stop; exit 0' HUP INT TERM
  while kill -0 "$PARENT" 2>/dev/null; do sleep 1 & TICK=$!; wait "$TICK"; TICK=; done
  kill -TERM "$OWNER" 2>/dev/null || :
) </dev/null &
WATCHER=$!
status=0; wait "$CHILD" || status=$?
CHILD=
exit "$status"
