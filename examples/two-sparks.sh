#!/bin/sh
# Purpose: start one engine across two DGX Sparks from a controller, check it, and stop it.
# Run:     sh examples/two-sparks.sh          (dry run: prints every command; no SSH, no files, any Mac or Linux)
#          VADE_CONFIG=/your/two-sparks.conf sh examples/two-sparks.sh --live
# Needs:   vade on PATH (cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml).
#          --live also needs the INSTALL.md controller setup: passwordless SSH to both Sparks,
#          the verified checkpoint on each, and a cable between them (RDMA, or SEQPAR_RDMA=0 for TCP).
# Expected output: see "Two Sparks" in examples/README.md.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

if [ "${1:-}" != --live ]; then
  # Fill this checkout's folders into a private copy of the example config.
  DIR=$(mktemp -d)
  DIR=$(CDPATH= cd -- "$DIR" && pwd -P)
  # awk copies both folders as literal text (ENVIRON); sed would read & | and \ in them as syntax.
  ROOT=$ROOT DIR=$DIR awk '
    { i = index($0, "/path/to/vade"); if (i) $0 = substr($0, 1, i - 1) ENVIRON["ROOT"] substr($0, i + 13) }
    index($0, "state_dir = ~/.vade-state") == 1 { $0 = "state_dir = " ENVIRON["DIR"] "/state" substr($0, 26) }
    { print }' "$ROOT/examples/two-sparks.conf" > "$DIR/two-sparks.conf"
  export VADE_CONFIG="$DIR/two-sparks.conf" VADE_HOME="$DIR/home"
  # Capture the plan first: a pipeline's status is sed's, so a failed preview would exit 0.
  PLAN=$(vade up --sparks 2 --dry-run) || exit $?
  printf '%s\n' "$PLAN" | sed -n '1,/^  Endpoint /p;$p'
  exit 0
fi

: "${VADE_CONFIG:?Set VADE_CONFIG to your edited copy of examples/two-sparks.conf}"
unset VADE_API_URL      # vade verify reads it before VADE_CONFIG; check the engine started here
# Ask vade for the endpoint it will serve (your config's local_port), as the dry run prints it.
PLAN=$(vade up --sparks 2 --dry-run) || exit $?
API=$(printf '%s\n' "$PLAN" | sed -n 's/^  Endpoint  *//p')
[ -n "$API" ] || { echo "vade up --dry-run printed no Endpoint row" >&2; exit 1; }
# Keep the verify receipt out of the checkout: vade install refuses an untracked file there.
RECEIPT=$(mktemp -d)
RECEIPT=$(CDPATH= cd -- "$RECEIPT" && pwd -P)/vade-verify.json
vade install            # stage the source on both Sparks and build there
trap 'vade down' EXIT   # stop the engine and release both Sparks, even when a step fails
trap 'exit 1' HUP INT TERM
vade up --sparks 2      # rank 0 on spark-a, rank 1 on spark-b, tunnel to local_port
vade status
curl --fail-with-body -sS "$API/models"; echo
echo "Verify receipt: $RECEIPT (kept after this script exits)"
vade verify --receipt "$RECEIPT"   # drafting ON and OFF must give identical token IDs; reports "1 machine"
