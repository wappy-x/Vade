#!/bin/sh
# Purpose: point the Codex CLI at Vade's Responses API through a private Codex home, then ask it one question.
# Run:     sh examples/codex.sh                 (one-shot: prints the answer)
#          sh examples/codex.sh --interactive   (opens Codex in this folder)
# Needs:   a running engine, curl, Rust 1.99.0, Codex (`codex`) on PATH, and vade-harness, built with
#          cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml --bins
#          VADE_API_URL picks another endpoint (default http://127.0.0.1:18080).
# Expected output: see "Codex CLI" in examples/README.md.
# Your own ~/.codex is never read or changed: setup writes a new home with its own
# config.toml, and launch.sh starts Codex there in a clean environment.
set -eu
. "$(dirname -- "$0")/api-common.sh"
BASE=${VADE_API_URL:-http://127.0.0.1:18080}
BASE=${BASE%/}
BASE=${BASE%/v1}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
HARNESS=${VADE_HARNESS:-$ROOT/harness/target/release/vade-harness}

# Read the served model and its limits from the running engine; setup never guesses them.
CAPABILITIES=$(curl --fail-with-body -sS "$BASE/v1/capabilities")

# The home must be a new folder under an existing private parent, given by its real path.
PARENT=$(mktemp -d)
PARENT=$(CDPATH= cd -- "$PARENT" && pwd -P)
"$HARNESS" setup codex --endpoint "$BASE" --model "$(field model)" \
  --target direct --dedicated-home --write "$PARENT/codex" \
  --context-window "$(field context_limit)" --max-output-tokens "$(field output_limit)" 2>/dev/null
echo "Codex home: $PARENT/codex (config.toml selects provider vade, wire_api responses)" >&2

if [ "${1:-}" = --interactive ]; then
  exec "$PARENT/codex/launch.sh"
fi
exec "$PARENT/codex/launch.sh" exec --skip-git-repo-check \
  'In one sentence: what does a GPU do that a CPU does not? Do not run any commands.' < /dev/null
