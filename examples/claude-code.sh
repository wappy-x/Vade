#!/bin/sh
# Purpose: point Claude Code at Vade's Anthropic Messages API and ask it one question.
# Run:     sh examples/claude-code.sh                 (one-shot: prints the answer)
#          sh examples/claude-code.sh --interactive   (opens Claude Code in this folder)
# Needs:   a running engine, curl, Python 3, and Claude Code (`claude`) on PATH.
#          VADE_API_URL picks another endpoint (default http://127.0.0.1:18080).
# Expected output: see "Claude Code" in examples/README.md.
# This is not a live-qualified client route; harness/docs/passthrough-matrix.md says what is checked.
set -eu
BASE=${VADE_API_URL:-http://127.0.0.1:18080}
BASE=${BASE%/}
BASE=${BASE%/v1}

# Read the served model and its context window from the running engine.
CAPABILITIES=$(curl --fail-with-body -sS "$BASE/v1/capabilities")
field() { printf '%s' "$CAPABILITIES" | python3 -I -c "import json, sys; print(json.load(sys.stdin)['$1'])"; }
MODEL=$(field model)
CONTEXT=$(field context_limit)
echo "Vade at $BASE serves $MODEL with a $CONTEXT-token context." >&2

# Messages lives at the API root, without /v1. A local engine without a key
# accepts any token. Claude Code's background calls use the same model.
export ANTHROPIC_BASE_URL="$BASE"
export ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN:-no-key-required}
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CONTEXT"
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export ENABLE_TOOL_SEARCH=false

# --bare skips hooks, plugins, memory and CLAUDE.md. Without it Claude Code adds a
# system-role entry to the messages, which the engine refuses today with
# 400 Unexpected role "system" (engine/crates/serve/src/anthropic.rs).
if [ "${1:-}" = --interactive ]; then
  exec claude --bare
fi
exec claude --bare -p 'In one sentence: what does a GPU do that a CPU does not?'
