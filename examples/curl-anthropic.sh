#!/bin/sh
# Purpose: call Vade's Anthropic Messages API with plain curl: count tokens, get one answer, stream one.
# Run:     sh examples/curl-anthropic.sh
# Needs:   a running engine (vade launch on the Spark, or vade up from a controller) and curl.
#          VADE_API_URL picks another endpoint (default http://127.0.0.1:18080; a trailing /v1 is fine).
# Expected output: see "curl against the Anthropic API" in examples/README.md.
set -eu
BASE=${VADE_API_URL:-http://127.0.0.1:18080}
BASE=${BASE%/}
BASE=${BASE%/v1}

# Messages accepts any model name as an alias for the loaded model, so Anthropic
# clients can keep their own. No anthropic-version header is needed.
echo '== POST /v1/messages/count_tokens'
curl --fail-with-body -sS "$BASE/v1/messages/count_tokens" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp4-local",
    "messages": [{"role": "user", "content": "Name the capital of France in one word."}]
  }'
echo

# "thinking": {"type": "disabled"} returns only the answer; no sampling fields means greedy.
echo '== POST /v1/messages'
curl --fail-with-body -sS "$BASE/v1/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp4-local",
    "max_tokens": 16,
    "thinking": {"type": "disabled"},
    "system": "Answer in one word.",
    "messages": [{"role": "user", "content": "Name the capital of France."}]
  }'
echo

echo '== POST /v1/messages, streamed (server-sent events)'
curl --fail-with-body -sSN "$BASE/v1/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp4-local",
    "max_tokens": 16,
    "thinking": {"type": "disabled"},
    "stream": true,
    "messages": [{"role": "user", "content": "Count from 1 to 5, separated by spaces."}]
  }'
