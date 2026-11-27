#!/bin/sh
# Purpose: call Vade's OpenAI-compatible API with plain curl: list the model, get one answer, stream one.
# Run:     sh examples/curl-openai.sh
# Needs:   a running engine (vade launch on the Spark, or vade up from a controller) and curl.
#          VADE_API_URL picks another endpoint (default http://127.0.0.1:18080; a trailing /v1 is fine).
# Expected output: see "curl against the OpenAI API" in examples/README.md.
set -eu
BASE=${VADE_API_URL:-http://127.0.0.1:18080}
BASE=${BASE%/}
BASE=${BASE%/v1}

echo '== GET /v1/models'
curl --fail-with-body -sS "$BASE/v1/models"
echo

# No sampling fields, so the engine decodes greedily: the same answer every run.
# "enable_thinking": false skips the model's thinking and returns only the answer.
echo '== POST /v1/chat/completions'
curl --fail-with-body -sS "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp4-local",
    "messages": [{"role": "user", "content": "Name the capital of France in one word."}],
    "max_tokens": 16,
    "enable_thinking": false
  }'
echo

echo '== POST /v1/chat/completions, streamed (server-sent events)'
curl --fail-with-body -sSN "$BASE/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp4-local",
    "messages": [{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
    "max_tokens": 16,
    "enable_thinking": false,
    "stream": true
  }'
