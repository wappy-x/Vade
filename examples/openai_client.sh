#!/bin/sh
# Streaming, tool calls, and prompt reuse through the OpenAI-compatible API.
# Run: sh examples/openai_client.sh [stream|tools|cache ...]
# Needs: Rust 1.99.0, curl, and a running engine. No packages to install.
# VADE_API_URL selects the endpoint. VADE_API_KEY_FILE selects a bearer-key file.
# The cache demo needs an engine started with VADE_FLASHNEXT_PREFIX_CACHE=1 (one Spark or two).
set -eu
. "$(dirname -- "$0")/api-common.sh"
BASE=${VADE_API_URL:-http://127.0.0.1:18080}
BASE=${BASE%/}; BASE=${BASE%/v1}
DIR=$(mktemp -d); CHILD=
cleanup() {
  if [ -n "$CHILD" ]; then kill "$CHILD" 2>/dev/null || :; wait "$CHILD" 2>/dev/null || :; fi
  rm -rf -- "$DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
get() { "$VADE_TOOLS" json-get "$@"; }
value() { "$VADE_TOOLS" json-value "$@"; }
quote() { "$VADE_TOOLS" json-quote; }
request() {
  path=$1; shift
  if [ -n "${VADE_API_KEY_FILE:-}" ]; then
    set -- --header "Authorization: Bearer $(cat "$VADE_API_KEY_FILE")" "$@"
  fi
  curl --fail-with-body -sS --max-time 600 --header 'Content-Type: application/json' "$BASE$path" "$@"
}
chat() { request /v1/chat/completions --data-binary "$1"; }
MODEL=$(request /v1/models | get data 0 id)
[ -n "$MODEL" ] || { echo 'No served model.' >&2; exit 1; }
M=$(printf '%s' "$MODEL" | quote)
printf '[model] %s at %s\n' "$MODEL" "$BASE"
stream() {
  mkfifo "$DIR/stream"
  request /v1/chat/completions -N --data-binary "{\"model\":$M,\"stream\":true,\"max_tokens\":64,\"enable_thinking\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Write a haiku about a small computer that runs a large model.\"}]}" > "$DIR/stream" &
  CHILD=$!; usage=null; done_event=0; chunks=0
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | tr -d '\r')
    case $line in 'data: [DONE]') done_event=1; continue ;; 'data: '*) chunk=${line#data: } ;; *) continue ;; esac
    [ "$done_event" -eq 0 ] || { echo 'Data after stream completion.' >&2; return 1; }
    printf '%s' "$chunk" | "$VADE_TOOLS" json-stream-check
    chunks=$((chunks+1))
    printf '%s' "$chunk" | "$VADE_TOOLS" json-print choices 0 delta content
    current=$(printf '%s' "$chunk" | value usage)
    [ "$current" = null ] || usage=$current
  done < "$DIR/stream"
  status=0; wait "$CHILD" || status=$?; CHILD=
  [ "$status" -eq 0 ] || return "$status"
  [ "$done_event" -eq 1 ] && [ "$chunks" -gt 0 ] || { echo 'Incomplete event stream.' >&2; return 1; }
  rm "$DIR/stream"
  printf '\n[usage] prompt %s, completion %s\n' "$(printf '%s' "$usage" | get prompt_tokens)" "$(printf '%s' "$usage" | get completion_tokens)"
}
tools() {
  schema='{"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}'
  messages='{"role":"user","content":"What is the weather in Paris right now?"}'
  body="\"model\":$M,\"tools\":[$schema],\"max_tokens\":256,\"enable_thinking\":false"
  reply=$(chat "{$body,\"messages\":[$messages]}")
  message=$(printf '%s' "$reply" | value choices 0 message)
  call=$(printf '%s' "$message" | value tool_calls 0)
  args=$(printf '%s' "$call" | get function arguments)
  city=$(printf '%s' "$args" | value city)
  id=$(printf '%s' "$call" | value id)
  [ "$id" != null ] && [ "$city" != null ] || { echo 'No weather tool call.' >&2; exit 1; }
  printf '[finish_reason] %s\n' "$(printf '%s' "$reply" | get choices 0 finish_reason)"
  printf '[tool call] %s(%s)\n' "$(printf '%s' "$call" | get function name)" "$args"
  result=$(printf '{"city":%s,"temperature_c":18,"sky":"clear"}' "$city" | quote)
  reply=$(chat "{$body,\"messages\":[$messages,$message,{\"role\":\"tool\",\"tool_call_id\":$id,\"content\":$result}]}")
  printf '[answer] %s\n' "$(printf '%s' "$reply" | get choices 0 message content)"
}
cache() {
  notes='You review code against these house rules.'; i=1
  while [ "$i" -le 120 ]; do notes="$notes Rule $i: keep function $i under $((10+i)) lines."; i=$((i+1)); done
  notes=$(printf '%s' "$notes" | quote)
  messages="{\"role\":\"system\",\"content\":$notes},{\"role\":\"user\",\"content\":\"How many lines may function 7 have? Answer with the number.\"}"
  for turn in 1 2; do
    reply=$(chat "{\"model\":$M,\"max_tokens\":16,\"enable_thinking\":false,\"prompt_cache_key\":\"examples-review\",\"messages\":[$messages]}")
    answer=$(printf '%s' "$reply" | get choices 0 message content)
    printf '[turn %s] prompt %s tokens, cached %s: %s\n' "$turn" "$(printf '%s' "$reply" | get usage prompt_tokens)" "$(printf '%s' "$reply" | get usage prompt_tokens_details cached_tokens)" "$answer"
    answer=$(printf '%s' "$answer" | quote)
    messages="$messages,{\"role\":\"assistant\",\"content\":$answer},{\"role\":\"user\",\"content\":\"And function 9? Answer with the number.\"}"
  done
}
[ "$#" -gt 0 ] || set -- stream tools cache
for demo do
  printf '== %s\n' "$demo"
  case $demo in stream) stream ;; tools) tools ;; cache) cache ;; *) echo 'Use stream, tools, or cache.' >&2; exit 1 ;; esac
done
