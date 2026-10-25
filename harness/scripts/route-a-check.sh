#!/bin/sh
# Route A self-check: run on a Spark from the checkout root, after INSTALL.md
# steps 1 to 3 (toolchains, clone and kernel build, verified model).
#
#   sh harness/scripts/route-a-check.sh
#
# Steps: toolchain, build, model (vade detect), vade launch, one chat request,
# vade verify, vade down. Each step prints PASS or FAIL with its elapsed
# seconds; step output goes to a private log folder that the summary names.
#
# The check runs in a new private Vade home of its own, ~/.vade-route-a.XXXXXX,
# so it never reads or stops ~/.vade or an engine started from it, and no
# other vade command can find its engine. That is a first setup: the launch
# builds the engine and kernels there. The script tries to stop its engine at
# the end, also when an earlier step failed or the script is interrupted
# (Ctrl-C, kill, a dropped SSH session), and reports FAIL when that stop
# fails. After a PASS it removes that home; after a FAIL it keeps it for
# inspection and names it.
#
# VADE_DRY=1 prints the commands without running them (works on any computer).
# VADE_MODELS picks the checkpoint folder, as for vade launch (default
# ~/models); the model step fails when vade would load one outside it.
# VADE_HOME, VADE_CONFIG, VADE_API_URL, VADE_BUNDLE and VADE_SOURCE are
# ignored: the check always builds this checkout into its own home.
set -u
umask 077
unset CDPATH
dry=${VADE_DRY:-0}
root=$(cd -P "$(dirname "$0")/../.." && pwd) || exit 2
if [ ! -f "$root/cli/Cargo.toml" ]; then
    printf '%s\n' "route-a-check: run this from a Vade checkout (no cli/Cargo.toml under $root)." >&2
    exit 2
fi
vade=$root/cli/target/release/vade
export PATH="$HOME/.cargo/bin:$PATH"
# Pin every instance this script means: the local launcher, this checkout's
# source, the CLI path above and the endpoint in its own engine record.
unset VADE_CONFIG VADE_API_URL VADE_BUNDLE VADE_SOURCE CARGO_BUILD_TARGET
if [ "$dry" = 1 ]; then
    logs="(VADE_DRY: no logs)" vade_home="$HOME/.vade-route-a.XXXXXX"
else
    logs=$(mktemp -d "${TMPDIR:-/tmp}/vade-route-a.XXXXXX") && logs=$(cd -P "$logs" && pwd) || exit 2
    vade_home=$(mktemp -d "$HOME/.vade-route-a.XXXXXX") && vade_home=$(cd -P "$vade_home" && pwd) || exit 2
fi
export VADE_HOME="$vade_home"
record=$vade_home/state/run.conf
step_no=0 passed=0 failed=0 launched=0 model_path= endpoint= began=$(date +%s)

say() { printf '%s\n' "$*"; }
# result NAME STATUS START [DETAIL]
result() {
    printf '[%d/7] %-10s %-4s %4d s%s\n' "$step_no" "$1" "$2" "$(( $(date +%s) - $3 ))" "${4:+  $4}"
    case $2 in PASS) passed=$((passed + 1)) ;; FAIL) failed=$((failed + 1)) ;; esac
}
# run NAME COMMAND...: the command's output goes to NAME.log; 0 means success.
run() {
    name=$1
    shift
    if [ "$dry" = 1 ]; then
        say "  would run: $*"
        return 0
    fi
    "$@" > "$logs/$name.log" 2>&1 < /dev/null
}
# tail_log NAME: show the end of a failed step's log.
tail_log() {
    [ "$dry" = 1 ] && return 0
    tail -n 15 "$logs/$1.log" | sed 's/^/    /'
}
# unescape QUOTES: decode one value from stdin. QUOTES=1 reads vade's quoted
# display form (Rust {:?}: \\ \" \'), QUOTES=0 a key=value record (\\). Any
# other escape (a newline or control character in the path) fails, so a
# spelling is never mistaken for the path.
unescape() {
    awk -v quotes="$1" '{
        out = ""
        for (i = 1; i <= length($0); i++) {
            c = substr($0, i, 1)
            if (c == "\\") {
                c = substr($0, ++i, 1)
                if (c != "\\" && !(quotes && (c == "\"" || c == "\047"))) exit 1
            }
            out = out c
        }
        printf "%s", out
    }'
}
# field KEY FILE: one decoded value from a key=value record.
field() { sed -n "s/^$1=//p" "$2" 2>/dev/null | head -n 1 | unescape 0; }
# owned_endpoint: the running engine's base URL from this run's record, checked
# the way vade itself reads it (phase running, same port as the saved config).
owned_endpoint() {
    port=$(field port "$record")
    case $port in '' | *[!0-9]*) return 1 ;; esac
    [ "$(field phase "$record")" = running ] && [ "$(field port "$vade_home/config")" = "$port" ] || return 1
    printf 'http://127.0.0.1:%s\n' "$port"
}
# stop_owned: vade down when this run's launch left an engine record in its
# private home; 0 means nothing of this run is left running.
stop_owned() {
    [ "$launched" = 1 ] && [ -e "$record" ] || return 0
    "$vade" down > "$logs/down.log" 2>&1 < /dev/null && [ ! -e "$record" ]
}
# on_exit CODE: stop this run's engine if a step did not get to, then remove
# the private home after a PASS (keep it after a FAIL) and exit with CODE, or
# 1 when the stop fails after a success.
on_exit() {
    code=$1
    trap - EXIT HUP INT TERM
    if [ "$launched" = 1 ] && [ -e "$record" ]; then
        if stop_owned; then
            say "route-a-check: stopped the engine this run started"
        else
            say "route-a-check: could not stop the engine this run started; see $logs/down.log, then run: VADE_HOME=\"$vade_home\" \"$vade\" down"
            [ "$code" != 0 ] || code=1
        fi
    fi
    if [ "$dry" != 1 ]; then
        if [ "$code" = 0 ]; then
            rm -rf "${vade_home:?}"
        elif ! rmdir "$vade_home" 2>/dev/null; then
            say "route-a-check: kept this run's Vade home for inspection: $vade_home (remove it with rm -rf when done)"
        fi
    fi
    [ "$code" -lt 129 ] || say "route-a-check: interrupted (exit $code). Logs: $logs"
    exit "$code"
}
trap 'on_exit $?' EXIT
trap 'on_exit 129' HUP
trap 'on_exit 130' INT
trap 'on_exit 143' TERM

say "route-a-check: checkout $root"
say "route-a-check: private VADE_HOME $vade_home, VADE_MODELS ${VADE_MODELS:-(unset)}"
say "route-a-check: logs $logs"

# 1. Toolchain and GPU: the INSTALL.md step 1 and 2 results this needs.
step_no=1 t=$(date +%s) problems=
if [ "$dry" = 1 ]; then
    say "  would check: uname -m, nvidia-smi (GB10, no compute processes), cc git curl sha256sum, rustup toolchains 1.99.0 and nightly-2026-10-01 (rust-src, nvptx64-nvidia-cuda)"
    result toolchain DRY "$t"
else
    [ "$(uname -m)" = aarch64 ] || problems="$problems; this is $(uname -m), not a Spark (aarch64)"
    for tool in cc git curl sha256sum nvidia-smi rustup; do
        command -v "$tool" > /dev/null 2>&1 || problems="$problems; $tool is missing"
    done
    if command -v nvidia-smi > /dev/null 2>&1; then
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
        case $gpu in *GB10*) ;; *) problems="$problems; nvidia-smi shows '${gpu:-no GPU}', not GB10" ;; esac
        apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        [ -z "$apps" ] || problems="$problems; the GPU is in use, for example by a running vade engine (compute pids: $(printf '%s' "$apps" | tr '\n' ' '))"
    fi
    if command -v rustup > /dev/null 2>&1; then
        cargo +1.99.0 --version > /dev/null 2>&1 || problems="$problems; Rust 1.99.0 is not installed"
        rustup component list --toolchain nightly-2026-10-01 --installed 2>/dev/null | grep -q '^rust-src' \
            || problems="$problems; nightly-2026-10-01 with rust-src is not installed"
        rustup target list --toolchain nightly-2026-10-01 --installed 2>/dev/null | grep -q '^nvptx64-nvidia-cuda$' \
            || problems="$problems; nightly-2026-10-01 has no nvptx64-nvidia-cuda target"
    fi
    if [ -z "$problems" ]; then result toolchain PASS "$t"; else result toolchain FAIL "$t" "${problems#; }"; fi
fi

# 2. Build the CLI as Route A does, into cli/target whatever CARGO_TARGET_DIR
# says, and check from Cargo's own report that it produced the CLI run below.
step_no=2 t=$(date +%s)
set -- cargo +1.99.0 build --release --offline --manifest-path "$root/cli/Cargo.toml" \
    --target-dir "$root/cli/target" --message-format=json-render-diagnostics
if [ "$failed" -gt 0 ]; then
    result build SKIP "$t"
elif [ "$dry" = 1 ]; then
    say "  would run: $*"
    result build DRY "$t"
elif ! "$@" > "$logs/build.json" 2> "$logs/build.log" < /dev/null; then
    result build FAIL "$t" "see $logs/build.log"
    tail_log build
elif ! grep -qF "\"executable\":\"$(printf '%s' "$vade" | sed 's/[\\"]/\\&/g')\"" "$logs/build.json"; then
    result build FAIL "$t" "Cargo did not report $vade (a build.target setting?); see $logs/build.json"
else
    result build PASS "$t"
fi

# 3. The checkpoint vade launch will load, chosen by vade itself: VADE_MODELS,
# else ~/models (the private home has no saved model yet).
step_no=3 t=$(date +%s)
if [ "$failed" -gt 0 ]; then
    result model SKIP "$t"
elif [ "$dry" = 1 ]; then
    run model "$vade" detect
    result model DRY "$t"
elif ! run model "$vade" detect; then
    result model FAIL "$t" "see $logs/model.log (INSTALL.md step 3)"
    tail_log model
elif ! model_path=$(sed -n 's/^Models: "\(.*\)"$/\1/p' "$logs/model.log" | tail -n 1 | unescape 1) || [ -z "$model_path" ]; then
    result model FAIL "$t" "cannot read the model path from vade detect; see $logs/model.log"
elif ! real=$(cd -P "$model_path" 2>/dev/null && pwd); then
    result model FAIL "$t" "vade detect named $model_path, which cannot be opened"
else
    wanted=
    if [ -n "${VADE_MODELS:-}" ]; then
        case $VADE_MODELS in
            "~") wanted=$HOME ;;
            "~/"*) wanted=$HOME/${VADE_MODELS#"~/"} ;;
            *) wanted=$VADE_MODELS ;;
        esac
        wanted=$(cd -P "$wanted" 2>/dev/null && pwd) || wanted=
    fi
    if [ -n "${VADE_MODELS:-}" ] && [ -z "$wanted" ]; then
        result model FAIL "$t" "VADE_MODELS folder $VADE_MODELS cannot be opened"
    elif [ -n "$wanted" ] && [ "$real" != "$wanted" ] && [ "${real#"$wanted"/}" = "$real" ]; then
        result model FAIL "$t" "vade would load $model_path, not one under VADE_MODELS $wanted"
    else
        result model PASS "$t" "$model_path"
    fi
fi

# 4. vade launch: first setup, engine build and start, until the engine is
# ready. It always uses the local launcher; with no terminal it skips chat.
step_no=4 t=$(date +%s)
if [ "$failed" -gt 0 ]; then
    result launch SKIP "$t"
elif [ "$dry" = 1 ]; then
    run launch "$vade" launch
    result launch DRY "$t"
else
    # Only this run knows its private home, so an engine record there is this
    # run's engine, and cleanup stops it.
    launched=1
    if ! run launch "$vade" launch || [ ! -f "$record" ]; then
        result launch FAIL "$t" "see $logs/launch.log"
        tail_log launch
    elif ! loaded=$(field model_path "$record"); then
        result launch FAIL "$t" "cannot read the loaded model path from $record"
    elif [ "$loaded" != "$model_path" ]; then
        result launch FAIL "$t" "the engine loaded $loaded, not step 3's $model_path"
    else
        result launch PASS "$t"
    fi
fi

# 5. One chat request through the OpenAI-compatible API, thinking off, at the
# endpoint this run's engine record names.
step_no=5 t=$(date +%s)
if [ "$failed" -gt 0 ]; then
    result chat SKIP "$t"
elif [ "$dry" = 1 ]; then
    endpoint="<endpoint>"
    say "  would run: curl GET $endpoint/v1/models, then POST $endpoint/v1/chat/completions \"Say hello in five words.\" ($endpoint from $record)"
    result chat DRY "$t"
elif ! endpoint=$(owned_endpoint); then
    endpoint=
    result chat FAIL "$t" "no running endpoint in $record"
else
    model=$(curl --fail --silent --show-error --max-time 30 "$endpoint/v1/models" 2> "$logs/chat.log" \
        | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | head -n 1)
    body='{"model":"'"$model"'","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"enable_thinking":false}'
    if [ -n "$model" ] && curl --fail-with-body --silent --show-error --max-time 300 \
        -H 'Content-Type: application/json' -d "$body" "$endpoint/v1/chat/completions" \
        > "$logs/chat.json" 2>> "$logs/chat.log" && grep -q '"content": *"[^"]' "$logs/chat.json"; then
        reply=$(sed -n 's/.*"content": *"\([^"]*\)".*/\1/p' "$logs/chat.json" | head -n 1 | cut -c1-80)
        result chat PASS "$t" "$model at $endpoint/v1: $reply"
    else
        result chat FAIL "$t" "no reply from $endpoint/v1 (model '${model}'); see $logs/chat.log and chat.json"
        tail_log chat
    fi
fi

# 6. vade verify against that same endpoint: drafting OFF and ON must give the
# same token IDs. The receipt goes to the log folder, so the checkout stays
# clean for a later Route B.
step_no=6 t=$(date +%s)
if [ "$failed" -gt 0 ]; then
    result verify SKIP "$t"
elif run verify "$vade" verify --url "$endpoint" --receipt "$logs/vade-verify.json"; then
    if [ "$dry" = 1 ]; then
        result verify DRY "$t"
    else
        result verify PASS "$t" "$(grep -m 1 'IDENTICAL' "$logs/verify.log")"
    fi
else
    result verify FAIL "$t" "see $logs/verify.log"
    tail_log verify
fi

# 7. vade down for this run's engine, in its private home.
step_no=7 t=$(date +%s)
if [ "$dry" = 1 ]; then
    run down "$vade" down
    result down DRY "$t"
elif [ "$launched" = 0 ]; then
    result down SKIP "$t" "nothing was started"
elif [ ! -e "$record" ]; then
    launched=0
    result down SKIP "$t" "vade launch left no engine record"
elif stop_owned; then
    launched=0
    result down PASS "$t"
else
    launched=0
    result down FAIL "$t" "see $logs/down.log; then run: VADE_HOME=\"$vade_home\" \"$vade\" down"
    tail_log down
fi

total=$(( $(date +%s) - began ))
if [ "$dry" = 1 ]; then
    say "route-a-check: DRY (nothing ran, $total s)"
elif [ "$failed" = 0 ] && [ "$passed" = 7 ]; then
    say "route-a-check: PASS ($passed/7 steps, $total s)"
else
    say "route-a-check: FAIL ($passed/7 passed, $failed failed, $total s). Logs: $logs"
    exit 1
fi
