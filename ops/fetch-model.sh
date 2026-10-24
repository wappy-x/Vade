#!/bin/sh
# Download a pinned snapshot, verify it, then publish a trusted sibling manifest.
# Qualified FP4-Local snapshot (33 files, 102262310289 bytes; INSTALL.md step 3):
# sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"
# Requires python3 with venv/pip, sha256sum, find, sort and download access.
# No HF token or click-through gate for this public revision (API checked 2026-10-24).
# Publisher licence: nvidia-open-model-agreement. Budget 120 GiB free for a fresh
# model + installation on one volume; see INSTALL.md for cache/headroom details.
# Expected sibling manifest: cli/examples/qwen3.8-flash-next-fp4-local.sha256.
set -eu
umask 077
exec 3>&2 2>/dev/null
export LC_ALL=C HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_TELEMETRY=1
state= child= locked= isolating=
reason='Model setup failed; check the tools and available disk space.'
cleanup() {
    result=$?
    trap - 0
    trap '' INT TERM HUP
    if [ -n "$child" ]; then
        kill -TERM "$child" 2>/dev/null || :
        wait "$child" 2>/dev/null || :
    fi
    if [ -n "$locked" ]; then
        if [ -n "$isolating" ] && [ -d "$state/huggingface" ]; then
            if ! mv "$state/huggingface" "$model/.cache/"; then
                reason='Could not restore download metadata; preserve the sibling fetch directory.'
                result=1
            fi
        fi
        rmdir "$state/lock" || result=1
    fi
    [ "$result" -eq 0 ] || printf '%s\n' "$reason" >&3
    exit "$result"
}
trap cleanup 0
trap 'reason="Interrupted; run the same command again."; exit 130' INT
trap 'reason="Interrupted; run the same command again."; exit 143' TERM HUP
run() {
    "$@" >/dev/null 2>&1 &
    child=$!
    wait "$child"
    child=
}
reason='Supply a repository, full revision hash, and model directory.'
[ "$#" -eq 3 ]
repo=$1 revision=$2
case "$revision" in *[!0-9a-f]*|'') exit 1 ;; esac
[ "${#revision}" -eq 40 ]
reason='Install python3 with venv, pip, sha256sum, find and sort first.'
for tool in python3 sha256sum find sort; do command -v "$tool" >/dev/null; done
reason='Could not prepare the model directory; check its permissions.'
mkdir -p -- "$3"
model=$(cd -- "$3" && pwd -P)
[ "$model" != / ]
state=$model.fetch
reason='Could not prepare the private sibling fetch directory.'
[ ! -L "$state" ]
if [ ! -d "$state" ]; then mkdir -m 700 "$state"; fi
reason='Another fetch may be running; check the sibling fetch lock directory.'
mkdir "$state/lock"
locked=yes
export HF_HOME="$state/hub" HF_TOKEN_PATH="$state/hub/token" HF_HUB_CACHE="$state/hub/cache"
unset HUGGING_FACE_HUB_TOKEN
reason='Could not create the private Python environment; install python3-venv.'
run python3 -m venv "$state/venv"
reason='Could not install huggingface_hub 2.2.0; check pip and internet access.'
run "$state/venv/bin/python" -m pip install --disable-pip-version-check --no-input 'huggingface_hub==2.2.0'
hf=$state/venv/bin/hf
reason='Download failed; check internet access, disk space and repository access, then rerun.'
run "$hf" download "$repo" --revision "$revision" --local-dir "$model"
reason='Could not set download metadata aside; stop other downloads and rerun.'
[ ! -e "$state/huggingface" ]
isolating=yes
mv "$model/.cache/huggingface" "$state/huggingface"
reason='Model verification failed; check for missing, extra or damaged files before rerunning.'
run "$hf" cache verify "$repo" --revision "$revision" --local-dir "$model" \
    --fail-on-missing-files --fail-on-extra-files
reason='Could not restore download metadata; preserve the sibling fetch directory.'
mv "$state/huggingface" "$model/.cache/"
reason='Could not write or check the checksum manifest; check files and disk space.'
cd "$model"
find . \( -type f -o -type l \) ! -path './.cache/huggingface/*' -exec sha256sum {} + > "$state/unsorted"
sort "$state/unsorted" > "$state/manifest"
run sha256sum -c "$state/manifest"
count=0
while IFS= read -r entry; do count=$((count + 1)); done < "$state/manifest"
[ "$count" -gt 0 ]
mv "$state/manifest" "$model.sha256"
rmdir "$state/lock"
locked=
printf 'verified %s files\n' "$count"
