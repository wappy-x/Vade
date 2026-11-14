#!/bin/sh
# Setup on one Spark: the prebuilt bundle when it fits, else a source build.
# From a checkout:  sh scripts/install.sh
# With no checkout: curl -fsSL https://vade.lol/install.sh | sh
# No sudo, shell profile edits, or model downloads.
# Every statement is inside main, and main runs on the last line, so a cut-off
# download is a syntax error and runs nothing. Keep main "$@" as the last line:
# the release upload checks it.
main() {
set -eu

# With --json, standard output carries JSON lines only: step events, then one last
# line with "event":"result". Descriptor 5 is that output; every other line goes
# to standard error. Fields are only ever added. Every program this script starts
# gets "5>&-": one that stays alive must not keep the reader of the JSON waiting.
json_text() { # $1 as the inside of a JSON string, on one line
    printf '%s\n' "$1" | sed -e ':a' -e '$!{' -e 'N' -e 'ba' -e '}' -e 's/\n/ /g' \
        -e 's/[[:cntrl:]]/ /g' -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}
seconds() { # whole seconds since the start, or null without a clock
    if [ -n "${started-}" ]; then printf '%s' "$(($(date +%s) - started))"; else printf null; fi
}
step() { # $1 = the step that starts now
    [ "${json-false}" = false ] ||
        printf '{"schema":"vade.install.v1","ok":true,"error":null,"next":null,"event":"step","step":"%s","seconds":%s}\n' "$1" "$(seconds)" >&5
}
failed() { # $1 = "what happened. Next: what to do"; the fixed word comes from $fault
    [ "${json-false}" = true ] || return 0
    message=${1%% Next: *} next=${1#* Next: }
    if [ "$next" = "$1" ]; then next=null; else next="\"$(json_text "$next")\""; fi
    [ "${sent-false}" = false ] || return 0
    # The line is built first; then signals wait, so "sent" and the line are never apart.
    line=$(printf '{"schema":"vade.install.v1","ok":false,"error":{"code":"%s","message":"%s"},"next":%s,"event":"result","action":"%s","check":%s,"seconds":%s}' \
        "${fault-failed}" "$(json_text "$message")" "$next" "${action-install}" "${check-false}" "$(seconds)")
    trap '' HUP INT TERM; sent=true; printf '%s\n' "$line" >&5
}
fail() {
    ticker_stop
    if [ "${check-false}" = true ]; then
        printf '%s\n' "Check: the install would stop here: $*"
        failed "$*"
        exit 0
    fi
    printf '%s\n' "install: $*" >&2
    if [ "${action-install}" != install ]; then :
    elif [ -n "${root-}" ]; then
        printf 'Retry: sh "%s/scripts/install.sh"%s\n' "$root" "${retry-}" >&2
    elif [ "${piped-false}" = true ]; then
        printf 'Retry: curl -fsSL https://vade.lol/install.sh | sh%s\n' "${retry:+ -s --$retry}" >&2
    fi
    failed "$*"
    exit 1
}
# A long step with no output of its own prints one "still" line every 15 s (cli/docs/ux.md, rule 7).
ticker_pid=
ticker() { # $1 = what runs now
    command -v sleep >/dev/null 2>&1 || return 0
    # The sleep holds none of our outputs, so a stopped ticker never keeps a reader waiting.
    ( waited=0
      while sleep 15 >/dev/null 2>&1 5>&- </dev/null; do
          waited=$((waited + 15))
          printf 'Install: still %s (%s s elapsed).\n' "$1" "$waited"
      done ) &
    ticker_pid=$!
}
ticker_stop() {
    [ -n "$ticker_pid" ] || return 0
    kill "$ticker_pid" 2>/dev/null || true
    wait "$ticker_pid" 2>/dev/null || true
    ticker_pid=
}
# The rules for every removal below. A home is a full path of two or more parts
# with no empty, "." or ".." part, and it is not your home folder or a folder
# above it. No part of a removed path, from / to the last name, is a link. Its
# real path is strictly inside the real home. Names come from fixed lists or
# from a listing read line by line: never a file name pattern or an unquoted list.
plain_home() { # $1 = a home folder
    case $1 in /*/*) ;; *) return 1 ;; esac
    case "$1/" in *//*|*/./*|*/../*) return 1 ;; esac
    case "$HOME/" in "$1"/*) return 1 ;; esac
    # Also by its real path: your home folder reached through a link is still yours.
    home_real=$(real_dir "$HOME") || home_real=
    [ -z "$home_real" ] || case "$home_real/" in "$1"/*) return 1 ;; esac
}
no_links() { # true when no part of $1, from / to the last name, is a link
    links_rest=${1#/} links_at=
    while [ -n "$links_rest" ]; do
        links_at="$links_at/${links_rest%%/*}"
        case $links_rest in */*) links_rest=${links_rest#*/} ;; *) links_rest= ;; esac
        [ ! -L "$links_at" ] || return 1
    done
}
real_dir() { (CDPATH= cd -P -- "$1" 2>/dev/null && pwd -P); }
# Every write, move and removal in a checked folder runs from inside it. cd -P pins the
# folder, and its physical path must equal the checked one: a link swapped in before the
# cd is refused. After the cd, a rename or a swapped parent cannot redirect a relative name.
in_dir() ( # $1 = a full folder path with no link in it, the rest = a command that uses relative names
    in_dir_at=$1; shift
    no_links "$in_dir_at" && CDPATH= cd -P -- "$in_dir_at" && [ "$(pwd -P)" = "$in_dir_at" ] || exit 1
    "$@"
)
by_name() { # $1 = a full path, the rest = a command and its flags: it gets ./<last name> in the parent
    by_name_at=$1; shift
    case ${by_name_at##*/} in ''|.|..) return 1 ;; esac
    in_dir "${by_name_at%/*}" "$@" "./${by_name_at##*/}"
}
inside() { # $1 = home, $2 = a path: true when the rules above let $2 go
    plain_home "$1" && no_links "$2" || return 1
    case $2 in "$1"/?*) ;; *) return 1 ;; esac
    case "/${2#"$1"/}/" in *//*|*/./*|*/../*) return 1 ;; esac
    [ "$(real_dir "$1")" = "$1" ] && [ "$(real_dir "$(dirname -- "$2")")" = "$(dirname -- "$2")" ]
}
# A Vade home carries the mark .vade-home, a file with the one line vade-home-v1.
# The install writes it, and the uninstall removes nothing from a home without it.
# A home from an older installer has no mark; it is known by stage/install.conf,
# which only vade install writes, with the line format=vade-local-install-v1.
marked() { # $1 = a home: true when it is a Vade home
    if [ -f "$1/.vade-home" ] && [ ! -L "$1/.vade-home" ]; then
        [ "$(sed -n 1p "$1/.vade-home" 2>/dev/null)" = vade-home-v1 ]; return
    fi
    [ -f "$1/stage/install.conf" ] && [ ! -L "$1/stage" ] && [ ! -L "$1/stage/install.conf" ] &&
        [ -n "$(sed -n '/^format=vade-local-install-v1$/p' "$1/stage/install.conf" 2>/dev/null)" ]
}
# A Vade checkout holds this file, the CLI, the engine and the harness.
checkout() {
    [ -f "$1/scripts/install.sh" ] && [ -f "$1/cli/Cargo.toml" ] && [ -f "$1/engine/Cargo.toml" ] &&
        [ -f "$1/harness/Cargo.toml" ]
}
# What an older installer leaves when it stops before vade install ends: a bundles
# folder and source.XXXXXXXX clones. Each entry must prove itself by its exact shape:
# - a clone: a real folder that is a whole Vade checkout;
# - in bundles, a real folder that is either an unpacked bundle, vade-*-linux-aarch64-gb10
#   whose MANIFEST starts with format=vade-release-v1 and names that folder, or a
#   download cache, .partial-<64 hex digits>, that holds only bundle.part and an empty lock.
# One entry that does not prove itself and the whole folder is refused. An empty bundles
# folder or an empty cache proves nothing: only a bundle or a cache with content sets bundle_proof.
each_entry() { # $1 = a folder, $2 = a test: true when $2 holds for every entry (patterns must be on)
    each_result=0
    for each in "$1"/* "$1"/.[!.]* "$1"/..?*; do
        [ -e "$each" ] || [ -L "$each" ] || continue
        "$2" "$each" || { each_result=1; break; }
    done
    return "$each_result"
}
with_patterns() { # run "$@" with file name patterns on, then put the setting back
    case $- in *f*) patterns_off=true ;; *) patterns_off=false ;; esac
    set +f
    patterns_result=0
    "$@" || patterns_result=1
    [ "$patterns_off" = false ] || set -f
    return "$patterns_result"
}
proven_clone() { [ ! -L "$1" ] && [ -d "$1" ] && checkout "$1"; } # $1 = a source.XXXXXXXX folder
cache_entry() { # $1 = a path in a download cache
    case ${1##*/} in
        bundle.part) [ ! -L "$1" ] && [ -f "$1" ] ;;
        lock) [ ! -L "$1" ] && [ -d "$1" ] && empty_dir "$1" ;;
        *) return 1 ;;
    esac
}
proven_bundle() { # $1 = a path in bundles
    [ ! -L "$1" ] && [ -d "$1" ] || return 1
    case ${1##*/} in
        vade-*-linux-aarch64-gb10)
            [ -f "$1/MANIFEST" ] && [ ! -L "$1/MANIFEST" ] || return 1
            [ "$(sed -n 1p "$1/MANIFEST" 2>/dev/null)" = format=vade-release-v1 ] || return 1
            while IFS= read -r manifest_line || [ -n "$manifest_line" ]; do
                [ "$manifest_line" != "name=${1##*/}" ] || { bundle_proof=true; return 0; }
            done < "$1/MANIFEST"
            return 1 ;;
        .partial-*)
            cache_sum=${1##*/.partial-}
            [ "${#cache_sum}" -eq 64 ] || return 1
            case $cache_sum in *[!0-9a-f]*) return 1 ;; esac
            each_entry "$1" cache_entry || return 1
            empty_dir "$1" || bundle_proof=true ;;
        *) return 1 ;;
    esac
}
leftover_entry() { # $1 = a path at the top of a folder without a mark
    case ${1##*/} in
        .install-lock) [ ! -L "$1" ] && [ -d "$1" ] ;;
        bundles) [ ! -L "$1" ] && [ -d "$1" ] || return 1
            bundle_proof=false
            each_entry "$1" proven_bundle || return 1
            [ "$bundle_proof" = false ] || left_any=true ;;
        source.????????) proven_clone "$1" && left_any=true ;;
        *) return 1 ;;
    esac
}
leftovers_only() { # $1 = a folder: true when it holds only proven leftovers, and at least one
    left_any=false
    with_patterns each_entry "$1" leftover_entry && [ "$left_any" = true ]
}
vade_home() { marked "$1" || leftovers_only "$1"; } # $1 = a folder: true when it is Vade's
mark_here() { (umask 077 && set -C && printf '%s\n' vade-home-v1 > ./.vade-home); }
write_mark() { # $1 = a Vade home without its mark: write it, never through a link or over a file
    [ ! -e "$1/.vade-home" ] && [ ! -L "$1/.vade-home" ] || return 0
    in_dir "$1" mark_here ||
        fault=bad_home fail "cannot write $1/.vade-home. Next: check that folder and its permissions, then retry."
}
empty_dir() { # $1 = a folder: true when it holds nothing but the names $2 and $3
    # Name patterns list the folder here, so they are on for this loop even in a shell started with -f.
    case $- in *f*) patterns_off=true ;; *) patterns_off=false ;; esac
    set +f
    empty_result=0
    for entry in "$1"/* "$1"/.[!.]* "$1"/..?*; do
        [ -e "$entry" ] || [ -L "$entry" ] || continue
        case ${entry##*/} in "${2-}"|"${3-}") ;; *) empty_result=1; break ;; esac
    done
    [ "$patterns_off" = false ] || set -f
    return "$empty_result"
}
# One install or uninstall at a time on a home: the folder .install-lock is the lock.
lock_dir= made_home=false
take_lock() { # $1 = the home, $2 = install to mark the home
    if [ ! -d "$1" ]; then
        [ -d "$(dirname -- "$1")" ] ||
            fault=bad_home fail "the folder above $1 does not exist. Next: create it, or choose another VADE_HOME, then retry."
        (umask 077 && by_name "$1" mkdir --) ||
            fault=bad_home fail "cannot create $1. Next: check that you own its parent folder, then retry."
        made_home=true
    fi
    # A folder that already holds other files is not taken over: the uninstall would treat it all as Vade's.
    [ "${2-}" != install ] || vade_home "$1" || empty_dir "$1" .install-lock ||
        fault=bad_home fail "$1 already holds files and is not a Vade home (no .vade-home mark), so nothing was written. Next: look at what is in it; if it is only an earlier Vade attempt, remove it, else choose a new or empty folder for VADE_HOME; then retry."
    (umask 077 && by_name "$1/.install-lock" mkdir --) 2>/dev/null ||
        fault=busy fail "another install or uninstall is running on $1. Next: wait for it to end; if none runs, remove the empty folder $1/.install-lock, then retry."
    lock_dir="$1/.install-lock"
    # The mark is the first file in a new home, so a stopped install still leaves a Vade home.
    [ "${2-}" != install ] || write_mark "$1"
}
drop_lock() {
    [ -n "$lock_dir" ] || return 0
    by_name "$lock_dir" rmdir -- 2>/dev/null || true
    # A home this run made and left empty goes too, so a refused install leaves nothing.
    if [ "$made_home" = true ]; then
        made="${lock_dir%/.install-lock}"
        ! empty_dir "$made" .vade-home || by_name "$made/.vade-home" rm -f -- 2>/dev/null || true
        by_name "$made" rmdir -- 2>/dev/null || true
    fi
    lock_dir= made_home=false
}
# Every exit cleans up, and with --json every exit prints one result line.
sent=false download= partial= clone_tmp= installer_tmp= script_tmp=
cleanup() {
    ticker_stop
    clean_download 2>/dev/null || true
    [ -z "$clone_tmp" ] || by_name "$clone_tmp" rm -rf -- 2>/dev/null || true
    # A file that mktemp named in TMPDIR, outside the home and never checked: plain rm -f.
    [ -z "$installer_tmp" ] || rm -f -- "$installer_tmp" 2>/dev/null || true
    [ -z "$script_tmp" ] || by_name "$script_tmp" rm -f -- 2>/dev/null || true
    clone_tmp= installer_tmp= script_tmp=
    drop_lock
}
on_exit() {
    exit_status=$?
    # No signal and no failed command ends the handler before the lock goes and the result line is out.
    trap '' HUP INT TERM
    set +e
    cleanup
    [ "${json-false}" = true ] && [ "$sent" = false ] || return 0
    fault=${fault-stopped} failed "the installer stopped early with exit code $exit_status. Next: read the lines above, then run the same command again."
}
on_signal() { # $1 = exit code, $2 = the signal
    trap '' HUP INT TERM
    printf 'install: stopped by %s.\n' "$2" >&2
    fault=interrupted failed "the installer was stopped by $2. Next: run the same command again."
    exit "$1"
}
if [ "${1-}" = --help ]; then
    printf '%s\n' 'Usage: sh scripts/install.sh [--launch] [--source] [--check] [--json]' \
        '   or: curl -fsSL https://vade.lol/install.sh | sh -s -- [--launch] [--source] [--check] [--json]' \
        '   or: sh scripts/install.sh --uninstall [--all] [--check] [--json]' \
        'Install the prebuilt bundle for this Spark, then run vade install. When the bundle does not fit' \
        'or cannot be downloaded, say why, install pinned Rust if missing, and build vade, the GPU' \
        'kernels, the engine and chat from this checkout instead. With no checkout, the source build' \
        'clones public main into a new folder in ~/.vade and runs the installer of that clone.' \
        'The model is not needed yet: the script prints the vade pull command for it.' \
        'It checks this Spark first, one line each: the GPU, the driver, that the GPU is free, the disk.' \
        '--launch starts the engine and opens chat after installation; it needs the model.' \
        '--source always builds from source.' \
        '--check prints what the install would do, writes nothing, and exits 0.' \
        '--json prints JSON lines on standard output and every other line on standard error. The last' \
        'line is the result: {"schema":"vade.install.v1","ok":true or false,"error":...,"next":...}.' \
        'The exit code is 0 only when "ok" is true, or with --check.' \
        '--uninstall stops the engine that vade started, then removes the programs, builds, caches and' \
        'state in the Vade home (default ~/.vade) and the script in ~/.local/bin. Your own chat files' \
        '(sessions, instructions, memory, tool settings) and the controller config stay. With --all they' \
        'go too, and only the model files and the Rust toolchains are left. With --check it lists what' \
        'it would remove and removes nothing. Run it on each Spark.' \
        'Needs an idle NVIDIA GB10 GPU with a driver that supports CUDA 13.0 or newer.' \
        'VADE_HOME (default ~/.vade) must be a full path with no link in it. One install or uninstall' \
        'runs at a time on it. With --json a signal ends the run with exit 129, 130 or 143 and a result line.' \
        'VADE_DL_BASE sets the download location (default https://dl.vade.lol/vade); VADE_CHANNEL' \
        'sets the channel (default main).' \
        'CARGO_NET_OFFLINE=true requires already cached toolchains and build-std.'
    exit 0
fi
clone_url=https://github.com/wappy-x/vade.git
launch=false source=false check=false json=false all=false action=install retry= root= piped=false started=
for argument in "$@"; do [ "$argument" != --json ] || json=true; done
if [ "$json" = true ]; then exec 5>&1 1>&2; fi
trap on_exit EXIT
trap 'on_signal 129 SIGHUP' HUP
trap 'on_signal 130 SIGINT' INT
trap 'on_signal 143 SIGTERM' TERM
if command -v date >/dev/null 2>&1; then started=$(date +%s) || started=; fi
case $started in ''|*[!0-9]*) started= ;; esac
for argument in "$@"; do
    case $argument in
        --launch) launch=true retry="$retry --launch" ;;
        --source) source=true retry="$retry --source" ;;
        --check) check=true ;;
        --json) retry="$retry --json" ;;
        --uninstall) action=uninstall ;;
        --all) all=true ;;
        *) check=false; fault=bad_arguments fail 'unknown arguments. Next: run the installer with --help.' ;;
    esac
done
[ "$json" = false ] || [ "$launch" = false ] ||
    fault=bad_arguments fail '--json and --launch do not go together: chat needs a terminal. Next: run the installer with --json, then vade up.'
[ "$action" = install ] || { [ "$launch" = false ] && [ "$source" = false ]; } ||
    fault=bad_arguments fail '--uninstall takes only --all, --check and --json. Next: run the installer with --help.'
[ "$action" = uninstall ] || [ "$all" = false ] ||
    fault=bad_arguments fail '--all goes with --uninstall. Next: run the installer with --help.'
[ -n "${HOME-}" ] || fault=no_home fail 'HOME is missing. Next: run the installer from your normal login shell on the Spark.'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] ||
    fault=not_a_spark fail 'run this on a Linux aarch64 DGX Spark. Next: open a terminal on your Spark and retry.'
# The Vade home. Set but empty is refused, like any home that breaks the rules above.
if [ -n "${VADE_HOME+x}" ]; then vade_home=$VADE_HOME; else vade_home=$HOME/.vade; fi
plain_home "$vade_home" && no_links "$vade_home" ||
    fault=bad_home fail "VADE_HOME ($vade_home) must be a full path of two or more parts, with no empty, . or .. part and no link in it, and not your home folder. Next: unset VADE_HOME, or set it to the real folder (pwd -P prints it), then retry."
# --uninstall: stop the engine that vade started, then remove what the installer
# and vade wrote. It removes names from fixed lists only: never a model, never a
# file it does not know, and never the home folder while something is left in it.
uninstall() {
    step uninstall
    home_dir=$vade_home
    launcher="$home_dir/bin/vade"
    config_base=${XDG_CONFIG_HOME:-$HOME/.config}
    config_dir="$config_base/vade"
    # The controller config goes only with --all, and only from a plain folder.
    if [ "$all" = true ] && [ "$home_dir" = "$HOME/.vade" ] && { [ -e "$config_dir" ] || [ -L "$config_dir" ]; }; then
        plain_home "$config_base" && no_links "$config_base" ||
            fault=bad_config_home fail "XDG_CONFIG_HOME ($config_base) must be a full path of two or more parts, with no empty, . or .. part and no link in it, so nothing was removed. Next: unset XDG_CONFIG_HOME or set it to the real folder, then retry."
    fi
    # Nothing goes from a folder that is not a Vade home, also with --check.
    # A folder known only by its leftovers loses them item by item, never a whole bundles folder.
    leftover_home=false
    [ ! -e "$home_dir" ] || marked "$home_dir" || ! leftovers_only "$home_dir" || leftover_home=true
    [ ! -e "$home_dir" ] || marked "$home_dir" || [ "$leftover_home" = true ] ||
        fault=bad_home fail "$home_dir is not a Vade home: it has no .vade-home mark, no stage/install.conf from vade install, and not only what an earlier Vade attempt leaves, so nothing was removed. Next: check that VADE_HOME names the folder Vade was installed in; if it holds an earlier Vade attempt, look at what is in it and remove it by hand, then retry."
    [ "$check" = true ] || [ ! -d "$home_dir" ] || take_lock "$home_dir"
    # A home known once stays known: an older home gets its mark before stage/install.conf goes.
    [ "$check" = true ] || [ ! -d "$home_dir" ] || write_mark "$home_dir"
    path_script="$HOME/.local/bin/vade"
    # A record means vade started an engine. Its files must outlive it: vade down reads them.
    # This Spark keeps state/run.conf; a controller keeps one session.conf for each group it started.
    recorded() {
        record="$home_dir/state/run.conf"
        [ ! -f "$record" ] || return 0
        command -v find >/dev/null 2>&1 || return 1
        record=$(for folder in "$home_dir/state" "$home_dir/controller"; do
            [ ! -d "$folder" ] || find "$folder" -type f \( -name run.conf -o -name session.conf \) 2>/dev/null
        done | head -n 1)
        [ -n "$record" ]
    }
    if [ "$check" = false ] && [ -x "$launcher" ] && "$launcher" --version >/dev/null 2>&1 </dev/null 5>&-; then
        printf '%s\n' 'Uninstall: stopping the engine that vade started (vade down).'
        "$launcher" down </dev/null 5>&- ||
            fault=engine_running fail "vade down failed, so nothing was removed. Next: fix the cause above, run \"$launcher\" down, then retry."
    fi
    if recorded; then
        [ "$check" = true ] ||
            fault=engine_running fail "an engine that vade started is still recorded in $record, so nothing was removed. Next: stop it with the vade that started it (vade down), then retry."
        printf '%s\n' 'Check: an engine is recorded; the uninstall would run vade down first.'
    fi
    removed=0 kept=0
    remove() { # $1 = the folder it must be inside, $2 = a path from the fixed lists, $3 = file when Vade keeps a file there
        [ -e "$2" ] || [ -L "$2" ] || return 0
        if ! inside "$1" "$2"; then
            kept=$((kept + 1))
            printf 'Kept %s (a link, or not a plain path inside %s: the uninstall never removes one)\n' "$2" "$1"
            return 0
        fi
        if [ "${3-}" = file ] && [ -d "$2" ]; then
            kept=$((kept + 1))
            printf 'Kept %s (a folder where Vade keeps a file: the uninstall never removes one)\n' "$2"
            return 0
        fi
        removed=$((removed + 1))
        if [ "$check" = true ]; then printf 'Check: would remove %s\n' "$2"; return 0; fi
        # A file goes with rm -f: a folder put at its path in the meantime stays.
        if [ "${3-}" = file ]; then by_name "$2" rm -f --; else by_name "$2" rm -rf --; fi || fault=cannot_remove fail "cannot remove $2; the uninstall stopped there. Next: check its owner and permissions, then retry."
        printf 'Removed %s\n' "$2"
    }
    keep() { # $1 = a path that stays, $2 = why
        [ -e "$1" ] || return 0
        kept=$((kept + 1))
        printf 'Kept %s (%s)\n' "$1" "$2"
    }
    # In the config folder Vade writes only config (vade setup) and config.before or
    # config.before-N (an older config that setup moved aside). Those files go, then the
    # folder if nothing else is in it; a folder that keeps other files stays.
    config_name() { # $1 = a path in the config folder: true when Vade writes that name
        config_file=${1##*/}
        case $config_file in
            config|config.before) ;;
            config.before-*) case ${config_file#config.before-} in ''|*[!0-9]*) return 1 ;; esac ;;
            *) return 1 ;;
        esac
    }
    remove_config_file() { ! config_name "$1" || remove "$config_dir" "$1" file; return 0; }
    remove_config() {
        if ! inside "$config_base" "$config_dir" || [ ! -d "$config_dir" ]; then
            kept=$((kept + 1))
            printf 'Kept %s (a link, or not a plain folder inside %s: the uninstall never removes one)\n' "$config_dir" "$config_base"
            return 0
        fi
        with_patterns each_entry "$config_dir" remove_config_file || true
        if [ "$check" = true ]; then
            if with_patterns each_entry "$config_dir" config_name; then
                removed=$((removed + 1)); printf 'Check: would remove %s\n' "$config_dir"; return 0
            fi
        elif by_name "$config_dir" rmdir -- 2>/dev/null; then
            removed=$((removed + 1)); printf 'Removed %s\n' "$config_dir"; return 0
        fi
        kept=$((kept + 1))
        printf 'Kept %s (it holds files that are not from Vade)\n' "$config_dir"
    }
    # bin may hold your own programs: only Vade's script goes, then bin if it is empty.
    if [ "$leftover_home" = true ]; then
        # Each proven bundle and cache goes by itself; then bundles, only if it is empty.
        remove_bundle() { proven_bundle "$1" && remove "$home_dir" "$1"; return 0; }
        with_patterns each_entry "$home_dir/bundles" remove_bundle || true
        [ "$check" = true ] || [ ! -d "$home_dir/bundles" ] || [ -L "$home_dir/bundles" ] || by_name "$home_dir/bundles" rmdir -- 2>/dev/null || true
    else
        remove "$home_dir" "$home_dir/bundles"
    fi
    for name in bin/vade bin/vade.new stage state build-jobs controller config config.local .lock; do
        case $name in stage|state|build-jobs|controller) remove "$home_dir" "$home_dir/$name" ;; *) remove "$home_dir" "$home_dir/$name" file ;; esac
    done
    # A script write that a signal stopped leaves bin/.vade.XXXXXXXX (vade install or this script).
    if command -v ls >/dev/null 2>&1 && [ -d "$home_dir/bin" ] && [ ! -L "$home_dir/bin" ]; then
        names=$(ls -A "$home_dir/bin" 2>/dev/null) || names=
        while IFS= read -r name; do
            case $name in .vade.????????) remove "$home_dir" "$home_dir/bin/$name" file ;; esac
        done <<EOF
$names
EOF
    fi
    [ "$check" = true ] || [ ! -d "$home_dir/bin" ] || [ -L "$home_dir/bin" ] || by_name "$home_dir/bin" rmdir -- 2>/dev/null || true
    # The clones of a source build: source.XXXXXXXX folders that hold this installer.
    # The listing is read one whole line at a time: a space or a * in a name stays in it.
    if command -v ls >/dev/null 2>&1; then
        names=$(ls -A "$home_dir" 2>/dev/null) || names=
        while IFS= read -r clone; do
            case $clone in source.????????) ! proven_clone "$home_dir/$clone" || remove "$home_dir" "$home_dir/$clone" ;; esac
        done <<EOF
$names
EOF
    fi
    for name in sessions projects history commands instructions.md MEMORY.md permissions.json trust.json mcp.json hooks.conf hf-token; do
        if [ "$all" = false ]; then keep "$home_dir/$name" 'yours; --uninstall --all removes it'
        else case $name in sessions|projects|commands) remove "$home_dir" "$home_dir/$name" ;; *) remove "$home_dir" "$home_dir/$name" file ;; esac; fi
    done
    # The script in ~/.local/bin and the controller config belong to the default
    # home alone: with another VADE_HOME they serve another install and stay.
    if [ "$home_dir" = "$HOME/.vade" ]; then
        # Never a file that is not ours: the script carries the installer's mark on line 2.
        if [ -f "$path_script" ] && [ ! -L "$path_script" ] &&
            [ "$(sed -n 2p "$path_script" 2>/dev/null)" = '# Written by the Vade installer. vade-path-shim-v1' ]; then
            remove "$HOME/.local/bin" "$path_script" file
        fi
        if [ "$all" = false ]; then keep "$config_dir" 'your controller config; --uninstall --all removes it'
        elif [ -e "$config_dir" ] || [ -L "$config_dir" ]; then remove_config; fi
    fi
    drop_lock
    if [ "$check" = false ] && [ -d "$home_dir" ]; then
        # The mark goes last, and only when nothing else is left: a home that keeps your files stays a Vade home.
        ! empty_dir "$home_dir" .vade-home || [ ! -f "$home_dir/.vade-home" ] || by_name "$home_dir/.vade-home" rm -f --
        # Only an empty folder goes; rmdir refuses one that still holds a file.
        if by_name "$home_dir" rmdir -- 2>/dev/null; then printf 'Removed %s\n' "$home_dir"
        elif [ "$all" = true ]; then keep "$home_dir" 'it holds files that are not from Vade'; fi
    fi
    keep "$HOME/models" 'the model files'
    [ ! -d "$HOME/.rustup" ] || printf 'Kept %s (the Rust toolchains; remove them with: rustup self uninstall)\n' "$HOME/.rustup"
    if [ "$check" = true ]; then printf 'Check: the uninstall would remove %s of these; nothing was removed.\n' "$removed"
    elif [ "$removed" -eq 0 ]; then printf '%s\n' 'Uninstall: nothing to remove. Vade is not installed here.'
    else printf 'Uninstall done: %s removed.\n' "$removed"; fi
    if [ "$json" = true ]; then
        # The line is built first, so no signal falls between "sent" and the line itself.
        line=$(printf '{"schema":"vade.install.v1","ok":true,"error":null,"next":null,"event":"result","action":"uninstall","check":%s,"all":%s,"home":"%s","removed":%s,"seconds":%s}' \
            "$check" "$all" "$(json_text "$home_dir")" "$removed" "$(seconds)")
        trap '' HUP INT TERM; sent=true; printf '%s\n' "$line" >&5
    fi
    exit 0
}
[ "$action" = install ] || uninstall
# Refuse unusable launcher paths before any download, build or home write.
# Check again at publication, since another process can change these paths.
launcher_paths() {
    [ ! -L "$vade_home/bin" ] && { [ ! -e "$vade_home/bin" ] || [ -d "$vade_home/bin" ]; } &&
        [ ! -d "$vade_home/bin/vade" ]
}
launcher_paths ||
    fault=bad_home fail "cannot write $vade_home/bin/vade. Next: check that path and its permissions (bin must be a folder, not a link), then retry."
[ -z "${VADE_CONFIG+x}" ] ||
    fail 'VADE_CONFIG selects a controller. Next: unset VADE_CONFIG, then rerun the installer for this Spark.'
[ -z "${VADE_BUNDLE+x}" ] ||
    fail 'VADE_BUNDLE selects a bundle. Next: unset VADE_BUNDLE and rerun this script, or run vade launch from that bundle.'
workspace=$(pwd -P) || fail 'current folder is unavailable. Next: open an existing project folder and retry.'
# Piped, $0 names the shell, not this file: then no folder here is the source.
case $0 in */install.sh|install.sh)
    if [ -f "$0" ]; then
        root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || root=
        [ -z "$root" ] || checkout "$root" || root=
    fi ;;
esac
if [ -n "$root" ]; then cd "$root"; else piped=true; fi
# Keep install progress and errors. Print the chat hint after the login PATH check.
install_vade() {
    # Descriptor 3 returns the child status. Descriptor 4 keeps stdout live.
    # The child gets no descriptor 5: a program it leaves running must not hold the JSON output open.
    # Each line is passed on when it arrives. A filter program would hold the lines
    # back in blocks when the output is a file or a pipe, and a log would show no step until the end.
    install_status=$(
        {
            { result=0; "$vade" "$@" </dev/null 3>&- 4>&- 5>&- || result=$?
              printf '%s\n' "$result" >&3; } |
                while IFS= read -r line || [ -n "$line" ]; do
                    # The patterns open with "(": inside "$( )" an old bash needs the brackets to pair up.
                    case $line in ('Then start the engine and open chat: '*) ;; (*) printf '%s\n' "$line" ;; esac
                done >&4
        } 3>&1
    ) || return 1
    case $install_status in ''|*[!0-9]*) return 1 ;; esac
    return "$install_status"
} 4>&1
# The checks of this Spark, one line each, before any download or build. Every
# check runs, so one run names all that is missing; then the first failure stops
# the install. vade install checks the GPU, the driver and the disk again.
step check
problem= problem_fault=
ok() { printf '  ok    %-12s %s\n' "$1" "$2"; }
bad() { # $1 = label, $2 = fixed word for --json, $3 = "what is wrong. Next: what to do"
    printf '  FAIL  %-12s %s\n' "$1" "${3%% Next: *}"
    [ -n "$problem" ] || problem=$3 problem_fault=$2
}
printf '%s\n' 'Checks of this Spark:'
ok 'this box' 'Linux aarch64'
for tool in curl nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 || {
        bad "$tool" missing_tool "missing $tool. Next: ask your administrator to provide the prerequisites in docs/INSTALL.md#step-1-prerequisites, then retry."; }
done
gpu_processes=
if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader); then
        case " $gpu " in
            *[!A-Za-z0-9]GB10[!A-Za-z0-9]*) ok GPU "$gpu" ;;
            *) bad GPU not_a_spark "this needs an NVIDIA GB10 GPU (a DGX Spark); nvidia-smi reports: $gpu. Next: run this script on a Spark." ;;
        esac
    else
        bad GPU driver_failed 'NVIDIA driver check failed (nvidia-smi error above). Next: ask your administrator to restore nvidia-smi, then retry.'
    fi
    # Older drivers print "CUDA Version:"; newer drivers print "CUDA UMD Version:".
    cuda=$(nvidia-smi | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\)\.[0-9].*/\2/p' | head -n 1)
    if [ -n "$cuda" ] && [ "$cuda" -ge 13 ]; then ok driver "CUDA $cuda"
    else bad driver driver_too_old 'the NVIDIA driver must support CUDA 13.0 or newer (the "CUDA Version" or "CUDA UMD Version" that nvidia-smi prints). Next: ask your administrator to update the driver, then retry.'; fi
    if gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); then
        if [ -z "$gpu_processes" ]; then ok 'GPU is free' 'no compute process'
        elif [ "$check" = true ]; then printf '  FAIL  %-12s %s\n' 'GPU is free' 'the GPU is in use.'
        else bad 'GPU is free' gpu_in_use 'the GPU is in use. Next: stop that workload through the launcher that started it, then retry.'; fi
    else
        bad 'GPU is free' driver_failed 'NVIDIA driver check failed (nvidia-smi error above). Next: ask your administrator to restore nvidia-smi, then retry.'
    fi
fi
# The disk that will hold the Vade home. The check asks what vade install asks
# (cli/src/launch.rs INSTALL_SPACE, cli/src/detect.rs check_resources): 8 GiB for a
# first install and nothing for an update (stage/install.conf is there), plus 1 GiB
# when the model folder (VADE_MODELS, else ~/models) is on the same disk.
disk_of() { # $1 = a full path: sets disk_at, disk_kb and disk_mount for the first folder that exists at or above it
    disk_at=$1 disk_kb= disk_mount=
    case $disk_at in /*) ;; *) return 1 ;; esac
    while [ ! -d "$disk_at" ] && [ "$disk_at" != / ]; do disk_at=$(dirname -- "$disk_at"); done
    command -v df >/dev/null 2>&1 || return 1
    disk_line=$(df -Pk "$disk_at" 2>/dev/null | tail -n 1)
    disk_kb=$(printf '%s\n' "$disk_line" | sed -n 's/^[^ ]* *[0-9][0-9]* *[0-9][0-9]* *\([0-9][0-9]*\) .*/\1/p')
    disk_mount=$(printf '%s\n' "$disk_line" | sed -n 's/^.*% \(.*\)$/\1/p')
}
disk_of "$vade_home" || true
place=$disk_at free_kb=$disk_kb home_mount=$disk_mount
models_dir=${VADE_MODELS:-$HOME/models}
# A model folder whose disk cannot be read counts as the same disk: the larger need.
shared=true
if [ -n "$home_mount" ] && disk_of "$models_dir" && [ -n "$disk_mount" ] && [ "$disk_mount" != "$home_mount" ]; then shared=false; fi
if [ -f "$vade_home/stage/install.conf" ]; then need_what=update need_kb=0; else need_what=install need_kb=8388608; fi
[ "$shared" = false ] || need_kb=$((need_kb + 1048576))
need_text="the $need_what needs $((need_kb / 1048576)) GiB"
[ "$shared" = false ] || need_text="$need_text with the model on the same disk"
case $free_kb in
    ''|*[!0-9]*) printf '  skip  %-12s %s\n' disk 'not read here; vade install checks it' ;;
    *) free_gib="$((free_kb / 1048576)).$((free_kb % 1048576 * 10 / 1048576))"
       if [ "$free_kb" -ge "$need_kb" ]; then ok disk "$free_gib GiB free in $place ($need_text)"
       else bad disk no_disk_space "the disk has $free_gib GiB free in $place, and $need_text (the model needs 103 GB more). Next: free space there, or set VADE_HOME to a folder on a larger disk, then retry."; fi ;;
esac
# A build from source that was asked for: name its tools now, not after the download.
if [ "$source" = true ]; then
    if command -v cc >/dev/null 2>&1; then ok 'C compiler' 'cc'
    else bad 'C compiler' missing_tool 'missing cc, which a build from source needs. Next: ask your administrator to provide the prerequisites in docs/INSTALL.md#step-1-prerequisites, then retry.'; fi
    if [ "$piped" = true ] && ! command -v git >/dev/null 2>&1; then
        bad git missing_tool 'missing git, which the source build needs to get the Vade source. Next: ask your administrator to install git, then retry.'
    fi
    if command -v rustup >/dev/null 2>&1 || [ -x "$HOME/.cargo/bin/rustup" ]; then ok Rust 'rustup found; missing pinned toolchains are added'
    elif [ "${CARGO_NET_OFFLINE-false}" = true ]; then bad Rust missing_tool 'missing rustup. Next: rerun this script online without CARGO_NET_OFFLINE=true to install the pinned Rust tools.'
    else ok Rust 'not here yet; the installer downloads the pinned toolchains'; fi
fi
[ -z "$problem" ] || fault=$problem_fault fail "$problem"
[ -z "$gpu_processes" ] || printf '%s\n' 'Check: the GPU is in use; the install would stop until that workload stops.'

# The result line of --check. $1 = what the install would do.
checked() {
    [ "$json" = true ] || return 0
    if [ -n "$gpu_processes" ]; then
        fault=gpu_in_use failed 'the GPU is in use. Next: stop that workload through the launcher that started it, then retry.'
    else
        line=$(printf '{"schema":"vade.install.v1","ok":true,"error":null,"next":null,"event":"result","action":"install","check":true,"would":"%s","seconds":%s}' \
            "$(json_text "$1")" "$(seconds)")
        trap '' HUP INT TERM; sent=true; printf '%s\n' "$line" >&5
    fi
}
# The prebuilt bundle. Each check below runs before the next: the local tools, OS,
# glibc and driver are read first, then latest.json is downloaded and the local
# glibc and driver are compared with it, then the bundle is downloaded and checked.
# A bad archive hash stops the install. Other failures can build from source.
prebuilt_dir= why= keep_partial=false plan=
clean_download() {
    ticker_stop
    if [ -n "$partial" ]; then
        if [ "$keep_partial" = true ]; then
            by_name "$partial/lock" rm -rf --
        else
            # Move the locked cache into the download folder before removal. A new
            # install can then make its own cache without losing files to this cleanup.
            # From inside the checked download folder, .. is its physical parent.
            in_dir "$download" mv -- "../${partial##*/}" ./partial 2>/dev/null || return 1
        fi
    fi
    [ -z "$download" ] || by_name "$download" rm -rf --
    download= partial=
}
version_at_least() { # $1 >= $2, dotted decimal numbers
    set -- "$1." "$2."
    while [ -n "$1$2" ]; do
        a=${1%%.*} b=${2%%.*}
        case "${a:-0}${b:-0}" in *[!0-9]*) return 1 ;; esac
        [ "${a:-0}" -eq "${b:-0}" ] || { [ "${a:-0}" -gt "${b:-0}" ]; return; }
        set -- "${1#*.}" "${2#*.}"
    done
}
field() { printf '%s\n' "$latest" | tr ',{}' '\n\n\n' | sed -n "s/^[[:space:]]*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\"[[:space:]]*\$/\1/p" | head -n 1; }
bundle_matches() { # $1 = unpacked bundle folder, $2 = sha40
    [ -f "$1/MANIFEST" ] && [ ! -L "$1/MANIFEST" ] && grep -qx "sha=$2" "$1/MANIFEST" &&
        (cd "$1" && sed -n 's/^file=//p' MANIFEST | sha256sum -c --quiet --strict - >/dev/null 2>&1)
}
outside() { # true when a member line on stdin is not $1 or under $1/
    while IFS= read -r member; do
        case $member in "$1"|"$1"/*) ;; *) return 0 ;; esac
    done
    return 1
}
prebuilt() {
    [ "$source" = false ] || { why='--source was given'; return 1; }
    for tool in tar gzip sha256sum getconf grep tr cut sort mv mkdir rm wc; do
        command -v "$tool" >/dev/null 2>&1 || { why="missing $tool"; return 1; }
    done
    # Tests point VADE_TEST_OS_RELEASE at a fixture file; a Spark reads /etc/os-release.
    os_release=${VADE_TEST_OS_RELEASE:-/etc/os-release}
    os=$( (. "$os_release" && printf '%s %s' "${ID-}" "${VERSION_ID-}") 2>/dev/null) || os=
    case $os in "ubuntu "*) version_at_least "${os#ubuntu }" 24.04 ||
        { why="Ubuntu ${os#ubuntu } is older than 24.04"; return 1; } ;;
        *) why="the bundle is made for Ubuntu 24.04 or newer, and this system is '${os:-unknown}'"; return 1 ;; esac
    glibc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed -n 's/^glibc \([0-9][0-9.]*\)$/\1/p')
    [ -n "$glibc" ] || { why='cannot read the glibc version'; return 1; }
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    [ -n "$driver" ] || { why='cannot read the NVIDIA driver version'; return 1; }
    base=${VADE_DL_BASE:-https://dl.vade.lol/vade}
    channel=${VADE_CHANNEL:-main}
    case $base in https://*) protocol='=https' ;; file:///*) protocol='=file' ;;
        *) why="VADE_DL_BASE must start with https:// (or file:/// for a local copy)"; return 1 ;; esac
    case $channel in ''|*[!A-Za-z0-9._-]*|.*) why='VADE_CHANNEL may hold only letters, digits, dot, dash and underscore'; return 1 ;; esac
    url="$base/$channel/latest.json"
    latest=$(curl -q --proto "$protocol" --tlsv1.2 --fail --location --silent --show-error \
        --connect-timeout 30 --max-time 30 --max-filesize 65536 "$url" 2>&1 5>&-) ||
        { why="cannot download $url ($(printf '%s' "$latest" | head -n 1))"; return 1; }
    sha=$(field sha) file=$(field file) sum=$(field sha256) need_driver=$(field min_driver) need_glibc=$(field glibc)
    short=$(printf '%.7s' "$sha")
    case $sha in *[!0-9a-f]*|'') why="$url has no valid sha"; return 1 ;; esac
    case $sum in *[!0-9a-f]*|'') why="$url has no valid sha256"; return 1 ;; esac
    [ "${#sha}" -eq 40 ] && [ "${#sum}" -eq 64 ] || { why="$url has an invalid sha or sha256"; return 1; }
    case $file in vade-*-"$short"-linux-aarch64-gb10.tar.gz) ;; *) why="$url names an unexpected file"; return 1 ;; esac
    case $file in */*|*..*) why="$url names an unexpected file"; return 1 ;; esac
    version_at_least "$glibc" "${need_glibc:-x}" ||
        { why="the bundle needs glibc ${need_glibc:-?} and this system has $glibc"; return 1; }
    version_at_least "$driver" "${need_driver:-x}" ||
        { why="the bundle needs NVIDIA driver ${need_driver:-?} or newer and this Spark has $driver"; return 1; }
    stem=${file%.tar.gz}
    prebuilt_dir="$vade_home/bundles/$stem"
    if [ -e "$prebuilt_dir" ] || [ -L "$prebuilt_dir" ]; then
        bundle_matches "$prebuilt_dir" "$sha" ||
            { why="$prebuilt_dir exists and does not match its MANIFEST; move it away to download again"; return 1; }
        plan="use $prebuilt_dir (Vade $sha, already downloaded and checked), then run vade install."
        [ "$check" = false ] || printf 'Check: would %s\n' "$plan"
        return 0
    fi
    if [ "$check" = true ]; then
        plan="download $base/$channel/$file (sha256 $sum), unpack it to $prebuilt_dir, then run vade install."
        printf 'Check: would %s\n' "$plan"
        return 0
    fi
    step download
    printf 'Install: downloading the prebuilt bundle %s.\n' "$file"
    # Private like the folders vade makes: vade refuses a ~/.vade that others can read.
    # Downloads go only into a plain bundles folder in the home, never through a link.
    inside "$vade_home" "$vade_home/bundles" || { why="$vade_home/bundles is a link or not a plain folder"; return 1; }
    (umask 077 && by_name "$vade_home/bundles" mkdir -p --) || { why="cannot create $vade_home/bundles"; return 1; }
    new_name=$(in_dir "$vade_home/bundles" mktemp -d ./.download.XXXXXXXX) || { why="cannot create a folder in $vade_home/bundles"; return 1; }
    download="$vade_home/bundles/${new_name#./}"
    # The expected hash names the partial file. A new release cannot resume it.
    cache="$vade_home/bundles/.partial-$sum"
    if [ -L "$cache" ] || { [ -e "$cache" ] && [ ! -d "$cache" ]; }; then
        why='the bundle download cache is not a regular folder'; return 1
    fi
    (umask 077 && by_name "$cache" mkdir -p -- && in_dir "$cache" mkdir ./lock) 2>/dev/null ||
        { why="cannot lock the bundle download cache; if no other installer is running, remove $cache/lock and retry"; return 1; }
    partial=$cache keep_partial=true
    archive="$partial/bundle.part"
    if [ -L "$archive" ] || { [ -e "$archive" ] && [ ! -f "$archive" ]; }; then
        why='the partial bundle is not a regular file'; return 1
    fi
    [ ! -f "$archive" ] || [ "$(wc -c < "$archive")" -le 536870912 ] || by_name "$archive" rm -f --
    attempt=0
    # Keep a complete checked archive if an earlier run stopped before unpacking.
    if [ ! -f "$archive" ] || [ "$(sha256sum "$archive" | cut -d ' ' -f 1)" != "$sum" ]; then
        while :; do
            attempt=$((attempt + 1))
            code=0
            ticker "downloading $file"
            status=$(in_dir "$partial" curl -q --proto "$protocol" --tlsv1.2 --fail --location --silent --show-error \
                --connect-timeout 30 --speed-limit 1024 --speed-time 60 --max-filesize 536870912 \
                --continue-at - --write-out '%{http_code}' --output ./bundle.part "$base/$channel/$file" \
                2>/dev/null 5>&-) || code=$?
            ticker_stop
            if [ -f "$archive" ] && [ "$(wc -c < "$archive")" -gt 536870912 ]; then
                keep_partial=false; why='the bundle download exceeds the size limit'; return 1
            fi
            case "$code:$status" in
                33:*|*:416) by_name "$archive" rm -f -- ;;
                0:*) break ;;
                5:*|6:*|7:*|16:*|18:*|28:*|35:*|52:*|55:*|56:*|92:*|*:408|*:429|*:5??) ;;
                *) keep_partial=false; why="download of $base/$channel/$file failed"; return 1 ;;
            esac
            [ "$attempt" -lt 3 ] || { why="download of $base/$channel/$file failed; partial bytes were kept for retry"; return 1; }
        done
    fi
    keep_partial=false
    if [ "$(sha256sum "$archive" | cut -d ' ' -f 1)" != "$sum" ]; then
        printf 'install: %s does not match the sha256 in latest.json; nothing installed.\n' "$file" >&2
        fault=bad_download failed "$file does not match the sha256 in latest.json; nothing installed. Next: run the same install command again."
        exit 1
    fi
    # The checked archive stays in the cache and is read there. Each step that writes
    # runs inside its own checked folder: the unpack folder, then a move up from it.
    members=$(in_dir "$partial" gzip -dc ./bundle.part | tar -tf -) || { why="$file is not a readable archive"; return 1; }
    if printf '%s\n' "$members" | outside "$stem" ||
        printf '%s\n' "$members" | grep -e '\.\./' -e '/\.\.$' >/dev/null; then
        why="$file has files outside $stem/"; return 1
    fi
    # Only regular files and folders: no links, devices or other special members.
    types=$(in_dir "$partial" gzip -dc ./bundle.part | tar -tvf - | cut -c 1 | sort -u | tr -d '\n') ||
        { why="$file is not a readable archive"; return 1; }
    case $types in -|d-|-d) ;; *) why="$file has members that are not files or folders"; return 1 ;; esac
    in_dir "$download" mkdir ./unpack && in_dir "$partial" gzip -dc ./bundle.part | in_dir "$download/unpack" tar -xf - ||
        { why="cannot unpack $file"; return 1; }
    bundle_matches "$download/unpack/$stem" "$sha" || { why="$file does not match its MANIFEST"; return 1; }
    # .. is the physical parent of the folder in_dir checked, so ../.. is bundles itself.
    in_dir "$download/unpack" mv -- "./$stem" "../../$stem" || { why="cannot move the bundle to $prebuilt_dir"; return 1; }
    clean_download
    printf 'Install: prebuilt Vade %s is in %s.\n' "$sha" "$prebuilt_dir"
}
route=source
# From here the install writes in the Vade home, so it holds the lock (cleanup drops it).
# --check says it too when the folder is someone else's.
[ "$check" = false ] || [ ! -d "$vade_home" ] || vade_home "$vade_home" || empty_dir "$vade_home" .install-lock ||
    fault=bad_home fail "$vade_home already holds files and is not a Vade home (no .vade-home mark), so nothing was written. Next: look at what is in it; if it is only an earlier Vade attempt, remove it, else choose a new or empty folder for VADE_HOME; then retry."
[ "$check" = true ] || take_lock "$vade_home" install
if prebuilt; then
    [ "$check" = false ] || { checked "$plan"; exit 0; }
    vade="$prebuilt_dir/vade"
    step install
    if install_vade install; then
        route=prebuilt
        printf 'Install: installed prebuilt Vade %s.\n' "$sha"
    else
        # With a controller config that install reaches the other Sparks. A source
        # build is for this Spark alone, so it would end with rc 0 and hide the fault.
        [ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/vade/config" ] ||
            fault=install_failed fail "\"$vade\" install failed, and your controller config selects these Sparks, so a source build here does not help. Next: follow the diagnostic above, run \"$vade\" doctor, then retry."
        why="\"$vade\" install failed (diagnostic above)"
        vade=
    fi
else
    vade=
    clean_download
fi
# With no checkout: clone public main into a new folder, then run the installer
# of the clone, which holds the pinned tools of that source. The folder is in
# the Vade home, not in /tmp: the vade program stays in the clone, and vade
# install and vade setup read the clone, so it must outlive a reboot.
clone_source() {
    if [ "$check" = true ]; then
        plan="clone $clone_url into a new folder in $vade_home, then build from source there."
        printf 'Check: would %s\n' "$plan"
        checked "$plan"
        exit 0
    fi
    command -v git >/dev/null 2>&1 ||
        fault=missing_tool fail 'missing git, which the source build needs to get the Vade source. Next: ask your administrator to install git, then retry.'
    # Private like the folders vade makes: vade refuses a ~/.vade that others can read.
    (umask 077 && by_name "$vade_home" mkdir -p --) ||
        fail "cannot create $vade_home. Next: check that you own its parent folder, then retry."
    new_name=$(in_dir "$vade_home" mktemp -d ./source.XXXXXXXX) ||
        fail "cannot create a folder in $vade_home. Next: check its permissions and disk space, then retry."
    clone="$vade_home/${new_name#./}"
    clone_tmp=$clone
    step clone
    printf 'Install: cloning %s into %s.\n' "$clone_url" "$clone"
    # A transfer under 1 KB/s for 60 s stops; git never asks for a password.
    # git clones into the empty folder it runs in: in_dir refuses a swapped one.
    GIT_TERMINAL_PROMPT=0 in_dir "$clone" git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 \
        clone --quiet --depth 1 "$clone_url" . </dev/null 5>&- ||
        fault=clone_failed fail "cannot clone $clone_url (git error above). Next: check that this Spark reaches github.com, then retry."
    checkout "$clone" ||
        fail "the clone of $clone_url is not a Vade checkout. Next: retry; if it fails again, report it with the Vade version."
    # The clone stays: its installer runs next and takes the lock itself.
    clone_tmp=
    drop_lock
    trap - EXIT HUP INT TERM
    # Standard input is this script, so the clone's installer reads the terminal
    # or nothing. A build failure there keeps the clone and prints its own Retry line.
    if [ "$launch" = true ] && (: </dev/tty) 2>/dev/null; then
        exec sh "$clone/scripts/install.sh" --source --launch </dev/tty
    fi
    # With --json the clone's installer writes the JSON lines: it is the same public main as this script.
    [ "$json" = false ] || exec sh "$clone/scripts/install.sh" --source --json </dev/null 1>&5 5>&-
    exec sh "$clone/scripts/install.sh" --source </dev/null
}
# The source build below is the same as before the prebuilt bundle existed.
if [ -z "$vade" ]; then
printf 'Prebuilt: %s; building from source.\n' "$why"
# Only a build needs a compiler, so a bundle install runs without one. This comes
# before the clone, so a Spark with no compiler downloads nothing.
command -v cc >/dev/null 2>&1 ||
    fault=missing_tool fail 'missing cc, which a build from source needs. Next: ask your administrator to provide the prerequisites in docs/INSTALL.md#step-1-prerequisites, then retry.'
[ "$piped" = false ] || clone_source
if [ "$check" = true ]; then
    printf '%s\n' "Check: would install the pinned Rust toolchains if missing, build vade in $root/cli/target," \
        'fetch the Rust build-std files, then run vade install to build the GPU kernels, the engine and chat.'
    checked "install the pinned Rust toolchains if missing, build vade in $root/cli/target, fetch the Rust build-std files, then run vade install to build the GPU kernels, the engine and chat."
    exit 0
fi
export PATH="$HOME/.cargo/bin:$PATH"
export VADE_SOURCE="$root"
# Install only absent components: a cached setup makes no toolchain downloads.
online() {
    [ "${CARGO_NET_OFFLINE-false}" != true ] ||
        fail "missing $*. Next: rerun this script online without CARGO_NET_OFFLINE=true to install the pinned Rust tools."
}
if ! command -v rustup >/dev/null 2>&1; then
    online rustup
    installer=$(mktemp "${TMPDIR:-/tmp}/vade-rustup.XXXXXXXX") ||
        fail 'cannot create a temporary installer. Next: check TMPDIR permissions and disk space, then retry.'
    installer_tmp=$installer
    printf '%s\n' 'Install: downloading the official Rust installer (no shell profile changes).'
    curl -q --proto '=https' --tlsv1.2 --fail --location --show-error --silent \
        --output "$installer" https://sh.rustup.rs 5>&- ||
        fail 'Rust installer download failed. Next: check access to https://sh.rustup.rs and rerun the Retry command below.'
    sh "$installer" -y --default-toolchain none --no-modify-path 5>&- ||
        fail 'Rust installation failed. Next: read the Rust error above, fix it, then rerun the Retry command below.'
    rm -f -- "$installer"
    installer_tmp=
fi
host=1.99.0
device=nightly-2026-10-01
if ! rustup run "$host" rustc --version >/dev/null 2>&1; then
    online "Rust $host"
    rustup toolchain install "$host" --profile minimal --no-self-update 5>&- ||
        fail "host compiler setup failed. Next: rustup toolchain install $host --profile minimal --no-self-update; then rerun this script."
fi
components=$(rustup component list --installed --toolchain "$device" 2>/dev/null || true)
complete=true
for component in rust-src llvm-tools llvm-bitcode-linker rust-std-nvptx64-nvidia-cuda; do
    case "$components" in *"$component"*) ;; *) complete=false ;; esac
done
if ! rustup run "$device" rustc --version >/dev/null 2>&1 || [ "$complete" = false ]; then
    online "Rust $device with GPU components"
    rustup toolchain install "$device" --profile minimal --no-self-update \
        --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda 5>&- ||
        fail 'GPU compiler setup failed. Next: finish the pinned GPU toolchain command in docs/INSTALL.md#install-rust-source-builds-only, then rerun this script.'
fi
step build
printf '%s\n' 'Install: building the vade command.'
CARGO_TARGET_DIR="$root/cli/target" cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml 5>&- ||
    fault=build_failed fail 'CLI build failed. Next: fix the compiler error above (including any missing linker), then rerun the Retry command below.'
vade="$root/cli/target/release/vade"
# vade install checks this Spark and builds the GPU kernels once, offline.
# It needs no model on disk; vade launch saves defaults once the model is there.
# That build needs the Rust build-std files; fetch only those here.
sysroot=$(rustup run "$device" rustc --print sysroot) && [ -n "$sysroot" ] ||
    fail "cannot find the $device sysroot. Next: rustup toolchain install $device --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda; then rerun this script."
printf '%s\n' 'Install: fetching the Rust build-std files for the GPU kernels (none when cached).'
cargo "+$device" fetch --locked --manifest-path "$sysroot/lib/rustlib/src/rust/library/Cargo.toml" 5>&- ||
    fail 'Rust build-std fetch failed. Next: check access to https://index.crates.io and rerun the Retry command below; with CARGO_NET_OFFLINE=true, run it once online first.'
# A source install is for this Spark alone: a controller config does not change that.
step install
install_vade __install-here ||
    fault=install_failed fail "local install failed. Next: follow the diagnostic above, run \"$vade\" doctor, then retry."
fi
# One folder stays on PATH across versions: a small script that runs the build
# just installed, rewritten on each install. Not a copy or a link: the real
# program must run from its own folder to find its bundle or checkout.
bin_path=
if [ "$vade_home" != "$HOME/.vade" ]; then bin_path="$vade_home/bin"; fi
quoted=$(printf '%s\n' "$vade" | sed "s/'/'\\\\''/g")
# A script goes to a new temporary name that mktemp makes in its folder, then is renamed
# over the old one: never written through a link, never over a file that was there.
# Each step runs from inside the checked folder (in_dir) on relative names.
fill_script() { # in the folder: $1 = the temporary name, $2 = the script's name, $3 = its text
    # mktemp made an empty plain file: refuse anything else put at that name since.
    [ -f "./$1" ] && [ ! -L "./$1" ] && [ ! -s "./$1" ] || return 1
    printf '%s\n' "$3" > "./$1" && chmod 700 "./$1" && mv -f "./$1" "./$2"
}
write_script() { # $1 = a folder by its physical path, $2 = the script's name, $3 = its text
    new_name=$(in_dir "$1" mktemp ./.vade.XXXXXXXX) || return 1
    case ${new_name#./} in .vade.*) script_tmp="$1/${new_name#./}" ;; *) return 1 ;; esac
    if in_dir "$1" fill_script "${script_tmp##*/}" "$2" "$3"; then script_tmp=; return 0; fi
    by_name "$script_tmp" rm -f --; script_tmp=; return 1
}
launcher_paths &&
    { [ -d "$vade_home/bin" ] || (umask 077 && by_name "$vade_home/bin" mkdir --); } &&
    write_script "$vade_home/bin" vade "$(printf '#!/bin/sh\nexec '\''%s'\'' "$@"' "$quoted")" ||
    fail "cannot write $vade_home/bin/vade. Next: check that path and its permissions (bin must be a folder, not a link), then retry."
# Older bundles: this one and the one before it stay. The script above runs the new bundle now, so no older one is the install.
# Only an unpacked bundle goes: a real folder with our name pattern whose MANIFEST starts
# with format=vade-release-v1 and names that folder (proven_bundle above).
# The listing is read one whole line at a time, and each removal follows the rules at the top.
bundle_dir() { # true when $1 is a name in bundles/ that is an older unpacked bundle
    case $1 in vade-*-linux-aarch64-gb10) ;; *) return 1 ;; esac
    [ "$vade_home/bundles/$1" != "$prebuilt_dir" ] && proven_bundle "$vade_home/bundles/$1"
}
if [ "$route" = prebuilt ] && command -v ls >/dev/null 2>&1; then
    if inside "$vade_home" "$vade_home/bundles"; then
        names=$(ls -A "$vade_home/bundles" 2>/dev/null) || names=
        newest=
        while IFS= read -r name; do
            bundle_dir "$name" || continue
            [ -z "$newest" ] || [ "$vade_home/bundles/$name" -nt "$vade_home/bundles/$newest" ] || continue
            newest=$name
        done <<EOF
$names
EOF
        while IFS= read -r name; do
            bundle_dir "$name" && [ "$name" != "$newest" ] && inside "$vade_home" "$vade_home/bundles/$name" || continue
            by_name "$vade_home/bundles/$name" rm -rf -- 2>/dev/null && printf 'Install: removed the older bundle %s.\n' "$name"
        done <<EOF
$names
EOF
    else
        printf 'Install: kept the older bundles: %s/bundles is a link or not a plain folder.\n' "$vade_home"
    fi
fi
# A second script for new shells: the stock Ubuntu ~/.profile puts ~/.local/bin on
# PATH at login when the folder exists. No shell profile changes, no link, and
# never over a file that is not ours. It names ~/.vade, so a named VADE_HOME skips it.
shim="$HOME/.local/bin/vade"
shim_marker='# Written by the Vade installer. vade-path-shim-v1'
shim_written=false
if [ -z "${VADE_HOME-}" ] || [ "$VADE_HOME" = "$HOME/.vade" ]; then
    if { [ ! -e "$shim" ] && [ ! -L "$shim" ]; } ||
        { [ -f "$shim" ] && [ ! -L "$shim" ] && [ "$(sed -n 2p "$shim" 2>/dev/null)" = "$shim_marker" ]; }; then
        if [ -L "$HOME/.local" ] || [ -L "$HOME/.local/bin" ]; then
            printf 'Skipped %s: %s/.local or %s/.local/bin is a link, and the installer never writes through one.\n' "$shim" "$HOME" "$HOME"
        # The write runs inside the folder by its physical path: a link put at .local or
        # .local/bin after this check makes in_dir refuse it.
        elif { [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin"; } 2>/dev/null &&
            shim_home=$(real_dir "$HOME") && [ ! -L "$HOME/.local" ] && [ ! -L "$HOME/.local/bin" ] &&
            write_script "${shim_home%/}/.local/bin" vade "$(printf '#!/bin/sh\n%s\nexec "$HOME/.vade/bin/vade" "$@"' "$shim_marker")" 2>/dev/null; then
            shim_written=true
        else
            printf 'Skipped %s: this account cannot write it.\n' "$shim"
        fi
    else
        printf 'Skipped %s: that file is not from Vade, so the installer left it alone.\n' "$shim"
    fi
fi
printf '%s\n' 'Install complete. Next steps:' \
    '1. If the model is not on this Spark yet, download and check it (102.26 GB):' \
    '   vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300' \
    '2. Start the engine:' \
    '   vade up' \
    ''
path_line="export PATH=\"${bin_path:-\$HOME/.vade/bin}:\$PATH\""
# Ask the login shell after writing the scripts. A failed probe keeps the export advice.
# Check the first command found, so another vade cannot hide this install.
login_vade=
if [ -n "${SHELL-}" ] && command -v timeout >/dev/null 2>&1; then
    # No descriptor 5 for the login shell: a profile may leave a program running, and it must not hold the JSON output open.
    login_reply=$(timeout 5 "$SHELL" -lc 'printf "vade-login-command=%s\n" "$(command -v vade 2>/dev/null)"' </dev/null 2>/dev/null 5>&-) || login_reply=
    login_vade=$(printf '%s\n' "$login_reply" | sed -n 's/^vade-login-command=//p' | tail -n 1)
fi
if [ "$login_vade" = "$vade_home/bin/vade" ] || { [ "$shim_written" = true ] && [ "$login_vade" = "$shim" ]; }; then
    launch_command=vade
    path_advice='Open a new terminal, then run: vade up'
else
    launch_command='~/.vade/bin/vade'
    if [ -n "$bin_path" ]; then
        quoted_bin=$(printf '%s\n' "$bin_path" | sed "s/'/'\\\\''/g")
        launch_command="'$quoted_bin/vade'"
    fi
    path_advice="Run: $path_line"
fi
printf 'Then start the engine and open chat: %s launch\n' "$launch_command"
printf '%s\n' "$path_advice"
if [ "$json" = true ]; then
    # The build is what the installed vade says it is; null when it does not say.
    build=$("$vade_home/bin/vade" version --json 2>/dev/null </dev/null 5>&- |
        sed -n 's/.*"sha":"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1) || build=
    if [ -n "$build" ]; then build="\"$build\""; else build=null; fi
    on_path=false
    [ "$launch_command" != vade ] || on_path=true
    # The line is built first, so no signal falls between "sent" and the line; after it, signals wait.
    line=$(printf '{"schema":"vade.install.v1","ok":true,"error":null,"next":"%s","event":"result","action":"install","check":false,"route":"%s","build":%s,"home":"%s","vade":"%s","path_line":"%s","on_login_path":%s,"seconds":%s}' \
        "$(json_text "$vade_home/bin/vade up")" "$route" "$build" "$(json_text "$vade_home")" \
        "$(json_text "$vade_home/bin/vade")" "$(json_text "$path_line")" "$on_path" "$(seconds)")
    trap '' HUP INT TERM; sent=true; printf '%s\n' "$line" >&5
fi
cleanup
trap - EXIT
[ "$json" = true ] || trap - HUP INT TERM
if [ "$launch" = true ]; then
    cd "$workspace" || fail 'your starting folder is unavailable. Next: open an existing project folder, then use the launch command above.'
    [ "$piped" = false ] || (: </dev/tty) 2>/dev/null || {
        printf '%s\n' 'No terminal to open chat in, so the installer stops here. Run the commands above in a terminal.'
        exit 0
    }
    # Piped, standard input is this script, so chat reads the terminal.
    [ "$piped" = false ] || exec "$vade" launch </dev/tty
    exec "$vade" launch
fi
}
main "$@"
