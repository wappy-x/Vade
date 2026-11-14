#!/bin/sh
# Test the installer with fake host tools. No GPU or network is used.
# Run from the repository root: sh scripts/test-install.sh
set -eu
PATH=/usr/bin:/bin
export PATH
unset CDPATH
umask 077
root=$(cd -P "$(dirname "$0")/.." && pwd)
matrix=$(mktemp -d "${TMPDIR:-/tmp}/vade-install-matrix.XXXXXXXX")
trap 'rm -rf "$matrix"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
matrix=$(cd -P "$matrix" && pwd)

shells='sh bash'
for tool in dash zsh; do
    if command -v "$tool" >/dev/null 2>&1; then shells="$shells $tool"; fi
done
if [ "$shells" = 'sh bash' ]; then
    echo 'FAIL setup: missing dash or zsh'
    exit 1
fi
for tool in sh bash timeout; do
    command -v "$tool" >/dev/null 2>&1 || {
        printf 'FAIL setup: missing %s\n' "$tool"
        exit 1
    }
done
system_timeout=$(command -v timeout)

# A file in PATH is either a local fake or a known system file tool.
# In particular, no real curl, git, compiler, Rust tool or GPU tool is in PATH.
# Leave tar out so the default install selects source before any download.
fixture() {
    mkdir -p "$case_dir/bin" "$case_dir/home" "$case_dir/prefix" "$case_dir/tmp" \
        "$case_dir/work" "$case_dir/checkout/scripts" "$case_dir/checkout/cli" \
        "$case_dir/checkout/engine" "$case_dir/checkout/harness"
    cp "$root/scripts/install.sh" "$case_dir/checkout/scripts/install.sh"
    for crate in cli engine harness; do : > "$case_dir/checkout/$crate/Cargo.toml"; done
    for profile in .profile .bashrc .zshrc; do
        printf '%s\n' 'Keep this profile.' > "$case_dir/home/$profile"
    done
    printf '%s\n' 'Keep this prefix.' > "$case_dir/prefix/keep"
    for tool in sh dirname mktemp rm rmdir mkdir mv cp chmod tail sed head; do
        ln -s "$(command -v "$tool")" "$case_dir/bin/$tool"
    done
    cat > "$case_dir/bin/uname" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' uname >> "$MATRIX_CALLS"
case $1 in -s) echo "$MATRIX_OS" ;; -m) echo "$MATRIX_ARCH" ;; *) exit 1 ;; esac
EOF
    cat > "$case_dir/bin/nvidia-smi" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' nvidia-smi >> "$MATRIX_CALLS"
case ${1-} in
    --query-gpu=name) echo 'NVIDIA GB10' ;;
    --query-compute-apps=pid) [ -z "${MATRIX_GPU_BUSY-}" ] || echo 4242 ;;
    '') echo 'CUDA Version: 13.0' ;;
    *) exit 1 ;;
esac
EOF
    cat > "$case_dir/bin/curl" <<'EOF'
#!/bin/sh
printf '%s\n' unexpected-download >> "$MATRIX_CALLS"
exit 99
EOF
    cat > "$case_dir/bin/cc" <<'EOF'
#!/bin/sh
printf '%s\n' unexpected-compiler >> "$MATRIX_CALLS"
exit 99
EOF
    cat > "$case_dir/bin/timeout" <<'EOF'
#!/bin/sh
shift
exec "$@"
EOF
    cat > "$case_dir/bin/login-shell" <<'EOF'
#!/bin/sh
set -eu
[ "$1" = -lc ]
. "$HOME/.profile"
exec /bin/sh -c "$2"
EOF
    cat > "$case_dir/bin/git" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' 'git clone' >> "$MATRIX_CALLS"
[ "$GIT_TERMINAL_PROMPT" = 0 ]
for target do :; done
# The installer runs git inside the new folder and clones into it.
[ "$target" != . ] || target=$(pwd -P)
case $target in "$HOME"/.vade/source.*) ;; *) exit 99 ;; esac
cp -R "$MATRIX_CASE/checkout/." "$target/"
EOF
    cat > "$case_dir/bin/rustup" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' "rustup $*" >> "$MATRIX_CALLS"
case $1 in
    component) printf '%s\n' rust-src llvm-tools llvm-bitcode-linker rust-std-nvptx64-nvidia-cuda ;;
    run) [ "${4-}" != --print ] || printf '%s\n' "$MATRIX_CASE/sysroot" ;;
    *) exit 99 ;;
esac
EOF
    cat > "$case_dir/bin/cargo" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' "cargo $*" >> "$MATRIX_CALLS"
[ "$CARGO_NET_OFFLINE" = true ]
case $PWD in "$MATRIX_CASE"/*) ;; *) exit 99 ;; esac
case $2 in
    build)
        [ "$CARGO_TARGET_DIR" = "$PWD/cli/target" ]
        mkdir -p "$CARGO_TARGET_DIR/release"
        cp "$MATRIX_CASE/fake-vade" "$CARGO_TARGET_DIR/release/vade"
        ;;
    fetch) ;;
    *) exit 99 ;;
esac
EOF
    cat > "$case_dir/fake-vade" <<'EOF'
#!/bin/sh
set -eu
[ "$1" = __install-here ]
case $VADE_SOURCE in "$MATRIX_CASE"/*) ;; *) exit 99 ;; esac
printf '%s\n' "vade $*" >> "$MATRIX_CALLS"
printf 'Then start the engine and open chat: %s launch\n' "$0"
EOF
    for tool in uname nvidia-smi curl cc git rustup cargo timeout login-shell; do
        chmod 700 "$case_dir/bin/$tool"
    done
    chmod 700 "$case_dir/fake-vade"
    : > "$case_dir/calls"
}

snapshot() {
    (cd "$case_dir"
        find home prefix tmp checkout work -print
        find home prefix tmp checkout work -type f -exec cksum {} \;
    ) | LC_ALL=C sort
}

install_shell() {
    case $shell in
        sh) set -- /bin/sh "$@" ;;
        bash) set -- "$(command -v bash)" --noprofile --norc --posix "$@" ;;
        dash) set -- "$(command -v dash)" "$@" ;;
        zsh) set -- "$(command -v zsh)" --emulate sh -f "$@" ;;
    esac
    case $test_case in login-hanging) set -- "$system_timeout" 15 "$@" ;; esac
    env -i PATH="$case_dir/bin" HOME="$case_dir/home" PREFIX="$case_dir/prefix" \
        TMPDIR="$case_dir/tmp" XDG_CONFIG_HOME="$case_dir/home/.config" \
        XDG_CACHE_HOME="$case_dir/home/.cache" XDG_DATA_HOME="$case_dir/home/.local/share" \
        CARGO_HOME="$case_dir/home/.cargo" RUSTUP_HOME="$case_dir/home/.rustup" \
        CARGO_NET_OFFLINE=true MATRIX_CASE="$case_dir" MATRIX_CALLS="$case_dir/calls" \
        MATRIX_OS="$host_os" MATRIX_ARCH="$host_arch" SHELL="$login_shell" \
        MATRIX_GPU_BUSY="${gpu_busy-}" MATRIX_DF_KB="${df_kb-}" MATRIX_DF_APART="${df_apart-}" MATRIX_DOWN_FAILS="${down_fails-}" \
        ${vade_home_env:+"VADE_HOME=$vade_home_env"} \
        "$@"
}

reject_calls() {
    if grep -qE "$1" "$case_dir/calls"; then
        printf 'Unexpected tool call: %s\n' "$1"
        exit 1
    fi
}

run_case() (
    set -eu
    host_os=Linux host_arch=aarch64
    case $test_case in
        host-linux-x86_64) host_arch=x86_64 ;;
        host-macos) host_os=Darwin host_arch=arm64 ;;
    esac
    fixture
    login_shell= profile_contents='Keep this profile.'
    case $test_case in
        login-shim) profile_contents='if [ -d "$HOME/.local/bin" ]; then PATH="$HOME/.local/bin:$PATH"; fi; export PATH' ;;
        login-bin) profile_contents='PATH="$HOME/.vade/bin:$PATH"; export PATH' ;;
        login-missing) profile_contents='PATH="$MATRIX_CASE/bin"; export PATH' ;;
        login-failed) profile_contents='exit 1' ;;
        login-hanging)
            profile_contents='printf "%s\n" started > "$MATRIX_CASE/login-started"; exec /bin/sleep 30'
            rm "$case_dir/bin/timeout"
            ln -s "$system_timeout" "$case_dir/bin/timeout"
            ;;
    esac
    case $test_case in login-*)
        login_shell="$case_dir/bin/login-shell"
        printf '%s\n' "$profile_contents" > "$case_dir/home/.profile"
        ;;
    esac
    snapshot > "$case_dir/before"
    cd "$case_dir/work"
    mode=${test_case%%-*}
    action=${test_case#*-}
    set --
    case $action in
        default) ;;
        source) set -- --source ;;
        check) set -- --check ;;
        help) set -- --help ;;
    esac
    case $mode in
        checkout|login)
            # Run from a different folder to test discovery through the file path.
            install_shell "$case_dir/checkout/scripts/install.sh" "$@" \
                > "$case_dir/stdout" 2> "$case_dir/stderr"
            ;;
        pipe)
            cat "$case_dir/checkout/scripts/install.sh" | install_shell -s -- "$@" \
                > "$case_dir/stdout" 2> "$case_dir/stderr"
            ;;
        half)
            bytes=$(wc -c < "$case_dir/checkout/scripts/install.sh")
            dd if="$case_dir/checkout/scripts/install.sh" of="$case_dir/cut.sh" \
                bs=1 count="$((bytes / 2))" 2>/dev/null
            if cat "$case_dir/cut.sh" | install_shell -s \
                > "$case_dir/stdout" 2> "$case_dir/stderr"; then
                echo 'The half download must fail to parse.'
                exit 1
            fi
            [ -s "$case_dir/stderr" ]
            ;;
        no)
            # A complete function without its last call must also run nothing.
            # Save the producer status because a pipe can hide it.
            if (
                set +e
                sed '$d' "$case_dir/checkout/scripts/install.sh"
                producer_status=$?
                printf '%s\n' "$producer_status" > "$case_dir/producer-status"
                exit "$producer_status"
            ) | install_shell -s > "$case_dir/stdout" 2> "$case_dir/stderr"; then
                consumer_status=0
            else
                consumer_status=$?
            fi
            [ "$(cat "$case_dir/producer-status")" = 0 ]
            [ "$consumer_status" = 0 ]
            [ ! -s "$case_dir/stderr" ]
            ;;
        host)
            if install_shell "$case_dir/checkout/scripts/install.sh" \
                > "$case_dir/stdout" 2> "$case_dir/stderr"; then
                echo 'The installer must reject this host.'
                exit 1
            else
                [ "$?" = 1 ]
            fi
            ;;
    esac
    case $test_case in
        half-pipe|no-call)
            [ ! -s "$case_dir/calls" ]
            [ ! -s "$case_dir/stdout" ]
            ;;
        host-*)
            grep -qxF 'install: run this on a Linux aarch64 DGX Spark. Next: open a terminal on your Spark and retry.' "$case_dir/stderr"
            [ ! -s "$case_dir/stdout" ]
            printf '%s\n' uname > "$case_dir/expected-calls"
            if [ "$host_os" = Linux ]; then printf '%s\n' uname >> "$case_dir/expected-calls"; fi
            cmp "$case_dir/expected-calls" "$case_dir/calls"
            ;;
        *-help)
            [ ! -s "$case_dir/calls" ]
            grep -q '^Usage: sh scripts/install.sh' "$case_dir/stdout"
            ;;
        *-check)
            reject_calls '^(git|cargo|rustup|vade) '
            case $mode in
                checkout) grep -q '^Check: would install the pinned Rust toolchains' "$case_dir/stdout" ;;
                pipe) grep -q '^Check: would clone ' "$case_dir/stdout" ;;
            esac
            ;;
        *)
            case $action in
                default) grep -qF 'Prebuilt: missing tar; building from source.' "$case_dir/stdout" ;;
                source) grep -qF 'Prebuilt: --source was given; building from source.' "$case_dir/stdout" ;;
            esac
            case $test_case in
                login-shim|login-bin)
                    grep -qxF 'Open a new terminal, then run: vade up' "$case_dir/stdout"
                    grep -qxF 'Then start the engine and open chat: vade launch' "$case_dir/stdout"
                    [ "$(grep -c 'export PATH=' "$case_dir/stdout" || true)" = 0 ]
                    ;;
                *)
                    grep -qxF 'Run: export PATH="$HOME/.vade/bin:$PATH"' "$case_dir/stdout"
                    grep -qxF 'Then start the engine and open chat: ~/.vade/bin/vade launch' "$case_dir/stdout"
                    ;;
            esac
            [ "$(grep -cE '^(Open a new terminal|Run: export PATH=)' "$case_dir/stdout")" = 1 ]
            [ "$(grep -c '^Then start the engine and open chat:' "$case_dir/stdout")" = 1 ]
            grep -q '^cargo .* build --release --offline --locked ' "$case_dir/calls"
            grep -q '^cargo .* fetch --locked ' "$case_dir/calls"
            [ "$(tail -n 1 "$case_dir/calls")" = 'vade __install-here' ]
            grep -q '^Install complete. Next steps:' "$case_dir/stdout"
            [ -x "$case_dir/home/.vade/bin/vade" ]
            [ -x "$case_dir/home/.local/bin/vade" ]
            case $mode in
                checkout|login) reject_calls '^git clone$' ;;
                pipe) grep -q '^git clone$' "$case_dir/calls" ;;
            esac
            ;;
    esac
    reject_calls '^unexpected-'
    case $test_case in login-hanging) [ -f "$case_dir/login-started" ] ;; esac
    case $test_case in
        *-check|*-help|half-pipe|no-call|host-*)
            snapshot > "$case_dir/after"
            cmp "$case_dir/before" "$case_dir/after"
            ;;
    esac
    [ "$(cat "$case_dir/home/.profile")" = "$profile_contents" ]
    for profile in .bashrc .zshrc; do
        [ "$(cat "$case_dir/home/$profile")" = 'Keep this profile.' ]
    done
    [ "$(cat "$case_dir/prefix/keep")" = 'Keep this prefix.' ]
)

failed=0
# Every write, move and removal in the Vade home must run through in_dir or by_name
# (scripts/install.sh): from inside a checked folder, on relative names. This reads
# install.sh and lists each command line outside those two helpers where an argument
# after rm, rmdir, mv, cp, ln, chmod, mkdir, mktemp, tar, touch, curl or git (in the
# same command), or the target of a > or >> redirection, starts with a home variable.
# The first argument of in_dir and by_name is the checked anchor and does not count;
# [ ... ] tests only read. A relative name such as ./x, ../x or ../../x is fine: a
# step up from the checked folder is its physical parent. A relative path through a
# folder, such as ./x/y, counts: in_dir did not check x. A line
# that is listed must be on the allowlist below, word for word, under its reason.
# Each reason must hold against another process of the same user that renames
# folders in the home while the script runs.
unanchored_writes() { # $1 = install.sh, $2 = the allowlist
    awk -v allow="$2" '
    BEGIN {
        while ((getline entry < allow) > 0) {
            sub(/^[ \t]+/, "", entry)
            if (entry ~ /^# /) { reason = entry; continue }
            if (entry == "") continue
            if (reason == "") { print "allowlist line without a reason: " entry; bad = 1 }
            allowed[entry] = 1; reason = ""
        }
        home = "(vade_home|home_dir|HOME|download|partial|cache|clone|clone_tmp|archive|prebuilt_dir|lock_dir|made|config_dir|config_base|script_tmp|shim|path_script|launcher|[12])"
        ref = "[$][{]?" home "[}]?([^A-Za-z0-9_]|$)"
        argument = "(^|[ \t=])\"?(" ref "|[.]/[^ \t/\"]+/)"
        word = "(^|[ \t!`])(rm|rmdir|mv|cp|ln|chmod|mkdir|mktemp|tar|touch|curl|git)([ \t]|$)"
        redirect = ">>?[ \t]*\"?(" ref "|[.]/[^ \t/\"]+/)"
    }
    {
        line = (joined == "" ? $0 : joined " " $0)
        if (line ~ /\\$/) { joined = substr(line, 1, length(line) - 1); next }
        joined = ""
        text = line; sub(/^[ \t]+/, "", text)
        if (text ~ /^#/) next
        if (text ~ /^in_dir\(\) \(/) { helper = ")"; next }
        if (text ~ /^by_name\(\) \{/) { helper = "}"; next }
        if (helper != "") { if (text == helper) helper = ""; next }
        rest = text
        gsub(/(in_dir|by_name)[ \t]+"[^"]*"/, "ANCHOR", rest)
        gsub(/\[ [^]]* \]/, "TEST", rest)
        hit = 0
        # Commands end at && || ; | ( ) and at a { or } that stands alone (not ${...}).
        count = split(rest, part, /&&|[|][|]|;|[|]|[()]|(^|[ \t])[{}]([ \t]|$)/)
        for (i = 1; i <= count; i++) {
            if (part[i] ~ redirect) hit = 1
            if (match(part[i], word) && substr(part[i], RSTART) ~ argument) hit = 1
        }
        if (hit) {
            if (text in allowed) { used[text] = 1; next }
            print "unanchored: " NR ": " text; bad = 1
        }
    }
    END {
        for (entry in allowed) if (!(entry in used)) { print "allowlist line not found: " entry; bad = 1 }
        exit bad
    }' "$1"
}
cat > "$matrix/anchored-allow" <<'EOF'
# Makes only empty folders under your home folder, not under the Vade home; mkdir -p deletes and overwrites nothing. The script write that follows runs through in_dir with the physical path and refuses a .local or .local/bin that was swapped.
elif { [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin"; } 2>/dev/null &&
EOF
if out=$(unanchored_writes "$root/scripts/install.sh" "$matrix/anchored-allow"); then
    printf 'PASS static: every write, move and removal in the home is anchored\n'
else
    printf 'FAIL static: every write, move and removal in the home is anchored\n%s\n' "$out"
    failed=1
fi

for shell in $shells; do
    for test_case in checkout-default pipe-default checkout-source pipe-source \
        checkout-check pipe-check checkout-help pipe-help half-pipe no-call \
        host-linux-x86_64 host-macos login-shim login-bin login-missing login-failed login-hanging; do
        case_dir="$matrix/$shell-$test_case"
        mkdir "$case_dir"
        set +e
        run_case > "$case_dir/result" 2>&1
        status=$?
        set -e
        label=$shell
        case $shell in bash) label='bash --posix' ;; zsh) label='zsh --emulate sh' ;; esac
        if [ "$status" = 0 ]; then
            printf 'PASS %s: %s\n' "$label" "$test_case"
        else
            printf 'FAIL %s: %s\n' "$label" "$test_case"
            cat "$case_dir/result" >&2
            if [ -f "$case_dir/stderr" ]; then cat "$case_dir/stderr" >&2; fi
            failed=1
        fi
    done
done
# The checks printed up front, --json and --uninstall. Each case sets up its own
# files; like the cases above, none uses a real GPU, network, compiler or vade.
run_new_case() (
    set -eu
    host_os=Linux host_arch=aarch64 login_shell= gpu_busy= df_kb= df_apart= down_fails= vade_home_env=
    fixture
    for tool in cat ls find; do ln -s "$(command -v "$tool")" "$case_dir/bin/$tool"; done
    home="$case_dir/home"
    out="$case_dir/stdout" err="$case_dir/stderr"
    fake_df() {
        cat > "$case_dir/bin/df" <<'EOF'
#!/bin/sh
# With MATRIX_DF_APART=1 a models folder is on another disk.
mount=/
case "${MATRIX_DF_APART-}:$2" in 1:*/models) mount=/srv ;; esac
printf '%s\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on' "/dev/fake 4000000000 1000 $MATRIX_DF_KB 1% $mount"
EOF
        chmod 700 "$case_dir/bin/df"
    }
    run() { status=0; install_shell "$case_dir/checkout/scripts/install.sh" "$@" > "$out" 2> "$err" || status=$?; }
    has() { grep -qxF -- "$2" "$1" || { printf 'Missing line in %s: %s\n' "$1" "$2"; exit 1; }; }
    lacks() { ! grep -qF -- "$2" "$1" || { printf 'Unexpected text in %s: %s\n' "$1" "$2"; exit 1; }; }
    unchanged() { snapshot > "$case_dir/after"; cmp "$case_dir/before" "$case_dir/after"; }
    # Standard output holds JSON lines only, and each one parses when python3 is here.
    json_only() {
        [ -s "$out" ]
        ! grep -qv '^{"schema":"vade.install.v1","ok":[a-z]*,"error":.*}$' "$out"
        [ "$(grep -c '"event":"result"' "$out")" = 1 ]
        tail -n 1 "$out" | grep -q '"event":"result"'
        if [ -n "$python" ]; then "$python" -c 'import json,sys
for line in sys.stdin: json.loads(line)' < "$out"; fi
    }
    result_has() { tail -n 1 "$out" | grep -qF -- "$1" || { printf 'Result lacks: %s\n' "$1"; tail -n 1 "$out"; exit 1; }; }
    # A Vade home as an install and some use leave it, with one file that is not from Vade.
    # A clone as the source build makes it: a whole Vade checkout.
    make_clone() { mkdir -p "$1/scripts" "$1/cli" "$1/engine" "$1/harness"; for file in scripts/install.sh cli/Cargo.toml engine/Cargo.toml harness/Cargo.toml; do : > "$1/$file"; done; }
    # What an older installer leaves: an unpacked bundle, a download cache and a clone, each in its exact shape.
    make_leftovers() {
        sum=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
        mkdir -p "$1/bundles/vade-0.1.0-abcdefg-linux-aarch64-gb10" "$1/bundles/.partial-$sum/lock"
        printf '%s\n' format=vade-release-v1 name=vade-0.1.0-abcdefg-linux-aarch64-gb10 sha=abc > "$1/bundles/vade-0.1.0-abcdefg-linux-aarch64-gb10/MANIFEST"
        : > "$1/bundles/.partial-$sum/bundle.part"
        make_clone "$1/source.abcd1234"
    }
    make_home() {
        mkdir -p "$1/bin" "$1/stage/builds/aaaaaaaaaaaaaaaaaaaaaaaa" "$1/state" "$1/bundles/vade-x" \
            "$1/build-jobs" "$1/source.abcd1234/scripts" "$1/controller/state/groups/pair" "$1/sessions"
        make_clone "$1/source.abcd1234"
        # Names that only look like a clone stay: no installer inside, or another length.
        mkdir -p "$1/source.notclone" "$1/source.abc"
        printf '%s\n' engine > "$1/stage/builds/aaaaaaaaaaaaaaaaaaaaaaaa/flash-serve"
        printf '%s\n' 'port = 18080' > "$1/config"
        : > "$1/.lock"
        printf '%s\n' 'my notes' > "$1/instructions.md"
        printf '%s\n' '{}' > "$1/trust.json"
        printf '%s\n' 'a chat' > "$1/sessions/s1"
        printf '%s\n' 'not from Vade' > "$1/my-own-file"
        printf '%s\n' vade-home-v1 > "$1/.vade-home"
        cat > "$1/bin/vade" <<'EOF'
#!/bin/sh
set -eu
case ${1-} in
    --version) echo 'vade 0.1.0' ;;
    down)
        printf '%s\n' 'vade down' >> "$MATRIX_CALLS"
        [ -z "${MATRIX_DOWN_FAILS-}" ] || { echo 'error: cannot stop the engine' >&2; exit 1; }
        rm -f "$(dirname "$(dirname "$0")")/state/run.conf"
        echo 'Vade is already stopped.' ;;
    *) exit 99 ;;
esac
EOF
        chmod 700 "$1/bin/vade"
    }
    # What sits outside the Vade home: the script on PATH, the controller config, a model.
    make_outside() {
        mkdir -p "$home/.local/bin" "$home/.config/vade" "$home/models"
        printf '%s\n' '#!/bin/sh' '# Written by the Vade installer. vade-path-shim-v1' 'exec "$HOME/.vade/bin/vade" "$@"' > "$home/.local/bin/vade"
        printf '%s\n' 'ssh = spark-a' > "$home/.config/vade/config"
        printf '%s\n' weights > "$home/models/m"
    }
    gone() { for path do [ ! -e "$path" ] || { printf 'Still there: %s\n' "$path"; exit 1; }; done; }
    there() { for path do [ -e "$path" ] || { printf 'Missing: %s\n' "$path"; exit 1; }; done; }
    programs="bin/vade bundles stage state build-jobs controller config .lock source.abcd1234"
    cd "$case_dir/work"
    case $test_case in
        checks-pass)
            df_kb=50000000; fake_df
            run
            [ "$status" = 0 ]
            has "$out" 'Checks of this Spark:'
            has "$out" '  ok    this box     Linux aarch64'
            has "$out" '  ok    GPU          NVIDIA GB10'
            has "$out" '  ok    driver       CUDA 13'
            has "$out" '  ok    GPU is free  no compute process'
            has "$out" "  ok    disk         47.6 GiB free in $home (the install needs 9 GiB with the model on the same disk)"
            lacks "$out" FAIL
            # Every check line comes before the first step that downloads or builds.
            [ "$(grep -n '^Checks of this Spark:' "$out" | cut -d : -f 1)" -lt "$(grep -n '^Prebuilt: ' "$out" | cut -d : -f 1)" ]
            [ "$(grep -n '  ok    disk ' "$out" | cut -d : -f 1)" -lt "$(grep -n '^Prebuilt: ' "$out" | cut -d : -f 1)" ]
            has "$out" 'Install complete. Next steps:'
            ;;
        checks-disk-edge)
            # A first install asks what vade install asks: 8 GiB, plus 1 GiB when the model
            # folder is on the same disk (9437184 KiB). One KiB less fails.
            df_kb=9437183; fake_df
            run
            [ "$status" = 1 ]
            has "$out" "  FAIL  disk         the disk has 8.9 GiB free in $home, and the install needs 9 GiB with the model on the same disk (the model needs 103 GB more)."
            reject_calls '^(git|cargo|rustup|vade) '
            df_kb=9437184
            run
            [ "$status" = 0 ]
            has "$out" "  ok    disk         9.0 GiB free in $home (the install needs 9 GiB with the model on the same disk)"
            ;;
        checks-disk-apart)
            # The model folder on another disk: 8 GiB (8388608 KiB), as vade install asks.
            mkdir "$home/models"; df_apart=1
            df_kb=8388607; fake_df
            run
            [ "$status" = 1 ]
            has "$out" "  FAIL  disk         the disk has 7.9 GiB free in $home, and the install needs 8 GiB (the model needs 103 GB more)."
            reject_calls '^(git|cargo|rustup|vade) '
            df_kb=8388608
            run
            [ "$status" = 0 ]
            has "$out" "  ok    disk         8.0 GiB free in $home (the install needs 8 GiB)"
            ;;
        checks-disk-update)
            # An update (stage/install.conf is there) needs no new room, plus 1 GiB with the model on the same disk.
            mkdir -p "$home/.vade/stage"; chmod 700 "$home/.vade"
            printf '%s\n' vade-home-v1 > "$home/.vade/.vade-home"
            printf '%s\n' 'format=vade-local-install-v1' 'generation=aaaaaaaaaaaaaaaaaaaaaaaa' > "$home/.vade/stage/install.conf"
            df_kb=1048575; fake_df
            run
            [ "$status" = 1 ]
            has "$out" "  FAIL  disk         the disk has 0.9 GiB free in $home/.vade, and the update needs 1 GiB with the model on the same disk (the model needs 103 GB more)."
            reject_calls '^(git|cargo|rustup|vade) '
            df_kb=1048576
            run
            has "$out" "  ok    disk         1.0 GiB free in $home/.vade (the update needs 1 GiB with the model on the same disk)"
            mkdir "$home/models"; df_apart=1; df_kb=0
            run
            has "$out" "  ok    disk         0.0 GiB free in $home/.vade (the update needs 0 GiB)"
            ;;
        safety-swap-before-cd)
            # A link put at a checked folder after its check and before the cd: the
            # pwd -P compare refuses it, and the folder outside keeps its files.
            mkdir -p "$home/.vade/stage" "$case_dir/outside/stage"
            printf '%s\n' mine > "$case_dir/outside/stage/keep"
            sed -n -e '/^no_links() {/,/^}/p' -e '/^in_dir() (/,/^)/p' -e '/^by_name() {/,/^}/p' \
                "$case_dir/checkout/scripts/install.sh" > "$case_dir/helpers.sh"
            [ "$(grep -c '^[a-z_]*() [({]' "$case_dir/helpers.sh")" = 3 ]
            cat > "$case_dir/swap.sh" <<'EOF'
. "$MATRIX_CASE/helpers.sh"
# The check passes, then the swap: the same moment as a swap just after no_links.
no_links() {
    [ ! -e "$MATRIX_CASE/swapped" ] || return 0
    : > "$MATRIX_CASE/swapped"
    mv "$HOME/.vade" "$HOME/saved"
    /bin/ln -s "$MATRIX_CASE/outside" "$HOME/.vade"
}
if by_name "$HOME/.vade/stage" rm -rf --; then echo removed; else echo refused; fi
EOF
            status=0
            install_shell "$case_dir/swap.sh" > "$out" 2> "$err" || status=$?
            [ "$status" = 0 ]
            has "$out" refused
            [ -e "$case_dir/swapped" ]
            [ "$(cat "$case_dir/outside/stage/keep")" = mine ]
            [ -d "$home/saved/stage" ]
            ;;
        checks-skip)
            run
            [ "$status" = 0 ]
            has "$out" '  skip  disk         not read here; vade install checks it'
            ;;
        checks-source)
            run --source
            [ "$status" = 0 ]
            has "$out" '  ok    C compiler   cc'
            has "$out" '  ok    Rust         rustup found; missing pinned toolchains are added'
            ;;
        checks-busy|checks-disk|checks-two)
            case $test_case in checks-busy|checks-two) gpu_busy=1 ;; esac
            case $test_case in checks-disk|checks-two) df_kb=1000000; fake_df ;; esac
            snapshot > "$case_dir/before"
            run
            [ "$status" = 1 ]
            disk="the disk has 0.9 GiB free in $home, and the install needs 9 GiB with the model on the same disk (the model needs 103 GB more)."
            busy='install: the GPU is in use. Next: stop that workload through the launcher that started it, then retry.'
            case $test_case in checks-busy|checks-two) has "$out" '  FAIL  GPU is free  the GPU is in use.' ;; esac
            case $test_case in checks-disk|checks-two) has "$out" "  FAIL  disk         $disk" ;; esac
            # Both faults are listed; the install line names the first one.
            case $test_case in
                checks-disk) has "$err" "install: $disk Next: free space there, or set VADE_HOME to a folder on a larger disk, then retry." ;;
                *) has "$err" "$busy" ;;
            esac
            has "$err" "Retry: sh \"$case_dir/checkout/scripts/install.sh\""
            [ "$(grep -c '^install: ' "$err")" = 1 ]
            reject_calls '^(git|cargo|rustup|vade) '
            unchanged
            ;;
        checks-check-busy)
            gpu_busy=1
            snapshot > "$case_dir/before"
            run --check
            [ "$status" = 0 ]
            has "$out" '  FAIL  GPU is free  the GPU is in use.'
            has "$out" 'Check: the GPU is in use; the install would stop until that workload stops.'
            grep -q '^Check: would install the pinned Rust toolchains' "$out"
            unchanged
            ;;
        json-install|json-pipe)
            if [ "$test_case" = json-pipe ]; then
                status=0
                cat "$case_dir/checkout/scripts/install.sh" | install_shell -s -- --json > "$out" 2> "$err" || status=$?
                grep -q '^git clone$' "$case_dir/calls"
            else
                run --json
            fi
            [ "$status" = 0 ]
            json_only
            for field in '"ok":true,"error":null,' '"action":"install","check":false,"route":"source","build":null,' \
                "\"next\":\"$home/.vade/bin/vade up\"" "\"home\":\"$home/.vade\"" "\"vade\":\"$home/.vade/bin/vade\"" \
                '"path_line":"export PATH=\"$HOME/.vade/bin:$PATH\"","on_login_path":false,'; do result_has "$field"; done
            for name in check build install; do grep -q "\"event\":\"step\",\"step\":\"$name\"" "$out"; done
            # Every line a person reads went to standard error.
            has "$err" 'Checks of this Spark:'
            has "$err" 'Install complete. Next steps:'
            [ "$(tail -n 1 "$case_dir/calls")" = 'vade __install-here' ]
            [ -x "$home/.vade/bin/vade" ]
            ;;
        json-left-running)
            # A login profile leaves a program running. The reader of the JSON must not wait for it:
            # the result arrives through a pipe well before that program ends.
            login_shell="$case_dir/bin/login-shell"
            printf '%s\n' '/bin/sleep 25 > /dev/null 2>&1 &' > "$home/.profile"
            status=0
            "$system_timeout" 15 /bin/sh -c '"$@" | cat > "$0"' "$out" \
                env -i PATH="$case_dir/bin" HOME="$home" TMPDIR="$case_dir/tmp" CARGO_NET_OFFLINE=true \
                MATRIX_CASE="$case_dir" MATRIX_CALLS="$case_dir/calls" MATRIX_OS=Linux MATRIX_ARCH=aarch64 \
                SHELL="$login_shell" /bin/sh "$case_dir/checkout/scripts/install.sh" --json 2> "$err" || status=$?
            [ "$status" = 0 ] || { echo "the pipe stayed open: status $status"; exit 1; }
            json_only
            result_has '"ok":true,"error":null,'
            printf '%s\n' 'Keep this profile.' > "$home/.profile"
            ;;
        progress-live)
            # Without a terminal, a line that vade prints reaches the log while vade still runs.
            # The fake vade prints one line, then waits for the test to see that line.
            cat > "$case_dir/fake-vade" <<'EOF'
#!/bin/sh
set -eu
[ "$1" = __install-here ] || exit 99
printf '%s\n' "vade $*" >> "$MATRIX_CALLS"
printf '%s\n' 'Fixture step one.'
waited=0
until [ -e "$MATRIX_CASE/seen" ]; do
    /bin/sleep 0.2; waited=$((waited + 1))
    [ "$waited" -lt 60 ] || { echo 'the line was held back' >&2; exit 9; }
done
printf 'Then start the engine and open chat: %s launch\n' "$0"
EOF
            install_shell "$case_dir/checkout/scripts/install.sh" > "$out" 2> "$err" &
            installer=$!
            waited=0
            until grep -qx 'Fixture step one.' "$out" 2>/dev/null; do
                /bin/sleep 0.2; waited=$((waited + 1))
                [ "$waited" -lt 100 ] || break
            done
            : > "$case_dir/seen"
            status=0; wait "$installer" || status=$?
            [ "$status" = 0 ] || { echo "the line did not arrive while vade ran: status $status"; exit 1; }
            has "$out" 'Fixture step one.'
            lacks "$out" 'Then start the engine and open chat: /'
            has "$out" 'Install complete. Next steps:'
            ;;
        json-check)
            snapshot > "$case_dir/before"
            run --check --json
            [ "$status" = 0 ]
            json_only
            result_has '"ok":true,"error":null,"next":null,"event":"result","action":"install","check":true,"would":"install the pinned Rust toolchains if missing, build vade in '
            reject_calls '^(git|cargo|rustup|vade) '
            unchanged
            ;;
        json-busy|json-check-busy)
            gpu_busy=1
            snapshot > "$case_dir/before"
            if [ "$test_case" = json-busy ]; then run --json; [ "$status" = 1 ]; check=false
            else run --check --json; [ "$status" = 0 ]; check=true; fi
            json_only
            result_has "\"ok\":false,\"error\":{\"code\":\"gpu_in_use\",\"message\":\"the GPU is in use.\"},\"next\":\"stop that workload through the launcher that started it, then retry.\",\"event\":\"result\",\"action\":\"install\",\"check\":$check,"
            [ "$test_case" = json-check-busy ] || [ "$(grep -c . "$out")" = 2 ]
            reject_calls '^(git|cargo|rustup|vade) '
            unchanged
            ;;
        json-bad-flag)
            snapshot > "$case_dir/before"
            run --json --nope
            [ "$status" = 1 ]
            json_only
            result_has '"ok":false,"error":{"code":"bad_arguments","message":"unknown arguments."},"next":"run the installer with --help."'
            [ ! -s "$case_dir/calls" ]
            run --all
            [ "$status" = 1 ]
            has "$err" 'install: --all goes with --uninstall. Next: run the installer with --help.'
            run --uninstall --source
            [ "$status" = 1 ]
            has "$err" 'install: --uninstall takes only --all, --check and --json. Next: run the installer with --help.'
            run --json --launch
            [ "$status" = 1 ]
            result_has '"code":"bad_arguments","message":"--json and --launch do not go together: chat needs a terminal."'
            unchanged
            ;;
        json-odd-path)
            # A quote and a backslash in a path stay valid JSON.
            vade_home_env="$home/od\"d\\home"
            run --uninstall --json
            [ "$status" = 0 ]
            json_only
            result_has '/home/od\"d\\home","removed":0,'
            if [ -n "$python" ]; then
                [ "$(tail -n 1 "$out" | "$python" -c 'import json,sys; print(json.loads(sys.stdin.read())["home"])')" = "$vade_home_env" ]
            fi
            ;;
        uninstall-check)
            make_home "$home/.vade"; make_outside
            : > "$home/.vade/state/run.conf"
            snapshot > "$case_dir/before"
            run --uninstall --check
            [ "$status" = 0 ]
            for name in $programs; do has "$out" "Check: would remove $home/.vade/$name"; done
            has "$out" "Check: would remove $home/.local/bin/vade"
            has "$out" 'Check: an engine is recorded; the uninstall would run vade down first.'
            has "$out" "Kept $home/.vade/sessions (yours; --uninstall --all removes it)"
            has "$out" "Kept $home/models (the model files)"
            has "$out" 'Check: the uninstall would remove 10 of these; nothing was removed.'
            reject_calls '^vade '
            unchanged
            ;;
        uninstall-default|uninstall-json)
            make_home "$home/.vade"; make_outside
            : > "$home/.vade/state/run.conf"
            if [ "$test_case" = uninstall-json ]; then
                run --uninstall --json
                json_only
                result_has "\"ok\":true,\"error\":null,\"next\":null,\"event\":\"result\",\"action\":\"uninstall\",\"check\":false,\"all\":false,\"home\":\"$home/.vade\",\"removed\":10,"
                grep -q '"event":"step","step":"uninstall"' "$out"
                has "$err" 'Uninstall done: 10 removed.'
            else
                run --uninstall
                has "$out" 'Uninstall: stopping the engine that vade started (vade down).'
                has "$out" 'Uninstall done: 10 removed.'
            fi
            [ "$status" = 0 ]
            [ "$(grep -c '^vade ' "$case_dir/calls")" = 1 ]
            grep -qx 'vade down' "$case_dir/calls"
            for name in $programs; do gone "$home/.vade/$name"; done
            gone "$home/.local/bin/vade"
            there "$home/.vade/source.notclone" "$home/.vade/source.abc"
            # The person's own files, the controller config and the model stay.
            there "$home/.vade/sessions/s1" "$home/.vade/instructions.md" "$home/.vade/trust.json" "$home/.vade/my-own-file" \
                "$home/.config/vade/config" "$home/models/m" "$home/.local/bin"
            # A second run finds nothing and changes nothing.
            snapshot > "$case_dir/before"
            run --uninstall
            [ "$status" = 0 ]
            has "$out" 'Uninstall: nothing to remove. Vade is not installed here.'
            unchanged
            ;;
        uninstall-all|uninstall-all-clean)
            make_home "$home/.vade"; make_outside
            [ "$test_case" = uninstall-all ] || { rm "$home/.vade/my-own-file"; rmdir "$home/.vade/source.notclone" "$home/.vade/source.abc"; }
            run --uninstall --all
            [ "$status" = 0 ]
            for name in $programs sessions instructions.md trust.json; do gone "$home/.vade/$name"; done
            gone "$home/.local/bin/vade" "$home/.config/vade"
            there "$home/models/m" "$home/.config"
            if [ "$test_case" = uninstall-all ]; then
                # A file that is not from Vade is never removed, so its folder stays.
                there "$home/.vade/my-own-file"
                has "$out" "Kept $home/.vade (it holds files that are not from Vade)"
                # The config folder counts twice: Vade's config file, then the empty folder.
                has "$out" "Removed $home/.config/vade/config"
                has "$out" 'Uninstall done: 15 removed.'
            else
                gone "$home/.vade"
                has "$out" "Removed $home/.vade"
            fi
            ;;
        uninstall-down-fails|uninstall-no-program|uninstall-group-record)
            make_home "$home/.vade"; make_outside
            record="$home/.vade/state/run.conf"
            # A controller's record of a group that the fake vade down does not clear.
            [ "$test_case" != uninstall-group-record ] || record="$home/.vade/controller/state/groups/pair/session.conf"
            : > "$record"
            case $test_case in uninstall-down-fails) down_fails=1 ;; uninstall-no-program) rm "$home/.vade/bin/vade" ;; esac
            snapshot > "$case_dir/before"
            run --uninstall
            [ "$status" = 1 ]
            if [ "$test_case" = uninstall-down-fails ]; then
                has "$err" "install: vade down failed, so nothing was removed. Next: fix the cause above, run \"$home/.vade/bin/vade\" down, then retry."
            else
                has "$err" "install: an engine that vade started is still recorded in $record, so nothing was removed. Next: stop it with the vade that started it (vade down), then retry."
            fi
            lacks "$err" 'Retry:'
            lacks "$out" Removed
            unchanged
            ;;
        uninstall-foreign-script)
            make_home "$home/.vade"; make_outside
            printf '%s\n' '#!/bin/sh' 'echo another program named vade' > "$home/.local/bin/vade"
            run --uninstall
            [ "$status" = 0 ]
            there "$home/.local/bin/vade"
            lacks "$out" "$home/.local/bin/vade"
            has "$out" 'Uninstall done: 9 removed.'
            ;;
        uninstall-other-home)
            # Another VADE_HOME: the default install, its script on PATH and the config are not touched.
            make_home "$home/.vade"; make_outside; make_home "$home/second"
            vade_home_env="$home/second"
            (cd "$case_dir" && find home/.vade home/.local home/.config home/models -print -type f -exec cksum {} \; | LC_ALL=C sort) > "$case_dir/default-before"
            run --uninstall --all
            [ "$status" = 0 ]
            for name in $programs sessions instructions.md; do gone "$home/second/$name"; done
            there "$home/second/my-own-file"
            (cd "$case_dir" && find home/.vade home/.local home/.config home/models -print -type f -exec cksum {} \; | LC_ALL=C sort) > "$case_dir/default-after"
            cmp "$case_dir/default-before" "$case_dir/default-after"
            lacks "$out" "$home/.local/bin/vade"
            lacks "$out" "$home/.config/vade"
            ;;
        safety-home-link)
            # A linked home is refused; the folder it names keeps every file.
            make_home "$case_dir/elsewhere"; make_outside
            ln -s "$case_dir/elsewhere" "$home/.vade"
            (cd "$case_dir" && find elsewhere -print | LC_ALL=C sort) > "$case_dir/elsewhere-before"
            run --uninstall --all --json
            [ "$status" = 1 ]
            json_only
            result_has '"error":{"code":"bad_home"'
            (cd "$case_dir" && find elsewhere -print | LC_ALL=C sort) > "$case_dir/elsewhere-after"
            cmp "$case_dir/elsewhere-before" "$case_dir/elsewhere-after"
            there "$home/.config/vade/config" "$home/.local/bin/vade"
            reject_calls '^vade '
            ;;
        safety-config-link)
            # A linked XDG_CONFIG_HOME: --all stops before any removal, and the folder it names keeps its file.
            make_home "$home/.vade"; make_outside
            mkdir -p "$case_dir/elsewhere/vade"; printf '%s\n' keep > "$case_dir/elsewhere/vade/config"
            rm -r "$home/.config"; ln -s "$case_dir/elsewhere" "$home/.config"
            snapshot > "$case_dir/before"
            run --uninstall --all
            [ "$status" = 1 ]
            has "$err" "install: XDG_CONFIG_HOME ($home/.config) must be a full path of two or more parts, with no empty, . or .. part and no link in it, so nothing was removed. Next: unset XDG_CONFIG_HOME or set it to the real folder, then retry."
            there "$case_dir/elsewhere/vade/config"
            unchanged
            ;;
        safety-bad-homes)
            # Refused before anything is read or removed. --check keeps the old script harmless here too.
            for bad in / /tmp "$home/.vade/../.vade" "$home/./.vade" "$home//.vade" "$home" relative/home; do
                vade_home_env=$bad
                for action in --uninstall ''; do
                    run $action --check --json
                    [ "$status" = 0 ] || { echo "status $status for $bad $action"; exit 1; }
                    json_only
                    result_has '"ok":false,"error":{"code":"bad_home"'
                    lacks "$err" 'would remove'
                done
            done
            reject_calls '^(git|cargo|rustup|vade) '
            ;;
        safety-names)
            # A clone name with a space or a * is one name; a name a pattern would find here stays.
            make_home "$home/.vade"; make_outside
            for name in 'source.ab cd 12' 'source.????????'; do make_clone "$home/.vade/$name"; done
            # A clone name with only a scripts/install.sh is not a checkout, so it stays.
            mkdir -p "$home/.vade/source.halfway1/scripts"; : > "$home/.vade/source.halfway1/scripts/install.sh"
            mkdir -p "$case_dir/work/source.zzzzzzzz/scripts"; : > "$case_dir/work/source.zzzzzzzz/scripts/install.sh"
            run --uninstall
            [ "$status" = 0 ]
            gone "$home/.vade/source.ab cd 12" "$home/.vade/source.????????" "$home/.vade/source.abcd1234"
            there "$case_dir/work/source.zzzzzzzz/scripts/install.sh" "$home/.vade/source.notclone" "$home/.vade/my-own-file" \
                "$home/.vade/source.halfway1/scripts/install.sh"
            has "$out" 'Uninstall done: 12 removed.'
            ;;
        safety-leaf-link)
            # A link in place of a listed name is kept, and what it names is untouched.
            make_home "$home/.vade"; make_outside
            rm -r "$home/.vade/stage"; mkdir -p "$case_dir/elsewhere/builds"; ln -s "$case_dir/elsewhere" "$home/.vade/stage"
            run --uninstall
            [ "$status" = 0 ]
            has "$out" "Kept $home/.vade/stage (a link, or not a plain path inside $home/.vade: the uninstall never removes one)"
            there "$case_dir/elsewhere/builds"
            [ -L "$home/.vade/stage" ]
            ;;
        safety-lock-held)
            # A lock left by a running install: the uninstall and a second install stop and change nothing.
            make_home "$home/.vade"; make_outside
            mkdir "$home/.vade/.install-lock"
            snapshot > "$case_dir/before"
            run --uninstall --json
            [ "$status" = 1 ]
            json_only
            result_has '"error":{"code":"busy","message":"another install or uninstall is running on '
            run --json
            [ "$status" = 1 ]
            result_has '"error":{"code":"busy",'
            reject_calls '^(git|cargo|rustup|vade) '
            unchanged
            ;;
        safety-two-installs)
            # Two installs at once: the second stops at the lock, the first ends well and drops it.
            cat > "$case_dir/fake-vade" <<'EOF'
#!/bin/sh
set -eu
[ "$1" = __install-here ] || exit 99
printf '%s\n' "vade $*" >> "$MATRIX_CALLS"
printf '%s\n' 'Fixture step one.'
waited=0
until [ -e "$MATRIX_CASE/seen" ]; do
    /bin/sleep 0.2; waited=$((waited + 1))
    [ "$waited" -lt 100 ] || exit 9
done
EOF
            install_shell "$case_dir/checkout/scripts/install.sh" > "$case_dir/first" 2>&1 &
            first=$!
            waited=0
            until grep -qx 'Fixture step one.' "$case_dir/first" 2>/dev/null; do
                /bin/sleep 0.2; waited=$((waited + 1))
                [ "$waited" -lt 100 ] || break
            done
            [ -d "$home/.vade/.install-lock" ]
            run --json
            second=$status
            run --uninstall
            third=$status
            : > "$case_dir/seen"
            status=0; wait "$first" || status=$?
            [ "$second" = 1 ] && [ "$third" = 1 ] && [ "$status" = 0 ] || { echo "status $second $third $status"; exit 1; }
            has "$err" "install: another install or uninstall is running on $home/.vade. Next: wait for it to end; if none runs, remove the empty folder $home/.vade/.install-lock, then retry."
            gone "$home/.vade/.install-lock"
            there "$home/.vade/bin/vade" "$home/.vade/.vade-home"
            [ "$(grep -c '^vade __install-here' "$case_dir/calls")" = 1 ]
            ;;
        safety-foreign-home)
            # A folder that is not a Vade home: nothing goes, also with --check, and the install does not take it over.
            foreign="$case_dir/local"
            mkdir -p "$foreign/bin" "$foreign/state" "$foreign/share"
            printf '%s\n' tool > "$foreign/bin/pipx"; printf '%s\n' data > "$foreign/state/app"; printf '%s\n' doc > "$foreign/share/y"
            vade_home_env=$foreign
            listing() { (cd "$case_dir" && find local -print -type f -exec cksum {} \; | LC_ALL=C sort); }
            listing > "$case_dir/local-before"
            refusal="$foreign is not a Vade home: it has no .vade-home mark, no stage/install.conf from vade install, and not only what an earlier Vade attempt leaves, so nothing was removed."
            run --uninstall --check
            [ "$status" = 0 ]
            has "$out" "Check: the install would stop here: $refusal Next: check that VADE_HOME names the folder Vade was installed in; if it holds an earlier Vade attempt, look at what is in it and remove it by hand, then retry."
            lacks "$out" 'would remove'
            run --uninstall --all --json
            [ "$status" = 1 ]
            json_only
            result_has "\"error\":{\"code\":\"bad_home\",\"message\":\"$refusal\"}"
            for flags in --json '--check --json'; do
                run $flags
                json_only
                result_has "\"error\":{\"code\":\"bad_home\",\"message\":\"$foreign already holds files and is not a Vade home (no .vade-home mark), so nothing was written.\"}"
            done
            listing > "$case_dir/local-after"
            cmp "$case_dir/local-before" "$case_dir/local-after"
            reject_calls '^(git|cargo|rustup|vade) '
            ;;
        safety-marked-home)
            # A Vade home with your own files beside Vade's: only Vade's own items go, also from bin.
            make_home "$home/.vade"; make_outside
            printf '%s\n' mine > "$home/.vade/bin/my-tool"
            run --uninstall --check
            [ "$status" = 0 ]
            has "$out" "Check: would remove $home/.vade/bin/vade"
            ! grep -qxF "Check: would remove $home/.vade/bin" "$out"
            there "$home/.vade/bin/vade"
            run --uninstall --all
            [ "$status" = 0 ]
            gone "$home/.vade/bin/vade" "$home/.vade/state" "$home/.vade/sessions" "$home/.vade/stage"
            there "$home/.vade/bin/my-tool" "$home/.vade/my-own-file" "$home/.vade/.vade-home"
            has "$out" "Kept $home/.vade (it holds files that are not from Vade)"
            ;;
        safety-legacy-home)
            # A home from an older installer has no mark: stage/install.conf from vade install shows it is Vade's.
            make_home "$home/.vade"; make_outside
            rm "$home/.vade/.vade-home"
            snapshot > "$case_dir/before"
            run --uninstall
            [ "$status" = 1 ]
            grep -q "^install: $home/.vade is not a Vade home: " "$err"
            unchanged
            printf '%s\n' 'format=vade-local-install-v1' 'generation=aaaaaaaaaaaaaaaaaaaaaaaa' > "$home/.vade/stage/install.conf"
            run --uninstall
            [ "$status" = 0 ]
            gone "$home/.vade/bin" "$home/.vade/stage"
            there "$home/.vade/my-own-file" "$home/.vade/sessions/s1"
            # Known once, known again: the uninstall left the mark, so a new install and --all both work.
            [ "$(cat "$home/.vade/.vade-home")" = vade-home-v1 ]
            run
            [ "$status" = 0 ] || { cat "$err"; exit 1; }
            there "$home/.vade/bin/vade" "$home/.vade/sessions/s1"
            run --uninstall --all
            [ "$status" = 0 ] || { cat "$err"; exit 1; }
            gone "$home/.vade/sessions" "$home/.vade/bin/vade"
            there "$home/.vade/my-own-file"
            ;;
        safety-old-leftovers)
            # A true older leftover: the uninstall removes it item by item, and so does an uninstall after an install.
            make_leftovers "$home/.vade"
            run --uninstall --check --json
            [ "$status" = 0 ]
            json_only
            result_has '"ok":true,"error":null,'
            run --uninstall
            [ "$status" = 0 ] || { cat "$err"; exit 1; }
            has "$out" "Removed $home/.vade/bundles/vade-0.1.0-abcdefg-linux-aarch64-gb10"
            has "$out" "Removed $home/.vade/bundles/.partial-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            has "$out" "Removed $home/.vade/source.abcd1234"
            gone "$home/.vade"
            make_leftovers "$home/.vade"
            run
            [ "$status" = 0 ] || { cat "$err"; exit 1; }
            there "$home/.vade/bin/vade" "$home/.vade/.vade-home"
            run --uninstall
            [ "$status" = 0 ] || { cat "$err"; exit 1; }
            gone "$home/.vade"
            ;;
        safety-leftovers-refused)
            # Each folder holds something that does not prove itself Vade's: refused whole by the uninstall,
            # its --check and the install, and left exactly as it was.
            mkdir -p "$case_dir/elsewhere"; make_clone "$case_dir/elsewhere/clone"; mkdir "$case_dir/elsewhere/bundles"
            for variant in user-bundles not-a-checkout extra-in-bundles linked-clone linked-bundles empty dotfile download-folder short-cache; do
                f="$case_dir/v-$variant"
                case $variant in
                    user-bundles) mkdir -p "$f/bundles/my project"; printf '%s\n' mine > "$f/bundles/notes.txt" ;;
                    not-a-checkout) mkdir -p "$f/source.abcdefgh/scripts"; printf '%s\n' '#!/bin/sh' > "$f/source.abcdefgh/scripts/install.sh" ;;
                    extra-in-bundles) make_leftovers "$f"; printf '%s\n' mine > "$f/bundles/notes.txt" ;;
                    linked-clone) mkdir -p "$f"; ln -s "$case_dir/elsewhere/clone" "$f/source.abcd1234" ;;
                    linked-bundles) mkdir -p "$f"; ln -s "$case_dir/elsewhere/bundles" "$f/bundles" ;;
                    empty) mkdir -p "$f" ;;
                    dotfile) make_leftovers "$f"; printf '%s\n' x > "$f/.cache" ;;
                    download-folder) make_leftovers "$f"; mkdir -p "$f/bundles/.download.abcd1234" ;;
                    short-cache) mkdir -p "$f/bundles/.partial-0123"; : > "$f/bundles/.partial-0123/bundle.part" ;;
                esac
                vade_home_env=$f
                (cd "$case_dir" && find "v-$variant" elsewhere -print | LC_ALL=C sort) > "$case_dir/variant-before"
                run --uninstall
                [ "$status" = 1 ] || { echo "$variant: uninstall status $status"; cat "$out"; exit 1; }
                grep -q "^install: $f is not a Vade home: " "$err" || { echo "$variant: no refusal"; cat "$err"; exit 1; }
                run --uninstall --check
                [ "$status" = 0 ] && lacks "$out" 'would remove'
                grep -q "^Check: the install would stop here: $f is not a Vade home: " "$out"
                if [ "$variant" != empty ]; then
                    run
                    [ "$status" = 1 ] || { echo "$variant: install status $status"; exit 1; }
                    grep -q "^install: $f already holds files and is not a Vade home" "$err"
                fi
                (cd "$case_dir" && find "v-$variant" elsewhere -print | LC_ALL=C sort) > "$case_dir/variant-after"
                cmp "$case_dir/variant-before" "$case_dir/variant-after" || { echo "$variant changed"; exit 1; }
            done
            reject_calls '^(git|cargo|rustup|vade) '
            ;;
        *) echo "unknown case $test_case"; exit 1 ;;
    esac
    reject_calls '^unexpected-'
    for profile in .profile .bashrc .zshrc; do
        [ "$(cat "$home/$profile")" = 'Keep this profile.' ]
    done
    [ "$(cat "$case_dir/prefix/keep")" = 'Keep this prefix.' ]
)

python=
if command -v python3 >/dev/null 2>&1; then python=$(command -v python3); fi
for shell in $shells; do
    for test_case in checks-pass checks-disk-edge checks-disk-apart checks-disk-update checks-skip checks-source checks-busy checks-disk checks-two checks-check-busy progress-live \
        json-install json-pipe json-left-running json-check json-busy json-check-busy json-bad-flag json-odd-path \
        uninstall-check uninstall-default uninstall-json uninstall-all uninstall-all-clean \
        uninstall-down-fails uninstall-no-program uninstall-group-record uninstall-foreign-script uninstall-other-home \
        safety-swap-before-cd safety-home-link safety-config-link safety-bad-homes safety-names safety-leaf-link safety-lock-held safety-two-installs \
        safety-foreign-home safety-marked-home safety-legacy-home safety-old-leftovers safety-leftovers-refused; do
        case_dir="$matrix/$shell-$test_case"
        mkdir "$case_dir"
        set +e
        run_new_case > "$case_dir/result" 2>&1
        status=$?
        set -e
        label=$shell
        case $shell in bash) label='bash --posix' ;; zsh) label='zsh --emulate sh' ;; esac
        if [ "$status" = 0 ]; then
            printf 'PASS %s: %s\n' "$label" "$test_case"
        else
            printf 'FAIL %s: %s\n' "$label" "$test_case"
            cat "$case_dir/result" >&2
            if [ -f "$case_dir/stderr" ]; then cat "$case_dir/stderr" >&2; fi
            failed=1
        fi
    done
done
exit "$failed"
