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
for tool in sh bash; do
    command -v "$tool" >/dev/null 2>&1 || {
        printf 'FAIL setup: missing %s\n' "$tool"
        exit 1
    }
done

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
    for tool in sh dirname mktemp rm mkdir mv cp chmod tail sed head; do
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
    --query-compute-apps=pid) ;;
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
    cat > "$case_dir/bin/git" <<'EOF'
#!/bin/sh
set -eu
printf '%s\n' 'git clone' >> "$MATRIX_CALLS"
[ "$GIT_TERMINAL_PROMPT" = 0 ]
for target do :; done
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
EOF
    for tool in uname nvidia-smi curl cc git rustup cargo; do
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
    env -i PATH="$case_dir/bin" HOME="$case_dir/home" PREFIX="$case_dir/prefix" \
        TMPDIR="$case_dir/tmp" XDG_CONFIG_HOME="$case_dir/home/.config" \
        XDG_CACHE_HOME="$case_dir/home/.cache" XDG_DATA_HOME="$case_dir/home/.local/share" \
        CARGO_HOME="$case_dir/home/.cargo" RUSTUP_HOME="$case_dir/home/.rustup" \
        CARGO_NET_OFFLINE=true MATRIX_CASE="$case_dir" MATRIX_CALLS="$case_dir/calls" \
        MATRIX_OS="$host_os" MATRIX_ARCH="$host_arch" \
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
        checkout)
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
            grep -q '^cargo .* build --release --offline --locked ' "$case_dir/calls"
            grep -q '^cargo .* fetch --locked ' "$case_dir/calls"
            [ "$(tail -n 1 "$case_dir/calls")" = 'vade __install-here' ]
            grep -q '^Install complete. Next steps:' "$case_dir/stdout"
            [ -x "$case_dir/home/.vade/bin/vade" ]
            [ -x "$case_dir/home/.local/bin/vade" ]
            case $mode in
                checkout) reject_calls '^git clone$' ;;
                pipe) grep -q '^git clone$' "$case_dir/calls" ;;
            esac
            ;;
    esac
    reject_calls '^unexpected-'
    case $test_case in
        *-check|*-help|half-pipe|no-call|host-*)
            snapshot > "$case_dir/after"
            cmp "$case_dir/before" "$case_dir/after"
            ;;
    esac
    for profile in .profile .bashrc .zshrc; do
        [ "$(cat "$case_dir/home/$profile")" = 'Keep this profile.' ]
    done
    [ "$(cat "$case_dir/prefix/keep")" = 'Keep this prefix.' ]
)

failed=0
for shell in $shells; do
    for test_case in checkout-default pipe-default checkout-source pipe-source \
        checkout-check pipe-check checkout-help pipe-help half-pipe no-call \
        host-linux-x86_64 host-macos; do
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
exit "$failed"
