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

fail() {
    if [ "${check-false}" = true ]; then
        printf '%s\n' "Check: the install would stop here: $*"
        exit 0
    fi
    printf '%s\n' "install: $*" >&2
    if [ -n "${root-}" ]; then
        printf 'Retry: sh "%s/scripts/install.sh"%s\n' "$root" "${retry-}" >&2
    elif [ "${piped-false}" = true ]; then
        printf 'Retry: curl -fsSL https://vade.lol/install.sh | sh%s\n' "${retry:+ -s --$retry}" >&2
    fi
    exit 1
}
if [ "${1-}" = --help ]; then
    printf '%s\n' 'Usage: sh scripts/install.sh [--launch] [--source] [--check]' \
        '   or: curl -fsSL https://vade.lol/install.sh | sh -s -- [--launch] [--source] [--check]' \
        'Install the prebuilt bundle for this Spark, then run vade install. When the bundle does not fit' \
        'or cannot be downloaded, say why, install pinned Rust if missing, and build vade, the GPU' \
        'kernels, the engine and chat from this checkout instead. With no checkout, the source build' \
        'clones public main into a new folder in ~/.vade and runs the installer of that clone.' \
        'The model is not needed yet: the script prints the vade pull command for it.' \
        '--launch starts the engine and opens chat after installation; it needs the model.' \
        '--source always builds from source.' \
        '--check prints what the install would do, writes nothing, and exits 0.' \
        'Needs an idle NVIDIA GB10 GPU with a driver that supports CUDA 13.0 or newer.' \
        'VADE_DL_BASE sets the download location (default https://dl.vade.lol/vade); VADE_CHANNEL' \
        'sets the channel (default main).' \
        'CARGO_NET_OFFLINE=true requires already cached toolchains and build-std.'
    exit 0
fi
clone_url=https://github.com/wappy-x/vade.git
launch=false source=false check=false retry= root= piped=false
for argument in "$@"; do
    case $argument in
        --launch) launch=true retry="$retry --launch" ;;
        --source) source=true retry="$retry --source" ;;
        --check) check=true ;;
        *) check=false; fail 'unknown arguments. Next: run the installer with --help.' ;;
    esac
done
[ -n "${HOME-}" ] || fail 'HOME is missing. Next: run the installer from your normal login shell on the Spark.'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] ||
    fail 'run this on a Linux aarch64 DGX Spark. Next: open a terminal on your Spark and retry.'
[ -z "${VADE_CONFIG+x}" ] ||
    fail 'VADE_CONFIG selects a controller. Next: unset VADE_CONFIG, then rerun the installer for this Spark.'
[ -z "${VADE_BUNDLE+x}" ] ||
    fail 'VADE_BUNDLE selects a bundle. Next: unset VADE_BUNDLE and rerun this script, or run vade launch from that bundle.'
for tool in cc curl nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 ||
        fail "missing $tool. Next: ask your administrator to provide the prerequisites in INSTALL.md#step-1-prerequisites, then retry."
done
workspace=$(pwd -P) || fail 'current folder is unavailable. Next: open an existing project folder and retry.'
# A Vade checkout holds this file, the CLI, the engine and the harness.
checkout() {
    [ -f "$1/scripts/install.sh" ] && [ -f "$1/cli/Cargo.toml" ] && [ -f "$1/engine/Cargo.toml" ] &&
        [ -f "$1/harness/Cargo.toml" ]
}
# Piped, $0 names the shell, not this file: then no folder here is the source.
case $0 in */install.sh|install.sh)
    if [ -f "$0" ]; then
        root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) || root=
        [ -z "$root" ] || checkout "$root" || root=
    fi ;;
esac
if [ -n "$root" ]; then cd "$root"; else piped=true; fi
# Check the GPU and driver before any download; vade install checks them again.
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) ||
    fail 'NVIDIA driver check failed (nvidia-smi error above). Next: ask your administrator to restore nvidia-smi, then retry.'
case " $gpu " in *[!A-Za-z0-9]GB10[!A-Za-z0-9]*) ;; *) fail "this needs an NVIDIA GB10 GPU (a DGX Spark); nvidia-smi reports: $gpu. Next: run this script on a Spark." ;; esac
# Older drivers print "CUDA Version:"; newer drivers print "CUDA UMD Version:".
cuda=$(nvidia-smi | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\)\.[0-9].*/\2/p' | head -n 1)
[ -n "$cuda" ] && [ "$cuda" -ge 13 ] ||
    fail 'the NVIDIA driver must support CUDA 13.0 or newer (the "CUDA Version" or "CUDA UMD Version" that nvidia-smi prints). Next: ask your administrator to update the driver, then retry.'
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ||
    fail 'NVIDIA driver check failed (nvidia-smi error above). Next: ask your administrator to restore nvidia-smi, then retry.'
[ -z "$gpu_processes" ] || [ "$check" = true ] ||
    fail 'the GPU is in use. Next: stop that workload through the launcher that started it, then retry.'
[ -z "$gpu_processes" ] || printf '%s\n' 'Check: the GPU is in use; the install would stop until that workload stops.'

# The prebuilt bundle. Each check below runs before the next: the local tools, OS,
# glibc and driver are read first, then latest.json is downloaded and the local
# glibc and driver are compared with it, then the bundle is downloaded and checked.
# Any failure leaves the bundle and builds from source as before.
prebuilt_dir= why= download=
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
    for tool in tar gzip sha256sum getconf grep tr cut sort mv mkdir rm; do
        command -v "$tool" >/dev/null 2>&1 || { why="missing $tool"; return 1; }
    done
    vade_home=${VADE_HOME:-$HOME/.vade}
    case $vade_home in /*) ;; *) why='VADE_HOME is not an absolute path'; return 1 ;; esac
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
        --connect-timeout 30 --max-time 30 --max-filesize 65536 "$url" 2>&1) ||
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
        [ "$check" = false ] || printf 'Check: would use %s (Vade %s, already downloaded and checked), then run vade install.\n' "$prebuilt_dir" "$sha"
        return 0
    fi
    if [ "$check" = true ]; then
        printf 'Check: would download %s (sha256 %s), unpack it to %s, then run vade install.\n' "$base/$channel/$file" "$sum" "$prebuilt_dir"
        return 0
    fi
    printf 'Install: downloading the prebuilt bundle %s.\n' "$file"
    # Private like the folders vade makes: vade refuses a ~/.vade that others can read.
    (umask 077 && mkdir -p "$vade_home/bundles") || { why="cannot create $vade_home/bundles"; return 1; }
    download=$(mktemp -d "$vade_home/bundles/.download.XXXXXXXX") || { why="cannot create a folder in $vade_home/bundles"; return 1; }
    trap 'rm -rf "$download"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # A stalled transfer (under 1 KB/s for 60 s) or a file over 512 MiB stops the
    # download; the bundle is about 20 MB.
    curl -q --proto "$protocol" --tlsv1.2 --fail --location --silent --show-error \
        --connect-timeout 30 --speed-limit 1024 --speed-time 60 --max-filesize 536870912 \
        --output "$download/$file" "$base/$channel/$file" || { why="download of $base/$channel/$file failed"; return 1; }
    [ "$(sha256sum "$download/$file" | cut -d ' ' -f 1)" = "$sum" ] ||
        { why="$file does not match the sha256 in latest.json"; return 1; }
    members=$(gzip -dc "$download/$file" | tar -tf -) || { why="$file is not a readable archive"; return 1; }
    if printf '%s\n' "$members" | outside "$stem" ||
        printf '%s\n' "$members" | grep -e '\.\./' -e '/\.\.$' >/dev/null; then
        why="$file has files outside $stem/"; return 1
    fi
    # Only regular files and folders: no links, devices or other special members.
    types=$(gzip -dc "$download/$file" | tar -tvf - | cut -c 1 | sort -u | tr -d '\n') ||
        { why="$file is not a readable archive"; return 1; }
    case $types in -|d-|-d) ;; *) why="$file has members that are not files or folders"; return 1 ;; esac
    mkdir "$download/unpack" && gzip -dc "$download/$file" | (cd "$download/unpack" && tar -xf -) ||
        { why="cannot unpack $file"; return 1; }
    bundle_matches "$download/unpack/$stem" "$sha" || { why="$file does not match its MANIFEST"; return 1; }
    mv "$download/unpack/$stem" "$prebuilt_dir" || { why="cannot move the bundle to $prebuilt_dir"; return 1; }
    rm -rf "$download"
    trap - EXIT INT TERM
    printf 'Install: prebuilt Vade %s is in %s.\n' "$sha" "$prebuilt_dir"
}
if prebuilt; then
    [ "$check" = false ] || exit 0
    vade="$prebuilt_dir/vade"
    if "$vade" install </dev/null; then
        printf 'Install: installed prebuilt Vade %s.\n' "$sha"
    else
        why="\"$vade\" install failed (diagnostic above)"
        vade=
    fi
else
    vade=
    [ -z "${download-}" ] || rm -rf "$download"
    trap - EXIT INT TERM
fi
# With no checkout: clone public main into a new folder, then run the installer
# of the clone, which holds the pinned tools of that source. The folder is in
# the Vade home, not in /tmp: the vade program stays in the clone, and vade
# install and vade setup read the clone, so it must outlive a reboot.
clone_source() {
    vade_home=${VADE_HOME:-$HOME/.vade}
    case $vade_home in /*) ;; *) fail 'VADE_HOME is not an absolute path. Next: unset VADE_HOME or set it to a full path, then retry.' ;; esac
    if [ "$check" = true ]; then
        printf 'Check: would clone %s into a new folder in %s, then build from source there.\n' "$clone_url" "$vade_home"
        exit 0
    fi
    command -v git >/dev/null 2>&1 ||
        fail 'missing git, which the source build needs to get the Vade source. Next: ask your administrator to install git, then retry.'
    # Private like the folders vade makes: vade refuses a ~/.vade that others can read.
    (umask 077 && mkdir -p "$vade_home") ||
        fail "cannot create $vade_home. Next: check that you own its parent folder, then retry."
    clone=$(mktemp -d "$vade_home/source.XXXXXXXX") ||
        fail "cannot create a folder in $vade_home. Next: check its permissions and disk space, then retry."
    trap 'rm -rf "$clone"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf 'Install: cloning %s into %s.\n' "$clone_url" "$clone"
    # A transfer under 1 KB/s for 60 s stops; git never asks for a password.
    GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 \
        clone --quiet --depth 1 "$clone_url" "$clone" </dev/null ||
        fail "cannot clone $clone_url (git error above). Next: check that this Spark reaches github.com, then retry."
    checkout "$clone" ||
        fail "the clone of $clone_url is not a Vade checkout. Next: retry; if it fails again, report it with the Vade version."
    trap - EXIT INT TERM
    # Standard input is this script, so the clone's installer reads the terminal
    # or nothing. A build failure there keeps the clone and prints its own Retry line.
    if [ "$launch" = true ] && (: </dev/tty) 2>/dev/null; then
        exec sh "$clone/scripts/install.sh" --source --launch </dev/tty
    fi
    exec sh "$clone/scripts/install.sh" --source </dev/null
}
# The source build below is the same as before the prebuilt bundle existed.
if [ -z "$vade" ]; then
printf 'Prebuilt: %s; building from source.\n' "$why"
[ "$piped" = false ] || clone_source
if [ "$check" = true ]; then
    printf '%s\n' "Check: would install the pinned Rust toolchains if missing, build vade in $root/cli/target," \
        'fetch the Rust build-std files, then run vade install to build the GPU kernels, the engine and chat.'
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
    trap 'rm -f "$installer"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf '%s\n' 'Install: downloading the official Rust installer (no shell profile changes).'
    curl -q --proto '=https' --tlsv1.2 --fail --location --show-error --silent \
        --output "$installer" https://sh.rustup.rs ||
        fail 'Rust installer download failed. Next: check access to https://sh.rustup.rs and rerun the Retry command below.'
    sh "$installer" -y --default-toolchain none --no-modify-path ||
        fail 'Rust installation failed. Next: read the Rust error above, fix it, then rerun the Retry command below.'
    rm -f "$installer"
    trap - EXIT INT TERM
fi
host=1.99.0
device=nightly-2026-10-01
if ! rustup run "$host" rustc --version >/dev/null 2>&1; then
    online "Rust $host"
    rustup toolchain install "$host" --profile minimal --no-self-update ||
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
        --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda ||
        fail 'GPU compiler setup failed. Next: finish the pinned GPU toolchain command in INSTALL.md#install-rust-source-builds-only, then rerun this script.'
fi
printf '%s\n' 'Install: building the vade command.'
CARGO_TARGET_DIR="$root/cli/target" cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml ||
    fail 'CLI build failed. Next: fix the compiler error above (including any missing linker), then rerun the Retry command below.'
vade="$root/cli/target/release/vade"
# vade install checks this Spark and builds the GPU kernels once, offline.
# It needs no model on disk; vade launch saves defaults once the model is there.
# That build needs the Rust build-std files; fetch only those here.
sysroot=$(rustup run "$device" rustc --print sysroot) && [ -n "$sysroot" ] ||
    fail "cannot find the $device sysroot. Next: rustup toolchain install $device --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda; then rerun this script."
printf '%s\n' 'Install: fetching the Rust build-std files for the GPU kernels (none when cached).'
cargo "+$device" fetch --locked --manifest-path "$sysroot/lib/rustlib/src/rust/library/Cargo.toml" ||
    fail 'Rust build-std fetch failed. Next: check access to https://index.crates.io and rerun the Retry command below; with CARGO_NET_OFFLINE=true, run it once online first.'
"$vade" install </dev/null ||
    fail "local install failed. Next: follow the diagnostic above, run \"$vade\" doctor, then retry."
fi
# One folder stays on PATH across versions: a small script that runs the build
# just installed, rewritten on each install. Not a copy or a link: the real
# program must run from its own folder to find its bundle or checkout.
vade_home=${VADE_HOME:-$HOME/.vade}
bin_path=${VADE_HOME:+$VADE_HOME/bin}
quoted=$(printf '%s\n' "$vade" | sed "s/'/'\\\\''/g")
[ ! -d "$vade_home/bin/vade" ] && (umask 077 && mkdir -p "$vade_home/bin" &&
    printf '#!/bin/sh\nexec '\''%s'\'' "$@"\n' "$quoted" > "$vade_home/bin/vade.new" &&
    chmod 700 "$vade_home/bin/vade.new" && mv -f "$vade_home/bin/vade.new" "$vade_home/bin/vade") ||
    fail "cannot write $vade_home/bin/vade. Next: check that path and its permissions, then retry."
# A second script for new shells: the stock Ubuntu ~/.profile puts ~/.local/bin on
# PATH at login when the folder exists. No shell profile changes, no link, and
# never over a file that is not ours. It names ~/.vade, so a named VADE_HOME skips it.
shim="$HOME/.local/bin/vade"
shim_marker='# Written by the Vade installer. vade-path-shim-v1'
shim_written=false
if [ -z "${VADE_HOME-}" ] || [ "$VADE_HOME" = "$HOME/.vade" ]; then
    if { [ ! -e "$shim" ] && [ ! -L "$shim" ]; } ||
        { [ -f "$shim" ] && [ ! -L "$shim" ] && [ "$(sed -n 2p "$shim" 2>/dev/null)" = "$shim_marker" ]; }; then
        if (mkdir -p "$HOME/.local/bin" &&
            printf '#!/bin/sh\n%s\nexec "$HOME/.vade/bin/vade" "$@"\n' "$shim_marker" > "$shim.new" &&
            chmod 700 "$shim.new" && mv -f "$shim.new" "$shim") 2>/dev/null; then
            shim_written=true
        else
            rm -f "$shim.new" 2>/dev/null || true
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
    'vade launch starts the engine and opens chat.' \
    ''
found=$(command -v vade 2>/dev/null || true)
path_line="export PATH=\"${bin_path:-\$HOME/.vade/bin}:\$PATH\""
if [ "$found" = "$vade_home/bin/vade" ] || { [ "$shim_written" = true ] && [ "$found" = "$shim" ]; }; then
    printf '%s\n' 'vade is on your PATH.'
elif [ "$shim_written" = true ]; then
    printf 'Open a new terminal, or run: %s\n' "$path_line"
else
    printf 'Run: %s\n' "$path_line"
fi
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
