#!/bin/sh
# Source setup on one Spark. No sudo, shell profile edits, or model downloads.
set -eu

fail() {
    printf '%s\n' "install: $*" >&2
    [ -z "${root-}" ] || printf 'Retry: sh "%s/scripts/install.sh"%s\n' "$root" "$([ "${launch-false}" = true ] && printf ' --launch')" >&2
    exit 1
}
if [ "${1-}" = --help ]; then
    printf '%s\n' 'Usage: sh scripts/install.sh [--launch]' \
        'Install pinned Rust if missing, build Vade, and prepare one local Spark.' \
        '--launch starts the engine and opens chat after installation.' \
        'Use an existing checkpoint under ~/models, or set VADE_MODELS.' \
        'CARGO_NET_OFFLINE=true requires already cached toolchains and build-std.'
    exit 0
fi
launch=false
case "$#:${1-}" in 0:) ;; 1:--launch) launch=true ;; *) fail 'unknown arguments. Next: sh scripts/install.sh --help' ;; esac
[ -n "${HOME-}" ] || fail 'HOME is missing. Next: run the installer from your normal login shell on the Spark.'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] ||
    fail 'run this on a Linux aarch64 DGX Spark. Next: open a terminal on your Spark and retry.'
[ -z "${VADE_CONFIG+x}" ] ||
    fail 'VADE_CONFIG selects a controller. Next: unset VADE_CONFIG, then rerun sh scripts/install.sh for this Spark.'
[ -z "${VADE_BUNDLE+x}" ] ||
    fail 'VADE_BUNDLE selects a bundle. Next: unset VADE_BUNDLE for this source install, or run the bundle vade launch.'
for tool in cc curl nvidia-smi; do
    command -v "$tool" >/dev/null 2>&1 ||
        fail "missing $tool. Next: ask your administrator to provide the prerequisites in INSTALL.md#step-1-prerequisites, then retry."
done
workspace=$(pwd -P) || fail 'current folder is unavailable. Next: open an existing project folder and retry.'
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P) ||
    fail 'cannot find the checkout. Next: run sh scripts/install.sh from the Vade checkout.'
cd "$root"
gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ||
    fail 'NVIDIA driver check failed. Next: ask your administrator to restore nvidia-smi, then retry.'
[ -z "$gpu_processes" ] ||
    fail 'the GPU is in use. Next: stop that workload through the launcher that started it, then retry.'
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
# That build needs the Rust build-std files; fetch only those here.
sysroot=$(rustup run "$device" rustc --print sysroot) && [ -n "$sysroot" ] ||
    fail "cannot find the $device sysroot. Next: rustup toolchain install $device --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda; then rerun this script."
printf '%s\n' 'Install: fetching the Rust build-std files for the GPU kernels (none when cached).'
cargo "+$device" fetch --locked --manifest-path "$sysroot/lib/rustlib/src/rust/library/Cargo.toml" ||
    fail 'Rust build-std fetch failed. Next: check access to https://index.crates.io and rerun the Retry command below; with CARGO_NET_OFFLINE=true, run it once online first.'
"$vade" install </dev/null ||
    fail "local install failed. Next: follow the diagnostic above, run \"$vade\" doctor, then retry."
printf 'Install complete. Start and chat:\n  "%s" launch\nCheck and start:\n  "%s" setup\nTo type vade by name in a new shell, run this there first:\n  export PATH="$HOME/.cargo/bin:%s:$PATH"\n' "$vade" "$vade" "$root/cli/target/release"
if [ "$launch" = true ]; then
    cd "$workspace" || fail 'your starting folder is unavailable. Next: open an existing project folder, then use the launch command above.'
    exec "$vade" launch
fi
