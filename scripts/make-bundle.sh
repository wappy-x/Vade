#!/bin/sh
# Native Linux ARM64 release producer. Only system tools and the pinned Rust
# toolchains are used; the archive contains no toolchain or CUDA link stub.
set -eu

fail() { printf '%s\n' "bundle: $*" >&2; exit 1; }
if [ "${1-}" = --help ]; then
    printf '%s\n' 'Usage: sh scripts/make-bundle.sh [OUTPUT_DIR]' \
        'Build committed HEAD for Linux aarch64; OUTPUT_DIR must not exist (default: .local/bundles).'
    exit 0
fi
[ "$#" -le 1 ] || fail 'expected at most one output directory'
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] || fail 'run on Linux aarch64 (no GPU required)'
for tool in git cargo rustup cc readelf tar zstd sha256sum find sed sort date mktemp \
    grep cut cp chmod mkdir mv rm rmdir wc tr dirname basename; do
    command -v "$tool" >/dev/null 2>&1 || fail "missing build tool: $tool"
done
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
[ -z "$(git status --porcelain --untracked-files=normal)" ] || fail 'commit or remove source changes before building a release'
revision=$(git rev-parse --verify HEAD)
short=$(printf '%.7s' "$revision")
epoch=$(git show -s --format=%ct "$revision")
archive="vade-launch-bundle-v1-$short-aarch64.tar.zst"
output=${1:-"$root/.local/bundles"}
mkdir -p -- "$(dirname -- "$output")"
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
output="$output_parent/$(basename -- "$output")"
[ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output already exists; choose a fresh output directory'
# A sibling lock serializes producers targeting this directory. Stage on the
# destination filesystem so one rename publishes the archive and checksum.
lock="$output.lock"
mkdir -- "$lock" 2>/dev/null || fail 'output is reserved by another producer; choose a fresh output directory'
work= publication=
cleanup() {
    [ -z "$work" ] || rm -rf -- "$work"
    [ -z "$publication" ] || rm -rf -- "$publication"
    rmdir -- "$lock"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
publication=$(mktemp -d "$output_parent/.bundle-publish.XXXXXXXX")
mkdir -p "$root/.local"
work=$(mktemp -d "$root/.local/bundle-build.XXXXXXXX")
started=$(date +%s)
printf 'build_directory: %s\n' "$work"

# Build an immutable tracked snapshot, never stray files or cached binaries in
# the working tree. REVISION is the engine archive provenance stamp.
mkdir "$work/source" "$work/payload" "$work/link-only"
git archive --format=tar "$revision" > "$work/source.tar"
tar -xf "$work/source.tar" -C "$work/source"
source="$work/source"
printf '%s\n' "$revision" > "$source/engine/REVISION"
unset RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS \
    CARGO_BUILD_TARGET RUSTC RUSTDOC CARGO KERNEL_BIN SPARK_PTX_VARIANT \
    SPARK_LAYER_PTX SPARK_CUBIN LD_LIBRARY_PATH
export CARGO_TARGET_DIR="$work/target" RUSTUP_AUTO_INSTALL=0 RUSTUP_TOOLCHAIN=1.99.0
export RUSTFLAGS='-D warnings'
# Remapping diagnostics alone does not change env!("CARGO_MANIFEST_DIR").
# Sanitize that compile-time value separately; Cargo still supplies real paths
# when running build scripts and cargo-run tools. Append remaps in the wrapper
# so the kernel builder cannot discard them when setting its own RUSTFLAGS.
export VADE_BUNDLE_SOURCE="$source" VADE_BUNDLE_BUILD="$work"
cat > "$work/rustc-portable" <<'EOF'
#!/bin/sh
set -eu
case ${CARGO_MANIFEST_DIR-} in
    "$VADE_BUNDLE_SOURCE"/*) export CARGO_MANIFEST_DIR="./${CARGO_MANIFEST_DIR#"$VADE_BUNDLE_SOURCE"/}" ;;
esac
case ${CARGO_MANIFEST_PATH-} in
    "$VADE_BUNDLE_SOURCE"/*) export CARGO_MANIFEST_PATH="./${CARGO_MANIFEST_PATH#"$VADE_BUNDLE_SOURCE"/}" ;;
esac
exec "$@" "--remap-path-prefix=$HOME=/build-home" \
    "--remap-path-prefix=$VADE_BUNDLE_BUILD=/vade-build" \
    "--remap-path-prefix=$VADE_BUNDLE_SOURCE=."
EOF
chmod 755 "$work/rustc-portable"
export RUSTC_WRAPPER="$work/rustc-portable"

# flash-serve links the CUDA driver but does not execute it during a build.
# Supply only its declared symbols for the linker, with the real driver's
# SONAME. No runtime search path is added; Spark loads its installed driver.
find "$source/engine/crates" -type f -name '*.rs' -exec \
    sed -n 's/.*fn \(cu[A-Z][A-Za-z0-9_]*\)[[:space:]]*(.*/\1/p' {} + \
    | sort -u > "$work/cuda-symbols"
[ -s "$work/cuda-symbols" ] || fail 'no CUDA driver declarations found'
printf '#include <stdlib.h>\n' > "$work/link-only/cuda.c"
while IFS= read -r symbol; do
    printf 'int %s(void) { abort(); }\n' "$symbol" >> "$work/link-only/cuda.c"
done < "$work/cuda-symbols"
cc -shared -fPIC -Wl,-soname,libcuda.so.1 -o "$work/link-only/libcuda.so" "$work/link-only/cuda.c"
export LIBRARY_PATH="$work/link-only"

cd "$source/engine"
# The first build-std may fetch Rust compiler components. Set
# CARGO_NET_OFFLINE=true when these are already cached (as on a Spark).
cargo +1.99.0 kernels
cargo +1.99.0 build --release --offline --locked -p spark-tp --bin flash-serve
cd "$source/cli"
cargo +1.99.0 build --release --offline --locked --bin vade
cd "$source/harness"
cargo +1.99.0 build --release --offline --locked --bin vade-harness

payload="$work/payload"
cp "$CARGO_TARGET_DIR/release/flash-serve" "$payload/flash-serve"
cp "$CARGO_TARGET_DIR/release/vade" "$payload/vade"
cp "$CARGO_TARGET_DIR/release/vade-harness" "$payload/vade-harness"
cp "$source/engine/target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx" "$payload/flash.ptx"
chmod 755 "$payload/flash-serve" "$payload/vade" "$payload/vade-harness"
chmod 644 "$payload/flash.ptx"
cp "$source/LICENSE" "$source/NOTICE" "$payload/"
mkdir -p "$payload/engine/crates/tokenizer"
cp "$source/engine/LICENSE" "$source/engine/NOTICE" "$payload/engine/"
cp -R "$source/engine/licenses" "$payload/engine/licenses"
cp "$source/engine/crates/tokenizer/UNICODE-NOTICE.txt" \
    "$source/engine/crates/tokenizer/QWEN-NOTICE.txt" "$payload/engine/crates/tokenizer/"
for binary in flash-serve vade vade-harness; do
    readelf -h "$payload/$binary" > "$work/elf-header"
    grep -q 'Class:.*ELF64' "$work/elf-header" || fail "$binary is not ELF64"
    grep -q 'Machine:.*AArch64' "$work/elf-header" || fail "$binary is not ARM64"
    readelf -d "$payload/$binary" > "$work/elf-dynamic"
    if grep -Eq '\((RPATH|RUNPATH)\)' "$work/elf-dynamic"; then
        fail "$binary contains a runtime library search path"
    fi
done
readelf -d "$payload/flash-serve" > "$work/engine-dynamic"
grep -Fq 'Shared library: [libcuda.so.1]' "$work/engine-dynamic" || fail 'engine does not reference the installed CUDA driver SONAME'
cd "$payload"
digest() { sha256sum "$1" | cut -d ' ' -f 1; }
cat > launcher-bundle.conf <<EOF
format=vade-launch-bundle-v1
target=aarch64-unknown-linux-gnu
engine_sha=$revision
cli_sha=$revision
harness_sha=$revision
engine_sha256=$(digest flash-serve)
ptx_sha256=$(digest flash.ptx)
harness_sha256=$(digest vade-harness)
vade_sha256=$(digest vade)
EOF
chmod 644 launcher-bundle.conf
# Refuse the release if any compiler or generated input retained producer paths.
# This also catches future env! paths that diagnostic remapping cannot sanitize.
for artifact in flash-serve flash.ptx vade vade-harness launcher-bundle.conf; do
    for forbidden in "$work" "$(basename -- "$work")" "$root" "$HOME"; do
        [ -z "$forbidden" ] || if grep -aFq -- "$forbidden" "$artifact"; then
            fail "$artifact contains a producer path"
        fi
    done
done
# Explicit members keep build inputs, link stubs and local paths out of releases.
tar --format=gnu --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner \
    -cf "$work/bundle.tar" LICENSE NOTICE engine flash-serve flash.ptx launcher-bundle.conf vade vade-harness
zstd -q -T1 -10 "$work/bundle.tar" -o "$publication/$archive"
cd "$publication"
sha256sum "$archive" > SHA256SUMS
sha256sum -c SHA256SUMS
# Check immediately before renaming, and let GNU mv refuse any intervening
# destination (including an empty directory or symlink). -T prevents nesting.
[ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output appeared during build; leaving it untouched'
mv -T -n -- "$publication" "$output"
[ ! -e "$publication" ] || fail 'output appeared during publication; leaving it untouched'
publication=
finished=$(date +%s)
printf 'bundle: %s\nbytes: %s\nbuild_seconds: %s\n' \
    "$output/$archive" "$(wc -c < "$output/$archive" | tr -d ' ')" "$((finished - started))"
