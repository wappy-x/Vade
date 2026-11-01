#!/bin/sh
# Native Linux ARM64 release producer. Only system tools and the pinned Rust
# toolchains are used; the archive contains no toolchain or CUDA link stub.
set -eu

fail() { printf '%s\n' "bundle: $*" >&2; exit 1; }
if [ "${1-}" = --help ]; then
    printf '%s\n' 'Usage: sh scripts/make-bundle.sh [OUTPUT_DIR]' \
        'Build committed HEAD for Linux aarch64; OUTPUT_DIR must not exist (default: .local/bundles).' \
        '   or: sh scripts/make-bundle.sh --from-payload PAYLOAD_DIR --cli-dir CLI_DIR [OUTPUT_DIR]' \
        'Package already built binaries without a build (Linux or macOS with git, tar, gzip and od; the' \
        'payload commit must be in this checkout). PAYLOAD_DIR holds' \
        'build.json, source.tar and artifacts/engine/target; CLI_DIR holds vade, vade-harness and REVISION.' \
        'Writes vade-<version>-<sha7>-linux-aarch64-gb10.tar.gz and SHA256SUMS.'
    exit 0
fi
if [ "${1-}" = --from-payload ]; then
    # Package a payload that is already built (for example by a cross build), with no
    # build here. It runs on Linux or macOS: GNU tar and BSD tar each give the same
    # bytes on every run; MANIFEST holds per-file hashes that do not depend on tar.
    [ "$#" -ge 4 ] && [ "$#" -le 5 ] && [ "$3" = --cli-dir ] || fail 'expected --from-payload PAYLOAD_DIR --cli-dir CLI_DIR [OUTPUT_DIR]'
    for tool in git tar gzip od find sed sort grep head tail cut tr touch chmod cp mkdir mktemp mv rm wc dirname basename; do
        command -v "$tool" >/dev/null 2>&1 || fail "missing tool: $tool"
    done
    if command -v sha256sum >/dev/null 2>&1; then sum() { sha256sum "$@"; }
    elif command -v shasum >/dev/null 2>&1; then sum() { shasum -a 256 "$@"; }
    else fail 'missing tool: sha256sum or shasum'; fi
    digest() { sum "$1" | cut -d ' ' -f 1; }
    root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
    payload=$(CDPATH= cd -- "$2" && pwd -P) || fail 'payload folder is missing'
    clidir=$(CDPATH= cd -- "$4" && pwd -P) || fail 'CLI folder is missing'
    engine="$payload/artifacts/engine/target/release/flash-serve"
    ptx="$payload/artifacts/engine/target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx"
    for file in "$payload/build.json" "$payload/source.tar" "$engine" "$ptx" "$clidir/vade" "$clidir/vade-harness" "$clidir/REVISION"; do
        [ -f "$file" ] && [ ! -L "$file" ] || fail "missing payload file: $file"
    done
    revision=$(git get-tar-commit-id < "$payload/source.tar") || fail 'source.tar has no commit id'
    printf '%s\n' "$revision" | grep -Eqx '[0-9a-f]{40}' || fail 'source.tar has no commit id'
    grep -Fq "\"sha\": \"$revision\"" "$payload/build.json" || fail 'build.json names another commit than source.tar'
    [ "$(cat "$clidir/REVISION")" = "$revision" ] || fail 'CLI_DIR/REVISION names another commit than the payload'
    json() { sed -n "s/.*\"$1\": *\"\\([0-9a-f]*\\)\".*/\\1/p" "$payload/build.json"; }
    [ "$(json binary_sha256)" = "$(digest "$engine")" ] || fail 'flash-serve does not match build.json'
    [ "$(json ptx_sha256)" = "$(digest "$ptx")" ] || fail 'PTX does not match build.json'
    git -C "$root" cat-file -e "$revision^{commit}" 2>/dev/null || fail "commit $revision is not in this checkout; fetch it first"
    epoch=$(git -C "$root" show -s --format=%ct "$revision")
    date=$(TZ=UTC git -C "$root" show -s --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ "$revision")
    stamp=$(TZ=UTC git -C "$root" show -s --format=%cd --date=format-local:%Y%m%d%H%M.%S "$revision")
    version=$(tar -xOf "$payload/source.tar" cli/Cargo.toml | sed -n 's/^version = "\([0-9A-Za-z.+-]*\)"$/\1/p' | head -n 1)
    [ -n "$version" ] || fail 'cannot read the version from cli/Cargo.toml'
    short=$(printf '%.7s' "$revision")
    stem="vade-$version-$short-linux-aarch64-gb10"
    archive="$stem.tar.gz"
    # The PTX ISA version sets the oldest driver that can load the kernels. Add a row
    # (from the CUDA release notes) when the kernel toolchain moves to a newer ISA.
    isa=$(sed -n '1,40s/^\.version \([0-9.]*\)$/\1/p' "$ptx" | head -n 1)
    gpu=$(sed -n '1,40s/^\.target \(sm_[0-9a-z]*\).*$/\1/p' "$ptx" | head -n 1)
    case $isa in
        8.8) ptx_cuda=12.9 ptx_driver=575.51.03 ;;
        *) fail "no driver row for PTX ISA '$isa'; add it from the CUDA release notes" ;;
    esac
    [ "$gpu" = sm_121 ] || fail "PTX targets '$gpu', not the GB10 (sm_121)"
    output=${5:-"$root/.local/bundles"}
    [ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output already exists; choose a fresh output directory'
    mkdir -p -- "$(dirname -- "$output")"
    output="$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)/$(basename -- "$output")"
    [ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output already exists; choose a fresh output directory'
    work=$(mktemp -d "$(dirname -- "$output")/.bundle-publish.XXXXXXXX")
    trap 'rm -rf -- "$work"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    stage="$work/$stem"
    mkdir "$work/src" "$work/out" "$stage"
    (cd "$work/src" && tar -xf "$payload/source.tar" LICENSE NOTICE engine/LICENSE engine/NOTICE engine/licenses \
        engine/crates/tokenizer/UNICODE-NOTICE.txt engine/crates/tokenizer/QWEN-NOTICE.txt) || fail 'license files are missing from source.tar'
    mkdir -p "$stage/engine/crates/tokenizer"
    cp "$engine" "$stage/flash-serve"
    cp "$ptx" "$stage/flash.ptx"
    cp "$clidir/vade" "$clidir/vade-harness" "$stage/"
    cp "$work/src/LICENSE" "$work/src/NOTICE" "$stage/"
    cp "$work/src/engine/LICENSE" "$work/src/engine/NOTICE" "$stage/engine/"
    cp -R "$work/src/engine/licenses" "$stage/engine/licenses"
    cp "$work/src/engine/crates/tokenizer/UNICODE-NOTICE.txt" "$work/src/engine/crates/tokenizer/QWEN-NOTICE.txt" "$stage/engine/crates/tokenizer/"
    for binary in flash-serve vade vade-harness; do
        # ELF magic, 64-bit class, little endian, machine 183 (AArch64).
        head=$(od -An -tu1 -N20 "$stage/$binary" | tr -s ' \n' '  ' | sed 's/^ //; s/ $//')
        case $head in "127 69 76 70 2 1 "*" 183 0") ;; *) fail "$binary is not a Linux aarch64 ELF64 file" ;; esac
        for forbidden in "$payload" "$clidir" "$root" "$HOME"; do
            if grep -aFq -- "$forbidden" "$stage/$binary"; then fail "$binary contains a producer path"; fi
        done
    done
    grep -aFq libcuda.so.1 "$stage/flash-serve" || fail 'engine does not reference the installed CUDA driver SONAME'
    glibc=$(grep -aho 'GLIBC_2\.[0-9][0-9.]*' "$stage/flash-serve" "$stage/vade" "$stage/vade-harness" |
        sed 's/^GLIBC_//' | sort -u -t . -k1,1n -k2,2n -k3,3n | tail -n 1)
    [ -n "$glibc" ] || fail 'cannot read the glibc version the binaries need'
    cat > "$stage/launcher-bundle.conf" <<CONF
format=vade-launch-bundle-v1
target=aarch64-unknown-linux-gnu
engine_sha=$revision
cli_sha=$revision
harness_sha=$revision
engine_sha256=$(digest "$stage/flash-serve")
ptx_sha256=$(digest "$stage/flash.ptx")
harness_sha256=$(digest "$stage/vade-harness")
vade_sha256=$(digest "$stage/vade")
CONF
    cat > "$stage/README-BUNDLE" <<README
Vade $version ($revision), prebuilt for one GB10 system (a DGX Spark or
another GB10 machine).

This folder has the vade command, the inference engine (flash-serve), the GPU
kernels as PTX (flash.ptx) and the chat program (vade-harness). It has no
model weights.

You need Linux aarch64 with glibc $glibc or newer and an NVIDIA driver from
release 580 or newer (CUDA 13.0). MANIFEST lists every file
with its SHA-256 hash.

Install and start:

    ./vade install
    ./vade launch

./vade install checks the hashes in launcher-bundle.conf and copies the files
under ~/.vade (or VADE_HOME). Read INSTALL.md in the source repository for
the model download and for more Sparks.
README
    (cd "$stage" && find . -type f ! -name MANIFEST | sed 's|^\./||' | LC_ALL=C sort) > "$work/files"
    {
        printf '%s\n' format=vade-release-v1 "name=$stem" "version=$version" "sha=$revision" "date=$date" \
            target=gb10 arch=aarch64-unknown-linux-gnu "gpu=$gpu" "ptx_isa=$isa" "ptx_cuda=$ptx_cuda" \
            "ptx_min_driver=$ptx_driver" min_cuda=13.0 min_driver=580 "glibc=$glibc"
        while IFS= read -r file; do printf 'file=%s  %s\n' "$(digest "$stage/$file")" "$file"; done < "$work/files"
    } > "$stage/MANIFEST"
    for artifact in "$stage/flash.ptx" "$stage/launcher-bundle.conf" "$stage/MANIFEST"; do
        for forbidden in "$payload" "$clidir" "$root" "$HOME"; do
            if grep -aFq -- "$forbidden" "$artifact"; then fail "$(basename -- "$artifact") contains a producer path"; fi
        done
    done
    # Fixed modes, times and member order; no owner names or extended attributes.
    find "$stage" -type d -exec chmod 755 {} +
    find "$stage" -type f -exec chmod 644 {} +
    chmod 755 "$stage/flash-serve" "$stage/vade" "$stage/vade-harness"
    find "$stage" -exec env TZ=UTC touch -t "$stamp" {} +
    (cd "$work" && find "$stem" | LC_ALL=C sort) > "$work/members"
    tarcmd=tar
    command -v gtar >/dev/null 2>&1 && tarcmd=gtar
    if "$tarcmd" --version 2>/dev/null | grep -q 'GNU tar'; then
        (cd "$work" && "$tarcmd" --format=ustar --no-recursion --owner=0 --group=0 --numeric-owner \
            --mtime="@$epoch" -cf "$work/bundle.tar" -T "$work/members")
    else
        (cd "$work" && COPYFILE_DISABLE=1 "$tarcmd" --format=ustar --no-recursion --uid 0 --gid 0 --numeric-owner \
            --no-xattrs --no-mac-metadata --no-acls --no-fflags -cf "$work/bundle.tar" -T "$work/members")
    fi
    gzip -n -9 -c "$work/bundle.tar" > "$work/out/$archive"
    # Check the archive as an installer reads it: unpack, then compare every file hash.
    mkdir "$work/check"
    (cd "$work/check" && gzip -dc "$work/out/$archive" | tar -xf - && cd "$stem" &&
        sed -n 's/^file=//p' MANIFEST | sum -c - >/dev/null) || fail 'unpacked archive does not match MANIFEST'
    (cd "$work/out" && sum "$archive" > SHA256SUMS && sum -c SHA256SUMS >/dev/null) || fail 'archive checksum failed'
    [ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output appeared during packaging; leaving it untouched'
    mv "$work/out" "$output"
    printf 'bundle: %s\nbytes: %s\nsha256: %s\ntar: %s\n' "$output/$archive" \
        "$(wc -c < "$output/$archive" | tr -d ' ')" "$(digest "$output/$archive")" "$("$tarcmd" --version | head -n 1)"
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
