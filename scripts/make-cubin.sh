#!/bin/sh
# Compile the exact kernel PTX into the GB10 image for a release build. It needs
# no GPU and no driver. The compiler is pinned by version and hash, and a change
# to it needs a new token id proof, so there is no setting for another compiler.
set -eu

version=13.0.88
archive_sha256=01b01e10aa2662ad1b3aeab3317151d7d6d4a650eeade55ded504f6b7fced18e
ptxas_sha256=a1941a04ca4fd233b2fbe50c625b1e72b3d5f79ebe80209a272c85482dfbb487
# The compiler's own defaults, written out. Without -m64 -O3 only the note that
# records the flags differs; every code section is the same. The PTX carries each
# .maxnreg, so no register limit is added here.
ptxas_args='--gpu-name=sm_121 -m64 -O3'
name=cuda_nvcc-linux-sbsa-$version-archive
url=https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-sbsa/$name.tar.xz

fail() { printf '%s\n' "cubin: $*" >&2; exit 1; }
if [ "${1-}" = --help ] || [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    printf '%s\n' 'Usage: sh scripts/make-cubin.sh PTX OUTPUT_DIR [ARCHIVE]' \
        "Compile PTX with ptxas $version on Linux aarch64; OUTPUT_DIR must not exist." \
        "ARCHIVE is a saved copy of $name.tar.xz; without it the script downloads the file." \
        'Writes flash.cubin, flash.cubin.conf and flash.ptx (the bytes the compiler read).'
    [ "${1-}" = --help ] && exit 0
    exit 2
fi
if command -v sha256sum >/dev/null 2>&1; then digest() { sha256sum "$1" | cut -d ' ' -f 1; }
elif command -v shasum >/dev/null 2>&1; then digest() { shasum -a 256 "$1" | cut -d ' ' -f 1; }
else fail 'missing tool: sha256sum or shasum'; fi

[ -f "$1" ] && [ ! -L "$1" ] || fail "missing PTX: $1"
ptx=$(CDPATH= cd -- "$(dirname -- "$1")" && pwd -P)/$(basename -- "$1")
[ ! -e "$2" ] && [ ! -L "$2" ] || fail 'output already exists; choose a fresh output directory'
archive=
if [ "$#" = 3 ]; then
    archive=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
fi
mkdir -p -- "$(dirname -- "$2")"
output=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
work=$(mktemp -d "$output.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -z "$archive" ]; then
    command -v curl >/dev/null 2>&1 || fail 'missing tool: curl'
    archive=$work/$name.tar.xz
    curl -fsSL --connect-timeout 30 --max-time 600 -o "$archive" "$url" || fail "cannot download $url"
fi
[ "$(digest "$archive")" = "$archive_sha256" ] || fail "$name.tar.xz does not have the pinned SHA-256"
mkdir "$work/toolkit" "$work/out"
# GNU tar and BSD tar detect the compression; busybox tar needs -J.
tar -xf "$archive" -C "$work/toolkit" "$name/bin/ptxas" 2>/dev/null ||
    tar -xJf "$archive" -C "$work/toolkit" "$name/bin/ptxas" || fail "cannot unpack $name.tar.xz"
compiler=$work/toolkit/$name/bin/ptxas
[ "$(digest "$compiler")" = "$ptxas_sha256" ] || fail 'ptxas does not have the pinned SHA-256'
chmod 755 "$compiler"
reported=$("$compiler" --version) || fail 'ptxas does not run here; run on Linux aarch64'
printf '%s\n' "$reported" | grep -q ", V$version\$" || fail "ptxas does not report version $version"

cp "$ptx" "$work/out/flash.ptx"
sed -n '1,40p' "$work/out/flash.ptx" | grep -q '^\.target sm_121$' || fail 'PTX does not target the GB10 (sm_121)'
before=$(digest "$work/out/flash.ptx")
# A relative name, so that no producer path can reach the image. Word splitting of
# ptxas_args is wanted here.
(cd "$work/out" && "$compiler" $ptxas_args flash.ptx -o flash.cubin) || fail 'ptxas failed'
# The image is valid only for the exact bytes that were compiled.
cmp -s "$ptx" "$work/out/flash.ptx" || fail 'PTX changed during the compile'
[ "$before" = "$(digest "$ptx")" ] || fail 'PTX changed during the compile'
[ -s "$work/out/flash.cubin" ] || fail 'ptxas wrote no image'
cat > "$work/out/flash.cubin.conf" <<CONF
format=vade-cubin-v1
target=sm_121
ptx_sha256=$before
cubin_sha256=$(digest "$work/out/flash.cubin")
min_driver_api=13000
ptxas_version=$version
ptxas_sha256=$ptxas_sha256
ptxas_args=$ptxas_args
CONF
[ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output appeared during the compile; leaving it untouched'
mv "$work/out" "$output"
printf 'cubin: %s\nbytes: %s\nptx_sha256: %s\nptxas: %s\n' "$output/flash.cubin" \
    "$(wc -c < "$output/flash.cubin" | tr -d ' ')" "$before" "$version"
