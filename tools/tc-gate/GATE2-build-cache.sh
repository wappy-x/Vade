#!/bin/bash
# GATE2 custom build cache: server, resident-decode-check and image.
# Derived from the device-proven KITF helper; custom proof is still required.
# Control builds use the original uncached TCGATE recipe, never this cache.
set -Eeuo pipefail
trap 'echo "CACHE_BUILD_FAIL line=$LINENO"' ERR
label=$1; commit=$2; root=$HOME/${3:?box folder}
if [ -n "${4:-}${VADE_GATE_CONTROL:-}" ]; then echo CACHE_CONTROL_REFUSED; exit 2; fi
[ -z "${4:-}" ] && [ -z "${VADE_GATE_CONTROL:-}" ] || { echo CACHE_CONTROL_REFUSED; exit 2; }
export PATH="$HOME/.cargo/bin:$PATH" RUSTUP_AUTO_INSTALL=0 CARGO_NET_OFFLINE=true RUSTFLAGS='-D warnings'
case "$commit" in *[!0-9a-f]*|'') echo REVISION_MISMATCH; exit 1;; esac
[ ${#commit} = 40 ] || exit 1
cd "$root"
t0=$(date +%s)
archive=$(sha256sum "src-$label.tar" | cut -d' ' -f1)
# Match the engine builder: PATH assembler first, then the CUDA installation.
ptxas_bin=$(command -v ptxas || printf '%s\n' /usr/local/cuda/bin/ptxas)
# Recipe + archive content + actual installed compiler identities, not commit dates.
recipe=$( { sha256sum "$0" | cut -d' ' -f1; rustc +1.99.0 -vV || exit 1; rustc +nightly-2026-10-01 -vV || exit 1; "$ptxas_bin" --version || exit 1; } | sha256sum | cut -d' ' -f1)
cache=$HOME/.cache/vade-gate-build-v1/$recipe
mkdir -p "$cache"
# Lock only builds using this cache; held by this process and its children.
exec 9>"$cache/build.lock"
flock -w 120 9 || { echo CACHE_LOCK_FAIL; exit 1; }
entry=$cache/$commit-$archive
k=$root/kit-$label
mkdir -p "$k"
if [ -f "$entry/SHA256SUMS" ] && [ "$(cat "$entry/BUILD-REVISION")" = "$commit" ] && (cd "$entry" && sha256sum --status -c SHA256SUMS); then
  cp "$entry/flash-serve" "$entry/resident-decode-check" "$entry/flash.ptx" "$entry/BUILD-REVISION" "$entry/SHA256SUMS" "$k/"
  echo "CACHE_HIT commit=$commit archive=$archive recipe=$recipe"
else
  echo "CACHE_MISS commit=$commit archive=$archive recipe=$recipe"
  tar xf "src-$label.tar"
  cd "src-$label/engine"
  grep -qx "$commit" REVISION || { echo REVISION_MISMATCH; exit 1; }
  target=$cache/target
  mkdir -p "$target"
  ln -s "$target" target
  export CARGO_TARGET_DIR=$target
  # Archives preserve commit times, including old commits. Never trust those times.
  # Discard a future-dated target; otherwise force every source newer than prior outputs.
  touch "$root/cache-now"
  if [ -n "$(find "$target" -type f -newer "$root/cache-now" -print -quit)" ]; then
    echo CACHE_FUTURE_TARGET_REBUILD
    rm -rf "$target"; mkdir -p "$target"
  fi
  touch "$root/cache-build-start"
  sleep 1
  # The archive identity marks sources newer than REVISION as dirty. Give every
  # untouched archived file exactly ONE fresh timestamp, including REVISION.
  touch "$root/cache-source-stamp"
  find .. -type f -exec touch -r "$root/cache-source-stamp" {} +
  build(){
    local t
    t=$(date +%s)
    cargo +1.99.0 kernels > "$root/build-$label-kernels.log" 2>&1 || { echo KERNELS_FAIL; tail -15 "$root/build-$label-kernels.log"; return 1; }
    echo "BUILD_PHASE kernels seconds=$(( $(date +%s)-t ))"
    t=$(date +%s)
    cargo +1.99.0 build --offline --locked --release -p spark-tp --bin flash-serve --bin resident-decode-check > "$root/build-$label-bins.log" 2>&1 || { echo BINS_FAIL; tail -15 "$root/build-$label-bins.log"; return 1; }
    echo "BUILD_PHASE bins seconds=$(( $(date +%s)-t ))"
  }
  build
  image=$target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx
  binary=$target/release/flash-serve
  checker=$target/release/resident-decode-check
  if ! { [ "$image" -nt "$root/cache-build-start" ] && [ "$binary" -nt "$root/cache-build-start" ] && [ "$checker" -nt "$root/cache-build-start" ]; }; then
    echo CACHE_STALE_TARGET_REBUILD
    rm -rf "$target"; mkdir -p "$target"
    build
  fi
  [ "$image" -nt "$root/cache-build-start" ] && [ "$binary" -nt "$root/cache-build-start" ] && [ "$checker" -nt "$root/cache-build-start" ] || { echo STALE_BUILD_FAIL; exit 1; }
  cp "$binary" "$k/flash-serve"; cp "$checker" "$k/resident-decode-check"; cp "$image" "$k/flash.ptx"
  printf '%s\n' "$commit" > "$k/BUILD-REVISION"
  (cd "$k" && sha256sum flash-serve resident-decode-check flash.ptx BUILD-REVISION > SHA256SUMS)
  # An incomplete entry is never reusable. Publish the manifest last under the lock.
  mkdir -p "$entry"
  cp "$k/flash-serve" "$k/resident-decode-check" "$k/flash.ptx" "$k/BUILD-REVISION" "$entry/"
  cp "$k/SHA256SUMS" "$entry/SHA256SUMS"
fi
(cd "$k" && sha256sum --status -c SHA256SUMS && cat SHA256SUMS)
echo "CONTROL=none"
echo "BUILD_COMPLETE $label secs=$(( $(date +%s)-t0 ))"
