#!/bin/sh
# Download and verify a public, pinned model with Rust and curl.
# Usage: sh scripts/fetch-model.sh REPOSITORY FULL_REVISION MODEL_DIRECTORY
# Rust checks the remote file list, sizes and hashes before it writes MODEL_DIRECTORY.sha256.
# Partial downloads stay in MODEL_DIRECTORY.fetch for the next attempt.
set -eu
umask 077
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
if [ -z "${VADE_TOOLS:-}" ]; then
  cargo +1.99.0 build --quiet --release --offline --locked --manifest-path "$ROOT/tools/Cargo.toml" \
    --target-dir "$ROOT/tools/target" --bin vade-tools
  VADE_TOOLS=$ROOT/tools/target/release/vade-tools
fi
exec "$VADE_TOOLS" fetch-model "$@"
