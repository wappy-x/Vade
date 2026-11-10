#!/bin/sh
# Shared JSON support. Rust parses JSON; curl supplies HTTPS and HTTP streaming.
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
if [ -z "${VADE_TOOLS:-}" ]; then
  cargo +1.99.0 build --quiet --offline --locked --manifest-path "$ROOT/tools/Cargo.toml" \
    --target-dir "$ROOT/tools/target" --bin vade-tools
  VADE_TOOLS=$ROOT/tools/target/debug/vade-tools
fi
field() { printf '%s' "$CAPABILITIES" | "$VADE_TOOLS" json-get "$1"; }
