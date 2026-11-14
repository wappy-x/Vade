#!/bin/sh
# Purpose: reproduce a published vade bench number (one Spark, drafting ON: C1 45.466 and C8 185.144 tok/s).
# Run:     sh examples/bench.sh              (measures the running engine; about 2 minutes)
#          sh examples/bench.sh --self-test  (no engine or GPU: checks the client against a built-in mock)
# Needs:   vade on PATH (cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml),
#          and for a real run an engine started with vade launch or vade up at the default settings.
# Expected output: see "Reproduce a benchmark number" in examples/README.md.
# The reference is the zero-flag run in the receipt of the one-Spark v0.1.0 drafting-ON card:
# bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/README.txt
set -eu
if [ "${1:-}" = --self-test ]; then
  exec vade bench --self-test
fi

# Zero flags is the published recipe: the recorded endpoint, the model /health names,
# the bundled prompt, C1 C2 C4 C8 C16 once each, exactly 256 greedy tokens per chat.
# VADE_API_URL adds --url, for an engine that no local installation recorded.
OUT=$(mktemp -d)
set -- --output "$OUT/vade-bench.json"
if [ -n "${VADE_API_URL:-}" ]; then set -- --url "$VADE_API_URL" "$@"; fi
vade bench "$@"
echo
echo "Published (1 Spark, v0.1.0): C1 45.466 | C2 79.747 | C4 143.598 | C8 185.144 | C16 189.042 tok/s"
echo "Receipt: $OUT/vade-bench.json, table: $OUT/vade-bench.txt"
