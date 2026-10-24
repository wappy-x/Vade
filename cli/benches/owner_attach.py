#!/usr/bin/env python3
"""Opt-in Mac mock attach benchmark; run on an idle host, outside correctness CI."""
import argparse
import os
from pathlib import Path
import re
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=20)
parser.add_argument("--target-seconds", type=float, default=1.0)
args = parser.parse_args()
if args.samples < 5 or args.target_seconds <= 0:
    parser.error("use at least five samples and a positive target")

cli = Path(__file__).resolve().parents[1]
cargo = Path.home() / ".cargo/bin/cargo"
environment = dict(os.environ, RUSTUP_TOOLCHAIN="1.99.0", CARGO_NET_OFFLINE="true")
command = [str(cargo), "test", "--offline", "--manifest-path", str(cli / "Cargo.toml"),
           "--bin", "vade", "launch::tests::resident_attach_with_real_subprocesses_reuses_saved_state",
           "--", "--exact", "--nocapture", "--test-threads=1"]
samples = []
# First invocation compiles and warms caches; only the measured attach interval
# is sampled. Each invocation verifies calls/state and runs three shell doubles.
for sample in range(args.samples + 1):
    result = subprocess.run(command, env=environment, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise SystemExit(result.stdout + result.stderr)
    match = re.search(r"Mac resident attach with three real subprocesses: ([0-9.]+) s", result.stdout)
    if match is None:
        raise SystemExit("Attach measurement missing from test output")
    if sample:
        samples.append(float(match.group(1)))
ordered = sorted(samples)
p95 = ordered[(95 * len(ordered) + 99) // 100 - 1]
print(f"Mock attach: n={len(samples)}, median={statistics.median(samples):.3f}s, p95={p95:.3f}s; target<{args.target_seconds:.3f}s")
raise SystemExit(0 if p95 < args.target_seconds else 1)
