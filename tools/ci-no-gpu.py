#!/usr/bin/env python3
"""CI-only CUDA linkage: any device call aborts; never ship this library.

Run on a Linux host without CUDA after building the real stamped PTX. The
workflow scopes its linker/runtime search paths to this generated directory.
Device tests must use libtest ignore reasons, so accidental execution fails.
"""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / 'engine/target/ci-no-gpu'
out.mkdir(parents=True, exist_ok=True)
symbols = sorted({name for path in (root / 'engine/crates').rglob('*.rs')
                  for name in re.findall(r'\bfn\s+(cu[A-Z]\w*)\s*\(', path.read_text())})
if not symbols:
    raise RuntimeError('no CUDA declarations found')
source = out / 'cuda.rs'
source.write_text('\n'.join(
    '#[unsafe(no_mangle)]\npub extern "C" fn ' + name + '() -> i32 { '
    'panic!("device call in host CI: ' + name + '") }' for name in symbols))
subprocess.run(['rustc', '--edition=2024', '--crate-type=cdylib', '-C', 'panic=abort',
                str(source), '-o', str(out / 'libcuda.so')], check=True)
print(f'CI-only CUDA linkage: {len(symbols)} symbols; device calls abort')
