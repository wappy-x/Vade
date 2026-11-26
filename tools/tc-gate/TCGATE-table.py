#!/usr/bin/env python3
"""Usage: TCGATE-table.py A B [--extra RUN] [--tile RUN] [--small RUN] [--quality RUN]
A and B can join folders with +. --extra adds turns/tool evidence outside the fresh four-load repeat.
Exit 0 means served exactness and both smoke controls passed; missing evidence exits 3.
Word sweeps for the switched format and subtle sum-order controls still need separate receipts.
"""
import argparse, importlib.util, os, pathlib, subprocess, sys
spec = importlib.util.spec_from_file_location('verdict', pathlib.Path(__file__).with_name('TCGATE-verdict.py'))
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('a'); p.add_argument('b')
for n in ('extra', 'tile', 'small', 'quality'): p.add_argument('--' + n)
x = p.parse_args(); a, b = v.Run(x.a), v.Run(x.b)
full = v.Run(x.a + ('+' + x.extra if x.extra else '')); rows = []; ok = True

def row(name, proves, tags, bad):
    global ok
    state = 'NOT PROVEN' if bad is None else ('PASS' if not bad else 'FAIL: ' + bad[0])
    ok &= bad == []
    rows.append((name, proves, tags, state))
row('1', 'fresh-server repeat, rule 1', a.name + ' / ' + b.name, (v.part1(a, b, []) + v.inventory(a) + v.inventory(b)) if b.commit and not v.inventory(a) and not v.inventory(b) else None)
for n, fn, proves in ((2, v.part2, 'served groups, rules 2/5; available word sweeps'), (3, v.part3, 'continuation and cache/cold, rule 3'), (4, v.part4, 'strict/plain depths 1/3/7, rule 4')):
    row(str(n), proves, full.name, fn(full, []))
for name, path, nums in (('tile smoke control', x.tile, (1, 3)), ('small smoke control', x.small, (2, 4))):
    if not path: row(name, 'broad fault must change tokens', '-', None); continue
    r = v.Run(path); failures = list(r.problems)
    expected = 'tile' if name.startswith('tile') else 'small'
    if r.control != expected: failures.append(f'wrong control identity: {r.control}, expected {expected}')
    for n in nums:
        bad = v.part1(full, r, [], control=True) if n == 1 else {2:v.part2,3:v.part3,4:v.part4}[n](r, [])
        differences = [f for f in bad or [] if 'first difference at token' in f or 'DIFFER' in f]
        if not differences: failures.append(f'part {n} MISSED (no token difference)')
    row(name, 'parts ' + '/'.join(map(str, nums)) + ' detect planted magnitude fault', r.name, failures)
print('| Part | What it proves | Run tag | Result |\n|---|---|---|---|')
for fields in rows: print('| ' + ' | '.join(str(z).replace('|', '/') for z in fields) + ' |')
print('\nRule 6 and full rule 2 require format-specific device-to-twin word sweeps. Smoke controls do not prove sensitivity to a different sum order.')
if x.quality:
    root = pathlib.Path(os.environ.get('VADE_NIGHT_ROOT', pathlib.Path(__file__).resolve().parent.parent))
    cmd = [sys.executable, str(pathlib.Path(__file__).with_name('TCGATE-quality-report.py')), str(root/'quality-tc/histories.json'), str(root/'quality-tc/maker-reference.json'), x.quality, '--free', str(root/'baseline/ids.json')] + full.dirs
    q = subprocess.run(cmd, capture_output=True, text=True); print('\n' + q.stdout)
    ok &= q.returncode == 0 and 'QUALITY VALID: yes' in q.stdout
else: print('\nQuality: NOT RUN.')
print('\nSERVED CHECKS AND SMOKE CONTROLS: ' + ('PASS' if ok else 'INCOMPLETE OR FAILED'))
print('LANDING: NOT PROVEN by this table. Format word/twin, sum-order control, path and quality review remain required.')
sys.exit(0 if ok else 3)
