#!/usr/bin/env python3
"""TCGATE-quality-report.py (2026-11-15, seat TCGATE). Runs on the Mac. Stdlib only.

  TCGATE-quality-report.py HISTORIES.json MAKER_REFERENCE.json QUALITY_RUN [--free BASELINE_IDS.json GATE_RUN...]

QUALITY_RUN is a run folder of TCGATE-gate.sh with the part "quality" (its QL/out-QL files).
Measure 1, against main: at each answer position of main's own answers to the four loads, does this build pick main's
  token, given main's history (same-history top-1)? Share by load and in total. On main itself this must be 100%.
  With --free: the first position where this build's free-running answer leaves main's, from gate run folders.
Measure 2, against the maker's own engine: at each of the 208 answer positions of the four gate cells of 2026-11-10,
  does this build's top token equal the maker's top token on the same history? Main on 2026-11-10: 198 of 208.
A history that did not finish, a cold check that differs, or a chat prompt whose free run does not start like the
history is printed as NOT VALID and makes the last line say so."""
import json, os, sys, glob
main_run = False
args = sys.argv[1:]; free = []
if '--free' in args: i = args.index('--free'); free = args[i + 1:]; args = args[:i]
hfile, rfile, run = args[:3]
main_run = 'A: build of 1508efce96fd732edf93704c95c34b813afcebaf started' in open(os.path.join(run, 'drive.log')).read()
hist = {h['name']: h for h in json.load(open(hfile))}; ref = json.load(open(rfile))
res = {}
for p in sorted(glob.glob(os.path.join(run, 'QL', 'out-QL', '*.json'))):
    try: r = json.load(open(p))
    except Exception: continue
    if isinstance(r, dict) and 'top' in r and 'name' in r: res[r['name']] = r
print(f"QUALITY RUN {os.path.basename(run.rstrip('/'))}: histories with results {len(res)} of {len(hist)}")
bad = []
def valid(name, r, main_run):
    h = hist[name]; notes = []
    if r['done'] != len(h['history']): notes.append(f"only {r['done']} of {len(h['history'])} positions")
    for k, c in r.get('cold', {}).items():
        if not c.get('equal'): notes.append(f"cold check at position {k}: cold {c.get('cold')} against kept {c.get('kept')}")
    if len(r.get('cold', {})) < 2: notes.append("cold checks missing")
    if main_run and 'chat_request' in h and r.get('free8_equals_history') is not True: notes.append('free 8 tokens do not match history')
    if r.get('errors'): notes.append('; '.join(r['errors'])[:160])
    return notes
# measure 1
tot = {}; first = {}
print("MEASURE 1: same-history top-1 against main (main's history, this build's top token)")
for name in sorted(n for n in hist if hist[n]['measure'] == 'main'):
    h = hist[name]; r = res.get(name)
    if r is None: print(f"   {name:<22} NO RESULT"); bad.append(name); continue
    H = h['history']; same = sum(1 for a, b in zip(r['top'], H) if a == b); n = r['done']
    miss = [i for i, (a, b) in enumerate(zip(r['top'], H)) if a != b]
    notes = valid(name, r, main_run)
    if notes: bad.append(name)
    load = h['load']; t = tot.setdefault(load, [0, 0]); t[0] += same; t[1] += n
    print(f"   {name:<22} {same:>4} of {n:>4}  first position that differs: {miss[0] if miss else 'none'}  positions that differ: {len(miss)}  free 8 start like main: {r.get('free8_equals_history')}{'  NOT VALID: ' + ' | '.join(notes) if notes else ''}")
for load, (a, b) in sorted(tot.items()): print(f"   LOAD {load:<14} {a} of {b} = {100 * a / b:.3f}%" if b else f"   LOAD {load} no positions")
A = sum(v[0] for v in tot.values()); B = sum(v[1] for v in tot.values())
print(f"MEASURE 1 TOTAL: {A} of {B}" + (f" = {100 * A / B:.3f}%" if B else ""))
# measure 2
print("MEASURE 2: same-history top-1 against the maker's own engine (history of 2026-11-10; main that day 198 of 208)")
A2 = B2 = 0
for name in sorted(n for n in hist if hist[n]['measure'] == 'maker'):
    h = hist[name]; r = res.get(name)
    if r is None: print(f"   {name:<22} NO RESULT"); bad.append(name); continue
    top = ref[name]['maker_top1']; n = r['done']; same = sum(1 for a, b in zip(r['top'], top) if a == b)
    old = sum(1 for a, b in zip(h['history'][:n], top) if a == b); moved = sum(1 for a, b in zip(r['top'], h['history']) if a != b)
    notes = valid(name, r, main_run)
    if notes: bad.append(name)
    A2 += same; B2 += n
    print(f"   {name:<22} this build {same:>3} of {n:>3}   rc.9 history itself {old:>3} of {n:>3}   positions where this build leaves the rc.9 token: {moved}{'  NOT VALID: ' + ' | '.join(notes) if notes else ''}")
print(f"MEASURE 2 TOTAL: {A2} of {B2}")
# free runs
if free:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ids as idsmod
    base = json.load(open(free[0])); seen = {}
    for key, src, got in idsmod.lists(free[1:]):
        if key not in base: continue
        a = base[key]; n = min(len(a), len(got)); d = next((j for j in range(n) if a[j] != got[j]), None)
        if d is None and len(a) != len(got): d = n
        s = seen.setdefault(key, [n, d]); s[0] = max(s[0], n)
        if d is not None and (s[1] is None or d < s[1]): s[1] = d
    print("FREE RUNS against main (first position where the free-running answer differs)")
    for key in sorted(seen): print(f"   {key:<22} {'same as main for all ' + str(seen[key][0]) + ' tokens' if seen[key][1] is None else 'first difference at token ' + str(seen[key][1]) + ' of ' + str(seen[key][0])}")
    print(f"FREE RUNS: answers equal to main {sum(1 for v in seen.values() if v[1] is None)} of {len(seen)}")
print("QUALITY VALID: " + ("yes" if not bad else f"NO: {sorted(set(bad))}"))

sys.exit(0 if not bad else 3)
