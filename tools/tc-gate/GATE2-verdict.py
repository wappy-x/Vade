#!/usr/bin/env python3
"""TCGATE-verdict.py (2026-11-15, seat TCGATE): the gate that replaces "the same token IDs as main" after a rebase.
Stdlib only. It reads the run folders that TCGATE-gate.sh wrote and proves the exactness rules on the token IDs the
run itself produced. It never reads a baseline: the proof must hold on NEW token IDs.

  TCGATE-verdict.py RUN_A                  parts 2, 3, 4 of one run (part 1 says NOT RUN)
  TCGATE-verdict.py RUN_A RUN_B            part 1 (twice the same) across the two runs, and parts 2, 3, 4 of each run
  --expect-fail=1,3                        a wrong-on-purpose control: the named parts MUST fail; prints CONTROL CAUGHT
                                           or CONTROL MISSED for each, exit 0 only when every named part failed
  --only=2,4                               judge only these parts (a control run that holds only some parts)
A run may be several folders joined with "+", for example runs/a1+runs/a2: the gate runs in short parts (a served turn
is 12 minutes at most), each part a folder of its own with its own servers. The folders of one side must be runs of
ONE commit and must not hold the same server twice.

Part 1  TWICE THE SAME (rule 1): every token ID list of run A equals the list of the same request in run B, complete
        and in full length; both runs hold the same requests; both built the same commit; the servers of the two runs
        are different processes (start records), and no list of the four loads came from the prompt cache.
Part 2  ALONE EQUALS GROUP (rules 2 and 5): each chat alone equals the same chat in every group of 2, 4 and 8 chats
        it ran in, and the copy chat alone equals each of its 2 and 8 copies. The server log must show batch steps of
        2, 4 and 8 rows, or the groups did not form. If the run holds the check tool's row sweeps, each must have
        ended with exit 0.
Part 3  PROMPT PATH EQUALS DECODE PATH (rule 3): the kept second turn equals the same turn read cold, and the kept
        turn did use the cache; a continuation read cold equals the tokens the decode steps made.
Part 4  DRAFTED EQUALS PLAIN (rule 4): for each kind of answer, strict drafting at every depth the run holds equals
        plain decoding, in full length. That the drafting path did run is checked by depth, not by request (MEASURED
        on main: at depth 1 two of three kinds of answer propose 1 token in 256 and accept none; at depth 7 one kind
        stays at passes of 6 rows): for each depth some request accepts drafted tokens and some request reaches a
        check pass of depth + 1 rows; every drafted request has a check pass of more than one row. Plain decoding
        must also agree between the servers of the run.
Last line: GATE VERDICT: PASS | FAIL (parts) | INCOMPLETE (parts not run)."""
import sys, os, json, re, glob

def load(p):
    try: return json.load(open(p))
    except Exception: return None

class Run:
    def __init__(self, d):
        if '+' in d:
            parts = [Run(x) for x in d.split('+')]
            self.d = parts[0].d; self.dirs = [p.d for p in parts]; self.name = '+'.join(p.name for p in parts)
            self.lists = {}; self.chat = []; self.many = []; self.turn_checks = None; self.turns = {}; self.problems = []; self.logs = {}; self.starts = {}; self.drive = ''
            commits = {p.commit for p in parts}; controls = {p.control for p in parts}
            self.commit = parts[0].commit if len(commits) == 1 else None; self.control = parts[0].control if len(controls) == 1 else 'mixed'
            if len(commits) != 1: self.problems.append(f"the folders of one side built different commits: {sorted(str(c) for c in commits)}")
            for p in parts:
                for k in p.lists:
                    if k in self.lists: self.problems.append(f"the request {k} is in two folders of one side")
                for k in p.logs:
                    if k in self.logs: self.problems.append(f"the server {k} is in two folders of one side")
                self.lists.update(p.lists); self.chat += p.chat; self.many += p.many; self.turns.update(p.turns); self.problems += p.problems
                self.logs.update(p.logs); self.starts.update(p.starts); self.drive += p.drive
                if p.turn_checks is not None: self.turn_checks = p.turn_checks
            self.has_tool = any(p.has_tool for p in parts)
            return
        self.d = d.rstrip('/'); self.name = os.path.basename(self.d); self.dirs = [self.d]
        self.has_tool = os.path.isdir(os.path.join(self.d, 'TOOL'))
        self.lists = {}     # request name -> ids
        self.chat = []      # (server, workload, arm, depth, ids, draft_fields, file)
        self.many = []      # (file, kind, count, [(spec, ids)])
        self.turn_checks = None; self.turns = {}
        self.problems = []
        if not os.path.isfile(os.path.join(self.d, 'LOADS-DONE')): self.problems.append('run has no LOADS-DONE marker')
        for root, _, files in os.walk(self.d):
            for f in sorted(files):
                p = os.path.join(root, f); rel = os.path.relpath(p, self.d)
                if '-warm' in f: continue
                if f == 'receipt.jsonl':
                    for line in open(p):
                        if not line.strip(): continue
                        r = json.loads(line)
                        self.lists[f"{rel}#{r.get('cell')}"] = r.get('output_ids') or []
                        if not r.get('ok'): self.problems.append(f"{rel}#{r.get('cell')}: request not ok")
                elif f.endswith('-checks.json'):
                    self.turn_checks = load(p)
                elif f.endswith('.json') and 'health' not in f:
                    r = load(p)
                    if not isinstance(r, dict): continue
                    if isinstance(r.get('streams'), list):
                        streams = [(str(s.get('spec')), [int(x) for x in s.get('token_ids') or []]) for s in r['streams']]
                        self.many.append((rel, r.get('kind'), r.get('count'), streams))
                        for i, (spec, ids) in enumerate(streams): self.lists[f"{rel}#{i}:{spec}"] = ids
                    elif 'workload' in r and 'arm' in r:
                        ids = [int(x) for x in (r.get('token_ids') or [])]
                        self.lists[rel] = ids
                        self.chat.append((rel.split(os.sep)[0], r['workload'], r['arm'], r.get('mtp_depth'), ids, r.get('draft_fields'), rel))
                        if r.get('http_status') != 200 or r.get('error'): self.problems.append(f"{rel}: http {r.get('http_status')} {str(r.get('error'))[:80]}")
                        if (r.get('request_body') or {}).get('cache') is not False: self.problems.append(f"{rel}: the request did not switch the prompt cache off")
                    elif 'output_ids' in r and 'name' in r:
                        self.lists[rel] = r['output_ids']; self.turns[r['name']] = r
                        if not r.get('ok'): self.problems.append(f"{rel}: request not ok")
        self.logs = {os.path.basename(p)[len('rank0-'):-len('.log')]: p for p in glob.glob(os.path.join(self.d, '*', 'rank0-*.log'))}
        self.starts = {}
        for name, p in self.logs.items():
            text = open(p, errors='replace').read()
            m = re.search(r'TCGATE SERVER START rank=0 epoch=([\d.]+) pid=(\d+).*?serve_sha=(\w+) ptx_sha=(\w+)', text)
            self.starts[name] = (m.groups() if m else None, text.count('ready bind='))
        drive = os.path.join(self.d, 'drive.log'); self.drive = open(drive, errors='replace').read() if os.path.exists(drive) else ''
        m = re.search(r'A: build of ([0-9a-f]{40}) started', self.drive); self.commit = m.group(1) if m else None
        m = re.search(r'control: (\S+)\)', self.drive); self.control = m.group(1) if m else 'none'
    def count(self, server, pattern):
        p = self.logs.get(server)
        return len(re.findall(pattern, open(p, errors='replace').read())) if p else 0

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return f"first difference at token {i} of {n}: {a[i]} against {b[i]}"
    return f"lengths {len(a)} and {len(b)}" if len(a) != len(b) else None

def inventory(r):
    """The complete fresh repeat includes all four loads and the deep draft cells."""
    bad = []
    for server, arms in (('PL', ('P',)), ('DR', ('P', 'S2', 'S3')), ('DP', ('P', 'S1', 'S3', 'S7'))):
        for workload in ('code', 'prose', 'structured'):
            for arm in arms:
                cells = [c for c in r.chat if c[0] == server and c[1] == workload and c[2] == arm]
                if len(cells) != 1 or len(cells[0][4]) != 256:
                    bad.append(f'{server} {workload} {arm}: need one complete 256-token cell')
    key = 'PL/out-PL-read/receipt.jsonl#g8192d'
    if len(r.lists.get(key, [])) != 160: bad.append('8K read cell missing or not 160 tokens')
    if not r.many: bad.append('chat groups missing')
    for rel, _, count, streams in r.many:
        if count != len(streams) or any(len(ids) != 128 for _, ids in streams):
            bad.append(f'{rel}: need {count} complete 128-token streams')
    return bad

def part1(a, b, out, control=False):
    """control: run B is a wrong-on-purpose build of another commit that may hold fewer requests. Then only the
    requests both runs hold are compared, and only a token difference is a finding."""
    bad = []; tokens = 0
    if not a.lists or not b.lists: bad.append("a run has no token lists")
    if not a.starts or not b.starts: bad.append("a run has no server start records")
    if not control and set(a.starts) != set(b.starts): bad.append("the runs do not hold the same server start records")
    if not control and (a.commit != b.commit or not a.commit): bad.append(f"the runs built different commits: {a.commit} and {b.commit}")
    only_a = sorted(set(a.lists) - set(b.lists)); only_b = sorted(set(b.lists) - set(a.lists))
    if not control and (only_a or only_b): bad.append(f"the runs do not hold the same requests: only in A {only_a[:3]}, only in B {only_b[:3]}")
    same = 0; common = sorted(set(a.lists) & set(b.lists))
    for k in common:
        x, y = a.lists[k], b.lists[k]; tokens += min(len(x), len(y))
        if x and x == y: same += 1
        else: bad.append(f"{k}: {first_diff(x, y) or 'empty list'}")
    if not common: bad.append("no list to compare")
    # the two runs must be different server processes of the same build
    for name in sorted(set(a.starts) & set(b.starts)):
        sa, sb = a.starts[name], b.starts[name]
        if not sa[0] or not sb[0]: bad.append(f"server {name}: no start record"); continue
        if sa[0][:2] == sb[0][:2]: bad.append(f"server {name}: the two runs show the same server process")
        if sa[1] != 1 or sb[1] != 1: bad.append(f"server {name}: a run started its server {sa[1]} and {sb[1]} times")
        if not control and sa[0][2:] != sb[0][2:]: bad.append(f"server {name}: the two runs used different builds: {sa[0][2:]} and {sb[0][2:]}")
    for r in (a, b):
        for p in r.problems: bad.append(f"run {r.name}: {p}")
    out.append(f"   lists equal {same} of {len(common)}, tokens compared {tokens}, servers compared {sorted(set(a.starts) & set(b.starts))}, commit {a.commit}")
    return bad

def part2(r, out):
    if not r.many and not r.has_tool: return None
    bad = list(r.problems); pairs = 0; tokens = 0
    if r.many:
        alone = {}
        for rel, kind, count, streams in r.many:
            if count == 1 and len(streams) == 1: alone[streams[0][0]] = (rel, streams[0][1])
            if count != len(streams): bad.append(f"{rel}: stream count {len(streams)} differs from declared count {count}")
        groups = [m for m in r.many if m[2] and m[2] > 1]
        if not alone: bad.append("no chat ran alone")
        if not groups: bad.append("no group of chats ran")
        sizes = set()
        for rel, kind, count, streams in groups:
            sizes.add(count)
            for i, (spec, ids) in enumerate(streams):
                if spec not in alone: bad.append(f"{rel}: chat {spec} never ran alone"); continue
                ref = alone[spec][1]
                if ids and ids == ref: pairs += 1; tokens += len(ids)
                else: bad.append(f"{rel} slot {i} chat {spec} (group of {count}) against alone: {first_diff(ref, ids) or 'empty list'}")
        for need in (2, 4, 8):
            if need not in sizes: bad.append(f"no group of {need} chats in the run")
            n = r.count('PL', rf'SERVED BATCH rank=0 rows={need} ')
            if n == 0: bad.append(f"the server log shows no batch step of {need} rows: the group of {need} did not form")
        out.append(f"   chat-in-group lists equal to the chat alone: {pairs} ({tokens} tokens); chats that ran alone: {len(alone)}; group sizes {sorted(sizes)}; batch steps in the server log: " + ', '.join(f"{n} rows x {r.count('PL', rf'SERVED BATCH rank=0 rows={n} ')}" for n in (2, 4, 8)))
    if r.has_tool:
        modes = re.findall(r'TOOL (\S+) exit=(\d+)', r.drive)
        required = {'prompt-rows', 'nv-tc-groups', 'dense-tc-rows'}
        missing = required - {m for m, _ in modes}
        if missing: bad.append(f"missing required check tool modes: {sorted(missing)}")
        for mode, code in modes:
            if code != '0': bad.append(f"check tool {mode} ended with exit {code}")
        out.append("   check tool row sweeps: " + ', '.join(f"{m} exit={c}" for m, c in modes))
    return bad

def part3(r, out):
    if r.turn_checks is None: return None
    bad = list(r.problems)
    for c in r.turn_checks.get('checks', []):
        if not c.get('equal'): bad.append(f"{c['name']}: DIFFER (first difference at token {c.get('first_difference')})")
        elif not c.get('valid'): bad.append(f"{c['name']}: not a valid check: {c.get('note')}")
    names = [c['name'] for c in r.turn_checks.get('checks', [])]
    for need in ('turn2-kept-against-turn2-cold', 'long-continuation-k32', 'short-continuation-k1', 'short-continuation-k17', 'short-continuation-k40'):
        if need not in names: bad.append(f"the check {need} did not run")
    kept = r.turns.get('turn2', {}); resumes = r.count('K32', r'SERVED RESUME rank')
    out.append(f"   checks run {len(names)}; kept turn cached_tokens={kept.get('cached_tokens')} of {kept.get('prompt_tokens')} prompt tokens; resume lines in the server log {resumes}")
    return bad

def part4(r, out):
    servers = sorted({c[0] for c in r.chat if c[0] in ('DR', 'DP')})
    if not servers: return None
    bad = list(r.problems); pairs = 0; tokens = 0; depths = set(); cover = {}
    for server in servers:
        for wl in sorted({c[1] for c in r.chat if c[0] == server}):
            plain = [c for c in r.chat if c[0] == server and c[1] == wl and c[2] == 'P']
            strict = [c for c in r.chat if c[0] == server and c[1] == wl and c[2] != 'P']
            if not plain: bad.append(f"{server} {wl}: no plain run"); continue
            if not strict: bad.append(f"{server} {wl}: no drafted run"); continue
            ref = plain[0][4]
            for _, _, arm, depth, ids, fields, rel in strict:
                depths.add(depth)
                if ids and ids == ref: pairs += 1; tokens += len(ids)
                else: bad.append(f"{rel} (strict depth {depth}) against plain: {first_diff(ref, ids) or 'empty list'}")
                c = cover.setdefault((server, depth), {"accepted": 0, "proposed": 0, "max_rows": 0, "multi": 0, "requests": 0})
                f = fields or {}; c["requests"] += 1; c["accepted"] += f.get('accepted') or 0; c["proposed"] += f.get('proposed') or 0
                c["max_rows"] = max(c["max_rows"], f.get('max_rows') or 0); c["multi"] += max(0, (f.get('rows') or 0) - (f.get('cycles') or 0))
                if not fields: bad.append(f"{rel}: no draft line in the server log for this request")
                elif (f.get('rows') or 0) <= (f.get('cycles') or 0): bad.append(f"{rel}: no check pass of more than one row, so the request did not use the drafting path")
    for (server, depth), c in sorted(cover.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        if c["accepted"] == 0: bad.append(f"server {server} depth {depth}: no request accepted a drafted token")
        if c["max_rows"] != (depth or 0) + 1: bad.append(f"server {server} depth {depth}: no request reached a check pass of {(depth or 0) + 1} rows (largest {c['max_rows']})")
    for required_depth in (1, 3, 7):
        if required_depth not in depths: bad.append(f"strict depth {required_depth} did not run")
    for server in servers:
        if r.count(server, r'SERVED VERIFY rank=0 .* rows=[2-9]') == 0: bad.append(f"server {server}: the log shows no check pass of more than one row")
    # plain decoding of one request must agree between every server of the run (mtp 0, 3 and 7 are three servers)
    for wl in sorted({c[1] for c in r.chat}):
        plain = [c for c in r.chat if c[1] == wl and c[2] == 'P']
        for c in plain[1:]:
            if c[4] != plain[0][4]: bad.append(f"plain {wl}: {c[6]} against {plain[0][6]}: {first_diff(plain[0][4], c[4])}")
    out.append(f"   drafted lists equal to plain: {pairs} ({tokens} tokens); servers {servers}; depths {sorted(d for d in depths if d is not None)}; plain runs compared across servers: {sum(1 for c in r.chat if c[2] == 'P')}")
    out.append("   drafting that ran, by server and depth (requests, proposed, accepted, rows beyond one a pass, largest pass): " + '; '.join(f"{s} d{d}: {c['requests']}, {c['proposed']}, {c['accepted']}, {c['multi']}, {c['max_rows']}" for (s, d), c in sorted(cover.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0))))
    return bad

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    expect = set(); only = None
    for a in sys.argv[1:]:
        if a.startswith('--expect-fail='): expect = {int(x) for x in a.split('=')[1].split(',') if x}
        if a.startswith('--only='): only = {int(x) for x in a.split('=')[1].split(',') if x}
    if not 1 <= len(args) <= 2: print(__doc__); return 2
    runs = [Run(d) for d in args]
    names = {1: 'TWICE THE SAME', 2: 'ALONE EQUALS GROUP', 3: 'PROMPT PATH EQUALS DECODE PATH', 4: 'DRAFTED EQUALS PLAIN'}
    state = {}; findings = {}
    for r in runs: print(f"RUN {r.name}: commit {r.commit}, control {r.control}, lists {len(r.lists)}, servers {sorted(r.logs)}")
    for n in (1, 2, 3, 4):
        if only is not None and n not in only: continue
        out = []; bad = None
        if n == 1:
            bad = part1(runs[0], runs[1], out, control=bool(expect)) if len(runs) == 2 else None
            if bad is not None and not expect: bad += inventory(runs[0]) + inventory(runs[1])
        else:
            fn = {2: part2, 3: part3, 4: part4}[n]; bad = None
            for r in runs:
                b = fn(r, out)
                if b is not None: bad = (bad or []) + [f"run {r.name}: {x}" for x in b]
        findings[n] = bad or []
        if bad is None: state[n] = 'NOT RUN'; print(f"GATE PART {n} {names[n]}: NOT RUN")
        elif bad: state[n] = 'FAIL'; print(f"GATE PART {n} {names[n]}: FAIL ({len(bad)} findings)"); [print(x) for x in out]; [print(f"   FAIL {x}") for x in bad[:12]]
        else: state[n] = 'PASS'; print(f"GATE PART {n} {names[n]}: PASS"); [print(x) for x in out]
    failed = [n for n, s in state.items() if s == 'FAIL']; missing = [n for n, s in state.items() if s == 'NOT RUN']
    if expect:
        ok = True
        for n in sorted(expect):
            # a control counts as caught only by a token difference, never by a missing request or a missing log line
            token = [x for x in findings.get(n, []) if 'first difference at token' in x or 'DIFFER' in x]
            caught = state.get(n) == 'FAIL' and bool(token); ok &= caught
            print(f"CONTROL {'CAUGHT' if caught else 'MISSED'} by part {n} {names[n]} (state {state.get(n)}, findings that are token differences: {len(token)})")
        print("CONTROL VERDICT: " + ("EVERY NAMED PART CAUGHT THE WRONG BUILD" if ok else "A PART DID NOT CATCH THE WRONG BUILD"))
        return 0 if ok else 1
    passed = [n for n, s in state.items() if s == 'PASS']; skipped = [n for n in (1, 2, 3, 4) if n not in state]
    if failed: print(f"GATE VERDICT: FAIL (parts {failed}; passed {passed}; not run {missing + skipped})"); return 1
    if missing or skipped: print(f"GATE VERDICT: INCOMPLETE (parts not run: {sorted(missing + skipped)}; parts passed: {passed})"); return 3
    print("GATE VERDICT: PASS (parts 1, 2, 3, 4)"); return 0

if __name__ == '__main__': sys.exit(main())
