#!/usr/bin/env python3
"""Draw the 2026-10-24 benchmark figure from the committed receipts.

Run from anywhere: python3 -I tools/benchmarks-chart.py
Writes docs/assets/benchmarks-2026-10-24-light.svg and -dark.svg. Every value
label is the digit string printed in a receipt; nothing is computed or rounded.
FOLDERS is the explicit list of receipts drawn, each with the build its README
must name: the home panel uses the release-build cards, the cloud panels the
morning cards and timing runs (no release-build cloud cards exist).
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = ROOT / 'bench' / 'receipts' / '2026-10-24'
OUT = ROOT / 'docs' / 'assets'
RELEASE = '73f7a9dc157fe4e0d0217aacc4fe57a16812bb9b'
MORNING = '4681d8b2c83f3df09926046dd6bd8b7fd124ce69'
PREFIX = '2026-10-24_'
FOLDERS = {
    'home_off': (RELEASE, '1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off'),
    'home_on': (RELEASE, '1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off'),
    'vm1_off': (MORNING, '1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off'),
    'vm1_on': (MORNING, '1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card'),
    'vm2_off': (MORNING, '1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off'),
    'vm2_on': (MORNING, '1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card'),
}

THEMES = {
    'light': dict(surface='#fcfcfb', ring='rgba(11,11,11,0.10)', ink='#0b0b0b', ink2='#52514e',
                  axis='#c3c2b7', off='#2a78d6', on='#eb6834'),
    'dark': dict(surface='#1a1a19', ring='rgba(255,255,255,0.10)', ink='#f0efec', ink2='#c3c2b7',
                 axis='#383835', off='#3987e5', on='#d95926'),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
WIDTH, PAD, GAP = 880, 24, 16
BAR, SURFACE_GAP, RADIUS = 24, 2, 4
SCALE_MAX, SCALE_PX = 180.0, 196


def folder(key):
    build, name = FOLDERS[key]
    path = RECEIPTS / (PREFIX + name)
    readme = (path / 'README.txt').read_text()
    if build not in readme:
        sys.exit(f'{path.name}: README does not name build {build[:7]}')
    return path, readme


def card(key):
    """Single-stream prose/code and C8 aggregate, as printed on the card."""
    path, _ = folder(key)
    text = ''.join(p.read_text() for p in sorted(path.glob('*.card.txt')))
    found = {}
    for name, pattern in (('prose', r'│\s+PROSE\s+([0-9]+\.[0-9])\s'),
                          ('code', r'│\s+CODE\s+([0-9]+\.[0-9])\s'),
                          ('c8', r'Aggregate tok/s:.*\bC8 ([0-9]+\.[0-9])\b')):
        values = re.findall(pattern, text)
        if len(values) != 1:
            sys.exit(f'{path.name}: expected one {name} value, found {values}')
        found[name] = values[0]
    return found


def timing(key):
    """Drafting ON single-stream medians from a timing-run README (not a card)."""
    path, readme = folder(key)
    if 'NOT A RIGMARK CARD' not in readme:
        sys.exit(f'{path.name}: expected a timing run, not a card')
    found = {}
    for name in ('code', 'prose'):
        values = re.findall(rf'(?<![A-Za-z])ON\s+{name}\s+([0-9]+\.[0-9])\b', readme)
        if not values:
            sys.exit(f'{path.name}: no ON {name} median')
        found[name] = values[0]
    return found


def panels():
    home_off, home_on = card('home_off'), card('home_on')
    vm1_off, vm1_on = card('vm1_off'), timing('vm1_on')
    vm2_off, vm2_on = card('vm2_off'), timing('vm2_on')
    def groups(off, on, on_kind):
        return [(label, sub, [('off', off[k]), (on_kind, on[k]) if k in on else None])
                for k, label, sub in (('code', 'Code', '1 stream'), ('prose', 'Prose', '1 stream'),
                                      ('c8', 'C8', '8 streams, sum'))]
    return [
        ('Home DGX Spark, one Spark', 'OFF card 1444, ON card 1438', groups(home_off, home_on, 'on')),
        ('Cloud GB10, one VM', 'OFF card 1046, ON timing run 1042', groups(vm1_off, vm1_on, 'timing')),
        ('Cloud GB10, two VMs', 'OFF card 1006, ON timing run 1114', groups(vm2_off, vm2_on, 'timing')),
    ]


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def width(text, size):
    """Rough advance width of system sans text, for collision checks only."""
    narrow, wide = set('.,:;il1|() '), set('mwMW')
    return sum(size * (0.28 if c in narrow else 0.85 if c in wide else 0.58) for c in text)


def bar(x, base, height, fill):
    r = min(RADIUS, height / 2)
    top = base - height
    return (f'<path d="M{x:.1f} {base:.1f}V{top + r:.1f}Q{x:.1f} {top:.1f} {x + r:.1f} {top:.1f}'
            f'H{x + BAR - r:.1f}Q{x + BAR:.1f} {top:.1f} {x + BAR:.1f} {top + r:.1f}V{base:.1f}Z" fill="{fill}"/>')


def text(x, y, body, size, fill, anchor='start', weight=None, tabular=False):
    extra = f' font-weight="{weight}"' if weight else ''
    extra += ' font-variant-numeric="tabular-nums"' if tabular else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(body)}</text>'


def render(theme, data):
    t = THEMES[theme]
    out, boxes, rects = [], [], []
    def label(x, y, body, size, fill, anchor='start', **kw):
        w = width(body, size)
        left = x - w / 2 if anchor == 'middle' else x - w if anchor == 'end' else x
        boxes.append((left, y - size * 0.8, left + w, y + size * 0.2, body))
        out.append(text(x, y, body, size, fill, anchor, **kw))

    label(PAD, 38, 'Decode speed, tok/s (higher = faster)', 17, t['ink'], weight=600)
    label(PAD, 60, 'Qwen3.8-Flash-Next-FP4-Local; home: release build 73f7a9d, cloud: main 4681d8b. Each label is the value its receipt prints.', 12, t['ink2'])
    x = PAD
    for kind, name in (('off', 'Drafting OFF, card'), ('on', 'Drafting ON, card'), ('timing', 'Drafting ON, timing run, not a card')):
        fill = {'off': t['off'], 'on': t['on'], 'timing': 'url(#timing)'}[kind]
        out.append(f'<rect x="{x}" y="75" width="12" height="12" rx="2" fill="{fill}"/>')
        label(x + 18, 85.5, name, 12, t['ink'])
        x += 18 + width(name, 12) + 22

    top, base_off, panel_w = 108, 262, (WIDTH - 2 * PAD - 2 * GAP) / 3
    for i, (title, sub, groups) in enumerate(data):
        left = PAD + i * (panel_w + GAP)
        label(left, top + 16, title, 13, t['ink'], weight=600)
        label(left, top + 33, sub, 11, t['ink2'])
        base = top + base_off
        out.append(f'<rect x="{left:.1f}" y="{base:.1f}" width="{panel_w:.1f}" height="1" fill="{t["axis"]}"/>')
        group_w = panel_w / len(groups)
        for j, (name, note, bars) in enumerate(groups):
            centre = left + group_w * (j + 0.5)
            x0 = centre - BAR - SURFACE_GAP / 2
            heights = [float(item[1]) / SCALE_MAX * SCALE_PX if item else 0 for item in bars]
            for k, item in enumerate(bars):
                if item is None:
                    continue
                kind, value = item
                height = heights[k]
                if not 0 < height <= SCALE_PX:
                    sys.exit(f'{value} is outside the drawn scale')
                bx = x0 + k * (BAR + SURFACE_GAP)
                fill = {'off': t['off'], 'on': t['on'], 'timing': 'url(#timing)'}[kind]
                out.append(bar(bx, base, height, fill))
                rects.append((bx, base - height, bx + BAR, base, value))
                # A label wider than its bar overhangs the neighbour; beside a taller
                # neighbour it is aligned to the bar's outer edge instead of centred.
                if width(value, 11) > BAR and heights[1 - k] > height:
                    lx, anchor = (bx + BAR, 'end') if k == 0 else (bx, 'start')
                else:
                    lx, anchor = bx + BAR / 2, 'middle'
                label(lx, base - height - 6, value, 11, t['ink'], anchor, tabular=True)
            label(centre, base + 18, name, 11.5, t['ink'], 'middle', weight=600)
            label(centre, base + 32, note, 10.5, t['ink2'], 'middle')

    y = top + base_off + 62
    for line, fill in (
        ('Cloud C8 with drafting ON is left out: no card, and the timing run’s later ON rounds failed on build 4681d8b.', t['ink2']),
        ('Home OFF and ON cards ran on different home Sparks with the same build.', t['ink2']),
        ('Timing runs time whole requests; cards print decode estimates, so textured and solid bars are not the same estimator.', t['ink2']),
        ('tok/s = completion tokens per wall second; C8 = 8 concurrent streams; cards thinking OFF, context 32768, greedy', t['ink2']),
        ('Sources, bench/receipts/2026-10-24/ (home: release build 73f7a9d; cloud: main 4681d8b); one scale for all panels:', t['ink2']),
    ):
        label(PAD, y, line, 11, fill)
        y += 17
    for key in ('home_off', 'home_on', 'vm1_off', 'vm1_on', 'vm2_off', 'vm2_on'):
        label(PAD + 12, y, PREFIX + FOLDERS[key][1], 10.5, t['ink2'])
        y += 15
    height = int(y + 8)

    for a in range(len(boxes)):
        l, tp, r, b, body = boxes[a]
        if l < PAD - 0.5 or r > WIDTH - PAD + 0.5:
            sys.exit(f'{theme}: text runs past the margin: {body}')
        for l2, t2, r2, b2, body2 in boxes[a + 1:]:
            if l < r2 and l2 < r and tp < b2 and t2 < b:
                sys.exit(f'{theme}: labels collide: {body!r} and {body2!r}')
        for l2, t2, r2, b2, value in rects:
            if l < r2 + 1 and l2 - 1 < r and tp < b2 and t2 < b:
                sys.exit(f'{theme}: label {body!r} touches the {value} bar')

    headline = ('Home DGX Spark, release build 73f7a9d, drafting OFF to ON: code {0} to {1} tok/s, prose {2} to {3}, '
                'C8 {4} to {5}. Cloud GB10, main 4681d8b, drafting OFF: one VM code {6}, prose {7}, C8 {8}; '
                'two VMs code {9}, prose {10}, C8 {11}. Cloud drafting ON timing runs, not cards: '
                'one VM code {12}, prose {13}; two VMs code {14}, prose {15}.').format(*headline_values(data))
    defs = (f'<defs><pattern id="timing" width="6" height="6" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{t["on"]}" fill-opacity="0.28"/>'
            f'<rect width="2.5" height="6" fill="{t["on"]}"/></pattern></defs>')
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
            f'height="{height}" role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">Vade decode speed on 2026-10-24 receipts, home release build 73f7a9d, cloud main 4681d8b</title>\n'
            f'<desc id="d">{esc(headline)}</desc>\n{defs}\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="8" fill="{t["surface"]}" stroke="{t["ring"]}"/>\n')
    return head + '\n'.join(out) + '\n</svg>\n'


def headline_values(data):
    """The values in the order the description sentence names them."""
    home, vm1, vm2 = ([bars for _, _, bars in groups] for _, _, groups in data)
    code, prose, c8 = 0, 1, 2
    return [home[code][0][1], home[code][1][1], home[prose][0][1], home[prose][1][1],
            home[c8][0][1], home[c8][1][1],
            vm1[code][0][1], vm1[prose][0][1], vm1[c8][0][1],
            vm2[code][0][1], vm2[prose][0][1], vm2[c8][0][1],
            vm1[code][1][1], vm1[prose][1][1], vm2[code][1][1], vm2[prose][1][1]]


def main():
    data = panels()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f'benchmarks-2026-10-24-{theme}.svg'
        path.write_text(render(theme, data))
        print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
