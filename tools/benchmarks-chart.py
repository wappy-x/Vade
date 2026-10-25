#!/usr/bin/env python3
"""Draw the v0.1.0 benchmark figure from the committed receipts.

Run from anywhere: python3 -I tools/benchmarks-chart.py
Writes docs/assets/benchmarks-2026-10-24-light.svg and -dark.svg. Every value
label is the digit string printed in a receipt; nothing is computed or rounded.
FOLDERS is the explicit list of RigMark cards drawn. Every one is a v0.1.0
release card, and its README must name the release commit RELEASE.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
RECEIPTS = ROOT / 'bench' / 'receipts' / '2026-10-24'
OUT = ROOT / 'docs' / 'assets'
RELEASE = '5a226519a2a44a0ff7ea9be43118b8b9ec0d8f9e'
PREFIX = '2026-10-24_'
FOLDERS = {
    'home1_off': '1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off',
    'home1_on': '1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off',
    'home2_off': '2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off',
    'home2_on': '2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off',
    'vm1_off': '1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma',
    'vm1_on': '1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma',
    'vm2_off': '2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a',
    'vm2_on': '2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a',
    'vm4_off': '2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a',
    'vm4_on': '2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a',
}
# One row of panels per place; each panel is an OFF card and an ON card.
ROWS = (
    (('One DGX Spark', 'home1'), ('Two DGX Sparks', 'home2')),
    (('One cloud GB10 VM', 'vm1'), ('Two cloud GB10 VMs', 'vm2'), ('Four cloud GB10 VMs', 'vm4')),
)

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
BASE_OFF, ROW_GAP = 262, 64


def folder(key):
    path = RECEIPTS / (PREFIX + FOLDERS[key])
    readme = (path / 'README.txt').read_text()
    if RELEASE not in readme:
        sys.exit(f'{path.name}: README does not name release commit {RELEASE[:7]}')
    return path


def card(key):
    """Single-stream prose/code and C8 aggregate, as printed on the card."""
    path = folder(key)
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


def run(key):
    """The card's time of day, as its folder name starts."""
    return FOLDERS[key].split('_', 1)[0]


def panels():
    rows = []
    for row in ROWS:
        out = []
        for title, key in row:
            off, on = card(key + '_off'), card(key + '_on')
            groups = [(label, note, [('off', off[k]), ('on', on[k])])
                      for k, label, note in (('code', 'Code', '1 chat'), ('prose', 'Prose', '1 chat'),
                                             ('c8', '8 chats', 'together'))]
            out.append((title, f'OFF card {run(key + "_off")}, ON card {run(key + "_on")}', groups))
        rows.append(out)
    return rows


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def width(text, size, tabular=False):
    """Rough advance width of system sans text, for collision checks only.
    Tabular figures give every digit, the 1 too, the same width."""
    narrow, wide = set('.,:;il|() "') | (set() if tabular else {'1'}), set('mwMW')
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
    fills = {'off': t['off'], 'on': 'url(#on)'}
    out, boxes, rects = [], [], []
    def label(x, y, body, size, fill, anchor='start', **kw):
        w = width(body, size, kw.get('tabular', False))
        left = x - w / 2 if anchor == 'middle' else x - w if anchor == 'end' else x
        boxes.append((left, y - size * 0.8, left + w, y + size * 0.2, body))
        out.append(text(x, y, body, size, fill, anchor, **kw))

    label(PAD, 38, 'Decode speed, tok/s (higher = faster)', 17, t['ink'], weight=600)
    label(PAD, 60, 'Qwen3.8-Flash-Next-FP4-Local on Vade v0.1.0 (5a22651). Each label is the value its card prints.', 12, t['ink2'])
    x = PAD
    for kind, name in (('off', 'Drafting OFF'), ('on', 'Drafting ON (the default)')):
        out.append(f'<rect x="{x}" y="75" width="12" height="12" rx="2" fill="{fills[kind]}"/>')
        label(x + 18, 85.5, name, 12, t['ink'])
        x += 18 + width(name, 12) + 22

    top = 108
    for row in data:
        panel_w = (WIDTH - 2 * PAD - (len(row) - 1) * GAP) / len(row)
        for i, (title, sub, groups) in enumerate(row):
            left = PAD + i * (panel_w + GAP)
            label(left, top + 16, title, 13, t['ink'], weight=600)
            label(left, top + 33, sub, 11, t['ink2'])
            base = top + BASE_OFF
            out.append(f'<rect x="{left:.1f}" y="{base:.1f}" width="{panel_w:.1f}" height="1" fill="{t["axis"]}"/>')
            group_w = panel_w / len(groups)
            for j, (name, note, bars) in enumerate(groups):
                centre = left + group_w * (j + 0.5)
                x0 = centre - BAR - SURFACE_GAP / 2
                heights = [float(value) / SCALE_MAX * SCALE_PX for _, value in bars]
                for k, (kind, value) in enumerate(bars):
                    height = heights[k]
                    if not 0 < height <= SCALE_PX:
                        sys.exit(f'{value} is outside the drawn scale')
                    bx = x0 + k * (BAR + SURFACE_GAP)
                    out.append(bar(bx, base, height, fills[kind]))
                    rects.append((bx, base - height, bx + BAR, base, value))
                    # A label wider than its bar overhangs the neighbour; beside a neighbour
                    # as tall or taller it is aligned to the bar's outer edge instead of centred.
                    if width(value, 11, True) > BAR and heights[1 - k] >= height:
                        lx, anchor = (bx + BAR, 'end') if k == 0 else (bx, 'start')
                    else:
                        lx, anchor = bx + BAR / 2, 'middle'
                    label(lx, base - height - 6, value, 11, t['ink'], anchor, tabular=True)
                label(centre, base + 18, name, 11.5, t['ink'], 'middle', weight=600)
                label(centre, base + 32, note, 10.5, t['ink2'], 'middle')
        top += BASE_OFF + ROW_GAP

    y = top + 6
    for line in (
        'Every bar is a RigMark card for Vade v0.1.0, thinking OFF, context 32768, greedy decoding.',
        'Drafting is ON by default; for the OFF cards, every request sent "draft":"plain".',
        'tok/s = completion tokens per wall second; 8 chats = eight concurrent chats, counted together.',
        '8-chat rates vary: at eight chats the engine serves some chats without drafting.',
        'Sources, bench/receipts/2026-10-24/; one scale for all panels:',
    ):
        label(PAD, y, line, 11, t['ink2'])
        y += 17
    for key in FOLDERS:
        label(PAD + 12, y, PREFIX + FOLDERS[key], 10.5, t['ink2'])
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

    defs = (f'<defs><pattern id="on" width="6" height="6" patternUnits="userSpaceOnUse" '
            f'patternTransform="rotate(45)"><rect width="6" height="6" fill="{t["on"]}" fill-opacity="0.45"/>'
            f'<rect width="3" height="6" fill="{t["on"]}"/></pattern></defs>')
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
            f'height="{height}" role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">Vade v0.1.0 decode speed, RigMark cards of 2026-10-24</title>\n'
            f'<desc id="d">{esc(headline(data))}</desc>\n{defs}\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="8" fill="{t["surface"]}" stroke="{t["ring"]}"/>\n')
    return head + '\n'.join(out) + '\n</svg>\n'


def headline(data):
    """One sentence per panel, drafting OFF to ON, from the drawn labels."""
    parts = []
    for row in data:
        for title, _, groups in row:
            (_, _, code), (_, _, prose), (_, _, c8) = groups
            chats = (f'{c8[0][1]} either way' if c8[0][1] == c8[1][1]
                     else f'{c8[0][1]} to {c8[1][1]}')
            parts.append(f'{title}, drafting OFF to ON: code {code[0][1]} to {code[1][1]} tok/s, '
                         f'prose {prose[0][1]} to {prose[1][1]}, 8 chats {chats}.')
    return ' '.join(parts)


def main():
    data = panels()
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f'benchmarks-2026-10-24-{theme}.svg'
        path.write_text(render(theme, data))
        print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
