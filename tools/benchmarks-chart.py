#!/usr/bin/env python3
"""Draw the v0.1.0 one-Spark vs two-Spark figure from the committed receipts.

Run from anywhere: python3 -I tools/benchmarks-chart.py
Writes docs/assets/benchmarks-2026-10-24-light.svg and -dark.svg. Every value
label is the digit string printed in a receipt; nothing is computed or rounded.
FOLDERS is the explicit list of RigMark cards drawn: one Spark and two Sparks,
drafting ON (the default), on our own Sparks. SINGLES and PAIRS are every
same-day home card with the same settings; the headline is checked against all
of them, and the eight-chat note quotes their range. Each is a v0.1.0 release
card, and its README must name the release commit RELEASE. Drafting OFF and the
cloud VMs stay in the tables of docs/benchmarks.md.
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
    'one': '1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off',
    'two': '2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off',
}
ON = '5a22651-drafting-on-thinking-off'
SINGLES = (FOLDERS['one'], f'1945_qwen3.8-flash-next-nvfp4_local-1spark_{ON}',
           f'1946_qwen3.8-flash-next-nvfp4_local-1spark_{ON}', f'1946_qwen3.8-flash-next-nvfp4_local-1spark_{ON}-home-5',
           f'1947_qwen3.8-flash-next-nvfp4_local-1spark_{ON}', f'1947_qwen3.8-flash-next-nvfp4_local-1spark_{ON}-home-6')
PAIRS = (FOLDERS['two'], f'2022_qwen3.8-flash-next-nvfp4_local-2spark_{ON}',
         f'2022_qwen3.8-flash-next-nvfp4_local-2spark_{ON}-home-5-home-6')
COUNT = {3: 'three', 6: 'six'}
NAMES = {'one': '1 Spark', 'two': '2 Sparks'}
# Writing speed shares one tok/s scale; the first-reply panel has its own, in seconds.
GROUPS = (('code', 'Code, 1 chat'), ('prose', 'Prose, 1 chat'), ('c8', '8 chats together'))
HEADLINE = 'A second Spark answers a long prompt faster. One chat does not write faster.'
SUBHEAD = ('With drafting ON (the default), one chat writes slower on two Sparks. '
           'Eight chats vary from run to run.')

THEMES = {
    'light': dict(surface='#fcfcfb', ring='rgba(11,11,11,0.10)', ink='#0b0b0b', ink2='#52514e',
                  axis='#c3c2b7', one='#2a78d6', two='#eb6834'),
    'dark': dict(surface='#1a1a19', ring='rgba(255,255,255,0.10)', ink='#f0efec', ink2='#c3c2b7',
                 axis='#383835', one='#3987e5', two='#d95926'),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
WIDTH, PAD = 880, 24
BAR, SURFACE_GAP, RADIUS = 20, 2, 4
LABEL_W = 66
RIGHT_X = 572
SPEED_MAX, SPEED_PX = 180.0, 376
TTFT_MAX, TTFT_PX = 4.5, 190


def folder(name):
    path = RECEIPTS / (PREFIX + name)
    if not (path / 'README.txt').is_file():
        sys.exit(f'{path.name}: receipt folder or its README.txt is missing')
    readme = (path / 'README.txt').read_text(encoding='utf-8')
    if RELEASE not in readme:
        sys.exit(f'{path.name}: README does not name release commit {RELEASE[:7]}')
    return path


def card(name):
    """Single-stream prose/code, C8 aggregate and cold 8K TTFT, as printed on the card."""
    path = folder(name)
    text = ''.join(p.read_text(encoding='utf-8') for p in sorted(path.glob('*.card.txt')))
    found = {}
    for name, pattern in (('prose', r'│\s+PROSE\s+([0-9]+\.[0-9])\s'),
                          ('code', r'│\s+CODE\s+([0-9]+\.[0-9])\s'),
                          ('c8', r'Aggregate tok/s:.*\bC8 ([0-9]+\.[0-9])\b'),
                          ('ttft', r'│\s+8K\s+[0-9,]+\s+([0-9]+\.[0-9]+)\s')):
        values = re.findall(pattern, text)
        if len(values) != 1:
            sys.exit(f'{path.name}: expected one {name} value, found {values}')
        found[name] = values[0]
    return found


def check_headline(singles, pairs):
    """The headline is words, not numbers; stop unless every pair beats every single Spark."""
    def worst(cards, k, pick):
        return pick(float(c[k]) for c in cards)
    claims = (('every pair answers the 8K prompt faster than every single Spark',
               worst(pairs, 'ttft', max) < worst(singles, 'ttft', min)),
              ('one chat writes slower on every pair than on every single Spark',
               all(worst(pairs, k, max) < worst(singles, k, min) for k in ('code', 'prose'))))
    for claim, holds in claims:
        if not holds:
            sys.exit(f'the cards no longer say: {claim}; rewrite HEADLINE and SUBHEAD')


def spread(cards, k):
    """Lowest and highest value of one field, as the cards print them."""
    values = sorted((c[k] for c in cards), key=float)
    return values[0], values[-1]


def c8_note(singles, pairs):
    (a, b), (c, d) = spread(singles, 'c8'), spread(pairs, 'c8')
    return (f'Across all our runs, 8 chats gave {a} to {b} tok/s',
            f'on {COUNT[len(singles)]} single Sparks and {c} to {d} on {COUNT[len(pairs)]} pairs.')


def esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def width(text, size, tabular=False):
    """Rough advance width of system sans text, for collision checks only.
    Tabular figures give every digit, the 1 too, the same width."""
    narrow, wide = set('.,:;il|() "') | (set() if tabular else {'1'}), set('mwMW')
    return sum(size * (0.28 if c in narrow else 0.85 if c in wide else 0.58) for c in text)


def bar(x, top, length, fill):
    """A horizontal bar from the baseline at x, rounded at its data end only."""
    r = min(RADIUS, length / 2)
    end, bottom = x + length, top + BAR
    return (f'<path d="M{x:.1f} {top:.1f}H{end - r:.1f}Q{end:.1f} {top:.1f} {end:.1f} {top + r:.1f}'
            f'V{bottom - r:.1f}Q{end:.1f} {bottom:.1f} {end - r:.1f} {bottom:.1f}H{x:.1f}Z" fill="{fill}"/>')


def text(x, y, body, size, fill, anchor='start', weight=None, tabular=False):
    extra = f' font-weight="{weight}"' if weight else ''
    extra += ' font-variant-numeric="tabular-nums"' if tabular else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(body)}</text>'


def render(theme, data, note, checked):
    t = THEMES[theme]
    out, boxes, rects = [], [], []
    def label(x, y, body, size, fill, anchor='start', **kw):
        w = width(body, size, kw.get('tabular', False))
        left = x - w / 2 if anchor == 'middle' else x - w if anchor == 'end' else x
        boxes.append((left, y - size * 0.8, left + w, y + size * 0.2, body))
        out.append(text(x, y, body, size, fill, anchor, **kw))

    def pair(left, top, values, scale_max, scale_px, unit=''):
        """Two labelled bars, 1 Spark above 2 Sparks, from one baseline."""
        base = left + LABEL_W
        out.append(f'<rect x="{base - 1:.1f}" y="{top - 3:.1f}" width="1" height="{2 * BAR + SURFACE_GAP + 6}" fill="{t["axis"]}"/>')
        for k, key in enumerate(('one', 'two')):
            y = top + k * (BAR + SURFACE_GAP)
            length = float(values[key]) / scale_max * scale_px
            if not 0 < length <= scale_px:
                sys.exit(f'{values[key]} is outside the drawn scale')
            out.append(bar(base, y, length, t[key]))
            rects.append((base, y, base + length, y + BAR, values[key]))
            label(left, y + 14.5, NAMES[key], 12, t['ink'])
            label(base + length + 6, y + 14.5, values[key] + unit, 12, t['ink'], tabular=True)

    label(PAD, 40, HEADLINE, 19, t['ink'], weight=650)
    label(PAD, 63, SUBHEAD, 13, t['ink2'])
    x = PAD
    for key in ('one', 'two'):
        out.append(f'<rect x="{x}" y="80" width="12" height="12" rx="2" fill="{t[key]}"/>')
        label(x + 18, 90.5, NAMES[key], 12.5, t['ink'], weight=600)
        x += 18 + width(NAMES[key], 12.5) + 22
    label(x + 4, 90.5, 'Vade v0.1.0 on our own DGX Sparks, drafting ON', 12, t['ink2'])

    top = 128
    label(PAD, top, 'Writing speed', 14, t['ink'], weight=600)
    label(PAD, top + 18, 'tokens per second, higher is better', 12, t['ink2'])
    label(RIGHT_X, top, 'First reply after an 8K-token prompt', 14, t['ink'], weight=600)
    label(RIGHT_X, top + 18, 'seconds, lower is better', 12, t['ink2'])
    y = top + 46
    for name, title in GROUPS:
        label(PAD, y, title, 12.5, t['ink'], weight=600)
        pair(PAD, y + 9, {k: data[k][name] for k in data}, SPEED_MAX, SPEED_PX)
        y += 9 + 2 * BAR + SURFACE_GAP
        if name == 'c8':
            y += 4
            for line in note:
                y += 16
                label(PAD, y, line, 11.5, t['ink2'])
        y += 30
    panel_bottom = y - 30
    pair(RIGHT_X, top + 46 - 3.5, {k: data[k]['ttft'] for k in data}, TTFT_MAX, TTFT_PX, ' s')

    y += 2
    for line in (
        'Each bar is one RigMark card for Vade v0.1.0 (5a22651), thinking OFF, context 32768, greedy decoding.',
        'Two Sparks run as a pipeline: each holds part of the model\'s layers and passes its work to the next.',
        checked,
        'Drafting OFF and the cloud VM runs are in the tables on the benchmarks page.',
        'Sources, bench/receipts/2026-10-24/:',
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
        if l < RIGHT_X and r > RIGHT_X - 12 and tp < panel_bottom and b > top - 14:
            sys.exit(f'{theme}: text runs into the right panel: {body}')
        for l2, t2, r2, b2, body2 in boxes[a + 1:]:
            if l < r2 and l2 < r and tp < b2 and t2 < b:
                sys.exit(f'{theme}: labels collide: {body!r} and {body2!r}')
        for l2, t2, r2, b2, value in rects:
            if l < r2 + 1 and l2 - 1 < r and tp < b2 and t2 < b:
                sys.exit(f'{theme}: label {body!r} touches the {value} bar')

    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
            f'height="{height}" role="img" aria-labelledby="t d" font-family="{FONT}">\n'
            f'<title id="t">Vade v0.1.0 on one DGX Spark and on two, drafting ON, RigMark cards of 2026-10-24</title>\n'
            f'<desc id="d">{esc(describe(data, note))}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="8" fill="{t["surface"]}" stroke="{t["ring"]}"/>\n')
    return head + '\n'.join(out) + '\n</svg>\n'


def describe(data, note):
    """The figure in words, from the drawn labels."""
    one, two = data['one'], data['two']
    return (f'{HEADLINE} {SUBHEAD} Drafting ON, 1 Spark then 2 Sparks: code {one["code"]} and {two["code"]} tok/s, '
            f'prose {one["prose"]} and {two["prose"]}, 8 chats together {one["c8"]} and {two["c8"]}; '
            f'first reply after an 8K-token prompt {one["ttft"]} s and {two["ttft"]} s. {" ".join(note)}')


def main():
    data = {key: card(name) for key, name in FOLDERS.items()}
    singles, pairs = [card(name) for name in SINGLES], [card(name) for name in PAIRS]
    check_headline(singles, pairs)
    note = c8_note(singles, pairs)
    checked = (f'The headline holds on all {COUNT[len(singles)]} single Sparks and all '
               f'{COUNT[len(pairs)]} pairs we measured with these settings.')
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f'benchmarks-2026-10-24-{theme}.svg'
        path.write_text(render(theme, data, note, checked), encoding='utf-8')
        print(f'wrote {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
