#!/usr/bin/env python3
"""Check local documentation links and anchors without network access."""
from pathlib import Path
import os
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


def anchors(path):
    text = path.read_text()
    found = set(re.findall(r'''(?:id|name)=["']([^"']+)''', text))
    seen = {}
    for line in text.splitlines():
        heading = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not heading:
            continue
        title = re.sub(r'<[^>]+>', '', heading[1])
        anchor = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        found.add(anchor + (f'-{count}' if count else ''))
    return found


def check(root, paths):
    issues = []
    pattern = r'''!?\[[^\]]*\]\(([^
]*?)\)|(?:src|srcset|href)=["']([^"']+)["']|(?m:^[ 	]{0,3}\[[^\]]+\]:[ 	]*(\S+))'''
    for name in paths:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text()
        for match in re.finditer(pattern, text):
            target = next(value for value in match.groups() if value is not None).strip()
            if target.startswith('<'):
                target = target[1:].split('>', 1)[0]
            else:
                target = re.split(r''' ["']''', target)[0]
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('//'):
                continue
            url = urlsplit(target)
            destination = path.parent / unquote(url.path) if url.path else path
            line = text[:match.start()].count('\n') + 1
            if not destination.exists():
                issues.append(f'{name}:{line}: missing path: {target}')
            elif url.fragment and destination.suffix == '.md' and unquote(url.fragment) not in anchors(destination):
                issues.append(f'{name}:{line}: missing anchor: {target}')
    return issues


def main():
    root = Path(__file__).resolve().parent.parent
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '*.md', '*.txt'],
        cwd=root, env=env,
    ).decode().split('\0')
    paths = sorted(set(filter(None, names)))
    issues = check(root, paths)
    for issue in issues:
        print(issue)
    print(f'{len(paths)} documentation files; {len(issues)} broken local references')
    return bool(issues)


if __name__ == '__main__':
    sys.exit(main())
