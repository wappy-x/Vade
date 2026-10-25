#!/usr/bin/env python3
"""Check a producer output pair and its complete manifest without extracting it."""

import hashlib
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile


def _check_members(tar):
    """Validate archive paths and member kinds before inspecting the manifest."""
    entries = tar.getmembers()
    members = {entry.name.rstrip("/"): entry for entry in entries}
    if len(members) != len(entries):
        raise ValueError("duplicate archive member")
    for name, entry in members.items():
        if name.startswith("/") or ".." in name.split("/") or not (entry.isfile() or entry.isdir()):
            raise ValueError("unsafe archive member: " + name)
    required = {"LICENSE", "NOTICE", "flash-serve", "flash.ptx", "vade", "vade-harness", "launcher-bundle.conf",
                "engine/LICENSE", "engine/NOTICE", "engine/crates/tokenizer/UNICODE-NOTICE.txt",
                "engine/crates/tokenizer/QWEN-NOTICE.txt"}
    if not required <= members.keys():
        raise ValueError("missing bundle member")
    if any(not members[name].isfile() for name in required):
        raise ValueError("bundle member is not a regular file")
    if not any(name.startswith("engine/licenses/") and entry.isfile() for name, entry in members.items()):
        raise ValueError("missing engine licenses")
    if any(name not in required and name != "engine" and not name.startswith("engine/") for name in members):
        raise ValueError("unexpected bundle member")
    return members


def _check_manifest(tar, members, short, forbidden):
    """Check source identity, payload hashes, modes and retained producer paths."""
    manifest_bytes = tar.extractfile("launcher-bundle.conf").read()
    lines = manifest_bytes.decode().splitlines()
    manifest = dict(line.split("=", 1) for line in lines)
    artifacts = {"engine": "flash-serve", "ptx": "flash.ptx", "harness": "vade-harness", "vade": "vade"}
    expected = {"format", "target", "engine_sha", "cli_sha", "harness_sha"} | {key + "_sha256" for key in artifacts}
    if len(lines) != len(manifest) or manifest.keys() != expected:
        raise ValueError("invalid manifest fields")
    if manifest["format"] != "vade-launch-bundle-v1" or manifest["target"] != "aarch64-unknown-linux-gnu":
        raise ValueError("wrong bundle format or target")
    revision = manifest["engine_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not revision.startswith(short):
        raise ValueError("invalid source revision")
    if manifest["cli_sha"] != revision or manifest["harness_sha"] != revision:
        raise ValueError("source revisions differ")
    for key, name in artifacts.items():
        data = tar.extractfile(name).read()
        if hashlib.sha256(data).hexdigest() != manifest[key + "_sha256"]:
            raise ValueError("manifest hash mismatch: " + name)
        if name != "flash.ptx" and members[name].mode & 0o111 != 0o111:
            raise ValueError("missing executable mode: " + name)
        for path in forbidden:
            if path.encode() in data or path.encode() in manifest_bytes:
                raise ValueError("producer path retained: " + name)


def check_bundle(directory, forbidden=()):
    """Return the verified archive path, or raise the first validation error."""
    directory = Path(directory)
    checksum = (directory / "SHA256SUMS").read_text()
    match = re.fullmatch(r"([0-9a-f]{64})  (vade-launch-bundle-v1-([0-9a-f]{7})-aarch64\.tar\.zst)\n", checksum)
    if not match:
        raise ValueError("invalid SHA256SUMS")
    digest, name, short = match.groups()
    archive = directory / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError("compressed archive hash mismatch")
    raw = subprocess.run(["zstd", "-q", "-d", "-c", str(archive)], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = _check_members(tar)
        _check_manifest(tar, members, short, forbidden)
    return archive


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("usage: python3 tools/check-bundle.py OUTPUT_DIR [FORBIDDEN_PATH ...]")
        print("bundle manifest and archive hashes: PASS (" + str(check_bundle(sys.argv[1], sys.argv[2:])) + ")")
    except (ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        sys.exit("bundle check: " + str(error))
