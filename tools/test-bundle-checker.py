#!/usr/bin/env python3
"""Bundle checker regressions using in-memory tar fixtures, without build tools.

Only zstd decompression is simulated. Archive parsing, layout checks, hashes,
manifest validation and producer-path checks use the real checker.
"""

import hashlib
import importlib.util
import io
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("bundle_check", Path(__file__).with_name("check-bundle.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
REVISION = "1234567890abcdef1234567890abcdef12345678"


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bundle checker ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / ("vade-launch-bundle-v1-" + REVISION[:7] + "-aarch64.tar.zst")
        self.files = {name: b"notice\n" for name in
                      ("LICENSE", "NOTICE", "engine/LICENSE", "engine/NOTICE", "engine/licenses/LICENSE",
                       "engine/crates/tokenizer/UNICODE-NOTICE.txt", "engine/crates/tokenizer/QWEN-NOTICE.txt")}
        self.files.update({name: ("portable " + name).encode() for name in
                           ("flash-serve", "flash.ptx", "vade-harness", "vade")})
        self.manifest = dict(format="vade-launch-bundle-v1", target="aarch64-unknown-linux-gnu",
                             engine_sha=REVISION, cli_sha=REVISION, harness_sha=REVISION)
        for key, name in (("engine", "flash-serve"), ("ptx", "flash.ptx"), ("harness", "vade-harness"), ("vade", "vade")):
            self.manifest[key + "_sha256"] = hashlib.sha256(self.files[name]).hexdigest()

    def manifest_bytes(self, **changes):
        fields = dict(self.manifest, **changes)
        return "".join(key + "=" + value + "\n" for key, value in fields.items()).encode()

    def write_bundle(self, *, manifest=None, extra=(), modes=None, types=None):
        files = dict(self.files)
        files["launcher-bundle.conf"] = manifest if manifest is not None else self.manifest_bytes()
        target = io.BytesIO()
        with tarfile.open(fileobj=target, mode="w") as tar:
            for name, data in list(files.items()) + list(extra):
                entry = tarfile.TarInfo(name)
                entry.type = (types or {}).get(name, tarfile.REGTYPE)
                entry.size = len(data) if entry.isfile() else 0
                entry.mode = (modes or {}).get(name, 0o755)
                tar.addfile(entry, io.BytesIO(data) if entry.isfile() else None)
        self.archive.write_bytes(target.getvalue())
        self.write_checksum()

    def write_checksum(self):
        (self.root / "SHA256SUMS").write_text(hashlib.sha256(self.archive.read_bytes()).hexdigest() + "  " + self.archive.name + "\n")

    def check(self, forbidden=()):
        with patch.object(CHECK.subprocess, "run") as decompress:
            decompress.return_value.stdout = self.archive.read_bytes()
            return CHECK.check_bundle(self.root, forbidden)

    def test_valid_archive_is_checked_without_extraction(self):
        self.write_bundle()
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        self.assertEqual(self.check(), self.archive)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_checksum_rejects_corruption_and_malformed_lines(self):
        self.write_bundle()
        self.archive.write_bytes(self.archive.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "^compressed archive hash mismatch$"):
            self.check()
        self.write_checksum()
        checksum = self.root / "SHA256SUMS"
        valid = checksum.read_text()
        for text in ("", valid.rstrip("\n"), valid + valid, valid.replace("  ", " ")):
            with self.subTest(checksum=text):
                checksum.write_text(text)
                with self.assertRaisesRegex(ValueError, "^invalid SHA256SUMS$"):
                    self.check()

    def test_archive_rejects_duplicate_unsafe_and_unexpected_members(self):
        for name, error in (("LICENSE", "duplicate archive member"), ("LICENSE/", "duplicate archive member"),
                            ("/absolute", "unsafe archive member: /absolute"),
                            ("../escape", "unsafe archive member: ../escape"), ("stray", "unexpected bundle member")):
            with self.subTest(name=name):
                self.write_bundle(extra=((name, b"stray"),))
                with self.assertRaisesRegex(ValueError, "^" + re.escape(error) + "$"):
                    self.check()

    def test_archive_requires_payload_and_licenses(self):
        for name, error in (("vade", "missing bundle member"), ("engine/licenses/LICENSE", "missing engine licenses")):
            with self.subTest(name=name):
                data = self.files.pop(name)
                self.write_bundle()
                self.files[name] = data
                with self.assertRaisesRegex(ValueError, "^" + re.escape(error) + "$"):
                    self.check()

    def test_archive_rejects_links_and_required_directories(self):
        for kind, error in ((tarfile.SYMTYPE, "unsafe archive member: vade"),
                            (tarfile.LNKTYPE, "unsafe archive member: vade"),
                            (tarfile.DIRTYPE, "bundle member is not a regular file")):
            with self.subTest(kind=kind):
                self.write_bundle(types={"vade": kind})
                with self.assertRaisesRegex(ValueError, "^" + error + "$"):
                    self.check()

    def test_archive_errors_precede_manifest_errors(self):
        self.write_bundle(manifest=b"malformed", extra=(("LICENSE", b"duplicate"),))
        with self.assertRaisesRegex(ValueError, "^duplicate archive member$"):
            self.check()
        del self.files["vade"]
        self.write_bundle(manifest=b"malformed")
        with self.assertRaisesRegex(ValueError, "^missing bundle member$"):
            self.check()

    def test_manifest_fields_and_source_identity_are_checked(self):
        for key, value, error in (("format", "unknown", "wrong bundle format or target"),
                                  ("target", "unknown", "wrong bundle format or target"),
                                  ("engine_sha", "invalid", "invalid source revision"),
                                  ("engine_sha", "f" * 40, "invalid source revision"),
                                  ("cli_sha", "f" * 40, "source revisions differ"),
                                  ("harness_sha", "f" * 40, "source revisions differ"),
                                  ("extra", "field", "invalid manifest fields")):
            with self.subTest(field=key, value=value):
                self.write_bundle(manifest=self.manifest_bytes(**{key: value}))
                with self.assertRaisesRegex(ValueError, "^" + error + "$"):
                    self.check()
        manifest = self.manifest_bytes()
        for invalid in (manifest + b"engine_sha=" + REVISION.encode() + b"\n",
                        manifest.replace(b"format=vade-launch-bundle-v1\n", b"")):
            with self.subTest(manifest=invalid):
                self.write_bundle(manifest=invalid)
                with self.assertRaisesRegex(ValueError, "^invalid manifest fields$"):
                    self.check()

    def test_payload_hashes_modes_and_producer_paths_are_checked(self):
        original = self.files["flash.ptx"]
        self.files["flash.ptx"] += b"corruption"
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, r"^manifest hash mismatch: flash\.ptx$"):
            self.check()
        self.files["flash.ptx"] = original
        for mode in (0o644, 0o700, 0o750):
            with self.subTest(mode=mode):
                self.write_bundle(modes={"vade": mode})
                with self.assertRaisesRegex(ValueError, "^missing executable mode: vade$"):
                    self.check()
        self.write_bundle(modes={"flash.ptx": 0o644})
        self.assertEqual(self.check(), self.archive)
        for forbidden in ("portable", "aarch64-unknown-linux-gnu"):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "^producer path retained: flash-serve$"):
                    self.check((forbidden,))


if __name__ == "__main__":
    unittest.main()
