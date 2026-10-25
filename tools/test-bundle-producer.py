#!/usr/bin/env python3
"""Fast bundle regressions: real shell/layout/hashing, simulated build tools.

No Rust, GPU, Git writes or third-party Python packages are needed. The tag
workflow separately builds real artifacts and runs check-bundle.py on them.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("bundle_check", ROOT / "tools/check-bundle.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

TOOL = ROOT / "tools/fixtures/bundle-producer-tool.py"


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bundle producer ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/make-bundle.sh", self.root / "scripts/make-bundle.sh")
        for name in ("LICENSE", "NOTICE", "engine/LICENSE", "engine/NOTICE", "engine/licenses/LICENSE",
                     "engine/crates/tokenizer/UNICODE-NOTICE.txt", "engine/crates/tokenizer/QWEN-NOTICE.txt",
                     "cli/Cargo.toml", "harness/Cargo.toml"):
            file = self.root / "fixture" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("fixture notice or manifest\n")
        (self.root / "fixture/engine/crates/cuda.rs").write_text("fn cuInit(flags: u32);\n")
        tools = self.root / "bin"
        tools.mkdir()
        script = tools / "tool"
        script.write_text("#!" + sys.executable + "\n" + TOOL.read_text())
        script.chmod(0o755)
        for name in ("uname", "git", "cargo", "rustup", "cc", "readelf", "tar", "zstd", "sha256sum", "rustc-probe", "mv"):
            (tools / name).symlink_to("tool")
        self.output = self.root / "published"
        self.env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                        TEST_ROOT=str(self.root), TEST_OUTPUT=str(self.output), TEST_REAL_MV=shutil.which("mv"))
        # The checker uses the same transparent compression fixture, but checks
        # archive bytes, member hashes and manifest fields independently.
        self.enterContext(patch.dict(os.environ, {name: self.env[name] for name in
                                                 ("PATH", "TEST_ROOT", "TEST_OUTPUT", "TEST_REAL_MV")}))

    def produce(self, mode="", succeeds=True):
        result = subprocess.run(["sh", str(self.root / "scripts/make-bundle.sh"), str(self.output)],
                                env=dict(self.env, TEST_MODE=mode), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
        self.assertFalse(Path(str(self.output) + ".lock").exists())
        self.assertEqual(list(self.root.glob(".bundle-publish.*")), [])
        self.assertEqual(list((self.root / ".local").glob("bundle-build.*")), [])
        return result

    def check_pair(self):
        archive = CHECK.check_bundle(self.output, (str(self.root), os.environ["HOME"], "bundle-build."))
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["SHA256SUMS", archive.name])
        return archive

    def test_layout_manifest_hashes_and_compiler_path_sanitization(self):
        self.produce()
        self.check_pair()

    def test_existing_pair_and_empty_directory_are_preserved(self):
        self.produce()
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.produce(succeeds=False)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})
        shutil.rmtree(self.output)
        self.output.mkdir()
        self.produce(succeeds=False)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_dangling_output_symlink_is_preserved(self):
        self.output.symlink_to(self.root / "absent")
        self.produce(succeeds=False)
        self.assertTrue(self.output.is_symlink())

    def test_competing_producer_cannot_publish(self):
        self.produce("concurrent")
        self.check_pair()

    def test_destination_appearing_during_build_is_preserved(self):
        for mode in ("destination", "empty-destination", "rename-race", "empty-rename-race"):
            with self.subTest(mode=mode):
                self.produce(mode, succeeds=False)
                if mode in ("destination", "rename-race"):
                    self.assertEqual((self.output / "existing").read_text(), "keep me")
                else:
                    self.assertEqual(list(self.output.iterdir()), [])
                shutil.rmtree(self.output)

    def test_refusals_and_failure_cleanup(self):
        for mode in ("dirty", "build-fails", "interrupt", "path-leak", "remapped-path-leak", "wrong-arch", "rpath"):
            with self.subTest(mode=mode):
                self.produce(mode, succeeds=False)
                self.assertFalse(self.output.exists())


def load_tests(loader, tests, pattern):
    """Keep checker regressions in the existing CI entry point after the split."""
    spec = importlib.util.spec_from_file_location("bundle_checker_tests", ROOT / "tools/test-bundle-checker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests.addTests(loader.loadTestsFromModule(module))
    return tests


if __name__ == "__main__":
    unittest.main()
