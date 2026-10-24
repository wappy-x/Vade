#!/usr/bin/env python3
"""Fast producer regression: real shell/layout/hashing, simulated build tools.

No Rust, GPU, Git writes or third-party Python packages are needed. The tag
workflow separately builds real artifacts and runs check-bundle.py on them.
"""

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("bundle_check", ROOT / "tools/check-bundle.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)

# Simulate only expensive/platform-specific tools. Compiler calls still pass
# through the producer's real wrapper; the probe verifies its env and arguments.
TOOL = r'''
import hashlib, os, pathlib, shutil, signal, subprocess, sys, tarfile
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["TEST_ROOT"])
output = pathlib.Path(os.environ["TEST_OUTPUT"])
mode = os.environ.get("TEST_MODE", "")
if name == "uname":
    print("Linux" if args == ["-s"] else "aarch64")
elif name == "git":
    if args[0] == "status":
        print(" M tracked" if mode == "dirty" else "", end="")
    elif args[0] == "rev-parse":
        print("1234567890abcdef1234567890abcdef12345678")
    elif args[0] == "show":
        print("1700000000")
    elif args[0] == "archive":
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as tar:
            for file in sorted((root / "fixture").iterdir()):
                tar.add(file, arcname=file.name)
    else:
        sys.exit("unexpected git invocation")
elif name == "rustup":
    pass
elif name == "cc":
    pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"link-only stub")
elif name == "cargo":
    if mode == "build-fails":
        sys.exit(42)
    if mode == "interrupt":
        os.kill(os.getppid(), signal.SIGTERM)
        sys.exit(0)
    if "kernels" in args:
        if mode in ("destination", "empty-destination"):
            output.mkdir()
            if mode == "destination":
                (output / "existing").write_text("keep me")
        if mode == "concurrent":
            attempt = subprocess.run(["sh", str(root / "scripts/make-bundle.sh"), str(output)], capture_output=True)
            assert attempt.returncode != 0 and b"reserved" in attempt.stderr, attempt
        target = pathlib.Path.cwd() / "target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx"
    else:
        target = pathlib.Path(os.environ["CARGO_TARGET_DIR"]) / "release" / args[args.index("--bin") + 1]
    env = dict(os.environ, CARGO_MANIFEST_DIR=str(pathlib.Path.cwd()),
               CARGO_MANIFEST_PATH=str(pathlib.Path.cwd() / "Cargo.toml"), TEST_ARTIFACT=str(target))
    subprocess.run([os.environ["RUSTC_WRAPPER"], str(root / "bin/rustc-probe"), "--crate-name", "probe"], env=env, check=True)
elif name == "rustc-probe":
    source = os.environ["VADE_BUNDLE_SOURCE"]
    work = os.environ["VADE_BUNDLE_BUILD"]
    assert os.environ["CARGO_MANIFEST_DIR"].startswith("./"), os.environ["CARGO_MANIFEST_DIR"]
    assert os.environ["CARGO_MANIFEST_PATH"].startswith("./"), os.environ["CARGO_MANIFEST_PATH"]
    assert args[-3:] == ["--remap-path-prefix=" + os.environ["HOME"] + "=/build-home",
                         "--remap-path-prefix=" + work + "=/vade-build",
                         "--remap-path-prefix=" + source + "=."]
    data = ("portable artifact " + os.environ["CARGO_MANIFEST_DIR"]).encode()
    if mode == "path-leak":
        data += work.encode()
    if mode == "remapped-path-leak":
        data += pathlib.Path(work).name.encode()
    target = pathlib.Path(os.environ["TEST_ARTIFACT"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
elif name == "readelf":
    if args[0] == "-h":
        print("Class: ELF64\nMachine: " + ("X86-64" if mode == "wrong-arch" else "AArch64"))
    else:
        print("(NEEDED) Shared library: [libcuda.so.1]")
        if mode == "rpath":
            print("(RUNPATH) producer-path")
elif name == "tar":
    if "-xf" in args:
        with tarfile.open(args[args.index("-xf") + 1]) as tar:
            tar.extractall(args[args.index("-C") + 1], filter="data")
    else:
        position = args.index("-cf")
        with tarfile.open(args[position + 1], "w", format=tarfile.GNU_FORMAT) as tar:
            for file in args[position + 2:]:
                tar.add(file, arcname=file)
elif name == "zstd":
    if "-d" in args:
        sys.stdout.buffer.write(pathlib.Path(args[-1]).read_bytes())
    else:
        shutil.copyfile(args[args.index("-10") + 1], args[args.index("-o") + 1])
elif name == "sha256sum":
    if args[0] == "-c":
        for line in pathlib.Path(args[1]).read_text().splitlines():
            digest, file = line.split("  ", 1)
            assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest() == digest
            print(file + ": OK")
    else:
        print(hashlib.sha256(pathlib.Path(args[0]).read_bytes()).hexdigest() + "  " + args[0])
elif name == "mv":
    if mode in ("rename-race", "empty-rename-race"):
        output.mkdir()
        if mode == "rename-race":
            (output / "existing").write_text("keep me")
    # Linux CI uses GNU mv itself. macOS's mv lacks -T: emulate its no-clobber
    # directory behavior for the platform-independent layout unit tests.
    if sys.platform != "darwin":
        os.execv(os.environ["TEST_REAL_MV"], ["mv"] + args)
    source, destination = map(pathlib.Path, args[-2:])
    if not destination.exists() and not destination.is_symlink():
        source.rename(destination)
else:
    sys.exit("unexpected fake build tool: " + name)
'''


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
        script.write_text("#!" + sys.executable + "\n" + TOOL)
        script.chmod(0o755)
        for name in ("uname", "git", "cargo", "rustup", "cc", "readelf", "tar", "zstd", "sha256sum", "rustc-probe", "mv"):
            (tools / name).symlink_to("tool")
        self.output = self.root / "published"
        self.env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
                        TEST_ROOT=str(self.root), TEST_OUTPUT=str(self.output), TEST_REAL_MV=shutil.which("mv"))
        self.previous_path = os.environ["PATH"]
        os.environ["PATH"] = self.env["PATH"]
        self.addCleanup(os.environ.__setitem__, "PATH", self.previous_path)
        # The checker uses the same transparent compression fixture, but checks
        # archive bytes, member hashes and manifest fields independently.
        for name in ("TEST_ROOT", "TEST_OUTPUT", "TEST_REAL_MV"):
            previous = os.environ.get(name)
            os.environ[name] = self.env[name]
            self.addCleanup(self.restore_env, name, previous)

    @staticmethod
    def restore_env(name, previous):
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous

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

    def test_checker_rejects_corrupt_archive_and_manifest_hash(self):
        self.produce()
        archive = self.check_pair()
        original = archive.read_bytes()
        archive.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "compressed archive hash"):
            self.check_pair()
        # Repackage corrupted payload with a valid outer checksum: the manifest
        # check must still reject it, just as the launcher would.
        target = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(original)) as old, tarfile.open(fileobj=target, mode="w") as new:
            for entry in old.getmembers():
                data = old.extractfile(entry).read() if entry.isfile() else None
                if entry.name == "flash.ptx":
                    data += b"corruption"
                    entry.size = len(data)
                new.addfile(entry, io.BytesIO(data) if data is not None else None)
        archive.write_bytes(target.getvalue())
        (self.output / "SHA256SUMS").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name + "\n")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            self.check_pair()


if __name__ == "__main__":
    unittest.main()
