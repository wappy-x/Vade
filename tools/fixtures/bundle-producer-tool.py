"""Simulate expensive or platform-specific tools for bundle producer tests.

The test harness runs this file through tool-name symlinks. Compiler calls
still pass through the producer's real wrapper; the probe checks its environment
and arguments. These tools never build or execute a real inference engine.
"""

import hashlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tarfile

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
