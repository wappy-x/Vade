# Prebuilt installation

Vade can run from ready-made binaries instead of compiling them. There are two
formats, and they serve different setups:

| Format | Use it when | In the v0.1.0 release |
| --- | --- | --- |
| Launcher bundle (`vade-launch-bundle-v1`) | You run Vade on one Spark | Yes |
| Controller bundle (`vade-prebuilt-v1`) | You install with `vade install` from a controller | No, you assemble it yourself |

Most of this page covers the launcher bundle. The
[last section](#prebuilt-binaries-for-a-controller-install) covers the
controller format. For other terms, see the [glossary](../../docs/GLOSSARY.md).

## One Spark: download and launch

The launcher bundle has everything Vade needs to install its engine and its
chat program (the harness) on Linux aarch64.

### What you need

Your Spark needs:

- a working NVIDIA driver (`nvidia-smi` and `libcuda.so.1`)
- `curl`, `sha256sum`, `tar` and `zstd`

You don't need Rust or the CUDA toolkit to run the bundle.

The bundle doesn't include model weights, and it doesn't download them. Vade
finds weights you already have, so keep your existing checkpoint at
`~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`. The
[first-run guide](../../harness/docs/first-run.md) explains how to choose a
model and how the local launcher behaves.

The bundle and its `SHA256SUMS` file are on the
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0).
If you'd rather build from source, follow [INSTALL.md](../../INSTALL.md), or
make the bundle yourself as described in
[Produce the launcher bundle](#produce-the-launcher-bundle).

### Download, check and unpack

Start in a new, empty directory, so that unpacking can't replace any existing
files. Then set two values. The bundle filename is the one listed in that
release's `SHA256SUMS`:

```sh
mkdir vade-launch
cd vade-launch
RELEASE_URL='https://github.com/wappy-x/vade/releases/download/v0.1.0'
BUNDLE='vade-launch-bundle-v1-5a22651-aarch64.tar.zst'
```

Now run three commands. The first downloads both files, the second checks the
archive and unpacks it, and the third starts Vade:

```sh
curl --fail --location --output "$BUNDLE" "$RELEASE_URL/$BUNDLE" && curl --fail --location --output SHA256SUMS "$RELEASE_URL/SHA256SUMS"
sha256sum -c SHA256SUMS && tar --zstd -xf "$BUNDLE"
./vade launch
```

The checksum covers the whole compressed archive, including the CLI and the
manifest inside it. It catches changed bytes, but it isn't a publisher
signature, so always get the archive and the checksum from the same release.

### What `./vade launch` does

The launcher reads `launcher-bundle.conf`, the manifest next to `vade`. It
verifies the hashes of the engine, the PTX file (the GPU kernels) and the
harness. It then installs them under `~/.vade`, or under `VADE_HOME` if you set
it to an absolute path.

If you run a different copy of `vade`, set `VADE_BUNDLE` to the absolute path
of the unpacked bundle directory instead. `VADE_BUNDLE` names a directory, not
the compressed archive.

### Check the running engine

When the launcher says it's ready, leave chat with `/exit` and verify the
running engine:

```sh
./vade verify
```

The verifier finds the endpoint from the launcher's saved state, even on a
non-default port. You can override it with `--url http://127.0.0.1:18080`.

If you need to recover, run `./vade status`. It prints a
`Vade is ready at ...` line; pass that endpoint to `./vade verify --url`.

To stop the engine that Vade owns, run `./vade down`.

### Run a benchmark

Run `./vade bench` with no flags. Use the same `VADE_HOME` and `VADE_CONFIG`
as the running engine. The benchmark finds the endpoint by itself.

It writes its results (receipts) to `./vade-bench.json` and `./vade-bench.txt`
in the current folder, and replaces any older ones. To write them somewhere
else, use `--output FILE` (`--out FILE` is the same flag). It doesn't create any install or
build folders to hold them.

By default the benchmark uses whatever drafting the server is set to. In
drafting, the model's multi-token-prediction (MTP) head proposes tokens and
the full model checks them. To measure plain decoding instead, run
`./vade bench --draft plain`. It asks for plain decoding on every request, so
you don't need to restart the server.

Each receipt records:

- the draft mode you asked for
- the build SHA from `/health`
- the drafting depth the server reports

With FP4-Local checkpoints, the drafting depth that `/health` reports has a
known caveat. [Benchmark options](../README.md#measure-a-running-server)
explains it and has more on receipts.

## What's in the launcher bundle

The producer script (see the next section) writes two files: the archive,
`vade-launch-bundle-v1-<sha7>-aarch64.tar.zst`, and `SHA256SUMS`. The archive
unpacks these files straight into the current directory:

| File | What it is |
| --- | --- |
| `vade` | Linux aarch64 launcher CLI |
| `flash-serve` | Linux aarch64 inference engine |
| `flash.ptx` | The production GPU kernels as PTX, stamped with the build they came from |
| `vade-harness` | Linux aarch64 chat program |
| `launcher-bundle.conf` | Launcher format, target, revisions and file hashes |
| `LICENSE`, `NOTICE`, `engine/` notices and licenses | The license and notice files that ship with the source |

The manifest, `launcher-bundle.conf`, matches the local launcher's
`vade-launch-bundle-v1` format:

```ini
format=vade-launch-bundle-v1
target=aarch64-unknown-linux-gnu
engine_sha=<full-source-sha40>
cli_sha=<full-source-sha40>
harness_sha=<full-source-sha40>
engine_sha256=<flash-serve-SHA256>
ptx_sha256=<flash.ptx-SHA256>
harness_sha256=<vade-harness-SHA256>
vade_sha256=<vade-SHA256>
```

All three revision fields (`engine_sha`, `cli_sha` and `harness_sha`) name the
same archived repository commit. They record provenance: where the build came
from.

Today the local launcher checks the format, the target, and the engine, PTX
and harness hashes. It doesn't compare the revision fields with the CLI.

## Produce the launcher bundle

### Install the toolchains

Build from a clean, committed checkout on Linux aarch64. Install the same
pinned toolchains that CI uses, then run the script:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
sh scripts/make-bundle.sh
```

The producer script also needs:

- Git
- a C compiler (`cc`)
- binutils
- GNU `tar`
- `zstd`
- GNU `sha256sum`

It uses system build tools and the repository's own Rust crates. Those crates
use only the standard library, so the build adds no third-party crate
dependencies. The first nightly `build-std` run may fetch the compiler's own
dependencies.

You don't need a GPU or the CUDA toolkit to emit PTX or build the bundle.

### Choose where the output goes

`sh scripts/make-bundle.sh [OUTPUT_DIR]` writes to `.local/bundles` unless you
name another folder. That folder must not exist yet, not even as an empty
folder. To keep bundles from several builds, give each build its own folder.

While it runs:

- It takes a lock next to the output folder.
- It builds the archive and checksum in a temporary folder, then renames that
  folder into place, so both files appear together.
- If the build fails, it removes its temporary files and the lock.
- Two builds running at once can't overwrite a finished pair.

### What the script builds

The script snapshots the recorded commit with `git archive` and runs
`cargo kernels`. It then builds release versions of `flash-serve`, `vade` and
`vade-harness`.

Linking needs `libcuda.so.1`, so the script uses a temporary stand-in with that
name that only provides symbols. The stand-in isn't shipped, and the binaries
don't look for it. On the Spark, the NVIDIA driver supplies the real library.

The script also strips build-machine paths from the binaries. It refuses any
binary, PTX or manifest that still contains one.

### How releases are built and tested

The [release workflow](../../.github/workflows/release.yml) runs on tag pushes
matching `v*`, on `ubuntu-24.04-arm`. It installs the same pinned toolchains,
runs the same script unchanged, and checks `SHA256SUMS`. It then attaches the
bundle and checksum to the tag's GitHub release. The tag workflow does the
full native build, and it checks both the archive checksum and the extracted
manifest hashes before it publishes.

Pull-request CI runs `python3 tools/test-bundle-producer.py` without
compiling the full engine. It tests that the producer:

- lays out the files and writes the manifest hashes correctly
- refuses the cases it should refuse, and cleans up after a failure
- can't overwrite a finished bundle when two builds collide
- leaves no build-machine paths in its output

The v0.1.0 bundle was validated by CI and by fresh installs on home and cloud
Sparks.

## Prebuilt binaries for a controller install

The controller is the computer that runs `vade install`. It uses a separate
format, `vade-prebuilt-v1`. The `make-bundle.sh` script above builds only the
launcher bundle (`vade-launch-bundle-v1`). It doesn't build a controller
bundle.

The v0.1.0 release doesn't include a `vade-prebuilt-v1` bundle. Unless you
assemble one yourself, a controller install builds from source on each Spark.

### What a controller bundle contains

With a controller bundle, `vade install` can copy its binaries instead of
compiling them. Point it at them with the existing `prebuilt_dir` setting.

Each bundle is a folder named after one engine revision and one CLI revision,
`<prebuilt_dir>/<engine-sha40>-<cli-sha40>`. It contains:

| File | Where it runs |
| --- | --- |
| `flash-serve` | The Spark (Linux aarch64) |
| `flash.ptx` | The Spark, loaded by its CUDA driver |
| `vade` | The Spark, as the helper: the copy of the CLI that install runs there (Linux aarch64) |
| `host-vade` (optional pair) | The computer that runs `vade install` |
| `host-vade-harness` (optional pair) | The computer that runs `vade chat` |

The bundle's `manifest.conf` records its source revisions and hashes:

```ini
format=vade-prebuilt-v1
target=aarch64-unknown-linux-gnu
engine_sha=<full-engine-source-sha>
cli_sha=<full-cli-source-sha>
flash-serve.sha256=<SHA256>
flash.ptx.sha256=<SHA256>
vade.sha256=<SHA256>
host_target=aarch64-apple-darwin
harness_sha=<full-harness-source-sha>
host-vade.sha256=<SHA256>
host-vade-harness.sha256=<SHA256>
```

Build every binary from the recorded source revision. Vade trusts the
directory and manifest you supply, so get them from a trusted build or
release. Vade still takes the revisions to install, and what the engine
supports, from your local source checkouts. Installation doesn't fetch
a release or download a model.

### The optional host pair

The host pair is `host-vade` and `host-vade-harness`. Use the host target that
matches the computer you install from: `aarch64-apple-darwin`,
`x86_64-apple-darwin`, `aarch64-unknown-linux-gnu`, or
`x86_64-unknown-linux-gnu`.

If both host files are present and pass their checks, install needs no Rust
compiler on your computer. It copies the chat program and CLI into the new
installed build.

If you leave out all the host fields, those two are built from source. The
engine and helper can still come from the bundle.

### Checks and fallback

Before copying, Vade checks the source revisions, the host target, each
binary's architecture and the SHA256 values. It checks the copies again before
switching to them.

If the bundle fails a check, Vade prints one line saying why and builds from
source instead, with the usual prerequisites. If that build fails, your
previous install stays active.

Vade reuses binaries already on a Spark only when the Spark's SSH target and
install folder are unchanged and the binaries' hashes still match.

### Model files and install-time checks

Install leaves model files where they are. It checks that the checkpoint's
shards are present, and when nothing has changed it reuses a saved list of the
files instead of scanning the folder again. That saved list is only a
speed-up: it doesn't replace trusted publisher checksums or the engine's own
model checks.

During install, current engines also do two jobs side by side:

- They fill the CUDA module cache.
- They check the checkpoint against the engine's usual checksums.

Neither job runs the model or opens a port. Later starts reuse the results
while the files and checksums are unchanged. Install prints how long this took
and how many bytes it checked.

Install skips these jobs on a Spark that is already running an engine Vade
started. If they fail, install keeps the previous install.

They also need an engine and a helper on the Spark that both support them. The
helper reports whether it does when Vade queries it. With an older engine or
helper, install skips the jobs and carries on as usual. With an older helper,
it also prints one line saying it is continuing without this step.
