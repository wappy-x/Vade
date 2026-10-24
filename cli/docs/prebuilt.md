# Prebuilt installation

## One Spark: download and launch

The launcher bundle contains everything Vade needs to install its engine and
chat program on Linux aarch64. The Spark needs a working NVIDIA driver
(`nvidia-smi` and `libcuda.so.1`), `curl`, `sha256sum`, `tar` and `zstd`.
Rust and the CUDA toolkit are not needed to run the bundle.

Keep the existing checkpoint at
`~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`. Vade discovers existing weights;
the bundle does not contain or download them. See the
[first-run guide](../../harness/docs/first-run.md) for model selection and local
launcher behavior.

Use the download directory URL and exact bundle filename from a trusted
release. This repository is still a private preview; these placeholders do not
claim that a public release exists. Private release assets require authorized
access. Start in a new, empty directory so extraction cannot replace existing
files, then set these two values to the ones from that release:

```sh
mkdir vade-launch
cd vade-launch
RELEASE_URL='https://github.com/<owner>/<repo>/releases/download/<tag>'
BUNDLE='vade-launch-bundle-v1-<sha7>-aarch64.tar.zst'
```

The three commands are:

```sh
curl --fail --location --output "$BUNDLE" "$RELEASE_URL/$BUNDLE" && curl --fail --location --output SHA256SUMS "$RELEASE_URL/SHA256SUMS"
sha256sum -c SHA256SUMS && tar --zstd -xf "$BUNDLE"
./vade launch
```

The checksum covers the compressed archive, including the CLI and manifest.
Obtain the archive and checksum from the trusted release; the checksum detects
changed bytes but is not a publisher signature. The launcher finds
`launcher-bundle.conf` beside `vade`, verifies the engine, PTX and harness hashes,
and installs them under `~/.vade` (or an absolute `VADE_HOME`). Alternatively,
set `VADE_BUNDLE` to the absolute extracted bundle directory when using another
copy of `vade`. `VADE_BUNDLE` names a directory, not the compressed archive.

After the launcher reports ready, leave chat with `/exit` and verify the running
engine:

```sh
./vade verify
```

The verifier discovers the endpoint from the launcher's saved state, including
an alternate port. Use `--url http://127.0.0.1:18080` to override discovery. For
recovery, run `./vade status` and pass the endpoint from its `Vade is ready at ...`
line to `./vade verify --url`. `./vade down` stops the owned engine.

Run `./vade bench` with no flags and the same `VADE_HOME` and `VADE_CONFIG` as
the running installation. It discovers the recorded endpoint and saves
`./vade-bench.json` and its sibling `./vade-bench.txt` in the current directory,
replacing existing receipts. Use `--out FILE` (or `--output FILE`) to select
another receipt path. The default creates no installation or build directories
for receipt storage.

The benchmark uses the server's drafting policy as served;
`./vade bench --draft plain` requests plain decoding on every request without a
restart. Receipts retain the requested draft mode, the build SHA from `/health`,
and the server-reported MTP depth. See [benchmark options](../README.md#measure-a-running-server)
for the known FP4-Local health-depth reporting caveat and receipt details.

## Produce the launcher bundle

Build from a clean, committed checkout on Linux aarch64. Install the same
pinned toolchains used by CI:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
sh scripts/make-bundle.sh
```

The producer also requires Git, a C compiler (`cc`), binutils, GNU `tar`, `zstd`
and GNU `sha256sum`. It uses system build tools and the repository's Rust
standard-library-only crates; it adds no third-party crate dependencies. The
first nightly `build-std` invocation may fetch the compiler's own dependencies.
No GPU or CUDA toolkit is needed to emit PTX or build the bundle.

`sh scripts/make-bundle.sh [OUTPUT_DIR]` writes to `.local/bundles` by default.
The output directory must not already exist, even if empty. The producer reserves
it with a sibling lock, builds the archive and checksum in a temporary directory,
then publishes both together by renaming that directory. A failed build removes
its temporary files and lock; concurrent builds cannot replace an existing pair.
Choose another output directory to retain bundles from multiple builds.

It builds a `git archive` snapshot of the recorded commit, runs `cargo kernels`,
and builds release versions of `flash-serve`, `vade` and `vade-harness`. A
temporary CUDA stub with SONAME `libcuda.so.1` supplies symbols only while
linking. The stub is not shipped and no stub search path is embedded in the
release binaries; the Spark's driver supplies `libcuda.so.1` at runtime.
The producer removes build paths from compiled source locations and environment
constants, then rejects any binary, PTX or manifest that retains its build path.

The outputs are `vade-launch-bundle-v1-<sha7>-aarch64.tar.zst` and `SHA256SUMS`.
The archive extracts these files directly into the current directory:

| File | Purpose |
| --- | --- |
| `vade` | Linux aarch64 launcher CLI |
| `flash-serve` | Linux aarch64 inference engine |
| `flash.ptx` | Stamped production GPU kernels |
| `vade-harness` | Linux aarch64 chat program |
| `launcher-bundle.conf` | Launcher format, target, revisions and file hashes |
| `LICENSE`, `NOTICE`, `engine/` notices and licenses | Existing distribution notices |

The manifest matches the local launcher's `vade-launch-bundle-v1` format:

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

All three revision fields identify the same archived repository commit. They
record provenance; the current local consumer validates the format, target
and engine/PTX/harness hashes, without comparing revision fields to the CLI.

The [release workflow](../../.github/workflows/release.yml) runs on tag pushes
matching `v*` using `ubuntu-24.04-arm`. It installs the same pinned toolchains,
invokes this producer unchanged, checks `SHA256SUMS`, and attaches the bundle
and checksum to the tag's GitHub release. The workflow itself requires a
GitHub Actions run for validation; local checks cover its syntax and shared
producer, not a completed hosted release.

Pull-request CI runs `python3 tools/test-bundle-producer.py` to exercise the
producer's layout, manifest hashes, refusals, failure cleanup, publication
collisions and compiler path hygiene without compiling the full engine. The tag
workflow performs the full native build and checks both the archive checksum and
the extracted manifest hashes before publication.

## Controller prebuilt installation

The controller uses the separate `vade-prebuilt-v1` format described below.
The launcher producer above emits `vade-launch-bundle-v1`; it does not assemble
the controller's revision-pair directory.

`vade install` can copy trusted release binaries instead of compiling them.
Keep using the existing `prebuilt_dir` setting. Each bundle lives at
`<prebuilt_dir>/<engine-sha40>-<cli-sha40>` and contains:

| File | Runs on |
| --- | --- |
| `flash-serve` | Linux AArch64 Spark |
| `flash.ptx` | Spark CUDA driver |
| `vade` | Linux AArch64 Spark helper |
| `host-vade` (optional pair) | The computer running `vade install` |
| `host-vade-harness` (optional pair) | The computer running `vade chat` |

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

For the host pair, use the target matching the installation computer:
`aarch64-apple-darwin`, `x86_64-apple-darwin`,
`aarch64-unknown-linux-gnu`, or `x86_64-unknown-linux-gnu`.
Build every binary from the recorded source revision. The supplied directory
and manifest are the trust input; obtain them from a trusted build or release.
Local source checkouts still identify the selected revisions and engine
capabilities. The staged helper separately advertises its preparation support
through the existing read-only path query. Installation does not fetch a release
or download a model.

With both host artifacts present and verified, installation needs no local Rust
compiler and copies the harness and CLI into its new generation. Omitting all
host fields preserves the source build of those two tools. The Spark engine and
helper can still come from the verified bundle in that case.

Before staging, Vade checks the source revisions, host target, executable
architecture and SHA256 values. It verifies the copied bytes again before
activation. A rejected bundle produces one line explaining the fallback, then
uses the source build and its normal prerequisites. Failed source builds keep
the previous activation. Existing remote binaries are reused only for the same
machine SSH target and destination, after their hashes match again.

Model files stay in their existing directory. The shard-presence check uses a
cached filesystem inventory when unchanged; that cache does not replace trusted
publisher checksums or the engine's model verification.

Current engines also prepare the CUDA module cache during installation and
verify the existing checkpoint with the engine's normal checksum rules. Those
tasks run together, without inference or a listening server. Later starts reuse
the resulting caches while their file identities and checksums remain valid.
Installation prints the preparation times and verified byte count. It skips
preparation on a machine already running an owned server and preserves the
previous activation if preparation fails. Older engines or helpers keep their
ordinary installation path: preparation runs only when both the engine and the
verified staged helper support it. An older helper that omits the preparation
capability receives no preparation request, and installation prints one line
explaining that it is continuing without preparation.
