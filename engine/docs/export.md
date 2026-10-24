# Export

The engine exporter is a retained exact-file snapshot utility. It copies only
allowlisted files and never commits, pushes or publishes.

**The bundled allowlist is historical, not a current release manifest.** It
predates repository consolidation, omits newer modules and still names an
engine-local `rust-toolchain.toml`; the active compiler pin now lives at the
[repository root](../../rust-toolchain.toml). The current source is distributed
as this complete repository. Do not use the historical allowlist to prepare a
current build or public squash.

The procedure below describes the utility for maintainers who first prepare a
complete source tree and updated allowlist, including its compiler pin. A
preview must pass its own build before it can be treated as a usable snapshot.

<details>
<summary><b>Terms on this page</b>: allowlist, denylist, manifest, SHA-256 hash, preview and release export, PTX</summary>

<dl>
<dt>Allowlist</dt>
<dd><code>scripts/export-allowlist.txt</code>, the files that may be
copied.</dd>
<dt>Denylist</dt>
<dd>A private file of local names that must never be exported.</dd>
<dt>Manifest</dt>
<dd><code>EXPORT-MANIFEST.json</code>: source revision and each file's
hash.</dd>
<dt>SHA-256 hash</dt>
<dd>A fingerprint of a file's bytes; any change alters it.</dd>
<dt>Preview export</dt>
<dd>A trial copy that may include uncommitted files; never a release.</dd>
<dt>Release export</dt>
<dd>A copy made from committed source, recorded in the manifest.</dd>
<dt>PTX</dt>
<dd>NVIDIA's portable GPU instruction format; the kernel build makes it.</dd>
</dl>

</details>

## The rules

**Only allowlisted files leave, the histories stay separate, and only a
person publishes.**

- **Separate histories.** Never mirror or merge the private history into the
  shareable repository.
- **Allowlist only.** Only the exact files named in the allowlist are copied.
- **Never copied.** Git metadata, build output, model weights, secrets and
  private development notes. The public root `AGENTS.md` and `CLAUDE.md` are
  contributor documentation and belong in a current source export; include
  them when preparing its updated allowlist.
- **No commits.** The tool never commits or pushes. Publishing, and any
  licensing change, are owner decisions.

## Make a preview

**A preview needs no commit.** Use it for clean builds and setup checks.

### Do it yourself

1. Create the denylist: an untracked, nonempty file outside the source tree,
   one private name per line (host aliases, usernames, other local names).
   Lines starting with `#` are ignored. Point `PRIVATE_DENYLIST` at it.
2. After refreshing the complete source tree and allowlist, from that source
   root write the preview:

   ```sh
   cargo run -p vade-export --offline -- --source . \
     --destination ./export-out --denylist "$PRIVATE_DENYLIST" --preview
   ```

3. Check that it builds:

   ```sh
   (cd export-out && CARGO_TARGET_DIR=../target/export-check \
     cargo check --workspace --exclude spark-rust-kernels --offline)
   ```

### Ask your coding agent

```text
Make a preview export of this checkout by following the "Make a preview"
section of docs/export.md. PRIVATE_DENYLIST is already set to a private file
outside the checkout; never print its contents. Write the preview to
./export-out, then run the workspace check inside it. Report the exporter's
summary line and any rejected file. Do not commit, push or publish anything.
```

<details>
<summary><b>What a preview allows</b>, and what it cannot do</summary>

- Uncommitted and untracked allowlisted files, and an output folder inside
  the source tree.
- Its manifest says `"preview": true` and `"source_revision": null`.
- It cannot overwrite a release export, and its output must never become an
  input.

</details>

## Make a release snapshot

**Only after the source has been reviewed and committed.**

### Do it yourself

1. Commit and review the source. The tracked source must be clean, and the
   allowlist entries committed.
2. Set `EXPORT_DIRECTORY` to the separate shareable checkout.
3. From the committed tree, run the exporter without `--preview`:

   ```sh
   cargo run -p vade-export --offline -- --source . \
     --destination "$EXPORT_DIRECTORY" --denylist "$PRIVATE_DENYLIST"
   ```

4. Check the workspace in the exported tree, and inspect the exported diff.
5. Only then commit in the shareable repository, and review the author and
   message.

The manifest records the source revision and every file's hash. Each file's
bytes must match that revision. No remote URL is recorded.

### Ask your coding agent

```text
Make a release export of the committed checkout by following the "Make a
release snapshot" section of docs/export.md. PRIVATE_DENYLIST and
EXPORT_DIRECTORY are already set; never print the denylist. Stop if the
tracked source has uncommitted changes. After the export, run the workspace
check in EXPORT_DIRECTORY and show the diff of the exported files. Do not
commit, push or publish anything.
```

## What the exporter checks

**Both modes reject unsafe paths, secrets, local details, most network
addresses, binary files and unknown files in the destination.** Every check
finishes before any file changes.

<details>
<summary><b>Every check</b>, the two binary exceptions, and the run order</summary>

<dl>
<dt>Paths</dt>
<dd>Unsafe paths and symlinks, including symlinked destinations.</dd>
<dt>Secrets</dt>
<dd>Private keys, access tokens and credential assignments.</dd>
<dt>Local details</dt>
<dd>Personal absolute paths, MAC addresses and denylist names.</dd>
<dt>Network addresses</dt>
<dd>IPv4 literals, except loopback, wildcard and documentation ones.</dd>
<dt>Binary files</dt>
<dd>Anything that is not UTF-8 text, except the two cases below.</dd>
<dt>Destination</dt>
<dd>Unknown files in the destination, and output edited by hand.</dd>
</dl>

The two binary exceptions:

- **Reference score rows:** allowlisted `.f32` files under `fixtures/`. Each
  must hold at least 32,000 finite floats, none larger than 1,000,000 in
  absolute value. Text patterns are skipped, but denylist names are still
  checked in any readable run of 12 or more characters. Review their model
  and vocabulary size separately.
- **PNG images:** only pixels, with no text or metadata chunks.

The fixtures' JSON prompts are made-up test conversations. No checkpoint
weights are exported.

Run order: the denylist must sit outside both trees; every check finishes
before any file changes; the manifest is written last, so an interrupted run
cannot pass for a complete snapshot.

</details>

## What still needs a person

**Pattern checks support review; they do not replace it.**

- Review every newly allowlisted file.
- Review the shareable repository's author and commit messages.
- Keep build output outside any destination you intend to regenerate.

<details>
<summary><b>Build notes</b>, and what the export includes besides the engine</summary>

- **All hosts.** From `engine/` in a complete repository export, build the GPU
  kernels with `cargo +1.99.0 kernels` before the workspace checks in
  [Validation](validation.md#on-any-computer-host-checks). Stale PTX is rejected
  on macOS as well as Linux.
- **Mac.** Host checks need no GPU and do not validate Linux CUDA execution.

The Python official-model oracle is kept privately outside this repository
and is excluded from the export. Besides the engine, the export includes these
Rust tools:

- **`tools/export`**, the exporter. Rust, with no outside crates; it reuses
  the model crate's JSON reader and computes SHA-256 itself. Before the older
  Python exporter was removed, it reproduced that exporter's 128-file
  snapshot and manifest byte for byte.
- **`crates/gpu/build-kernels`** and **`.cargo/config.toml`**: the kernel
  build tool, and the alias that runs it as `cargo kernels`. Rust; it drives
  the pinned nightly compiler, then copies and stamps the PTX. There are no C
  sources and no shell scripts.
- **`crates/tokenizer/tools/generate_unicode.rs`** rebuilds the Unicode
  tables, reproducing the checked-in output exactly.

</details>

**Where next:** [Provenance](provenance.md) · [Link checks](link-checks.md) ·
[Documentation index](README.md)
