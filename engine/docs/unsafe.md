# Unsafe code

**Where the engine uses `unsafe`, why it can't avoid it there, and what stops it
from spreading.** This page is for people reading or reviewing engine code.
The rule itself is in [AGENTS.md](../../AGENTS.md#house-rules): no new unsafe.

Vade is written in Rust using only the standard library, so nothing wraps the
NVIDIA driver, the RDMA library or the GPU target for us. `unsafe` exists in
the engine only at four kinds of boundary:

1. **GPU kernels.** Kernels are Rust compiled for NVPTX. They receive raw
   device addresses, use inline PTX for instructions Rust has no word for
   (tensor-core multiplies, asynchronous copies, some conversions) and call
   LLVM intrinsics.
2. **The CUDA driver.** The engine talks to `libcuda` through a small,
   hand-declared C interface: memory, modules, streams, graphs and launches.
3. **Kernel launches.** A launch passes an array of untyped argument
   pointers. The compiler can't check that they match the kernel's
   parameters, or that the memory they point to outlives the GPU's work.
4. **RDMA transport between Sparks.** `libibverbs` is loaded at run time
   and called through C function pointers. Peers write into memory that is
   registered with the network card and read back through raw pointers.

A few ordinary system calls sit next to these. They're declared by hand
because the standard library doesn't expose them: detaching a process and
handling signals (`fork`, `setsid`, `dup2`, `flock`, `signal`, `_exit`),
reading checkpoint files (`mmap`, `posix_fadvise`, `fcntl`), and setting CPU
affinity and socket options.

## Where it lives

**The table counts every `unsafe` keyword outside comments and string
literals, by what follows it, in tracked files under `engine/` on main at
`56b0f4ca0`.** Tests and benchmark tools are included. A block is
`unsafe { … }`. A function is `unsafe fn`, including every kernel entry
point. Foreign functions are the declarations inside an `unsafe extern`
block. Function-pointer types are `unsafe extern "C" fn(…)` fields that hold
loaded `libibverbs` symbols.

| Package | Blocks | Unsafe fns | Other | What the unsafe is for |
| --- | ---: | ---: | --- | --- |
| [`spark-rust-kernels`](../crates/gpu/kernels/README.md) | 12 | 1,054 | 13 intrinsic blocks, 16 intrinsics | Kernel bodies. Of the 1,054 functions, 470 are `extern "ptx-kernel"` entry points and the rest are device helpers that take raw pointers. |
| [`spark-gpu-probe`](../crates/gpu/README.md) | 260 | 57 | 3 extern blocks, 5 foreign fns | The driver interface in `driver.rs`. The `launch_flash*` family and the typed launch wrappers in `backend.rs`, `batch.rs`, `fp8.rs`, `fp4.rs` and `moe.rs`. Graph capture and replay, and pinned host memory. |
| [`spark-engine`](../crates/engine/README.md) | 213 | 9 | 1 impl, 4 extern blocks, 7 foreign fns | Mostly kernel launches from model code (`src/flash/`, `src/hybrid*`), each calling a GPU-crate unsafe function. Also `mmap`, `posix_fadvise` and `fcntl` for checkpoint reads, and one direct driver copy in the `load-check` tool. |
| [`spark-tp`](../crates/tp/README.md) | 95 | 13 | 3 impls, 8 extern blocks, 12 foreign fns, 10 fn-pointer types | RDMA queue pairs and one-sided writes into pinned exchange slots (`rdma.rs`, `transport.rs`). The servers also use signal handlers, CPU affinity, socket options and checkpoint file advice. |
| [`spark-seqpar`](../crates/seqpar/README.md) | 88 | 19 | 5 impls, 8 fn-pointer types | Ring transport over TCP or RDMA (`transport.rs`, `transport/gather.rs`, `rdma.rs`): registered K/V and mailbox memory, viewed as slices while peers fill it. |
| [`vade-bench`](../tools/bench/README.md) | 59 | 5 | 2 extern blocks, 3 foreign fns, 5 fn-pointer types, 3 attributes | The `kernel-bench` binary, which drives the driver and kernels directly, and `memory_probe.rs`, a separate allocation probe built with `rustc`. The HTTP benchmark library has none. |
| [`engine-fabric`](../crates/fabric/README.md) | 6 | 0 | 1 extern block, 4 foreign fns, 14 fn-pointer types | `verbs.rs`: the `rdma-core` C layout and the lazy `dlopen` of `libibverbs.so.1`. Linux only. |
| [`spark-serve`](../crates/serve/README.md) | 5 | 0 | 1 extern block, 5 foreign fns | Only `src/bin/spark-serve/lifecycle.rs`, which detaches the server. The crate denies unsafe code everywhere else. |
| [`spark-tp-kernels`](../crates/tp/kernels/README.md) | 1 | 3 | 1 attribute | The optional TP projection kernels. |
| `tools/oracles` | 3 | 0 | 1 extern block, 2 foreign fns | One microbenchmark helper that starts and stops the CUDA profiler. |

The rest of the engine has none, and five packages make that a compile error:

| Package | Lint |
| --- | --- |
| [`engine-base`](../crates/base/README.md), [`engine-hash`](../crates/hash/README.md), [`engine-model`](../crates/model/README.md), [`engine-tokenizer`](../crates/tokenizer/README.md), [`vade-export`](../tools/export/README.md) | `unsafe_code = "forbid"` in `Cargo.toml`. No module can opt back in. |
| [`spark-serve`](../crates/serve/README.md) | `unsafe_code = "deny"`. Only `src/bin/spark-serve/lifecycle.rs` opts back in, with a visible `#![allow(unsafe_code)]`. |
| [`vade-bench`](../tools/bench/README.md) | `unsafe_code = "deny"`. In `kernel-bench`, the `cuda` module (`kernel_cuda.rs` and the `kernel_*.rs` files it includes) and one shared parameter module opt back in with `#[allow(unsafe_code)]`. `memory_probe.rs` is built separately with `rustc`, so the package lint doesn't cover it. |
| `spark-gpu-probe` | `module_loading.rs` alone carries `#![forbid(unsafe_code)]`; it keeps the module-load control flow safe and leaves the CUDA calls to the driver adapter. |

`spark-build-kernels` has no unsafe code and no lint. The CLI and the harness
sit outside the engine workspace. They have their own small C interfaces, for
signals, terminals and process control, and this page doesn't cover them.

## What checks it

**There are three checks, and only the first is automatic everywhere.**

1. **The compiler.** In the five `forbid` packages, any `unsafe` fails the
   build. In the two `deny` packages it fails outside the modules that
   allow it by name. CI builds with `RUSTFLAGS=-D warnings`.
2. **The ratchet.** [`cli/tests/ratchet.rs`](../../cli/tests/ratchet.rs) holds
   counts that may fall but never rise:
   third-party dependencies, files over 1,500 lines and hardcoded Spark-count
   branches. **It doesn't count unsafe sites today.** Nothing automatic stops
   a new unsafe block in the packages in the first table.
3. **Review.** The no-new-unsafe rule is checked by reading the diff. Before
   you ask for review, run this from the repository root:

   ```sh
   git diff main -- '*.rs' | grep -E '^\+.*\bunsafe\b'
   ```

   Every hit needs a reason, and an added unsafe block, function, impl or
   extern declaration is a refusal, not a discussion. Code moving between
   files counts as removed in one place and added in another; say so in the
   change description. Comments and strings that happen to contain the word
   also match, so read each hit.

## Writing a SAFETY note

**Put a `// SAFETY:` comment directly above each unsafe block, naming the
invariant the caller upholds.** It should answer the questions the compiler
couldn't. Say who owns the memory, how many bytes or elements are valid, and
how long the memory stays alive (for GPU and RDMA work, until the stream or
the completion queue has finished with it). Say which thread may touch it.
Say what layout the other side expects. Two notes already in the tree show
the shape:

```rust
// SAFETY: GdnChunkNext is repr(C) u64/u32/f32 fields without padding (asserted above).
```

```rust
// SAFETY: the immutable tensor owns this checked contiguous
// range; prior packing is synchronized and out has count bytes.
```

For an unsafe function, say what its caller must guarantee: a comment on
the definition, or a `# Safety` section in its doc comment. `launch_flash`
states its contract this way in
[`backend.rs`](../crates/gpu/src/backend.rs). The caller must match the
kernel's parameter list and keep every referenced allocation alive until the
ordered stream has completed the kernel. Each call site's note should then
name the parameter block it passes and the buffers that keep its addresses
alive.

Coverage is far from complete. On main at `56b0f4ca0`, nine comments
explain an unsafe block: five `// SAFETY:` notes (the two above and three in
[`tp/src/next_bw.rs`](../crates/tp/src/next_bw.rs)) and four older
`// Safety:` comments (two in
[`engine/src/flash/session.rs`](../crates/engine/src/flash/session.rs) and
two in [`seqpar/src/transport.rs`](../crates/seqpar/src/transport.rs)).
Adding a note changes no code. If you touch an unsafe block, add or correct
its note in the same change.

## Related pages

- [Module map](module-map.md): where each engine module lives, including the
  boundaries above.
- [Architecture](architecture.md#gpu-kernels-and-the-driver): kernels, the
  driver and `libibverbs` from the outside.
- [Validation](validation.md): what host checks prove and what needs a Spark.

**Back:** [Documentation index](README.md) · [Engine packages](../crates/README.md)
