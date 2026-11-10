# harness/src/terminal

**Rust modules for `terminal`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`clock.rs`](clock.rs) | Local display timestamps. |
| [`clipboard.rs`](clipboard.rs) | Exact answer copying and OSC 52 encoding. |
| [`input.rs`](input.rs) | Input channels and incremental key decoding, with decoder tests. |
| [`frame.rs`](frame.rs) | Frame construction, dock layout and screen output. |
| [`text.rs`](text.rs) | Display sanitization, approval escaping and character cell widths, with focused tests. |
| [`editor.rs`](editor.rs) | UTF-8 scalar-boundary editor. |
| [`render.rs`](render.rs) | First-party terminal presentation. |
| [`transcript.rs`](transcript.rs) | The bounded, width-aware transcript rows behind the terminal view. |

[`tests.rs`](tests.rs) keeps complete-screen goldens and approval-review integration checks. Text, decoder, transcript and rendering unit tests live with their implementations. Public text helpers remain available through `terminal`.

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `terminal`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
