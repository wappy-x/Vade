# harness/src/tools

**Rust modules for `tools`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`discovery.rs`](discovery.rs) | Literal, bounded discovery over the same directory handles as reads/edits. |

## Read code with a bounded working set

Native coding tools keep source reads inline at 4 KiB including provenance.
Start with `workspace_search` scoped to the likely file or component. Give one
`literal` or up to eight `literals` to find related names in one scan:

```json
{"path":"harness/src","literals":["read_line_ranges","workspace_read_lines"],"limit":8}
```

Search counts each matching line once, in deterministic path/line order. Its
snippets center on the first match; `byte_column` gives the one-based UTF-8 byte
column at which the snippet starts. `text_truncated`, `scan_truncated` and
`skipped` disclose omitted text or incomplete discovery. Narrow the path when
needed, then read only the relevant code. `offset`/`next_offset` paginate a live
view; changes between calls can shift matches.

`workspace_read_lines` accepts a single inclusive `start_line`/`end_line` or up
to eight ordered, disjoint `ranges` with at most 200 selected lines in total:

```json
{"path":"harness/src/tools/discovery.rs","ranges":[{"start_line":55,"end_line":90},{"start_line":145,"end_line":175}]}
```

All ranges come from one guarded UTF-8 file snapshot and share the inline byte
budget. Results carry the relative `path`, requested ranges and exact numbered
lines. `next_range` is the zero-based requested-range index and `next_line` is
the first selected line that did not fit. Both are null when the requested
ranges are complete, even if the source has later unrelated lines. Source size
is capped at 4 MiB. Large individual lines require small `workspace_read` byte
ranges; offsets must be UTF-8 boundaries. Avoid reading every page of a source
file or artifact without a specific need. Retained artifact bytes are useful
for a particular omitted diagnostic; original captures remain in the journal.

`workspace_edit_lines` edits a fully displayed range from the latest
`workspace_read_lines` call for that path. Vade keeps the latest read of the
eight most recently read paths, and two reads of unchanged bytes count as one. Copy its `edit_tag`, `path` and inclusive
`start_line`/`end_line`; supply `operation` (`replace`, `delete`, `insert_after`)
and exact `new_text`. Delete uses empty text; insert uses one anchor line.
Empty files use 1..1. Include CRLF explicitly; text before later lines must end
with a newline. Inserting after an unterminated last line adds its separator.
The tag covers the whole file, and retained exact bytes also guard against hash
collisions. Hidden/truncated lines cannot be edited. Re-read after each edit or
resume; stale files refuse with current lines. Normal approval/checks apply.

## Check and repair after edits

On macOS, `workspace_shell` accepts exactly one of `argv` or `check_profile`,
plus `timeout_ms` (1–120000). Profiles select the existing offline host checks
for `engine-host`, `engine-bench-host`, `harness-host` or `cli-host`; the engine
host check excludes GPU kernels. Profiles require an explicitly selected Rust
toolchain via `--shell-toolchain` and retain normal command approval and sandbox
rules. After a focused edit, check its component, read failure diagnostics,
repair, then repeat the check before claiming success. Tests use explicit argv
when relevant. Bounded command captures retain the exit status and disclose
omitted bytes; a timeout or unknown outcome must be reported honestly. Host
checks do not qualify GPU correctness or performance.

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `tools`.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
