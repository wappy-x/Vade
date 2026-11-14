# engine-hash

SHA-256 for checkpoint verification. The public API hashes every input byte;
unchanged-file cache decisions remain in `engine-model`.

The implementation reuses the safe, portable SHA-256 code from the export tool.
It uses only Rust's standard library, has no dependencies, and forbids unsafe
code. Every architecture uses the same compression, stream framing and padding;
there is no hardware-specific SHA2 implementation.

Hashing streams through a 64 KiB buffer rather than allocating the input size.
Tests cover known SHA-256 vectors, padding and read boundaries, generated streams,
and file-reader error handling. Large-stream release checks are separate from
ordinary host test runs; performance measurements must identify their CPU and
build mode. Mac measurements do not establish Spark performance.

File verification on supported Linux AArch64 and x86-64 systems reads the same
open inode through an independent direct-I/O descriptor and aligned 1 MiB heap
storage. Large checkpoint files require a filesystem that accepts these direct
reads and access to `/proc/self/fd`. Buffered reads are limited to 64 KiB total
per file, allowing small metadata and a short final tail. If direct I/O is
unavailable or rejected while a larger remainder is unread, verification fails
instead of reading that remainder into the page cache. Interrupted reads retry;
other I/O errors propagate. The caller's offset, original flags, expected digest
and before/after file-identity checks remain unchanged.

Non-Linux hosts, including macOS development and test hosts, use an explicitly
reported portable buffered mode. That mode has bounded userspace memory but
does not promise page-cache avoidance; the Linux serving guarantee does not
apply to it. Stream callers likewise choose their own reader and I/O policy.

Each `checkpoint_hash_file` diagnostic reports `io_mode`, `direct_bytes`,
`buffered_bytes` and `bytes`. Modes are `direct`, `direct-with-buffered-tail`,
`buffered-bounded`, `buffered-portable` and `empty`; an open direct descriptor
alone is not evidence that a direct read succeeded. Failures report
`state=failed` and `io_policy=direct-or-bounded-buffer` on Linux, or
`io_policy=portable-buffered` on other hosts. On a supported Linux
filesystem, large files are never silently hashed using an unbounded buffered
fallback. The Spark gate must still measure file-cache growth on the actual
checkpoint filesystem, together with these diagnostics; userspace scratch
bounds do not establish filesystem-cache bounds.

`hash_buffer_bytes_per_worker_max` replaces the earlier
`hash_buffer_bytes_per_worker` diagnostic key. Its value is 1,118,207 bytes rather
than 65,536 because it includes the 64 KiB stream buffer, 1 MiB direct-I/O buffer
and alignment slack. This is the maximum I/O-buffer allocation per file
worker, not total resident memory or file-cache usage. The maximum worker
count remains eight. HTTP checkpoint metadata fields are unchanged.
