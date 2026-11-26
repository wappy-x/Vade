# Contributing

Thank you for the interest in Vade. Small, complete changes land fastest.

## Ground rules

1. Rust standard library only. Zero third-party crates.
2. Unsafe code only at the GPU boundary, with a safety note and a test.
3. Exactness is the product: the same request with the same settings must return the same token IDs on 1, 2 and 4 Sparks, batched or alone, drafting on or off. A change that alters token IDs needs a strong reason in the pull request.
4. Keep diffs small. One pull request, one idea.
5. No model-specific tuning and no new build steps beyond the pinned ones.

## Before you open a pull request

1. `cargo test` passes.
2. `cargo clippy` is clean. CI denies warnings.
3. New code has a test that fails without it.
4. The repository ratchet still passes.

## How to send it

Open a pull request against `main`. Name the commit after what it does. Include the exact command output for every number the pull request claims.

## What merges fastest

Bug fixes with a failing test, exactness proof work, and documentation fixes. For a large feature, open an issue first and wait for a maintainer reply before writing the code.
