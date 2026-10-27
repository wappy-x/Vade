## What this changes

Say what was wrong or missing, and what Vade does now. Link the issue if there
is one.

## Tests you ran

List the commands and what they printed. For engine changes, also say which
Sparks you ran on and attach or link the receipts.

```sh
# for example, from engine/
cargo +1.99.0 test --offline --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- check-doc-links
```

## Checklist

- [ ] **Tests:** the checks under "Check your change" in CONTRIBUTING.md that
  apply to this change pass, and I listed them above.
- [ ] **Docs:** pages, help text and the Unreleased section of CHANGELOG.md
  are updated for anything a user would notice, or nothing user-visible
  changed.
- [ ] **Standard library only:** no new third-party dependencies of any kind
  (registry, Git, path or vendored).
- [ ] **No new unsafe:** no new `unsafe` blocks, functions, impls or `extern`
  declarations, and existing SAFETY comments still hold.
- [ ] **Exactness kept:** the same request on the same setup still returns the
  same token IDs, alone, batched, cached, chunked, and with drafting on or off.
  For engine changes I attached the receipts that show it. For host-only or
  docs changes, tick it and say "host-only" under Tests.

The Rust is formatted densely on purpose. Please match the code around your
change by hand and don't include a `cargo fmt` sweep.
