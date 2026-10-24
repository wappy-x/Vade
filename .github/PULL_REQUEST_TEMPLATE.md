## Change

Describe the problem and the resulting behavior.

## Validation

List the commands run and their results. Attach receipt paths and relevant
output, including an exactness receipt for engine changes.

## Checklist

- [ ] Rust changes are std-only.
- [ ] No third-party dependencies were added (including registry, Git or vendored dependencies).
- [ ] No new `unsafe` code was added.
- [ ] `cargo fmt --check` results are recorded when Rust changes; pre-existing failures are identified and no bulk formatting sweep is included.
- [ ] Relevant tests and CI checks are green.
- [ ] Engine changes include an exactness receipt identifying the build, model, settings and token comparison; otherwise this is marked not applicable with a reason.

Existing Rust uses deliberate dense formatting, so a formatting check may report
pre-existing differences. Keep formatting changes confined to the code being
changed.
