# Build and exactness gates

Use `vade test build` and `vade test determinism`. A gate blocks a change until
its required checks pass. The [Rust gate guide](../../../cli/docs/test.md)
describes inputs, cases, receipts and exit codes, including the established
native and HTTP byte comparisons.

The former engine shell gates and operational wrappers have been retired.
Host checks verify tool behavior; GPU correctness and performance require
qualification runs on real Sparks, as described in [CONTRIBUTING.md](../../../.github/CONTRIBUTING.md#exactness-and-evidence).

See [Validation](../../docs/validation.md) and the
[engine exactness contract](../../README.md#determinism).

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
