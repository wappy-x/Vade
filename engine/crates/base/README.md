# engine-base

Portable metadata utilities shared by the engine. The bounded JSON reader
preserves numbers as decimal strings so dimensions and offsets do not round
through floating point; admission helpers read the MemAvailable memory input and format byte-budget and
token-limit refusals without making admission decisions. The crate also supplies a common
error and result type. It uses only the Rust standard library, has no
dependencies, and forbids unsafe code. It is a member of the engine workspace
and is covered by the [workspace host checks](../README.md).

**Source:** [JSON](src/json.rs) · [Admission diagnostics](src/admission.rs) ·
[Crate map](../README.md)
