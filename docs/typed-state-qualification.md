# Typed hybrid state qualification

`Session::target_state_bytes()` records the committed position, all owned FP32
recurrent cells, packed BF16 convolution history, and visible attention K/V.
`Session::committed_state_bytes()` appends the prompt replay tokens, in-step MTP
cursor/last hidden row/visible K/V, retained recurrent/history checkpoints, and
the retained MTP replay row. Snapshot presence is explicit. Geometry and byte
extents come from the same immutable local model configuration.

These receipts synchronize/read device storage and allocate temporary copies.
They are qualification tools, never request-path work. Retain large receipts on
the native test machine. External draft models, empty MTP sessions, speculative
MTP cursors, and incomplete replay checkpoints are refused by the complete
receipt. Pending or invalid target owners are refused by the target receipt.

Compare target bytes across plain/MTP execution. Compare complete bytes only
with matching draft/replay ownership and identical arithmetic segmentation.
Full-vocabulary logits remain a separate receipt. A scalar GDN reference is in
`engine/tools/bench/src/gdn_reference.rs`; its host transcendental functions do
not establish CUDA approximation parity.

The world-2 production refusal remains in place. Host reference and state-copy
tests do not establish physical registration, payload residency, capacity,
native full-model exactness, or HTTP readiness.

The private typed loader core is also callable by internal module tests with a
validated rank domain. Both production entry points enforce world admission
before reaching it. Native qualification can therefore exercise the same loader
without enabling a public world-2 serving path.
