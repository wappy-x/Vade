# engine-fabric

**Shared communication interfaces and peer primitives for the engine's
multi-Spark code.** Model ownership stays in `engine-model`; sequence and
tensor transports stay in their calling crates.

<dl>
<dt><code>discovery</code>, <code>verbs</code></dt>
<dd>Runtime peer discovery and the Linux-only rdma-core interface.</dd>
<dt><code>layout</code></dt>
<dd>Link capabilities derived from rank count and ring or path topology.</dd>
<dt><code>exchange</code>, <code>profile</code></dt>
<dd>The Flash collective interface and bounded communication profiling.</dd>
</dl>

**How it fits:** [spark-seqpar](../seqpar/README.md) and
[spark-tp](../tp/README.md) share these primitives.
`spark-seqpar` retains compatibility re-exports of `discovery` and Linux
`verbs`; callers can also import them directly from `engine_fabric`.

**Back:** [Crate map](../README.md) · **Docs:** [Architecture](../../docs/architecture.md)
