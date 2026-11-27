# Typed decode graphs

The official Qwen3.6-35B-A3B-FP8 checkpoint is served over world-1 HTTP in
eager execution: `spark-serve` sets its graph storage budget to 0. The HTTP
bytes gate remains pending, so context and throughput are unadvertised.
Multi-rank serving (including world 2), multimodal inputs, sequence-parallel
peers and fast drafting remain refused. The captures below apply to native
typed decode with a funded graph allowance.

Typed world-one decode captures the existing complete layer loop and optional
device argmax with the driver's whole-model graph allowance. Tokens and cache
descriptors upload before replay; kernels, grids and argument bytes are unchanged.
Attention positions remain in the descriptors, without graph-specific kernels.

Each executor retains four shapes keyed by row count, output mode, verification
layout, descriptor addresses and weak session identities. Score-bucket growth,
KV replacement, session replacement and prompt work invalidate affected captures.
Graph metadata shares the device allocation ceiling; unsuccessful healthy capture
uses eager execution. CUDA execution and cleanup faults remain fatal.

MTP predictions and target verification have separate graphs at their host decision
boundaries. MTP observation remains eager: its KV append carries a scalar position.
Profilers and the host routing-rejection hook use eager execution. Output and router
readback waits remain outside capture. Native answer bytes and timing require GPU qualification.
