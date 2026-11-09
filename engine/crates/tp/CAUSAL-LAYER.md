# Causal KDA layer probe

The checkpoint layer probe executes a complete KDA layer with a dense
feed-forward block. It derives all dimensions and stored parameter types
from the admitted checkpoint and its layer recipe. The layer constructor
rejects other layer kinds. The probe reads prefix parameters before it calls
that constructor.

Run the probe on a Linux host with CUDA:

```sh
checkpoint-prefix-check layer-step suite MODEL INPUT_IDS NEW_OUT
checkpoint-prefix-check layer-step compare-suite LEFT RIGHT
```

The suite runs the host reference first. It then starts fresh device
processes. Each process compares every produced plane with the host bytes.
The cross-device command compares both saved result trees.
The suite source defines the repeat count and input bound.

The layer includes attention mHC, input norm, projections, causal convolution,
KDA prepare and recurrence, gated norm, attention output, both residual
operations, post norm, dense projections, and capped SwiGLU.

MXFP8 projections use dynamic activation groups from BF16 source values.
They retain the stored weight and scale bytes. The fixed FP32 projection
returns a BF16 row. The bounded adapter stages the activation quantizer
through the host. It makes no speed claim.

Plain projections use complete FP32 rows widened from stored parameters.
They fold input features in ascending order. They round their completed
ordinary output to BF16. The shared KDA operation uses a fixed feature tree
and the existing portable exponential. Sigmoid beta rounds before recurrence.
The recurrent state stays FP32. Convolved values, recurrent output, gated
output, SwiGLU, and residual boundaries use BF16 values in FP32 lanes.
History retains the original widened BF16 projection values.

The host and device read immutable old causal state. They return new state
only after the full layer succeeds. The device completes every producer
before its consumer or receipt callback. A callback failure drops unpublished
state. It cannot undo files that the callback already wrote.

Each receipt pins the input IDs, source revision, metadata, consumed
parameters, scope, plane names, and plane bytes. It records the MXFP8 storage
types. Plain parameter records contain decoded values without their stored
type names. The probe rejects nonfinite output values.

This reference describes the declared candidate recipe. The older
`logical-f32-v1` stack omits these casts and activation quantizers. Its bytes
are not this reference's expected bytes. Backend parity remains open.

The probe runs complete rows on each device. It does not prove distributed
transport or tensor splits. It does not admit a service Factory. Other
attention kinds, expert layers, the full stack, and generated IDs are outside
this layer probe. See [the bounded decoder](BOUNDED-DECODER.md) for the full
stack probe. Long context and drafting remain open.
