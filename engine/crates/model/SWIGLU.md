# SwiGLU bound

`swiglu::SwiGlu` holds an optional upper bound. Its default has no bound.
`TextDescription::swiglu` reads the bound from the deferred operation data.
It does not enable GLM execution.

The saved GLM config sets `text_config.dtype` to `bfloat16` and
`text_config.swiglu_limit` to `10.0`. See the
[config fixture](tests/fixtures/glm/config.json), lines 2985 and 3230.
The saved Transformers `modeling_glm5_next.py` defines the dense forward
pass at lines 97-103 and the expert activation at lines 136-141. It clamps
the gate only above the bound. It clamps the up value at both bounds.
Both clamps precede SiLU and multiplication. The source has no cast to
FP32 in these operations. The source identity has the limits stated in the
[fixture record](tests/fixtures/glm/README.md).

For BF16 tensors, the operation contract is:

1. Round each projection output to BF16, nearest even.
2. Convert the scalar bound to BF16, nearest even.
3. Clamp the gate above the bound. Clamp up between the negative bound and
   the bound. The clamps retain the tensor dtype.
4. Compute SiLU with FP32 working values. Round its result to BF16.
5. Multiply that BF16 value by the BF16 up value. Round the product to BF16.

The host reference returns a widened FP32 value with the BF16 result bits.
Its FP32 mode omits the BF16 conversions. The reference uses the existing
fixed-order exponential from the NVFP4 host oracle. It does not claim
bit identity with the PyTorch exponential on all inputs.

The existing `layer_swiglu` kernel accepts a trailing bound parameter.
Zero means no bound. Existing host calls pass zero and keep their current
exponential and rounding order. A capped call uses the fixed-order
exponential from `gpu::next_math`. This is also the arithmetic used by the
capped host reference. The bound is operation data, not a model-name test
or an environment option. It must stay positive and finite in FP32 and
BF16. The host rejects other bounds before launch.

This change covers the standalone epilogue. It does not bind the deferred
GLM model to the existing fused dense or routed-expert kernels. Those
bindings still need their own dtype and byte checks.

`swiglu-bytes` is a device check binary. It checks capped output against
the host oracle. It checks uncapped output against a supplied base PTX
module. It also checks repeat and chunked runs. The base PTX must come
from the parent revision recorded with the kit. The kit command uses
shipped binaries and PTX. It needs no Rust toolchain on the Spark.
Host tests do not establish device byte identity or token ID exactness.
