# Host CSF block decoder

[`csf::Csf32`](src/csf.rs) decodes `csf_32x2_v1` on the CPU. The caller
supplies the logical scale shape, quantization type, and component bytes.
The decoder does not use a model name. It does not read a checkpoint or
change model admission. It has no GPU path.

`expand` returns exact row-major scale bytes. `decode_into` writes that same
layout into a caller buffer. For NVFP4, this is the E4M3 scale layout used by
[`flash::nvfp4::dequantize`](src/flash/nvfp4.rs).
`dequantize_nvfp4` uses that existing host function to produce FP32 weights.
It keeps the existing order of scale multiplication.

## Public definition

The implementation uses local copies of these public files. No remote source
or model payload was needed.

- FlashInfer, revision `5a894aff27b53b627567ca72ee79454df66879ba`:
  [`flashinfer/experimental/b12x/loader/_csf_blocks.py`](https://github.com/local-inference-lab/flashinfer/blob/5a894aff27b53b627567ca72ee79454df66879ba/flashinfer/experimental/b12x/loader/_csf_blocks.py).
  `CsfBlockPlane.validate` defines the component bounds.
  `export_csf_blocks` and `_from_logical_components` define packing.
  `block_order` and `logical_csf_components` define the logical mapping.
  File SHA256:
  `ed29be272ed682238a66f6e33f8d97d72f17652c9b743c3d8d23615764f05496`.
- vLLM, revision `8244d1c44348ae8a5bbfc9f1b607021652283ed4`:
  [`vllm/model_executor/model_loader/csf_utils.py`](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/vllm/model_executor/model_loader/csf_utils.py).
  `CsfTensorReader.matrix` binds the `csf32_*` components to `CsfBlockPlane`.
- The same vLLM revision:
  [`tests/quantization/csf_fixtures.py`](https://github.com/local-inference-lab/vllm/blob/8244d1c44348ae8a5bbfc9f1b607021652283ed4/tests/quantization/csf_fixtures.py).
  `matrix` defines the synthetic known vectors.
  `decode_planes` gives an independent logical decoder.
  File SHA256:
  `7861967ce921cd9e26edcc5b6f147072f4e57fa34850c4490f6d1b1b2163625a`.

These source projects use Apache-2.0. The Rust decoder is written for Vade.
No Python code, dependency, or model data is included.

## Component contract

The following rules come from `_csf_blocks.py` above. Rows must be positive
and divisible by 32. Scale columns must be positive. NVFP4 uses 4-bit codes.
MXFP4 uses 1-bit codes. These widths come from quantization metadata.

Let `R` be the row count, `C` the scale column count, and `b` the code width.
Let `B = R / 32 * ceil(C / 2)` be the block count.
Let `G = ceil(R / 128) * ceil(C / 4)` be the tile group count.
Let `P = ceil(G / 32)` be the page count.

| Component | Type on disk | Required extent |
| --- | --- | --- |
| `csf32_bases` | U8 | `R` elements |
| `csf32_codes` | U8 | `R * C * b / 8` elements |
| `csf32_page_bases` | Little-endian U32 | `P` elements |
| `csf32_offsets` | Little-endian U16 | `B + P` elements |
| `csf32_exceptions` | Little-endian U16 | Exact extent given by all page prefixes |

The Rust API takes raw bytes for each component. It checks the full byte
length, including the element width. Each call handles a separate plane.
Its page bases are relative to that plane's exception bytes.

Visit tile groups in row-major order. Each group covers up to 128 rows and
4 scale columns. Within a group, visit column pairs first. Within each pair,
visit the 32-row blocks. Tail groups are clipped to the logical shape.
Each block stores row-major fields with the low field first. A final single
column stores only its own fields. It has no padded column in the stream.

Add each field to its row base. The row base must not exceed `256 - 2^b`.
Then apply exceptions. The low byte of an exception is `2 * row + column`
within its block. The high byte replaces the scale byte exactly. Positions
must be in bounds, unique, and increasing within each block.

A page holds 32 whole tile groups. Each page has a separate zero prefix and
a terminal prefix. Prefixes count exception records. Page bases count bytes.
Pages must cover a consecutive exception array with no gaps or unused tail.
The array must be smaller than `2^32` bytes, as required by the source encoder.

## Validation and limits

Shape products and component sizes use checked arithmetic. The decoder checks
every component before it writes output. It rejects short and extra data,
invalid prefixes, invalid page bases, and invalid exception positions.
Allocation failure from `expand` returns an error.

Byte expansion preserves every scale bit pattern. NVFP4 conversion to FP32
also checks the existing finite, nonnegative scale requirement. The global
weight scale must be positive and finite. MXFP4 supports byte expansion only.

[`tests/csf32.rs`](tests/csf32.rs) uses hand-built data and a test encoder.
Known vectors are derived by hand from the public `matrix` fixture. They are
not captured Python outputs. Round-trip tests use the public packing rules.
Malformed-input tests cover lengths, geometry, palettes, prefixes, and records.

This API does not validate real checkpoint headers or enable GLM execution.
Model admission, storage plans, and device loading need separate work.
