# FFN plan regression data

The plan files contain the full `Debug` bytes of synthetic Qwen Flash Next
config plans. They were saved with the parser from base commit
`32d6374fd90c59a80aae81dae4ee2da43530f2be`, before the FFN plan change.
The test inputs come from `flash_fixture` and `flash_formats` in
[`plan/mod.rs`](../../../src/plan/mod.rs).

The inputs cover FP8 and mixed NVFP4/MXFP8 storage. They include decoder and
MTP layers. The test compares current plan bytes with these saved bytes.
It also adds invalid deferred schedule fields to each input. Those fields
must not reach the deferred FFN parser on the Qwen path.

These are config plans. No weight data or device results are stored here.
