# Published GLM prompt vectors

`vllm-test-cases.json` is an unchanged vLLM fixture. It contains request data
and expected prompt bytes. It contains no weights or checkpoint token IDs.

Source: [vLLM fixture at its introduction commit](https://github.com/vllm-project/vllm/blob/d14f72c7fe1316ae9d186f1eed554e8099dba705/rust/src/chat/src/renderer/glm5_next/fixtures/test_cases.json).
The SHA256 is `8660be3b335be039a57af9e180403be0baee8cd1455f151d7297a20285f39497`.
The [test](../../../src/tokenizer/glm_external_tests.rs) verifies this hash.
The fixture uses the Apache License 2.0. Copyright belongs to the vLLM contributors.
See [the notice](../../../../../NOTICE) and [license](../../../../../../LICENSE).

The [upstream tests](https://github.com/vllm-project/vllm/blob/d14f72c7fe1316ae9d186f1eed554e8099dba705/rust/src/chat/src/renderer/glm5_next/tests.rs)
check these prompt bytes against their native renderer. They also compare
typed requests with their existing HF renderer. The
[merged change](https://github.com/vllm-project/vllm/pull/62424) records those checks.
This repository does not execute that runtime or repeat its reported runs.

The local test calls the typed GLM renderer. It compares the fixed expected
strings directly. It does not derive expected strings from Vade's renderer.
It preserves the source order of tool fields and arguments.
It maps source truth values to the typed Boolean option.
It maps missing or null content to the template's empty string.
These mappings test renderer behavior. They do not widen HTTP input admission.

The test prints each excluded case and its reason. Custom roles, media,
structured tool output, and assistant continuation have no matching typed API.
The fixture also includes invalid arguments and tool definitions.
The test checks their refusals. Its fixed counts reject silent coverage loss.

This is an external prompt comparison, not complete checkpoint text parity.
The upstream Flash template changes the tool-result sort implementation.
Its SHA256 is `2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4`.
The local checkpoint template stays pinned to its existing bytes.
The compared text cases match Vade's renderer for its pinned template.
This does not prove equality between the template runtimes.
No file here records an upstream tokenizer run on the pinned checkpoint.
Exact checkpoint IDs, decoding, and runtime provenance remain separate gates.

Back: [server tests](../../README.md).
