# Chat template references

[glm.jinja](glm.jinja) is the unchanged template from
[checkpoint revision 76348ddb2be2e35e39ecfc00e75dc07703f4f3d5](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/blob/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5/chat_template.jinja).
It has the same bytes as the
[upstream template](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/91b69215804f2d2036a8867f6ce44848e278dcda/chat_template.jinja).
The [NOTICE](../../../NOTICE) records its source and MIT license.
The [serving bridge](../src/tokenizer.rs) checks the full template text.
It accepts trailing line-ending changes. It rejects other GLM template edits.

The [Rust renderer](../src/tokenizer/glm.rs) implements the text-only rules.
It does not run Jinja. It rejects forced tool choices.
The GLM renderer omits `developer` messages, as the pinned template does.
Other renderers keep their current role rules.
Tool definition keys keep the template's raw bytes.
Nested tool fields use JSON escapes. String call arguments stay raw.
Tool references, deferred tool loading, and media content are outside its scope.
The loader retains the complete generation EOS list.
Request preparation rejects a backend whose stop set differs from that list.
The [host proof table](../tests/fixtures/glm-text/README.md) lists the source
vectors and the open external comparison. GLM engine admission stays closed.

Back: [server crate](../README.md).
