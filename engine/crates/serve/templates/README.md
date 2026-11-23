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

[inkling2-mini.jinja](inkling2-mini.jinja) is the unchanged template from
[checkpoint revision b85c3cfa615fc413cc9af3379f491b963f3b586a](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/chat_template.jinja).
Its SHA-256 is `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`.
[inkling2-mini-response.json](inkling2-mini-response.json) is the
`response_template` field of the same revision's `tokenizer_config.json`,
written with one-space indents. The [NOTICE](../../../NOTICE) records the source
and the Apache-2.0 license of both files.
The [serving bridge](../src/tokenizer.rs) selects the renderer by the bytes of
the template. It reads no model name. It accepts trailing line-ending changes.
It refuses a template with the same end-of-turn control and other bytes, and a
`response_template` that differs from the copy.

The [Rust renderer](../src/tokenizer/inkling.rs) implements the text rules. It
does not run Jinja. These are its own rules:

- The template has no thinking switch. An explicit `enable_thinking` false
  selects effort 0, the lowest level of the template, and wins over an effort
  value. A service default of thinking off and the `/no_think` words in message
  text are no client ask: the prompt keeps the default effort of the template.
- A `developer` message renders as a system message. The template refuses it.
- Tool call arguments given as JSON text are read as an object. The template
  takes an object only.
- A control spelling in caller text is encoded as ordinary text, alone. The
  reference library makes the same cut with `sanitize_control_tokens`.
- The request reader is stricter than the template. It takes the exact effort
  words and a tool with its `function` wrapper. The template also takes an
  effort word with spaces around it and a tool with no wrapper.
- Image and audio parts and a forced tool choice are refused.

The [reply parser](../src/chat_output/inkling.rs) is a state machine on token
IDs. It reads answer text, reasoning, tool calls and the end of the turn.

- A tool call is one message. Its body is a JSON object with the members
  `name` and `args`. The parser holds the body until the end of its message.
- The name and the arguments come from the object, as in the reply grammar of
  the checkpoint. The parser does not read the name in the message header.
- The arguments are the text that the model wrote. The parser does not change
  an argument to the type of the tool's schema.
- A body that is not a call of a declared tool is shown as answer text. This
  includes a body that is not one JSON object, a name that the request did not
  declare, and a body that a length limit cut before the end of its message.
- A request with no declared tool has no call to make. A tool call body is
  answer text there.

The [reference files](../tests/fixtures/inkling-chat/README.md) list the
sources and the proof limits.

Back: [server crate](../README.md).
