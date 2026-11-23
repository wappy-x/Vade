# Reference token IDs for the message markup

These files check the chat renderer and the reply parser of the
[checked template copy](../../../templates/inkling2-mini.jinja) against the
maker's own code. They hold token IDs, text and token bytes. They hold no
weights and no code of the maker.

| File | Content |
| --- | --- |
| `reference-prompts.jsonl` | 26 chat requests and the prompt token IDs that the maker's template and tokenizer gave for each: 13 prompts, each with effort `none` and with no effort field. |
| `reference-cases.jsonl` | 85 chat requests of the lab, each with the answer of the maker's template and tokenizer: the rendered text and the prompt token IDs for 74, and the error text for 11. The cases cover the effort words and numbers, the order of messages, content parts, reasoning in the history, tools, tool calls, tool results and control spellings in caller text. The IDs are those of the library's guarded mode (`sanitize_control_tokens`), which keeps control tokens out of caller text. For the 9 cases where the plain mode gives other IDs, the line holds them as `ids_unguarded`. |
| `reference-replies.jsonl` | 7 replies of the model as token IDs, with the reasoning and the answer cut out of the reference's own text of the same reply. 4 end with the end-of-turn token. 3 end at a length limit. |
| `reference-tool-replies.jsonl` | 18 replies of the model, as token IDs, for 9 requests with tools, each with effort `none` and with no effort field. Each line has the names of the declared tools and the expected events: reasoning, answer text and calls. 8 replies hold 10 tool calls. 2 replies end at the length limit inside a tool call. |
| `vocabulary.tsv` | The bytes, in hex, of each of the 1,017 token IDs that the four files use. The word `control` marks an added token. |

## Source

- Checkpoint files: `thinkingmachines/Inkling-2-Mini` at revision
  `b85c3cfa615fc413cc9af3379f491b963f3b586a`, license Apache-2.0.
  `tokenizer.json` gave the token bytes. `chat_template.jinja` has SHA-256
  `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`.
- Reference code: the `transformers` library, release 5.24.0. Its
  `apply_chat_template` gave the prompt IDs. Its greedy generation on the
  checkpoint gave the replies of both reply files, on one DGX Spark.
- The same library gave the text, the token IDs and the error text of the 85
  cases. It ran the template and the tokenizer only.
- The requests: six prompts in the RigMark message form, and seven fixed
  prompts of the lab. One long prompt of the reference set (8,015 IDs) is left
  out to keep the files small.

## What the tests prove

[`src/tokenizer/inkling_tests.rs`](../../../src/tokenizer/inkling_tests.rs)
renders each request. It checks that the reference list has the same control
tokens at the same places, and that the ordinary tokens between them have
exactly the bytes of the rendered text. It does not run the tokenizer on the
text. So it proves the rendered prompt, and it needs no split rule.
For a line with text, it also checks that the rendered text equals the text of
the reference. For a line with an error text, it checks that this engine
refuses the request too.

[`tests/inkling_output.rs`](../../inkling_output.rs) feeds each reply of the
two reply files to the state machine on token IDs and cuts the result at every
byte.

[`tests/inkling_text_contract.rs`](../../inkling_text_contract.rs) is ignored by
default. It loads the public tokenizer files from `.local/inkling-chat` and
checks that the encoded prompt equals the reference list, ID for ID. It needs
the split rule of that tokenizer in the tokenizer crate.

## Limits

- Four of the 85 cases differ from the reference by a rule of this engine. The
  unit test holds the list and proves each row. This engine refuses two cases
  that the template takes, so 72 of the 74 token ID lists are compared.
- The comparison of control spellings in caller text is against the guarded
  mode of the library. The plain mode of the library turns such a spelling
  into a control token. This engine never does.
- The expected events of `reference-tool-replies.jsonl` are not the output of
  the reference library's reply reader. A script of the lab made them from the
  same token IDs. It follows the reply grammar of the checkpoint and the rules
  of this parser, and it is not the code under test. A call that the length
  limit cut is answer text there, by a rule of this engine.
- The other replies with tool calls in the tests are hand-made. One has the
  form of the reference library's own parsing test.
- A match of prompt IDs is not proof that the model answers right.

Back: [tests](../../README.md).
