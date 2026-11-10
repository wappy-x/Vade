# Chat renderers

The [serving bridge](../tokenizer.rs) selects the renderer from checkpoint
template data. The [GLM renderer](glm.rs) handles text, thoughts, function
calls, and string tool results. The tokenizer crate keeps the Qwen renderer.

The GLM golden tests cite lines in the [pinned template](../../templates/glm.jinja).
They use the block-trimming rules in the
[Transformers template compiler](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/chat_template_utils.py).

The [external fixture test](glm_external_tests.rs) compares published vLLM
prompt bytes. Its [source record](../../tests/fixtures/glm-upstream/README.md)
names the scope and exclusions. It does not run an upstream tokenizer.

Back: [source map](../README.md).

The [published prompt check](glm_external_tests.rs) compares retained external
prompt strings. Its [fixture note](../../tests/fixtures/glm-upstream/README.md)
states the exclusions. It does not prove checkpoint tokenizer runtime parity.
