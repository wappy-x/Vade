# Rules for agents in this repository

Follow every rule below for every change.

1. Every new Rust source file starts with this exact first line:
   `// Part of the tally crate. License: Apache-2.0.`
2. Never change, add or delete anything under `vendor/`. That code is a copy
   of another project.
3. Give every public function a `///` doc comment.
4. Record every new public function in CHANGELOG.md, under the heading
   `## Unreleased`, as one line in the form `- added: <module>::<function>`.
   Put new lines directly below the heading, above older lines.
5. Do not add dependencies to Cargo.toml.
