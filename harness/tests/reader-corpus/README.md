# Reader test files

These small files are local test data. The files contain no private source
data. The office files use short test namespace names. Image files contain
the headers that the readers use. These cases do not test pixel decoding.

Each input has saved Markdown in `golden/<input name>.md.golden`.
The goldens keep the format header, warnings, and page, slide, sheet,
chapter, or file markers.
The text cases cover UTF-8, UTF-16 in both byte orders, and Windows-1252.
The other cases cover each format in the reader registry.

Run the tests from `harness/`:

```sh
RUSTFLAGS='-D warnings' cargo +1.99.0 test --offline --locked -p vade-harness --lib readers::corpus_tests -- --nocapture
```

The tests cut each input at every byte. They change each byte, try wrong
files, and test random bytes from a fixed seed. Package tests rebuild ZIP
checksums after each XML part change. This lets broken XML reach its reader.
Each probe uses the default caps and small caps.

Other tests set large header fields and check the input, output, page, and
decode caps. A refused file is a valid result. A panic fails the test.

To add a case, add its input and saved Markdown. Then add its name to
`CORPUS` in `src/readers/corpus_tests.rs`. Review each change to a golden.
Tests never rewrite these files.

Encoded text inputs use `.text` so the documentation check does not read
them as UTF-8. Goldens use `.md.golden` because their links belong to the
source documents.
