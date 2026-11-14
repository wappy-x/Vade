# Golden token IDs

This folder holds the reference answers for the exactness law, one file for
each served model. A file gives a hash of the token IDs that each gate
request must return. The `check-goldens` program compares the receipts of a
run with one file. It prints one line: `GOLDENS PASS` or `GOLDENS FAIL` with
the first difference.

| File | Model | Setup that made the rows |
| --- | --- | --- |
| [glm-5.3-flash-fp4-local.tsv](glm-5.3-flash-fp4-local.tsv) | GLM diagnostic text service (not qualified) | Two Sparks, plain decode |
| [qwen3.8-flash-next-fp4-local.tsv](qwen3.8-flash-next-fp4-local.tsv) | Qwen3.8 Flash-Next FP4-Local | One Spark and two Sparks, no draft used |

The files hold hashes only. They hold no prompt text and no token IDs.

## Check a run

A run needs a Spark. The check does not: it reads the receipt files that the
run wrote. Run it on any host, from the repository root:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml --bin check-goldens -- \
  tools/goldens/glm-5.3-flash-fp4-local.tsv RUN_DIR/receipt.jsonl
```

Give one receipt file or more. A receipt file has one JSON object on each
line. The check reads these fields: `cell` (the request name), `output_ids`
(the token IDs of the reply), `prompt_tokens` and `ok`.

The check passes only when all of these are true:

- Every row names a cell that is in the file.
- Every row has the golden prompt size and the golden token IDs.
- No row has `"ok": false`.
- Every cell in the file has one row or more.

On a failure, the line names the receipt file, the line number, the cell and
the first token that differs. The exit status is 1.

```text
GOLDENS FAIL first difference: run/receipt.jsonl:3 cell=c1040: token 3 of 32 differs (got ID 7); expected sha256 4e96..., got 7ab5...
```

## What a file holds

Fields are separated by tabs. A line that starts with `#` is a comment.

| Row | Fields |
| --- | --- |
| `format` | `vade-goldens-v1` |
| `model` | The model name. It is also the file name. |
| `epoch` | The answer set. It changes only when the answers change. |
| `note` | A name and free text for people. The check ignores note rows. |
| `cell` | Request name, prompt token count, reply token count, sha256, prefix fingerprint |

The sha256 covers the reply token IDs written as decimal numbers, one on each
line. For a file `reply.ids` in that form, `shasum -a 256 reply.ids` gives
the same value. The sha256 decides the result.

The prefix fingerprint has eight hex digits for each reply token: the start
of the sha256 of the reply up to that token. The check uses it only to find
the first token that differs.

## The host half

The tools test suite checks that every file in this folder is well formed,
and it runs in CI:

```sh
cargo +1.99.0 test --offline --locked --manifest-path tools/Cargo.toml --bin check-goldens
```

This does not prove that an engine gives these answers. Only a run on Sparks
and the check above prove that.

## Change the answers

A change to a `cell` row means that the engine gives different token IDs.
That needs the owner's approval ([AGENTS.md](../../AGENTS.md), Exactness law).

1. Make the new answers on Sparks, in two fresh loads or more.
2. Make the new file from those receipts:

   ```sh
   cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml --bin check-goldens -- \
     --make MODEL EPOCH RUN_1/receipt.jsonl RUN_2/receipt.jsonl > new.tsv
   ```

   The program makes no file when one cell has two different answers.
3. Give the new file the next epoch. Add `note` rows for the setup, the
   checkpoint, the requests and the source run. Do not edit a `cell` row by
   hand.
4. Commit the file with the change that needs it, and name the ruling.

Do not change another model's file in the same commit.
