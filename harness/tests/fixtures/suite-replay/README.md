# Suite replay records

These records come from `suite --dry-run --only markdown-spec-qa,refuse-outside-path,structured-output`.
The command used the scripts in `tests/agent-suite` on a host. No model or
Spark ran. `replay.json` records the Spark count as `0`.

Each request body was replaced with `{"messages":[]}`. Each elapsed time
was set to `0`. The replies remain as the suite recorded them. The token
IDs and usage fields come from the script server. They are not model data.

`broken.jsonl` is a cut-off record for the error test.

[`suite_replay.rs`](../../suite_replay.rs) replays these replies. It also
changes the recorded patch and answer to test each quality rule and a
failed completion check.
