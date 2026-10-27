//! Option help for the long-context evaluation command.
/// Help for `vade test longctx` or one of its subcommands.
pub fn longctx(command: Option<&str>) -> &'static str {
    match command {
        Some("prepare") => {
            r#"usage: vade test longctx prepare [-h]
                                 --tokenizer-command TOKENIZER_COMMAND [TOKENIZER_COMMAND ...]
                                 --frozen-bodies FROZEN_BODIES
                                 --ifeval-inputs IFEVAL_INPUTS --output OUTPUT
                                 [--documents {4,5,6,7,8}] [--seed SEED]
                                 [--keys-per-depth KEYS_PER_DEPTH]
                                 [--checkpoint-hash CHECKPOINT_HASH]
                                 [--max-tokens MAX_TOKENS]

Prepare the pinned legacy Flash-Next FP8 suite. This does not qualify FP4-Local.
--checkpoint-hash changes manifest metadata. Run and compare accept only the
pinned FP8 checkpoint hash.

options:
  -h, --help            show this help message and exit
  --tokenizer-command TOKENIZER_COMMAND [TOKENIZER_COMMAND ...]
  --frozen-bodies FROZEN_BODIES
  --ifeval-inputs IFEVAL_INPUTS
  --output OUTPUT
  --documents {4,5,6,7,8}
  --seed SEED
  --keys-per-depth KEYS_PER_DEPTH
  --checkpoint-hash CHECKPOINT_HASH
  --max-tokens MAX_TOKENS
"#
        }
        Some("run") => {
            r#"usage: vade test longctx run [-h] --manifest MANIFEST
                             [--top-k {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}]
                             [--floor-factor FLOOR_FACTOR] [--report REPORT]
                             --a-url A_URL --b-url B_URL [--c-url C_URL]
                             --output OUTPUT [--timeout TIMEOUT]
                             [--api-key-env API_KEY_ENV] [--a-build A_BUILD]
                             [--b-build B_BUILD] [--c-build C_BUILD]

Requires the pinned legacy Flash-Next FP8 manifest and compatible FP8 servers.
This command does not qualify FP4-Local.

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST
  --top-k {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}
  --floor-factor FLOOR_FACTOR
  --report REPORT
  --a-url A_URL
  --b-url B_URL
  --c-url C_URL
  --output OUTPUT
  --timeout TIMEOUT
  --api-key-env API_KEY_ENV
  --a-build A_BUILD
  --b-build B_BUILD
  --c-build C_BUILD
"#
        }
        Some("compare") => {
            r#"usage: vade test longctx compare [-h] --manifest MANIFEST
                                 [--top-k {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}]
                                 [--floor-factor FLOOR_FACTOR]
                                 [--report REPORT] --a-dumps A_DUMPS
                                 --b-dumps B_DUMPS [--c-dumps C_DUMPS]

Requires the pinned legacy Flash-Next FP8 manifest.
This command does not qualify FP4-Local.

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST
  --top-k {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32}
  --floor-factor FLOOR_FACTOR
  --report REPORT
  --a-dumps A_DUMPS
  --b-dumps B_DUMPS
  --c-dumps C_DUMPS
"#
        }
        _ => {
            r#"usage: vade test longctx [-h] {prepare,run,compare} ...

Deterministic long-context split/floor evaluation; Rust standard library.
This is the pinned legacy Flash-Next FP8 suite. It does not qualify FP4-Local.
prepare uses the existing CPU tokenizer example; capture uses /v1/score and
/v1/completions; compare also reads native full-vocabulary FP32 dump
directories. Generated prompts and captures are runtime artifacts, never
source fixtures.

positional arguments:
  {prepare,run,compare}
    prepare             freeze exact 32k/128k prompts using the existing CPU
                        tokenizer
    run                 capture paired servers and compare
    compare             compare existing native/NDJSON dumps

options:
  -h, --help            show this help message and exit
"#
        }
    }
}
