# Download models with vade pull

`vade pull` downloads a model from the Hugging Face Hub into a folder that
Vade can serve. It continues interrupted downloads, checks every file and
records what it downloaded. `vade models` lists the models on this machine and
tells you which Vade command can serve each one.

## The checkpoints Vade serves

Each command below downloads one checkpoint at the publisher revision that
Vade is qualified on. The file counts and sizes come from
`vade pull --dry-run` on 2026-10-29.

- **Qwen3.8 Flash-Next FP4-Local.** It has 32 files and 102.26 GB. Serve it
  with `vade up`. `vade launch` starts the same engine and opens chat.

  ```sh
  vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
  ```

- **Qwen3.8 Flash-Next FP8.** It has 143 files and 185.56 GB. Serve it with
  `vade up --fp8` from a controller ([FP8 install](install-fp8.md)).

  ```sh
  vade pull Qwen/Qwen3.8-Flash-Next-FP8@236dfdf285828023ca3bcd3f37366c58a3469b13
  ```

- **Legacy Qwen3-8B BF16.** It has 14 files and 16.40 GB. Serve it with
  `vade run qwen3-8b --weights-source ~/models/Qwen/Qwen3-8B`.

  ```sh
  vade pull Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218
  ```

The FP4-Local command puts the model in
`~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`. When no checkpoint is saved
yet, `vade up` looks in that folder first, so you can run it next with no
other setting.

A checkpoint that an earlier `vade up` or `vade launch` saved comes before
`VADE_MODELS` and before that folder. To serve a new folder in that case, use a new Vade
home. The `Next:` line of the pull output gives the command with a home that
is not in use yet, such as
`VADE_HOME=~/.vade-<org>-<name> VADE_MODELS=<folder> vade launch`. A `Note:`
line says why.

The download takes a long time. Over SSH, run it inside `tmux`, or under
`nohup` with its output in a log file:

```sh
nohup vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300 > ~/vade-pull.log 2>&1 &
```

To follow it, run `tail -f ~/vade-pull.log`. If the connection drops or you
stop the pull, run the same command again. It continues from where it stopped.

## Usage

```text
vade pull <org>/<name>[@revision] [--include PATTERN]... [--to DIR] [--dry-run]
vade models [DIR] [--json]
```

- `<org>/<name>` is the repository name on the Hub.
- `@revision` is a branch, a tag or a full commit hash. The default is `main`.
  Pull resolves it to one commit first and downloads every file from that
  commit, so a push to the branch during the download cannot mix two versions.
  Give the full commit hash to get the same files every time.
- `--to DIR` sets the model folder. The default is `~/models/<org>/<name>`.
  The folder can be a symlink, for example to a larger disk. Pull then writes
  into the folder it points to, and puts the checksum list beside that folder,
  where `vade up` looks for it. A symlink inside the folder stops the pull.
- `--dry-run` lists the files and their sizes. It downloads nothing and does
  not create the model folder.

### Which files pull downloads

Without `--include`, pull downloads the files Vade needs to serve a model:

- the weights (`*.safetensors`) and their index;
- every JSON file, which includes `config.json`, `generation_config.json`, the
  tokenizer configuration and any quantization configuration;
- the tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`,
  `tokenizer.model`, `*.tiktoken`);
- the chat template (`*.jinja`);
- `README.md`, `LICENSE` and `NOTICE`.

Pull skips other files, such as `.gitattributes`, `*.bin` and `*.pth` weights
and images. It also skips any path that has a part starting with a dot.

`--include PATTERN` replaces this selection. Give it more than once to add more
patterns. No pattern selects a path that has a part starting with a dot. So a
repository cannot write into the pull state or over `.vade-model.json`.

In a pattern, `*` matches any text, including `/`, and `?` matches one
character. Put the pattern in quotes so that the shell does not expand it:

```sh
vade pull org/name --include '*.json' --include 'tokenizer*' --to ~/models/org/name-metadata
```

## What pull checks

Pull reads the file list from the Hub API at the resolved commit
(`/api/models/<org>/<name>/tree/<commit>?recursive=true`, page by page). For
each file it then does these steps:

1. If the file is already in the folder, pull checks it. Pull does not read a
   file again when the same folder's last pull checked it and it has not
   changed since.
2. Otherwise, pull downloads it from `/<org>/<name>/resolve/<commit>/<path>`
   into a partial file in `.vade-pull/` inside the model folder. A partial
   file from an earlier attempt continues from its last byte. Pull follows
   the redirect that the Hub sends for a file, even for a small file. When a
   reply sends more bytes than the listing gives, pull stops it and removes
   the partial file. When a reply ends early, pull continues it. Pull never
   takes a short reply as the whole file.
3. Pull checks the size, then the hash: the SHA-256 that the Hub keeps for
   large (LFS) files, or the Git blob ID for small files. When a file fails
   the check, pull removes it and stops with the file name and both values.
4. Pull moves the checked file into place.

A transfer that stalls (less than 1 byte per second for 120 seconds), a broken
connection and HTTP 408, 429 and 5xx answers get up to six attempts. The wait
between attempts starts at 2 seconds and doubles each time. An HTTP 401, 403
or 404 answer stops the pull with the next step to take. Ctrl-C stops the pull
and its download, also while pull checks a file. The partial file stays for
the next run, and pull writes no records. A symlink, a FIFO or any other file
that is not a regular file where pull writes in `.vade-pull/` stops the pull.
A file listing whose next page is on another address, or whose `Link` header
does not parse, stops it too. An API reply over 64 MiB, or with more than
1 MiB of headers, stops the pull with any curl version.

Only one pull can write to a model folder at a time. A second pull of the same
folder stops with a message. The lock is a file in `.vade-pull/`, and the
operating system releases it when the pull exits.

## What pull writes

When every file passes its check, pull writes two files:

- **`.vade-model.json`** in the model folder. It records the repository, the
  revision you asked for, the commit, the Hub address, the time in UTC, the
  license from the model card, the total bytes, the support state and, for
  each file, its path, size, SHA-256 and how it was checked.
- **`<folder>.sha256`** beside the model folder, such as
  `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local.sha256`. It has one
  `sha256sum` row for each file. For an FP4-Local checkpoint, `vade launch`,
  `vade up` and `vade install` read this list and hash every listed file
  before they serve the model. The FP8 route keeps the engine's own receipt
  and does not read this list. `vade run qwen3-8b` checks its own pinned
  hashes. The [fetch helper](../ops/fetch-model.sh) writes a list with the
  same name and form.

A pull with `--include` into a folder that an earlier pull of the same commit
filled keeps the rows of the other files in both records. Before it changes a
file, pull stops in these cases:

- A pull with `--include` would leave out of its records a file that is in
  the folder, such as a shard of a checkpoint that another tool downloaded.
- A pull with `--include` follows an update that stopped after it replaced
  a file. Run `vade pull` without `--include` to finish the update first.
- The list beside the folder names a file that is still in the folder and
  that the new records would leave out.

In each case, pull the whole selection again without `--include`, or use
`--to` with a new folder.

Pull then prints the support state and the command that serves the model.

## Support state

Pull and `vade models` give each folder one of these states:

| State | Meaning |
| --- | --- |
| Supported: Qwen3.8 Flash-Next FP4-Local | The checks `vade launch` runs on the config, the quantization configuration, the shard index, the shards, the generation configuration and the tokenizer all pass. `vade launch` and `vade up` serve it |
| Supported: Qwen3.8 Flash-Next FP8 | The same checks pass for the FP8 format. Serve it from a controller with `vade up --fp8` |
| Supported: the pinned Qwen3-8B BF16 files | Every file of the `vade run qwen3-8b` catalog is present with its pinned SHA-256 |
| Not supported yet | No route can serve the folder. The reason is the first check that failed, such as an unknown architecture or a missing shard |
| Incomplete | A pull of this folder did not finish, or an update stopped after it replaced a file. Run the same `vade pull` again |

These checks read metadata only, never the weights. A supported state does not
prove that the model runs. The device checks in
[adding a model](adding-a-model.md) qualify a model. A model that downloads
but shows "not supported yet" needs the work that page describes before Vade
can serve it.

## vade models

`vade models` looks up to three folders deep under `~/models`, or under the
folder you name. It shows each model folder with its size on disk, the
repository and commit from `.vade-model.json` and its support state. A
symlink in the folder that points to a model folder is listed too. Add
`--json` to get one `vade.models.v1` object for scripts.

```text
$ vade models /tmp/pulls
Models in /tmp/pulls:
  MODEL        SIZE  SOURCE                                            STATE
  fp4       70.3 MB  nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e  Not supported yet
                                                                       Model shard "tensors/model-00001-of-00021.safetensors" is missing or unreadable.
```

This folder came from a pull with `--include` patterns that left out the
weights, so the launcher checks stop at the first missing shard.

## Gated and private models

Some repositories are private, or they require you to accept the publisher's
terms. For these repositories, do these steps:

1. Accept the terms on the model page while you are signed in to the Hub.
2. Make a read token in your Hugging Face settings.
3. Set `HF_TOKEN` to the token, or save it as the only text in
   `~/.vade/hf-token` (`$VADE_HOME/hf-token` when `VADE_HOME` is set). The file
   can be a symlink, but it must be a regular file of at most 4 KiB. Make it
   readable only by you: `chmod 600 ~/.vade/hf-token`.

`HF_TOKEN` takes priority over the file. Pull gives the token to curl on its
standard input, so it does not show in the process list. curl 7.58 and later
send it only to the Hub host, not to the download host that the Hub redirects
to. Pull never prints the token or writes it to a file. The checkpoints in the
list above download without a token.

## Mirrors and transport

Set `HF_ENDPOINT` to use a Hub mirror, for example
`HF_ENDPOINT=https://hub.example.org`. The address must use HTTPS and have no
path. Pull accepts plain HTTP only for an address on this machine
(`127.0.0.1`, `localhost` or `[::1]`). The tests use such an address.

Vade has no TLS code of its own. Pull runs the system `curl` from `PATH` with
a fixed list of arguments and no shell. It skips your `.curlrc`, allows only
HTTPS (also after redirects), resumes with `--continue-at -` and sets a
30-second connect timeout. Proxy variables such as `https_proxy` still apply.

## Difference from the fetch helper

[`ops/fetch-model.sh`](../ops/fetch-model.sh) is the older downloader. It needs a
full commit hash and a source checkout. It downloads every file in the
repository, prints nothing until it finishes and refuses a folder that holds
any other file. `vade pull` is part of the `vade` command. It downloads only
the files that Vade needs, accepts a branch or a tag, shows progress, works
with gated models and records `.vade-model.json`. Both check every size and
hash, and both write the same `<folder>.sha256` list, so `vade launch` checks a
folder from either one in the same way.
