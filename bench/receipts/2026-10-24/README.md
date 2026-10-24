# 2026-10-24 FP4-Local receipts

These folders are sanitized copies of the source folders quoted in the
[current benchmark tables](../../../docs/benchmarks.md#current-2026-10-24).
Each folder keeps its original name, except that the cloud provider's name in
three timing-run folder names is replaced by `cloud`. The `73f7a9d` folders
are the release-build home cards, measured on main `73f7a9d`; the v0.1.0
engine differs from it only in how startup admission counts free memory (main
`2c64d0a`). The `4681d8b` folders are the morning cards on main `4681d8b`.

| Folder | Kind | Files |
| --- | --- | --- |
| [1438 one Spark, drafting ON, thinking OFF, release build](2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1444 one Spark, drafting OFF, thinking OFF, release build](2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1445 one Spark, drafting ON, thinking ON/low, release build](2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [1458 one Spark, drafting OFF, thinking ON/low, release build](2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [1444 home pair, drafting OFF, release build](2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [1452 home pair, drafting ON, release build](2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [0650 one Spark, drafting ON, thinking OFF](2026-10-24_0650_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [0656 one Spark, drafting OFF, thinking OFF](2026-10-24_0656_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [0657 one Spark, drafting ON, thinking ON/low](2026-10-24_0657_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [0710 one Spark, drafting OFF, thinking ON/low](2026-10-24_0710_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [0711 home pair, drafting ON](2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [0720 home pair, drafting OFF](2026-10-24_0720_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [1006 two cloud VMs, drafting OFF](2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1046 one cloud VM, drafting OFF](2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1042 one cloud VM, ON/OFF/ON](2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card/) | Timing run, not a card | `README.txt`, `timing-on.json` |
| [1114 two cloud VMs, ON/OFF/ON](2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card/) | Timing run, not a card | `README.txt`, `pair-timing-on.json`, `pair-c8-on.json`, `pair-on-health.json` |
| [1259 two cloud VMs, engine of main 407b6ae](2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card/) | Timing run, not a card | `README.txt`, `pair-timing-fix.json`, `pair-c8-fix.json`, `pair-fix-health.json` |

## What was copied

From every source folder: each `*.card.txt`, the `README.txt`, and each other
`*.json` file of at most 200 KB. Not copied:

- `*.card.json`, the full RigMark JSON receipts (1.6–2.0 MB each). They
  contain the raw prompts and replies. The `JSON sha256` prefix printed on a
  card identifies that unpublished file.
- `SHA256SUMS`, because it hashes the unsanitized originals.
- Run logs and rank logs (`*.log`).
- The `receipts/` and `logs/` subfolders of the two release-build home pair
  folders (raw per-request bodies and replies, per-Spark command, environment
  and telemetry files, server logs). Their `README.txt` lists them; they stay
  private.

Folders cited inside a README but absent from the benchmark tables, such as
the 2026-10-22 comparison cards, are not included.

## Sanitization

Each copied file was rewritten line by line with these replacements and no
other edits:

| Original | Replaced with |
| --- | --- |
| Home Spark hostnames | `spark-a` and `spark-b` (the home pair), `spark-c` and `spark-d` (morning one-Spark drafting ON and OFF cards), `spark-e` and `spark-f` (release-build one-Spark drafting ON and OFF cards) |
| Cloud VM hostnames | `<cloud-vm>` (the second VM keeps its `-w1` suffix) |
| Cloud cluster name and entry ID | `<cloud-cluster>` and `<cloud-id>` |
| Cloud provider name | `<cloud>` in text, `cloud` in folder names |
| Private fabric addresses | `<fabric>` |
| Private filesystem paths (absolute, home-relative (`~/...`) and private lab paths) | `<path>` |

The same rules also cover user names and the remaining home Spark names; none
occurred in these files. The labels `spark-c` to `spark-f` name four different
machines: the morning and release-build one-Spark cards did not run on the same
Sparks. The two release-build home pair folders arrived with Spark names,
fabric addresses (`<fabric-a>`, `<fabric-b>`) and the control secret already
replaced by their producer; their `~/...` paths were then replaced by `<path>`. Loopback addresses (`127.0.0.1`) and port numbers are
unchanged. In the RigMark card boxes, trailing padding was adjusted on edited
lines so the right border stays aligned where possible.

The `README.txt` notes were then edited once more for publication: internal
script, lane and merge names were removed or described in words, the release
pair's run ID was replaced by `<run id>`, and clock times inside the notes were
converted from the local clock to UTC (`Z`). Folder names keep the local clock.

No measurement changed. A token-by-token comparison of every number in each
original file against its sanitized copy found differences only inside the
replaced hostnames, IDs, addresses and paths, and in the README note clock
times converted to UTC. Every number cited in the
benchmark tables appears verbatim in the matching receipt, except one derived
value: the 1259 run's ON prose median 31.6 is the median of the two samples 29.8
and 33.4 that its README lists.
