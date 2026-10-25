# 2026-10-24 FP4-Local receipts

These folders are sanitized copies of the source folders quoted in the
[current benchmark tables](../../../docs/benchmarks.md#current-results-2026-10-24).
The table's Kind column uses four terms:

- **RigMark card**: the result file that one run of the public RigMark
  benchmark writes.
- **Custom runner summary**: the summary that the custom runner for our
  two-Spark pair writes. It is not a RigMark card.
- **Timing run, not a card**: a separate measurement on the cloud GB10 VMs.
  Its results are in the folder's README, with the timing JSON next to it.
- **Replicas run, not a card**: two separate one-machine servers behind the
  local gateway, measured at C8 and C16. Its results are in the folder's README.

Each folder name starts with the date and the local clock time of the run, so
1438 is the 14:38 run. Folders keep their original names, except that in
three timing-run folder names the cloud provider's name is replaced by
`cloud`.

The `73f7a9d` folders are the release-build cards from our own Sparks,
measured on main `73f7a9d`. The v0.1.0 engine changes one thing compared with
that build: its startup memory check, called admission in the engine, now
counts free memory differently (main `2c64d0a`). The `4681d8b` folders are the
morning cards on main `4681d8b`.

| Folder | Kind | Files |
| --- | --- | --- |
| [1438 One DGX Spark, drafting ON, thinking OFF, release build](2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1444 One DGX Spark, drafting OFF, thinking OFF, release build](2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1445 One DGX Spark, drafting ON, thinking ON/low, release build](2026-10-24_1445_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [1458 One DGX Spark, drafting OFF, thinking ON/low, release build](2026-10-24_1458_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [1444 Two DGX Sparks, drafting OFF, release build](2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [1452 Two DGX Sparks, drafting ON, release build](2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [0650 One DGX Spark, drafting ON, thinking OFF](2026-10-24_0650_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [0656 One DGX Spark, drafting OFF, thinking OFF](2026-10-24_0656_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [0657 One DGX Spark, drafting ON, thinking ON/low](2026-10-24_0657_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-on-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [0710 One DGX Spark, drafting OFF, thinking ON/low](2026-10-24_0710_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-thinking-on/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [0711 Two DGX Sparks, drafting ON](2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [0720 Two DGX Sparks, drafting OFF](2026-10-24_0720_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) | Custom runner summary | `README.txt`, `thinking-off.card.txt` |
| [1006 Two cloud GB10 VMs, drafting OFF](2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1046 One cloud GB10 VM, drafting OFF](2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1042 One cloud GB10 VM, drafting ON/OFF/ON](2026-10-24_1042_qwen3.8-flash-next-nvfp4_local-cloud-1vm-drafting-timing-not-a-card/) | Timing run, not a card | `README.txt`, `timing-on.json` |
| [1114 Two cloud GB10 VMs, drafting ON/OFF/ON](2026-10-24_1114_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-timing-not-a-card/) | Timing run, not a card | `README.txt`, `pair-timing-on.json`, `pair-c8-on.json`, `pair-on-health.json` |
| [1259 Two cloud GB10 VMs, drafting ON/OFF/ON, engine of main 407b6ae](2026-10-24_1259_qwen3.8-flash-next-nvfp4_local-cloud-2spark-drafting-on-after-fix-timing-not-a-card/) | Timing run, not a card | `README.txt`, `pair-timing-fix.json`, `pair-c8-fix.json`, `pair-fix-health.json` |

## v0.1.0 release cards (5a22651)

These folders come from the v0.1.0 release test. Every machine installed the
public build `5a22651` fresh from GitHub, following `INSTALL.md`, and then ran
the cards against that install. The machines were our six home Sparks, named
`home-1` to `home-6`; four cloud GB10 VMs, named `vm-a` to `vm-d`, that form
one cluster we call cloud A; and one cloud container with a GB10, which we call
cloud B. Each folder's `README.txt` says how that machine was installed and
started.

Some folder names end with the machine or cloud name. Rows marked *(cloud A)*
ran on one of the cloud A VMs; their notes do not say which one.

Two folders are replicas runs: two separate one-machine servers behind the
local gateway, measured at C8 and C16. They are not RigMark cards; their
results are in the folder's `README.txt` and summary files.

| Folder | Kind | Files |
| --- | --- | --- |
| [1937 One DGX Spark (home-1), drafting OFF, thinking OFF](2026-10-24_1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1938 One DGX Spark (home-3), drafting OFF, thinking OFF](2026-10-24_1938_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1938 One DGX Spark (home-5), drafting OFF, thinking OFF](2026-10-24_1938_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-home-5/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [1940 One DGX Spark (home-2), drafting OFF, thinking OFF](2026-10-24_1940_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1941 One DGX Spark (home-4), drafting OFF, thinking OFF](2026-10-24_1941_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1941 One DGX Spark (home-6), drafting OFF, thinking OFF](2026-10-24_1941_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-home-6/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [1943 One DGX Spark (home-1), drafting ON, thinking OFF](2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1945 One DGX Spark (home-3), drafting ON, thinking OFF](2026-10-24_1945_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1946 One DGX Spark (home-2), drafting ON, thinking OFF](2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1946 One DGX Spark (home-5), drafting ON, thinking OFF](2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-home-5/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [1947 One DGX Spark (home-4), drafting ON, thinking OFF](2026-10-24_1947_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1947 One DGX Spark (home-6), drafting ON, thinking OFF](2026-10-24_1947_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-home-6/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [2054 One DGX Spark (home-5), drafting ON, thinking ON/low](2026-10-24_2054_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-on-home-5/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2054 One DGX Spark (home-6), drafting ON, thinking ON/low](2026-10-24_2054_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-on-home-6/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2108 One DGX Spark (home-5), drafting OFF, thinking ON/low](2026-10-24_2108_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-on-home-5/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2108 One DGX Spark (home-6), drafting OFF, thinking ON/low](2026-10-24_2108_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-on-home-6/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2111 One DGX Spark (home-1), drafting ON, thinking ON/low](2026-10-24_2111_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-on-home-1/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2114 One DGX Spark (home-2), drafting ON, thinking ON/low](2026-10-24_2114_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-on-home-2/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2125 One DGX Spark (home-1), drafting OFF, thinking ON/low](2026-10-24_2125_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-on-home-1/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2128 One DGX Spark (home-2), drafting OFF, thinking ON/low](2026-10-24_2128_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-on-home-2/) | RigMark card | `README.txt`, `thinking-on.card.txt` |
| [2012 Two DGX Sparks (home-1 + home-2), drafting OFF, thinking OFF](2026-10-24_2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [2014 Two DGX Sparks (home-3 + home-4), drafting OFF, thinking OFF](2026-10-24_2014_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2015 Two DGX Sparks (home-5 + home-6), drafting OFF, thinking OFF](2026-10-24_2015_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-home-5-home-6/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [2019 Two DGX Sparks (home-1 + home-2), drafting ON, thinking OFF](2026-10-24_2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [2022 Two DGX Sparks (home-3 + home-4), drafting ON, thinking OFF](2026-10-24_2022_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2022 Two DGX Sparks (home-5 + home-6), drafting ON, thinking OFF](2026-10-24_2022_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-home-5-home-6/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json`, `vade-verify.json` |
| [2051 Two DGX Sparks as replicas (home-1 + home-2), drafting OFF and ON, thinking OFF](2026-10-24_2051_qwen3.8-flash-next-nvfp4_local-2spark-replicas_5a22651-drafting-off-on-thinking-off/) | Replicas run, not a card | `README.txt`, `numbers.txt`, `vade-bench.json`, `vade-bench.txt`, `vade-verify-gateway.json`, `vade-verify-replica-18192.json`, `vade-verify-replica-18193.json` |
| [1946 One cloud GB10 VM (vm-a), drafting OFF, thinking OFF](2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1946 One cloud GB10 VM (vm-b), drafting OFF, thinking OFF](2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vmb/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1954 One cloud GB10 VM (vm-a), drafting ON, thinking OFF](2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1954 One cloud GB10 VM (vm-b), drafting ON, thinking OFF](2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vmb/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1955 One cloud GB10 VM (cloud A), drafting OFF, thinking OFF](2026-10-24_1955_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [1956 One cloud GB10 VM (cloud A), drafting OFF, thinking OFF](2026-10-24_1956_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2003 One cloud GB10 VM (cloud A), drafting ON, thinking OFF](2026-10-24_2003_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2005 One cloud GB10 VM (cloud A), drafting ON, thinking OFF](2026-10-24_2005_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2032 One cloud GB10 VM (cloud A), drafting OFF, thinking OFF](2026-10-24_2032_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2032 One cloud GB10 VM (cloud A, second VM), drafting OFF, thinking OFF](2026-10-24_2032_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-w2/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2040 One cloud GB10 VM (cloud A, second VM), drafting ON, thinking OFF](2026-10-24_2040_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-w2/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2041 One cloud GB10 VM (cloud A), drafting ON, thinking OFF](2026-10-24_2041_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2027 Two cloud GB10 VMs (vm-a + vm-b), drafting OFF, thinking OFF](2026-10-24_2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2035 Two cloud GB10 VMs (vm-a + vm-b), drafting ON, thinking OFF](2026-10-24_2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt`, `vade-bench.json` |
| [2113 Two cloud GB10 VMs as replicas (vm-a + vm-b), drafting ON and OFF, thinking OFF](2026-10-24_2113_qwen3.8-flash-next-nvfp4_local-2spark-replicas_5a22651-drafting-on-off-thinking-off-cloud-a/) | Replicas run, not a card | `README.txt`, `health-gateway-end.json`, `health-gateway-start.json`, `health-replica-a-end.json`, `health-replica-a-start.json`, `health-replica-b-end.json`, `health-replica-b-start.json`, `replicas-numbers.json`, `vade-bench-vm-a-bundle.json`, `vade-bench-vm-b-bundle.json`, `vade-verify-replica-a.json`, `vade-verify-replica-b.json` |
| [2204 Four cloud GB10 VMs (vm-a to vm-d), drafting OFF, thinking OFF](2026-10-24_2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [2213 Four cloud GB10 VMs (vm-a to vm-d), drafting ON, thinking OFF](2026-10-24_2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1943 One cloud GB10 VM (cloud B), drafting OFF, thinking OFF](2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-b/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [1949 One cloud GB10 VM (cloud B), drafting ON, thinking OFF](2026-10-24_1949_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-b/) | RigMark card | `README.txt`, `thinking-off.card.txt` |
| [2026 One cloud GB10 VM (cloud B), drafting ON, thinking ON/low](2026-10-24_2026_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-on-cloud-b/) | RigMark card | `README.txt`, `thinking-on.card.txt` |

For these folders we also copied the other `*.txt` summaries. We did not copy
the replicas runs' per-request JSON files, which are larger than the size limit
below, or the measurement scripts (`*.py`).

## What was copied

From every source folder we copied each `*.card.txt`, the `README.txt`, and
every other `*.json` file of at most 200 KB. We did not copy:

- `*.card.json`, the full RigMark JSON receipts (1.6–2.0 MB each). They
  contain the raw prompts and replies. The `JSON sha256` prefix printed on a
  card identifies that unpublished file.
- `SHA256SUMS`, because it hashes the unsanitized originals.
- Run logs and rank logs (`*.log`).
- The `receipts/` and `logs/` subfolders of the two release-build folders from
  our own pair (raw per-request bodies and replies, per-Spark command,
  environment and telemetry files, server logs). Their `README.txt` lists
  them; they stay private.

Folders that a README cites but the benchmark tables do not, such as the
2026-10-22 comparison cards, are not included.

## Sanitization

We rewrote each copied file line by line with these replacements and made no
other edits:

| Original | Replaced with |
| --- | --- |
| Hostnames of our own Sparks | `spark-a` and `spark-b` (our two-Spark pair), `spark-c` and `spark-d` (morning one-Spark drafting ON and OFF cards), `spark-e` and `spark-f` (release-build one-Spark drafting ON and OFF cards) |
| Cloud VM hostnames | `<cloud-vm>` (the second VM keeps its `-w1` suffix) |
| Cloud cluster name and entry ID | `<cloud-cluster>` and `<cloud-id>` |
| Cloud provider name | `<cloud>` in text, `cloud` in folder names |
| Private fabric addresses | `<fabric>` |
| Private filesystem paths: absolute, home-relative (`~/...`) and private lab paths | `<path>` |
| Hostnames of our six home Sparks (v0.1.0 folders) | `home-1` to `home-6`, in text and in folder names |
| Cloud A VM hostnames (v0.1.0 folders) | `vm-a`, `vm-b`, `vm-c` and `vm-d` |
| Cloud A provider name (v0.1.0 folders) | `cloud-a` in text and in folder names |
| Cloud B provider name (v0.1.0 folders) | `cloud B` in text, `cloud-b` in folder names |
| Cloud entry IDs and SSH host addresses (v0.1.0 folders) | `<cloud-id>` (none occurred) |
| Other private addresses (v0.1.0 folders) | `<ip>` (none occurred) |
| Point-to-point link and fabric addresses (v0.1.0 folders) | `<link-a>` to `<link-d>`, `<fabric-a>`, `<fabric-b>`, `<peer>`, `<box>` |
| User names (v0.1.0 folders) | removed |

In the earlier folders, the same rules also covered user names and the names
of our other Sparks; none of them occurred there. The labels `spark-c` to `spark-f` name four
different machines: the morning and release-build one-Spark cards did not run
on the same Sparks.

In the two release-build folders from our pair, the Spark names, fabric
addresses (`<fabric-a>`, `<fabric-b>`) and the control secret had already been
replaced before the folders reached us; we then replaced their `~/...` paths
with `<path>`.

In the v0.1.0 folders, the cloud A worker pairs became `vm-a+vm-b` and
`vm-c+vm-d`. Their `README.txt` notes were not edited by hand after that: they
keep the local clock times, our internal job labels and measurement-script
names.

Loopback addresses (`127.0.0.1`) and port numbers are unchanged. In the
RigMark card boxes, we adjusted trailing padding on edited lines so the right
border stays aligned where possible.

We then edited the `README.txt` notes once more for publication:

- internal script, lane (work stream) and merge names were removed or
  described in words;
- the release pair's run ID was replaced by `<run id>`;
- clock times inside the notes were converted from the local clock to UTC
  (`Z`). Folder names keep the local clock.

No measurement changed. A token-by-token comparison of every number in each
original file against its sanitized copy found differences only inside the
replaced hostnames, IDs, addresses and paths, and in the README note clock
times converted to UTC. Every number cited in the benchmark tables appears
verbatim in the matching receipt, with one derived exception: the 1259 run's
ON prose median 31.6 is the median of the two samples 29.8 and 33.4 that its
README lists.
