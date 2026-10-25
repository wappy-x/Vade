# Hardware and network setup

**How to buy, cable, address and check the network that joins DGX Sparks for Vade.**

New to some of these terms? Each one is explained the first time it appears,
and the [Glossary](GLOSSARY.md) has them all.

> **In 30 seconds**
>
> - **Two Sparks and one cable** is the simplest linked setup. We have published measurements on two and four of our own Sparks.
> - Use a **200 Gb/s QSFP cable**, straight from one Spark to the other. No switch needed.
> - Give each cable **two small private subnets** (one per rail) and **MTU 9000**.
> - Check it: a jumbo ping, then an RDMA test. Expect **about 196 Gb/s per cable**.
> - Vade finds the RDMA devices by itself from the addresses you give it.
> - **No RDMA?** Vade can run in [TCP mode](../INSTALL.md#rdma-or-tcp) (`SEQPAR_RDMA=0`). Our cloud GB10 cards used TCP over 200G links; we have not measured TCP over ordinary Ethernet.

<br>

## On this page

1. [What to buy](#1-what-to-buy)
2. [How a Spark's network ports work](#2-how-a-sparks-network-ports-work)
3. [Pick a layout](#3-pick-a-layout)
4. [Cable it](#4-cable-it)
5. [Give every link an address](#5-give-every-link-an-address)
6. [Prepare the OS](#6-prepare-the-os)
7. [Check every link](#7-check-every-link)
8. [Do not change these](#8-do-not-change-these)
9. [Troubleshooting](#9-troubleshooting)

<br>

---

## 1. What to buy

| Item | How many | Notes |
| --- | --- | --- |
| **NVIDIA DGX Spark** | 2 or more | GB10 chip, 128 GB memory shared by CPU and GPU. FP4-Local fits on one Spark; more Sparks read long prompts faster. The legacy FP8 checkpoint needs a pair. |
| **QSFP cable, 200 Gb/s** | 1 per link | Direct-attach copper (DAC), QSFP56 200G or QSFP112. Short runs (about 1 m or less) are typical for passive copper. Check NVIDIA's [Spark clustering page](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) for its current cable list. |
| **Ordinary network** | 1 | The Sparks' normal Ethernet or Wi-Fi, for SSH and downloads. Keep it separate from the QSFP links. |
| **A controller** (optional) | 0 or 1 | A Linux or macOS computer that runs the `vade` command, the SSH tunnel and the coding assistant, and drives the Sparks. A Spark can also act as its own controller. |
| **Switch** (optional) | 0 | Not needed for any layout on this page. See [with a switch](#with-a-switch). |

**Cable speed matters.** A 100 Gb/s cable links at half speed. Vade ignores any
interface slower than 100 Gb/s when it looks for fabric links. Our links report
`200000Mb/s` over 4 lanes; check yours with `ethtool` (see [step 7](#7-check-every-link)).

<br>

---

## 2. How a Spark's network ports work

Each Spark has a **ConnectX-7** network card (NVIDIA's high-speed network chip)
with **two QSFP ports** on the back.

The surprise: **each QSFP port shows up in Linux as two network interfaces.**
Each half sits on its own PCIe path and carries about half the cable's speed.
This guide calls each half a **rail**.

| Physical port | Rail | Network interface (typical) | RDMA device (typical) |
| --- | --- | --- | --- |
| Port 0 | rail 0 | `enp1s0f0np0` | `rocep1s0f0` |
| Port 0 | rail 1 | `enP2p1s0f0np0` | `roceP2p1s0f0` |
| Port 1 | rail 0 | `enp1s0f1np1` | `rocep1s0f1` |
| Port 1 | rail 1 | `enP2p1s0f1np1` | `roceP2p1s0f1` |

- One rail alone moves about **100–109 Gb/s** one way.
- Both rails together move about **196 Gb/s** one way. That is the full cable.
- Vade splits each large transfer across both rails of a cable.

**Confirm the names on your machine.** Run `ibdev2netdev` (or `rdma link show`).
Do not trust the labels printed on the case: on at least one Spark, the port
marked "P0" is the one Linux calls port 1.

**RDMA** (remote direct memory access) lets one Spark write straight into the
other's memory without going through the operating system. On Ethernet it runs
as **RoCE v2** (RDMA over Converged Ethernet). The Spark has no GPUDirect RDMA,
so Vade stages transfers through ordinary host memory
([NVIDIA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)).

<br>

---

## 3. Pick a layout

How Vade spreads the model over several Sparks depends on the checkpoint.
Each Spark runs one **rank** (one engine process), and **rank 0** answers HTTP.

### FP4-Local, the current checkpoint

FP4-Local fits on one Spark. With more Sparks, Vade splits it by layers: each
Spark holds a block of consecutive layers, runs them, and passes the result to
the next Spark. On two Sparks this is the **layer pipeline**; on more than two
it is a **chain**.

| Sparks | Layout | Cables | How to start it |
| --- | --- | --- | --- |
| **1** | One Spark on its own | none | `vade launch` on the Spark, or `vade up` from a controller |
| **2** | Layer pipeline | 1 | `vade up --sparks 2` |
| **3 to 24** | Layer chain, in cable order | one fewer than the Sparks | `vade up --sparks N`, or `--hosts a,b,c` in cable order |
| **Many chats** | Replicas: a separate one-Spark server on each Spark, behind a local gateway | none | `vade up --replicas N` (2 to 16) |

A chain does not need the cable that would close it into a ring. Before you
choose more than two Sparks, run `vade install` from current source, so the
copy staged on each Spark supports the chain. Vade's startup checks cannot
prove that a cabling layout we have not tested gives the same tokens on every
Spark, so test yours before you rely on it. The steps are in
[INSTALL.md, Two and more Sparks](../INSTALL.md#two-and-more-sparks).

### Legacy FP8

The older Flash-Next FP8 checkpoint needs at least two Sparks. It splits the
model with **tensor parallelism (TP)**: every layer is cut into equal slices,
and the Sparks work on each token together. Its layer sizes divide evenly by
2, 4 or 8, but not by 3 or 6.

Every legacy FP8 launch needs the `--fp8-legacy` flag, on four Sparks too.
Without it, `vade up` stops and tells you to add it.

| Sparks | Layout | Cables | Status |
| --- | --- | --- | --- |
| **2** | 2-way TP, ring | 1 | `vade up --fp8-legacy`. Two is the legacy default. |
| **4** | 4-way TP, path (chain) | 3 (a ring of 4 also works) | `vade up --fp8-legacy --sparks 4`; use the [four-Spark guide](four-sparks.md). We do not compare its speed on this page. |
| **6** | 6-way TP, path | 5 (a ring of 6 also works) | `vade up --fp8-legacy --sparks 6` starts it, but the model splits unevenly. **Not recommended.** |
| **1, 3, 5, 7+** | none | none | `vade up --fp8-legacy` refuses: "Legacy FP8 needs 2, 4, or 6 machines." |

**Ring or path?** Here two ranks use a **ring**, where data goes around in a
loop. Four and six ranks use a **path**, a chain with two open ends, so you do
not need the cable that would close the ring.

**Tip: cable a ring anyway.** In a ring, every pair of neighbors has its own
direct cable. Six Sparks in a ring can serve as three separate pairs, or as
one long chain, without recabling.

<br>

---

## 4. Cable it

**One rule covers every layout:** port 0 of each Spark goes to port 1 of the
next Spark. Vade does not require it, but it keeps addresses predictable.

### Two Sparks (direct)

```text
  spark-1                      spark-2
 ┌────────┐                   ┌────────┐
 │ port 0 ├───── cable 1 ─────┤ port 1 │
 │ port 1 │                   │ port 0 │   (second ports unused)
 └────────┘                   └────────┘
```

### Four Sparks (ring)

```text
   spark-1 ──── cable 1 ──── spark-2
      │                        │
   cable 4                  cable 2
      │                        │
   spark-4 ──── cable 3 ──── spark-3
```

| Cable | From | To |
| --- | --- | --- |
| 1 | spark-1 port 0 | spark-2 port 1 |
| 2 | spark-2 port 0 | spark-3 port 1 |
| 3 | spark-3 port 0 | spark-4 port 1 |
| 4 | spark-4 port 0 | spark-1 port 1 |

For a four-Spark chain or path, cables 1–3 are enough. Cable 4 closes the ring, so
spark-1 + spark-2 and spark-3 + spark-4 can also run as two separate pairs.

### Six Sparks (ring)

Same rule, six cables: spark-N port 0 → spark-(N+1) port 1, and spark-6 port 0
→ spark-1 port 1. Our six-Spark ring measured the same speed on all six cables.

### With a switch

Instead of cabling neighbors directly, you can plug each Spark into a 200 Gb/s
Ethernet switch that supports RoCE. Each Spark's `fabric.<peer>` setting is
then its own address on the shared switch subnet, and both rails need a subnet.
**Not yet tested: Vade has not been run through a switch.** Direct cables are
simpler and are what we measured.

<br>

---

## 5. Give every link an address

Every rail of every cable gets its **own small private subnet** (`/24`). The two
ends of the cable use host numbers `.1` and `.2`. No gateway, no routing: these
links only connect direct neighbors.

**Example scheme (copy it, or pick your own):** use `10.77.0.0/16`, two `/24`s
per cable. The end using its **port 0** is `.1`; the end using its **port 1** is `.2`.

| Cable | Ends | Rail 0 subnet | Rail 1 subnet |
| --- | --- | --- | --- |
| 1 | spark-1 port 0 ↔ spark-2 port 1 | `10.77.0.0/24` | `10.77.1.0/24` |
| 2 | spark-2 port 0 ↔ spark-3 port 1 | `10.77.2.0/24` | `10.77.3.0/24` |
| 3 | spark-3 port 0 ↔ spark-4 port 1 | `10.77.4.0/24` | `10.77.5.0/24` |
| 4 | spark-4 port 0 ↔ spark-1 port 1 | `10.77.6.0/24` | `10.77.7.0/24` |

So on cable 1, for example:

| Spark | Interface | Address |
| --- | --- | --- |
| spark-1 | `enp1s0f0np0` (port 0, rail 0) | `10.77.0.1/24` |
| spark-1 | `enP2p1s0f0np0` (port 0, rail 1) | `10.77.1.1/24` |
| spark-2 | `enp1s0f1np1` (port 1, rail 0) | `10.77.0.2/24` |
| spark-2 | `enP2p1s0f1np1` (port 1, rail 1) | `10.77.1.2/24` |

**Before you pick a range,** make sure nothing else uses it, on every Spark:

```sh
ip -4 route show table all | grep '10\.77\.'     # expect no output
```

Avoid your home or office LAN range, VPN ranges and Docker's `172.17.0.0/16`.

**MTU 9000 on every fabric interface, at both ends.** MTU is the largest packet
size. 9000 ("jumbo frames") is faster than the default 1500: we measured
196 Gb/s at 9000 and 185 Gb/s at 1500. With Ethernet MTU 9000, the RDMA path
MTU shows as **4096**. That is expected.

**What Vade needs from you** is one address per neighbor. In the Vade config,
each machine's `fabric.<peer>` is **its own rail-0 address on the cable to that
peer**. Vade discovers rail 1 by itself: it looks for a second subnet that both
ends share on a 100 Gb/s-or-faster interface, and uses it when present.

```ini
[machine spark-1]
# spark-1's own address on cable 1
fabric.spark-2 = 10.77.0.1

[machine spark-2]
# spark-2's own address on cable 1
fabric.spark-1 = 10.77.0.2
```

The full config file is in [cli/examples/vade.conf](../cli/examples/vade.conf).
Never commit your real addresses; keep the config in an ignored `.local/` folder.

<br>

---

## 6. Prepare the OS

The repository ships **OS-setup shell scripts** in
[ops/scripts/spark-setup/](../ops/scripts/spark-setup/). They are plain shell
scripts, separate from the Rust build. **Every script only previews by
default**; add `--apply` after you read the preview.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `init-config.sh pair\|ring` | Makes private placeholder files: `config.env` and `links.tsv` (one row per rail). |
| 2 | `render.sh` | Writes a netplan file (`40-cx7.yaml`): static address, MTU 9000, no DHCP, per fabric interface. Also a hosts fragment. |
| 3 | `apply.sh … packages` | Installs the OS's own RDMA tools: verbs library, providers, `perftest`. (Use `rdma link show` if `ibdev2netdev` is missing.) |
| 4 | `apply.sh … network` | Installs the netplan file. Refuses if a rail carries your SSH session or is already configured. You then run `netplan try` from the console. |
| 5 | `apply.sh … rdma` | Adds your account to an RDMA group, sets device permissions and **unlimited locked memory (memlock)**. |
| 6 | `apply.sh … kernel` | Only for kernel `7.0.0-1019-nvidia`: adds NVIDIA's `kho=off` fix. Never reboots for you. |
| 7 | `updates.sh pause\|restore` | Pauses automatic OS updates during benchmarks, then restores them. |
| 8 | `verify.sh` | PASS/FAIL for driver, RDMA access, memlock, addresses, carrier, MTU, GIDs, jumbo ping and SSH. `--bw-server`/`--bw-client` run the bandwidth test. |

Every change is recorded in a transaction you can roll back.
The full walkthrough, with every command, is **[Spark system setup](spark-setup.md)**.

> **Where this stands:** these scripts have passed Mac syntax and mock checks. The
> complete blank-OS run on real Sparks is **not yet tested**. Our working network
> was built by hand from NVIDIA's playbooks
> ([two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks),
> [three in a ring](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring)).

**Locked memory, in plain words.** RDMA has to pin memory so the OS cannot
move it. Linux limits how much an account may pin. Vade needs that limit to be
`unlimited` **in the exact SSH session that starts it**, and `vade up` refuses
to start otherwise. Some SSH routes (for example Tailscale SSH) skip the normal
login limits. Use ordinary OpenSSH over the LAN.

<br>

---

## 7. Check every link

Run these on each Spark, for every fabric interface, with your own interface
names and addresses in place of the examples.

### Checklist

- [ ] Cable detected at full speed
- [ ] MTU 9000
- [ ] RDMA device maps to the right interface
- [ ] Jumbo ping reaches the neighbor
- [ ] RDMA bandwidth test passes, both directions, both rails
- [ ] `ulimit -l` says `unlimited` in your launch session

### Commands and what you should see

**Carrier and speed**

```sh
ethtool enp1s0f0np0 | grep -E 'Speed|Link detected'
```

```text
	Speed: 200000Mb/s
	Link detected: yes
```

**MTU**

```sh
ip link show enp1s0f0np0 | grep -o 'mtu [0-9]*'
```

```text
mtu 9000
```

**RDMA device ↔ interface**

```sh
ibdev2netdev
```

```text
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
...
```

**Jumbo ping** (8972 bytes of data + 28 bytes of headers = 9000; `-M do`
forbids splitting, so this fails loudly if any MTU is wrong)

```sh
ping -c 3 -M do -s 8972 -I enp1s0f0np0 10.77.0.2
```

```text
3 packets transmitted, 3 received, 0% packet loss
```

**RDMA bandwidth.** Start the server end first, then the client end, using
the setup scripts' private config (`$CFG`, see [spark-setup.md](spark-setup.md#8-verify-every-rail-and-the-actual-launch-session)).
The `1` is the row number of that rail in each Spark's own `links.tsv`. Without
`--apply` the script only previews.

```sh
# server end (spark-1)
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-server 1 --apply
# client end (spark-2)
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-client 1 --apply
```

Under the hood it runs `perftest`'s `ib_write_bw` for 5 seconds (`-D 5`),
with a 25-second timeout. If you would rather run it by hand, here is the same
test for rail 0 of cable 1. Rail 1 uses its own device and address, and port
`18516` instead of `18515` (the defaults `init-config.sh` writes into
`links.tsv`). Run the server command on spark-1 and the client command on
spark-2. At each prompt, type that Spark's numeric IPv4 RoCE v2 GID index,
which you recorded in step 3 of
[spark-setup.md](spark-setup.md#record-each-rails-real-gid):

```sh
# server end (spark-1)
read -r -p 'Server GID index: ' GID_INDEX
timeout 25s ib_write_bw -d rocep1s0f0 -i 1 -x "$GID_INDEX" -p 18515 --bind_source_ip=10.77.0.1 \
  -m 4096 -F --report_gbits -D 5 -s 65536
# client end (spark-2), pointing at the server's rail address
read -r -p 'Client GID index: ' GID_INDEX
timeout 25s ib_write_bw -d rocep1s0f1 -i 1 -x "$GID_INDEX" -p 18515 --bind_source_ip=10.77.0.2 \
  -m 4096 -F --report_gbits -D 5 -s 65536 10.77.0.1
```

### Numbers we measured

| Test | Our result |
| --- | --- |
| One rail, one direction | about 100–109 Gb/s |
| Both rails of one cable, one direction | **about 196 Gb/s** at MTU 9000 (185 Gb/s at MTU 1500) |
| Both rails, both directions at once (`ib_write_bw -b`) | about 391 Gb/s total per cable |
| Six-Spark ring, all six cables | 48 of 48 RDMA tests passed at those speeds |

`Gb/s` is gigabits per second. Divide by 8 for gigabytes: 196 Gb/s ≈ 24.5 GB/s.

**Right after `vade up`, check Vade itself.** `vade logs` shows the last 64 KiB
of each rank's log. On FP4-Local with more than one Spark, each rank prints a
line such as
`flashnext pp stage rank=<r> world=<n> layers=<first>..<end> upstream=<bool> downstream=<bool>`:
its rank, the number of ranks (`world`), the block of layers it runs, and
whether it receives from a previous Spark (`upstream`) and sends to a next one
(`downstream`).

On legacy FP8, each rank prints a line that starts like
`flash_serve transport rank=0 world=2 rdma=true`, and a `seqpar_links` line.
`rdma=false` means that rank is on TCP.
On two Sparks, `rails=2` means Vade is using both halves of the cable;
`rails=1` means it found only one. On 4 or 6 Sparks it reads
`rails_right=2 rails_left=2`; the first rank shows `rails_left=1` and the last
rank `rails_right=1`, because each has a neighbor on one side only.

**Coming from vLLM or SGLang?** Vade does not use NCCL; it has its own RDMA
transport. If you already run NVIDIA's `nccl-tests`, our three-Spark ring
measured about **23 GB/s** bus bandwidth for `all_gather` at 1 GB, which
matches the published two-Spark reference.

<br>

---

## 8. Do not change these

| Leave this alone | Why |
| --- | --- |
| **MTU 9000 at both ends** | A mismatch silently drops big packets. Jumbo ping catches it. |
| **No gateway or DHCP on fabric interfaces** | These are point-to-point links. A gateway can steal your default route. |
| **Fabric subnets separate from your LAN, VPN and Docker** | Overlaps send traffic out the wrong interface. |
| **DGX OS's own driver, firmware and RDMA packages** | Do not install another OFED or NIC driver. Vade uses the stock `libibverbs.so.1`. |
| **`kho=off` on kernel `7.0.0-1019-nvidia`** | NVIDIA's fix for a multi-Spark RoCE bug. See the [advisory](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254). |
| **Unlimited memlock for the Vade account** | Without it, RDMA memory registration fails. |
| **Your LAN and Wi-Fi profiles** | The setup scripts never touch them. You need them for SSH. |
| **The interface carrying your SSH session** | Reconfiguring it locks you out. Use the local console for network changes. |

If you run a firewall on the fabric, allow these between direct neighbors:
ICMP, SSH, **RoCE v2 (UDP 4791)** and **Vade's TCP ports 29960 and 29961**
(`tp_port` and `tp_port + 1`).

<br>

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Link detected: no` | Cable not seated, wrong port, or bad cable | Reseat both ends. Try another cable. Check `ibdev2netdev` for which port is Up. |
| Speed `100000Mb/s`, not `200000Mb/s` | 100 Gb/s cable | Use a 200 Gb/s cable. |
| Ping fails completely | Address on the wrong interface (case labels ≠ Linux names), or subnets differ at the two ends | Run `ibdev2netdev` and `ip -br address` on both ends. Move the address to the interface that is Up. |
| Small ping works, `-s 8972 -M do` fails | MTU is not 9000 somewhere | Set MTU 9000 on both ends of that rail. |
| `ib_write_bw` cannot open the device or register memory | Not in the RDMA group, or memlock limited | Run the `rdma` setup step, then **log out and back in**. Check `ulimit -l`. |
| Bandwidth about half of 196 Gb/s per cable | Only one rail configured or used | Give rail 1 its own subnet at both ends. On legacy FP8, check `vade logs` for `rails=2`. |
| Bandwidth about 185, not 196 Gb/s | MTU 1500 | Set MTU 9000. |
| `vade up`: "RDMA launch needs unlimited soft and hard memlock in this SSH session" | The SSH alias Vade uses does not get unlimited locked memory (some routes, such as Tailscale SSH, skip login limits) | Point the config's `ssh =` at an OpenSSH alias over the LAN that prints `unlimited` for `ulimit -Sl` and `ulimit -Hl`. Or use TCP mode (`export SEQPAR_RDMA=0` on the controller). |
| Engine log: "RDMA socket has no QSFP interface" or "no IPv4 RoCE-v2 GID on neighbour interface" | `fabric.<peer>` is not an address on a 100 Gb/s-or-faster ConnectX-7 interface, or that interface has no RoCE v2 GID yet (link down) | Use the Spark's own fabric address on that cable. Check carrier. |
| Engine log: "RDMA required but unavailable on one or more ranks" | RDMA mode (the default) and one rank could not open RDMA | Check memlock, RDMA group membership and `ibdev2netdev` on that rank. Without a QSFP link, use TCP mode (`export SEQPAR_RDMA=0` on the controller). |
| `vade install` or `vade doctor`: "… needs … for RDMA. Have the administrator prepare RDMA, or set SEQPAR_RDMA=0 for TCP over the existing network, then rerun this command." | The RDMA library, an RDMA device or unlimited memlock is missing on that Spark | Run the OS-setup `packages` and `rdma` steps, then log in again. Or use TCP mode. |
| `vade install` or `vade doctor`: fabric check fails | `fabric.<peer>` values swapped or wrong, or ICMP blocked | Each value is the machine's **own** address on its link to that peer. The check uses ping, so allow ICMP between neighbors. |
| RoCE errors or hangs across Sparks on kernel `7.0.0-1019-nvidia` | Known kernel regression | Add `kho=off` (setup step 6) or move to NVIDIA's fixed kernel. |
| `rx_err_lane_*_phy` counters keep rising | Marginal cable or connector | Small corrected-error counts are normal. If they climb, reseat or replace the cable. |

Still stuck? Before you ask for help, collect `ibdev2netdev`, `ip -br address`,
`ethtool <iface>` and `vade logs` from both ends, with your real addresses removed.

<br>

**Back:** [README](../README.md) · **Next:** [INSTALL.md, Route B](../INSTALL.md#route-b-from-a-controller) and [Two and more Sparks](../INSTALL.md#two-and-more-sparks) → · Legacy FP8 only: [legacy fresh install](fresh-install.md)
