# Eight Sparks: one ring of eight

**How to cable eight Sparks (GB10 machines) into one ring without a switch,
give every link an address, check the ring and start the first two-Spark and
eight-Spark runs.**

Read [Hardware and network setup](network-setup.md) first. This page uses its
words (port, rail, ring, path) and its rules (MTU 9000, one subnet per
rail, no gateway). The OS steps are in [Spark system setup](spark-setup.md).

> **In 30 seconds**
>
> - A Spark has two QSFP ports. Eight cables join the eight Sparks into **one
>   ring**: each Spark has one cable to each of its two ring neighbors.
> - A run of 2, 4 or 8 Sparks uses an **arc**: Sparks that are neighbors on
>   the ring, listed in cable order. Rank 0, the first, answers HTTP.
> - From a ring of six, you move one cable end and add two cables.
> - Check each new Spark against the Sparks you already have **before** you
>   connect a cable: memory, firmware, driver and PCIe links.
> - `vade doctor --fabric --ring` checks that every Spark has exactly two
>   neighbors, that every cable passes on both rails, and that the ring closes.
>   It does not measure bandwidth. The bandwidth test does that.

<br>

## On this page

1. [The ring and its arcs](#1-the-ring-and-its-arcs)
2. [Cable plan](#2-cable-plan)
3. [Address plan](#3-address-plan)
4. [Controller config: machines and groups](#4-controller-config-machines-and-groups)
5. [Check the ring](#5-check-the-ring)
6. [Start a run on an arc](#6-start-a-run-on-an-arc)
7. [Saturday, step by step](#7-saturday-step-by-step)
8. [What vade prints for eight Sparks](#8-what-vade-prints-for-eight-sparks)

This page calls the Sparks **spark-1** to **spark-8**. These are labels for
this page, not host names. Put a label with the same name on each Spark and on
both ends of each cable before you start. The mapping from labels to your
machines and the real addresses belong in your private notes, not in this
repository.

<br>

---

## 1. The ring and its arcs

Each Spark has two QSFP ports, port 0 and port 1. Cable *c* joins port 0 of
spark-*c* to port 1 of the next Spark. Cable 8 joins spark-8 back to spark-1
and closes the ring.

```text
   spark-1 ── 1 ── spark-2 ── 2 ── spark-3 ── 3 ── spark-4
      │                                               │
      8                                               4
      │                                               │
   spark-8 ── 7 ── spark-7 ── 6 ── spark-6 ── 5 ── spark-5
```

**Place the Sparks as the drawing shows:** spark-1 to spark-4 in one row,
spark-8 to spark-5 in a second row beside it. Then every cable joins two
Sparks that stand side by side, cables 4 and 8 included. Short passive copper
cables (about 1 m or less, see
[network-setup.md](network-setup.md#1-what-to-buy)) do not reach from one end
of a straight row of eight to the other. Give each Spark's external power
adapter its own space with air around it. Do not stack the adapters.

**The arc rule.** A run on 2, 4 or 8 Sparks uses an arc: Sparks that follow
each other on the ring. List them in cable order, so each Spark is cabled to
the next one in the list. The list is the rank order. Rank 0 is the first
Spark and answers HTTP. The arc can start at any Spark and go in either
direction. This page uses the first Spark of the arc in cable order as rank
0. Whether the choice of rank 0 changes speed is to be measured.

**Named arcs.** The example config
([cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf)) names
these arcs as groups. Groups in one layout use no Spark twice, so they can run
at the same time.

| Group | Sparks in rank order | Rank 0 | Cables between the ranks | Layout |
| --- | --- | --- | --- | --- |
| `eight` | spark-1, spark-2, …, spark-8 | spark-1 | 1 to 7 | `eight` |
| `quad-a` | spark-1, spark-2, spark-3, spark-4 | spark-1 | 1, 2, 3 | `quads` |
| `quad-b` | spark-5, spark-6, spark-7, spark-8 | spark-5 | 5, 6, 7 | `quads` |
| `pair-1` | spark-1, spark-2 | spark-1 | 1 | `doubles` |
| `pair-2` | spark-3, spark-4 | spark-3 | 3 | `doubles` |
| `pair-3` | spark-5, spark-6 | spark-5 | 5 | `doubles` |
| `pair-4` | spark-7, spark-8 | spark-7 | 7 | `doubles` |
| `pair-23` | spark-2, spark-3 | spark-2 | 2 | `doubles-b` |
| `pair-45` | spark-4, spark-5 | spark-4 | 4 | `doubles-b` |
| `pair-67` | spark-6, spark-7 | spark-6 | 6 | `doubles-b` |
| `pair-81` | spark-8, spark-1 | spark-8 | 8 | `doubles-b` |
| `one-1` … `one-8` | one Spark each | that Spark | none | `singles` |

With `doubles` and `doubles-b`, every cable carries one pair. Other arcs of
four, such as spark-3 to spark-6 or spark-7, spark-8, spark-1, spark-2, start
with `--hosts` ([section 6](#6-start-a-run-on-an-arc)).

**What runs on each arc in this build.**

| Sparks | FP4-Local | FP8 |
| --- | --- | --- |
| 1 | One Spark, no fabric. `--replicas 8` runs eight independent servers behind one local gateway. | Refused. |
| 2 | The tensor split, the only two-Spark path. It needs RDMA. | Two-way TP. |
| 4 | The layer chain of four. | Four-way TP on a path. Today it gives other token IDs than two Sparks ([benchmarks.md](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)). |
| 8 | The layer chain of eight. | Refused: `FP8 needs 2, 4, or 6 machines` ([cli/src/config.rs](../cli/src/config.rs)). |

FP4-Local runs tensor parallel only on two Sparks (the tensor split). On four
or eight Sparks it runs the layer chain: each Spark holds a block of layers.
This build has no four-Spark or eight-Spark tensor split for FP4-Local.

**One server per group.** Each group is its own server on its own
`local_port`. No gateway joins two groups, so a client must send its chats to
each group's port itself. Only `vade up --replicas N` puts one local gateway
in front of several engines, and each of those engines runs on one Spark.

A run of three or more Sparks uses a path (`TP_TOPOLOGY=path`, see
`topology()` in [cli/src/state.rs](../cli/src/state.rs)). So a run on all
eight uses cables 1 to 7, and cable 8 is not part of its path. For two Sparks,
the session topology is a ring over their one cable.

**Why one ring.** Every pair of neighbors and every run of four neighbors is an
arc of the one ring, so all the layouts above and the run on all eight start
without recabling. Two rings of four would give each quad a closing cable, but
a run of four uses a path in this build, so it would not use that cable, and a
run on all eight would need two cables moved.

**Faults.** One failed cable leaves a path of eight. Arcs that do not use that
cable still run. A failed Spark stops the arcs that hold it.

**Memory.** Each Spark has 128 GB shared by CPU and GPU. An arc of four has
512 GB, and all eight have 1,024 GB.

<br>

---

## 2. Cable plan

Follow one row at a time, with the cable in your hand. "Port 0" and "port 1"
are the names Linux uses, not the labels on the case: on at least one Spark
the port marked "P0" is the one Linux calls port 1. Run `ibdev2netdev` on each
Spark to see which port is which (see
[network-setup.md, step 2](network-setup.md#2-how-a-sparks-network-ports-work)).

| Cable | From | To |
| --- | --- | --- |
| 1 | spark-1 port 0 | spark-2 port 1 |
| 2 | spark-2 port 0 | spark-3 port 1 |
| 3 | spark-3 port 0 | spark-4 port 1 |
| 4 | spark-4 port 0 | spark-5 port 1 |
| 5 | spark-5 port 0 | spark-6 port 1 |
| 6 | spark-6 port 0 | spark-7 port 1 |
| 7 | spark-7 port 0 | spark-8 port 1 |
| 8 | spark-8 port 0 | spark-1 port 1 |

After each cable, check that both ends show `(Up)` in `ibdev2netdev`.

**From a ring of six.** If six Sparks already run as one ring cabled with the
same rule (cables 1 to 6, where cable 6 goes from spark-6 port 0 to spark-1
port 1):

1. Move the cable 6 end from spark-1 port 1 to **spark-7 port 1**.
2. Add cable 7: spark-7 port 0 to spark-8 port 1.
3. Add cable 8: spark-8 port 0 to **spark-1 port 1**.

spark-2 to spark-6 keep all their cables.

If your ring of six was not cabled with this port rule, find the cable 6 end
on spark-1 by its address, not by its port number. `ip -4 -br addr` shows the
interface that holds the cable 6 address, and `ibdev2netdev` shows its port.
`sudo ethtool -p <interface> 10` blinks that port's light for 10 seconds,
if the port supports it.
Put a tag on that cable end before you stop anything. The cable to spark-2
stays where it is.

<br>

---

## 3. Address plan

Eight cables with two rails each need **16 subnets**. On your Sparks, give
each rail its own `/24`, as [spark-setup.md](spark-setup.md#choose-the-addresses)
says. The setup scripts accept only `/24`: a `links.tsv` row with another
prefix is refused.

**The rule.** Cable *c* has two rails. Rail *r* (0 or 1) uses subnet number
`n = 2c - 1 + r`: cable 1 uses subnets 1 and 2, cable 8 uses subnets 15
and 16. Take 16 unused `/24` subnets from one private range and number them 1
to 16. The port 0 end of the cable is host 1 of its subnets, the port 1 end
host 2. An address belongs to its cable end, so it moves with that end.

Before you use the range, make sure that nothing else on any Spark uses it:
`ip -4 route show table all` must show no route into it that your plan does
not add, and it must not overlap your LAN, VPN or containers. Write the real
addresses only in your private configs.

**The example addresses.** The tables below and
[cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf) use
`192.0.2.0/24`, a range reserved for documentation. One `/24` cannot hold 16
`/24` subnets, so the examples cut it into one `/29` per rail: subnet *n* is
`192.0.2.(8(n-1))/29`. Cable 1 rail 0 is `192.0.2.0/29` with `.1` and
`.2`, and cable 1 rail 1 is `192.0.2.8/29` with `.9` and `.10`. Do not put
these addresses or the `/29` prefix on your Sparks.

### Subnets per cable

| Cable | Port 0 end, host 1 | Port 1 end, host 2 | Rail 0: subnet, example | Rail 1: subnet, example |
| --- | --- | --- | --- | --- |
| 1 | spark-1 | spark-2 | 1, `192.0.2.0/29` | 2, `192.0.2.8/29` |
| 2 | spark-2 | spark-3 | 3, `192.0.2.16/29` | 4, `192.0.2.24/29` |
| 3 | spark-3 | spark-4 | 5, `192.0.2.32/29` | 6, `192.0.2.40/29` |
| 4 | spark-4 | spark-5 | 7, `192.0.2.48/29` | 8, `192.0.2.56/29` |
| 5 | spark-5 | spark-6 | 9, `192.0.2.64/29` | 10, `192.0.2.72/29` |
| 6 | spark-6 | spark-7 | 11, `192.0.2.80/29` | 12, `192.0.2.88/29` |
| 7 | spark-7 | spark-8 | 13, `192.0.2.96/29` | 14, `192.0.2.104/29` |
| 8 | spark-8 | spark-1 | 15, `192.0.2.112/29` | 16, `192.0.2.120/29` |

### Addresses per Spark

These are the example addresses. On your Sparks, each row is host 1 or host 2
of a `/24`, as the rule says. The interface names are the typical names from
[network-setup.md](network-setup.md#2-how-a-sparks-network-ports-work).
Confirm them on each Spark with `ibdev2netdev`.

| Spark | Port | Rail 0 (`enp1s0f*`) | Rail 1 (`enP2p1s0f*`) | Cable | Neighbor |
| --- | --- | --- | --- | --- | --- |
| spark-1 | 0 | `192.0.2.1/29` | `192.0.2.9/29` | 1 | spark-2 |
| spark-1 | 1 | `192.0.2.114/29` | `192.0.2.122/29` | 8 | spark-8 |
| spark-2 | 0 | `192.0.2.17/29` | `192.0.2.25/29` | 2 | spark-3 |
| spark-2 | 1 | `192.0.2.2/29` | `192.0.2.10/29` | 1 | spark-1 |
| spark-3 | 0 | `192.0.2.33/29` | `192.0.2.41/29` | 3 | spark-4 |
| spark-3 | 1 | `192.0.2.18/29` | `192.0.2.26/29` | 2 | spark-2 |
| spark-4 | 0 | `192.0.2.49/29` | `192.0.2.57/29` | 4 | spark-5 |
| spark-4 | 1 | `192.0.2.34/29` | `192.0.2.42/29` | 3 | spark-3 |
| spark-5 | 0 | `192.0.2.65/29` | `192.0.2.73/29` | 5 | spark-6 |
| spark-5 | 1 | `192.0.2.50/29` | `192.0.2.58/29` | 4 | spark-4 |
| spark-6 | 0 | `192.0.2.81/29` | `192.0.2.89/29` | 6 | spark-7 |
| spark-6 | 1 | `192.0.2.66/29` | `192.0.2.74/29` | 5 | spark-5 |
| spark-7 | 0 | `192.0.2.97/29` | `192.0.2.105/29` | 7 | spark-8 |
| spark-7 | 1 | `192.0.2.82/29` | `192.0.2.90/29` | 6 | spark-6 |
| spark-8 | 0 | `192.0.2.113/29` | `192.0.2.121/29` | 8 | spark-1 |
| spark-8 | 1 | `192.0.2.98/29` | `192.0.2.106/29` | 7 | spark-7 |

**From a ring of six.** The moved cable 6 end takes its addresses with it:
spark-1 port 1 gives `192.0.2.82` and `192.0.2.90` to spark-7 port 1, and
gets `192.0.2.114` and `192.0.2.122` (cable 8). spark-7 and spark-8 get every
row above. spark-2 to spark-6 keep all their addresses.

Every fabric interface uses one IPv4 address, MTU 9000, no gateway and no
DHCP. The
[setup scripts](spark-setup.md#1-make-a-private-cable-map) write these
settings from a private `links.tsv` with one row per rail. Use the `ring`
template on every Spark: each Spark has two cables.

<br>

---

## 4. Controller config: machines and groups

The controller config has one `[machine]` section per Spark. Each
`fabric.<peer>` value is the Spark's **own rail 0 address on the cable to that
peer**. Vade finds rail 1 by itself. Two Sparks are linked in the config when
each names the other. Start from the complete example,
[cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf), keep
your copy in a private file such as `.local/eight.conf`, and select it with
`VADE_CONFIG`.

```ini
[machine spark-1]
ssh = spark-1-ssh-alias
fabric.spark-2 = 192.0.2.1
fabric.spark-8 = 192.0.2.114
```

**Groups.** A `[group NAME]` section names the Sparks of one server.

| Key | Meaning |
| --- | --- |
| `hosts` | Machine names in **rank order**, separated by commas. Rank 0, the first, answers HTTP. List them in cable order along the arc. |
| `local_port` | The port on the controller for this server. The default is the global `local_port`. |
| `layout` | Optional. Groups with the same layout run at the same time. They must not share a machine or a local port. Groups in different layouts may share machines. |

```ini
[group quad-a]
hosts = spark-1,spark-2,spark-3,spark-4
local_port = 18080
layout = quads

[group quad-b]
hosts = spark-5,spark-6,spark-7,spark-8
local_port = 18090
layout = quads
```

**Rules that `vade` checks:**

- Every host is a configured machine, listed once.
- With `vade up`, a group must be an **arc** of the cabling in the listed
  order: each host is linked to the next. A link from the last host back to
  the first is allowed, not needed. A group of one needs no link. If the
  hosts follow the cables in another order, the error gives that order.
- In one layout, no two groups share a machine, and their local port ranges
  do not overlap. A group holds `local_port` and, for replicas, one port per
  machine after it.

**Use a group.** Set `VADE_GROUP` to the group name. It selects the group's
machines, its own ownership record (under `state_dir/groups/NAME`) and its
own local port. Use the same `VADE_GROUP` for every command of that server.
Like `VADE_CONFIG`, `VADE_GROUP` selects the controller, even when this
computer also has a local installation. While any group holds an ownership
record, `vade launch` and `vade run qwen3-8b` refuse to create local defaults
from the default controller config.

With `VADE_GROUP` set, `--sparks` is refused, and `--hosts` must repeat the
group's own list: the group names the machines. `--replicas N` starts one
replica on each machine of the group, so N must equal the group size. `up`,
`down`, `install` and `update` share one lock, so start the groups one after
the other.

**Install and update with running groups.** `vade install` and `vade update`
stage the new build on every machine in the config. They treat the engines
of every group's ownership record as owned by this controller, not as outside
jobs, so a running group does not stop the install. A running group keeps
its old build. `vade update` offers to restart only the group that
`VADE_GROUP` selects. Restart each other running group yourself:
`VADE_GROUP=NAME vade down`, then `VADE_GROUP=NAME vade up`. When you decline
the restart, the printed restart commands carry the `VADE_GROUP=NAME` prefix.

`vade config` shows the selected group. The launch plan shows it as its first
row. Without `VADE_GROUP`, `vade status` and `vade down` name the groups that
still hold an ownership record, and `vade status` says `Vade is stopped.` only
when no group holds one. `vade status --json` lists them in a `groups` array.
Vade finds the records under `state_dir/groups/`, so a group that you renamed
or removed while its server runs still counts: install treats its engines as
owned, and local defaults are refused. Put its `[group NAME]` section back,
then run `VADE_GROUP=NAME vade down`. Stop every group before you rename or
remove it.

<br>

---

## 5. Check the ring

```sh
VADE_CONFIG=.local/eight.conf vade doctor --fabric --ring --dry-run
VADE_CONFIG=.local/eight.conf vade doctor --fabric --ring
```

`vade doctor --fabric` reads every machine in the config. It makes three SSH
calls per Spark (interfaces, pings, clock) and changes nothing. `--ring` adds
the `ring` and `ring line rate` rows. Without `--ring`, a config of any shape
(a pair, a path, a ring of six) gets the other rows only, as before. It prints
one row per check:

| Row | It passes when |
| --- | --- |
| `cabling` | Always passes. Shows each connected part of the config: `ring of 8: …`, `path of 8: …`, `pair: …` or `no link: …`. |
| `ring` (with `--ring`) | Every machine in the config has exactly two linked neighbors, and the links close into one ring that holds every machine. A missing cable 8 gives `The ring does not close: path of 8: …` and names the two ends to cable. |
| `group NAME` | The group's hosts are an arc of the cabling in the listed order: `arc of 4 in rank order: …`, with `the ring closes` when the last host is also cabled to the first. |
| `<spark> port to <peer>` | An interface on the Spark holds its `fabric.<peer>` address, links at 100,000 Mb/s or more, has MTU 9000 or more, and has an RDMA device in state `ACTIVE`. With `SEQPAR_RDMA=0` (TCP), the RDMA device is not required. |
| `<a> - <b> cable` | Both ends name each other in the config. Rail 0 is the pair of interfaces with the `fabric.<peer>` addresses, in one subnet. Rail 1 is a second subnet on two other fabric interfaces, one at each end; an address alias on the rail 0 interface does not count. **Each rail on its own**: the two ends hold different addresses, both ends link at 100,000 Mb/s or more with the same MTU of 9000 or more and an `ACTIVE` RDMA device, and each end answers a ping and a 9000-byte ping that must not be split (`ping -M do -s 8972`) from the other end. A ping to an address that the pinging Spark holds itself fails: it never leaves the Spark. No other cable uses either subnet. The cable fails when either rail fails. |
| `<spark> clock` | The Spark's clock is within 1,000 ms of the controller's clock, beyond the SSH round trip, and `timedatectl` reports time sync on. |
| `ring line rate` (with `--ring`) | Rail 0 of every cable links at the same speed as the fastest rail 0 in the ring, and rail 1 the same. A cable that links slower on one rail fails this row and names the cable, the rail, the Spark and the interface. Every cable must have both rails to compare. |

**No bandwidth number.** `vade doctor --fabric` checks link state, speed,
MTU, the RDMA device and pings. It does not move data at full rate, so a
cable can pass every row and still be slow. Run the bandwidth test on every
new or moved cable ([step 3 of section 7](#7-saturday-step-by-step)).

Our links report `200000Mb/s` in `ethtool`
([network-setup.md](network-setup.md#1-what-to-buy)). The rate row compares
the cables with each other. It does not assume that number, because the speed
that each rail interface reports is to be confirmed on the eight Sparks.

The `ring` row and the cable rows together give the ring check: every Spark
sees exactly its two neighbors, both of its cables pass on both rails, and the
whole ring closes. An address on the wrong port, a swapped `fabric.<peer>`
value, a link with only one end in the config, a missing or faulty rail 1, a
rail 1 whose link is down, the same address at both ends of a rail, or an MTU
of 1500 at one end each fail one named row. Each failed row gives the next
step. A host name in a `fabric.<peer>` value is skipped; this check reads
numeric IPv4 addresses only. When the SSH probe of a Spark fails, its
`fabric ports` row fails, and its port and cable rows are skipped until that
row passes.

Then check each group as `vade up` will start it:

```sh
VADE_GROUP=pair-4 vade doctor
VADE_GROUP=eight vade doctor
```

That runs the normal checks (SSH, GPU, disk, memory, model files, ports) on
the group's Sparks, and pings each link of the group.

<br>

---

## 6. Start a run on an arc

Each recipe gives the hosts in rank order. Use the group form when the arc is
a group in your config, and the `--hosts` form for any other arc. With
`--hosts` and no `VADE_GROUP`, the run uses the default ownership record and
`local_port`, so start one such run at a time.

**Two Sparks (world 2).** FP4-Local runs the tensor split.

```sh
VADE_GROUP=pair-4 vade up                  # spark-7, spark-8 (cable 7)
vade up --hosts spark-7,spark-8            # the same arc, without a group
vade up --hosts spark-8,spark-1            # the pair on cable 8
```

**Four Sparks (world 4).** FP4-Local runs the layer chain of four. FP8 runs
four-way TP with `--fp8` and a config that selects the FP8 checkpoint
(`vade up --fp8` refuses an FP4-Local config). Each group below is its own
server on its own local port.

```sh
VADE_GROUP=quad-a vade up                                    # spark-1 to spark-4
VADE_GROUP=quad-b vade up                                    # spark-5 to spark-8, at the same time
vade up --hosts spark-3,spark-4,spark-5,spark-6              # an arc across cable 4
vade up --hosts spark-7,spark-8,spark-1,spark-2              # an arc across cable 8
vade up --fp8 --hosts spark-1,spark-2,spark-3,spark-4        # FP8, with an FP8 config
```

FP8 on four Sparks gives other token IDs than FP8 on two Sparks today
([benchmarks.md](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).
Compare FP8 token IDs only between runs on the same number of Sparks.

**Eight Sparks (world 8).** FP4-Local runs the layer chain of eight. FP8 is
refused.

```sh
VADE_GROUP=eight vade up
vade up --sparks 8          # the first eight machines in config order
```

`--sparks N` takes the first N machines in config order, so keep the
`[machine]` sections in cable order (spark-1 to spark-8). Then `--sparks 2`,
`--sparks 4` and `--sparks 8` are arcs that start at spark-1.

Stop a run with the same selection: `VADE_GROUP=NAME vade down`, or plain
`vade down` for a `--hosts` or `--sparks` run.

<br>

---

## 7. Saturday, step by step

Do the steps in order. Do not start a step until the one before it passes.
Write down the time and the result of each step.

**Before the new Sparks arrive**

- [ ] Print the [cable plan](#2-cable-plan) and the
      [address plan](#3-address-plan). Label eight cables 1 to 8, both ends,
      and the eight Sparks spark-1 to spark-8.
- [ ] Write the private mapping: which machine is each label, and the real
      address of each row in [section 3](#3-address-plan), from your own
      private range.
- [ ] On every Spark, check that the private range is free:
      `ip -4 route show table all`.
- [ ] Write the private controller config from
      [cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf).
      Run `vade doctor --fabric --ring --dry-run` with it: no row fails.

**1. Unbox and check, before any QSFP cable**

- [ ] Place spark-7 and spark-8 as in [section 1](#1-the-ring-and-its-arcs).
      Connect power and the management LAN only. Keep a local console ready.
- [ ] Finish the OS first boot. Do not update the firmware yet: first write
      down the versions below, then decide.
- [ ] On spark-7, spark-8 and one of the other six, read these values. None
      of the commands needs `sudo` or changes a setting. Each value on a new
      Spark must be the same as on the other Spark. When one is different,
      decide before you cable: match it, or write down why you accept it.

      | What | Command |
      | --- | --- |
      | Memory | `grep MemTotal /proc/meminfo` |
      | Board, BIOS, embedded controller | `cat /sys/class/dmi/id/{product_name,bios_version,bios_date,ec_firmware_release}` (some boards have no `ec_firmware_release`) |
      | Kernel and boot options | `uname -r`; `cat /proc/cmdline` |
      | GPU driver and CUDA | `nvidia-smi` (first lines) |
      | GPU clock, power limit, PCIe link | `nvidia-smi --query-gpu=clocks.max.graphics,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv` |
      | ConnectX-7 firmware | `cat /sys/class/infiniband/*/fw_ver` (four devices, one version) |
      | ConnectX-7 PCIe links | `cat /sys/class/infiniband/*/device/current_link_speed /sys/class/infiniband/*/device/current_link_width` (the same as `max_link_speed` and `max_link_width`) |
      | Wi-Fi off | `nmcli radio wifi` shows `disabled` |

- [ ] Match the kernel and the driver of the other six. On kernel
      `7.0.0-1019-nvidia`, apply the `kho=off` step
      ([spark-setup.md, step 5](spark-setup.md#5-check-the-kernel-before-registering-memory)).

**2. Cable**

- [ ] Stop every server on the six Sparks. Plain `vade status` shows
      `Vade is stopped.` and names no group. Never move a cable that a running
      rank uses.
- [ ] Find and tag the cable 6 end on spark-1
      ([from a ring of six](#2-cable-plan)).
- [ ] Move and add cables as in [from a ring of six](#2-cable-plan): move the
      cable 6 end to spark-7 port 1, add cables 7 and 8.
- [ ] On each Spark, `ibdev2netdev` shows both ports `(Up)`, and `ethtool`
      shows the same speed on every fabric interface as on the other six.

**3. Address**

Make network changes from the local console of the Spark, not over SSH on a
fabric port.

- [ ] On spark-7 and spark-8, run the OS setup steps 1 to 4 of
      [spark-setup.md](spark-setup.md) with the `ring` template and their
      addresses from [section 3](#3-address-plan).
- [ ] On spark-1, change the port 1 addresses only (cable 8).
- [ ] Run the bandwidth test on the moved and new cables (6, 7, 8), on each
      rail ([network-setup.md, step 7](network-setup.md#7-check-every-link)).
      The test passes on any positive value, so read the number. The
      [numbers we measured](network-setup.md#numbers-we-measured) on earlier
      cables: about 109 Gb/s one way on one rail, and 196.04 Gb/s one way
      with both rails at MTU 9000. A rail far below that has a fault: reseat
      the cable at both ends, then test again.
- [ ] Copy the checkpoint to spark-7 and spark-8 at the same absolute path
      as on the other Sparks
      ([INSTALL.md, step 3](../INSTALL.md#step-3-get-the-model)). It is
      102.26 GB. Copy it from a neighbor over a tested fabric cable, not over
      a slow LAN. Then check that it has the same files and sizes as on the
      neighbor.

**4. Ring check**

- [ ] `vade doctor --fabric --ring`: every row passes, `cabling` and `ring`
      show `ring of 8: spark-1, …, spark-8`, and `ring line rate` passes.
- [ ] `vade install`, so that every Spark has the same staged build.
- [ ] `VADE_GROUP=pair-4 vade doctor` and `VADE_GROUP=eight vade doctor`.

**5. First two-Spark run: the new pair**

- [ ] `VADE_GROUP=pair-4 vade up` (spark-7 and spark-8 on cable 7).
- [ ] `VADE_GROUP=pair-4 vade verify --prompt 8k --receipt verify-pair-4.json`
      prints `IDENTICAL`.
- [ ] Run the FP4-Local two-Spark card on the new pair, then
      `VADE_GROUP=pair-4 vade down`.
- [ ] Do the same `up`, `verify` (with `--receipt verify-pair-1.json`) and
      `down` on one existing pair, for example `VADE_GROUP=pair-1`. The two
      receipts must have the same `reference_ids_sha256` value: the same
      request gives the same token IDs on the new pair.

**6. First eight-Spark run**

- [ ] `VADE_GROUP=eight vade up`. The launch plan shows group `eight` and eight
      machines.
- [ ] `VADE_GROUP=eight vade verify --prompt 8k --receipt verify-eight.json`
      prints `IDENTICAL`, with the same `reference_ids_sha256` value as the
      pair receipts.
- [ ] Run the FP4-Local eight-Spark card (layer chain). FP8 on eight Sparks
      is refused by this build.
- [ ] `VADE_GROUP=eight vade down`.

**7. Arcs of four and layouts**

- [ ] `quad-a`, then `quad-b`: the FP4-Local four-Spark card on each, and the
      FP8 four-Spark card with an FP8 config. Compare the FP8 token IDs with
      other four-Spark FP8 runs only.
- [ ] The FP4-Local layouts at equal total load: 8 singles
      (`vade up --replicas 8`, one local port), 4 doubles (`doubles`, then
      `doubles-b`: four servers on four local ports), 2 quads (`quads`: two
      servers on two ports). With groups, the load client sends its chats to
      every group's port itself. Record the same cells for each layout, and
      check that the token IDs are the same across layouts for the same
      request.

**Where the numbers land.** Each card records the revision, the hardware, the
checkpoint, the settings, the arc (the hosts in rank order) and the raw
results. The eight-Spark and four-of-eight columns are defined in
[benchmarks.md](benchmarks.md#columns-for-eight-sparks). Published numbers go
there, with a small sanitized receipt in [bench/receipts](../bench/receipts/README.md),
after review. Until then, every value for eight Sparks is to be measured. An
arc or layout that we have not tested still needs its own exactness check on
real Sparks before you rely on it.

<br>

---

## 8. What vade prints for eight Sparks

These are the output lines that the ring of eight adds or changes. Nothing
changes for a config without groups or without `--ring`.

| Command | Line or row | When |
| --- | --- | --- |
| `vade doctor --fabric` | `group NAME`: `arc of 4 in rank order: spark-1, spark-2, spark-3, spark-4` | A group that is an arc. `; the ring closes` follows when the last host is cabled to the first. |
| `vade doctor --fabric` | `group NAME`: `Group NAME is not an arc of the cabling: A and B have no fabric link in the config.` | Two neighbors in the list are not cabled. The next step gives the cable order when one exists. |
| `vade doctor --fabric --ring` | `ring`: `ring of 8: spark-1, …, spark-8; every Spark has two neighbors` | The whole config is one closed ring. |
| `vade doctor --fabric --ring` | `ring`: `The ring does not close: path of 8: …` | One cable is missing from the config. |
| `vade doctor --fabric --ring` | `ring`: `The cabling is 2 rings, not one: …` or `A ring needs exactly two neighbors per Spark in the config: …` | Two rings, or a Spark with one, three or no neighbors. |
| `vade doctor --fabric --ring` | `ring line rate`: `rail 0 at … Mb/s and rail 1 at … Mb/s on all 8 cables` | Every rail links at the ring's rate. |
| `vade up` with `VADE_GROUP` | `Group NAME is not an arc of the cabling: …` | The group's hosts do not follow the cables. Nothing starts. |
| `vade config` | `Group`: `NAME (VADE_GROUP): spark-1, spark-2, …` | A group is selected. |

<br>

**Back:** [Network setup](network-setup.md) · **See also:** [Four Sparks](four-sparks.md) · [FP4-Local launch](../cli/docs/fp4-local.md) · [Benchmarks](benchmarks.md#columns-for-eight-sparks)
