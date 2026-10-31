# Eight Sparks: one ring of eight

**How to cable eight Sparks (GB10 machines) into one ring without a switch,
give every link an address, check the ring and start the first two-Spark and
eight-Spark runs. [Section 7](#7-day-one-at-the-rack) is the check list for
the day the new Sparks arrive.**

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
> - On eight Sparks, FP4-Local runs as one server (the layer chain of eight),
>   as eight one-Spark servers behind one address (`vade up --replicas 8`),
>   or as four pairs behind one address (`vade up --replicas 4 --pairs`).
>   FP8 does not run on eight Sparks.

<br>

## On this page

1. [The ring and its arcs](#1-the-ring-and-its-arcs)
2. [Cable plan](#2-cable-plan)
3. [Address plan](#3-address-plan)
4. [Controller config: machines and groups](#4-controller-config-machines-and-groups)
5. [Check the ring](#5-check-the-ring)
6. [Start a run on an arc](#6-start-a-run-on-an-arc)
7. [Day one at the rack](#7-day-one-at-the-rack)
8. [What to measure first](#8-what-to-measure-first)
9. [What vade prints for eight Sparks](#9-what-vade-prints-for-eight-sparks)

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
| 2 | The tensor split, the only two-Spark path. It needs RDMA. `--replicas 4 --pairs` runs four pairs behind one local gateway. | Two-way TP. |
| 4 | Over RDMA, the tensor split in parts (each Spark works on every token). Over TCP, the layer chain of four. | Four-way TP on a path. Today it gives other token IDs than two Sparks ([benchmarks.md](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)). |
| 8 | The layer chain of eight. | Refused: `FP8 needs 2, 4, or 6 machines` ([cli/src/config.rs](../cli/src/config.rs)). |

FP4-Local runs the tensor split on two Sparks, and the tensor split in parts on
four Sparks over RDMA. On four Sparks over TCP and on eight Sparks it runs the
layer chain: each Spark holds a block of layers. Eight Sparks do not run the
split: the mailbox has no room for eight ranks. The split in parts and the
chain have no prefix cache. The model has 48 layers, so on eight Sparks each
Spark runs 6 of them, and each token passes through all eight Sparks in turn
(`Stage::new` in
[engine/crates/engine/src/flash/next_pp.rs](../engine/crates/engine/src/flash/next_pp.rs)).
No card has measured the layer chain of eight yet, so this page gives no speed
for it.

**One address for several servers.** Only `vade up --replicas N` puts one
local gateway in front of several engines. Each engine runs on one Spark, or,
with `--pairs`, on a tensor-split pair. Without `--replicas`, a group is one
server on its own `local_port`. No `--replicas` mode puts four Sparks or more
in one server. So two quads are two servers on two ports, and a client must
send its chats to each port itself. See
[section 6](#several-servers-behind-one-address) for the commands.

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
replica on each machine of the group, so N must equal the group size. With
`--pairs`, `--replicas N` starts one pair on each two machines of the group,
so 2N must equal the group size. `up`, `down`, `install` and `update` share
one lock, so start the groups one after the other.

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
new or moved cable ([step 5 of section 7](#step-5-measure-each-moved-and-new-cable)).

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

**Four Sparks (world 4).** FP4-Local runs the four-way tensor split over RDMA
(each Spark works on every token, on a quarter of the experts; one-chat passes
also split most projections, and greedy steps split the output head) and the
layer chain of four over TCP. FP8 runs four-way TP with `--fp8` and a config that selects the FP8 checkpoint
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

### Several servers behind one address

`vade up --replicas N` starts N FP4-Local servers and one local gateway. It
does not take `--fp8`. The gateway listens on `local_port`, so clients use one
address. Each server gets its own tunnel on `local_port + 1` to
`local_port + N`. Each reply comes from one server. The gateway never joins
output from two servers. The rule that picks the server is in
[four-sparks.md](four-sparks.md#two-pairs-behind-one-address).

**Eight singles.** Eight one-Spark servers. They do not use the fabric, so a
cable fault does not stop them.

```sh
vade up --replicas 8 --dry-run
vade up --replicas 8                  # the first eight machines in config order
VADE_GROUP=eight vade up --replicas 8 # the same eight, as the group eight
```

**Four pairs.** Four tensor-split servers, two Sparks each. The first and
second machine form pair 1, the third and fourth pair 2, and so on. Each pair
must have its own cable: `vade up` checks that each pair is an arc and stops
before it starts anything when one is not. With the machines in cable order,
the pairs are spark-1 and spark-2 (cable 1), spark-3 and spark-4 (cable 3),
spark-5 and spark-6 (cable 5), and spark-7 and spark-8 (cable 7), the same
pairs as the `doubles` layout.

```sh
vade up --replicas 4 --pairs --dry-run
vade up --replicas 4 --pairs
VADE_GROUP=eight vade up --replicas 4 --pairs
vade up --replicas 4 --pairs --hosts spark-2,spark-3,spark-4,spark-5,spark-6,spark-7,spark-8,spark-1
```

The last line uses the other four cables (2, 4, 6 and 8), the same pairs as
the `doubles-b` layout.

With `VADE_GROUP`, N must equal the group size, or with `--pairs`, half of
it. `VADE_GROUP=quad-a vade up --replicas 2 --pairs` puts two pairs of
`quad-a` behind one address. `--replicas` takes 2 to 16 servers, and with
`--pairs` 2 to 12 pairs.

**Not in this build:** a server of four or eight Sparks behind a shared
address with other servers. `--pairs` is the only size above one Spark. Two
quads run as two servers, each on its own port:
`VADE_GROUP=quad-a vade up`, then `VADE_GROUP=quad-b vade up`.

Stop a replica run as you started it: `vade down`, or `VADE_GROUP=NAME vade
down`.

<br>

---

## 7. Day one at the rack

This is the check list for the day the new Sparks arrive. It assumes that six
Sparks already run as one ring (spark-1 to spark-6, cabled as in
[section 2](#2-cable-plan)) and that spark-7 and spark-8 are new. With eight
new Sparks, do step 1 on all of them, then cable all eight as in
[section 2](#2-cable-plan).

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

### Step 1. Check each new Spark, before any QSFP cable

- [ ] Place spark-7 and spark-8 as in [section 1](#1-the-ring-and-its-arcs).
      Connect power and the management LAN only. Keep a local console ready.
- [ ] Finish the OS first boot. Do not update the firmware yet: first write
      down the values below, then decide.
- [ ] Choose one of the six Sparks as the reference. On spark-7, spark-8 and
      the reference, read these values. None of the commands needs `sudo` or
      changes a setting, and the reference can stay busy.

      | What | Command | On a new Spark |
      | --- | --- | --- |
      | Memory | `grep MemTotal /proc/meminfo` | The same as the reference |
      | GPU name and driver | `nvidia-smi --query-gpu=name,driver_version --format=csv` | The same |
      | CUDA | `nvidia-smi` (the `CUDA Version` in the first lines); `nvcc --version` (often `/usr/local/cuda/bin/nvcc`) | The same |
      | OS release | `cat /etc/os-release` | The same |
      | Kernel and boot options | `uname -r`; `cat /proc/cmdline` | The same |
      | Board, BIOS, embedded controller | `cat /sys/class/dmi/id/{product_name,bios_version,bios_date,ec_firmware_release}` (some boards have no `ec_firmware_release`) | The same, or a difference that you accept in writing |
      | GPU clock, power limit, PCIe link | `nvidia-smi --query-gpu=clocks.max.graphics,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv` | The same |
      | GPU temperature at idle | `nvidia-smi --query-gpu=temperature.gpu --format=csv` | Write it down. Row 10 of [section 8](#8-what-to-measure-first) compares with it |
      | RDMA devices | `ls /sys/class/infiniband` | Four devices |
      | ConnectX-7 firmware | `cat /sys/class/infiniband/*/fw_ver` | One version on all four devices, the same as the reference |
      | ConnectX-7 PCIe links | `cat /sys/class/infiniband/*/device/current_link_speed /sys/class/infiniband/*/device/current_link_width` | The same as `max_link_speed` and `max_link_width`, on all four devices |
      | NCCL, only if you use `nccl-tests` | `ls /usr/lib/aarch64-linux-gnu/libnccl.so.*` | The same. Vade does not use NCCL |
      | Wi-Fi | `nmcli radio wifi` | `disabled` |

- [ ] **Stop rule.** A new Spark with less memory, a lower GPU clock or
      power limit, or a PCIe link below its maximum does not join the ring.
      Solve that first. For any other difference, decide before you cable:
      match the six, or write down why you accept it. Do not update firmware
      to clear a difference before you have written down both values.
- [ ] Match the kernel and the driver of the other six. On kernel
      `7.0.0-1019-nvidia`, apply the `kho=off` step
      ([spark-setup.md, step 5](spark-setup.md#5-check-the-kernel-before-registering-memory)).

### Step 2. Find the cable 6 end, before any cable moves

Only one cable end moves: the cable 6 end on spark-1. Find it by its
address, not by the label on the case.

- [ ] Stop every server on the six Sparks. Plain `vade status` shows
      `Vade is stopped.` and names no group. Never move a cable that a running
      rank uses.
- [ ] On spark-1, from its local console, run `ibdev2netdev` and
      `ip -4 -br addr`. Write down the interface that holds the cable 6 rail 0
      address and the interface that holds the cable 1 rail 0 address.
- [ ] `sudo ethtool -p <cable 6 interface> 10`. The port light that blinks is
      the cable 6 end. Put a tag on that cable end.
- [ ] `sudo ethtool -p <cable 1 interface> 10`. The other port must blink. If
      the same port blinks, stop: the private mapping is wrong.

Some ports do not blink. Then use step 3 to check the end you unplug.

### Step 3. Cable, in this order

After each cable, check both of its ends with `ibdev2netdev` (the port shows
`(Up)`) and `ethtool <interface>` (the same `Speed` as the fabric interfaces
of the other six).

| Order | Cable | What to do | Check |
| --- | --- | --- | --- |
| 1 | 6 | Unplug the tagged end from spark-1. Plug it into spark-7 port 1. | When you unplug: on spark-1, `ibdev2netdev` shows the cable 6 port `(Down)` and the cable 1 port still `(Up)`. If cable 1 went down, put that end back in the same port and go back to step 2. When you plug in: spark-6 and spark-7 |
| 2 | 7 | New cable: spark-7 port 0 to spark-8 port 1. | spark-7 and spark-8 |
| 3 | 8 | New cable: spark-8 port 0 to the spark-1 port that cable 6 left. | spark-8 and spark-1 |

Cables 1 to 5 do not move. If the port that cable 6 left on spark-1 is not
port 1, write down the real port. Its interfaces get the cable 8 addresses.
A port that links at `100000Mb/s` most likely has a 100 Gb/s cable
([network-setup.md, troubleshooting](network-setup.md#9-troubleshooting)).

### Step 4. Give the new cable ends their addresses

Make network changes from the local console of the Spark, not over SSH on a
fabric port.

- [ ] On spark-7 and spark-8, run the OS setup steps 1 to 7 of
      [spark-setup.md](spark-setup.md) with the `ring` template and their
      addresses from [section 3](#3-address-plan). Steps 4 to 7 (memlock,
      kernel, SSH between the Sparks, paused updates) are needed for the rail
      check below.
- [ ] On spark-1, change only the port that cable 6 left: it gets the cable 8
      addresses, and its peer is now spark-8.
- [ ] On spark-6, the cable 6 addresses stay. Only its peer on that cable
      changes, from spark-1 to spark-7
      ([spark-setup.md, name each peer rail](spark-setup.md#name-each-peer-rail)).
- [ ] Give spark-6 and spark-1 SSH to their new peers. spark-setup step 6 on
      spark-7 and spark-8 copies only their own keys, so it does not cover
      this direction.
  - On spark-6: the addresses `192.0.2.82` and `192.0.2.90` now belong to
    spark-7, but spark-6 still holds spark-1's host key for them. The rail
    check connects with strict host-key checking, so its SSH row fails, and
    the `FAIL` text does not say that the key changed. On spark-7's local
    console, read its host key fingerprint
    (`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`). Then, on spark-6,
    remove the old entries for these two addresses only
    (`ssh-keygen -R 192.0.2.82` and `ssh-keygen -R 192.0.2.90`, as the
    service account), and run
    [spark-setup.md, step 6](spark-setup.md#6-establish-ssh-between-the-sparks).
    Accept the new key only if it matches what spark-7's console showed.
  - On spark-1: the cable 8 addresses are new, so there is no old entry to
    remove. Compare spark-8's fingerprint on spark-8's console, then run
    spark-setup step 6 on spark-1.
  - Step 6 loops over every peer in the Spark's config. For the peer that did
    not change, `ssh-copy-id` finds the key already there.
- [ ] On spark-1 and spark-6, if updates were restored after the last
      benchmark window, pause them again
      ([spark-setup.md, step 7](spark-setup.md#7-pause-updates-only-for-the-benchmark-window)).
      If you skip this, the rail check fails its five update rows.
- [ ] On spark-1, spark-6, spark-7 and spark-8, run the rail check of
      [spark-setup.md, step 8](spark-setup.md#8-verify-every-rail-and-the-actual-launch-session)
      with `--apply`. It passes when its last line shows `INFO failures=0`.
      The rail rows then print `PASS`: address, MTU 9000, carrier, speed,
      RDMA `ACTIVE`, the RoCE v2 GID that matches the IPv4 address, the
      9000-byte ping and SSH to the peer. Each rail also prints one
      `UNVERIFIED` row, because this check does not run `ib_write_bw`. Step 5
      does. A `FAIL` row names the setting to fix: on a kernel other than
      `7.0.0-1019-nvidia`, for example, set `KERNEL_FIXED` as
      [spark-setup.md, step 5](spark-setup.md#5-check-the-kernel-before-registering-memory)
      says.
- [ ] On every Spark, `ip -4 -br addr` shows four fabric addresses, each
      with `/24`. Across the eight Sparks, each of the 16 subnets is on
      exactly two Sparks.

### Step 5. Measure each moved and new cable

`vade doctor` and the rail check do not move data at full rate. A cable can
pass both and still be slow, so measure it.

- [ ] Run the bandwidth test
      ([network-setup.md, step 7](network-setup.md#7-check-every-link)) on
      each rail of cables 6, 7 and 8, in both directions: swap the server and
      the client ends.
- [ ] In the same sitting, run the same test on one rail of a cable that did
      not move, for example cable 1. That value is your reference for the day.
- [ ] Read the numbers. The test passes on any positive value. The
      [numbers we measured](network-setup.md#numbers-we-measured) on earlier
      cables are about 109 Gb/s one way on one rail and 196.04 Gb/s one way
      on both rails at MTU 9000. For a both-rails value, start the rail 0 and
      rail 1 tests at the same time and add the two results.
- [ ] A rail clearly below the reference cable has a fault. Reseat the cable
      at both ends and test again. If it stays low, try another cable.

### Step 6. Check the ring

```sh
vade doctor --fabric --ring
```

It passes when every row passes:

- `cabling` and `ring` show `ring of 8: spark-1, spark-2, spark-3, spark-4,
  spark-5, spark-6, spark-7, spark-8`, and `ring` adds `every Spark has two
  neighbors`.
- Each `<a> - <b> cable` row passes on both rails. Each port links at
  100,000 Mb/s or more, has MTU 9000 and an `ACTIVE` RDMA device, and
  answers a small and a 9000-byte ping from the other end.
- Each `<spark> clock` row is within 1,000 ms of the controller.
- `ring line rate` shows `rail 0 at … Mb/s and rail 1 at … Mb/s on all 8
  cables`.

[Section 5](#5-check-the-ring) explains every row and each failure.

### Step 7. Build and model files

- [ ] Copy the checkpoint to spark-7 and spark-8 at the same absolute path
      as on the other Sparks
      ([INSTALL.md, step 3](../INSTALL.md#step-3-get-the-model)). It is
      102.26 GB. Copy it from a neighbor over a measured fabric cable, not over
      a slow LAN. Then check that it has the same files and sizes as on the
      neighbor.
- [ ] `vade install`, so that every Spark has the same staged build. It
      reports an error when a Spark does not have the model yet, so copy the
      model first.
- [ ] `VADE_GROUP=pair-4 vade doctor` and `VADE_GROUP=eight vade doctor`.
      They run the normal checks (SSH, GPU, disk, memory, model files, ports)
      on the group's Sparks.

### Step 8. The new pair gives the same token IDs

- [ ] `VADE_GROUP=pair-4 vade up` (spark-7 and spark-8 on cable 7).
- [ ] `VADE_GROUP=pair-4 vade verify --prompt 8k --receipt verify-pair-4.json`
      prints `IDENTICAL`. Then `VADE_GROUP=pair-4 vade down`.
- [ ] Do the same `up`, `verify` (with `--receipt verify-pair-1.json`) and
      `down` on one existing pair, for example `VADE_GROUP=pair-1`. The two
      receipts must have the same `reference_ids_sha256` value: the same
      request gives the same token IDs on the new pair.

### Step 9. Serve on all eight

Two layouts put several servers on the eight Sparks behind one address.
Start one at a time ([section 6](#several-servers-behind-one-address)).

```sh
vade up --replicas 8
vade verify --prompt 8k --receipt verify-singles.json
vade down

vade up --replicas 4 --pairs
vade verify --prompt 8k --receipt verify-pairs.json
vade down
```

`vade verify` with no `--url` goes through the gateway on `local_port`. To
check one server alone, give its tunnel:
`vade verify --prompt 8k --url http://127.0.0.1:18081 --receipt verify-server-1.json`.
Every receipt must have the same `reference_ids_sha256` value as the pair
receipts of step 8.

Eight singles do not use the fabric, so a cable fault does not stop them.
Four pairs each use one cable. Which layout gives more tokens per second on
eight Sparks is not measured yet ([section 8](#8-what-to-measure-first)).

### Step 10. Eight Sparks as one server

- [ ] `VADE_GROUP=eight vade up`. The launch plan shows group `eight` and eight
      machines. After the start, `VADE_GROUP=eight vade logs` shows one
      `flashnext pp stage rank=<r> world=8 layers=<first>..<end>` line per
      Spark, 6 layers each.
- [ ] `VADE_GROUP=eight vade verify --prompt 8k --receipt verify-eight.json`
      prints `IDENTICAL`, with the same `reference_ids_sha256` value as the
      pair receipts.
- [ ] `VADE_GROUP=eight vade down`.

If the start fails with `pipeline peer read timed out after 10000 ms`, one
Spark finished loading the model much later than the Spark before it. Each
rank waits 10 seconds for the next rank
(`LINK_TIMEOUT` in
[engine/crates/engine/src/flash/next_pp.rs](../engine/crates/engine/src/flash/next_pp.rs)).
Look in `VADE_GROUP=eight vade logs` for the rank that finished loading last,
then start again.

FP8 on eight Sparks is refused by this build.

### Step 11. Arcs of four

- [ ] `VADE_GROUP=quad-a vade up`, then `VADE_GROUP=quad-b vade up`: two
      servers on two local ports. Check each with `vade verify` and the same
      `VADE_GROUP`.
- [ ] FP8 on four Sparks needs an FP8 config. Compare its token IDs only
      with other four-Spark FP8 runs.

### When you are done

- [ ] Stop every server: `vade down` with the same `VADE_GROUP` as each
      `vade up`. Plain `vade status` then shows `Vade is stopped.` and names no
      group. Then restore updates on every Spark where you paused them,
      spark-7 and spark-8 included
      ([spark-setup.md, step 9](spark-setup.md#9-finish-benchmarking-and-restore-updates)).

<br>

---

## 8. What to measure first

Measure in this order. Each row names the record that the value goes into.
Each card records the revision, the hardware, the checkpoint, the settings,
the arc or layout (the labels in rank order) and the raw results. **No value
for eight Sparks is measured yet**, so this page gives no expected number.
An arc or layout that has not been tested still needs its own exactness check
on real Sparks before you rely on it.

| Order | What | How | Record |
| --- | --- | --- | --- |
| 1 | The values of each new Spark and the reference | [Step 1](#step-1-check-each-new-spark-before-any-qsfp-cable) | Your private notes, with the date |
| 2 | Bandwidth of each rail of cables 6, 7 and 8, both directions, and of one cable that did not move | [Step 5](#step-5-measure-each-moved-and-new-cable) | Your private notes: the raw `ib_write_bw` output |
| 3 | The same token IDs on the new pair and an old pair | [Step 8](#step-8-the-new-pair-gives-the-same-token-ids) | The two `vade verify` receipts |
| 4 | Speed of the new pair and an old pair, in the same sitting | The FP4-Local two-Spark card on `pair-4` and `pair-1` | One two-Spark card for each pair ([benchmarks.md](benchmarks.md)) |
| 5 | Eight singles behind one address: the card cells | `vade up --replicas 8`. The load client on `local_port` gives the total; the same client on each tunnel at the same time gives each server's value | A card for this layout: the total, and each server's value beside it |
| 6 | Four pairs behind one address: the same cells | `vade up --replicas 4 --pairs` | A card for this layout, as in row 5 |
| 7 | Eight Sparks as one server: `vade verify`, then the card | `VADE_GROUP=eight` | The **8 of 8** column ([benchmarks.md](benchmarks.md#columns-for-eight-sparks)) |
| 8 | Two quads: each alone, then both at the same time | `VADE_GROUP=quad-a`, `VADE_GROUP=quad-b` | The **4 of 8** and **2 × 4 of 8** columns |
| 9 | How long `vade status --json` takes during rows 5 and 7 | `time vade status --json`, with the same `VADE_GROUP` as the run | Your private notes. Its checks share a 15-second deadline |
| 10 | Heat while row 5 runs | `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv -l 1` on each Spark | Your private notes, beside the idle values of step 1 |

Rows 5 and 6 have no column in [benchmarks.md](benchmarks.md) yet. Through the
gateway, the client sees only the total. For each server's value, send the
same cells to each server's tunnel (`local_port + 1` to `local_port + N`) at
the same time, and record that run on its own. These are two different loads:
at 16 chats, the gateway run puts 16 chats on all servers together, and the
tunnel run puts 16 chats on each server. So the sum of the tunnel run is not
the gateway total. Label each run on the card, and do not put a gateway total
beside per-server values from the tunnel run. Within each run, follow the
rules of the **2 × 4 of 8** column: a multi-chat cell gives the sum of the
servers and each server's value beside it, and a one-chat cell gives each
server's value. Token IDs must be the same as in the one-, two-
and four-Spark columns for the same request. Published numbers go to
[benchmarks.md](benchmarks.md), with a small sanitized receipt in
[bench/receipts](../bench/receipts/README.md), after review.

<br>

---

## 9. What vade prints for eight Sparks

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
