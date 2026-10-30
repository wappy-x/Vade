# Eight Sparks: cabling, addresses and first checks

**How to cable eight Sparks (GB10 machines) without a switch, give every link an address,
check the fabric and run the first cards.**

Read [Hardware and network setup](network-setup.md) first. This page uses its
words (port, rail, ring, path) and its rules (MTU 9000, one subnet per
rail, no gateway). The OS steps are in [Spark system setup](spark-setup.md).

> **In 30 seconds**
>
> - A Spark has two QSFP ports. Without a switch, eight Sparks form rings.
> - **Option A, two rings of four** (recommended for day one). **Option B,
>   one ring of eight.** Six of the eight cables are the same in both.
> - To change from one option to the other, move two cable ends and change
>   the addresses on two ports. The [change](#6-change-between-the-two-options)
>   takes minutes.
> - The controller config describes the cables. `[group]` sections name the
>   servers that run at the same time. `vade doctor --fabric` checks every
>   cable, port and clock.

<br>

## On this page

1. [The two cabling options](#1-the-two-cabling-options)
2. [Cable plan](#2-cable-plan)
3. [Address plan](#3-address-plan)
4. [What each option gives](#4-what-each-option-gives)
5. [Recommendation for day one](#5-recommendation-for-day-one)
6. [Change between the two options](#6-change-between-the-two-options)
7. [Controller config: machines, groups and layouts](#7-controller-config-machines-groups-and-layouts)
8. [Check the fabric](#8-check-the-fabric)
9. [Day-one checklist](#9-day-one-checklist)

This page calls the Sparks **spark-1** to **spark-8**. These are labels for
this page, not host names. Put a label with the same name on each Spark and on
both ends of each cable before you start.

<br>

---

## 1. The two cabling options

Each Spark has two QSFP ports, port 0 and port 1. Each cable joins port 0 of
one Spark to port 1 of the next Spark. With no switch, each Spark has at most
two neighbors, so the cables form rings.

**Option A: two rings of four.** Eight cables. Ring 1 is spark-1 to spark-4.
Ring 2 is spark-5 to spark-8.

```text
   spark-1 ── 1 ── spark-2          spark-5 ── 5 ── spark-6
      │               │                │               │
      4               2                8               6
      │               │                │               │
   spark-4 ── 3 ── spark-3          spark-8 ── 7 ── spark-7
```

**Option B: one ring of eight.** Eight cables. Every Spark is in one ring.

```text
   spark-1 ── 1 ── spark-2 ── 2 ── spark-3 ── 3 ── spark-4
      │                                               │
      8                                               4
      │                                               │
   spark-8 ── 7 ── spark-7 ── 6 ── spark-6 ── 5 ── spark-5
```

Cables 1, 2, 3, 5, 6 and 7 are the same in both options. Only cables 4 and 8
go to a different Spark.

<br>

---

## 2. Cable plan

Follow one row at a time, with the cable in your hand. "Port 0" and "port 1"
are the names Linux uses, not the labels on the case: on at least one Spark
the port marked "P0" is the one Linux calls port 1. Run `ibdev2netdev` on each
Spark to see which port is which (see
[network-setup.md, step 2](network-setup.md#2-how-a-sparks-network-ports-work)).

### Option A: two rings of four

| Cable | From | To | Ring |
| --- | --- | --- | --- |
| 1 | spark-1 port 0 | spark-2 port 1 | 1 |
| 2 | spark-2 port 0 | spark-3 port 1 | 1 |
| 3 | spark-3 port 0 | spark-4 port 1 | 1 |
| 4 | spark-4 port 0 | **spark-1 port 1** | 1 |
| 5 | spark-5 port 0 | spark-6 port 1 | 2 |
| 6 | spark-6 port 0 | spark-7 port 1 | 2 |
| 7 | spark-7 port 0 | spark-8 port 1 | 2 |
| 8 | spark-8 port 0 | **spark-5 port 1** | 2 |

### Option B: one ring of eight

| Cable | From | To |
| --- | --- | --- |
| 1 | spark-1 port 0 | spark-2 port 1 |
| 2 | spark-2 port 0 | spark-3 port 1 |
| 3 | spark-3 port 0 | spark-4 port 1 |
| 4 | spark-4 port 0 | **spark-5 port 1** |
| 5 | spark-5 port 0 | spark-6 port 1 |
| 6 | spark-6 port 0 | spark-7 port 1 |
| 7 | spark-7 port 0 | spark-8 port 1 |
| 8 | spark-8 port 0 | **spark-1 port 1** |

After each cable, check that both ends show `(Up)` in `ibdev2netdev`.

<br>

---

## 3. Address plan

The addresses on this page come from `192.0.2.0/24`, a range reserved for
documentation, cut into one `/29` subnet per rail. Do not put them on your
Sparks as they are. Choose one free private range for the fabric, keep the
same layout, and write the real addresses only in your private configs.
Before you use the range, make sure that nothing else on any Spark uses it:
`ip -4 route show table all` must show no route into it that your plan does
not add, and it must not overlap your LAN, VPN or containers.

**The rule.** Cable *c* has two rails. Rail *r* (0 or 1) uses subnet
`192.0.2.(16(c-1)+8r)/29`. The port 0 end of the cable is the first host of
that subnet, the port 1 end the second: cable 1 rail 0 is `192.0.2.0/29` with
`.1` and `.2`, and cable 1 rail 1 is `192.0.2.8/29` with `.9` and `.10`. An
address belongs to its cable, so a cable keeps its addresses when you move it,
as long as its port 0 end stays.

### Subnets per cable

| Cable | Port 0 end, first host | Port 1 end, second host (A / B) | Rail 0 subnet | Rail 1 subnet |
| --- | --- | --- | --- | --- |
| 1 | spark-1 | spark-2 | `192.0.2.0/29` | `192.0.2.8/29` |
| 2 | spark-2 | spark-3 | `192.0.2.16/29` | `192.0.2.24/29` |
| 3 | spark-3 | spark-4 | `192.0.2.32/29` | `192.0.2.40/29` |
| 4 | spark-4 | spark-1 / spark-5 | `192.0.2.48/29` | `192.0.2.56/29` |
| 5 | spark-5 | spark-6 | `192.0.2.64/29` | `192.0.2.72/29` |
| 6 | spark-6 | spark-7 | `192.0.2.80/29` | `192.0.2.88/29` |
| 7 | spark-7 | spark-8 | `192.0.2.96/29` | `192.0.2.104/29` |
| 8 | spark-8 | spark-5 / spark-1 | `192.0.2.112/29` | `192.0.2.120/29` |

### Addresses per Spark, option A (two rings of four)

The interface names are the typical names from
[network-setup.md](network-setup.md#2-how-a-sparks-network-ports-work).
Confirm them on each Spark with `ibdev2netdev`.

| Spark | Port | Rail 0 (`enp1s0f*`) | Rail 1 (`enP2p1s0f*`) | Cable | Neighbor |
| --- | --- | --- | --- | --- | --- |
| spark-1 | 0 | `192.0.2.1/29` | `192.0.2.9/29` | 1 | spark-2 |
| spark-1 | 1 | `192.0.2.50/29` | `192.0.2.58/29` | 4 | spark-4 |
| spark-2 | 0 | `192.0.2.17/29` | `192.0.2.25/29` | 2 | spark-3 |
| spark-2 | 1 | `192.0.2.2/29` | `192.0.2.10/29` | 1 | spark-1 |
| spark-3 | 0 | `192.0.2.33/29` | `192.0.2.41/29` | 3 | spark-4 |
| spark-3 | 1 | `192.0.2.18/29` | `192.0.2.26/29` | 2 | spark-2 |
| spark-4 | 0 | `192.0.2.49/29` | `192.0.2.57/29` | 4 | spark-1 |
| spark-4 | 1 | `192.0.2.34/29` | `192.0.2.42/29` | 3 | spark-3 |
| spark-5 | 0 | `192.0.2.65/29` | `192.0.2.73/29` | 5 | spark-6 |
| spark-5 | 1 | `192.0.2.114/29` | `192.0.2.122/29` | 8 | spark-8 |
| spark-6 | 0 | `192.0.2.81/29` | `192.0.2.89/29` | 6 | spark-7 |
| spark-6 | 1 | `192.0.2.66/29` | `192.0.2.74/29` | 5 | spark-5 |
| spark-7 | 0 | `192.0.2.97/29` | `192.0.2.105/29` | 7 | spark-8 |
| spark-7 | 1 | `192.0.2.82/29` | `192.0.2.90/29` | 6 | spark-6 |
| spark-8 | 0 | `192.0.2.113/29` | `192.0.2.121/29` | 8 | spark-5 |
| spark-8 | 1 | `192.0.2.98/29` | `192.0.2.106/29` | 7 | spark-7 |

### Addresses per Spark, option B (one ring of eight)

All rows are the same as option A except these four:

| Spark | Port | Rail 0 | Rail 1 | Cable | Neighbor |
| --- | --- | --- | --- | --- | --- |
| spark-1 | 1 | `192.0.2.114/29` | `192.0.2.122/29` | 8 | spark-8 |
| spark-4 | 0 | `192.0.2.49/29` (same) | `192.0.2.57/29` (same) | 4 | spark-5 |
| spark-5 | 1 | `192.0.2.50/29` | `192.0.2.58/29` | 4 | spark-4 |
| spark-8 | 0 | `192.0.2.113/29` (same) | `192.0.2.121/29` (same) | 8 | spark-1 |

Only spark-1 port 1 and spark-5 port 1 get new addresses. The port 0 ends of
cables 4 and 8 keep theirs.

Every fabric interface uses MTU 9000, no gateway and no DHCP. The
[setup scripts](spark-setup.md#1-make-a-private-cable-map) write these
settings from a private `links.tsv` with one row per rail. Use the `ring`
template on every Spark: each Spark has two cables in both options.

<br>

---

## 4. What each option gives

Vade runs one server per **group**: a set of Sparks that work on the same
chats. A **layout** is a set of groups that run at the same time. With eight
Sparks, these layouts use all eight at equal total load:

| Layout | Groups | Option A, two rings of four | Option B, one ring of eight | What Vade runs in each group today |
| --- | --- | --- | --- | --- |
| **8 singles** | 8 groups of 1 | yes | yes | FP4-Local on one Spark. Or one `vade up --replicas 8` with one local gateway. No fabric. |
| **4 doubles** | spark-1+2, 3+4, 5+6, 7+8 (cables 1, 3, 5, 7) | yes | yes | FP4-Local tensor split over RDMA, or FP8 two-way TP. |
| **2 quads** | spark-1..4, spark-5..8 | yes: two **rings** of four | paths of four only (no closing cable). A group needs a ring, so start one quad at a time with `--hosts` | FP4-Local layer chain (a path is enough). FP8 four-way TP (path). |
| **1 eight** | spark-1..8 | **no**: no cable joins the two rings | yes | FP4-Local layer chain of eight. FP8 refuses eight: `vade up --fp8` takes 2, 4 or 6 Sparks. |

**A tensor split over four or eight Sparks is not in this build.** Today the
only tensor split for FP4-Local is the pair. A four-way split is in design.
When it lands, a quad can run one chat on four Sparks, and the closing cable
of a ring of four is available to it.

**Distance in cables.** In a ring of four, the farthest Spark is 2 cables
away. In a path of four it is 3. In a ring of eight it is 4, and in a path of
eight it is 7. Each cable carries about 196 Gb/s in each direction (see
[the measured link numbers](network-setup.md#numbers-we-measured)).

**Memory.** Each Spark has 128 GB shared by CPU and GPU. A group of four has
512 GB. All eight have 1,024 GB. Only a checkpoint that does not fit in four
Sparks needs one group of eight.

**Faults.** In option A, a failed cable or Spark stops only its own ring; the
other ring keeps serving. A ring with one failed cable is still a path of four,
so its layer chain can run. In option B, one failed cable leaves a path of
eight; groups that use that cable stop.

<br>

---

## 5. Recommendation for day one

**Use option A, two rings of four.** Use option B later, for a model or a
split that needs all eight Sparks in one group.

1. **The pair is the engine's unit today.** The tensor split is the only
   two-Spark path, and both options give the same four pairs. Option B adds
   nothing for pairs or singles.
2. **A ring of four is the right shape for four.** Each quad gets its closing
   cable, so the farthest Spark is 2 cables away, not 3. Option B gives each
   quad a path only.
3. **Isolation.** The two rings share no cable. One group can serve while the
   other runs tests or a new build. A fault, a reboot or a recabling in one
   ring does not stop the other.
4. **Nothing in this build needs eight in one group.** FP4-Local fits one
   Spark, and FP8 runs on 2, 4 or 6. We use the FP4-Local chain of
   eight to measure it, not to serve.
5. **The change is small.** To run the eight-Spark cards, change to option B
   for that session: two cable ends and the addresses on two ports
   ([section 6](#6-change-between-the-two-options)).

<br>

---

## 6. Change between the two options

Stop every server first (`VADE_GROUP=NAME vade down` for each group, then
plain `vade status` shows `Vade is stopped.` and names no group). Never move a cable that a running rank uses. Make network
changes from the local console of the Spark, not over SSH on a fabric port.

**From A (two rings) to B (one ring):**

1. Unplug the cable 4 end from spark-1 port 1 and the cable 8 end from
   spark-5 port 1.
2. Plug cable 4 into **spark-5 port 1**. Plug cable 8 into **spark-1 port 1**.
3. On spark-1, change port 1 rail 0 to `192.0.2.114/29` and rail 1 to
   `192.0.2.122/29`.
4. On spark-5, change port 1 rail 0 to `192.0.2.50/29` and rail 1 to
   `192.0.2.58/29`.
5. Select the controller config for option B (see
   [section 7](#7-controller-config-machines-groups-and-layouts)), then run
   `vade doctor --fabric`.

**From B to A:** do the same steps with the ends and addresses the other way
around: cable 4 to spark-1 port 1 (`192.0.2.50`, `192.0.2.58`), cable 8 to
spark-5 port 1 (`192.0.2.114`, `192.0.2.122`).

**From one ring of six to option A.** If six Sparks already run as one ring
cabled with this plan (cables 1 to 6, where cable 6 goes from spark-6 port 0
to spark-1 port 1):

1. Move the cable 6 end from spark-1 port 1 to **spark-7 port 1**.
2. Move the cable 4 end from spark-5 port 1 to **spark-1 port 1**.
3. Add cable 7 (spark-7 port 0 to spark-8 port 1) and cable 8 (spark-8 port 0
   to spark-5 port 1).
4. Change spark-1 port 1 from `192.0.2.82` and `192.0.2.90` to `192.0.2.50` and
   `192.0.2.58`. Change spark-5 port 1 from `192.0.2.50` and `192.0.2.58` to
   `192.0.2.114` and `192.0.2.122`.
5. Give spark-7 and spark-8 the addresses in [section 3](#3-address-plan).

spark-2, spark-3, spark-4 and spark-6 keep all their cables and addresses.

<br>

---

## 7. Controller config: machines, groups and layouts

The controller config has one `[machine]` section per Spark. Each
`fabric.<peer>` value is the Spark's **own rail 0 address on the cable to that
peer**. Vade finds rail 1 by itself. Two Sparks are linked in the config when
each names the other.

Keep one private config file per cabling option, for example
`.local/eight-a.conf` and `.local/eight-b.conf`, and select one with
`VADE_CONFIG`. The machine sections differ only for spark-1, spark-4, spark-5
and spark-8. The groups differ too: option B has no group of four that closes
into a ring. Start from the complete examples:
[cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf) for option A
and [cli/examples/eight-sparks-ring8.conf](../cli/examples/eight-sparks-ring8.conf)
for option B. Every group in each file passes the group ring check: its
`group NAME` row in `vade doctor --fabric`. The tests check those rows and a
dry run; the port, cable and clock rows need the real Sparks.

```ini
[machine spark-1]
ssh = spark-1-ssh-alias
fabric.spark-2 = 192.0.2.1
# Option A: cable 4 from spark-4. Option B: fabric.spark-8 = 192.0.2.114
fabric.spark-4 = 192.0.2.50
```

**Groups.** A `[group NAME]` section names the Sparks of one server.

| Key | Meaning |
| --- | --- |
| `hosts` | Machine names in **rank order**, separated by commas. Rank 0, the first, answers HTTP. List them in cable order around the ring. |
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
- With `vade up`, a group must close into a ring in the listed order: each
  host is linked to the next, and the last to the first. A group of two needs
  one link. A group of one needs none. If the hosts form a ring in another
  order, the error gives that order.
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

```sh
VADE_GROUP=quad-a vade up --dry-run
VADE_GROUP=quad-a vade up
VADE_GROUP=quad-b vade up
VADE_GROUP=quad-a vade status
VADE_GROUP=quad-b vade down
```

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

## 8. Check the fabric

```sh
VADE_CONFIG=.local/eight-a.conf vade doctor --fabric --dry-run
VADE_CONFIG=.local/eight-a.conf vade doctor --fabric
```

`vade doctor --fabric` reads every machine in the config. It makes three SSH
calls per Spark (interfaces, pings, clock) and changes nothing. It prints one row per check:

| Row | It passes when |
| --- | --- |
| `cabling` | Always passes. Shows each connected part of the config: `ring of 4: …`, `path of 4: …`, `pair: …` or `no link: …`. |
| `group NAME` | The group's hosts close into a ring in the listed order. |
| `<spark> port to <peer>` | An interface on the Spark holds its `fabric.<peer>` address, links at 100,000 Mb/s or more, has MTU 9000 or more, and has an RDMA device in state `ACTIVE`. With `SEQPAR_RDMA=0` (TCP), the RDMA device is not required. |
| `<a> - <b> cable` | Both ends name each other in the config. Rail 0 is the pair of interfaces with the `fabric.<peer>` addresses, in one subnet. Rail 1 is a second subnet on two other fabric interfaces, one at each end; an address alias on the rail 0 interface does not count. **Each rail on its own**: the two ends hold different addresses, both ends link at 100,000 Mb/s or more with the same MTU of 9000 or more and an `ACTIVE` RDMA device, and each end answers a ping and a 9000-byte ping that must not be split (`ping -M do -s 8972`) from the other end. A ping to an address that the pinging Spark holds itself fails: it never leaves the Spark. No other cable uses either subnet. The cable fails when either rail fails. |
| `<spark> clock` | The Spark's clock is within 1,000 ms of the controller's clock, beyond the SSH round trip, and `timedatectl` reports time sync on. |

The cable check is the "both neighbors agree" check: an address on the wrong
port, a swapped `fabric.<peer>` value, a link with only one end in the config,
a missing or faulty rail 1, a rail 1 whose link is down, the same address at
both ends of a rail, or an MTU of 1500 at one end each fail one named row. Each failed row gives the next step. A
host name in a `fabric.<peer>` value is skipped; this check reads numeric IPv4
addresses only.

Then check each group as `vade up` will start it:

```sh
VADE_GROUP=quad-a vade doctor
```

That runs the normal checks (SSH, GPU, disk, memory, model files, ports) on
the group's Sparks, and pings each link of the group.

<br>

---

## 9. Day-one checklist

Do the steps in order. Do not start a step until the one before it passes.
Write down the time and the result of each step.

**Before the new Sparks arrive**

- [ ] Print the [cable plan](#2-cable-plan) and the
      [address plan](#3-address-plan). Label eight cables 1 to 8, both ends.
- [ ] On every Spark, check that the private range you chose is free:
      `ip -4 route show table all`.
- [ ] Write both private controller configs (option A and option B) and the
      groups for the three layouts (singles, doubles, quads). Start from
      [cli/examples/eight-sparks.conf](../cli/examples/eight-sparks.conf) and
      [cli/examples/eight-sparks-ring8.conf](../cli/examples/eight-sparks-ring8.conf).

**1. Unbox**

- [ ] Place spark-7 and spark-8. Connect power and the management LAN. Keep a
      local console ready.
- [ ] Finish the OS first boot. Match the kernel (`uname -r`) and the
      driver (`nvidia-smi`) of the other six. On kernel `7.0.0-1019-nvidia`,
      apply the `kho=off` step
      ([spark-setup.md, step 5](spark-setup.md#5-check-the-kernel-before-registering-memory)).
- [ ] Put the same verified checkpoint at the same absolute path as on the
      other Sparks ([INSTALL.md, step 3](../INSTALL.md#step-3-get-the-model)).

**2. Cable**

- [ ] Stop every server on the six Sparks. Plain `vade status` shows
      `Vade is stopped.` and names no group.
- [ ] Move and add cables as in
      [from one ring of six to option A](#6-change-between-the-two-options).
- [ ] On each Spark, `ibdev2netdev` shows every cabled port `(Up)`.

**3. Address**

- [ ] On spark-7 and spark-8, run the OS setup steps 1 to 4 of
      [spark-setup.md](spark-setup.md) with the `ring` template and the
      addresses from [section 3](#3-address-plan).
- [ ] On spark-1 and spark-5, change the port 1 addresses only.
- [ ] Run the bandwidth test on the new and moved cables (4, 6, 7, 8), both
      rails ([network-setup.md, step 7](network-setup.md#7-check-every-link)).
      Expect about 196 Gb/s per cable with both rails.

**4. Doctor**

- [ ] `vade doctor --fabric` with the option A config: every row passes, and
      `cabling` shows two rings of four.
- [ ] `vade install`, so that every Spark has the same staged build.
- [ ] `VADE_GROUP=<group> vade doctor` for each group of each layout.

**5. First two-Spark card per new pair**

- [ ] Start the new pair (spark-7 + spark-8) as its own group and run the
      FP4-Local two-Spark card.
- [ ] Run the same requests on one existing pair. The token IDs must be
      identical for the same request and settings.

**6. Four-Spark cards**

- [ ] For each quad (spark-1..4 and spark-5..8): FP4-Local four-Spark card
      (layer chain) and FP8 four-Spark card.

**7. Layout cards**

- [ ] Run the three layouts at equal total load: 8 singles, 4 doubles, 2
      quads. Record the same cells for each layout, and check that the token
      IDs are identical across layouts for the same request.

**8. Eight-Spark card**

- [ ] [Change to option B](#6-change-between-the-two-options). Run
      `vade doctor --fabric` with the option B config (from
      [cli/examples/eight-sparks-ring8.conf](../cli/examples/eight-sparks-ring8.conf)):
      every row passes, and `cabling` shows one ring of eight.
- [ ] Run the FP4-Local eight-Spark card (layer chain). FP8 on eight Sparks
      is refused by this build.
- [ ] Change back to option A, and run `vade doctor --fabric` again.

Each card records the revision, the hardware, the checkpoint, the settings and
the raw results ([benchmarks](benchmarks.md)). A layout that we have not
tested still needs its own exactness check on real Sparks before you rely on
it.

<br>

**Back:** [Network setup](network-setup.md) · **See also:** [Four Sparks](four-sparks.md) · [FP4-Local launch](../cli/docs/fp4-local.md)
