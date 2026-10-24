# Prepare a fresh DGX OS Spark for Vade

> **New to this?** Read [Hardware and network setup](network-setup.md) first:
> what to buy, cabling diagrams, an example address scheme and expected link
> numbers, in plain words. This page is the full command reference.

This is the system setup before [INSTALL.md](../INSTALL.md): cable the Sparks,
give both halves of every QSFP link their own network, allow the Vade account to
use RDMA, and check the session that will launch Vade. Repeat it on every node.
A pair needs one direct cable; a ring needs a cable from each node to each of
its two neighbors. Keep the ordinary LAN connection for administration.

To convert two existing double-linked pairs, use
[Prepare four Sparks](four-sparks.md). It gives the exact cable moves, the
path endpoint templates and the migration boundary for an already configured
fabric; the fresh-interface apply step below does not overwrite that fabric.

**UNVERIFIED: this procedure has not been executed from a blank DGX OS.** It is
based on read-only inspection of working Sparks, primary documentation and
script checks. No system change, reboot or bandwidth test
was performed while writing it. An existing working network is evidence for the
target state, not proof that these new commands reconstruct it. The previously
recorded Vade install preserved OS settings; its stale canary receipt has been
[removed](../bench/receipts/README.md).

Run each complete command block below from the repository root in Bash. Each
block runs in its own subshell and enables failure handling: a failed command
stops that block even if your terminal has no `set -e` setting. Stop the procedure
and investigate a failure before running another block. Initial DGX OS setup,
the NVIDIA driver, a normal administrator account, Git and repository
access must already work. Use DGX OS packages and NVIDIA's supported firmware;
these scripts do not install an alternative driver. For future system changes,
use the local console and an idle maintenance window. This guide's `sudo`
commands are operator instructions; they were not run while writing this guide.
Create the private configuration as the account that will run Vade; that account
needs administrator access for the system steps.

Each setup step prints its proposed commands and file differences first.
An ordinary invocation writes no files and changes no system settings; add
`--apply` only after reviewing that preview. This also applies to creating local
examples and rendered files. Verification previews its checks by default;
`--apply` runs them, including peer SSH, ping and bandwidth traffic. The manual
console commands below, such as `netplan try`, remain separate operator actions.
Network activation and service restarts below require a successful apply that
reports `changed`. An identical setup reports `unchanged` and skips activation.
The private result files under `.local/` record this outcome; keep them until
you have reviewed the operation. Preview and failed apply never authorize activation.

## 1. Make a private cable map

One QSFP port exposes **two Ethernet interfaces and two corresponding RoCE
devices**, on separate PCIe paths. Configure both rails with MTU 9000. Two
Sparks on one cable therefore need two rows of configuration on each node;
a ring node needs four. The Ethernet MTU is 9000, while the RDMA path MTU
reported by verbs should be 4096. NVIDIA's
[ConnectX-7 table](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
lists the correspondence. Confirm it with `ibdev2netdev` or `rdma link show`;
port stickers and historical left/right descriptions are insufficient.

Choose an unused RFC 1918 address block after checking LAN, VPN and all routing
tables on **every** node. Assign a distinct `/24` to each rail of each physical
cable. Give its endpoints host numbers 1 and 2; use the same subnet at both
ends. A pair uses two subnets; an N-node ring uses 2N. Keep them disjoint from
existing networks. No gateway or forwarding is needed for these direct links.

| Connection | Rail 0 | Rail 1 |
| --- | --- | --- |
| Pair's single cable | First unused subnet, endpoints 1/2 | Second unused subnet, endpoints 1/2 |
| Ring's next cable | Next unused subnet, endpoints 1/2 | Following unused subnet, endpoints 1/2 |

Write down which actual port at each end is connected; a cable may join
different port numbers. A ring peer's address is local to that cable. It is
normal for a peer's fabric alias to resolve differently on different nodes.
Give **each peer rail** a distinct local alias. Use the rail-0 alias of each
direct neighbor as the usual peer name; rail-1 gets a separate alias for checks.
Keep the OS hostname and management addresses separate. Nonadjacent ring nodes
use LAN SSH; this recipe does not route the fabric between nonadjacent nodes.

Generate examples **under ignored `.local/`**, then edit them for this node:

```sh
(
set -euo pipefail
bash ops/scripts/spark-setup/init-config.sh pair .local/spark-setup/node
# After reviewing the placeholder files and commands:
bash ops/scripts/spark-setup/init-config.sh pair .local/spark-setup/node --apply
# For a ring, use `ring` instead of `pair` on a new directory.
CFG="$PWD/.local/spark-setup/node/config.env"
ip -br link
ip -br address
ip -4 route show table all
nmcli -f DEVICE,TYPE,STATE device status
# Edit .local/spark-setup/node/config.env and links.tsv with your editor.
)
```

`init-config.sh` refuses to overwrite existing configuration. `config.env` is
trusted local shell code. Set `SERVICE_USER` to the account used by INSTALL.md,
and optionally `SERVICE_UNIT` to an existing Vade systemd service. Leave it empty
for the guide's SSH launches. `SSH_KEY_FILE` names that account's own key path;
the setup scripts do not read or distribute private keys. `KERNEL_FIXED` is an
explicit, evidence-backed approval of an exact kernel, described below.

The seven whitespace-separated fields in `links.tsv` are:

| Field | Fill with |
| --- | --- |
| `netdev` | This rail's Ethernet interface from `ip link` |
| `rdma_device` | Its verbs device from `ibdev2netdev` or `rdma link show` after package installation |
| `local_ipv4/24` | This node's chosen rail address and prefix |
| `peer_ipv4` | The directly cabled peer's address on the same rail |
| `peer_alias` | A unique local name for that peer rail |
| `ipv4_rocev2_gid` | The local GID index matching that interface and IPv4 address |
| `test_tcp_port` | A free high TCP port, matching the peer's row for this rail |

Each row must have a unique device, subnet, alias and test port on this node.
The generator's test ports are suggestions: adjust the peer endpoint's row to
match even if that row has a different number. Before addresses exist, use GID
index `0` **only as a staging placeholder**. Replace it with the actual IPv4
RoCE-v2 index after step 3; verification rejects a mismatched GID. Device names
can be filled from the sysfs inspection below if `ibdev2netdev` is not installed.
No actual names, addresses or filled examples belong in Git.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
for d in /sys/class/infiniband/*; do
  printf '\n%s\n' "${d##*/}"
  ls "$d/device/net"
done
git check-ignore .local/spark-setup/node/config.env
bash ops/scripts/spark-setup/render.sh "$CFG" .local/spark-setup/rendered
# After reviewing the file differences:
bash ops/scripts/spark-setup/render.sh "$CFG" .local/spark-setup/rendered --apply
cat .local/spark-setup/rendered/40-cx7.yaml
cat .local/spark-setup/rendered/hosts.fragment
)
```

## 2. Install RDMA tools from the OS

Use the distro's libibverbs loader, Mellanox providers, verbs utilities and
`perftest`. Vade uses host-memory RDMA; Spark does not support GPUDirect RDMA
([NVIDIA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)).
Do this while updates are enabled and the package manager is idle. Package
installation is repeatable; it is **UNVERIFIED on a fresh image**.
The package step also installs SSH, Netplan, NetworkManager and diagnostic tools.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" packages
# After reviewing the package commands:
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" packages --apply
if command -v ibdev2netdev >/dev/null; then ibdev2netdev; else rdma link show; fi
ibv_devinfo
ldconfig -p | awk '$1 == "libibverbs.so.1" { found=1 } END { exit !found }'
ls -l /dev/infiniband
)
```

## 3. Install and activate the fabric network

The rendered netplan file uses NetworkManager **per fabric interface**, with
static addresses and MTU 9000. It leaves the management interface out of that
file. Both endpoints must be configured before a link can pass its checks.
Before any write, the network step checks the live addresses and routes,
NetworkManager's active and saved profile ownership, and Netplan's effective
definitions, including `match:` rules. It refuses an already configured fabric
or any rail carrying the current SSH connection. Missing or ambiguous ownership
or SSH connection information also stops it. Run from the local console when
SSH provenance cannot be established; the script checks for a real local console.
Over SSH, preserve `SSH_CONNECTION` through sudo as shown below. An exactly
identical desired file set is a read-only no-op; it does not regenerate or
activate the network.

The OS helpers use shell and the system tools' line-oriented output. If they
cannot prove ownership or session safety from a supported output format, they
refuse the operation and explain which inspection needs administrator review.
Do not bypass that refusal or simplify an existing configuration just to make
the helper accept it.

In particular, IPv6 SSH provenance, Netplan `match:` or `set-name:` rules,
nonempty virtual-device sections, and opaque NetworkManager selectors need
separate review. The helper supports IPv4 management SSH or a real Linux
console, and simple unrelated Ethernet definitions. A working fabric with
more complex ownership should be kept and verified, rather than converted
by this setup step.

An existing destination file with different contents stops the operation,
even if it appears to define only these rails. Preserve that file and ask the
administrator to prepare the fabric separately. The script does not replace
or remove an existing configuration to make this recipe fit.

These scripts prepare unused interfaces on a fresh image. If a fabric already
works, keep it and use the verification step. If a conflicting profile exists,
have the administrator review that interface's configuration separately; this
recipe provides no automatic conversion or deletion. Preserve the LAN/Wi-Fi
profiles. DGX OS may have `90-NM-*` profiles. Conflicting hosts aliases outside
the managed block also stop installation. Avoid broad configuration dumps that
might reveal Wi-Fi credentials.

Once every preflight check passes, `--apply` records the immediate prior files,
including which files and directories were absent, in a private transaction
under `/var/lib/vade-spark-setup/transactions/`. It prints that transaction's
path. Keep it for rollback. Then it installs `40-cx7.yaml` and replaces only its
marked hosts block. It stages persistent files; the later console `netplan try`
generates and activates the candidate network. A failed apply restores the
recorded prior state automatically; retain the transaction and check any
reported recovery failure before proceeding.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
sudo --preserve-env=SSH_CONNECTION bash ops/scripts/spark-setup/apply.sh "$CFG" network
# After reviewing the proposed commands and file differences:
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/network-result.XXXXXX")
sudo --preserve-env=SSH_CONNECTION bash ops/scripts/spark-setup/apply.sh "$CFG" network \
  --result-file "$RESULT_DIR/result" --apply
RESULT=$(cat "$RESULT_DIR/result")
# From the local console, with both ends prepared:
case "$RESULT" in
  changed) sudo netplan try --timeout 120 ;;
  unchanged) printf 'Network already matches; activation skipped.\n' ;;
  *) printf 'Invalid apply result; stop before activation.\n' >&2; exit 1 ;;
esac
)
```

While `netplan try` waits, use a second terminal to check the management login
and both rails with `ip -br address`, `ip link` and a peer ping. Accept only if
they work. On timeout/cancel, **check that rollback actually restored the live
network**; Netplan documents rollback caveats. A failed trial leaves the
installed files in place; Netplan does not roll back `/etc/hosts`. Restore the
setup transaction before rebooting. The rollback commands for network, RDMA
and kernel changes are together below. See
[netplan try](https://netplan.readthedocs.io/en/stable/netplan-try/) and the
[YAML reference](https://netplan.readthedocs.io/en/stable/netplan-yaml/).

### Restore the immediate prior setup

Use the transaction path printed by the corresponding `--apply`, on the same
node. Roll back later transactions first if they changed the same files. Review
the rollback's file differences before applying it; preserve any unrelated edits
made since setup instead of overwriting them. Transactions remain private and
must not be deleted as cleanup. Package installation has no automatic rollback;
review the package manager's transaction history separately.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# Network rollback, from the console:
read -r -p 'Network transaction path printed by apply: ' SNAPSHOT
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/network-rollback-result.XXXXXX")
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" \
  --result-file "$RESULT_DIR/result" --apply
RESULT=$(cat "$RESULT_DIR/result")
# Persistent files are restored; generation and activation are still manual.
case "$RESULT" in
  changed) sudo netplan try --timeout 120 ;;
  unchanged) printf 'Network already restored; activation skipped.\n' ;;
  *) printf 'Invalid rollback result; stop before activation.\n' >&2; exit 1 ;;
esac
# Check management login and rails before accepting; check live rollback on failure.
)
```

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# RDMA rollback, from the console with a recovery session retained:
source ops/scripts/spark-setup/common.sh
load_config "$CFG"
read -r -p 'RDMA transaction path printed by apply: ' SNAPSHOT
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/rdma-rollback-result.XXXXXX")
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" \
  --result-file "$RESULT_DIR/result" --apply
RESULT=$(cat "$RESULT_DIR/result")
# The helper restores prior files, group membership and any group it created,
# reloads udev rules, restores saved device ownership/modes and reloads systemd.
case "$RESULT" in
  changed)
    sudo /usr/sbin/sshd -t
    sudo systemctl restart ssh.service
    if [[ -n $SERVICE_UNIT ]]; then sudo systemctl restart "$SERVICE_UNIT"; fi
    ;;
  unchanged) printf 'RDMA already restored; service restarts skipped.\n' ;;
  *) printf 'Invalid rollback result; stop before restarts.\n' >&2; exit 1 ;;
esac
# Restart a Vade service only while idle; log out and back in for prior limits/groups.
)
```

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# Kernel rollback, before the next maintenance reboot:
read -r -p 'Kernel transaction path printed by apply: ' SNAPSHOT
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" --apply
# The helper restores/removes the drop-in and grub.cfg exactly as recorded.
# Reboot manually from the console only during the agreed maintenance window.
uname -r
cat /proc/cmdline
)
```

Removing a mitigation file does not change the running kernel. After a reboot,
confirm that `/proc/cmdline` matches the restored configuration. Rollback cannot
revoke groups or limits already inherited by existing processes; new sessions
and idle-service restarts are needed. Inspect any failed reload or regeneration
before using the machine.

Find the IPv4 RoCE-v2 GID for each configured rail and replace the staging index
in `links.tsv`. The correct entry maps to the chosen Ethernet interface, has
type `RoCE v2`, and contains that rail's IPv4 address as an IPv4-mapped IPv6 GID.
Do not copy an index from another OS release or from an unrelated interface.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source ops/scripts/spark-setup/common.sh
load_config "$CFG"
for i in "${!RDMA_DEVS[@]}"; do
  p="/sys/class/infiniband/${RDMA_DEVS[$i]}/ports/1"
  printf '\nrail %s: %s %s\n' "$((i+1))" "${NETDEVS[$i]}" "${LOCAL_CIDRS[$i]}"
  for f in "$p"/gid_attrs/types/*; do
    idx=${f##*/}
    gid_type=$(cat "$f")
    [[ $gid_type == 'RoCE v2' ]] || continue
    printf 'index %s: ' "$idx"
    cat "$p/gid_attrs/ndevs/$idx" "$p/gids/$idx"
  done
done
)
```

## 4. Give the account device access and unlimited memlock

RDMA registration needs permission to open the rail's `/dev/infiniband/uverbs*`
device and lock host memory. The `rdma` step creates a dedicated access group,
adds the configured account, installs udev rules, and sets both soft and hard
memlock limits to unlimited for that account. It also stages an unlimited limit
for the SSH service and, if configured, the existing Vade service. It checks
that SSH uses PAM and its PAM stack includes `pam_limits.so`; it does not rewrite
authentication settings. These requirements follow
[RDMA-core](https://github.com/linux-rdma/rdma-core/blob/master/Documentation/libibverbs.md),
[Linux-PAM](https://github.com/linux-pam/linux-pam/blob/master/modules/pam_limits/limits.conf.5.xml)
and [systemd](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml).
The device refresh uses [udevadm's sysname matching](https://github.com/systemd/systemd/blob/v255/man/udevadm.xml)
for the actual RDMA device, rather than guessing its device-node path.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source ops/scripts/spark-setup/common.sh
load_config "$CFG"
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rdma
# After reviewing the proposed files, group changes and reload commands:
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/rdma-result.XXXXXX")
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" rdma \
  --result-file "$RESULT_DIR/result" --apply
RESULT=$(cat "$RESULT_DIR/result")
# From the console, retaining a recovery session; any Vade service must be idle:
case "$RESULT" in
  changed)
    sudo /usr/sbin/sshd -t
    sudo systemctl restart ssh.service
    if [[ -n $SERVICE_UNIT ]]; then sudo systemctl restart "$SERVICE_UNIT"; fi
    ;;
  unchanged) printf 'RDMA already matches; service restarts skipped.\n' ;;
  *) printf 'Invalid apply result; stop before restarts.\n' >&2; exit 1 ;;
esac
)
```

Log out and back in as `SERVICE_USER`, then return to the repository root to
check the new session:

```sh
(
set -euo pipefail
id
ulimit -Sl                         # unlimited
ulimit -Hl                         # unlimited
)
```

Existing SSH shells, tmux servers and user service managers keep inherited
limits and groups. Start a new session after the change. The changed branch
above restarts a configured Vade service; keep it idle for that operation.
If one is configured, check its actual process too:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source ops/scripts/spark-setup/common.sh
load_config "$CFG"
if [[ -n $SERVICE_UNIT ]]; then
  systemctl show "$SERVICE_UNIT" -p User -p LimitMEMLOCK -p LimitMEMLOCKSoft
  pid=$(systemctl show "$SERVICE_UNIT" -p MainPID --value)
  [[ $pid =~ ^[1-9][0-9]*$ ]] || { printf 'Service has no live process.\n' >&2; exit 1; }
  grep 'Max locked memory' "/proc/$pid/limits"
fi
)
```

A Tailscale SSH route or another non-PAM entry path may still give a small hard
limit. Use ordinary OpenSSH over the LAN or fabric and check the **exact shell
used to launch Vade**. Running verification under sudo would check root instead
of the service account. An SSH shell PASS does not qualify a systemd launch.
Fresh permission, daemon restart and unit behavior are **UNVERIFIED** here.

The step validates the account, group, PAM/SSH settings and any configured
service before writing. Its transaction records the original group
membership and whether the group existed, as well as prior file contents or
absence and existing RDMA device ownership and permissions. On failure it
restores that state; use the rollback commands in step 3 for a later manual recovery.

## 5. Check the kernel before registering memory

NVIDIA's [advisory](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254)
identifies `7.0.0-1019-nvidia` as affected by a multi-node RoCE regression and
publishes `kho=off`. NVIDIA began rolling out mitigation updates September 17,
2026, but **a particular fixed replacement build is UNVERIFIED here**.
Do not infer a fix from a higher version number. If using another kernel,
confirm its exact build against NVIDIA's release/advisory evidence, record that
source-approved candidate's exact `uname -r` in local `KERNEL_FIXED`, then run
the RDMA checks. The local approval permits those checks; a PASS on the kernel
selection alone does not establish a working replacement kernel.

For the known affected build, first inspect `/proc/cmdline`. The optional
`kernel` step saves the prior files and stages a small GRUB drop-in. It then
generates a private candidate with
[`grub-mkconfig`](https://www.gnu.org/software/grub/manual/grub/html_node/Invoking-grub_002dmkconfig.html),
checks its syntax with
[`grub-script-check`](https://www.gnu.org/software/grub/manual/grub/html_node/Invoking-grub_002dscript_002dcheck.html),
and records its bytes before publishing it; it does not reboot. Backups and
live command lines already contained NVIDIA's own KHO drop-in. The new drop-in
avoids adding a second identical `kho=off` token.
**UNVERIFIED: new GRUB generation and reboot persistence.**

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
uname -r
cat /proc/cmdline
# Only for 7.0.0-1019-nvidia when preparing this mitigation:
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" kernel
# After reviewing the GRUB file difference and generation command:
sudo bash ops/scripts/spark-setup/apply.sh "$CFG" kernel --apply
# Reboot manually from the console during the agreed maintenance window.
# After the reboot:
uname -r
cat /proc/cmdline                  # affected build must contain kho=off
)
```

The step checks the effective GRUB command line for a conflicting `kho=` option
before installing the drop-in. It records prior contents or absence and the
generated `/boot/grub/grub.cfg`; on failure it restores the immediate prior files.
Use step 3's kernel rollback to restore the exact saved drop-in and GRUB bytes,
including after interrupted installation; rollback does not regenerate GRUB.
No helper reboots the Spark.

`kho=off` is a mitigation, not evidence that every memory registration under
pressure is safe. The historical `flash-bench` finding at `4bd515e` concerned
registration after model loading. Later prerequisite changes registered and
sealed RDMA memory before weights were loaded. That old finding does not
qualify a current build; retain its startup and registration evidence. Use [INSTALL.md](../INSTALL.md)
for the current Flash launch. A small `ib_write_bw` run does not qualify
registration under model-serving memory pressure.

An earlier tested sysctl set `vm.compaction_proactiveness=0` to reduce historical
stalls. It is not a demonstrated RDMA prerequisite and these scripts do not
change it. Restoring that performance choice requires separate qualification;
its necessity on a fresh OS is **UNVERIFIED**.

## 6. Establish SSH between the Sparks

Every node needs its own account key and trusted peer host keys. Keep private
keys on their originating node. Do not reuse the historical shared private key.
From each service account, use its configured key path, create a key only if
absent, and copy only the public key to both directly cabled neighbors.
Compare the peer's SSH host-key fingerprint with its console output before
accepting it, for both the alias and the literal peer IPv4 address. Verification
connects to that address and binds the local rail. After a reinstall, remove an obsolete known-host entry only once
the new identity has been confirmed. Do not disable host-key checking.
This is **UNVERIFIED for newly installed nodes**; follow the ordinary SSH steps
in NVIDIA's [pair playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
or [ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring).

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source ops/scripts/spark-setup/common.sh
load_config "$CFG"
KEY_DIR=$(dirname "$SSH_KEY_FILE")
install -d -m 700 "$KEY_DIR"
if [[ ! -e $SSH_KEY_FILE ]]; then
  ssh-keygen -t ed25519 -f "$SSH_KEY_FILE"
fi
# On each peer console: ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
for i in "${!PEER_IPS[@]}"; do
  ssh-copy-id -i "$SSH_KEY_FILE.pub" "$SERVICE_USER@${PEER_IPS[$i]}"
  ssh -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes "$SERVICE_USER@${PEER_NAMES[$i]}" true
  ssh -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UpdateHostKeys=no "$SERVICE_USER@${PEER_IPS[$i]}" \
    'id && ulimit -Sl && ulimit -Hl'
  # INSTALL.md uses these same identity and authentication options for transfers.
  ssh -i "$SSH_KEY_FILE" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UpdateHostKeys=no "$SERVICE_USER@${PEER_NAMES[$i]}" true
done
)
```

For unattended launches, unlock a passphrase-protected key in the account's
SSH agent. Verify both directions and all rail aliases. The first authenticated
connection records the alias/address host key; verification subsequently requires the IP entry.
Management SSH between nonadjacent ring nodes needs its own trusted names/keys.
INSTALL.md explicitly asks for the sender's local private-key path for SSH,
SCP and rsync. On a Spark prepared here, supply this `SSH_KEY_FILE` path; on a
different source machine, use that machine's own authorized key. A peer check
with this custom key does not make OpenSSH select it automatically elsewhere.

## 7. Pause updates only for the benchmark window

Finish approved OS/package updates first. The helper records the original
enabled and active states once, then masks the APT daily timers/services and
unattended-upgrades service. It refuses to interrupt active APT daily jobs.
Absent units need no action. Repeated pauses keep the original restore state.
Restore the recorded state [after benchmark shutdown](#9-finish-benchmarking-and-restore-updates);
preexisting masks remain masks.
State is private under `/var/lib/vade-spark-setup/`. The saved rows also prove
that the underlying unit files and enablement have not changed before restoring
them. New snapshots identify their shell fingerprint format. Older snapshots
require administrator review; the helper preserves them and refuses to guess
or convert their hashes. Do not delete them as cleanup.
The helper does not control NVIDIA Dashboard OTA, manual package commands or
other custom updaters; their pause behavior is **UNVERIFIED**.
An active unattended-upgrades service is accepted only when its single command
matches the supported Ubuntu shutdown helper and output format. Alternate
commands or ambiguous service descriptions stop the pause before any write;
have the administrator review them separately.

```sh
(
set -euo pipefail
sudo bash ops/scripts/spark-setup/updates.sh pause
# After reviewing the saved-state and systemctl commands:
sudo bash ops/scripts/spark-setup/updates.sh pause --apply
bash ops/scripts/spark-setup/updates.sh status
)
```

## 8. Verify every rail and the actual launch session

Run the default check as the configured service account in a fresh ordinary SSH
shell. It previews the checks and peer commands without running them. After
reviewing, add `--apply` to get PASS/FAIL for architecture/driver, verbs library
and device access, current soft/hard memlock, kernel choice, addresses, carrier, MTUs,
RoCE-v2 GIDs, peer naming, jumbo ping, trusted peer SSH and paused update units.
A GID check accepts the kernel's eight full hexadecimal fields for an
IPv4-mapped address; alternate or ambiguous representations fail closed.
A configured service additionally gets unit/process limit checks. It returns
nonzero on a failed prerequisite. Bandwidth is explicitly **UNVERIFIED** in
the prerequisite mode, even when all its checks pass.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
mkdir -p .local/spark-setup/evidence
bash ops/scripts/spark-setup/verify.sh "$CFG" \
  | tee .local/spark-setup/evidence/prerequisites-preview.txt
# After reviewing the proposed checks and peer commands:
bash ops/scripts/spark-setup/verify.sh "$CFG" --apply \
  | tee .local/spark-setup/evidence/prerequisites.txt
)
```

With both endpoints idle and reserved under the current operations policy,
test each cable rail. This uses host memory and a 64 KiB transfer size for five
seconds, with a 25-second process timeout, RDMA MTU 4096 and the configured
device/GID/source IPv4. It prints the raw perftest rate in **Gb/s** and a PASS
only for successful completion with positive average bandwidth. That PASS
proves connectivity, not a throughput target. Use matching test ports and
perftest versions/options at the two endpoints.

In two terminals, start the server first, then its matching client promptly:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
mkdir -p .local/spark-setup/evidence
# Node at one end, its local row number (1-based):
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-server 1 \
  | tee .local/spark-setup/evidence/rail-1-server-preview.txt
# After reviewing the command; the peer must be ready to start its client:
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-server 1 --apply \
  | tee .local/spark-setup/evidence/rail-1-server.txt
)
```

On its direct peer, in the second terminal:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
mkdir -p .local/spark-setup/evidence
# Direct peer, matching rail's row number in THAT node's local config:
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-client 1 \
  | tee .local/spark-setup/evidence/rail-1-client-preview.txt
# After reviewing the command; its matching server must be running:
bash ops/scripts/spark-setup/verify.sh "$CFG" --bw-client 1 --apply \
  | tee .local/spark-setup/evidence/rail-1-client.txt
)
```

Repeat with endpoints reversed, then for every row on every cable. Row numbers
are local; a ring neighbor's corresponding row may be 3 rather than 1. Retain
each node/rail/direction result privately. A timeout or any FAIL blocks INSTALL.md.
Applied prerequisite checks and every bandwidth direction must pass before
declaring this network ready. After a maintenance reboot, repeat prerequisite and rail checks
to establish persistence. **UNVERIFIED: all these traffic/reboot checks; this
guide was checked on a Mac only.** The earlier notes recorded about 196 Gb/s with both halves
running concurrently on one cable at MTU 9000; that historical aggregate is not
the expected result of this single-rail smoke test. See
[upstream perftest usage](https://github.com/linux-rdma/perftest/blob/master/README).

If jumbo ping fails, compare both endpoints' subnet, cable/rail pairing, carrier
and MTU before RDMA debugging. If the provider opens but registration fails,
check the actual account's group membership and memlock, then the kernel note.
If a local firewall is enabled, review its rules for this isolated fabric:
allow ICMP, node SSH, the selected test TCP ports, RoCE-v2 UDP 4791 and Vade's
TCP 29960/29961 between directly cabled peers on both rails. Do not disable the
management firewall globally. Firewall changes are manual and **UNVERIFIED**.

Once checks pass, follow [INSTALL.md](../INSTALL.md) for compilers, weights,
builds, the two-rank launch and first HTTP answer. A ring's OS network readiness
does not qualify a new rank count; the documented Vade canary used two adjacent
Sparks with a direct link.

## 9. Finish benchmarking and restore updates

After the [installation and timing benchmark](../INSTALL.md#6-run-the-timing-benchmark),
run `vade down` and `vade status` on the Mac with the same private
`VADE_CONFIG`. Confirm both owned ranks have exited and both GPUs are empty
with `nvidia-smi` in your existing Spark sessions. If cleanup is incomplete,
preserve the ownership records and follow [CLI recovery](../cli/README.md#recovery).
Finish any other reserved benchmark work before restoring updates.

Only after that shutdown, run this separate block on each Spark. Restoration
may start pending package jobs; investigate any failure before continuing.

```sh
(
set -euo pipefail
sudo bash ops/scripts/spark-setup/updates.sh restore
# After reviewing the restore commands:
sudo bash ops/scripts/spark-setup/updates.sh restore --apply
)
```

## Verification status

On the inspected working Sparks, verbs queries reported ACTIVE Ethernet ports and
RDMA MTU 4096. Ordinary SSH sessions had unlimited soft/hard memlock; the SSH
system service's recorded limits were still 8 MiB. This guide renders a new
netplan file instead of recovering an earlier `40-cx7.yaml`.

The complete blank-OS path, package installation, new network activation, device
permission changes, new key setup, update pause/restore and reboot persistence
remain **UNVERIFIED on hardware**. Confirm cabling after reinstall, and use the
per-node verification receipts and the Vade install canary to establish the
restored setup.
