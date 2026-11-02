# Prepare a fresh DGX OS Spark for Vade

> **New to this?** Start with [Hardware and network setup](network-setup.md).
> It covers what to buy, cabling diagrams, an example address scheme and the
> link numbers to expect, in plain words. This page is the full command
> reference.

If you link Sparks, do this system setup on every Spark before
[INSTALL.md](../INSTALL.md). One Spark on its own needs none of it. You will:

- cable the Sparks
- give each half of every QSFP link its own network
- let the Vade account use RDMA
- check the session that will launch Vade

A pair needs one direct cable. A ring needs a cable from each Spark to each of
its two neighbors. Keep the ordinary LAN connection for administration.

Converting two existing pairs that each have two cables? Use
[Prepare four Sparks](four-sparks.md) instead. It gives the exact cable moves,
the templates for the two ends of a path, and the limits on what to change in
a fabric that is already configured. The network step below only sets up fresh
interfaces, so it does not overwrite that fabric.

> **Not yet tested from a blank DGX OS.** We wrote this page by reading working
> Sparks without changing them, together with NVIDIA's documentation and checks
> of the scripts themselves. Nothing was changed, rebooted or bandwidth-tested
> while writing it. A network that already works shows what you are aiming
> for, but it does not prove these commands will rebuild it. Our one earlier
> test install of Vade kept the OS settings it found, and its out-of-date
> record has been [removed](../bench/receipts/README.md).

## Before you start

These must already work:

- initial DGX OS setup and the NVIDIA driver
- a normal administrator account
- Git and access to the repository

A few ground rules:

- Use DGX OS packages and NVIDIA's supported firmware. These scripts do not
  install an alternative driver.
- Make later system changes from the local console, in an idle maintenance
  window.
- Create the private configuration as the account that will run Vade. That
  account needs administrator access for the system steps.
- The `sudo` commands here are instructions for you, the operator. They were
  not run while writing this guide.

### Words this page uses

| Term | Meaning here |
| --- | --- |
| [Rail](GLOSSARY.md#pr) | One half of a QSFP port: one Ethernet interface plus its matching RoCE device |
| [RoCE](GLOSSARY.md#pr) | RDMA carried over Ethernet; the Sparks' QSFP links use RoCE v2 |
| [GID](GLOSSARY.md#dg) | An address entry on a RoCE device; each rail needs the one that holds its IPv4 address |
| [Memlock](GLOSSARY.md#ho) | The limit on host memory a process may lock; RDMA registration needs it unlimited |
| Transaction | The private record a step saves before it changes files, so you can roll the change back |

### How the command blocks behave

- **Run each complete block from the repository root in Bash.** Each block runs
  in its own subshell with failure handling on. A failed command stops that
  block, even if your terminal has no `set -e` setting. If anything fails,
  stop and investigate before you run another block.
- **Every setup step previews first.** A plain run prints the commands it would
  run. Steps that edit configuration files also show the file differences.
  A preview writes no files and changes no system settings. Add `--apply`
  only after you have reviewed it. This also applies to local examples and
  rendered files.
- **Verification works the same way.** By default it only previews its checks.
  `--apply` runs them, including peer SSH, ping and bandwidth traffic.
- **Console commands stay manual.** Commands such as `netplan try` are separate
  actions for you to run.
- **Activation follows only a real change.** Steps that activate the network or
  restart services run only after an `--apply` succeeds and reports `changed`.
  If nothing needed changing, the step reports `unchanged` and activation is
  skipped. A preview or a failed apply never activates anything. The result
  files under `.local/` record which of these happened. Keep them until you
  have checked the step.

## 1. Make a private cable map

### How a QSFP port shows up

One QSFP port shows up as **two Ethernet interfaces and two matching RoCE
devices**, on separate PCIe paths. Each interface and its device make one
rail. So two Sparks on one cable need two rows of configuration on each Spark,
and each Spark in a ring needs four.

Configure both rails with Ethernet MTU 9000. The RDMA path MTU that verbs
reports should be 4096.

NVIDIA's
[ConnectX-7 table](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
lists which interface goes with which device. Confirm it with `ibdev2netdev`
or `rdma link show`. Port stickers and old left/right descriptions are not
enough.

### Choose the addresses

Check the LAN, VPN and all routing tables on **every** Spark. Then choose an
unused RFC 1918 (private) address block, kept separate from existing networks.

- Give each rail of each physical cable its own `/24`.
- Use the same subnet at both ends, with host numbers 1 and 2.
- A pair uses two subnets. A ring of N Sparks uses 2N.
- These direct links need no gateway or forwarding.

| Connection | Rail 0 | Rail 1 |
| --- | --- | --- |
| Pair's single cable | First unused subnet, endpoints 1/2 | Second unused subnet, endpoints 1/2 |
| Ring's next cable | Next unused subnet, endpoints 1/2 | Following unused subnet, endpoints 1/2 |

### Name each peer rail

Write down which port is actually connected at each end, because a cable may
join different port numbers.

Give **each peer rail** its own local alias. Use each direct neighbor's rail-0
alias as its usual peer name, and give rail 1 a separate alias for checks.
Keep these apart from the OS hostname and the management addresses.

In a ring, a peer's fabric address belongs to one cable only. So the same peer
alias can point at different addresses on different Sparks, and that is
normal. Sparks in a ring that are not cabled to each other use LAN SSH. This
recipe does not route the fabric between them.

### Generate and fill in the config

Generate the examples **under the ignored `.local/` folder**, then edit them
for this Spark:

```sh
(
set -euo pipefail
bash scripts/spark-setup/init-config.sh pair .local/spark-setup/node
# After reviewing the placeholder files and commands:
bash scripts/spark-setup/init-config.sh pair .local/spark-setup/node --apply
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
shell code that the scripts trust, so keep it local and put only values you
trust in it:

| Setting | Fill with |
| --- | --- |
| `SERVICE_USER` | The account INSTALL.md uses |
| `SERVICE_UNIT` | Optional: an existing Vade systemd service. Leave it empty for the guide's SSH launches. |
| `SSH_KEY_FILE` | That account's own key path. The setup scripts do not read or hand out private keys. |
| `KERNEL_FIXED` | Your explicit approval, backed by evidence, of one exact kernel (see step 5) |

`links.tsv` has one row per rail, with seven fields separated by whitespace:

| Field | Fill with |
| --- | --- |
| `netdev` | This rail's Ethernet interface from `ip link` |
| `rdma_device` | Its verbs device from `ibdev2netdev` or `rdma link show` after package installation |
| `local_ipv4/24` | This node's chosen rail address and prefix |
| `peer_ipv4` | The directly cabled peer's address on the same rail |
| `peer_alias` | A unique local name for that peer rail |
| `ipv4_rocev2_gid` | The local GID index matching that interface and IPv4 address |
| `test_tcp_port` | A free TCP port from 1024 to 65535, matching the peer's row for this rail |

A few rules for the rows:

- On this Spark, every row needs its own device, subnet, alias and test port.
- The generator's test ports are only suggestions. Change the peer's row for
  the same rail to match, even if that row has a different number.
- The addresses do not exist yet, so use GID index `0` **only as a staging
  placeholder**. You replace it with the real IPv4 RoCE-v2 index after step 3,
  and verification rejects a GID that does not match.
- Real names, addresses and filled examples never belong in Git.

The next block lists each RDMA device with its network interface. Use it to
fill in device names if `ibdev2netdev` is not installed yet. It then confirms
that Git ignores your config, and renders the netplan file and hosts fragment
for you to review:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
for d in /sys/class/infiniband/*; do
  printf '\n%s\n' "${d##*/}"
  ls "$d/device/net"
done
git check-ignore .local/spark-setup/node/config.env
bash scripts/spark-setup/render.sh "$CFG" .local/spark-setup/rendered
# After reviewing the file differences:
bash scripts/spark-setup/render.sh "$CFG" .local/spark-setup/rendered --apply
cat .local/spark-setup/rendered/40-cx7.yaml
cat .local/spark-setup/rendered/hosts.fragment
)
```

## 2. Install RDMA tools from the OS

Use the distribution's own libibverbs loader, Mellanox providers, verbs
utilities and `perftest`. The package step installs them, along with SSH,
Netplan, NetworkManager and diagnostic tools.

Vade does RDMA through host memory. The Spark does not support GPUDirect RDMA
([NVIDIA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)).

Install while updates are enabled and the package manager is idle. The step is
safe to repeat, but it has not yet been tested on a fresh image.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
sudo bash scripts/spark-setup/apply.sh "$CFG" packages
# After reviewing the package commands:
sudo bash scripts/spark-setup/apply.sh "$CFG" packages --apply
if command -v ibdev2netdev >/dev/null; then ibdev2netdev; else rdma link show; fi
ibv_devinfo
ldconfig -p | awk '$1 == "libibverbs.so.1" { found=1 } END { exit !found }'
ls -l /dev/infiniband
)
```

## 3. Install and activate the fabric network

The rendered netplan file uses NetworkManager **per fabric interface**, with
static addresses and MTU 9000. It leaves the management interface out. Both
ends must be configured before a link can pass its checks.

These scripts prepare unused interfaces on a fresh image. If a fabric already
works, keep it and go to the verification step. The recipe never converts,
replaces or deletes an existing configuration to make itself fit.

### What the step checks first

Before writing anything, the network step checks:

- the live addresses and routes
- which active and saved NetworkManager profiles own each interface
- Netplan's effective definitions, including `match:` rules

It refuses a fabric that is already configured, and any rail that carries your
current SSH connection. It also stops if ownership or SSH connection details
are missing or unclear.

If it cannot tell where your SSH session comes from, run it from the local
console. The script checks that the console is real. Over SSH, pass
`SSH_CONNECTION` through sudo as the block below does.

If the desired files are already exactly in place, the step only reads. It
does not regenerate or activate the network.

The helpers are shell scripts that read the text output of system tools. If
that output does not let them prove who owns an interface, or that your SSH
session is safe, they stop and say what an administrator needs to check. Do
not work around that, and do not simplify a working configuration just to get
past it.

### What it supports, and what stops it

The helper supports IPv4 management SSH or a real Linux console, and simple
unrelated Ethernet definitions. These need separate review:

- SSH sessions that arrive over IPv6
- Netplan `match:` or `set-name:` rules
- virtual-device sections that are not empty
- NetworkManager selectors the helper cannot interpret

If a working fabric has more complex ownership, keep it and verify it rather
than converting it with this step. Conflicts also stop the step:

- **A destination file already exists with different contents.** The step
  stops, even if that file seems to define only these rails. Keep the file and
  ask the administrator to prepare the fabric separately.
- **A conflicting profile exists.** Have the administrator review that
  interface's configuration separately.
- **Conflicting hosts aliases exist outside the managed block.** Installation
  stops.

Keep the LAN and Wi-Fi profiles, including any `90-NM-*` profiles DGX OS may
have. Avoid broad configuration dumps, which might reveal Wi-Fi credentials.

### Stage the files, then try the network

Once every preflight check passes, `--apply` first saves a transaction. It
records the files as they were just before the change, including which files
and directories did not exist, under
`/var/lib/vade-spark-setup/transactions/`. It prints the transaction's path;
keep it for rollback.

Then it installs `40-cx7.yaml` and replaces only its own marked block of hosts
entries. This only stages the files on disk. The network changes later, when
you run `netplan try` from the console, which generates and activates the
candidate network.

If the apply fails, it restores the recorded prior state automatically. Keep
the transaction, and check any reported recovery failure before you go on.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
sudo --preserve-env=SSH_CONNECTION bash scripts/spark-setup/apply.sh "$CFG" network
# After reviewing the proposed commands and file differences:
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/network-result.XXXXXX")
sudo --preserve-env=SSH_CONNECTION bash scripts/spark-setup/apply.sh "$CFG" network \
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
and both rails with `ip -br address`, `ip link` and a ping to the peer. Accept
only if they work.

On a timeout or cancel, **check that rollback really restored the live
network**, because Netplan documents some rollback caveats. A failed trial
leaves the installed files in place, and Netplan does not roll back
`/etc/hosts`. Restore the setup transaction before rebooting, using the
network commands in [Undo a setup step](#undo-a-setup-step). See
[netplan try](https://netplan.readthedocs.io/en/stable/netplan-try/) and the
[YAML reference](https://netplan.readthedocs.io/en/stable/netplan-yaml/).

### Record each rail's real GID

Once the rail addresses exist, find the IPv4 RoCE-v2 GID for each configured
rail. Put it in `links.tsv` in place of the staging index. The right entry:

- maps to the chosen Ethernet interface
- has type `RoCE v2`
- holds that rail's IPv4 address as an IPv4-mapped IPv6 GID

Do not copy an index from another OS release or from an unrelated interface.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source scripts/spark-setup/common.sh
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

To register memory for RDMA, the account needs two things: permission to open
the rail's `/dev/infiniband/uverbs*` device, and permission to lock host
memory. The `rdma` step:

- creates a dedicated access group and adds the configured account to it
- installs udev rules
- sets both the soft and hard memlock limits to unlimited for that account
- stages an unlimited limit for the SSH service and, if configured, the
  existing Vade service
- checks that SSH uses PAM and that its PAM stack includes `pam_limits.so`; it
  does not rewrite authentication settings

These requirements come from
[RDMA-core](https://github.com/linux-rdma/rdma-core/blob/master/Documentation/libibverbs.md),
[Linux-PAM](https://github.com/linux-pam/linux-pam/blob/master/modules/pam_limits/limits.conf.5.xml)
and [systemd](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml).
To refresh the device, the step uses
[udevadm's sysname matching](https://github.com/systemd/systemd/blob/v255/man/udevadm.xml),
which finds the actual RDMA device instead of guessing its device-node path.

Before writing, the step validates the account, the group, the PAM and SSH
settings and any configured service. Its transaction records:

- the original group membership, and whether the group existed
- the prior file contents, or that a file was absent
- the existing RDMA device ownership and permissions

If the step fails, it restores that state. For a manual recovery later, use
the RDMA commands in [Undo a setup step](#undo-a-setup-step).

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source scripts/spark-setup/common.sh
load_config "$CFG"
sudo bash scripts/spark-setup/apply.sh "$CFG" rdma
# After reviewing the proposed files, group changes and reload commands:
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/rdma-result.XXXXXX")
sudo bash scripts/spark-setup/apply.sh "$CFG" rdma \
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

### Check the new session

Log out and back in as `SERVICE_USER`, then go back to the repository root to
check the new session:

```sh
(
set -euo pipefail
id
ulimit -Sl                         # unlimited
ulimit -Hl                         # unlimited
)
```

Existing SSH shells, tmux servers and user service managers keep the limits
and groups they inherited, so start a new session after the change. The
`changed` branch above restarts a configured Vade service, so keep that
service idle while it happens. If you configured one, check its actual process
too:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source scripts/spark-setup/common.sh
load_config "$CFG"
if [[ -n $SERVICE_UNIT ]]; then
  systemctl show "$SERVICE_UNIT" -p User -p LimitMEMLOCK -p LimitMEMLOCKSoft
  pid=$(systemctl show "$SERVICE_UNIT" -p MainPID --value)
  [[ $pid =~ ^[1-9][0-9]*$ ]] || { printf 'Service has no live process.\n' >&2; exit 1; }
  grep 'Max locked memory' "/proc/$pid/limits"
fi
)
```

Check the **exact shell you will use to launch Vade**:

- A Tailscale SSH route, or another way in that skips PAM, may still give a
  small hard limit. Use ordinary OpenSSH over the LAN or fabric.
- Running verification under sudo would check root, not the service account.
- A PASS in an SSH shell does not qualify a systemd launch.

Fresh permissions, daemon restarts and unit behavior are not yet tested on
real Sparks.

## 5. Check the kernel before registering memory

NVIDIA's [advisory](https://forums.developer.nvidia.com/t/dgx-spark-update-advisory/383254)
identifies `7.0.0-1019-nvidia` as affected by a multi-node RoCE regression,
and publishes `kho=off` as the workaround. NVIDIA began rolling out mitigation
updates September 17, 2026, but we have not yet confirmed any particular fixed
build. Do not assume a fix from a higher version number.

### On any other kernel

Check its exact build against NVIDIA's release or advisory evidence. Record
that approved kernel's exact `uname -r` in your local `KERNEL_FIXED`, then run
the RDMA checks. That local approval lets the checks run. A PASS on the kernel
choice alone does not show that a replacement kernel works.

### On the affected build

First look at `/proc/cmdline`. Then run the optional `kernel` step. Before
installing anything, it checks the effective GRUB command line for a
conflicting `kho=` option. Then it:

- saves the prior files and stages a small GRUB drop-in
- generates a private candidate with
  [`grub-mkconfig`](https://www.gnu.org/software/grub/manual/grub/html_node/Invoking-grub_002dmkconfig.html)
- checks its syntax with
  [`grub-script-check`](https://www.gnu.org/software/grub/manual/grub/html_node/Invoking-grub_002dscript_002dcheck.html)
- records its bytes before publishing it

Its record holds the prior contents or absence of each file and the generated
`/boot/grub/grub.cfg`. On failure, it restores the files as they were just
before. It does not reboot.

The backups and live command lines we checked already had NVIDIA's own KHO
drop-in, so the new drop-in avoids adding a second `kho=off`. Two things are
not yet tested: generating GRUB from scratch, and the setting surviving a
reboot.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
uname -r
cat /proc/cmdline
# Only for 7.0.0-1019-nvidia when preparing this mitigation:
sudo bash scripts/spark-setup/apply.sh "$CFG" kernel
# After reviewing the GRUB file difference and generation command:
sudo bash scripts/spark-setup/apply.sh "$CFG" kernel --apply
# Reboot manually from the console during the agreed maintenance window.
# After the reboot:
uname -r
cat /proc/cmdline                  # affected build must contain kho=off
)
```

To undo it, use the kernel commands in [Undo a setup step](#undo-a-setup-step).
They restore the exact saved drop-in and GRUB bytes, including after an
interrupted install. Rollback does not regenerate GRUB, and no helper reboots
the Spark.

### What `kho=off` does not cover

`kho=off` works around NVIDIA's RoCE regression. It does not prove that every
memory registration is safe under load. A short `ib_write_bw` run does not
test registration under the memory pressure of a loaded model either.

An older build had a problem registering RDMA memory after the model loaded.
Current builds register and seal that memory before loading the weights, so
the old result does not apply. If you test a build, keep its startup log as
your record, and use [INSTALL.md](../INSTALL.md) for the current launch.

Earlier testing also set `vm.compaction_proactiveness=0` to reduce stalls seen
at the time. RDMA does not need it, and these scripts leave it alone. Nobody
has tested whether a fresh OS needs it, so bringing it back needs its own test.

## 6. Establish SSH between the Sparks

Every Spark needs its own account key and trusted peer host keys. From each
service account:

- Use its configured key path, and create a key only if there is none. Keep
  private keys on the Spark where they were made, and do not reuse the old
  shared private key.
- Copy only the public key to each directly cabled neighbor.
- Before you accept a peer's SSH host-key fingerprint, compare it with that
  peer's console output. Do this both for the alias and for the literal peer
  IPv4 address, because verification connects to that address and binds the
  local rail.
- Do not turn off host-key checking. After a reinstall, remove an obsolete
  known-host entry only once you have confirmed the new identity.

These steps are not yet tested on newly installed Sparks. For the ordinary SSH
steps, follow NVIDIA's [pair playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
or [ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring).

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
source scripts/spark-setup/common.sh
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

Then:

- For unattended launches, unlock a passphrase-protected key in the account's
  SSH agent.
- Check both directions and all rail aliases.
- The first authenticated connection records the host key for the alias or
  address. After that, verification requires the IP entry.
- Management SSH between Sparks in a ring that are not cabled to each other
  needs its own trusted names and keys.

INSTALL.md's SSH config names the sending machine's private key with an
`IdentityFile` line (see
[Check SSH from the controller](../INSTALL.md#check-ssh-from-the-controller)).
On a Spark prepared here, use this `SSH_KEY_FILE` path. On a
different source machine, use that machine's own authorized key. A peer check
with this custom key does not make OpenSSH pick it automatically anywhere
else.

## 7. Pause updates only for the benchmark window

Finish approved OS and package updates first. Then the helper:

- records the original enabled and active states, once
- masks the APT daily timers and services and the unattended-upgrades service
- refuses to interrupt APT daily jobs that are running
- skips units that do not exist

Pausing again keeps the original restore state. Restore the recorded state
[after benchmark shutdown](#9-finish-benchmarking-and-restore-updates). Masks
that existed before stay masked.

The saved state lives privately under `/var/lib/vade-spark-setup/`. Before
restoring, the helper checks that the unit files, and whether each one is
enabled, have not changed since the pause. Snapshots from older versions of
the helper store their checksums in a different format. The helper keeps them
but will not convert them, so an administrator has to review them. Do not
delete them during cleanup.

If unattended-upgrades is already running, the helper accepts it only when its
single command matches the supported Ubuntu shutdown helper and output format.
Any other command, or an unclear service description, stops the pause before
anything is written. Have the administrator review it separately.

The helper does not control NVIDIA Dashboard OTA, manual package commands or
other custom updaters. We have not tested how they behave during a pause.

```sh
(
set -euo pipefail
sudo bash scripts/spark-setup/updates.sh pause
# After reviewing the saved-state and systemctl commands:
sudo bash scripts/spark-setup/updates.sh pause --apply
bash scripts/spark-setup/updates.sh status
)
```

## 8. Verify every rail and the actual launch session

### Check the prerequisites

Run the default check as the configured service account, in a fresh ordinary
SSH shell. It previews the checks and peer commands without running them.
After reviewing, add `--apply` to get PASS/FAIL for:

- the architecture and driver
- the verbs library and device access
- current soft and hard memlock
- the kernel choice
- addresses, carrier and MTUs
- RoCE-v2 GIDs and peer naming
- jumbo ping and trusted peer SSH
- paused update units

If you configured a service, its unit and process limits are checked too.

The GID check accepts only the kernel's standard form for an IPv4 address
(eight full hexadecimal groups). Anything else counts as a failure. The script
exits nonzero if any prerequisite fails. This mode does not test bandwidth,
even when everything passes; the next check does.

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
mkdir -p .local/spark-setup/evidence
bash scripts/spark-setup/verify.sh "$CFG" \
  | tee .local/spark-setup/evidence/prerequisites-preview.txt
# After reviewing the proposed checks and peer commands:
bash scripts/spark-setup/verify.sh "$CFG" --apply \
  | tee .local/spark-setup/evidence/prerequisites.txt
)
```

### Test bandwidth on each rail

Test each rail of each cable with both ends idle, and reserved as your
operations policy requires. The test:

- uses host memory and a 64 KiB transfer size for five seconds
- has a 25-second process timeout
- uses RDMA MTU 4096 and the configured device, GID and source IPv4
- prints the raw perftest rate in **Gb/s**

It prints a PASS only if the run completes with positive average bandwidth.
That PASS proves the link connects, not that it meets a throughput target. Use
matching test ports, and the same perftest versions and options, at both ends.

In two terminals, start the server first, then its matching client right
after:

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
mkdir -p .local/spark-setup/evidence
# Node at one end, its local row number (1-based):
bash scripts/spark-setup/verify.sh "$CFG" --bw-server 1 \
  | tee .local/spark-setup/evidence/rail-1-server-preview.txt
# After reviewing the command; the peer must be ready to start its client:
bash scripts/spark-setup/verify.sh "$CFG" --bw-server 1 --apply \
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
bash scripts/spark-setup/verify.sh "$CFG" --bw-client 1 \
  | tee .local/spark-setup/evidence/rail-1-client-preview.txt
# After reviewing the command; its matching server must be running:
bash scripts/spark-setup/verify.sh "$CFG" --bw-client 1 --apply \
  | tee .local/spark-setup/evidence/rail-1-client.txt
)
```

Repeat with the ends swapped, then for every row on every cable. Row numbers
are local, so a ring neighbor's matching row may be 3 rather than 1. Keep each
node/rail/direction result privately.

On a timeout or any FAIL, stop: do not go on to INSTALL.md. The applied
prerequisite checks and every bandwidth direction must pass before you call
this network ready. After a maintenance reboot, repeat the prerequisite and
rail checks to show that the setup persists.

**We have not yet run any of these traffic or reboot checks on real Sparks.**
So far this guide has only been checked on a Mac.
[Hardware and network setup](network-setup.md#numbers-we-measured) records
196.04 Gb/s one way per cable and about 109 Gb/s one way for one rail
alone, from our cable tests of 2026-09-22 (setup record *Vade Timeline 02:
2026-09-22 to 09-24*, not in the repository). This smoke test runs
one rail at a time and only checks for a positive rate, so do not hold it to
the per-cable figure. See
[upstream perftest usage](https://github.com/linux-rdma/perftest/blob/master/README).

### If a check fails

- **Jumbo ping fails.** Compare both ends' subnet, cable/rail pairing, carrier
  and MTU before you debug RDMA.
- **The provider opens but registration fails.** Check the actual account's
  group membership and memlock, then the [kernel note](#5-check-the-kernel-before-registering-memory).
- **A local firewall is on.** Review its rules for this isolated fabric. Allow
  ICMP, node SSH, the selected test TCP ports, RoCE-v2 UDP 4791 and Vade's two
  TCP transport ports (the `tp_port` setting, 29960 by default, and
  `tp_port + 1`) between directly cabled peers on both rails. Do not turn off the
  management firewall globally. Firewall changes are manual and not yet
  tested.

### Next: install Vade

Once the checks pass, follow [INSTALL.md](../INSTALL.md) for the tools, the
code and the model, then
[Two and more Sparks](../INSTALL.md#two-and-more-sparks) for the multi-Spark
launch.

Passing these OS checks on a ring does not make it a tested Vade setup. The
one documented Vade test install used two neighboring Sparks on a direct
cable.

## 9. Finish benchmarking and restore updates

When you have finished with Vade and any benchmark runs, run `vade down` and
then `vade status` on the controller, with the same private `VADE_CONFIG`. Use `nvidia-smi` in your existing Spark sessions to confirm
that both owned ranks have exited and both GPUs are empty. If cleanup is
incomplete, keep the ownership records and follow
[CLI recovery](../cli/README.md#recovery). Finish any other reserved benchmark
work before restoring updates.

Only after that shutdown, run this separate block on each Spark. Restoring may
start pending package jobs, so look into any failure before you continue.

```sh
(
set -euo pipefail
sudo bash scripts/spark-setup/updates.sh restore
# After reviewing the restore commands:
sudo bash scripts/spark-setup/updates.sh restore --apply
)
```

## Undo a setup step

The network, RDMA and kernel steps each save a transaction when you run them
with `--apply`. Rolling back with it restores the immediate prior setup:

- Use the transaction path printed by the matching `--apply`, on the same
  Spark.
- If later transactions changed the same files, roll those back first.
- Review the rollback's file differences before applying it. Keep any
  unrelated edits made since setup rather than overwriting them.
- Transactions stay private and must not be deleted as cleanup.
- Package installation has no automatic rollback. Check the package manager's
  transaction history separately.

### Network

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# Network rollback, from the console:
read -r -p 'Network transaction path printed by apply: ' SNAPSHOT
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/network-rollback-result.XXXXXX")
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" \
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

### RDMA access and limits

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# RDMA rollback, from the console with a recovery session retained:
source scripts/spark-setup/common.sh
load_config "$CFG"
read -r -p 'RDMA transaction path printed by apply: ' SNAPSHOT
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
mkdir -p .local/spark-setup
RESULT_DIR=$(mktemp -d "$PWD/.local/spark-setup/rdma-rollback-result.XXXXXX")
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" \
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

### Kernel

```sh
(
set -euo pipefail
CFG="$PWD/.local/spark-setup/node/config.env"
# Kernel rollback, before the next maintenance reboot:
read -r -p 'Kernel transaction path printed by apply: ' SNAPSHOT
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT"
sudo bash scripts/spark-setup/apply.sh "$CFG" rollback "$SNAPSHOT" --apply
# The helper restores/removes the drop-in and grub.cfg exactly as recorded.
# Reboot manually from the console only during the agreed maintenance window.
uname -r
cat /proc/cmdline
)
```

Removing a mitigation file does not change the running kernel. After a reboot,
confirm that `/proc/cmdline` matches the restored configuration.

Rollback cannot take back groups or limits that existing processes have
already inherited. That needs new sessions and restarts of idle services. Look
into any failed reload or regeneration before you use the machine.

## Verification status

On the working Sparks we inspected, verbs queries reported ACTIVE Ethernet
ports and RDMA MTU 4096. Ordinary SSH sessions had unlimited soft and hard
memlock, but the SSH system service's recorded limits were still 8 MiB. This
guide renders a new netplan file rather than recovering an earlier
`40-cx7.yaml`.

These all remain **untested on hardware**:

- the complete path from a blank OS
- package installation
- new network activation
- device permission changes
- new key setup
- update pause and restore
- reboot persistence

Confirm the cabling after a reinstall. Then run the verification step again
on every Spark, keep its saved results, and do a test install of Vade to
confirm the restored setup.
