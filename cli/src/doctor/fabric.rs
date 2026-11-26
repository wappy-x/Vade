//! `vade doctor --fabric`: check every configured cable on every Spark. Three
//! SSH calls per Spark: one reads its IPv4 interfaces (speed, MTU, RDMA
//! device), one pings the far end of each rail of each cable twice (once with
//! a full 9000-byte frame), and one reads its clock. The checks below only read
//! that output; they change no setting. With `--ring`, two more rows expect
//! one closed ring of every machine at one line rate.
use crate::{checks::Report, config::Config, runner::{self, Runner, quote}, topology};
use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::time::{SystemTime, UNIX_EPOCH};

/// Slowest link that counts as a fabric cable, in Mb/s (one QSFP rail).
pub(crate) const FABRIC_MBPS: u64 = 100_000;
/// The MTU every fabric interface needs (docs/network-setup.md, step 5).
const FABRIC_MTU: u32 = 9000;
/// Largest clock difference to the controller, beyond the SSH round trip.
const CLOCK_SLACK_MS: u64 = 1000;

/// One IPv4 address on one interface of a Spark.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Interface {
    pub name: String,
    pub address: Ipv4Addr,
    pub prefix: u8,
    /// Link speed in Mb/s; 0 when the link is down or unknown.
    pub speed: u64,
    pub mtu: u32,
    /// The RDMA device behind the interface and its port state.
    pub rdma: Option<(String, String)>,
}

impl Interface {
    pub(crate) fn subnet(&self) -> (Ipv4Addr, u8) { (network(self.address, self.prefix), self.prefix) }
    fn holds(&self, address: Ipv4Addr) -> bool { network(address, self.prefix) == network(self.address, self.prefix) }
}

fn network(address: Ipv4Addr, prefix: u8) -> Ipv4Addr {
    let mask = u32::MAX.checked_shl(32 - u32::from(prefix.min(32))).unwrap_or(0);
    Ipv4Addr::from(u32::from(address) & mask)
}

/// What one Spark's probe and ping scripts printed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Probe {
    pub interfaces: Vec<Interface>,
    /// Neighbor address: (small ping answered, 9000-byte ping answered).
    pub pings: BTreeMap<Ipv4Addr, (bool, bool)>,
}

impl Probe {
    fn interface(&self, address: Ipv4Addr) -> Option<&Interface> { self.interfaces.iter().find(|interface| interface.address == address) }
}

/// The script that lists one Spark's IPv4 interfaces.
pub(crate) const PROBE_SCRIPT: &str = r#"set -u
ip -o -4 addr show | while read -r _ name _ cidr _; do
  name=${name%%@*}
  dev=-
  state=-
  for path in /sys/class/net/"$name"/device/infiniband/*; do
    if test -e "$path"; then dev=${path##*/}; break; fi
  done
  if test "$dev" != -; then state=$(sed 's/^[0-9]*: *//' /sys/class/infiniband/"$dev"/ports/1/state 2>/dev/null) || state=-; fi
  speed=$(cat /sys/class/net/"$name"/speed 2>/dev/null) || speed=0
  mtu=$(cat /sys/class/net/"$name"/mtu 2>/dev/null) || mtu=0
  printf 'iface %s %s %s %s %s %s\n' "$name" "$cidr" "${speed:-0}" "${mtu:-0}" "$dev" "${state:--}"
done
"#;

/// The script that pings `targets` from one Spark, small and 9000-byte frames.
pub(crate) fn ping_script(targets: &[Ipv4Addr]) -> String {
    let targets: Vec<String> = targets.iter().map(|address| quote(&address.to_string())).collect();
    format!(r#"set -u
for address in {}; do
  small=fail
  jumbo=fail
  ping -n -c 1 -W 2 "$address" >/dev/null 2>&1 && small=ok
  ping -n -c 1 -W 2 -M do -s 8972 "$address" >/dev/null 2>&1 && jumbo=ok
  printf 'ping %s %s %s\n' "$address" "$small" "$jumbo"
done
"#, if targets.is_empty() { "''".into() } else { targets.join(" ") })
}

/// The script that prints the Spark's clock in nanoseconds and its time sync state.
pub(crate) const CLOCK_SCRIPT: &str = "date +%s%N; timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown";

/// Read [`PROBE_SCRIPT`] or [`ping_script`] output into `probe`. Lines it does
/// not know are ignored.
pub(crate) fn parse_into(probe: &mut Probe, text: &str) {
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            ["iface", name, cidr, speed, mtu, device, state] if plain(name) => {
                let Some((address, prefix)) = cidr.split_once('/') else { continue };
                let (Ok(address), Ok(prefix)) = (address.parse::<Ipv4Addr>(), prefix.parse::<u8>()) else { continue };
                if prefix > 32 { continue; }
                probe.interfaces.push(Interface {
                    name: (*name).into(), address, prefix,
                    speed: speed.parse::<i64>().ok().filter(|speed| *speed > 0).map_or(0, |speed| speed as u64),
                    mtu: mtu.parse().unwrap_or(0),
                    rdma: (plain(device) && *device != "-").then(|| ((*device).into(), if plain(state) { (*state).into() } else { "-".into() })),
                });
            }
            ["ping", address, small, jumbo] => {
                if let Ok(address) = address.parse() { probe.pings.insert(address, (*small == "ok", *jumbo == "ok")); }
            }
            _ => {}
        }
    }
}

/// [`parse_into`] a new probe.
pub(crate) fn parse_probe(text: &str) -> Probe {
    let mut probe = Probe::default();
    parse_into(&mut probe, text);
    probe
}

fn plain(value: &str) -> bool {
    !value.is_empty() && value.len() <= 64 && value.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b':'))
}

/// Check one fabric interface: speed, MTU and, with RDMA selected, an active
/// RDMA device.
fn rail_port(interface: &Interface, rdma: bool) -> Result<String, String> {
    if interface.speed == 0 {
        return Err(format!("{} holds {} but its link is down (no speed reported). Check the cable and that both ends are seated.", interface.name, interface.address));
    }
    if interface.speed < FABRIC_MBPS {
        return Err(format!("{} holds {} but links at {} Mb/s. A fabric link needs 100000 Mb/s or more: check the cable and that this is a QSFP port.", interface.name, interface.address, interface.speed));
    }
    if interface.mtu < FABRIC_MTU {
        return Err(format!("{} has MTU {}. Set MTU 9000 at both ends of the cable.", interface.name, interface.mtu));
    }
    let device = match (&interface.rdma, rdma) {
        (Some((device, state)), true) if state == "ACTIVE" => format!("RDMA {device} active"),
        (Some((device, state)), true) => return Err(format!("RDMA device {device} on {} is {state}, not ACTIVE. Check the cable and carrier at both ends.", interface.name)),
        (None, true) => return Err(format!("{} has no RDMA device. Check ibdev2netdev and the RDMA packages, or set SEQPAR_RDMA=0 for TCP.", interface.name)),
        (Some((device, _)), false) => format!("RDMA {device}; TCP selected"),
        (None, false) => "no RDMA device; TCP selected".into(),
    };
    Ok(format!("{} {}/{}, {} Mb/s, MTU {}, {device}", interface.name, interface.address, interface.prefix, interface.speed, interface.mtu))
}

/// Check the interface that holds `own`, this Spark's address toward `peer`.
pub(crate) fn port(probe: &Probe, own: Ipv4Addr, peer: &str, rdma: bool) -> Result<String, String> {
    let interface = probe.interface(own).ok_or_else(|| format!(
        "No interface on this Spark has {own}, its fabric.{peer} address. Put the address on the QSFP port cabled to {peer}, or fix fabric.{peer}."
    ))?;
    rail_port(interface, rdma)
}

/// The two rails of one cable as (end a, end b) interface pairs. Rail 0 holds
/// the configured addresses. Rail 1 is the subnet that two other interfaces,
/// one at each end, share: fabric-speed ones first, else RDMA ports whose
/// link is down, so [`link`] names the down link. An address alias on the
/// rail 0 interface is not a second rail.
pub(crate) fn rails(a: (&str, Ipv4Addr, &Probe), b: (&str, Ipv4Addr, &Probe)) -> Result<[(Interface, Interface); 2], String> {
    let ((a_name, a_address, a_probe), (b_name, b_address, b_probe)) = (a, b);
    let missing = |name: &str, address| format!("{name} has no interface with {address}. Fix its port check first.");
    let a_zero = a_probe.interface(a_address).ok_or_else(|| missing(a_name, a_address))?;
    let b_zero = b_probe.interface(b_address).ok_or_else(|| missing(b_name, b_address))?;
    if a_zero.subnet() != b_zero.subnet() || !a_zero.holds(b_address) {
        return Err(format!(
            "{a_name} has {a_address}/{} and {b_name} has {b_address}/{}: not one subnet. Give both ends of this cable addresses in one subnet, or fix the fabric entries.",
            a_zero.prefix, b_zero.prefix
        ));
    }
    let other = |probe: &Probe, zero: &Interface| probe.interfaces.iter()
        .filter(|interface| interface.name != zero.name && interface.subnet() != zero.subnet()
            && (interface.speed >= FABRIC_MBPS || (interface.speed == 0 && interface.rdma.is_some())))
        .cloned().collect::<Vec<_>>();
    let b_other = other(b_probe, b_zero);
    let mut pairs = Vec::new();
    for a_one in other(a_probe, a_zero) {
        pairs.extend(b_other.iter().filter(|b_one| b_one.subnet() == a_one.subnet()).map(|b_one| (a_one.clone(), b_one.clone())));
    }
    let up = |pair: &&(Interface, Interface)| pair.0.speed >= FABRIC_MBPS && pair.1.speed >= FABRIC_MBPS;
    let one = pairs.iter().find(up).or_else(|| pairs.first()).cloned();
    let one = one.ok_or_else(|| format!(
        "Only one rail: {a_name} and {b_name} share no second subnet on two other fabric interfaces. Give the second rail of this cable its own subnet at both ends, or the link runs at about half speed."
    ))?;
    Ok([(a_zero.clone(), b_zero.clone()), one])
}

/// Check both rails of one cable on their own: each end's speed, MTU and RDMA
/// device, the same MTU at both ends, and small and 9000-byte pings in both
/// directions. The cable fails when either rail fails.
pub(crate) fn link(a: (&str, &Probe), b: (&str, &Probe), rails: &[(Interface, Interface); 2], rdma: bool) -> Result<String, String> {
    let ((a_name, a_probe), (b_name, b_probe)) = (a, b);
    for (index, (a_end, b_end)) in rails.iter().enumerate() {
        let rail = |error: String| format!("Rail {index}: {error}");
        if a_end.address == b_end.address {
            return Err(rail(format!("{a_name} and {b_name} both hold {}. Give each end of this rail its own address.", a_end.address)));
        }
        rail_port(a_end, rdma).map_err(|error| rail(format!("{a_name}: {error}")))?;
        rail_port(b_end, rdma).map_err(|error| rail(format!("{b_name}: {error}")))?;
        if a_end.mtu != b_end.mtu {
            return Err(rail(format!("MTU differs: {a_name} {}, {b_name} {}. Set MTU 9000 at both ends.", a_end.mtu, b_end.mtu)));
        }
        for (from, to, probe) in [(a_name, b_end.address, a_probe), (b_name, a_end.address, b_probe)] {
            // A Spark answers a ping to its own address over loopback.
            if probe.interface(to).is_some() {
                return Err(rail(format!("{from} holds {to}, the far end's address, itself: its ping proves nothing about the cable. Give each end of this rail its own address.")));
            }
            match probe.pings.get(&to) {
                Some((true, true)) => {}
                Some((true, false)) => return Err(rail(format!("{from} reaches {to}, but a 9000-byte ping fails. Set MTU 9000 on this rail, at both ends."))),
                _ => return Err(rail(format!("{from} gets no ping reply from {to}. Check the cable, the addresses and that ICMP is allowed between neighbors."))),
            }
        }
    }
    let [(zero, _), (one, _)] = rails;
    Ok(format!("rail 0 {}/{} and rail 1 {}/{}: 9000-byte pings answered both ways on both rails", zero.subnet().0, zero.prefix, one.subnet().0, one.prefix))
}

/// A Spark's clock against this computer's, from [`CLOCK_SCRIPT`] output read
/// between `before` and `after` (nanoseconds since 1970).
pub(crate) fn clock(text: &str, before: u128, after: u128) -> Result<String, String> {
    let mut lines = text.lines();
    let spark: u128 = lines.next().and_then(|line| line.trim().parse().ok())
        .ok_or("The Spark did not print its clock. Check that date supports +%s%N.")?;
    let sync = lines.next().map_or("unknown", str::trim);
    let window_ms = (after.saturating_sub(before) / 1_000_000) as u64;
    let (ahead, offset_ms) = if spark >= before { (true, (spark.saturating_sub(after) / 1_000_000) as u64) } else { (false, ((before - spark) / 1_000_000) as u64) };
    if offset_ms > CLOCK_SLACK_MS {
        return Err(format!("Clock is {offset_ms} ms {} this computer (SSH round trip {window_ms} ms). Turn on time sync on this Spark and on this computer.", if ahead { "ahead of" } else { "behind" }));
    }
    if sync != "yes" {
        let sync = if plain(sync) { sync } else { "an unknown state" };
        return Err(format!("Clock is within {CLOCK_SLACK_MS} ms of this computer, but time sync reports {sync}. Turn on time sync on this Spark."));
    }
    Ok(format!("within {CLOCK_SLACK_MS} ms of this computer (SSH round trip {window_ms} ms), time sync on"))
}

fn now() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |time| time.as_nanos()) }

/// The `ring line rate` row: rail `r` of every cable links at the fastest
/// speed that any cable's rail `r` reports. `rates` holds (cable row, rail,
/// Spark, interface) for both ends of both rails of every cable it checked.
fn line_rate(rates: &[(String, usize, String, Interface)], cables: usize) -> Result<String, String> {
    let fastest = |rail: usize| rates.iter().filter(|rate| rate.1 == rail).map(|rate| rate.3.speed).max().unwrap_or(0);
    let slow: Vec<String> = rates.iter().filter(|(_, rail, _, end)| end.speed < fastest(*rail))
        .map(|(row, rail, name, end)| format!("{row} rail {rail}: {name} {} at {} Mb/s", end.name, end.speed)).collect();
    if !slow.is_empty() {
        return Err(format!("Rail 0 of the other cables links at {} Mb/s and rail 1 at {} Mb/s, but {}. Check that cable: use a 200 Gb/s cable and seat both ends.", fastest(0), fastest(1), slow.join("; ")));
    }
    let checked = rates.len() / 4;
    if checked < cables {
        return Err(format!("Only {checked} of {cables} cables have both rails to compare. Fix the failed or skipped cable rows first."));
    }
    Ok(format!("rail 0 at {} Mb/s and rail 1 at {} Mb/s on all {cables} cables", fastest(0), fastest(1)))
}

/// Run every fabric check for every machine and group in `config`. A failed
/// check never stops the rest. `ring` adds the rows for one closed ring.
pub(crate) fn run<R: Runner>(report: &mut Report, runner: &mut R, config: &Config, rdma: bool, ring: bool, dry: bool) -> Result<(), String> {
    report.ok("cabling", &topology::cabling(&config.machines).join("; "))?;
    if ring {
        report.check("ring", topology::check_whole_ring(&config.machines), String::clone,
            "Cable every Spark to its two ring neighbors, give both ends of each cable a fabric entry, then run vade doctor --fabric --ring. See docs/eight-sparks.md.")?;
    }
    for group in &config.groups {
        let machines: Vec<_> = group.hosts.iter().filter_map(|host| config.machines.iter().find(|machine| machine.name == *host).cloned()).collect();
        let label = format!("group {}", group.name);
        let result = topology::check_arc(&format!("Group {}", group.name), &machines);
        report.check(&label, result, |()| topology::arc_text(&machines),
            "Fix the group's hosts order or its fabric entries, then run vade doctor --fabric.")?;
    }
    // A link with only one end in the config is a defect, never a skipped cable.
    for machine in &config.machines {
        for peer in machine.fabric.keys() {
            let Some(other) = config.machines.iter().find(|other| other.name == *peer) else { continue };
            if !other.fabric.contains_key(&machine.name) {
                report.fail(&format!("{} - {peer} cable", machine.name),
                    &format!("{} has fabric.{peer}, but [machine {peer}] has no fabric.{}. A cable needs both ends in the config.", machine.name, machine.name),
                    &format!("Add fabric.{} under [machine {peer}] with {peer}'s own address on that cable, or remove fabric.{peer} from [machine {}].", machine.name, machine.name))?;
            }
        }
    }
    let numeric = |value: &str| value.parse::<Ipv4Addr>().ok();
    let cables: Vec<(usize, usize)> = (0..config.machines.len())
        .flat_map(|a| (a + 1..config.machines.len()).map(move |b| (a, b)))
        .filter(|&(a, b)| topology::linked(&config.machines[a], &config.machines[b])).collect();
    if dry {
        for machine in &config.machines {
            let name = |check: &str| format!("{} {check}", machine.name);
            report.plan(&name("fabric ports"), "read IPv4 interfaces, speed, MTU and RDMA devices")?;
            report.plan(&name("clock"), "compare the Spark's clock with this computer's and read time sync")?;
        }
        for &(a, b) in &cables {
            report.plan(&format!("{} - {} cable", config.machines[a].name, config.machines[b].name),
                "both rails: one subnet each, speed, MTU and RDMA at both ends, small and 9000-byte pings both ways, no shared subnet")?;
        }
        if ring {
            report.plan("ring line rate", "rail 0 and rail 1 of every cable link at one speed each")?;
        }
        return Ok(());
    }
    // 1. Interfaces, and one row per configured port.
    let mut probes: Vec<Probe> = Vec::new();
    // A Spark whose probe did not run has no port facts: skip its port and cable rows.
    let mut probed: Vec<bool> = Vec::new();
    for machine in &config.machines {
        let name = |check: &str| format!("{} {check}", machine.name);
        let probe = match runner.run(&runner::ssh(&machine.ssh, PROBE_SCRIPT.into(), &name("fabric probe"))) {
            Ok(output) if output.success => Some(parse_probe(&output.stdout)),
            _ => {
                report.fail(&name("fabric ports"), "The fabric probe did not run over SSH.", "Check the ssh entry and that ip exists on the Spark, then run vade doctor --fabric.")?;
                None
            }
        };
        for (peer, value) in &machine.fabric {
            let row = name(&format!("port to {peer}"));
            match (numeric(value), &probe) {
                (None, _) => report.skip(&row, "fabric address is a host name; this check reads numeric IPv4 addresses")?,
                (Some(_), None) => report.skip(&row, &format!("the fabric probe of {} did not run; fix its fabric ports row first", machine.name))?,
                (Some(own), Some(probe)) => { report.check(&row, port(probe, own, peer, rdma), String::clone, "Fix this port, then run vade doctor --fabric. See docs/network-setup.md.")?; }
            }
        }
        probed.push(probe.is_some());
        probes.push(probe.unwrap_or_default());
    }
    // 2. Find both rails of each cable; ping the far end of each rail from both ends.
    let mut found = Vec::new();
    let mut targets: Vec<Vec<Ipv4Addr>> = vec![Vec::new(); config.machines.len()];
    for &(a, b) in &cables {
        let (ma, mb) = (&config.machines[a], &config.machines[b]);
        let row = format!("{} - {} cable", ma.name, mb.name);
        let (Some(a_address), Some(b_address)) = (numeric(&ma.fabric[&mb.name]), numeric(&mb.fabric[&ma.name])) else {
            report.skip(&row, "fabric address is a host name; this check reads numeric IPv4 addresses")?;
            continue;
        };
        if let Some(index) = [a, b].into_iter().find(|&index| !probed[index]) {
            report.skip(&row, &format!("the fabric probe of {} did not run; fix its fabric ports row first", config.machines[index].name))?;
            continue;
        }
        match rails((&ma.name, a_address, &probes[a]), (&mb.name, b_address, &probes[b])) {
            Err(error) => report.fail(&row, &error, "Fix this cable or its addresses, then run vade doctor --fabric. See docs/eight-sparks.md.")?,
            Ok(pair) => {
                for (a_end, b_end) in &pair {
                    targets[a].push(b_end.address);
                    targets[b].push(a_end.address);
                }
                found.push((a, b, row, pair));
            }
        }
    }
    for (index, machine) in config.machines.iter().enumerate() {
        let name = |check: &str| format!("{} {check}", machine.name);
        if !targets[index].is_empty() {
            match runner.run(&runner::ssh(&machine.ssh, ping_script(&targets[index]), &name("fabric ping"))) {
                Ok(output) if output.success => parse_into(&mut probes[index], &output.stdout),
                _ => report.fail(&name("fabric ping"), "The ping check did not run over SSH.", "Check the ssh entry and that ping exists on the Spark, then run vade doctor --fabric.")?,
            }
        }
        let before = now();
        let output = runner.run(&runner::ssh(&machine.ssh, CLOCK_SCRIPT.into(), &name("clock")));
        let after = now();
        let result = output.and_then(|output| if output.success { clock(&output.stdout, before, after) } else { Err("The clock check did not run over SSH.".into()) });
        report.check(&name("clock"), result, String::clone, "Turn on time sync (NTP) on the Spark, then run vade doctor --fabric.")?;
    }
    // 3. Check each cable, rail by rail, and that no two cables share a subnet.
    let mut subnets: BTreeMap<(Ipv4Addr, u8), String> = BTreeMap::new();
    let mut rates = Vec::new();
    for (a, b, row, pair) in found {
        for (index, (a_end, b_end)) in pair.iter().enumerate() {
            rates.push((row.clone(), index, config.machines[a].name.clone(), a_end.clone()));
            rates.push((row.clone(), index, config.machines[b].name.clone(), b_end.clone()));
        }
        let mut result = link((&config.machines[a].name, &probes[a]), (&config.machines[b].name, &probes[b]), &pair, rdma);
        for (index, (end, _)) in pair.iter().enumerate() {
            if let Some(other) = subnets.insert(end.subnet(), row.clone()).filter(|_| result.is_ok()) {
                result = Err(format!("Rail {index} uses subnet {}/{}, the same as a rail of the {other}. Give every rail of every cable its own subnet.", end.subnet().0, end.prefix));
            }
        }
        report.check(&row, result, String::clone, "Fix this cable or its addresses, then run vade doctor --fabric. See docs/eight-sparks.md.")?;
    }
    if ring {
        report.check("ring line rate", line_rate(&rates, cables.len()), String::clone,
            "Fix the named rail, then run vade doctor --fabric --ring. See docs/network-setup.md, step 7.")?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "fabric_tests.rs"]
mod tests;
