//! Inspect over SSH; only the owner executes the printed network configuration.
use spark_tp::Result;
use std::{collections::{BTreeMap, BTreeSet}, net::{Ipv4Addr, Ipv6Addr}, thread};

// These commands only read state. Ping is the only traffic this tool generates.
const INVENTORY: &str = r#"set -eu
export LC_ALL=C
printf '@ADDR4\n'
ip -o -4 address show
printf '@ADDR6\n'
ip -o -6 address show scope link
printf '@ROUTES\n'
ip -o -4 route show table all
printf '@PROFILES\n'
nmcli -g NAME connection show
printf '@PORTS\n'
for d in /sys/class/net/*; do
    speed=$(cat "$d/speed" 2>/dev/null) || continue
    [ "$speed" -ge 100000 ] 2>/dev/null || continue
    [ "$(cat "$d/carrier" 2>/dev/null)" = 1 ] || continue
    name=${d##*/}
    mac=$(cat "$d/address")
    mtu=$(cat "$d/mtu")
    uuid=$(nmcli -g GENERAL.CON-UUID device show "$name")
    switch=$(cat "$d/phys_switch_id" 2>/dev/null) || switch=--
    physical=$(cat "$d/phys_port_name" 2>/dev/null) || physical=--
    printf '%s %s %s %s %s %s %s\n' "$name" "$speed" "$mac" "$mtu" "${uuid:---}" "${switch:---}" "${physical:---}"
done
"#;

#[derive(Clone, Copy)]
struct Cidr { ip: Ipv4Addr, prefix: u8 }
impl Cidr {
    fn parse(s: &str) -> Option<Self> {
        let (ip, prefix) = s.split_once('/').unwrap_or((s, "32"));
        let prefix = prefix.parse().ok()?;
        (prefix <= 32).then_some(Self { ip: ip.parse().ok()?, prefix })
    }
    fn bounds(self) -> (u32, u32) {
        let mask = u32::MAX.checked_shl(32 - self.prefix as u32).unwrap_or(0);
        let start = u32::from(self.ip) & mask;
        (start, start | !mask)
    }
    fn overlaps(self, other: Self) -> bool {
        let (a, b) = self.bounds(); let (c, d) = other.bounds();
        a <= d && c <= b
    }
    fn private(self) -> bool {
        let [a, b, _, _] = self.ip.octets();
        a == 10 || a == 172 && (16..=31).contains(&b) || a == 192 && b == 168
    }
    fn display(self) -> String { format!("{}/{}", self.ip, self.prefix) }
}
struct Reservation { subnet: Cidr, device: String }
struct Port {
    name: String, mac: String, physical: String, speed: u64, mtu: u32, uuid: String,
    addresses: Vec<Cidr>, responders: BTreeSet<String>, note: String,
}
struct Host {
    alias: String, ports: Vec<Port>, reserved: Vec<Reservation>,
    addresses: Vec<Ipv4Addr>, local6: BTreeSet<Ipv6Addr>, profiles: BTreeSet<String>,
}

fn remote(host: &str, command: &str) -> Result<String> {
    let output = super::ssh_command().args(["-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no", host, command])
        .output().map_err(|e| format!("{host}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{host}: {}", String::from_utf8_lossy(&output.stderr).trim()));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("{host}: {e}"))
}
fn inspect(alias: &str) -> Result<Host> {
    let inventory = remote(alias, INVENTORY)?;
    let mut host = Host { alias: alias.into(), ports: Vec::new(), reserved: Vec::new(),
        addresses: Vec::new(), local6: BTreeSet::new(), profiles: BTreeSet::new() };
    let mut section = "";
    let mut addr4 = Vec::new();
    for line in inventory.lines() {
        if matches!(line, "@ADDR4" | "@ADDR6" | "@ROUTES" | "@PROFILES" | "@PORTS") {
            section = line; continue;
        }
        let w = line.split_whitespace().collect::<Vec<_>>();
        match section {
            "@ADDR4" => {
                let i = w.iter().position(|s| *s == "inet").ok_or("invalid IPv4 inventory")?;
                let cidr = w.get(i + 1).and_then(|s| Cidr::parse(s)).ok_or("invalid IPv4 address")?;
                let device = w.get(1).ok_or("missing IPv4 interface")?.split('@').next().unwrap().to_string();
                host.addresses.push(cidr.ip);
                host.reserved.push(Reservation { subnet: cidr, device: device.clone() });
                addr4.push((device, cidr));
            }
            "@ADDR6" => {
                if let Some(i) = w.iter().position(|s| *s == "inet6") {
                    if let Some(ip) = w.get(i + 1).and_then(|s| s.split('/').next()).and_then(|s| s.parse().ok()) {
                        host.local6.insert(ip);
                    }
                }
            }
            "@ROUTES" => {
                // All routing tables count, including VPNs and blackhole routes; a default does not reserve all IPv4.
                let dest = w.first().filter(|s| Cidr::parse(s).is_some()).or_else(|| w.get(1));
                if let Some(subnet) = dest.and_then(|s| Cidr::parse(s)).filter(|s| s.prefix != 0) {
                    let device = w.iter().position(|s| *s == "dev").and_then(|i| w.get(i + 1)).unwrap_or(&"").to_string();
                    host.reserved.push(Reservation { subnet, device });
                }
            }
            "@PROFILES" => { host.profiles.insert(line.to_string()); }
            "@PORTS" => {
                if w.len() != 7 { return Err(format!("{alias}: invalid port inventory")); }
                let physical = if w[5] == "--" || w[6] == "--" { w[2].into() } else { format!("{}/{}", w[5], w[6]) };
                host.ports.push(Port { name: w[0].into(), speed: w[1].parse().map_err(|e| format!("{e}"))?,
                    mac: w[2].to_ascii_lowercase(), physical, mtu: w[3].parse().map_err(|e| format!("{e}"))?, uuid: w[4].into(),
                    addresses: Vec::new(), responders: BTreeSet::new(), note: String::new() });
            }
            _ => return Err(format!("{alias}: unexpected inventory output")),
        }
    }
    if section != "@PORTS" { return Err(format!("{alias}: incomplete inventory")); }
    host.ports.sort_by(|a, b| a.name.cmp(&b.name));
    for port in &mut host.ports {
        port.addresses = addr4.iter().filter(|(d, _)| d == &port.name).map(|(_, a)| *a).collect();
        let device = super::quote(&port.name);
        // Do not use -c: the local multicast reply could end ping before the neighbour answers.
        let command = format!("export LC_ALL=C; ping -6 -n -w 3 {} 2>&1; printf '@NEIGH\\n'; ip -6 neigh show dev {device}",
            super::quote(&format!("ff02::1%{}", port.name)));
        let probe = remote(alias, &command)?;
        let (ping, neighbours) = probe.split_once("@NEIGH\n").ok_or("missing IPv6 neighbour inventory")?;
        let mut replies = BTreeSet::new();
        for line in ping.lines() {
            if let Some((_, tail)) = line.split_once(" bytes from ") {
                let address = tail.split_whitespace().next().unwrap_or("").trim_end_matches(':').split('%').next().unwrap();
                if let Ok(ip) = address.parse::<Ipv6Addr>() {
                    if ip.segments()[0] & 0xffc0 == 0xfe80 && !host.local6.contains(&ip) { replies.insert(ip); }
                }
            }
        }
        let mut macs = neighbour_macs(neighbours);
        for ip in &replies {
            if !macs.contains_key(ip) {
                // A multicast echo need not populate the local neighbour cache. Resolve only its actual responders.
                let command = format!("export LC_ALL=C; ping -6 -n -c 1 -w 1 {} >/dev/null 2>&1; ip -6 neigh show dev {device}",
                    super::quote(&format!("{ip}%{}", port.name)));
                macs.extend(neighbour_macs(&remote(alias, &command)?));
            }
        }
        for ip in &replies {
            if let Some(mac) = macs.get(ip) { port.responders.insert(mac.clone()); }
        }
        if replies.is_empty() { port.note = "no remote IPv6 link-local reply; check IPv6 and multicast ICMP".into(); }
        else if replies.iter().any(|ip| !macs.contains_key(ip)) { port.note = "some responders have no neighbour MAC; cabling unresolved".into(); }
    }
    Ok(host)
}
fn neighbour_macs(output: &str) -> BTreeMap<Ipv6Addr, String> {
    output.lines().filter_map(|line| {
        let w = line.split_whitespace().collect::<Vec<_>>();
        if w.contains(&"FAILED") || w.contains(&"INCOMPLETE") { return None; }
        let ip = w.first()?.parse().ok()?;
        let i = w.iter().position(|s| *s == "lladdr")?;
        Some((ip, w.get(i + 1)?.to_ascii_lowercase()))
    }).collect()
}
type Endpoint = (usize, usize);
type Physical = (usize, String);
fn physical(hosts: &[Host], endpoint: Endpoint) -> Physical {
    (endpoint.0, hosts[endpoint.0].ports[endpoint.1].physical.clone())
}
fn shared(a: &Port, b: &Port) -> Option<(Cidr, Cidr)> {
    a.addresses.iter().find_map(|a| b.addresses.iter().find_map(|b|
        (a.prefix == b.prefix && a.bounds() == b.bounds() && a.ip != b.ip).then_some((*a, *b))))
}
fn unused_subnet(reserved: &[Cidr]) -> Result<Cidr> {
    // Enumerate RFC 1918 /24s, skipping whole overlapping routes rather than assuming the LAN is /24.
    for (first, end) in [(10u32 << 24, 11u32 << 24), ((172 << 24) | (16 << 16), (172 << 24) | (32 << 16)),
        ((192 << 24) | (168 << 16), (192 << 24) | (169 << 16))] {
        let mut start = first;
        while start < end {
            let subnet = Cidr { ip: Ipv4Addr::from(start), prefix: 24 };
            let occupied_end = reserved.iter().filter(|s| subnet.overlaps(**s)).map(|s| s.bounds().1).max();
            if let Some(last) = occupied_end { start = (last as u64 + 256 & !255).min(end as u64) as u32; }
            else { return Ok(subnet); }
        }
    }
    Err("no unused private /24 across the supplied hosts' IPv4 addresses and routes".into())
}
fn configure(host: &mut Host, port_index: usize, address: Ipv4Addr) -> Vec<String> {
    let port = &host.ports[port_index];
    let settings = format!("ipv4.method manual ipv4.addresses {} ipv4.gateway '' ipv4.never-default yes 802-3-ethernet.mtu 9000 connection.autoconnect yes",
        super::quote(&format!("{address}/24")));
    let (command, selector) = if port.uuid != "--" && !port.uuid.is_empty() {
        let uuid = format!("uuid {}", super::quote(&port.uuid));
        (format!("sudo nmcli connection modify {uuid} {settings}"), uuid)
    } else {
        let base = format!("vade-qsfp-{}", port.name);
        let mut name = base.clone(); let mut suffix = 1;
        while host.profiles.contains(&name) { name = format!("{base}-{suffix}"); suffix += 1; }
        host.profiles.insert(name.clone());
        (format!("sudo nmcli connection add type ethernet ifname {} con-name {} {settings} ipv6.method link-local",
            super::quote(&port.name), super::quote(&name)), format!("id {}", super::quote(&name)))
    };
    vec![command, format!("sudo nmcli connection up {selector} ifname {}", super::quote(&port.name))]
}
fn jumbo(port: &Port, peer: Ipv4Addr) -> String {
    format!("ping -4 -c 3 -W 2 -M do -s 8972 -I {} {}", super::quote(&port.name), super::quote(&peer.to_string()))
}

pub fn run(aliases: &[String]) -> Result<()> {
    if aliases.len() < 2 { return Err("usage: tp-launch net-plan HOST1 HOST2 ... (SSH aliases from your ~/.ssh/config)".into()); }
    let mut unique = BTreeSet::new();
    for alias in aliases {
        if alias.is_empty() || alias.starts_with('-') || alias.contains(char::is_whitespace) || !unique.insert(alias) {
            return Err("net-plan needs distinct, nonempty SSH aliases without whitespace or leading '-'".into());
        }
    }
    // Each host is independent; no remote binaries, model weights, or build directory are needed.
    let mut hosts = thread::scope(|scope| {
        let tasks = aliases.iter().map(|alias| scope.spawn(move || inspect(alias))).collect::<Vec<_>>();
        tasks.into_iter().map(|task| task.join().map_err(|_| "network inspection thread failed".to_string())?).collect::<Result<Vec<_>>>()
    })?;
    let mut owners = BTreeMap::new();
    for (h, host) in hosts.iter().enumerate() {
        for (p, port) in host.ports.iter().enumerate() {
            if owners.insert(port.mac.clone(), (h, p)).is_some() {
                return Err("the same port MAC appears on multiple hosts; check duplicate aliases".into());
            }
        }
    }
    let mut peers: BTreeMap<Endpoint, Option<Endpoint>> = BTreeMap::new();
    let mut candidates = BTreeMap::new();
    let mut blocked = BTreeSet::new();
    let mut groups: BTreeMap<Physical, Vec<Endpoint>> = BTreeMap::new();
    for (h, host) in hosts.iter().enumerate() {
        for (p, port) in host.ports.iter().enumerate() {
            let found = port.responders.iter().filter_map(|mac| owners.get(mac).copied()).filter(|(peer, _)| *peer != h).collect::<Vec<_>>();
            let configured = found.iter().copied().filter(|&(h, p)| shared(port, &hosts[h].ports[p]).is_some()).collect::<Vec<_>>();
            let choices = if configured.is_empty() { &found } else { &configured };
            let peer = (choices.len() == 1 && port.note.is_empty()).then(|| choices[0]);
            if !port.note.is_empty() || configured.len() > 1 { blocked.insert((h, p)); }
            peers.insert((h, p), peer);
            candidates.insert((h, p), found);
            groups.entry(physical(&hosts, (h, p))).or_default().push((h, p));
        }
    }
    // Spark exposes two PCIe lanes of a physical QSFP port. Multicast reaches both;
    // sysfs groups them without depending on interface names or cable orientation.
    let neighbours = groups.iter().map(|(group, ports)| (group.clone(), ports.iter()
        .flat_map(|port| candidates[port].iter().map(|&peer| physical(&hosts, peer))).collect::<BTreeSet<_>>())).collect::<BTreeMap<_, _>>();
    for (group, members) in &groups {
        let adjacent = &neighbours[group];
        if adjacent.len() != 1 { continue; }
        let other = adjacent.first().unwrap();
        if group >= other || neighbours[other].len() != 1 || !neighbours[other].contains(group) { continue; }
        let available = |members: &[Endpoint]| {
            let mut ports = members.iter().copied().filter(|port| peers[port].is_none() && !blocked.contains(port)).collect::<Vec<_>>();
            ports.sort_by(|&a, &b| hosts[a.0].ports[a.1].mac.cmp(&hosts[b.0].ports[b.1].mac)); ports
        };
        let local = available(members); let remote = available(&groups[other]);
        for (a, b) in local.into_iter().zip(remote) {
            if candidates[&a].contains(&b) && candidates[&b].contains(&a) {
                peers.insert(a, Some(b)); peers.insert(b, Some(a));
            }
        }
    }
    let mut reports = vec!["# Read-only plan. No configuration or jumbo-ping commands below have been run.".to_string()];
    let mut unresolved = false;
    for (h, host) in hosts.iter().enumerate() {
        if host.ports.is_empty() { reports.push(format!("# {}: no QSFP ports with carrier and speed >= 100000 Mb/s", host.alias)); }
        for (p, port) in host.ports.iter().enumerate() {
            let choices = &candidates[&(h, p)];
            let peer = peers[&(h, p)];
            let state = if port.addresses.is_empty() { "unconfigured".into() }
                else { format!("already configured ({}); skipped", port.addresses.iter().map(|a| a.display()).collect::<Vec<_>>().join(", ")) };
            reports.push(format!("# {} {}: carrier, {} Mb/s, MTU {}, {state}", host.alias, port.name, port.speed, port.mtu));
            if let Some((h, p)) = peer { reports.push(format!("#   neighbour: {} {} (responder MAC matched)", hosts[h].alias, hosts[h].ports[p].name)); }
            else if !choices.is_empty() {
                reports.push(format!("#   unresolved or ambiguous lanes: {}; no configuration proposed", choices.iter().map(|&(h, p)| format!("{} {}", hosts[h].alias, hosts[h].ports[p].name)).collect::<Vec<_>>().join(", ")));
                unresolved = true;
            } else {
                reports.push(format!("#   {}", if port.note.is_empty() { "no neighbour among the supplied hosts; include its SSH alias to plan this link" } else { &port.note }));
                if !port.note.is_empty() { unresolved = true; }
            }
        }
    }
    let mut reserved = hosts.iter().flat_map(|h| h.reserved.iter().map(|r| r.subnet)).collect::<Vec<_>>();
    let observed = reserved.len();
    let mut commands = vec![Vec::new(); hosts.len()];
    let mut verification = vec![Vec::new(); hosts.len()];
    let mut rails = BTreeMap::<(usize, usize), usize>::new();
    let mut links = 0; let mut new_ports = 0;
    for (&a, &b) in &peers {
        let Some(b) = b else { continue; };
        if peers.get(&b) != Some(&Some(a)) {
            reports.push(format!("# {} {}: neighbour was not reciprocal; no configuration proposed", hosts[a.0].alias, hosts[a.0].ports[a.1].name));
            unresolved = true; continue;
        }
        if a > b { continue; }
        let pa = &hosts[a.0].ports[a.1]; let pb = &hosts[b.0].ports[b.1];
        let label = format!("{} {} <-> {} {}", hosts[a.0].alias, pa.name, hosts[b.0].alias, pb.name);
        let addresses = match (pa.addresses.is_empty(), pb.addresses.is_empty()) {
            (false, false) => shared(pa, pb).map(|(a, b)| (a.ip, b.ip)),
            (true, true) => {
                match unused_subnet(&reserved) {
                    Ok(subnet) => {
                        reserved.push(subnet);
                        Some((Ipv4Addr::from(subnet.bounds().0 + 1), Ipv4Addr::from(subnet.bounds().0 + 2)))
                    }
                    Err(error) => { reports.push(format!("# {label}: {error}")); None }
                }
            }
            _ => {
                let (existing, port) = if pa.addresses.is_empty() { (b, pb) } else { (a, pa) };
                port.addresses.iter().filter(|s| s.prefix == 24 && s.private()).find_map(|subnet| {
                    let conflict = hosts.iter().enumerate().any(|(h, host)| host.reserved.iter().any(|r|
                        subnet.overlaps(r.subnet) && !(h == existing.0 && r.device == port.name && r.subnet.prefix >= 24)));
                    if conflict || reserved[observed..].iter().any(|s| subnet.overlaps(*s)) { return None; }
                    let start = subnet.bounds().0;
                    (1..=254).map(|i| Ipv4Addr::from(start + i)).find(|ip| !hosts.iter().any(|h| h.addresses.contains(ip)))
                        .map(|ip| if existing == a { (subnet.ip, ip) } else { (ip, subnet.ip) })
                })
            }
        };
        let Some((ipa, ipb)) = addresses else {
            reports.push(format!("# {label}: no safe IPv4 plan without changing existing ports; skipped"));
            unresolved = true; continue;
        };
        links += 1;
        let rail = rails.entry((a.0, b.0)).or_default(); *rail += 1;
        reports.push(format!("# {label}: rail {rail}, {}", if pa.addresses.is_empty() || pb.addresses.is_empty() {
            format!("plan {}/24", Ipv4Addr::from(u32::from(ipa) & !255))
        } else { "already configured; no sudo commands".into() }));
        for (endpoint, own, peer) in [(a, ipa, ipb), (b, ipb, ipa)] {
            if hosts[endpoint.0].ports[endpoint.1].addresses.is_empty() {
                commands[endpoint.0].extend(configure(&mut hosts[endpoint.0], endpoint.1, own)); new_ports += 1;
            } else if hosts[endpoint.0].ports[endpoint.1].mtu < 9000 {
                verification[endpoint.0].push(format!("# {} has MTU below 9000; jumbo verification will fail; existing settings were skipped", hosts[endpoint.0].ports[endpoint.1].name));
            }
            verification[endpoint.0].push(jumbo(&hosts[endpoint.0].ports[endpoint.1], peer));
        }
    }
    reports.push(format!("# {links} reciprocal links; {new_ports} ports need configuration. Run each host's setup before the verification pings."));
    for (heading, per_host) in [("Setup (run on all hosts first)", commands), ("Jumbo verification (after setup)", verification)] {
        if per_host.iter().all(Vec::is_empty) { continue; }
        reports.push(format!("\n# {heading}:"));
        for (host, commands) in hosts.iter().zip(per_host) {
            if commands.is_empty() { continue; }
            reports.push(format!("\n# On {}:", host.alias)); reports.extend(commands);
        }
    }
    println!("{}", reports.join("\n"));
    if unresolved || links == 0 { return Err("some cabling could not be resolved read-only; review the diagnostics above".into()); }
    Ok(())
}
