//! Cable discovery for `vade up --sparks` and `vade setup --hosts`. Before setup
//! writes a config, it reaches every Spark the way `vade up` does (SSH with
//! BatchMode, so never a prompt) and runs the interface probe of `vade doctor
//! --fabric`. Two Sparks that share a subnet on fabric ports have a cable. The
//! cables give the chain, so the user can type the Sparks in any order and needs
//! `--link` only as an override. Setup never changes SSH settings: when a Spark
//! cannot be reached, it prints ssh's own error and the one command that the
//! user runs.
use crate::doctor::fabric::{self, Interface};
use crate::runner::{self, Runner};
use std::net::Ipv4Addr;

/// The last line of [`script`], so a login shell that prints nothing is not
/// taken for a Spark with no cables.
const END: &str = "vade-cables-end";

/// How many search steps [`chain`] takes before it keeps the best chain so far.
/// This covers every order of eight Sparks, and Sparks have few cables.
const SEARCH_STEPS: usize = 200_000;

/// The doctor probe, then the home folder and the address that SSH reached.
pub(super) fn script() -> String {
    format!("{}printf 'home %s\\n' \"$HOME\"\nset -- ${{SSH_CONNECTION:-}}\nprintf 'reached %s\\n' \"${{3:--}}\"\nprintf '{END}\\n'\n", fabric::PROBE_SCRIPT)
}

/// What [`script`] found on one Spark.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Spark {
    /// Up fabric ports only: 100 Gb/s or faster, with an active RDMA port.
    pub ports: Vec<Interface>,
    pub home: String,
    /// The address that SSH reached, or the destination when it is an address.
    pub reached: Option<Ipv4Addr>,
}

/// One cable: the address of each end, in the order the two Sparks were given.
pub(super) type Ends = (Ipv4Addr, Ipv4Addr);

/// What two Sparks share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Cable {
    None,
    One(Ends),
    /// Several addresses with no way to choose, such as a second address on one port.
    Unclear(Vec<Ends>),
}

/// Read [`script`] output from `destination`, or `None` when the script did not finish.
pub(super) fn parse(destination: &str, output: &str) -> Option<Spark> {
    output.lines().any(|line| line.trim() == END).then(|| {
        let value = |key: &str| output.lines().find_map(|line| line.strip_prefix(key)?.strip_prefix(' ').map(str::trim));
        let ports = fabric::parse_probe(output).interfaces.into_iter()
            .filter(|port| port.speed >= fabric::FABRIC_MBPS && port.rdma.as_ref().is_some_and(|(_, state)| state == "ACTIVE")).collect();
        let named = destination.rsplit('@').next().and_then(|host| host.parse().ok());
        Spark { ports, home: value("home").unwrap_or_default().to_owned(), reached: value("reached").and_then(|address| address.parse().ok()).or(named) }
    })
}

/// Sort key for a pair of ports. Rail 0 is `enp...` and rail 1 is `enP2p...`
/// (docs/network-setup.md), so the name is compared without its case: the bytes
/// of `P` sort before `p`, which would put rail 1 first.
fn name_key(a: &Interface, b: &Interface) -> (String, String) { (a.name.to_ascii_lowercase(), b.name.to_ascii_lowercase()) }

/// The cable between two Sparks: each pair of their ports in one subnet, with
/// two different addresses. The port that carries SSH counts only when no other
/// pair exists: a LAN port must not pass for a cable, but a rented pair can
/// have one link for both. Pairs on different ports at each end are the rails
/// of one cable, or two cables: the first by name is rail 0, which is the
/// address the config holds. Doctor and the engine find rail 1 by themselves.
pub(super) fn cable(a: &Spark, b: &Spark) -> Cable {
    let mut pairs: Vec<(&Interface, &Interface)> = a.ports.iter()
        .flat_map(|one| b.ports.iter().filter(move |two| one.subnet() == two.subnet() && one.address != two.address).map(move |two| (one, two)))
        .collect();
    let apart = |pair: &(&Interface, &Interface)| Some(pair.0.address) != a.reached && Some(pair.1.address) != b.reached;
    if pairs.iter().any(apart) { pairs.retain(apart); }
    pairs.sort_by_key(|(one, two)| name_key(one, two));
    let ends = |pair: &(&Interface, &Interface)| (pair.0.address, pair.1.address);
    let distinct = |list: Vec<&str>| { let mut seen = list.clone(); seen.sort_unstable(); seen.dedup(); seen.len() == list.len() };
    match pairs.as_slice() {
        [] => Cable::None,
        [pair, ..] if distinct(pairs.iter().map(|pair| pair.0.name.as_str()).collect()) && distinct(pairs.iter().map(|pair| pair.1.name.as_str()).collect()) => Cable::One(ends(pair)),
        _ => Cable::Unclear(pairs.iter().map(ends).collect()),
    }
}

/// The order of `count` Sparks, by index in the given order, that visits each
/// one once and steps only over `joined` pairs. The given order wins when it
/// fits. Else the answer starts with the first given Spark when it can, then
/// keeps the most pairs in their given order. With no such chain, the error
/// names two Sparks that stop the longest chain found: its last Spark, and the
/// first Spark that it left out.
pub(super) fn chain(count: usize, joined: &dyn Fn(usize, usize) -> bool) -> Result<Vec<usize>, (usize, usize)> {
    let given: Vec<usize> = (0..count).collect();
    if given.windows(2).all(|pair| joined(pair[0], pair[1])) { return Ok(given); }
    let mut search = Search { count, joined, steps: 0, path: Vec::with_capacity(count), best: None, longest: Vec::new() };
    for start in 0..count {
        search.path.push(start);
        search.visit();
        search.path.pop();
    }
    match search.best {
        Some((_, path)) => Ok(path),
        None => {
            let last = search.longest.last().copied().unwrap_or(0);
            Err((last, (0..count).find(|index| !search.longest.contains(index)).unwrap_or(last)))
        }
    }
}

/// Depth-first search state of [`chain`]. Indexes are tried in the given order,
/// so a tie keeps the path that follows that order first.
struct Search<'a> {
    count: usize,
    joined: &'a dyn Fn(usize, usize) -> bool,
    steps: usize,
    path: Vec<usize>,
    best: Option<((bool, usize), Vec<usize>)>,
    longest: Vec<usize>,
}

impl Search<'_> {
    fn visit(&mut self) {
        if self.steps >= SEARCH_STEPS { return; }
        self.steps += 1;
        if self.path.len() > self.longest.len() { self.longest = self.path.clone(); }
        if self.path.len() == self.count {
            let kept = (0..self.count).map(|at| self.path[at + 1..].iter().filter(|later| self.path[at] < **later).count()).sum();
            let score = (self.path[0] == 0, kept);
            if self.best.as_ref().is_none_or(|(best, _)| score > *best) { self.best = Some((score, self.path.clone())); }
            return;
        }
        let at = self.path[self.path.len() - 1];
        for next in 0..self.count {
            if !self.path.contains(&next) && ((self.joined)(at, next) || (self.joined)(next, at)) {
                self.path.push(next);
                self.visit();
                self.path.pop();
            }
        }
    }
}

/// Why setup cannot read the address of a cable end by itself.
pub(super) fn unclear(from: &str, to: &str, cable: &Cable) -> String {
    match cable {
        Cable::Unclear(found) => format!("{from} and {to} share {} addresses: {}", found.len(), found.iter().map(|(a, b)| format!("{a} and {b}")).collect::<Vec<_>>().join(", ")),
        _ => format!("{from} and {to} share no subnet on a fabric port (docs/network-setup.md sets them up)"),
    }
}

/// What the probes found on the given Sparks, with the `--link` cable ends.
pub(super) struct Cables<'a> {
    hosts: &'a [String],
    sparks: Vec<Option<Spark>>,
    links: &'a [(String, String, String)],
}

impl<'a> Cables<'a> {
    pub(super) fn new(hosts: &'a [String], sparks: Vec<Option<Spark>>, links: &'a [(String, String, String)]) -> Self { Self { hosts, sparks, links } }

    /// Whether every Spark answered. On a dry run none did.
    pub(super) fn read(&self) -> bool { self.sparks.iter().all(Option::is_some) }

    fn index(&self, host: &str) -> Option<usize> { self.hosts.iter().position(|name| name == host) }

    /// The cable that the probes found between two given Sparks.
    fn found(&self, a: usize, b: usize) -> Cable {
        match (&self.sparks[a], &self.sparks[b]) {
            (Some(one), Some(two)) => cable(one, two),
            _ => Cable::None,
        }
    }

    /// The Sparks in cable order, as typed, or the two Sparks that stop the chain.
    /// `--link` names a cable that the probes need not see.
    pub(super) fn chain(&self) -> Result<Vec<String>, (String, String)> {
        if !self.read() { return Ok(self.hosts.to_vec()); }
        let count = self.hosts.len();
        let linked = |a: usize, b: usize| self.links.iter().any(|(from, to, _)| (*from == self.hosts[a] && *to == self.hosts[b]) || (*from == self.hosts[b] && *to == self.hosts[a]));
        let joined: Vec<bool> = (0..count * count).map(|at| linked(at / count, at % count) || self.found(at / count, at % count) != Cable::None).collect();
        match chain(count, &|a, b| joined[a * count + b]) {
            Ok(order) => Ok(order.into_iter().map(|index| self.hosts[index].clone()).collect()),
            Err((a, b)) => Err((self.hosts[a].clone(), self.hosts[b].clone())),
        }
    }

    /// The address of `from` on its one cable to `to`.
    pub(super) fn address(&self, from: &str, to: &str) -> Option<String> {
        match self.found(self.index(from)?, self.index(to)?) {
            Cable::One((own, _)) => Some(own.to_string()),
            _ => None,
        }
    }

    /// Why [`Self::address`] has no answer for this cable end.
    pub(super) fn unclear(&self, from: &str, to: &str) -> String {
        let cable = self.index(from).zip(self.index(to)).map_or(Cable::None, |(a, b)| self.found(a, b));
        unclear(from, to, &cable)
    }

    /// `~/` paths expand on each Spark and must give the same folder on all of
    /// them, or vade up stops after a full install.
    pub(super) fn same_home(&self) -> Result<(), String> {
        let mut homes = self.hosts.iter().zip(&self.sparks).filter_map(|(host, spark)| Some((host, spark.as_ref()?.home.as_str())));
        let Some((first, home)) = homes.next() else { return Ok(()) };
        match homes.find(|(_, other)| other != &home) {
            Some((host, other)) => Err(format!("{first} has the home folder {home}, and {host} has {other}. Vade uses one folder path on every Spark, so use one account name on all of them, then run the same command again.")),
            None => Ok(()),
        }
    }
}

/// Run [`script`] on `destination` with the SSH options of `vade up`. This
/// computer runs it with no SSH. A failed connection shows ssh's own error with
/// its fix, which `retry` ends: the command to run after that fix.
pub(super) fn probe<R: Runner>(runner: &mut R, destination: &str, retry: &str) -> Result<Spark, String> {
    let exec = runner::ssh(destination, script(), &format!("{destination} cable check"));
    let output = runner.run(&exec)?;
    if output.success && let Some(spark) = parse(destination, &output.stdout) { return Ok(spark); }
    let stop = runner::ssh_check(destination).and_then(|check| runner.run(&check).ok()).filter(|check| !check.success)
        .and_then(|check| runner::SshStop::from_output(destination, &check.stdout));
    Err(match stop {
        Some(stop) => format!("{} {}", stop.error, stop.fix.replace("then run vade doctor.", &format!("then run {retry} again."))),
        None => format!("The cable check did not finish on {destination}. Run ssh {destination} ip -o -4 addr show to see what it prints, then run {retry} again."),
    })
}

#[cfg(test)]
#[path = "links_tests.rs"]
mod tests;
