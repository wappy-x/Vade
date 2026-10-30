use super::*;
use crate::runner::{Exec, Output};
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

/// Rail `rail` of cable `cable` is 192.0.2.(16(c-1)+8r)/29; the port 0 end is
/// host 1 and the port 1 end is host 2 (docs/eight-sparks.md).
fn rail(cable: u8, rail: u8, host: u8) -> Ipv4Addr { Ipv4Addr::new(192, 0, 2, 16 * (cable - 1) + 8 * rail + host) }

/// Two rings of four, which the ring check must refuse as one ring: (cable,
/// port 0 end, port 1 end).
const TWO_RINGS: [(u8, &str, &str); 8] = [(1, "s1", "s2"), (2, "s2", "s3"), (3, "s3", "s4"), (4, "s4", "s1"), (5, "s5", "s6"), (6, "s6", "s7"), (7, "s7", "s8"), (8, "s8", "s5")];
/// The ring of eight from docs/eight-sparks.md: cable c joins s<c> port 0 to
/// the next Spark's port 1.
const ONE_RING: [(u8, &str, &str); 8] = [(1, "s1", "s2"), (2, "s2", "s3"), (3, "s3", "s4"), (4, "s4", "s5"), (5, "s5", "s6"), (6, "s6", "s7"), (7, "s7", "s8"), (8, "s8", "s1")];

fn line(name: &str, address: Ipv4Addr, speed: i64, mtu: u32, device: &str, state: &str) -> String {
    format!("iface {name} {address}/29 {speed} {mtu} {device} {state}\n")
}

/// The interface lines of `machine` on the cables in `cables`, all healthy.
/// Interface names: p<port>r<rail>.
fn healthy(machine: &str, cables: &[(u8, &str, &str)]) -> String {
    let mut text = line("eth0", Ipv4Addr::new(198, 51, 100, 7), 1000, 1500, "-", "-");
    for &(cable, from, to) in cables {
        for (end, host, port) in [(from, 1, 0), (to, 2, 1)] {
            if end != machine { continue; }
            for r in 0..2 {
                text += &line(&format!("p{port}r{r}"), rail(cable, r, host), 200000, 9000, &format!("roce{port}{r}"), "ACTIVE");
            }
        }
    }
    text
}

/// `probe` with every address in `answered` answering both pings.
fn pinged(text: &str, answered: &[Ipv4Addr]) -> Probe {
    let mut probe = parse_probe(text);
    for address in answered { probe.pings.insert(*address, (true, true)); }
    probe
}

#[test]
fn probe_output_parses_and_ignores_odd_lines() {
    let probe = parse_probe(&format!("{}{}iface bad$name 203.0.113.1/24 1 1 - -\niface eth1 not-an-address 1 1 - -\nping 192.0.2.2 ok fail\nnoise\n",
        line("p0r0", rail(1, 0, 1), 200000, 9000, "rocep1s0f0", "ACTIVE"), line("p1r0", rail(4, 0, 2), -1, 9000, "-", "-")));
    assert_eq!(probe.interfaces.len(), 2);
    assert_eq!(probe.interfaces[0].rdma, Some(("rocep1s0f0".into(), "ACTIVE".into())));
    assert_eq!((probe.interfaces[1].speed, probe.interfaces[1].rdma.clone()), (0, None));
    assert_eq!(probe.pings[&rail(1, 0, 2)], (true, false));
}

#[test]
fn the_scripts_are_valid_shell_and_quote_targets() {
    let script = ping_script(&[rail(1, 0, 2), rail(4, 1, 1)]);
    assert!(script.contains("for address in 192.0.2.2 192.0.2.57; do"), "{script}");
    assert!(ping_script(&[]).contains("for address in ''; do"));
    for text in [script, ping_script(&[]), PROBE_SCRIPT.into(), CLOCK_SCRIPT.into()] {
        let status = std::process::Command::new("sh").args(["-n", "-c", &text]).status().unwrap();
        assert!(status.success(), "{text}");
    }
}

#[test]
fn a_port_needs_its_address_speed_mtu_and_an_active_rdma_device() {
    let own = rail(1, 0, 1);
    let probe = |speed, mtu, device: &str, state: &str| parse_probe(&line("p0r0", own, speed, mtu, device, state));
    assert_eq!(port(&probe(200000, 9000, "roce", "ACTIVE"), own, "s2", true).unwrap(), "p0r0 192.0.2.1/29, 200000 Mb/s, MTU 9000, RDMA roce active");
    let cases = [
        (port(&probe(200000, 9000, "roce", "ACTIVE"), rail(2, 0, 1), "s3", true), "No interface on this Spark has 192.0.2.17"),
        (port(&probe(1000, 9000, "roce", "ACTIVE"), own, "s2", true), "links at 1000 Mb/s"),
        (port(&probe(-1, 9000, "roce", "DOWN"), own, "s2", true), "its link is down"),
        (port(&probe(200000, 1500, "roce", "ACTIVE"), own, "s2", true), "MTU 1500"),
        (port(&probe(200000, 9000, "roce", "DOWN"), own, "s2", true), "is DOWN, not ACTIVE"),
        (port(&probe(200000, 9000, "-", "-"), own, "s2", true), "has no RDMA device"),
    ];
    for (result, expected) in cases { assert!(result.as_ref().unwrap_err().contains(expected), "{result:?}"); }
    assert!(port(&probe(200000, 9000, "-", "-"), own, "s2", false).unwrap().ends_with("no RDMA device; TCP selected"));
}

/// Check cable 1 between s1 and s2 with the two ends' interface lines and
/// the addresses each end hears.
fn cable_one(a_text: &str, b_text: &str, a_hears: &[Ipv4Addr], b_hears: &[Ipv4Addr]) -> Result<String, String> {
    let (a, b) = (pinged(a_text, a_hears), pinged(b_text, b_hears));
    let pair = rails(("s1", rail(1, 0, 1), &a), ("s2", rail(1, 0, 2), &b))?;
    link(("s1", &a), ("s2", &b), &pair, true)
}

#[test]
fn both_rails_of_a_cable_must_pass_on_their_own() {
    let cables = &TWO_RINGS[..1];
    let (a, b) = (healthy("s1", cables), healthy("s2", cables));
    let a_hears = [rail(1, 0, 2), rail(1, 1, 2)];
    let b_hears = [rail(1, 0, 1), rail(1, 1, 1)];
    assert_eq!(cable_one(&a, &b, &a_hears, &b_hears).unwrap(), "rail 0 192.0.2.0/29 and rail 1 192.0.2.8/29: 9000-byte pings answered both ways on both rails");
    let fails = |a: &str, b: &str, a_hears: &[Ipv4Addr], b_hears: &[Ipv4Addr], expected: &str| {
        let error = cable_one(a, b, a_hears, b_hears).unwrap_err();
        assert!(error.contains(expected), "expected {expected:?}: {error}");
    };
    // Rail 0 faults.
    fails(&a, &b.replace("192.0.2.2/29", "192.0.2.66/29"), &a_hears, &b_hears, "s2 has no interface with 192.0.2.2");
    fails(&a, &b.replace("p1r0 192.0.2.2/29 200000 9000", "p1r0 192.0.2.2/29 200000 1500"), &a_hears, &b_hears, "Rail 0: s2: p1r0 has MTU 1500");
    fails(&a, &b, &a_hears[1..], &b_hears, "Rail 0: s1 gets no ping reply from 192.0.2.2");
    // Rail 1 faults: MTU, RDMA state, no reply, a 9000-byte ping that fails.
    fails(&a, &b.replace("p1r1 192.0.2.10/29 200000 9000", "p1r1 192.0.2.10/29 200000 1500"), &a_hears, &b_hears, "Rail 1: s2: p1r1 has MTU 1500");
    fails(&a.replace("roce01 ACTIVE", "roce01 DOWN"), &b, &a_hears, &b_hears, "Rail 1: s1: RDMA device roce01 on p0r1 is DOWN");
    fails(&a, &b, &a_hears, &b_hears[..1], "Rail 1: s2 gets no ping reply from 192.0.2.9");
    let (a_probe, mut b_probe) = (pinged(&a, &a_hears), pinged(&b, &b_hears));
    b_probe.pings.insert(rail(1, 1, 1), (true, false));
    let pair = rails(("s1", rail(1, 0, 1), &a_probe), ("s2", rail(1, 0, 2), &b_probe)).unwrap();
    assert!(link(("s1", &a_probe), ("s2", &b_probe), &pair, true).unwrap_err().contains("Rail 1: s2 reaches 192.0.2.9, but a 9000-byte ping fails"));
    // A rail 1 address added as an alias on the rail 0 interface is not a second rail.
    let alias = |text: &str, from: &str| text.lines().filter(|line| !line.contains(from)).map(|line| format!("{line}\n")).collect::<String>()
        + &text.lines().find(|line| line.contains(from)).unwrap().replace(" p0r1 ", " p0r0 ").replace(" p1r1 ", " p1r0 ") + "\n";
    fails(&alias(&a, "192.0.2.9/29"), &alias(&b, "192.0.2.10/29"), &a_hears, &b_hears, "Only one rail");
    let one_rail = |text: &str| text.lines().filter(|line| !line.contains("r1 ")).map(|line| format!("{line}\n")).collect::<String>();
    fails(&one_rail(&a), &one_rail(&b), &a_hears, &b_hears, "Only one rail");
    // A rail 1 port whose link is down is that rail's fault, not a missing rail.
    let down = a.replace("p0r1 192.0.2.9/29 200000 9000 roce01 ACTIVE", "p0r1 192.0.2.9/29 -1 9000 roce01 DOWN");
    fails(&down, &b, &a_hears, &b_hears, "Rail 1: s1: p0r1 holds 192.0.2.9 but its link is down");
}

#[test]
fn a_rail_needs_two_addresses_and_pings_that_leave_the_spark() {
    let cables = &TWO_RINGS[..1];
    let (a, b) = (healthy("s1", cables), healthy("s2", cables));
    // Every address answers, as a Spark's own addresses do over loopback.
    let all = [rail(1, 0, 1), rail(1, 0, 2), rail(1, 1, 1), rail(1, 1, 2)];
    let check = |a: &str, b: &str, b_rail0: Ipv4Addr| {
        let (a, b) = (pinged(a, &all), pinged(b, &all));
        let pair = rails(("s1", rail(1, 0, 1), &a), ("s2", b_rail0, &b))?;
        link(("s1", &a), ("s2", &b), &pair, true)
    };
    // Rail 1 copied unchanged to both ends.
    let error = check(&a, &b.replace("192.0.2.10/29", "192.0.2.9/29"), rail(1, 0, 2)).unwrap_err();
    assert!(error.contains("Rail 1: s1 and s2 both hold 192.0.2.9"), "{error}");
    // Rail 0 with the same address at both ends.
    let error = check(&a, &b.replace("192.0.2.2/29", "192.0.2.1/29"), rail(1, 0, 1)).unwrap_err();
    assert!(error.contains("Rail 0: s1 and s2 both hold 192.0.2.1"), "{error}");
    // s1 also holds the far end's rail 1 address, so its ping never leaves it.
    let error = check(&(a.clone() + &line("lo1", rail(1, 1, 2), -1, 65536, "-", "-")), &b, rail(1, 0, 2)).unwrap_err();
    assert!(error.contains("Rail 1: s1 holds 192.0.2.10, the far end's address, itself"), "{error}");
}

/// Answers each Spark's probe from `probes`, its pings from the targets in
/// the script (all answered unless listed in `silent`), and its clock from
/// this computer's.
struct Scripted { probes: BTreeMap<String, String>, silent: BTreeSet<(String, Ipv4Addr)>, labels: Vec<String> }
impl Runner for Scripted {
    fn run(&mut self, exec: &Exec) -> Result<Output, String> {
        self.labels.push(exec.label.clone());
        let (machine, check) = exec.label.split_once(' ').unwrap();
        let stdout = match check {
            // A Spark with no probe text fails its probe, as an SSH failure does.
            "fabric probe" => match self.probes.get(machine) { Some(text) => text.clone(), None => return Ok(Output { success: false, stdout: String::new() }) },
            "fabric ping" => {
                let script = exec.args.last().unwrap();
                let targets = script.lines().find_map(|line| line.strip_prefix("for address in ")?.strip_suffix("; do")).unwrap();
                targets.split_whitespace().filter_map(|target| target.parse::<Ipv4Addr>().ok()).map(|target| {
                    let answer = if self.silent.contains(&(machine.to_string(), target)) { "fail fail" } else { "ok ok" };
                    format!("ping {target} {answer}\n")
                }).collect()
            }
            "clock" => format!("{}\nyes\n", now()),
            _ => panic!("unexpected command {}", exec.label),
        };
        Ok(Output { success: true, stdout })
    }
    fn interactive(&mut self, _: &Exec) -> Result<bool, String> { panic!("The fabric check never starts an interactive command.") }
    fn port_free(&mut self, _: u16) -> Result<(), String> { panic!("The fabric check never reads local ports.") }
    fn pause(&mut self, _: Duration) { panic!("The fabric check never waits.") }
}

fn eight_spark_text(groups: &str) -> String { cabled_text(&TWO_RINGS, groups) }

fn cabled_text(cables: &[(u8, &str, &str)], groups: &str) -> String {
    let mut text = "[global]\nstate_dir = /state\n".to_string();
    for index in 1..=8 {
        let name = format!("s{index}");
        text += &format!("[machine {name}]\nssh = {name}-ssh\n");
        for &(cable, from, to) in cables {
            if from == name { text += &format!("fabric.{to} = {}\n", rail(cable, 0, 1)); }
            if to == name { text += &format!("fabric.{from} = {}\n", rail(cable, 0, 2)); }
        }
    }
    text + groups
}

const GROUPS: &str = "[group left]\nhosts = s1,s2,s3,s4\nlayout = quads\n[group right]\nhosts = s5,s6,s7,s8\nlocal_port = 18090\nlayout = quads\n[group bad]\nhosts = s1,s3,s2,s4\nlocal_port = 18100\nlayout = test\n";

fn fabric_rows(text: &str, probes: BTreeMap<String, String>, silent: BTreeSet<(String, Ipv4Addr)>, dry: bool) -> (Result<(), String>, String, Vec<String>) {
    ring_rows(text, probes, silent, false, dry)
}

/// [`fabric_rows`] with `--ring` set by `ring`.
fn ring_rows(text: &str, probes: BTreeMap<String, String>, silent: BTreeSet<(String, Ipv4Addr)>, ring: bool, dry: bool) -> (Result<(), String>, String, Vec<String>) {
    let config = Config::parse(text, Path::new("test-config/config")).unwrap();
    let mut runner = Scripted { probes, silent, labels: Vec::new() };
    let mut report = Report::new("controller", dry);
    let capture = crate::output::testing::Capture::default();
    let result = crate::output::with_writer(capture.clone(), || run(&mut report, &mut runner, &config, true, ring, dry).and_then(|()| report.finish()));
    (result, String::from_utf8(capture.bytes.borrow().clone()).unwrap(), runner.labels)
}

fn row<'a>(printed: &'a str, name: &str) -> &'a str {
    printed.lines().find(|line| line.get(6..).is_some_and(|rest| rest.starts_with(name))).unwrap_or_else(|| panic!("{name}:\n{printed}"))
}

fn healthy_probes() -> BTreeMap<String, String> { probes_for(&TWO_RINGS) }

fn probes_for(cables: &[(u8, &str, &str)]) -> BTreeMap<String, String> {
    (1..=8).map(|index| { let name = format!("s{index}"); let text = healthy(&name, cables); (name, text) }).collect()
}

/// Arcs of the ring of eight: the whole ring, two quads, a quad across
/// cable 8, a pair on cable 8, and four Sparks with a gap.
const ARCS: &str = "[group eight]\nhosts = s1,s2,s3,s4,s5,s6,s7,s8\nlayout = eight\n[group quad-a]\nhosts = s1,s2,s3,s4\nlayout = quads\n\
    [group quad-b]\nhosts = s5,s6,s7,s8\nlocal_port = 18090\nlayout = quads\n[group cross]\nhosts = s7,s8,s1,s2\nlayout = cross\n\
    [group pair-81]\nhosts = s8,s1\nlayout = pairs\n[group gap]\nhosts = s1,s2,s4,s5\nlayout = gap\n";

#[test]
fn a_ring_of_eight_passes_the_ring_rows_and_every_arc() {
    let (result, printed, labels) = ring_rows(&cabled_text(&ONE_RING, ARCS), probes_for(&ONE_RING), BTreeSet::new(), true, false);
    assert!(result.unwrap_err().starts_with("1 doctor check failed."), "only the gap group fails:\n{printed}");
    assert!(row(&printed, "cabling").contains("ring of 8: s1, s2, s3, s4, s5, s6, s7, s8"), "{printed}");
    assert!(row(&printed, "ring ").starts_with("ok") && row(&printed, "ring ").contains("ring of 8: s1, s2, s3, s4, s5, s6, s7, s8; every Spark has two neighbors"), "{printed}");
    assert!(row(&printed, "group eight").contains("arc of 8 in rank order: s1, s2, s3, s4, s5, s6, s7, s8; the ring closes"), "{printed}");
    assert!(row(&printed, "group quad-b").contains("arc of 4 in rank order: s5, s6, s7, s8"), "{printed}");
    assert!(row(&printed, "group cross").starts_with("ok") && row(&printed, "group pair-81").starts_with("ok"), "{printed}");
    assert!(row(&printed, "group gap").starts_with("FAIL") && printed.contains("Group gap is not an arc of the cabling: s2 and s4"), "{printed}");
    for cable in ["s1 - s2 cable", "s4 - s5 cable", "s1 - s8 cable", "s5 port to s4", "s1 port to s8"] {
        assert!(row(&printed, cable).starts_with("ok"), "{cable}:\n{printed}");
    }
    assert!(row(&printed, "ring line rate").starts_with("ok") && printed.contains("rail 0 at 200000 Mb/s and rail 1 at 200000 Mb/s on all 8 cables"), "{printed}");
    assert_eq!(labels.len(), 24, "one probe, one ping and one clock call per Spark: {labels:?}");
    // Without --ring there are no ring rows.
    let (_, printed, _) = fabric_rows(&cabled_text(&ONE_RING, ""), probes_for(&ONE_RING), BTreeSet::new(), false);
    assert!(!printed.contains("ring line rate") && !printed.lines().any(|line| line.get(6..).is_some_and(|rest| rest.starts_with("ring "))), "{printed}");
}

#[test]
fn one_slow_rail_fails_the_ring_line_rate_only() {
    // Rail 1 of cable 5 links at 100000 Mb/s at the s6 end: still a fabric port, but not the ring's rate.
    let mut probes = probes_for(&ONE_RING);
    let slow = probes["s6"].replace("p1r1 192.0.2.74/29 200000", "p1r1 192.0.2.74/29 100000");
    probes.insert("s6".into(), slow);
    let (result, printed, _) = ring_rows(&cabled_text(&ONE_RING, ""), probes, BTreeSet::new(), true, false);
    assert!(result.unwrap_err().starts_with("1 doctor check failed."), "{printed}");
    assert!(row(&printed, "s5 - s6 cable").starts_with("ok"), "{printed}");
    assert!(row(&printed, "ring line rate").starts_with("FAIL") && printed.contains("but s5 - s6 cable rail 1: s6 p1r1 at 100000 Mb/s"), "{printed}");
    // A cable without two rails leaves the rate unproven.
    let mut probes = probes_for(&ONE_RING);
    for machine in ["s2", "s3"] {
        let text = probes[machine].lines().filter(|line| !line.contains("192.0.2.25/") && !line.contains("192.0.2.26/")).map(|line| format!("{line}\n")).collect::<String>();
        probes.insert(machine.into(), text);
    }
    let (_, printed, _) = ring_rows(&cabled_text(&ONE_RING, ""), probes, BTreeSet::new(), true, false);
    assert!(row(&printed, "s2 - s3 cable").starts_with("FAIL") && printed.contains("Only 7 of 8 cables have both rails to compare"), "{printed}");
}

#[test]
fn the_ring_row_names_a_missing_closing_cable_and_two_rings() {
    // Cable 8 (s8 - s1) is not in the config: a path of eight.
    let (result, printed, _) = ring_rows(&cabled_text(&ONE_RING[..7], ""), BTreeMap::new(), BTreeSet::new(), true, true);
    assert!(result.is_err() && row(&printed, "ring ").starts_with("FAIL") && printed.contains("The ring does not close: path of 8: s1, s2, s3, s4, s5, s6, s7, s8.")
        && printed.contains("next: Add the cable between s1 and s8, then add fabric.s8 under [machine s1]"), "{printed}");
    assert!(row(&printed, "ring line rate").starts_with("plan"), "{printed}");
    // Two rings of four are not one ring.
    let (_, printed, _) = ring_rows(&eight_spark_text(""), BTreeMap::new(), BTreeSet::new(), true, true);
    assert!(printed.contains("The cabling is 2 rings, not one"), "{printed}");
    // The ring of eight dry run passes with no SSH.
    let (result, printed, labels) = ring_rows(&cabled_text(&ONE_RING, ""), BTreeMap::new(), BTreeSet::new(), true, true);
    assert!(result.is_ok() && labels.is_empty(), "{printed}");
}

#[test]
fn eight_sparks_in_two_rings_pass_every_fabric_check() {
    let (result, printed, labels) = fabric_rows(&eight_spark_text(GROUPS), healthy_probes(), BTreeSet::new(), false);
    assert!(result.unwrap_err().starts_with("1 doctor check failed."), "only the bad group fails:\n{printed}");
    assert!(row(&printed, "cabling").contains("ring of 4: s1, s2, s3, s4; ring of 4: s5, s6, s7, s8"), "{printed}");
    assert!(row(&printed, "group left").contains("arc of 4 in rank order: s1, s2, s3, s4; the ring closes"), "{printed}");
    assert!(row(&printed, "group bad").starts_with("FAIL") && printed.contains("hosts = s1,s2,s3,s4"), "{printed}");
    for cable in ["s1 - s2 cable", "s1 - s4 cable", "s5 - s8 cable", "s6 - s7 cable", "s4 port to s1", "s8 clock"] {
        assert!(row(&printed, cable).starts_with("ok"), "{cable}:\n{printed}");
    }
    assert!(!printed.contains("s4 - s5 cable"), "{printed}");
    assert_eq!(labels.len(), 24, "one probe, one ping and one clock call per Spark: {labels:?}");

    // No reply on rail 1 of cable 3 fails that cable only.
    let silent = BTreeSet::from([("s3".to_string(), rail(3, 1, 2))]);
    let (_, printed, _) = fabric_rows(&eight_spark_text(GROUPS), healthy_probes(), silent, false);
    assert!(row(&printed, "s3 - s4 cable").starts_with("FAIL") && printed.contains("Rail 1: s3 gets no ping reply from 192.0.2.42"), "{printed}");
    assert!(row(&printed, "s2 - s3 cable").starts_with("ok"), "{printed}");

    // Rail 1 of cable 1 copied unchanged to s2: every ping answers, the cable fails.
    let mut probes = healthy_probes();
    let copied = probes["s2"].replace("192.0.2.10/29", "192.0.2.9/29");
    probes.insert("s2".into(), copied);
    let (result, printed, _) = fabric_rows(&eight_spark_text(""), probes, BTreeSet::new(), false);
    assert!(result.is_err() && row(&printed, "s1 - s2 cable").starts_with("FAIL") && printed.contains("Rail 1: s1 and s2 both hold 192.0.2.9"), "{printed}");
}

#[test]
fn a_spark_whose_probe_fails_skips_its_port_and_cable_rows() {
    let mut probes = probes_for(&ONE_RING);
    probes.remove("s3");
    let (result, printed, labels) = ring_rows(&cabled_text(&ONE_RING, ""), probes, BTreeSet::new(), true, false);
    assert!(result.is_err() && row(&printed, "s3 fabric ports").starts_with("FAIL"), "{printed}");
    for skipped in ["s3 port to s2", "s3 port to s4", "s2 - s3 cable", "s3 - s4 cable"] {
        assert!(row(&printed, skipped).starts_with("skip") && row(&printed, skipped).contains("the fabric probe of s3 did not run"), "{skipped}:\n{printed}");
    }
    assert!(row(&printed, "s2 port to s3").starts_with("ok") && row(&printed, "s4 - s5 cable").starts_with("ok"), "{printed}");
    assert!(!printed.contains("No interface on this Spark"), "{printed}");
    assert!(printed.contains("Only 6 of 8 cables have both rails to compare"), "{printed}");
    assert!(!labels.contains(&"s3 fabric ping".to_string()), "{labels:?}");
}

#[test]
fn two_cables_cannot_share_a_rail_subnet() {
    // Cable 6 (s6 - s7) reuses rail 1 subnet 192.0.2.72/29 of cable 5 (s5 - s6).
    let mut probes = healthy_probes();
    for machine in ["s6", "s7"] {
        let text = probes[machine].replace("192.0.2.89/29", "192.0.2.75/29").replace("192.0.2.90/29", "192.0.2.76/29");
        probes.insert(machine.into(), text);
    }
    let (_, printed, _) = fabric_rows(&eight_spark_text(""), probes, BTreeSet::new(), false);
    assert!(row(&printed, "s5 - s6 cable").starts_with("ok"), "{printed}");
    assert!(row(&printed, "s6 - s7 cable").starts_with("FAIL") && printed.contains("the same as a rail of the s5 - s6 cable"), "{printed}");
}

#[test]
fn a_link_with_one_end_in_the_config_is_a_failed_row() {
    // No groups: s2 names s1, but s1 does not name s2.
    let text = eight_spark_text("").replacen("fabric.s2 = 192.0.2.1\n", "", 1);
    for dry in [true, false] {
        let (result, printed, _) = fabric_rows(&text, healthy_probes(), BTreeSet::new(), dry);
        assert!(result.is_err(), "{printed}");
        assert!(row(&printed, "s2 - s1 cable").starts_with("FAIL") && printed.contains("s2 has fabric.s1, but [machine s1] has no fabric.s2"), "{printed}");
    }
}

#[test]
fn a_dry_run_plans_every_check_without_ssh() {
    let (result, printed, labels) = fabric_rows(&eight_spark_text(GROUPS), BTreeMap::new(), BTreeSet::new(), true);
    assert!(labels.is_empty(), "{labels:?}");
    assert!(result.is_err(), "the bad group still fails: {printed}");
    assert!(row(&printed, "s7 fabric ports").starts_with("plan") && row(&printed, "s5 - s8 cable").starts_with("plan"), "{printed}");
    let (result, printed, _) = fabric_rows(&eight_spark_text(""), BTreeMap::new(), BTreeSet::new(), true);
    assert!(result.is_ok(), "{printed}");
}

#[test]
fn clocks_must_be_close_and_synchronized() {
    let base = 1_800_000_000_000_000_000u128;
    let ms = 1_000_000u128;
    assert!(clock(&format!("{}\nyes\n", base + 5 * ms), base, base + 10 * ms).unwrap().starts_with("within 1000 ms"));
    assert!(clock(&format!("{}\nyes\n", base + 2_500 * ms), base, base + 10 * ms).unwrap_err().starts_with("Clock is 2490 ms ahead of"));
    assert!(clock(&format!("{}\nyes\n", base - 3_000 * ms), base, base + 10 * ms).unwrap_err().starts_with("Clock is 3000 ms behind"));
    assert!(clock(&format!("{base}\nno\n"), base, base).unwrap_err().contains("time sync reports no"));
    assert!(clock("Thu Oct 29\n", base, base).unwrap_err().contains("did not print its clock"));
}

#[test]
fn the_example_config_passes_its_dry_run_ring_check() {
    for ring in [false, true] {
        let (result, printed, labels) = ring_rows(include_str!("../../examples/eight-sparks.conf"), BTreeMap::new(), BTreeSet::new(), ring, true);
        assert!(result.is_ok() && labels.is_empty(), "{printed}");
        assert!(!printed.contains("FAIL"), "{printed}");
    }
}
