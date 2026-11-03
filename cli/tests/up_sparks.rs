//! `vade up --sparks a,b,c,d` on the real binary, with the fake tools of
//! tests/fixtures/fake_hosts first on PATH and four fake hosts. The tests read what
//! happens before the first install step. The install itself runs in process on
//! the same tools, in cli/src/app/tests/fake_hosts.rs, because a Mac cannot pass
//! the host platform check for a Linux bundle.
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The cables of the fake hosts, in chain order: host-a to host-b on two rails,
/// then host-b to host-c, then host-c to host-d.
const CABLES: [(&str, &str, &str); 4] = [
    ("host-a", "enp1s0f0np0 192.0.2.1", "host-b enp1s0f1np1 192.0.2.2"),
    ("host-a", "enP2p1s0f0np0 198.51.100.1", "host-b enP2p1s0f1np1 198.51.100.2"),
    ("host-b", "enp1s0f0np0 203.0.113.1", "host-c enp1s0f1np1 203.0.113.2"),
    ("host-c", "enp1s0f0np0 203.0.113.65", "host-d enp1s0f1np1 203.0.113.66"),
];
const HOSTS: [&str; 4] = ["host-a", "host-b", "host-c", "host-d"];

struct Fake { home: PathBuf }

impl Fake {
    /// Four fake hosts with their cables, and one LAN port each on one subnet.
    /// Every address is in a documentation range, cut to /26 for enough subnets.
    /// That port is too slow to count.
    fn new(name: &str) -> Self {
        let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let home = base.join(format!("vade-up-sparks-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(home.join("tools")).unwrap();
        std::fs::create_dir(home.join("bundle")).unwrap();
        use std::os::unix::fs::PermissionsExt;
        for entry in std::fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_hosts")).unwrap() {
            let path = entry.unwrap().path();
            let copy = home.join("tools").join(path.file_name().unwrap());
            std::fs::copy(&path, &copy).unwrap();
            std::fs::set_permissions(copy, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let fake = Self { home };
        for (index, host) in HOSTS.iter().enumerate() {
            std::fs::create_dir_all(fake.host(host)).unwrap();
            std::fs::write(fake.host(host).join("known"), "").unwrap();
            let lan = 193 + index;
            std::fs::write(fake.host(host).join("interfaces"), format!("iface enP7p1s0 203.0.113.{lan}/26 10000 1500 mlx5_9 ACTIVE\nhome /home/fake\nreached 203.0.113.{lan}\n")).unwrap();
        }
        fake
    }
    fn host(&self, host: &str) -> PathBuf { self.home.join("hosts").join(host) }
    fn add(&self, host: &str, port: &str, address: &str) {
        let path = self.host(host).join("interfaces");
        let lines = std::fs::read_to_string(&path).unwrap();
        std::fs::write(path, format!("iface {port} {address}/26 200000 9000 mlx5_0 ACTIVE\n{lines}")).unwrap();
    }
    /// Lay `cables`, and also the cable that closes a ring when `ring` is set.
    fn wire(&self, cables: &[(&str, &str, &str)], ring: bool) {
        for (from, own, far) in cables {
            let own: Vec<&str> = own.split(' ').collect();
            let far: Vec<&str> = far.split(' ').collect();
            self.add(from, own[0], own[1]);
            self.add(far[0], far[1], far[2]);
        }
        if ring {
            self.add("host-d", "enp1s0f0np0", "203.0.113.129");
            self.add("host-a", "enp1s0f1np1", "203.0.113.130");
        }
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_vade")).args(args).env_clear().current_dir(&self.home)
            .env("HOME", &self.home).env("XDG_CONFIG_HOME", self.home.join("xdg")).env("VADE_BUNDLE", self.home.join("bundle"))
            .env("FAKE_HOSTS", self.home.join("hosts")).env("PATH", format!("{}:/usr/bin:/bin", self.home.join("tools").display()))
            .stdin(Stdio::null()).output().unwrap()
    }
    fn config(&self) -> PathBuf { self.home.join("xdg/vade/config") }
    /// The ssh value of each machine in the written config, in rank order.
    fn order(&self) -> Vec<String> {
        std::fs::read_to_string(self.config()).unwrap().lines().filter_map(|line| line.strip_prefix("ssh = ")).map(str::to_owned).collect()
    }
}

impl Drop for Fake { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.home); } }

fn stdout(output: &Output) -> String { String::from_utf8_lossy(&output.stdout).into_owned() }
fn stderr(output: &Output) -> String { String::from_utf8_lossy(&output.stderr).into_owned() }

#[test]
fn names_in_cable_order_keep_that_order() {
    let fake = Fake::new("in-order");
    fake.wire(&CABLES, false);
    let output = fake.run(&["up", "--sparks", "host-a,host-b,host-c,host-d"]);
    let printed = stdout(&output);
    assert!(printed.contains("Cable order: host-a - host-b - host-c - host-d.\n"), "{printed}{}", stderr(&output));
    assert_eq!(fake.order(), HOSTS);
    let config = std::fs::read_to_string(fake.config()).unwrap();
    assert!(config.starts_with("# Written by vade setup. Start: vade up --hosts host-a,host-b,host-c,host-d\n"), "{config}");
    // Rail 0 of the two-rail cable, each end's own address. The LAN port is not a cable.
    for line in ["fabric.host-b = 192.0.2.1", "fabric.host-a = 192.0.2.2", "fabric.host-c = 203.0.113.1", "fabric.host-b = 203.0.113.2", "fabric.host-d = 203.0.113.65", "fabric.host-c = 203.0.113.66"] {
        assert!(config.lines().any(|row| row == line), "{line}:\n{config}");
    }
    assert!(!config.contains("198.51.100") && !config.contains("203.0.113.19"), "{config}");
    // The same config as a bundle install: no source keys, so no Rust and no checkout.
    assert!(config.lines().any(|row| row == "source = installed-bundle") && !config.contains("engine_source"), "{config}");
}

#[test]
fn shuffled_names_find_the_chain_and_print_it_in_one_line() {
    let fake = Fake::new("shuffled");
    fake.wire(&CABLES, false);
    let output = fake.run(&["up", "--sparks", "host-c,host-a,host-d,host-b"]);
    let printed = stdout(&output);
    // Neither end was typed first and both directions keep three pairs, so the first found stays.
    assert!(printed.contains("Cable order: host-a - host-b - host-c - host-d.\n"), "{printed}{}", stderr(&output));
    assert_eq!(fake.order(), HOSTS);
    assert_eq!(printed.matches("Cable order").count(), 1, "{printed}");
}

#[test]
fn a_closed_ring_keeps_the_given_order() {
    let fake = Fake::new("ring");
    fake.wire(&CABLES, true);
    let output = fake.run(&["up", "--sparks", "host-c,host-d,host-a,host-b"]);
    assert!(stdout(&output).contains("Cable order: host-c - host-d - host-a - host-b.\n"), "{output:?}");
    assert_eq!(fake.order(), ["host-c", "host-d", "host-a", "host-b"]);
}

#[test]
fn a_spark_with_no_shared_cable_gives_one_sentence_and_writes_nothing() {
    let fake = Fake::new("no-cable");
    fake.wire(&CABLES[..3], false);
    let output = fake.run(&["up", "--sparks", "host-a,host-b,host-c,host-d"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let text = stderr(&output);
    assert!(text.contains("host-c and host-d share no cable, so check that cable or add --link host-c:host-d=ADDRESS --link host-d:host-c=ADDRESS."), "{text}");
    assert!(!fake.config().exists() && !fake.home.join(".vade").exists());
    assert!(!stdout(&output).contains("Cable order"), "{output:?}");
    // The --link override names that cable, and the probes find the rest.
    fake.add("host-c", "enp2s0", "198.51.100.65");
    fake.add("host-d", "enp2s0", "198.51.100.129");
    let output = fake.run(&["up", "--sparks", "host-a,host-b,host-c,host-d", "--link", "host-c:host-d=198.51.100.65", "--link", "host-d:host-c=198.51.100.129"]);
    assert!(stdout(&output).contains("Cable order: host-a - host-b - host-c - host-d.\n"), "{output:?}");
    let config = std::fs::read_to_string(fake.config()).unwrap();
    assert!(config.contains("fabric.host-d = 198.51.100.65") && config.contains("fabric.host-c = 198.51.100.129"), "{config}");
}

#[test]
fn a_refused_key_shows_the_real_line_and_every_fix_at_once() {
    let fake = Fake::new("denied");
    fake.wire(&CABLES, false);
    std::fs::write(fake.host("host-b").join("denied"), "").unwrap();
    std::fs::remove_file(fake.host("host-d").join("known")).unwrap();
    let output = fake.run(&["up", "--sparks", "host-a,host-b,host-c,host-d"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let text = stderr(&output);
    assert!(text.contains("ssh host-b failed: fake@host-b: Permission denied (publickey). Vade never edits ~/.ssh."), "{text}");
    assert_eq!(text.matches("Read docs/INSTALL.md for the SSH check command.").count(), 2, "{text}");
    assert!(!text.contains("Run ssh "), "{text}");
    assert!(text.contains("ssh host-d failed: Host key verification failed."), "{text}");
    assert!(!text.contains("did not run the check script"), "{text}");
    assert!(!fake.config().exists());
    assert!(!fake.home.join(".ssh").exists() && !fake.host("host-b").join(".ssh").exists());
}

#[test]
fn a_dry_run_opens_no_connection_and_writes_nothing() {
    let fake = Fake::new("dry");
    fake.wire(&CABLES, false);
    let output = fake.run(&["up", "--sparks", "host-a,host-b", "--dry-run"]);
    assert!(output.status.success(), "{output:?}");
    assert!(!fake.home.join("hosts/calls.log").exists(), "{output:?}");
    assert!(!fake.config().exists());
    let printed = stdout(&output);
    assert!(printed.contains("Setup reads the cable addresses from each Spark when it runs."), "{printed}");
}

#[test]
fn the_config_from_discovery_is_the_config_from_the_same_links_by_hand() {
    let fake = Fake::new("hand");
    fake.wire(&CABLES, false);
    let output = fake.run(&["up", "--sparks", "host-d,host-b,host-a,host-c"]);
    assert!(stdout(&output).contains("Cable order: host-d - host-c - host-b - host-a.\n"), "{output:?}");
    let found = std::fs::read_to_string(fake.config()).unwrap();
    let links = [("host-d", "host-c", "203.0.113.66"), ("host-c", "host-d", "203.0.113.65"), ("host-c", "host-b", "203.0.113.2"), ("host-b", "host-c", "203.0.113.1"), ("host-b", "host-a", "192.0.2.2"), ("host-a", "host-b", "192.0.2.1")]
        .map(|(from, to, address)| format!("{from}:{to}={address}"));
    let mut args = vec!["setup", "--yes", "--hosts", "host-d,host-c,host-b,host-a"];
    for link in &links { args.extend(["--link", link]); }
    std::fs::rename(fake.config(), fake.home.join("first")).unwrap();
    // Public dry runs hide addresses and do not read hostnames. Compare the private files.
    let output = fake.run(&args);
    let hand = std::fs::read_to_string(fake.config()).unwrap_or_else(|error| panic!("{error}: {output:?}"));
    assert_eq!(hand, found);
}

#[test]
fn two_homes_stop_before_anything_is_written() {
    let fake = Fake::new("homes");
    fake.wire(&CABLES[..1], false);
    let path = fake.host("host-b").join("interfaces");
    std::fs::write(&path, std::fs::read_to_string(&path).unwrap().replace("home /home/fake", "home /home/other")).unwrap();
    let output = fake.run(&["up", "--sparks", "host-a,host-b"]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(stderr(&output).contains("host-a has the home folder /home/fake, and host-b has /home/other."), "{output:?}");
    assert!(!fake.config().exists());
}

/// One Spark, then two, then four: each new list wins, the old files stay, and
/// `vade up` alone reruns the last one.
#[test]
fn each_new_list_replaces_the_setup_and_vade_up_alone_reruns_it() {
    use std::os::unix::fs::PermissionsExt;
    let fake = Fake::new("replace");
    fake.wire(&CABLES, false);
    let root = fake.home.join(".vade");
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    std::fs::write(root.join("config"), "format=vade-owner-v1\n").unwrap();
    let output = fake.run(&["up", "--sparks", "host-a,host-b"]);
    assert!(stdout(&output).contains(&format!("Replacing the one-Spark setup with two Sparks. The old setup is now in {:?}.\n", root.join("config.local").display().to_string())), "{output:?}");
    assert_eq!(std::fs::read_to_string(root.join("config.local")).unwrap(), "format=vade-owner-v1\n");
    assert!(!root.join("config").exists());
    let two = std::fs::read_to_string(fake.config()).unwrap();
    let before = fake.home.join("xdg/vade/config.before");
    let output = fake.run(&["up", "--sparks", "host-c,host-d,host-a,host-b"]);
    assert!(stdout(&output).contains(&format!("Replacing the two-Spark setup with four Sparks. The old setup is now in {:?}.\n", before.display().to_string())), "{output:?}");
    assert_eq!(std::fs::read_to_string(&before).unwrap(), two, "nothing is deleted");
    // Neither end was typed first, and this direction keeps four of the six typed pairs.
    assert_eq!(fake.order(), ["host-d", "host-c", "host-b", "host-a"]);
    assert!(stdout(&fake.run(&["status"])).contains("Last setup: 4 Sparks (host-d, host-c, host-b, host-a).\n"));
    // vade up alone takes the four. A dry run shows the plan on any computer.
    let output = fake.run(&["up", "--dry-run"]);
    let printed = stdout(&output);
    assert!(output.status.success(), "{output:?}");
    assert!(printed.lines().any(|line| line.split_whitespace().collect::<Vec<_>>().join(" ") == "Machines host-d, host-c, host-b, host-a"), "{printed}");
    assert!(printed.lines().any(|line| line.split_whitespace().collect::<Vec<_>>().join(" ") == "Context 131072 tokens"), "{printed}");
}

#[test]
fn the_same_sparks_in_the_same_order_rewrite_nothing() {
    let fake = Fake::new("same");
    fake.wire(&CABLES, false);
    fake.run(&["up", "--sparks", "host-a,host-b"]);
    let first = std::fs::read_to_string(fake.config()).unwrap();
    let calls = |fake: &Fake| std::fs::read_to_string(fake.home.join("hosts/calls.log")).unwrap().matches("vade-cables-end").count();
    let probes = calls(&fake);
    let output = fake.run(&["up", "--sparks", "host-a,host-b"]);
    let printed = stdout(&output);
    assert!(!printed.contains("Replacing") && !printed.contains("Writing the controller config"), "{printed}");
    assert!(printed.contains("Cable order: host-a - host-b.\n"), "{printed}");
    // The first install failed here, so the same command runs it again.
    assert!(printed.contains("Setup step 1 of 3: vade install"), "{printed}");
    assert_eq!(std::fs::read_to_string(fake.config()).unwrap(), first);
    assert_eq!(calls(&fake), probes, "no new probe");
    assert!(!fake.home.join("xdg/vade/config.before").exists());
    // Another order finds the same chain, and that is the setup in place too.
    let order = fake.run(&["up", "--sparks", "host-b,host-a"]);
    assert!(stdout(&order).contains("Replacing the two-Spark setup with two Sparks."), "the typed order is a chain, and rank 0 changes: {order:?}");
    assert_eq!(fake.order(), ["host-b", "host-a"]);
}

#[test]
fn a_shuffled_list_of_the_sparks_in_place_rewrites_nothing() {
    let fake = Fake::new("shuffled-same");
    fake.wire(&CABLES, false);
    fake.run(&["up", "--sparks", "host-a,host-b,host-c,host-d"]);
    let first = std::fs::read_to_string(fake.config()).unwrap();
    let output = fake.run(&["up", "--sparks", "host-c,host-a,host-d,host-b"]);
    let printed = stdout(&output);
    assert!(printed.contains("Cable order: host-a - host-b - host-c - host-d.\n") && !printed.contains("Replacing"), "{printed}");
    assert_eq!(std::fs::read_to_string(fake.config()).unwrap(), first);
}

#[test]
fn fp8_stays_with_its_sparks_and_vade_up_alone_keeps_it() {
    let fake = Fake::new("fp8");
    fake.wire(&CABLES, false);
    fake.run(&["up", "--fp8", "--sparks", "host-a,host-b"]);
    let config = std::fs::read_to_string(fake.config()).unwrap();
    assert!(config.starts_with("# Written by vade setup. Start: vade up --fp8 --hosts host-a,host-b\n") && config.contains("model = qwen3.8-flash-next-fp8\n"), "{config}");
    let output = fake.run(&["up", "--dry-run"]);
    assert!(output.status.success() && stdout(&output).contains("qwen3.8-flash-next-fp8"), "{output:?}");
    // The list with no flag keeps the model of those Sparks.
    let output = fake.run(&["up", "--sparks", "host-a,host-b"]);
    assert!(!stdout(&output).contains("Replacing"), "{output:?}");
    assert_eq!(std::fs::read_to_string(fake.config()).unwrap(), config);
    // A flag that names the other model is a new setup.
    let output = fake.run(&["up", "--fp4-local", "--sparks", "host-a,host-b"]);
    assert!(stdout(&output).contains("Replacing the two-Spark setup with two Sparks."), "{output:?}");
    assert!(std::fs::read_to_string(fake.config()).unwrap().contains("model = qwen3.8-flash-next-fp4-local\n"));
}

#[test]
fn a_hand_written_config_is_not_touched_by_a_chain_it_holds() {
    let fake = Fake::new("hand-config");
    fake.wire(&CABLES, false);
    std::fs::create_dir_all(fake.config().parent().unwrap()).unwrap();
    let hand = format!("[global]\nsource = installed-bundle\nstage_dir = {0}/stage\nstate_dir = {0}/state\nmodel_revision = {1}\n\n[machine one]\nssh = host-a\nfabric.two = 192.0.2.1\n\n[machine two]\nssh = host-b\nfabric.one = 192.0.2.2\nfabric.three = 203.0.113.1\n\n[machine three]\nssh = host-c\nfabric.two = 203.0.113.2\n", fake.home.display(), "b".repeat(40));
    std::fs::write(fake.config(), &hand).unwrap();
    let output = fake.run(&["up", "--sparks", "host-a,host-b", "--dry-run"]);
    assert!(!stdout(&output).contains("Replacing"), "{output:?}");
    assert_eq!(std::fs::read_to_string(fake.config()).unwrap(), hand);
    // vade up alone on a config by hand keeps its one Spark.
    let output = fake.run(&["up", "--dry-run"]);
    assert!(stdout(&output).lines().any(|line| line.split_whitespace().collect::<Vec<_>>().join(" ") == "Machines one"), "{output:?}");
}
