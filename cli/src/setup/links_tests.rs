use super::*;

#[test]
fn setup_ssh_stop_fix_uses_the_destination_kind() {
    struct Failed;
    impl Runner for Failed {
        fn run(&mut self, _: &runner::Exec) -> Result<runner::Output, String> {
            Ok(runner::Output { success: false, stdout: "Host key verification failed.\n".into() })
        }
        fn interactive(&mut self, _: &runner::Exec) -> Result<bool, String> { unreachable!() }
        fn port_free(&mut self, _: u16) -> Result<(), String> { unreachable!() }
        fn pause(&mut self, _: std::time::Duration) { unreachable!() }
    }
    let _host = runner::fake_host(runner::Host::default());
    for target in ["user@192.0.2.10", "203.0.113.10", "fixture-ssh"] {
        let names = crate::config::PublicNames::hosts(&[target.into()]);
        let error = names.run(|| probe(&mut Failed, target, "vade setup")).unwrap_err();
        assert!(error.contains("Host key verification failed."), "{error}");
        if target == "fixture-ssh" {
            assert!(error.contains("Run ssh fixture-ssh true and accept the key") && error.ends_with("then run vade setup again."), "{error}");
        } else {
            assert!(error.contains("Read docs/INSTALL.md for the SSH check command."), "{error}");
            assert!(!error.contains(" true") && !error.contains("ssh-keyscan") && !error.contains(target), "{error}");
        }
    }
}

fn iface(name: &str, cidr: &str, speed: u64, state: &str) -> String { format!("iface {name} {cidr} {speed} 9000 mlx5_0 {state}\n") }

/// A Spark with these probe lines, reached over SSH at `reached` (or `-`).
fn spark(lines: &[String], reached: &str) -> Spark {
    parse("host", &format!("{}home /home/you\nreached {reached}\n{END}\n", lines.concat())).unwrap()
}

fn up(name: &str, cidr: &str) -> String { iface(name, cidr, 200_000, "ACTIVE") }

/// Every order of `0..count`.
fn orders(count: usize) -> Vec<Vec<usize>> {
    if count == 0 { return vec![Vec::new()]; }
    orders(count - 1).into_iter().flat_map(|order| (0..count).map(move |at| { let mut order = order.clone(); order.insert(at, count - 1); order })).collect()
}

/// `joined` for the cables in `edges`.
fn wired(edges: &[(usize, usize)]) -> impl Fn(usize, usize) -> bool + '_ {
    move |a, b| edges.contains(&(a, b)) || edges.contains(&(b, a))
}

#[test]
fn the_script_is_the_doctor_probe_with_three_more_lines() {
    let text = script();
    assert!(text.starts_with(fabric::PROBE_SCRIPT), "{text}");
    assert!(text.ends_with("printf 'vade-cables-end\\n'\n"), "{text}");
    let output = std::process::Command::new("sh").args(["-n", "-c", &text]).status().unwrap();
    assert!(output.success());
}

#[test]
fn only_up_fabric_ports_count() {
    let found = spark(&[
        up("enp1s0f0np0", "192.0.2.1/24"),
        // A LAN port on the same driver, a port with no link, and a port with no RDMA device.
        iface("enp5s0", "203.0.113.1/24", 10_000, "ACTIVE"),
        iface("enp1s0f1np1", "198.51.100.1/24", 0, "DOWN"),
        iface("enp2s0", "198.51.100.9/24", 200_000, "DOWN"),
        "iface enp3s0 192.0.2.77/24 200000 9000 - -\n".into(),
    ], "-");
    assert_eq!(found.ports.iter().map(|port| port.name.as_str()).collect::<Vec<_>>(), ["enp1s0f0np0"]);
    assert_eq!(found.home, "/home/you");
    assert_eq!(found.reached, None);
}

#[test]
fn a_script_that_did_not_end_gives_no_answer() {
    assert_eq!(parse("host", &up("enp1s0f0np0", "192.0.2.1/24")), None);
    assert_eq!(parse("host", ""), None);
    // A login banner around the lines changes nothing.
    assert_eq!(parse("host", &format!("Welcome\n{}{END}\n", up("enp1s0f0np0", "192.0.2.1/24"))).unwrap().ports.len(), 1);
}

#[test]
fn the_destination_address_counts_as_reached_with_no_ssh_session() {
    assert_eq!(parse("you@192.0.2.7", &format!("reached -\n{END}\n")).unwrap().reached, Some("192.0.2.7".parse().unwrap()));
    assert_eq!(parse("spark-a", &format!("reached -\n{END}\n")).unwrap().reached, None);
    assert_eq!(parse("spark-a", &format!("reached 192.0.2.8\n{END}\n")).unwrap().reached, Some("192.0.2.8".parse().unwrap()));
}

#[test]
fn one_shared_subnet_is_the_cable_and_its_prefix_is_each_ports_own() {
    let ends = |a: &str, b: &str| (a.parse().unwrap(), b.parse().unwrap());
    let a = spark(&[up("enp1s0f0np0", "192.0.2.1/24")], "-");
    let b = spark(&[up("enp1s0f1np1", "192.0.2.2/24")], "-");
    assert_eq!(cable(&a, &b), Cable::One(ends("192.0.2.1", "192.0.2.2")));
    assert_eq!(cable(&b, &a), Cable::One(ends("192.0.2.2", "192.0.2.1")));
    // A /30 is a subnet too, and a /24 around it is a different one.
    let (c, d) = (spark(&[up("p0", "203.0.113.1/30")], "-"), spark(&[up("p1", "203.0.113.2/30")], "-"));
    assert_eq!(cable(&c, &d), Cable::One(ends("203.0.113.1", "203.0.113.2")));
    assert_eq!(cable(&c, &spark(&[up("p1", "203.0.113.2/24")], "-")), Cable::None);
    assert_eq!(cable(&a, &c), Cable::None);
    // The same address on both ends is no cable.
    assert_eq!(cable(&a, &spark(&[up("p1", "192.0.2.1/24")], "-")), Cable::None);
}

#[test]
fn a_dual_rail_cable_gives_its_rail_0_address_whatever_the_order() {
    let a = spark(&[up("enP2p1s0f0np0", "198.51.100.1/24"), up("enp1s0f0np0", "192.0.2.1/24")], "-");
    let b = spark(&[up("enp1s0f1np1", "192.0.2.2/24"), up("enP2p1s0f1np1", "198.51.100.2/24")], "-");
    assert_eq!(cable(&a, &b), Cable::One(("192.0.2.1".parse().unwrap(), "192.0.2.2".parse().unwrap())));
}

#[test]
fn a_second_address_on_one_port_leaves_the_choice_to_the_user() {
    let a = spark(&[up("p0", "192.0.2.1/24"), up("p0", "198.51.100.1/24")], "-");
    let b = spark(&[up("p1", "192.0.2.2/24"), up("p1", "198.51.100.2/24")], "-");
    let Cable::Unclear(found) = cable(&a, &b) else { panic!() };
    assert_eq!(found.len(), 2);
    assert_eq!(unclear("a", "b", &cable(&a, &b)), "a and b share 2 addresses: 192.0.2.1 and 192.0.2.2, 198.51.100.1 and 198.51.100.2");
}

#[test]
fn the_ssh_port_counts_only_when_it_is_the_one_shared_port() {
    let lan = (up("lan0", "203.0.113.1/24"), up("lan0", "203.0.113.2/24"));
    let wire = (up("enp1s0f0np0", "192.0.2.1/24"), up("enp1s0f1np1", "192.0.2.2/24"));
    let a = spark(&[lan.0.clone(), wire.0.clone()], "203.0.113.1");
    let b = spark(&[lan.1.clone(), wire.1.clone()], "203.0.113.2");
    assert_eq!(cable(&a, &b), Cable::One(("192.0.2.1".parse().unwrap(), "192.0.2.2".parse().unwrap())));
    // One link for SSH and the engine, as a rented pair has.
    let a = spark(&[lan.0], "203.0.113.1");
    let b = spark(&[lan.1], "203.0.113.2");
    assert_eq!(cable(&a, &b), Cable::One(("203.0.113.1".parse().unwrap(), "203.0.113.2".parse().unwrap())));
}

#[test]
fn the_given_order_wins_when_it_is_a_chain() {
    // A closed ring of four: every rotation and both directions fit.
    let ring = [(0, 1), (1, 2), (2, 3), (3, 0)];
    assert_eq!(chain(4, &wired(&ring)), Ok(vec![0, 1, 2, 3]));
    // Every Spark on one switch: any order fits.
    assert_eq!(chain(24, &|a, b| a != b), Ok((0..24).collect()));
    assert_eq!(chain(1, &wired(&[])), Ok(vec![0]));
}

#[test]
fn a_shuffled_list_finds_the_only_chain_from_every_order_of_two_to_eight_sparks() {
    for count in 2..=8 {
        // The real line of cables, in cable order, as positions in each typed list.
        for order in orders(count) {
            let mut at = vec![0; count];
            for (cable_place, typed) in order.iter().enumerate() { at[*typed] = cable_place; }
            let found = chain(count, &|a, b| at[a].abs_diff(at[b]) == 1).unwrap();
            let forward: Vec<usize> = (0..count).map(|place| at.iter().position(|value| *value == place).unwrap()).collect();
            let backward: Vec<usize> = forward.iter().rev().copied().collect();
            assert!(found == forward || found == backward, "{order:?}: {found:?}");
            // Two directions fit: the one that starts with the first given name wins, if one does.
            if forward[0] == 0 { assert_eq!(found, forward); }
            if backward[0] == 0 { assert_eq!(found, backward); }
        }
    }
}

#[test]
fn a_ring_in_a_shuffled_list_follows_the_given_order_the_most() {
    let ring = [(0, 2), (2, 1), (1, 3), (3, 0)];
    // Typed 0 1 2 3 on the ring 0 2 1 3: both directions start at 0.
    // 0 2 1 3 keeps 5 of the 6 pairs in their given order, and 0 3 1 2 keeps 4.
    assert_eq!(chain(4, &wired(&ring)), Ok(vec![0, 2, 1, 3]));
}

#[test]
fn a_second_ring_keeps_the_direction_nearer_the_given_order() {
    // The ring 0 1 3 2: 0 1 3 2 keeps 5 pairs, and 0 2 3 1 keeps 4.
    let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
    assert_eq!(chain(4, &wired(&edges)), Ok(vec![0, 1, 3, 2]));
}

#[test]
fn the_first_given_name_can_sit_inside_the_chain() {
    // The cables run 1 - 0 - 2. No chain starts with Spark 0.
    assert_eq!(chain(3, &wired(&[(1, 0), (0, 2)])), Ok(vec![1, 0, 2]));
}

#[test]
fn no_chain_names_the_two_sparks_that_stop_it() {
    // Spark 3 has no cable.
    assert_eq!(chain(4, &wired(&[(0, 1), (1, 2)])), Err((2, 3)));
    // A star: three Sparks hang on Spark 0, so no chain holds all four.
    assert_eq!(chain(4, &wired(&[(0, 1), (0, 2), (0, 3)])), Err((2, 3)));
    // Two pairs with no cable between them.
    assert_eq!(chain(4, &wired(&[(0, 1), (2, 3)])), Err((1, 2)));
    assert_eq!(chain(2, &wired(&[])), Err((0, 1)));
}

#[test]
fn a_large_group_with_many_cables_ends_at_the_step_limit() {
    let found = chain(24, &|a, b| a != b && a + b != 1).unwrap();
    let mut sorted = found.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..24).collect::<Vec<_>>());
    assert_eq!(found[0], 0);
}
