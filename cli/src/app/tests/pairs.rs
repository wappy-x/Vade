//! `vade up --replicas N --pairs`: N two-Spark tensor splits behind one gateway.
use super::*;

#[test]
fn pairs_start_world_two_servers_with_one_tunnel_per_pair_behind_one_gateway() {
    let mut f = Fixture::fp4(4, false);
    f.app.up(&fp4_options(&["--replicas", "2", "--pairs"])).unwrap();
    let s = Session::load(&f.app.config.state_dir).unwrap().unwrap(); let (port, serve, tp) = (s.local_port, s.serve_port, f.app.config.tp_port);
    assert!(s.replicas && s.replica_sparks == 2 && s.chained() && s.two_sparks() && s.topology() == "ring" && s.servers().collect::<Vec<_>>() == [0, 2]);
    // Each pair is its own tensor split: rank 0 and 1 of world 2, linked only to its partner.
    let link = format!("127.0.0.1:{tp} 127.0.0.1:{tp}");
    assert_eq!((0..4).map(|i| s.engine_args(i)[0..4].join(" ")).collect::<Vec<_>>(), [format!("0 2 {link}"), format!("1 2 {link}"), format!("0 2 {link}"), format!("1 2 {link}")]);
    let starts: Vec<_> = f.app.runner.calls.iter().map(|e| text::parse(&String::from_utf8_lossy(&e.stdin)).unwrap_or_default()).filter(|r| r.get("op").is_some_and(|op| op == "start")).collect();
    assert_eq!(starts.iter().map(|r| format!("{}/{}", r["env.FLASH_SERVE_RANK"], r["env.FLASH_SERVE_WORLD_SIZE"])).collect::<Vec<_>>(), ["0/2", "1/2", "0/2", "1/2"]);
    assert!(starts.iter().all(|r| r["env.SEQPAR_RDMA"] == "1" && r["env.TP_TOPOLOGY"] == "ring"), "{starts:?}");
    // Only each pair's first machine serves HTTP, so only a and c get a tunnel.
    let tunnels: Vec<_> = f.app.runner.calls.iter().filter(|e| e.label == "controller tunnel start").map(|e| e.args[e.args.len() - 2..].join(" ")).collect();
    assert_eq!(tunnels, [format!("127.0.0.1:{}:127.0.0.1:{serve} a-ssh", port + 1), format!("127.0.0.1:{}:127.0.0.1:{serve} c-ssh", port + 2)]);
    assert_eq!(f.app.runner.gateway.clone().unwrap(), format!("vade-gateway --bind 127.0.0.1:{port} --upstream http://127.0.0.1:{} --upstream http://127.0.0.1:{} --stable-prefix", port + 1, port + 2));
    assert_eq!(super::super::chats_label(&s), "8 per replica, 16 across 2 replicas (engine-admitted at this context)");
    let capture = crate::output::testing::Capture::default();
    let printed = capture.bytes.clone();
    crate::output::with_writer(capture, || f.app.execute(Command::Status)).unwrap();
    let printed = String::from_utf8(printed.borrow().clone()).unwrap();
    assert!(printed.contains("tensor-split pairs behind the local gateway"), "{printed}");
    f.app.down().unwrap();
    assert!(f.app.runner.gateway.is_none() && !f.recorded(), "down stops the gateway and clears the record");
}

#[test]
fn pairs_need_replicas_an_even_host_list_and_a_link_inside_each_pair() {
    for (args, error) in [(&["--pairs"][..], "Use --pairs with --replicas N"), (&["--replicas", "2", "--pairs", "--hosts", "a,b"], "(2N with --pairs)"),
        (&["--replicas", "13", "--pairs"], "Use --replicas 2..12 with --pairs")] {
        let args = ["up"].into_iter().chain(args.iter().copied()).map(str::to_string).collect::<Vec<_>>();
        assert!(crate::cli::parse(&args).unwrap_err().contains(error), "{args:?}");
    }
    let options = fp4_options(&["--replicas", "2", "--pairs", "--hosts", "a,b,c,d"]);
    assert_eq!((options.sparks, options.replica_sparks, options.hosts.len()), (None, 2, 4));
    assert_eq!(fp4_options(&["--replicas", "2", "--pairs"]).sparks, Some(4));
    // One-Spark replicas need no fabric, but a pair is a tensor split and needs its own link.
    let mut f = Fixture::fp4(4, false);
    f.app.config.machines[2].fabric.remove("d");
    f.app.config.machines[3].fabric.remove("c");
    let error = f.app.up(&fp4_options(&["--replicas", "2", "--pairs"])).unwrap_err();
    assert!(error.starts_with("Pair 2 is not an arc of the cabling: c and d have no fabric link"), "{error}");
    // The refusal comes before any record, remote call, tunnel or gateway.
    assert!(!f.recorded() && f.app.runner.gateway.is_none(), "nothing recorded or started");
    assert!(!f.app.runner.calls.iter().any(|e| e.label.ends_with(" probe") || e.label.ends_with(" start") || e.program == "ssh"), "{:?}", f.app.runner.calls.iter().map(|e| &e.label).collect::<Vec<_>>());
    f.app.up(&fp4_options(&["--replicas", "4"])).unwrap();
    f.app.down().unwrap();
}

#[test]
fn a_pairs_record_round_trips_and_one_spark_replica_records_keep_their_fields() {
    let mut f = Fixture::fp4(4, false);
    f.app.up(&fp4_options(&["--replicas", "2", "--pairs"])).unwrap();
    let s = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(s.encode().contains("replica_sparks=2\n") && s.replica_sparks == 2, "{}", s.encode());
    // A pairs record restarts as pairs.
    let recorded = super::super::resident::recorded_options(&s);
    assert_eq!((recorded.replicas, recorded.replica_sparks), (true, 2));
    f.app.down().unwrap();
    f.app.up(&fp4_options(&["--replicas", "2"])).unwrap();
    let s = Session::load(&f.app.config.state_dir).unwrap().unwrap();
    assert!(!s.encode().contains("replica_sparks") && s.replica_sparks == 1 && !s.chained() && s.servers().count() == 2);
    assert!(!f.app.runner.gateway.clone().unwrap().contains("--stable-prefix"), "one-Spark replicas keep their gateway flags");
    f.app.down().unwrap();
}
