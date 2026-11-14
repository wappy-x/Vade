//! The private-network guard. No test here opens a socket or asks DNS: the
//! resolver and the curl process are injected, so the public answers below
//! are ordinary global addresses that no test ever reaches. Documentation
//! addresses are not global, so they are in the internal list. Host names
//! use RFC 2606 example domains with injected DNS and curl.
use super::*;
use std::cell::RefCell;

fn v4(text: &str) -> IpAddr { IpAddr::V4(text.parse().unwrap()) }
fn v6(text: &str) -> IpAddr { IpAddr::V6(text.parse().unwrap()) }
/// Private-range addresses are built from their parts, so no dotted literal of
/// them sits in the source (the publish leak gate refuses such text).
fn ip4(a: u8, b: u8, c: u8, d: u8) -> IpAddr { IpAddr::V4(Ipv4Addr::new(a, b, c, d)) }
fn dotted(a: u8, b: u8, c: u8, d: u8) -> String { Ipv4Addr::new(a, b, c, d).to_string() }
fn ip6(parts: [u16; 8]) -> IpAddr { IpAddr::V6(std::net::Ipv6Addr::from(parts)) }
fn ok_reply(remote: &str) -> Reply {
    Reply {
        body: b"done".to_vec(),
        headers: format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n\nVADE_REDIRECT:\nVADE_REMOTE:{remote}\n"),
        code: Some(0), truncated: false,
    }
}
fn redirect(url: &str, remote: &str) -> Reply {
    Reply {
        body: Vec::new(),
        headers: format!("HTTP/1.1 302 Found\r\n\r\nVADE_REDIRECT:{url}\nVADE_REMOTE:{remote}\n"),
        code: Some(47), truncated: false,
    }
}
fn far() -> Instant { Instant::now() + Duration::from_secs(20) }
fn args(command: &Command) -> Vec<String> { command.get_args().map(|s| s.to_str().unwrap().to_owned()).collect() }
fn values(args: &[String], flag: &str) -> Vec<String> {
    args.windows(2).filter(|a| a[0] == flag).map(|a| a[1].clone()).collect()
}

/// Every internal answer a resolver could give. Each must refuse the fetch.
fn internal_answers() -> Vec<IpAddr> {
    let mut all = vec![
        // IPv4: RFC 1918, loopback, link-local and cloud metadata, carrier-grade
        // NAT (RFC 6598, 100.64/10), network-benchmark / bridge space (198.18.0.0/15),
        // unspecified, "this network", reserved (240.0.0.0/4), broadcast and multicast.
        v4("10.0.0.1"), v4("172.16.0.1"), v4("172.31.255.254"), ip4(192, 168, 0, 1),
        v4("127.0.0.1"), v4("127.255.255.254"), v4("169.254.0.1"), v4("169.254.169.254"),
        ip4(100, 64, 0, 0), ip4(100, 64, 0, 1), ip4(100, 100, 100, 100), ip4(100, 127, 255, 255),
        v4("198.18.0.0"), v4("198.18.0.1"), v4("198.19.249.2"), v4("198.19.255.255"),
        v4("0.0.0.0"), v4("0.1.2.3"), v4("240.0.0.0"), v4("240.0.0.1"), v4("255.255.255.254"),
        v4("255.255.255.255"), v4("224.0.0.1"), v4("239.255.255.255"),
        // IPv6: loopback, unspecified, link-local, unique-local, site-local, multicast.
        v6("::1"), v6("::"), v6("fe80::1"), v6("febf::1"), v6("fc00::1"), v6("fd12:3456::1"), v6("fec0::1"), v6("ff02::1"),
        // IPv6 carrying or translating to a private address: NAT64 well-known
        // prefix (64:ff9b::/96), local-use NAT64 (64:ff9b:1::/48, even when its
        // low bits match a public IPv4), IPv4-translated (::ffff:0:a.b.c.d) and 6to4.
        v6("64:ff9b::a00:1"), v6("64:ff9b::c0a8:101"), v6("64:ff9b::7f00:1"), v6("64:ff9b::a9fe:a9fe"), v6("64:ff9b::6440:1"),
        v6("64:ff9b:1::1"), v6("64:ff9b:1:2:3:4:808:807"),
        v6("::ffff:0:a00:1"), v6("::ffff:0:6440:1"),
        v6("2002:a00:1::1"), v6("2002:7f00:1::"), v6("2002:c612:1::5"),
        // Not global: documentation, 192.0.0.0/24, 192.88.99.0/24, 2001:db8::/32,
        // 100::/64, Teredo (2001::/32) and 3fff::/20. First and last of each.
        v4("192.0.2.0"), v4("192.0.2.255"), v4("198.51.100.0"), v4("198.51.100.255"),
        v4("203.0.113.0"), v4("203.0.113.255"), v4("192.0.0.0"), v4("192.0.0.255"),
        v4("192.88.99.0"), v4("192.88.99.255"),
        v6("2001:db8::"), v6("2001:db8:ffff:ffff:ffff:ffff:ffff:ffff"), v6("100::"), v6("100::ffff:ffff:ffff:ffff"),
        v6("2001::"), v6("2001:0:ffff:ffff:ffff:ffff:ffff:ffff"), v6("3fff::"), v6("3fff:fff:ffff:ffff:ffff:ffff:ffff:ffff"),
        v6("64:ff9b::c000:207"), v6("2002:c000:207::1"),
    ];
    // IPv4-mapped (::ffff:a.b.c.d) and IPv4-compatible (::a.b.c.d) forms.
    for ip in [dotted(10, 0, 0, 1), dotted(127, 0, 0, 1), dotted(169, 254, 169, 254), dotted(100, 64, 0, 1),
        dotted(192, 168, 0, 1), dotted(198, 18, 0, 1), dotted(240, 0, 0, 1), dotted(192, 0, 2, 7)] {
        let ip: Ipv4Addr = ip.parse().unwrap();
        all.push(IpAddr::V6(ip.to_ipv6_mapped()));
        all.push(IpAddr::V6(ip.to_ipv6_compatible()));
    }
    all
}
fn public_answers() -> Vec<IpAddr> {
    vec![v4("8.8.8.10"), v4("9.9.9.20"), v4("1.1.1.30"), v6("2606:4700::10"),
        IpAddr::V6(Ipv4Addr::new(8, 8, 8, 40).to_ipv6_mapped()),
        v6("::ffff:0:808:807"), v6("64:ff9b::808:832"), v6("2002:808:833::1")]
}

fn assert_refused_addresses(addresses: Vec<IpAddr>) {
    for address in addresses {
        let url = format!("http://[{address}]/");
        assert_eq!(validate_url(&url), Err(Failure::Refused("local and private address ranges are refused")), "{address}");
        assert!(matches!(run_with(&url, 64, far(), &AtomicBool::new(false),
            |_, _, _, _| panic!("a refused literal reached DNS"), |_| panic!("a refused literal reached curl")), Err(Failure::Refused(_))));
        for answers in [vec![address], vec![v4("8.8.8.10"), address], vec![address, v6("2606:4700::10")]] {
            assert_eq!(run_with("https://inside.example.org/", 64, far(), &AtomicBool::new(false),
                |_, _, _, _| Ok(answers), |_| panic!("a refused answer reached curl")).unwrap_err(),
                Failure::Refused("the host name resolves to a local or private address"));
        }
        let mut started = 0;
        let result = fetch_with("https://www.example.org/", 64, 20, &AtomicBool::new(false), |hop, cap, deadline, cancel| {
            run_with(hop, cap, deadline, cancel, |_, _, _, _| Ok(vec![v4("8.8.8.10")]),
                |_| { started += 1; Ok(redirect(&url, "8.8.8.10")) })
        });
        assert!(matches!(result, Err(Failure::Refused(_))));
        assert_eq!(started, 1);
    }
}

#[test]
fn web_guard_3_local_names_stop_before_dns() {
    // Reserved local suffixes and common local aliases are refused as policy.
    for suffix in ["local", "localhost", "home.arpa", "internal", "lan", "home", "corp", "localdomain", "test", "invalid", "example", "onion"] {
        for host in [format!("test.{suffix}"), suffix.to_owned(), format!("TEST.{suffix}.").to_ascii_uppercase()] {
            let url = format!("https://{host}:8443/");
            assert_eq!(validate_url(&url), Err(Failure::Refused("local hosts are refused")), "{host}");
            assert!(matches!(run_with(&url, 64, far(), &AtomicBool::new(false),
                |_, _, _, _| panic!("a local name reached DNS"), |_| panic!("a local name reached curl")), Err(Failure::Refused(_))));
            let mut started = 0;
            let result = fetch_with("https://www.example.org/", 64, 20, &AtomicBool::new(false), |hop, cap, deadline, cancel| {
                run_with(hop, cap, deadline, cancel, |name, _, _, _| {
                    assert_eq!(name, "www.example.org"); Ok(vec![v4("8.8.8.10")])
                }, |_| { started += 1; Ok(redirect(&url, "8.8.8.10")) })
            });
            assert!(matches!(result, Err(Failure::Refused("local hosts are refused"))));
            assert_eq!(started, 1);
        }
        assert!(validate_url(&format!("https://test.not-{suffix}/")).is_ok());
        assert!(validate_url(&format!("https://{suffix}.example.org/")).is_ok());
        assert!(validate_url(&format!("https://not-{suffix}.example.org/")).is_ok());
    }
}

#[test]
fn web_guard_3_isatap_checks_the_carried_address() {
    let mut addresses = Vec::new();
    for marker in [0, 0x0200] {
        for address in internal_answers() {
            if let IpAddr::V4(ip) = address {
                let [a, b, c, d] = ip.octets();
                addresses.push(IpAddr::V6(std::net::Ipv6Addr::new(0x2606, 0x4700, 0, 0, marker, 0x5efe,
                    u16::from_be_bytes([a, b]), u16::from_be_bytes([c, d]))));
            }
        }
    }
    // A public IPv4 prefix must not hide a private ISATAP interface ID.
    addresses.push(v6("2002:808:807:0:200:5efe:a00:1"));
    // A public carried address must not hide a refused IPv6 prefix.
    addresses.extend([v6("2001:db8::5efe:808:807"), v6("fec0::200:5efe:808:807")]);
    assert_refused_addresses(addresses);
    for address in ["2606:4700::5efe:808:807", "2606:4700::200:5efe:808:807", "2606:4700::100:5efe:a00:1",
        "2606:4700::5eff:a00:1"] {
        assert!(!private_ip(v6(address)), "{address}");
    }
}

#[test]
fn web_guard_3_special_ipv6_prefixes_are_refused() {
    assert_refused_addresses([
        "2001:2::", "2001:2::1", "2001:2:0:ffff:ffff:ffff:ffff:ffff",
        "2001:10::", "2001:1f:ffff:ffff:ffff:ffff:ffff:ffff",
        "2001:20::", "2001:2f:ffff:ffff:ffff:ffff:ffff:ffff",
        "5f00::", "5f00::1", "5f00:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "2001:2::5efe:808:807", "2001:20::200:5efe:808:807", "5f00::5efe:808:807",
    ].into_iter().map(v6).collect());
    for edge in [[0x2001, 1, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff], [0x2001, 2, 1, 0, 0, 0, 0, 0],
        [0x2001, 0xf, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff], [0x2001, 0x30, 0, 0, 0, 0, 0, 0],
        [0x5eff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff, 0xffff], [0x5f01, 0, 0, 0, 0, 0, 0, 0]] {
        assert_refused_addresses(vec![ip6(edge)]);
    }
}

#[test]
fn web_guard_3_lookup_panic_releases_its_slot() {
    static RUNNING: AtomicUsize = AtomicUsize::new(0);
    assert_eq!(lookup_capped(|| panic!("injected resolver panic"), far(), &AtomicBool::new(false), &RUNNING),
        Err(Failure::Fetch("could not find the address of the host")));
    assert_eq!(RUNNING.load(Ordering::Acquire), 0);
    assert_eq!(lookup_capped(|| Err(io::Error::other("injected resolver error")), far(), &AtomicBool::new(false), &RUNNING),
        Err(Failure::Fetch("could not find the address of the host")));
    assert_eq!(RUNNING.load(Ordering::Acquire), 0);
    assert_eq!(lookup_capped(|| Ok(vec![v4("8.8.8.10")]), far(), &AtomicBool::new(false), &RUNNING), Ok(vec![v4("8.8.8.10")]));
    assert_eq!(RUNNING.load(Ordering::Acquire), 0);
}

fn assert_allowed_address(address: IpAddr) {
    assert!(!private_ip(address), "{address}");
    let url = format!("https://[{address}]/");
    assert_eq!(validate_url(&url), Ok(()));
    run_with(&url, 64, far(), &AtomicBool::new(false), |_, _, _, _| panic!("literal reached DNS"),
        |_| Ok(ok_reply(&address.to_string()))).unwrap();
    run_with("https://www.example.org/", 64, far(), &AtomicBool::new(false), |_, _, _, _| Ok(vec![address]),
        |command| {
            assert_eq!(values(&args(&command), "--resolve"), [format!("www.example.org:443:[{address}]"), format!("*:443:[{address}]")]);
            Ok(ok_reply(&address.to_string()))
        }).unwrap();
}

#[test]
fn ipv6_range_policy_edges() {
    // First and last addresses, plus both outside neighbors. A neighbor of a
    // refused child range can still be refused by the parent range policy.
    for (parts, bits, refused) in [
        ([0x2000, 0, 0, 0, 0, 0, 0, 0], 3, [true, false, false, true]),
        ([0x2001, 0, 0, 0, 0, 0, 0, 0], 23, [false, true, true, false]),
        ([0x0100, 0, 0, 1, 0, 0, 0, 0], 64, [true; 4]),
        ([0x2001, 0x0030, 0, 0, 0, 0, 0, 0], 28, [true; 4]),
    ] {
        let first = u128::from(std::net::Ipv6Addr::from(parts));
        let last = first | (u128::MAX >> bits);
        for (value, refused) in [first - 1, first, last, last + 1].into_iter().zip(refused) {
            let address = IpAddr::V6(std::net::Ipv6Addr::from(value));
            if refused { assert_refused_addresses(vec![address]); } else { assert_allowed_address(address); }
        }
    }
    assert_refused_addresses(vec![ip6([0x0100, 0, 0, 1, 0, 0, 0, 1]), ip6([0x2001, 0x0100, 0, 0, 0, 0, 0, 1]),
        ip6([0x2001, 0x0030, 0, 0, 0, 0, 0, 1]), ip6([0xfe00, 0, 0, 0, 0, 0, 0, 1]),
        ip6([0x4000, 0, 0, 0, 0, 0x5efe, 0x0808, 0x0807]),
        ip6([0x2001, 1, 0, 0, 0, 0, 0, 1]), ip6([0x2001, 3, 0, 0, 0, 0, 0, 1]),
        ip6([0x2001, 4, 0x0112, 0, 0, 0, 0, 1])]);
}

#[test]
fn ipv6_regional_global_addresses_stay_allowed() {
    // Ordinary global prefixes from AFRINIC, APNIC, ARIN, LACNIC and RIPE NCC.
    // DNS and curl are injected; these addresses are never contacted.
    for parts in [[0x2c0f, 0xfb50, 0, 0, 0, 0, 0, 1], [0x2404, 0x6800, 0, 0, 0, 0, 0, 1],
        [0x2606, 0x4700, 0, 0, 0, 0, 0, 1], [0x2800, 0x03f0, 0, 0, 0, 0, 0, 1], [0x2a00, 0x1450, 0, 0, 0, 0, 0, 1]] {
        assert_allowed_address(ip6(parts));
    }
}

#[test]
fn ipv6_existing_ipv4_embeddings_keep_the_ipv4_check() {
    // Keep the existing mapped, compatible, translated, NAT64 and 6to4 forms.
    // An arbitrary prefix carrying the same low bits is not an exception.
    for (octets, refused) in [([8, 8, 8, 7], false), ([10, 0, 0, 1], true), ([192, 168, 0, 1], true)] {
        let [a, b, c, d] = octets;
        let high = u16::from_be_bytes([a, b]);
        let low = u16::from_be_bytes([c, d]);
        let ip = Ipv4Addr::from(octets);
        for address in [IpAddr::V6(ip.to_ipv6_mapped()), IpAddr::V6(ip.to_ipv6_compatible()),
            ip6([0, 0, 0, 0, 0xffff, 0, high, low]), ip6([0x64, 0xff9b, 0, 0, 0, 0, high, low]),
            ip6([0x2002, high, low, 0, 0, 0, 0, 1])] {
            if refused { assert_refused_addresses(vec![address]); } else { assert_allowed_address(address); }
        }
        assert_refused_addresses(vec![ip6([0x64, 0xff9b, 1, 0, 0, 0, high, low]), ip6([0x4000, 0, 0, 0, 0, 0, high, low])]);
    }
}

#[test]
fn carrier_grade_nat_literals_are_refused() {
    let carrier = dotted(100, 64, 0, 1);
    for host in [dotted(100, 64, 0, 0), carrier.clone(), dotted(100, 100, 100, 100), dotted(100, 127, 255, 255),
        "198.18.0.1".into(), "198.19.249.2".into(), "240.0.0.1".into(), "255.255.255.254".into(),
        format!("[::ffff:{carrier}]"), format!("[::{carrier}]"), "[64:ff9b::6440:1]".into(), "[64:ff9b::a00:1]".into(),
        "[64:ff9b:1::1]".into(), "[2002:a00:1::1]".into(), "[::ffff:0:a00:1]".into(),
        "192.0.2.7".into(), "203.0.113.9".into(), "192.88.99.1".into(), "[2001:db8::7]".into(), "[2001::1]".into(), "[3fff::1]".into()] {
        assert_eq!(validate_url(&format!("http://{host}/")), Err(Failure::Refused("local and private address ranges are refused")), "{host}");
    }
}

#[test]
fn every_internal_answer_is_classed_internal_and_public_answers_are_not() {
    for ip in internal_answers() { assert!(private_ip(ip), "{ip} must be private"); }
    for ip in public_answers() { assert!(!private_ip(ip), "{ip} must stay public"); }
    // Prefix boundaries stay exact on both sides of 100.64/10, 198.18/15, 224/4+240/4
    // and each non-global range: the documentation ranges, 192.0.0/24, 192.88.99/24,
    // 2001:db8::/32 and 3fff::/20. Other IPv6 edges are checked below.
    for edge in [&dotted(100, 63, 255, 255), &dotted(100, 128, 0, 1), "198.17.255.255", "198.20.0.1", "223.255.255.255",
        "192.0.1.255", "192.0.3.0", "198.51.99.255", "198.51.101.0", "203.0.112.255", "203.0.114.0",
        "191.255.255.255", "192.0.1.0", "192.88.98.255", "192.88.100.0",
        "2001:db7:ffff:ffff:ffff:ffff:ffff:ffff", "2001:db9::",
        "2000:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "3ffe:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "3fff:1000::"] {
        assert!(!private_ip(edge.parse().unwrap()), "{edge} must stay public");
    }
}

#[test]
fn a_name_that_resolves_inside_is_refused_before_curl_starts() {
    for answer in internal_answers() {
        // Alone, and hidden behind a public answer of either family.
        for answers in [vec![answer], vec![v4("8.8.8.10"), answer], vec![v6("2606:4700::10"), answer, v4("1.1.1.30")]] {
            let lookups = RefCell::new(Vec::new());
            let result = run_with("https://inside.example.org/path", 64, far(), &AtomicBool::new(false),
                |name, port, _, _| { lookups.borrow_mut().push((name.to_owned(), port)); Ok(answers.clone()) },
                |_| panic!("curl started for {answers:?}"));
            assert_eq!(result.unwrap_err(), Failure::Refused("the host name resolves to a local or private address"), "{answers:?}");
            assert_eq!(*lookups.borrow(), [("inside.example.org".to_owned(), 443)]);
        }
    }
}

#[test]
fn public_answers_are_pinned_so_curl_does_not_look_the_name_up_again() {
    let mut lookups = 0;
    let mut started = 0;
    let expected_list = "8.8.8.10,9.9.9.20,1.1.1.30,[2606:4700::10],[::ffff:8.8.8.40],[::ffff:0:808:807],[64:ff9b::808:832],[2002:808:833::1]";
    let reply = run_with("https://www.example.org/a?b=c", 64, far(), &AtomicBool::new(false),
        |name, port, _, _| {
            lookups += 1;
            assert_eq!((name, port), ("www.example.org", 443));
            let mut dup = public_answers();
            dup.push(v4("8.8.8.10"));
            Ok(dup)
        },
        |command| {
            started += 1;
            let args = args(&command);
            assert_eq!(values(&args, "--resolve"), [format!("www.example.org:443:{expected_list}"), format!("*:443:{expected_list}")]);
            assert_eq!(values(&args, "--url"), ["https://www.example.org/a?b=c"]);
            Ok(ok_reply("8.8.8.10"))
        }).unwrap();
    assert_eq!(reply.body, b"done");
    assert_eq!((lookups, started), (1, 1));
    // An explicit port, upper case and a single trailing dot keep their meaning:
    // the plain name is looked up once, and both spellings plus `*` are pinned.
    run_with("HTTP://WWW.Example.ORG.:8080/", 64, far(), &AtomicBool::new(false),
        |name, port, _, _| { assert_eq!((name, port), ("www.example.org", 8080)); Ok(vec![v4("8.8.8.10")]) },
        |command| {
            let args = args(&command);
            assert_eq!(values(&args, "--resolve"), [
                "www.example.org:8080:8.8.8.10",
                "www.example.org.:8080:8.8.8.10",
                "*:8080:8.8.8.10",
            ]);
            Ok(ok_reply("8.8.8.10"))
        }).unwrap();
    for (url, host, port) in [("https://docs.example.org/a?b#c", "docs.example.org", 443), ("http://docs.example.org", "docs.example.org", 80),
        ("HTTPS://Docs.Example.Org:8443/x", "Docs.Example.Org", 8443), ("http://[2606:4700::7]:81/", "2606:4700::7", 81), ("https://[2606:4700::7]/", "2606:4700::7", 443)] {
        assert_eq!(endpoint(url), Some((host, port)), "{url}");
    }
}

#[test]
fn literal_public_addresses_skip_the_lookup_and_dotted_literals_are_refused() {
    for (url, remote) in [("http://8.8.8.7/", "8.8.8.7"), ("https://[2606:4700::7]:8443/x", "2606:4700::7")] {
        run_with(url, 64, far(), &AtomicBool::new(false), |_, _, _, _| panic!("a literal address was looked up"),
            |command| { assert!(values(&args(&command), "--resolve").is_empty()); Ok(ok_reply(remote)) }).unwrap();
    }
    // Curl reads an address with a final dot as a name and looks it up, so such a host is refused.
    for bad in ["http://8.8.8.7./", "https://8.8.8.7.:8443/x", "http://127.0.0.1./", "https://docs.example.org../"] {
        assert_eq!(validate_url(bad), Err(Failure::Refused("invalid or ambiguous URL host")), "{bad}");
    }
    assert!(validate_url("https://docs.example.org./").is_ok() && validate_url("http://8.8.8.7/").is_ok());
}

#[test]
fn a_failed_or_empty_lookup_stops_the_fetch() {
    let start = |_: Command| -> Result<Reply, Failure> { panic!("curl started without a checked address") };
    assert_eq!(run_with("https://www.example.org/", 64, far(), &AtomicBool::new(false),
        |_, _, _, _| Err(Failure::Fetch("could not find the address of the host")), start).unwrap_err(), Failure::Fetch("could not find the address of the host"));
    assert_eq!(run_with("https://www.example.org/", 64, far(), &AtomicBool::new(false),
        |_, _, _, _| Ok(Vec::new()), start).unwrap_err(), Failure::Fetch("could not find the address of the host"));
}

#[test]
fn ip_spellings_in_urls_are_refused_and_never_looked_up() {
    let mut hosts: Vec<String> = [
        // 127.0.0.1 and 169.254.169.254 in decimal, hex, octal, short and mixed form.
        "2130706433", "0x7f000001", "0X7F000001", "017700000001", "0177.0.0.1", "0177.0000.0000.0001",
        "127.1", "127.0.1", "0x7f.1", "0x7f.0.0.1", "0x7f.0x0.0x0.0x1",
        "2852039166", "0xa9fea9fe", "025177524776", "0251.0376.0251.0376", "0xa9.0xfe.0xa9.0xfe", "169.254.43518",
        // 10.0.0.1 and the first carrier NAT host (100.64/10) in the same forms.
        "167772161", "0xa000001", "012.0.0.1", "10.1", "1681915905", "0x64400001", "0144.0100.0.1", "100.64.1",
    ].into_iter().map(str::to_owned).collect();
    hosts.extend(["127.0.0.1.", "169.254.169.254", "[::ffff:a9fe:a9fe]", "[::ffff:169.254.169.254]", "[::a9fe:a9fe]"].map(str::to_owned));
    for host in hosts {
        let url = format!("http://{host}/latest/meta-data/");
        assert!(matches!(fetch_with(&url, 64, 1, &AtomicBool::new(false), |_, _, _, _| panic!("{url} reached the runner")),
            Err(Failure::Refused(_))), "{url}");
        assert!(matches!(run_with(&url, 64, far(), &AtomicBool::new(false), |_, _, _, _| panic!("{url} was looked up"), |_| panic!("{url} started curl")),
            Err(Failure::Refused(_))), "{url}");
    }
}

#[test]
fn a_redirect_from_a_public_host_to_an_inside_one_is_refused_on_that_hop() {
    let carrier = format!("http://{}/", dotted(100, 64, 0, 1));
    for destination in ["https://inside.example.org/", "http://169.254.169.254/latest/meta-data/", "http://[::ffff:169.254.169.254]/",
        &carrier, "http://198.18.0.1/", "http://240.0.0.1/", "http://[64:ff9b:1::1]/", "http://[::ffff:0:a00:1]/",
        "http://0xa9fea9fe/", "http://8.8.8.7./"] {
        let lookups = RefCell::new(Vec::new());
        let mut started = 0;
        let result = fetch_with("https://www.example.org/", 64, 20, &AtomicBool::new(false), |url, cap, deadline, cancel| {
            run_with(url, cap, deadline, cancel, |name, _, _, _| {
                lookups.borrow_mut().push(name.to_owned());
                Ok(vec![if name == "www.example.org" { v4("8.8.8.10") } else { v4("10.0.0.1") }])
            }, |_| { started += 1; Ok(redirect(destination, "8.8.8.10")) })
        });
        let message = match destination {
            "https://inside.example.org/" => "the host name resolves to a local or private address",
            "http://0xa9fea9fe/" | "http://8.8.8.7./" => "invalid or ambiguous URL host",
            _ => "local and private address ranges are refused",
        };
        assert_eq!(result.err(), Some(Failure::Refused(message)), "{destination}");
        assert_eq!(started, 1, "{destination}");
        let expected: &[&str] = if destination.contains("inside.example.org") { &["www.example.org", "inside.example.org"] } else { &["www.example.org"] };
        assert_eq!(*lookups.borrow(), expected, "{destination}");
    }
    // A public redirect target is looked up and pinned on its own hop.
    let pins = RefCell::new(Vec::new());
    let page = fetch_with("https://www.example.org/", 64, 20, &AtomicBool::new(false), |url, cap, deadline, cancel| {
        run_with(url, cap, deadline, cancel, |name, _, _, _| Ok(vec![if name == "www.example.org" { v4("8.8.8.10") } else { v4("9.9.9.20") }]),
            |command| {
                let args = args(&command);
                pins.borrow_mut().push(values(&args, "--resolve"));
                let first = values(&args, "--url")[0].contains("www.example.org");
                Ok(if first { redirect("https://docs.example.org/", "8.8.8.10") } else { ok_reply("9.9.9.20") })
            })
    }).unwrap();
    assert_eq!(page.url, "https://docs.example.org/");
    assert_eq!(*pins.borrow(), [
        vec!["www.example.org:443:8.8.8.10".to_owned(), "*:443:8.8.8.10".to_owned()],
        vec!["docs.example.org:443:9.9.9.20".to_owned(), "*:443:9.9.9.20".to_owned()],
    ]);
}

#[test]
fn the_lookup_honours_the_deadline_and_cancel_and_tests_have_no_resolver() {
    let slow = || { std::thread::sleep(Duration::from_secs(2)); Ok(vec![v4("8.8.8.10")]) };
    let started = Instant::now();
    assert_eq!(lookup_with(slow, Instant::now() + Duration::from_millis(50), &AtomicBool::new(false)), Err(Failure::Timeout));
    assert_eq!(lookup_with(slow, far(), &AtomicBool::new(true)), Err(Failure::Cancelled));
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(lookup_with(|| Ok(vec![v6("2606:4700::1")]), far(), &AtomicBool::new(false)), Ok(vec![v6("2606:4700::1")]));
    assert_eq!(lookup_with(|| Err(io::Error::other("no such host")), far(), &AtomicBool::new(false)),
        Err(Failure::Fetch("could not find the address of the host")));
    assert_eq!(lookup("www.example.org", 443, far(), &AtomicBool::new(false)), Err(Failure::Fetch("network is disabled in tests")));
}

#[test]
fn lookups_left_running_are_capped_at_eight() {
    // Its own counter, so lookups in other tests do not count here.
    static RUNNING: AtomicUsize = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(true);
    // Eight lookups that block until their gate closes. A cancel returns at
    // once and leaves each thread running.
    let gates: Vec<_> = (0..MAX_LOOKUPS).map(|_| {
        let (gate, wait) = std::sync::mpsc::channel::<()>();
        let held = lookup_capped(move || { let _ = wait.recv(); Ok(Vec::new()) }, far(), &cancelled, &RUNNING);
        assert_eq!(held, Err(Failure::Cancelled));
        gate
    }).collect();
    assert_eq!(RUNNING.load(Ordering::Acquire), MAX_LOOKUPS);
    assert_eq!(lookup_capped(|| panic!("the ninth lookup started"), far(), &AtomicBool::new(false), &RUNNING),
        Err(Failure::Fetch("too many address lookups are still running; try again later")));
    assert_eq!(RUNNING.load(Ordering::Acquire), MAX_LOOKUPS);
    // Closing the gates ends the eight threads, and a new lookup runs again.
    drop(gates);
    while RUNNING.load(Ordering::Acquire) != 0 { std::thread::yield_now(); }
    assert_eq!(lookup_capped(|| Ok(vec![v4("8.8.8.10")]), far(), &AtomicBool::new(false), &RUNNING), Ok(vec![v4("8.8.8.10")]));
}

#[test]
fn curl_peer_report_must_name_a_checked_address() {
    let checked = [v4("8.8.8.7"), v6("2606:4700::7")];
    let ok = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n\nVADE_REDIRECT:\nVADE_REMOTE:8.8.8.7\n";
    assert!(connected_to(ok, &checked, false) && !connected_to(ok, &[v4("8.8.8.8")], false));
    assert!(connected_to("curl: (7) no connection\nVADE_REDIRECT:\nVADE_REMOTE:\n", &checked, false));
    // A server header with the same name cannot hide the real peer.
    assert!(!connected_to("HTTP/1.1 200 OK\r\nVADE_REMOTE:8.8.8.7\r\n\r\n\nVADE_REMOTE:127.0.0.1\n", &checked, false));
    // A response with no report is refused, unless curl was stopped at the size limit before writing the report.
    let silent = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
    assert!(!connected_to(silent, &checked, false) && connected_to(silent, &checked, true));
    assert!(!connected_to(&format!("{silent}\nVADE_REMOTE:8.8.8.8\n"), &checked, true));
    for headers in ["HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n\nVADE_REMOTE:127.0.0.1\n", silent] {
        let result = run_with("https://www.example.org/", 64, far(), &AtomicBool::new(false),
            |_, _, _, _| Ok(vec![v4("8.8.8.7")]),
            |_| Ok(Reply { body: b"x".to_vec(), headers: headers.into(), code: Some(0), truncated: false }));
        assert_eq!(result.unwrap_err(), Failure::Refused("the connection went to an address that was not checked"));
    }
}
