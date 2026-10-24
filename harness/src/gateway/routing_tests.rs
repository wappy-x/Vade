use super::*;
const CAPS: &str = include_str!("../../tests/fixtures/engine-capabilities-v1.json");
fn head() -> Head {
    Head {
        first: String::new(),
        fields: Vec::new(),
    }
}
fn cluster(config: RoutingConfig) -> Cluster {
    let urls = (1..=3)
        .map(|i| format!("http://127.0.0.1:{}", 18000 + i))
        .collect::<Vec<_>>();
    let cluster = Cluster::new(&urls, config).unwrap();
    for i in 0..3 {
        cluster.apply_probe(i, 0, Signature::parse(CAPS), Instant::now());
    }
    cluster
}
fn body(key: &str) -> Vec<u8> {
    format!(
        r#"{{"model":"mock","prompt_cache_key":{}}}"#,
        json::quote(key)
    )
    .into_bytes()
}
fn choose(cluster: &Cluster, key: &str) -> Lease {
    cluster
        .select("/v1/chat/completions", None, &body(key), &head())
        .ok()
        .unwrap()
}
#[test]
fn execution_signature_separates_weights_tokenizer_template_and_context() {
    let (baseline, _) = Signature::parse(CAPS).unwrap();
    for (from, to, field) in [
        ("mock-weights", "other-weights", "identity.weights.fingerprint"),
        ("mock-tokenizer", "other-tokenizer", "identity.tokenizer_fingerprint"),
        ("mock-template", "other-template", "identity.template_fingerprint"),
        ("\"context_limit\": 4096", "\"context_limit\": 4097", "context_limit"),
    ] {
        let (changed, _) = Signature::parse(&CAPS.replace(from, to)).unwrap();
        assert_eq!(baseline.differences(&changed), [field]);
    }
    let (equal, domain) = Signature::parse(&CAPS.replace("mock-load-epoch", "other-epoch")).unwrap();
    assert!(baseline.differences(&equal).is_empty());
    assert_eq!(domain.as_deref(), Some("other-epoch"));
}
#[test]
fn least_loaded_reservations_affinity_and_saturation_are_atomic() {
    let cluster = cluster(RoutingConfig {
        endpoint_concurrency: 1,
        ..RoutingConfig::default()
    });
    let a = choose(&cluster, "a");
    assert_eq!(a.index, 0);
    let b = choose(&cluster, "a"); // Affine node is locally saturated.
    assert_eq!(b.index, 1);
    let c = choose(&cluster, "b");
    assert_eq!(c.index, 2);
    assert_eq!(
        cluster
            .select("/v1/chat/completions", None, &body("c"), &head())
            .err()
            .unwrap()
            .0,
        503
    );
    assert_eq!(cluster.stats().endpoints[1].fallbacks, 1);
    drop(a);
    drop(b);
    drop(c);
    let next = choose(&cluster, "a");
    assert_eq!(next.index, 1);
    assert_eq!(cluster.stats().endpoints[1].affinity_hits, 1);
    let alias = br#"{"model":"client-model-alias"}"#;
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        assert!(cluster.select(path, None, alias, &head()).is_ok());
    }
    for path in ["/v1/chat/completions", "/v1/responses"] {
        assert!(cluster.select(path, None, alias, &head()).is_err());
    }
}
#[test]
fn affinity_is_bounded_expires_and_invalidates_on_cache_domain_change() {
    let cluster = cluster(RoutingConfig {
        affinity_capacity: 2,
        ..RoutingConfig::default()
    });
    drop(choose(&cluster, "a"));
    drop(choose(&cluster, "b"));
    drop(choose(&cluster, "c"));
    assert_eq!(cluster.stats().affinity_entries, 2);
    let first = choose(&cluster, "c").index;
    cluster.apply_probe(
        first,
        0,
        Signature::parse(&CAPS.replace("mock-load-epoch", "next-epoch")),
        Instant::now(),
    );
    let next = choose(&cluster, "c").index;
    assert_ne!(first, next);
    for value in cluster.state().affinity.values_mut() {
        value.used = Instant::now() - cluster.0.config.affinity_ttl;
    }
    assert_eq!(cluster.stats().affinity_entries, 0);
    assert_eq!(
        cluster
            .stats()
            .endpoints
            .iter()
            .map(|s| s.fallbacks)
            .sum::<u64>(),
        1
    );
}
#[test]
fn client_key_takes_precedence_prefix_is_append_stable_and_scope_is_separate() {
    let cluster = cluster(RoutingConfig::default());
    let key =
        |text: &str, headers: &Head| cluster.key(json::view(text).unwrap(), "chat", None, headers);
    let first = r#"{"model":"mock","tools":[{"name":"read"}],"messages":[{"role":"system","content":"rules"},{"role":"user","content":"first"}]}"#;
    let later = first.replace("\"first\"}]", "\"first\"},{\"role\":\"assistant\",\"content\":\"reply\"},{\"role\":\"user\",\"content\":\"later\"}]");
    assert_eq!(key(first, &head()), key(&later, &head()));
    assert_ne!(
        key(first, &head()),
        key(&first.replace("rules", "new rules"), &head())
    );
    assert_ne!(
        key(first, &head()),
        key(&first.replace("read", "list"), &head())
    );
    assert_ne!(
        key(first, &head()),
        key(&first.replace("first", "other"), &head())
    );
    let explicit = r#"{"prompt_cache_key":"key","messages":[]}"#;
    assert_eq!(
        key(explicit, &head()),
        key(&explicit.replace("[]", "[123]"), &head())
    );
    assert_eq!(
        key(explicit, &head()),
        key(&explicit.replace("key\"", "\\u006bey\""), &head())
    );
    let mut scoped = head();
    scoped
        .fields
        .push(("authorization".into(), "mock-namespace".into()));
    assert_ne!(key(first, &head()), key(first, &scoped));
    let stats = cluster.stats().to_json();
    assert!(!stats.contains("mock-namespace") && !stats.contains("127.0.0.1"));
}
#[test]
fn unknown_stale_incompatible_or_changed_contract_cannot_receive_fallback() {
    let cluster = cluster(RoutingConfig::default());
    let a = choose(&cluster, "a").index;
    cluster.apply_probe(
        a,
        0,
        Signature::parse(&CAPS.replace("mock-weights", "other-weights")),
        Instant::now(),
    );
    assert!(!cluster.stats().endpoints[a].healthy);
    assert_ne!(choose(&cluster, "a").index, a);
    for node in &mut cluster.state().nodes {
        node.healthy_until = Some(Instant::now());
    }
    assert_eq!(
        cluster
            .select("/v1/chat/completions", None, &body("a"), &head())
            .err()
            .unwrap()
            .0,
        503
    );
    for bad in [
        CAPS.replace("vade.engine.v1", "vade.engine.v9"),
        CAPS.replace("\"identity\": {", "\"identity\": null, \"ignored\": {"),
    ] {
        assert!(Signature::parse(&bad).is_err());
    }
}
#[test]
fn different_models_form_separate_groups_same_model_different_fingerprints_are_ambiguous() {
    let urls = vec![
        "http://127.0.0.1:18001".into(),
        "http://127.0.0.1:18002".into(),
    ];
    let cluster = Cluster::new(&urls, RoutingConfig::default()).unwrap();
    cluster.apply_probe(0, 0, Signature::parse(CAPS), Instant::now());
    assert!(
        cluster
            .select("/v1/chat/completions", None, &body("a"), &head())
            .is_err()
    ); // Initial barrier.
    cluster.apply_probe(
        1,
        0,
        Signature::parse(&CAPS.replace("\"mock\"", "\"other\"")),
        Instant::now(),
    );
    assert_eq!(choose(&cluster, "a").index, 0);
    assert_eq!(
        cluster
            .select(
                "/v1/messages",
                None,
                br#"{"model":"client-model-alias"}"#,
                &head()
            )
            .err()
            .unwrap()
            .0,
        409
    );
    let other = br#"{"model":"other"}"#;
    assert_eq!(
        cluster
            .select("/v1/chat/completions", None, other, &head())
            .ok()
            .unwrap()
            .index,
        1
    );
    assert_eq!(
        cluster
            .select("/v1/capabilities", None, &[], &head())
            .err()
            .unwrap()
            .0,
        409
    );
    let ambiguous = Cluster::new(&urls, RoutingConfig::default()).unwrap();
    ambiguous.apply_probe(0, 0, Signature::parse(CAPS), Instant::now());
    ambiguous.apply_probe(
        1,
        0,
        Signature::parse(&CAPS.replace("mock-tokenizer", "different-tokenizer")),
        Instant::now(),
    );
    let error = ambiguous
            .select("/v1/chat/completions", None, &body("a"), &head())
            .err()
            .unwrap();
    assert_eq!(error.0, 409);
    assert!(error.1.contains("identity.tokenizer_fingerprint") && error.1.contains("restart the gateway"));
    for endpoint in &urls { assert!(error.1.contains(endpoint)); }
}
#[test]
fn overload_and_uncertain_failures_are_counted_and_old_probe_cannot_clear_failure() {
    let cluster = cluster(RoutingConfig::default());
    let mut lease = choose(&cluster, "a");
    let index = lease.index;
    lease.start();
    lease.status(429, Some("999999"));
    lease.finish(429, None);
    drop(lease);
    assert!(cluster.stats().endpoints[index].cooling_down);
    assert_eq!(cluster.stats().endpoints[index].overloads, 1);
    let mut next = choose(&cluster, "a");
    assert_ne!(index, next.index);
    let failed = next.index;
    next.start();
    drop(next); // Outcome unknown; never returned success.
    cluster.apply_probe(failed, 0, Signature::parse(CAPS), Instant::now());
    assert!(!cluster.stats().endpoints[failed].healthy);
    assert_eq!(cluster.stats().endpoints[failed].requests, 1);
    assert_eq!(cluster.stats().endpoints[failed].failures, 1);
    let serial = cluster.state().nodes[failed].health_serial;
    cluster.apply_probe(failed, serial, Signature::parse(CAPS), Instant::now());
    assert!(cluster.stats().endpoints[failed].healthy);
    assert!(cluster.stats().endpoints[failed].cooling_down);
    assert_eq!(cluster.stats().endpoints[failed].in_flight, 0);
}
#[test]
fn only_paired_successful_usage_counts_and_ratio_is_weighted() {
    let cluster = cluster(RoutingConfig::default());
    for (status, usage) in [
        (200, Some((100, 80))),
        (200, None),
        (200, Some((300, 120))),
        (503, Some((100, 99))),
    ] {
        let mut lease = choose(&cluster, "a");
        lease.start();
        lease.finish(status, usage);
    }
    let stats = &cluster.stats().endpoints[0];
    assert_eq!(stats.requests, 4);
    assert_eq!(stats.failures, 1);
    assert_eq!(stats.usage_responses, 2);
    assert_eq!(stats.cached_ratio(), Some(0.5));
    assert_eq!(cluster.stats().endpoints[1].cached_ratio(), None);
}
#[test]
fn bounds_duplicate_custom_paths_and_https_fail_before_io() {
    let urls: Vec<_> = (18001..18065).map(|port| format!("http://127.0.0.1:{port}")).collect();
    assert_eq!(Cluster::new(&urls, RoutingConfig::default()).unwrap().len(), 64);
    for urls in [
        vec![],
        (18001..18066).map(|port| format!("http://127.0.0.1:{port}")).collect(),
        vec!["https://127.0.0.1:18001".into()],
        vec!["http://127.0.0.1:18001/custom".into()],
        vec!["http://127.0.0.1:18001".into(); 2],
    ] {
        assert!(Cluster::new(&urls, RoutingConfig::default()).is_err());
    }
    assert!(
        RoutingConfig {
            affinity_capacity: 0,
            ..RoutingConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        RoutingConfig {
            endpoint_concurrency: 17,
            ..RoutingConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        RoutingConfig {
            affinity_ttl: Duration::ZERO,
            ..RoutingConfig::default()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn boundary_dispatch_preserves_model_group_affinity_and_fallback() {
    use super::super::{Upstream, prepare_dispatch, sampling};
    use crate::transport::BodyReader;
    use std::io::Read;

    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/completions"] {
        for explicit_key in [false, true] {
            let cluster = cluster(RoutingConfig::default());
            cluster.apply_probe(
                2,
                0,
                Signature::parse(&CAPS.replace("\"mock\"", "\"other\"")),
                Instant::now(),
            );
            let key = if explicit_key {
                r#", "prompt_cache_key":"boundary-session""#
            } else {
                ""
            };
            let small = format!(
                r#"{{ "model":"mock", "input":"prefix", "prompt":"prefix", "messages":[{{"role":"user","content":"prefix"}}]{key} }}"#
            );
            let mut original = small.as_bytes().to_vec();
            original.resize(json::MAX_INPUT_BYTES, b' ');
            let route = if path.ends_with("/responses") {
                "responses"
            } else if path.ends_with("/messages") {
                "messages"
            } else if path == "/v1/completions" {
                "completions"
            } else {
                "chat"
            };
            assert_eq!(api_route(path), route);
            let expected_key = cluster.key(
                json::view(&small).unwrap(),
                route,
                Some("client=1"),
                &head(),
            );
            let expected = cluster
                .select(path, Some("client=1"), &original, &head())
                .ok()
                .unwrap()
                .index();
            // Both accepted framing modes must reach the same dispatch preparation.
            for chunked in [false, true] {
                let (framed, wire) = if chunked {
                    let mut wire = format!("{:x}\r\n", original.len()).into_bytes();
                    wire.extend_from_slice(&original);
                    wire.extend_from_slice(b"\r\n0\r\n\r\n");
                    (
                        Head {
                            fields: vec![("transfer-encoding".into(), "chunked".into())],
                            ..head()
                        },
                        wire,
                    )
                } else {
                    (
                        Head {
                            fields: vec![("content-length".into(), original.len().to_string())],
                            ..head()
                        },
                        original.clone(),
                    )
                };
                let mut input = BodyReader::new(
                    std::io::Cursor::new(wire),
                    &framed,
                    true,
                    json::MAX_INPUT_BYTES,
                )
                .unwrap();
                let mut accepted = Vec::new();
                input.read_to_end(&mut accepted).unwrap();
                assert_eq!(accepted, original);
                let (lease, effective) = prepare_dispatch(
                    &Upstream::Cluster(cluster.clone()),
                    "POST",
                    path,
                    Some("client=1"),
                    &framed,
                    &accepted,
                )
                .ok()
                .unwrap();
                assert_eq!(lease.as_ref().unwrap().index(), expected);
                assert_eq!(
                    effective.len(),
                    original.len() + sampling::DEFAULTS_JSON.len() - 1
                );
                assert_eq!(effective, sampling::apply("POST", path, &framed, &original));
                assert!(cluster.state().affinity.contains_key(&(0, expected_key)));
                drop(lease);
                let fallback = cluster
                    .select_excluding(path, Some("client=1"), &accepted, &framed, Some(expected))
                    .ok()
                    .unwrap();
                assert_ne!(fallback.index(), expected);
                assert_ne!(fallback.index(), 2); // The other model group never receives a retry.
                drop(fallback);
                // Restore the original placement for the other framing mode.
                drop(
                    cluster
                        .select_excluding(
                            path,
                            Some("client=1"),
                            &accepted,
                            &framed,
                            Some(1 - expected),
                        )
                        .ok()
                        .unwrap(),
                );
            }
        }
    }
}

#[test]
fn text_completions_route_only_to_endpoints_that_advertise_them() {
    // The engine advertises /v1/completions; a replica that does not must get
    // the same 503 as any other unadvertised generation route, not a relay 404.
    let request = br#"{"model":"mock","prompt":"Once upon a time"}"#;
    let cluster = cluster(RoutingConfig::default());
    for path in ["/v1/completions", "/completions"] {
        assert!(cluster.select(path, None, request, &head()).is_ok(), "{path}");
    }
    let without = CAPS.replace(", \"/v1/completions\"]", "]");
    assert_ne!(without, CAPS);
    let urls = vec!["http://127.0.0.1:18001".into(), "http://127.0.0.1:18002".into()];
    let chat_only = Cluster::new(&urls, RoutingConfig::default()).unwrap();
    for i in 0..2 {
        chat_only.apply_probe(i, 0, Signature::parse(&without), Instant::now());
    }
    for path in ["/v1/completions", "/completions"] {
        let error = chat_only.select(path, None, request, &head()).err().unwrap();
        assert_eq!((error.0, error.1.as_ref()), (503, "compatible endpoints do not advertise this API route"), "{path}");
    }
    assert!(chat_only.select("/v1/chat/completions", None, &body("a"), &head()).is_ok());
    // Admission for a Completions body stays a context-fit probe, not a generation.
    let admission = br#"{"endpoint":"/v1/completions","request":{"model":"mock","prompt":"x"},"output_tokens":1}"#;
    assert!(chat_only.select("/v1/admission", None, admission, &head()).is_ok());
    // Mixed advertisement is a compatibility difference, named as such.
    let mixed = Cluster::new(&urls, RoutingConfig::default()).unwrap();
    mixed.apply_probe(0, 0, Signature::parse(CAPS), Instant::now());
    mixed.apply_probe(1, 0, Signature::parse(&without), Instant::now());
    let error = mixed.select("/v1/completions", None, request, &head()).err().unwrap();
    assert_eq!(error.0, 409);
    assert!(error.1.contains("fields differ: endpoints"), "{}", error.1);
    // A replica advertising Completions alone is still a generation endpoint.
    let completions_only = CAPS.replace(
        r#"["/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens", "/v1/completions"]"#,
        r#"["/v1/completions"]"#,
    );
    assert_ne!(completions_only, CAPS);
    assert!(Signature::parse(&completions_only).is_ok());
    assert!(Signature::parse(&completions_only.replace(r#"["/v1/completions"]"#, "[]")).is_err());
}

#[test]
fn text_completions_affinity_follows_the_prompt_prefix_in_its_own_namespace() {
    let cluster = cluster(RoutingConfig::default());
    let key = |route: &str, body: &str| cluster.key(json::view(body).unwrap(), route, None, &head());
    let story = r#"{"model":"mock","prompt":"Once upon a time"}"#;
    let recipe = r#"{"model":"mock","prompt":"Preheat the oven"}"#;
    assert_eq!(key("completions", story), key("completions", story));
    assert_ne!(key("completions", story), key("completions", recipe));
    // Same body under chat ignores `prompt`, as before: separate namespaces.
    assert_eq!(key("chat", story), key("chat", recipe));
    assert_ne!(key("completions", story), key("chat", story));
    // Appending beyond the hashed prefix keeps the key (growing continuations).
    let long = "a".repeat(PREFIX_BYTES);
    let first = format!(r#"{{"model":"mock","prompt":"{long}"}}"#);
    let grown = format!(r#"{{"model":"mock","prompt":"{long} and more"}}"#);
    assert_eq!(key("completions", &first), key("completions", &grown));
    // An explicit client key still wins over the prompt.
    let keyed = |prompt: &str| format!(r#"{{"model":"mock","prompt":"{prompt}","prompt_cache_key":"session"}}"#);
    assert_eq!(key("completions", &keyed("one")), key("completions", &keyed("two")));
    // Path classification, including the public alias and admission targets.
    for (path, route) in [
        ("/v1/completions", "completions"),
        ("/completions", "completions"),
        ("/v1/chat/completions", "chat"),
        ("/chat/completions", "chat"),
        ("/v1/responses", "responses"),
        ("/v1/messages", "messages"),
        ("/v1/messages/count_tokens", "messages"),
    ] {
        assert_eq!(api_route(path), route, "{path}");
    }
    // Admission and the following generation share an affinity entry.
    let generation = cluster.select("/v1/completions", None, story.as_bytes(), &head()).ok().unwrap();
    let index = generation.index();
    drop(generation);
    let admission = format!(r#"{{"endpoint":"/v1/completions","request":{story},"output_tokens":1}}"#);
    let before = cluster.stats().endpoints[index].affinity_hits;
    let admitted = cluster.select("/v1/admission", None, admission.as_bytes(), &head()).ok().unwrap();
    assert_eq!(admitted.index(), index);
    assert_eq!(cluster.stats().endpoints[index].affinity_hits, before + 1);
}
