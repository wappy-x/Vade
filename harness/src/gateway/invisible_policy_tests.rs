use super::*; fn head() -> Head { Head { first: "POST /v1/responses HTTP/1.1".into(), fields: Vec::new(), } } fn configured(flags: Flags) -> Runtime {
    Runtime::new(&["http://127.0.0.1:1".into()], flags).unwrap()
}
#[test]
fn opt_in_preserves_message_spans_and_explicit_options_across_turns() { let flags = Flags { stable: true, greedy: true, ..Flags::default() }; let runtime = configured(flags);
let first=br#"{ "model":"mock", "input":[{"role":"developer","content":"sys\n"},{"role":"user","content":"first"}],"extension":1.20e0 }"#;
let later=br#"{ "model":"mock", "input":[{"role":"developer","content":"sys\n"},{"role":"user","content":"first"},{"role":"assistant","content":"answer"},{"role":"user","content":"second"}],"extension":1.20e0 }"#;
let a = runtime.prepare("POST", "/v1/responses", &head(), first); let b = runtime.prepare("POST", "/v1/responses", &head(), later); assert_eq!( text(view(&a.body).unwrap(), "prompt_cache_key"),
text(view(&b.body).unwrap(), "prompt_cache_key") ); for (original, patched) in [ (first.as_slice(), a.body.as_ref()), (later.as_slice(), b.body.as_ref()), ] { let v = view(original).unwrap(); let w = view(patched).unwrap();
assert_eq!(v.get("input").unwrap().raw(), w.get("input").unwrap().raw()); assert_eq!(w.get("extension").unwrap().raw(), "1.20e0"); assert_eq!(w.get("temperature").unwrap().raw(), "0"); assert_eq!(w.get("cache").unwrap().raw(), "true"); }
for field in [ "temperature", "top_p", "top_k", "min_p", "typical_p", "seed", "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias", "sampling", "generation_config", ] { let raw = format!(
"{{\"input\":\"hello\",\"prompt_cache_key\":\"client\",\"cache\":false,\"{field}\":null}}" ); let p = runtime.prepare("POST", "/v1/responses", &head(), raw.as_bytes()); let v = view(&p.body).unwrap();
assert_eq!(text(v, "prompt_cache_key").as_deref(), Some("client")); assert_eq!(v.get("cache").unwrap().raw(), "false"); assert_eq!(v.get(field).unwrap().raw(), "null"); assert_eq!(p.body.as_ref(), raw.as_bytes()); }
let opaque = b"\xffnot-json"; let p = configured(Flags::default()).prepare("POST", "/other", &head(), opaque); assert!(matches!(p.body, Cow::Borrowed(_))); assert_eq!(p.body.as_ref(), opaque); }
#[test]
fn rendezvous_affinity_survives_reordering_and_only_moves_to_added_server() { let origins = [
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
        "http://127.0.0.1:3",
] .map(|s| Origin::parse(s).unwrap()); let reverse = vec![origins[1].clone(), origins[0].clone()]; for n in 0..100 { let key = format!("session{n}"); let old = affinity(&key, &origins[..2]); let reordered = affinity(&key, &reverse);
assert_eq!( origins[old].endpoint.authority, reverse[reordered].endpoint.authority ); let new = affinity(&key, &origins); assert!(new == 2 || old == new); } }
#[test]
fn capabilities_are_read_once_output_and_context_clamp_and_oversized_input_is_exact() { let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
let server = thread::spawn(move || { let mut caps = 0; for _ in 0..5 { let socket = listener.accept().unwrap().0; socket .set_read_timeout(Some(Duration::from_secs(3))) .unwrap(); let mut reader = BufReader::new(socket);
let h = transport::read_head(&mut reader).unwrap(); let route = h.first.split(' ').nth(1).unwrap().to_owned(); let mut body = BodyReader::new(reader, &h, true, 8192).unwrap(); let mut bytes = Vec::new();
body.read_to_end(&mut bytes).unwrap(); let reply = if route == "/v1/capabilities" { caps += 1;
r#"{"object":"engine.capabilities","contract_version":"vade.engine.v1","model":"mock","context_limit":100,"input_limit":99,"output_limit":16,"admission":"/v1/admission","admission_kind":"context_fit"}"#.to_owned() } else {
assert_eq!(route, "/v1/admission"); let v = view(&bytes).unwrap(); let request = v.get("request").unwrap(); let prompt = integer(request.get("mock_prompt")).unwrap(); let output = integer(v.get("output_tokens")).unwrap();
let admitted = prompt <= 99 && output <= 16 && prompt + output <= 100; format!(
"{{\"object\":\"engine.admission\",\"contract_version\":\"vade.engine.v1\",\"model\":\"mock\",\"prompt_tokens\":{prompt},\"output_tokens\":{output},\"context_limit\":100,\"input_limit\":99,\"output_limit\":16,\"admission_kind\":\"context_fit\",\"reserves_resources\":false,\"admitted\":{admitted}}}"
) }; write!(body.reader.get_mut(),"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",reply.len()).unwrap(); } assert_eq!(caps, 1); }); let runtime = Runtime::new( &[url], Flags {
stable: true, greedy: true, clamp: true, ..Flags::default() }, ) .unwrap(); let original=br#"{ "model":"mock", "input":[{"role":"user","content":"unchanged"}],"mock_prompt":95,"max_output_tokens":500 }"#;
let p = runtime.prepare("POST", "/v1/responses", &head(), original); let v = view(&p.body).unwrap(); assert_eq!(integer(v.get("max_output_tokens")), Some(5)); assert_eq!( view(original).unwrap().get("input").unwrap().raw(),
v.get("input").unwrap().raw() ); assert_eq!( view(p.report.as_bytes()) .unwrap() .get("clamps") .unwrap() .array() .unwrap() .count(), 2 ); let over = br#"{ "model":"mock","input":"too long","mock_prompt":101,"max_output_tokens":500 }"#;
let p = runtime.prepare("POST", "/v1/responses", &head(), over); assert!(p.body.starts_with(&over[..over.len() - 1])); let normalized = view(&p.body).unwrap(); assert_eq!(normalized.get("temperature").unwrap().raw(), "0"); assert_eq!(normalized.get("seed").unwrap().raw(), "20260905"); assert_eq!(normalized.get("top_p").unwrap().raw(), "1.0"); assert_eq!( integer( view(p.report.as_bytes()) .unwrap() .get("oversized_input") .unwrap() .get("tokens") ), Some(101) );
let normal = br#"{"model":"mock","input":"fits","mock_prompt":10,"max_output_tokens":4}"#; let p = runtime.prepare("POST", "/v1/responses", &head(), normal); assert_eq!( integer(view(&p.body).unwrap().get("max_output_tokens")), Some(4) );
server.join().unwrap(); }

#[test]
fn any_explicit_sampling_control_is_byte_identical_on_the_standalone_path() {
    for flags in [
        Flags::default(),
        Flags {
            greedy: true,
            ..Flags::default()
        },
    ] {
        let runtime = configured(flags);
        for path in [
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/responses?x=1",
            "/responses",
            "/v1/messages",
            "/messages",
        ] {
            for &(field, explicit) in super::super::super::sampling::EXPLICIT_SAMPLING_EXAMPLES {
                let original = format!(r#" {{ "{field}":{explicit}, "extension":1.20e0 }} "#);
                let prepared = runtime.prepare("POST", path, &head(), original.as_bytes());
                assert_eq!(
                    prepared.body.as_ref(),
                    original.as_bytes(),
                    "{path}: {field}"
                );
            }
        }
    }
}

fn assert_sampling_report(prepared: &Prepared<'_>, inserted: bool) {
    let report = view(prepared.report.as_bytes()).unwrap();
    assert_eq!(
        report.get("greedy_added").unwrap().raw(),
        if inserted { "true" } else { "false" }
    );
    let defaults = report.get("sampling_defaults_added").unwrap();
    assert_eq!(
        defaults.raw(),
        if inserted {
            r#"{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}"#
        } else {
            "null"
        }
    );
    let mut metrics = Metrics::new(true);
    metrics.policy = prepared.report.clone();
    let row = metrics.json();
    assert_eq!(
        view(row.as_bytes()).unwrap().get("policy").unwrap().raw(),
        prepared.report.as_str()
    );
}

fn check_sampling_reports(runtime: &Runtime) {
    for path in [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/responses?x=1",
        "/responses",
        "/v1/messages",
        "/messages",
    ] {
        let unsampled = br#"{ "input":{"seed":42}, "extension":1.20e0 }"#;
        let prepared = runtime.prepare("POST", path, &head(), unsampled);
        assert_sampling_report(&prepared, true);
        let body = view(&prepared.body).unwrap();
        for (field, expected) in
            [("temperature", "0"), ("seed", "20260905"), ("top_p", "1.0"),
             ("top_k", "0"), ("repetition_penalty", "1.0"),
             ("presence_penalty", "0.0"), ("frequency_penalty", "0.0")]
        {
            assert_eq!(body.get(field).unwrap().raw(), expected);
        }
        // Values already equal to the defaults are explicit, not another insertion.
        assert_sampling_report(
            &runtime.prepare("POST", path, &head(), &prepared.body),
            false,
        );
        for &(field, example) in super::super::super::sampling::EXPLICIT_SAMPLING_EXAMPLES {
            for explicit in [example, "null"] {
                let original = format!(r#" {{ "{field}":{explicit}, "extension":1.20e0 }} "#);
                let prepared = runtime.prepare("POST", path, &head(), original.as_bytes());
                assert_sampling_report(&prepared, false);
                let before = view(original.as_bytes()).unwrap();
                let after = view(&prepared.body).unwrap();
                for name in ["temperature", "seed", "top_p"] {
                    assert_eq!(
                        before.get(name).map(View::raw),
                        after.get(name).map(View::raw)
                    );
                }
            }
        }
    }
    for (method, path, body) in [
        ("GET", "/v1/responses", b"{}".as_slice()),
        ("POST", "/v1/admission", b"{}".as_slice()),
        ("POST", "/v1/responses", b"\xffopaque".as_slice()),
    ] {
        assert_sampling_report(&runtime.prepare(method, path, &head(), body), false);
    }
    let compressed = Head {
        fields: vec![("content-encoding".into(), "gzip".into())],
        ..head()
    };
    assert_sampling_report(
        &runtime.prepare("POST", "/v1/responses", &compressed, b"{}"),
        false,
    );
}

#[test]
fn default_policy_reports_sampling_defaults_only_when_inserted() {
    check_sampling_reports(&configured(Flags::default()));
}

#[test]
fn opt_in_policy_reports_sampling_defaults_only_when_inserted() {
    for flags in [
        Flags {
            affinity: true,
            ..Flags::default()
        },
        Flags {
            greedy: true,
            ..Flags::default()
        },
        Flags {
            stable: true,
            affinity: true,
            greedy: true,
            ..Flags::default()
        },
    ] {
        check_sampling_reports(&configured(flags));
    }
}

#[test]
fn boundary_policy_routing_decorations_and_clamps_use_client_bytes() {
    for path in ["/v1/chat/completions", "/v1/responses"] {
        for budget in [
            "",
            r#", "max_tokens":500, "max_output_tokens":500, "max_completion_tokens":500"#,
        ] {
            for prompt in [95, 101] {
                let small = format!(
                    r#"{{ "model":"mock", "input":"exact\n", "extension":1.20e0{budget} }}"#
                );
                let mut original = small.as_bytes().to_vec();
                original.resize(json::MAX_INPUT_BYTES, b' ');
                assert_eq!(original.len(), json::MAX_INPUT_BYTES);
                let framed = Head {
                    fields: vec![("content-length".into(), original.len().to_string())],
                    ..head()
                };
                let mut input = BodyReader::new(
                    io::Cursor::new(&original),
                    &framed,
                    true,
                    json::MAX_INPUT_BYTES,
                )
                .unwrap();
                let mut accepted = Vec::new();
                input.read_to_end(&mut accepted).unwrap();
                assert_eq!(accepted, original);
                let mut runtime = configured(Flags {
                    stable: true,
                    affinity: true,
                    clamp: false,
                    greedy: true,
                    no_sticky: false,
                });
                runtime
                    .origins
                    .push(Origin::parse("http://127.0.0.1:2").unwrap());
                runtime.router = Router::new(2, true);
                runtime.limits = vec![
                    Some(Limits {
                        context: 100,
                        input: 99,
                        output: 16,
                        model: "mock".into()
                    });
                    2
                ];
                let expected_key = key(view(&original), &head());
                let expected_index = affinity(&expected_key, &runtime.origins);
                let mut calls = Vec::new();
                let prepared = runtime.prepare_with(
                    "POST",
                    path,
                    &head(),
                    &accepted,
                    |index, endpoint, request, output, _| {
                        assert_eq!(index, expected_index);
                        assert_eq!(endpoint, path);
                        // Admission sees the original request with policy edits, before sampling defaults.
                        let value = view(request.trim_ascii()).unwrap();
                        for field in ["temperature", "seed", "top_p"] {
                            assert!(value.get(field).is_none());
                        }
                        assert_eq!(
                            text(value, "prompt_cache_key").as_deref(),
                            Some(expected_key.as_str())
                        );
                        assert_eq!(value.get("cache").unwrap().raw(), "true");
                        assert_eq!(value.get("extension").unwrap().raw(), "1.20e0");
                        for field in ["max_tokens", "max_output_tokens", "max_completion_tokens"] {
                            if let Some(n) = integer(value.get(field)) {
                                assert_eq!(n, output);
                            }
                        }
                        calls.push(output);
                        Ok(prompt)
                    },
                );
                assert_eq!(prepared.index, expected_index);
                assert_sampling_report(&prepared, true);
                let result = view(prepared.body.trim_ascii()).unwrap();
                assert_eq!(result.get("temperature").unwrap().raw(), "0");
                assert_eq!(result.get("seed").unwrap().raw(), "20260905");
                assert_eq!(result.get("top_p").unwrap().raw(), "1.0");
                let report = view(prepared.report.as_bytes()).unwrap();
                assert_eq!(report.get("error").unwrap().raw(), "null");
                if prompt == 95 {
                    assert_eq!(calls, [16, 5]);
                    assert_eq!(
                        text(result, "prompt_cache_key").as_deref(),
                        Some(expected_key.as_str())
                    );
                    assert_eq!(result.get("cache").unwrap().raw(), "true");
                    let preferred = if path.ends_with("/responses") {
                        "max_output_tokens"
                    } else {
                        "max_tokens"
                    };
                    assert_eq!(integer(result.get(preferred)), Some(5));
                    assert_eq!(report.get("cache_key_added").unwrap().raw(), "true");
                    assert_eq!(report.get("cache_default_added").unwrap().raw(), "true");
                } else {
                    assert_eq!(calls, [16]);
                    assert_eq!(
                        integer(report.get("oversized_input").unwrap().get("tokens")),
                        Some(101)
                    );
                    let expected =
                        super::super::super::sampling::apply("POST", path, &head(), &original);
                    assert_eq!(prepared.body, expected);
                }
            }
        }
    }
}

#[test]
fn boundary_default_policy_adds_only_exact_defaults() {
    for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
        let mut original = br#"{"model":"mock","input":""#.to_vec();
        original.resize(json::MAX_INPUT_BYTES - 2, b'x');
        original.extend_from_slice(b"\"}");
        let runtime = configured(Flags::default());
        let prepared = runtime.prepare("POST", path, &head(), &original);
        assert_sampling_report(&prepared, true);
        assert!(prepared.body.starts_with(&original[..original.len() - 1]));
        assert_eq!(
            &prepared.body[original.len() - 1..],
            br#","temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}"#
        );
    }
}

#[test]
fn stable_prefix_gives_messages_the_cache_fields_but_no_limit_changes() {
    // The engine reuses a prefix only for a request with cache:true or a key;
    // Claude Code sends neither, so without these every turn prefilled again.
    let body = br#"{ "model":"mock", "max_tokens":500, "system":"sys\n", "messages":[{"role":"user","content":"first"}], "extension":1.20e0 }"#;
    for clamp in [false, true] {
        // Runtime::new would fetch capabilities for clamp; set the limits by hand.
        let mut runtime = configured(Flags { stable: true, ..Flags::default() });
        runtime.flags.clamp = clamp;
        runtime.limits = vec![Some(Limits { context: 100, input: 99, output: 16, model: "mock".into() })];
        let prepared = runtime.prepare("POST", "/v1/messages?beta=true", &head(), body);
        let (original, patched) = (view(body).unwrap(), view(&prepared.body).unwrap());
        assert_eq!(patched.get("cache").unwrap().raw(), "true");
        assert!(text(patched, "prompt_cache_key").is_some_and(|k| k.starts_with("vade-")), "{}", prepared.report);
        for field in ["max_tokens", "system", "messages", "extension"] {
            assert_eq!(patched.get(field).unwrap().raw(), original.get(field).unwrap().raw(), "{field}");
        }
        assert!(prepared.report.contains(r#""cache_key_added":true,"cache_default_added":true"#) && prepared.report.contains(r#""clamps":[]"#), "{}", prepared.report);
    }
    // Explicit client choices survive, and without the flag Messages bytes are untouched.
    let explicit = br#"{ "model":"mock", "max_tokens":8, "cache":false, "prompt_cache_key":"client", "messages":[] }"#;
    let prepared = configured(Flags { stable: true, ..Flags::default() }).prepare("POST", "/v1/messages", &head(), explicit);
    let patched = view(&prepared.body).unwrap();
    assert_eq!((patched.get("cache").unwrap().raw(), text(patched, "prompt_cache_key").as_deref()), ("false", Some("client")));
    let mut plain = configured(Flags { affinity: true, ..Flags::default() });
    plain.flags.clamp = true;
    plain.limits = vec![Some(Limits { context: 100, input: 99, output: 16, model: "mock".into() })];
    let plain = plain.prepare("POST", "/v1/messages", &head(), body);
    let plain = view(&plain.body).unwrap();
    assert!(plain.get("cache").is_none() && plain.get("prompt_cache_key").is_none());
    assert_eq!(plain.get("max_tokens").unwrap().raw(), "500");
}
