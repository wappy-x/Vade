//! Real `vade-harness` processes against first-party loopback fixtures: context
//! and cache reporting, compaction, fork and resume, generation flags, cache
//! permission and unspecified sampling.
#[path = "support/home.rs"]
mod home;
#[allow(dead_code)]
#[path = "support/cli.rs"]
mod cli;
#[path = "support/contract.rs"]
mod contract;
use contract::*;
#[allow(dead_code)] // Only the field list and greedy defaults are used here.
#[path = "../../engine/crates/model/src/sampling_policy.rs"]
mod sampling_policy;

#[test]
fn cli_shows_exact_context_and_cached_tokens_and_new_resets_affinity() {
    for mode in ["direct", "agent"] {
        let workspace = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            let mut first_key = String::new();
            for turn in 0..2 {
                // Native /new drops its model/pool; direct keeps framed sockets.
                if turn == 1 && mode == "agent" {
                    reader = accept(&listener);
                }
                capabilities(&mut reader, CAPS);
                let body = admission(&mut reader, 117, true);
                let v = json::view(&body).unwrap();
                let key = v
                    .get("prompt_cache_key")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .into_owned();
                if turn == 0 {
                    first_key = key;
                } else {
                    assert_ne!(first_key, key);
                }
                assert_eq!(
                    v.get("messages").unwrap().array().unwrap().count(),
                    if mode == "agent" { 3 } else { 2 }
                );
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(117, 87)),
                );
            }
        });
        let mut command = home::harness();
        command.args([
            mode,
            "--state-in-project",
            "--verbose",
            "--endpoint",
            &endpoint,
            "--model",
            "mock",
            "--workspace",
            workspace.0.to_str().unwrap(),
            "--plain",
            "--max-output-tokens",
            "17",
        ]);
        let mut child = cli::Cli::start(&mut command);
        child.send("first\n");
        child.until(true, "session reported cached tokens:");
        child.send("/context\n/new\nsecond\n");
        child.until(true, "session reported cached tokens:");
        child.send("/context\n");
        let output = child.finish();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        for text in [&stdout, &stderr] {
            assert!(text.contains("exact 117 + output reservation 17 / context limit 4096"));
            assert!(text.contains("cached tokens 87"));
            assert!(text.contains("1/1 completed model responses"));
            assert!(!text.contains("sha256:mock"));
        }
        if mode == "agent" {
            assert!(stdout.contains("+ 17 output reserve / 4096 budget"));
        }
        server.join().unwrap();
    }
}

#[test]
fn cli_compaction_fork_switch_and_resume_reestablish_admitted_prefixes_and_report_cache_drop() {
    let workspace = WorkspaceFixture::new();
    let (listener, endpoint) = listen();
    let server = thread::spawn(move || {
        let mut reader = accept(&listener);
        let mut bodies: Vec<String> = Vec::new();
        let mut keys = Vec::new();
        // Three ordinary appends, compaction, fork, switch, fresh journal, and
        // a separate process's --resume, plus the opt-in discarded warm.
        for turn in 0..8 {
            if turn >= 4 {
                reader = accept(&listener);
            }
            if turn == 3 {
                // Manual /compact counts both the original completed history
                // and the reducing candidate before persistence or warm-up.
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 768);
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 512);
                capabilities(&mut reader, CAPS);
                rendered_history_admission(&mut reader, 512);
                let warm = admission(&mut reader, 512, true);
                let w = json::view(&warm).unwrap();
                assert_eq!(w.get("max_tokens").unwrap().raw(), "1");
                assert_eq!(
                    w.get("tools").unwrap().raw(),
                    json::view(&bodies[0]).unwrap().get("tools").unwrap().raw()
                );
                assert_eq!(
                    w.get("messages")
                        .unwrap()
                        .array()
                        .unwrap()
                        .nth(2)
                        .unwrap()
                        .get("name")
                        .unwrap()
                        .as_str()
                        .as_deref(),
                    Some("vade_context_summary")
                );
                assert_eq!(request(&mut reader).1, warm);
                respond(
                    &mut reader,
                    200,
                    "application/json",
                    r#"{"choices":[{"finish_reason":"length","message":{"role":"assistant","content":"discarded warm"}}],"usage":{"prompt_tokens":512,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":128}}}"#,
                );
            }
            capabilities(&mut reader, CAPS);
            let body = admission(&mut reader, 512, true);
            assert_eq!(
                request(&mut reader),
                ("POST /v1/chat/completions HTTP/1.1".into(), body.clone())
            );
            let v = json::view(&body).unwrap();
            let key = v
                .get("prompt_cache_key")
                .unwrap()
                .as_str()
                .unwrap()
                .into_owned();
            if turn > 0 && turn <= 3 {
                assert_eq!(
                    keys[0], key,
                    "same journal keeps affinity through compaction"
                );
            }
            if turn >= 4 {
                assert!(
                    !keys.contains(&key),
                    "a newly activated journal gets fresh affinity"
                );
            }
            if turn == 1 || turn == 2 {
                let previous = &bodies[turn - 1];
                let previous = without_cancel(previous);
                assert!(without_cancel(&body).starts_with(&previous[..previous.len() - 2]));
            }
            let messages = v.get("messages").unwrap().raw();
            if (3..=5).contains(&turn) || turn == 7 {
                assert!(messages.contains("User-supplied compaction summary"));
                assert!(!messages.contains("discardable old prose"));
                assert!(messages.contains("latest answer"));
            }
            if turn == 3 {
                assert!(!body.starts_with(&bodies[2][..bodies[2].len() - 2]));
                assert!(body.len() < bodies[2].len());
            }
            if turn == 4 {
                assert!(messages.contains("Re-read relevant files"));
            }
            if turn == 5 || turn == 7 {
                assert!(!messages.contains("child-only-task"));
            }
            if turn == 6 {
                assert_eq!(v.get("messages").unwrap().array().unwrap().count(), 3);
            }
            let text = if turn < 2 {
                "discardable old prose ".repeat(100)
            } else {
                "latest answer".into()
            };
            let event = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\n",
                json::quote(&text)
            );
            let cached = [0, 110, 220, 7, 0, 0, 0, 0][turn];
            respond(
                &mut reader,
                200,
                "text/event-stream",
                &format!("{event}{}{DONE}", usage(512, cached)),
            );
            bodies.push(body);
            keys.push(key);
        }
    });
    let launch = |resume: bool, input: &str| {
        let mut command = home::harness();
        command
            .args([
                "agent",
                "--state-in-project",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--workspace",
            ])
            .arg(&workspace.0)
            .args([
                "--plain",
                "--approval",
                "never",
                "--verbose",
                "--prewarm-compaction",
                if resume { "--resume" } else { "--session" },
                "parent",
            ]);
        let mut child = cli::Cli::start(&mut command);
        for line in input.lines() {
            child.send(&format!("{line}\n"));
            if !line.starts_with('/') {
                child.until(true, "Native run:");
            } else if line.starts_with("/compact ") {
                child.until(true, "Compacted context pre-warmed:");
            }
        }
        let output = child.finish();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        );
        (
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        )
    };
    let (stdout, stderr) = launch(
        false,
        "first task\nsecond task\nlatest task\n/compact 1 Old work resolved.\n/context\ncontinue compacted\n/context\n/fork child\n/context\nchild-only-task\n/switch parent\n/context\nparent followup\n/new\nfresh task\n/quit\n",
    );
    assert!(stdout.contains("the next prompt resends the full conversation"));
    assert!(stdout.contains("last model response: cached tokens unknown"));
    assert!(stdout.contains("cached tokens 7"));
    assert!(stdout.contains("compaction pre-warm:") && !stdout.contains("discarded warm"));
    assert!(stderr.contains("cached tokens 220") && stderr.contains("cached tokens 7"));
    assert!(stdout.contains("Compaction invalidates later causal KV"));
    let (stdout, _) = launch(true, "/context\nresumed followup\n/quit\n");
    assert!(stdout.contains("prompt boundary: resume/branch switch"));
    server.join().unwrap();
}

#[test]
fn cli_generation_flags_override_env_and_output_limit_refuses_before_generation() {
    for mode in ["direct", "agent"] {
        for (flag, env, expected, cache_env, cache) in [
            (Some("on"), "0", true, "1", true),
            (Some("off"), "1", false, "0", false),
            (None, "1", true, "1", true),
        ] {
            let f = WorkspaceFixture::new();
            let (listener, endpoint) = listen();
            let server = thread::spawn(move || {
                let mut reader = accept(&listener);
                capabilities(&mut reader, CAPS);
                let body = admission(&mut reader, 100, true);
                let request_value = json::parse(&body).unwrap();
                assert_eq!(
                    request_value.get("max_tokens"),
                    Some(&json::Value::Number("17".into()))
                );
                assert_eq!(
                    request_value.get("enable_thinking"),
                    Some(&json::Value::Bool(expected))
                );
                assert_eq!(request_value.get("cache"), Some(&json::Value::Bool(cache)));
                if mode == "agent" {
                    let system = request_value.get("messages").unwrap().as_array().unwrap()[0]
                        .get("content")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    assert!(system.contains("make that tool call in the same reply; never end a turn with only an announcement"));
                }
                assert_eq!(request(&mut reader).1, body);
                respond(
                    &mut reader,
                    200,
                    "text/event-stream",
                    &format!("{ANSWER}{}{DONE}", usage(100, 0)),
                );
            });
            let mut command = home::harness();
            command
                .args([
                    mode,
                    "--state-in-project",
                    "--endpoint",
                    &endpoint,
                    "--model",
                    "mock",
                    "--prompt",
                    "flags",
                    "--max-output-tokens",
                    "17",
                    "--workspace",
                ])
                .arg(&f.0)
                .env("VADE_HARNESS_THINKING", env)
                .env("VADE_HARNESS_PREFIX_CACHE", cache_env);
            if let Some(flag) = flag {
                command.args(["--thinking", flag]);
            }
            let output = cli::Cli::start(&mut command).finish();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            server.join().unwrap();
        }
        let f = WorkspaceFixture::new();
        let (listener, endpoint) = listen();
        let server = thread::spawn(move || {
            let mut reader = accept(&listener);
            capabilities(&mut reader, CAPS);
            assert_eq!(
                reader.read(&mut [0]).unwrap(),
                0,
                "no admission or generation after oversized output"
            );
        });
        let mut command = home::harness();
        command
            .args([
                mode,
                "--state-in-project",
                "--verbose",
                "--endpoint",
                &endpoint,
                "--model",
                "mock",
                "--prompt",
                "too much",
                "--max-output-tokens",
                "33",
                "--workspace",
            ])
            .arg(&f.0);
        let output = cli::Cli::start(&mut command).finish();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("the engine's output limit"));
        server.join().unwrap();
    }
}

#[test]
fn both_cli_modes_negotiate_cache_permission_and_honor_the_opt_out() {
    // Retention and affinity are independent advertised capabilities. The live
    // default engine has retention=false and affinity=true; a request cannot
    // turn retention on. Each environment setting lives in its own process.
    for mode in ["direct", "agent"] {
        let mut transcript = None;
        for (retention, affinity) in [(false, true), (true, true), (false, false), (true, false)] {
            let mut normalized = None;
            for setting in [None, Some("0"), Some("1")] {
                let workspace = WorkspaceFixture::new();
                let (listener, endpoint) = listen();
                let caps = CAPS.replace("\"prefix_cache\": true", &format!("\"prefix_cache\": {retention}"))
                    .replace("\"prompt_cache_key\": \"scheduling_hint\"", if affinity {
                        "\"prompt_cache_key\": \"scheduling_hint\""
                    } else { "\"prompt_cache_key\": null" });
                let server = thread::spawn(move || {
                    let mut reader = accept(&listener);
                    let mut generations = Vec::new();
                    let mut admitted = None;
                    while !reader.fill_buf().unwrap().is_empty() {
                        let (line, body) = request(&mut reader);
                        match line.as_str() {
                            "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", &caps),
                            "POST /v1/admission HTTP/1.1" => {
                                let view = json::view(&body).unwrap();
                                admitted = Some(view.get("request").unwrap().raw().to_owned());
                                let output = view.get("output_tokens").unwrap().raw().parse().unwrap();
                                respond(&mut reader, 200, "application/json", &admission_body(117, output, true));
                            }
                            "POST /v1/chat/completions HTTP/1.1" => {
                                assert_eq!(admitted.as_deref(), Some(body.as_str()), "cache decoration must precede exact admission");
                                generations.push(body);
                                respond(&mut reader, 200, "text/event-stream", &format!("{ANSWER}{}{DONE}", usage(117, 0)));
                            }
                            _ => panic!("unexpected cache-fixture request: {line}"),
                        }
                    }
                    generations
                });
                let mut command = home::harness();
                command.args([mode, "--state-in-project", "--endpoint", &endpoint, "--model", "mock", "--workspace"])
                    .arg(&workspace.0).args(["--plain", "--verbose", "--prompt", "hello", "--max-output-tokens", "17"])
                    .env_remove("VADE_API_KEY_FILE").env_remove("VADE_HARNESS_PREFIX_CACHE");
                if let Some(value) = setting { command.env("VADE_HARNESS_PREFIX_CACHE", value); }
                let output = cli::Cli::start(&mut command).finish();
                assert!(output.status.success(), "{mode} {setting:?}: {}", String::from_utf8_lossy(&output.stderr));
                assert!(String::from_utf8_lossy(&output.stdout).contains("answer"));
                assert!(String::from_utf8_lossy(&output.stderr).contains("cached tokens 0"));
                let bodies = server.join().unwrap();
                assert_eq!(bodies.len(), 1, "cache preference must not replay generation");
                let body = json::view(&bodies[0]).unwrap();
                assert_eq!(body.get("cache").map(|value| value.raw()),
                    (retention || affinity).then_some(if setting == Some("0") { "false" } else { "true" }));
                assert_eq!(body.get("prompt_cache_key").is_some(), affinity);
                let messages = body.get("messages").unwrap().raw().to_owned();
                if let Some(previous) = &transcript { assert_eq!(&messages, previous, "cache preference changed the transcript"); }
                else { transcript = Some(messages); }
                // Fresh sessions without sampling flags carry no sampling field,
                // and the cache setting changes nothing but `cache` and the
                // per-session `prompt_cache_key` (cache-defaults.md).
                assert!(!body.fields().unwrap().any(|(name, _)| sampling_policy::is_field(&name)),
                    "{mode}: cache negotiation must not add sampling fields: {}", bodies[0]);
                let json::Value::Object(fields) = json::parse(&bodies[0]).unwrap() else { panic!("object body") };
                let rest = json::stringify(&json::Value::Object(fields.into_iter()
                    .filter(|(name, _)| name != "cache" && name != "prompt_cache_key" && name != "cancel").collect())).unwrap();
                if let Some(previous) = &normalized {
                    assert_eq!(&rest, previous, "{mode} {retention}/{affinity} {setting:?}: cache setting changed another field");
                } else { normalized = Some(rest); }
            }
        }
    }
}

#[test]
fn cli_unspecified_sampling_stays_unspecified_across_retries_and_resume() {
    for mode in ["direct", "agent"] {
        for (flags, seed, temperature, resumed_flags) in [
            (&[][..], None, None, &[][..]),
            (&["--seed", "713"][..], Some(713u64), None, &[][..]),
            (&["--temperature", "0.5"][..], None, Some("0.5"), &["--temperature", "0.5"][..]),
            (&["--greedy"][..], None, Some("0"), &["--greedy"][..]),
            (&["--temperature", "0.5", "--seed", "713"][..], Some(713), Some("0.5"), &["--temperature", "0.5"][..]),
        ] {
            let workspace = WorkspaceFixture::new();
            let (listener, endpoint) = listen();
            let server = thread::spawn(move || {
                let mut captured = Vec::new();
                for connection in 0..if mode == "agent" { 2 } else { 1 } {
                    let mut reader = accept(&listener);
                    let mut admitted = None;
                    let mut turn = 0;
                    while !reader.fill_buf().unwrap().is_empty() {
                        let (line, body) = request(&mut reader);
                        match line.as_str() {
                            "GET /v1/capabilities HTTP/1.1" => respond(&mut reader, 200, "application/json", CAPS),
                            "POST /v1/admission HTTP/1.1" => {
                                let view = json::view(&body).unwrap();
                                admitted = Some(view.get("request").unwrap().raw().to_owned());
                                let output = view.get("output_tokens").unwrap().raw().parse().unwrap();
                                respond(&mut reader, 200, "application/json", &admission_body(117, output, true));
                            }
                            "POST /v1/chat/completions HTTP/1.1" => {
                                assert_eq!(admitted.as_deref(), Some(body.as_str()), "sampling must match exact admission");
                                let view = json::view(&body).unwrap();
                                let expected_seed = seed;
                                assert_eq!(view.get("seed").map(|value| value.raw().parse::<u64>().unwrap()), expected_seed);
                                assert_eq!(view.get("temperature").map(|value| value.raw()), temperature);
                                let explicit = view.fields().unwrap().any(|(name, _)| sampling_policy::is_field(&name));
                                assert_eq!(explicit, seed.is_some() || temperature.is_some(), "bookkeeping must not opt into named-field policy");
                                // This fixture selects the same field-presence branch as
                                // Engine. It does not simulate token selection or claim
                                // live arithmetic parity.
                                let label = if explicit { "explicit sampling" } else {
                                    let defaults = json::view(sampling_policy::GREEDY_DEFAULTS_JSON).unwrap();
                                    assert_eq!(defaults.get("temperature").unwrap().raw(), "0");
                                    "greedy defaults"
                                };
                                captured.push(body);
                                let event = format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{label}\"}},\"finish_reason\":\"stop\"}}]}}\n\n");
                                respond(&mut reader, 200, "text/event-stream", &format!("{event}{}{DONE}", usage(117, 0)));
                                turn += 1;
                            }
                            _ => panic!("unexpected sampling fixture request: {line}"),
                        }
                    }
                    assert_eq!(turn, if connection == 0 { 3 } else { 1 });
                }
                captured
            });
            let make = |resume: bool| {
                let mut command = home::harness();
                command.args([mode, "--state-in-project", "--endpoint", &endpoint, "--model", "mock", "--workspace"])
                    .arg(&workspace.0).args(["--plain", "--verbose", "--max-output-tokens", "17", "--no-auto-continue"])
                    .args(if resume { &["--resume", "sampler"][..] } else { &["--session", "sampler"][..] })
                    .args(if resume { resumed_flags } else { flags })
                    .env_remove("VADE_API_KEY_FILE");
                command
            };
            let mut child = cli::Cli::start(&mut make(false));
            for input in ["first\n", "/retry\n", "followup\n"] {
                child.send(input);
                child.until(true, "session reported cached tokens:");
            }
            child.send("/status\n");
            let output = child.finish();
            assert!(output.status.success(), "{mode} {flags:?}: {}", String::from_utf8_lossy(&output.stderr));
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains(if seed.is_some() || temperature.is_some() { "explicit sampling" } else { "greedy defaults" }));
            if seed.is_none() && temperature.is_none() {
                assert!(String::from_utf8_lossy(&output.stderr).contains("Sampling: upstream defaults (no sampling fields)"));
                assert!(stdout.contains("a greedy answer may repeat"));
            }
            if let Some(seed) = seed {
                assert!(stdout.contains(&format!("unchanged seed {seed}; use a different --seed")));
            }
            if mode == "agent" {
                let root = vade_harness::tools::Workspace::new(&workspace.0).unwrap();
                let (journal, _, _) = vade_harness::journal::Journal::resume(&root, "sampler", Limits::default()).unwrap();
                assert_eq!(journal.generation_seed(), seed);
                drop(journal);drop(root);
                let mut command = make(true);command.args(["--prompt", "resumed followup"]);
                let output = cli::Cli::start(&mut command).finish();
                assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            }
            let captured = server.join().unwrap();
            assert_eq!(captured.len(), if mode == "agent" { 4 } else { 3 });
            let first = json::parse(&captured[0]).unwrap();let retry = json::parse(&captured[1]).unwrap();
            assert_eq!(first.get("messages"), retry.get("messages"), "retry must preserve the retained conversation");
            if mode == "agent" { assert_eq!(retry.get("tool_choice").and_then(json::Value::as_str), Some("none")); }
        }
    }
}
