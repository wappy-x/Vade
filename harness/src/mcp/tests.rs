//! Tests for the MCP config, naming and result rules. Process tests with a
//! fake stdio server are in `tests/mcp.rs`.
use super::config::{self, Source};
use super::*;

fn value(text: &str) -> Value {
    json::parse(text).unwrap()
}

#[test]
fn entries_validate_names_fields_and_transport() {
    let spec = config::parse_entry("git-hub", &value(r#"{"command":"srv","args":["a","b"],"env":{"TOKEN":"x"},"cwd":"tools","enabled":false,"timeout_ms":250,"description":"other client key"}"#), Source::User).unwrap();
    assert_eq!((spec.command.as_str(), spec.args.len(), spec.env.len(), spec.cwd.as_deref(), spec.enabled, spec.timeout.as_millis(), spec.trusted), ("srv", 2, 1, Some("tools"), false, 250, true));
    assert_eq!(spec.directory(Path::new("/w")), Path::new("/w/tools"));
    let project = config::parse_entry("p", &value(r#"{"command":"srv","type":"stdio"}"#), Source::Project).unwrap();
    assert!(!project.trusted && project.enabled && project.timeout == config::DEFAULT_TIMEOUT);
    for (name, entry, reason) in [
        ("my_server", r#"{"command":"srv"}"#, "letters, digits or hyphens"),
        ("-x", r#"{"command":"srv"}"#, "starting with a letter or digit"),
        (&"x".repeat(33), r#"{"command":"srv"}"#, "letters, digits or hyphens"),
        ("s", r#"{"args":[]}"#, "command is required"),
        ("s", r#"{"command":""}"#, "command must be a nonempty string"),
        ("s", r#"{"command":"srv","args":[1]}"#, "args must be an array of strings"),
        ("s", r#"{"command":"srv","env":{"A":1}}"#, "env must be an object of strings"),
        ("s", r#"{"command":"srv","env":{"A=B":"1"}}"#, "invalid variable name"),
        ("s", r#"{"command":"srv","timeout_ms":0}"#, "1 to 600000"),
        ("s", r#"{"command":"srv","timeout_ms":600001}"#, "1 to 600000"),
        ("s", r#"{"command":"srv","enabled":"yes"}"#, "true or false"),
        ("s", r#"{"url":"https://example.com/mcp"}"#, "only stdio servers"),
        ("s", r#"{"command":"srv","type":"http"}"#, "only stdio servers"),
        ("s", r#"[]"#, "must be a JSON object"),
    ] {
        let error = config::parse_entry(name, &value(entry), Source::User).unwrap_err();
        assert!(error.contains(reason), "{entry}: {error}");
    }
}

#[test]
fn files_accept_either_server_key_and_trust_is_exact() {
    assert_eq!(config::servers(&value(r#"{"servers":{"a":{}}}"#)).unwrap().len(), 1);
    assert_eq!(config::servers(&value(r#"{"mcpServers":{"a":{},"b":{}}}"#)).unwrap().len(), 2);
    assert!(config::servers(&value(r#"{}"#)).unwrap().is_empty());
    assert!(config::servers(&value(r#"{"servers":{},"mcpServers":{}}"#)).is_err());
    assert!(config::servers(&value(r#"{"servers":[]}"#)).is_err());
    let entry = value(r#"{"command":"srv","args":["x"]}"#);
    let user = value(r#"{"trusted":[{"project":"/w/p","server":"s","entry":{"command":"srv","args":["x"]}}]}"#);
    assert!(config::is_trusted(&user, Path::new("/w/p"), "s", &entry));
    assert!(!config::is_trusted(&user, Path::new("/w/q"), "s", &entry));
    assert!(!config::is_trusted(&user, Path::new("/w/p"), "t", &entry));
    // A changed command needs a new yes.
    assert!(!config::is_trusted(&user, Path::new("/w/p"), "s", &value(r#"{"command":"srv","args":["y"]}"#)));
    assert!(!config::is_trusted(&value("{}"), Path::new("/w/p"), "s", &entry));
}

#[test]
fn pretty_output_parses_back_and_set_member_keeps_order() {
    let mut file = value(r#"{"note":"keep","servers":{"a":{"command":"x","args":["1","2"],"env":{}}},"trusted":[]}"#);
    config::set_member(&mut file, "servers", value(r#"{"b":{"command":"y"}}"#));
    config::set_member(&mut file, "extra", Value::Bool(true));
    let mut text = String::new();
    config::pretty(&file, 0, &mut text).unwrap();
    assert!(text.contains("\n  \"servers\": {\n    \"b\": {\n      \"command\": \"y\"\n    }\n  },"), "{text}");
    let back = value(&text);
    assert_eq!(back, file);
    let Value::Object(fields) = back else { unreachable!() };
    assert_eq!(fields.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["note", "servers", "trusted", "extra"]);
}

#[test]
fn tool_names_are_prefixed_sanitized_and_bounded() {
    assert_eq!(exposed_name("git", "create_issue").as_deref(), Some("git__create_issue"));
    assert_eq!(exposed_name("fs", "read.file/v2").as_deref(), Some("fs__read_file_v2"));
    assert_eq!(exposed_name("fs", "").as_deref(), None);
    assert_eq!(exposed_name("s", &"t".repeat(61)).map(|n| n.len()), Some(64));
    assert_eq!(exposed_name("s", &"t".repeat(62)), None);
    let mut notes = Vec::new();
    let tools = parse_tools("s", &value(r#"[
        {"name":"echo","description":"Echo text.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}},
        {"name":"bare"},
        {"name":"echo"},
        {"name":"a.b"},{"name":"a/b"},
        {"name":"array","inputSchema":{"type":"array"}},
        {"description":"no name"}
    ]"#).as_array().unwrap(), &mut notes);
    assert_eq!(tools.iter().map(|t| t.exposed.as_str()).collect::<Vec<_>>(), ["s__echo", "s__bare", "s__a_b"]);
    assert_eq!(tools[1].schema, value(r#"{"type":"object"}"#));
    assert_eq!(notes.len(), 4, "{notes:?}");
    let definition = json::stringify(&tools[0].definition("s")).unwrap();
    assert_eq!(definition, r#"{"type":"function","function":{"name":"s__echo","description":"MCP server s, tool echo. Echo text.","parameters":{"type":"object","properties":{"text":{"type":"string"}}}}}"#);
    let many: Vec<_> = (0..MAX_TOOLS_PER_SERVER + 3).map(|i| value(&format!(r#"{{"name":"t{i}"}}"#))).collect();
    let mut notes = Vec::new();
    assert_eq!(parse_tools("s", &many, &mut notes).len(), MAX_TOOLS_PER_SERVER);
    assert_eq!(notes.len(), 1);
    let huge = value(&format!(r#"[{{"name":"big","description":"{}"}}]"#, "d".repeat(500)));
    let wide = format!(r#"[{{"name":"wide","inputSchema":{{"type":"object","description":"{}"}}}}]"#, "w".repeat(MAX_SCHEMA_BYTES));
    let mut notes = Vec::new();
    assert_eq!(parse_tools("s", huge.as_array().unwrap(), &mut notes).len(), 1);
    assert!(parse_tools("s", value(&wide).as_array().unwrap(), &mut notes).is_empty() && notes[0].contains("16 KiB"));
}

#[test]
fn results_render_every_content_kind_and_respect_the_cap() {
    let result = value(r#"{"content":[
        {"type":"text","text":"line one"},
        {"type":"image","mimeType":"image/png","data":"AAAA"},
        {"type":"resource_link","uri":"file:///a","name":"a"},
        {"type":"resource","resource":{"uri":"file:///b","text":"body"}},
        {"type":"resource","resource":{"uri":"file:///c","blob":"AA=="}},
        {"type":"video"}
    ]}"#);
    assert_eq!(render_result(&result, 4096), "line one\n[image image/png, 4 base64 bytes not shown]\n[resource link file:///a a]\n[resource file:///b]\nbody\n[resource file:///c, binary not shown]\n[video content not shown]");
    assert_eq!(render_result(&value(r#"{"content":[],"structuredContent":{"n":1}}"#), 4096), r#"{"n":1}"#);
    let error = render_result(&value(r#"{"content":[{"type":"text","text":"bad \"input\""}],"isError":true}"#), 4096);
    assert_eq!(value(&error).get("message").and_then(Value::as_str), Some("bad \"input\""));
    assert_eq!(value(&error).get("error").and_then(Value::as_str), Some("mcp_tool_error"));
    for cap in [60, 61, 62, 63, 100] {
        let text = render_result(&value(&format!(r#"{{"content":[{{"type":"text","text":"{}"}}]}}"#, "é".repeat(100))), cap);
        assert!(text.len() <= cap && text.ends_with(" of 200 bytes]"), "{cap}: {text}");
    }
    assert_eq!(truncate("short".into(), 5), "short");
}

#[test]
fn router_sends_only_prefixed_names_to_mcp() {
    struct Native(usize);
    impl ToolExecutor for Native {
        fn execute(&mut self, _: ToolInvocation<'_>, _: &AtomicBool) -> Result<String, ToolError> {
            self.0 += 1;
            Ok("native".into())
        }
        fn last_side_effect_uncertain(&self) -> bool {
            true
        }
    }
    let call = |name: &str| crate::agent_wire::ToolCall { id: "c".into(), name: name.into(), arguments: "{}".into() };
    let arguments = value("{}");
    let cancel = AtomicBool::new(false);
    let hub = Mutex::new(Hub { root: PathBuf::from("/"), servers: Vec::new(), notes: Vec::new() });
    for policy in [ApprovalPolicy::Ask, ApprovalPolicy::Auto, ApprovalPolicy::Never] {
        let mut asked = Vec::new();
        let mut tools = McpTools::new(Native(0), Some(&hub), policy, |preview: Preview<'_>, _: &AtomicBool| {
            asked.push((preview.title, preview.approval_required));
            true
        });
        let native = call("workspace_read");
        assert_eq!(tools.execute(ToolInvocation { call: &native, arguments: &arguments, max_output_bytes: 4096 }, &cancel), Ok("native".into()));
        assert!(tools.last_side_effect_uncertain());
        let remote = call("gone__echo");
        let result = tools.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &cancel);
        assert!(!tools.last_side_effect_uncertain());
        assert_eq!(tools.inner.0, 1);
        if policy == ApprovalPolicy::Never {
            assert!(result.unwrap().contains(r#""error":"denied_by_policy""#));
        } else {
            assert!(result.unwrap().contains("mcp_unavailable"));
        }
        drop(tools);
        let expected: &[(&str, bool)] = match policy {
            ApprovalPolicy::Ask => &[("MCP tool", true)],
            ApprovalPolicy::Auto => &[("MCP tool", false)],
            ApprovalPolicy::Never => &[],
        };
        assert_eq!(asked, expected);
    }
    // A denied preview never reaches the hub, and the model learns who said
    // no: the user, or no one because no one could answer.
    let remote = call("s__echo");
    let approvals = crate::shell::permissions::Approvals::new(None, None, true).unwrap();
    for (interactive, error, cause) in [(true, "declined_by_user", "the user declined"), (false, "approval_required", "no one can approve it")] {
        approvals.set_interactive(interactive);
        let mut denied = McpTools::new(Native(0), Some(&hub), ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| false).with_approvals(&approvals);
        let text = denied.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &cancel).unwrap();
        assert!(text.contains(&format!(r#""error":"{error}""#)) && text.contains(cause) && !text.contains("tool server"), "{text}");
        assert_eq!(denied.inner.0, 0);
    }
    // A request the terminal could not show is not the user's no. A record
    // left on this thread before the request does not change a real no.
    let approvals = crate::shell::permissions::Approvals::new(None, None, true).unwrap();
    for shown in [false, true] {
        let me = std::thread::current().id();
        approvals.not_shown_for(me);
        let mut denied = McpTools::new(Native(0), Some(&hub), ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| {
            if !shown {
                approvals.not_shown_for(me);
            }
            false
        })
        .with_approvals(&approvals);
        let text = denied.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &cancel).unwrap();
        let error = if shown { "declined_by_user" } else { "approval_not_shown" };
        assert!(text.contains(&format!(r#""error":"{error}""#)), "{text}");
        assert_eq!(denied.inner.0, 0);
        assert!(!approvals.take_not_shown());
    }
    // Without the session's approval state the decline still names the user.
    let mut denied = McpTools::new(Native(0), Some(&hub), ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| false);
    assert!(denied.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &cancel).unwrap().contains("declined_by_user"));
    // A cancel during the question is a cancel, not a denial.
    let stopped = AtomicBool::new(true);
    let mut cancelled = McpTools::new(Native(0), Some(&hub), ApprovalPolicy::Ask, |_: Preview<'_>, _: &AtomicBool| false);
    assert_eq!(cancelled.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &stopped), Err(ToolError::Cancelled));
    let mut off = McpTools::new(Native(0), None, ApprovalPolicy::Auto, |_: Preview<'_>, _: &AtomicBool| true);
    assert!(off.execute(ToolInvocation { call: &remote, arguments: &arguments, max_output_bytes: 4096 }, &cancel).unwrap().contains("off in this chat"));
}

#[test]
fn large_lists_and_results_stay_bounded() {
    // A 129th tool that is invalid still ends the list at the cap.
    let mut listed: Vec<_> = (0..MAX_TOOLS_PER_SERVER).map(|i| value(&format!(r#"{{"name":"t{i}"}}"#))).collect();
    listed.push(value(r#"{"name":"bad","inputSchema":[]}"#));
    let mut notes = Vec::new();
    assert_eq!(parse_tools("s", &listed, &mut notes).len(), MAX_TOOLS_PER_SERVER);
    assert_eq!(notes, [format!("MCP s: kept the first {MAX_TOOLS_PER_SERVER} tools")]);
    // A long description is cut when it is read, not only when it is sent.
    let long = value(&format!(r#"[{{"name":"d","description":"{}"}}]"#, "é".repeat(5000)));
    let tools = parse_tools("s", long.as_array().unwrap(), &mut Vec::new());
    assert_eq!(tools[0].description.chars().count(), MAX_DESCRIPTION_CHARS);
    // Thousands of parts: the output is cut and the note counts every byte.
    let parts: Vec<_> = (0..2000).map(|_| format!(r#"{{"type":"text","text":"{}"}}"#, "y".repeat(999))).collect();
    let many = value(&format!(r#"{{"content":[{}]}}"#, parts.join(",")));
    for cap in [1, 20, 40, 4096] {
        let text = render_result(&many, cap);
        assert!(text.len() <= cap, "{cap}: {}", text.len());
        if cap >= 40 {
            assert!(text.ends_with(" of 1999999 bytes]"), "{text}");
        }
    }
    // A large tool error is cut inside the message, so the JSON stays closed,
    // also when every byte needs an escape.
    let quotes: Vec<_> = (0..200).map(|_| format!(r#"{{"type":"text","text":"{}"}}"#, "\\\"\\n".repeat(400))).collect();
    for parts in [&parts, &quotes] {
        let error = value(&format!(r#"{{"content":[{}],"isError":true}}"#, parts.join(",")));
        for cap in [80, 200, 4096] {
            let text = render_result(&error, cap);
            assert!(text.len() <= cap, "{cap}: {}", text.len());
            let parsed = json::parse(&text).unwrap_or_else(|_| panic!("{cap}: not JSON: {text}"));
            assert_eq!(parsed.get("error").and_then(Value::as_str), Some("mcp_tool_error"));
            let message = parsed.get("message").and_then(Value::as_str).unwrap();
            assert!(cap < 200 || message.contains("[output cut: kept "), "{text}");
        }
    }
    // Below the size of the note, keep only text, and never pass the cap.
    for cap in 0..40 {
        assert!(truncate("z".repeat(100), cap).len() <= cap);
    }
    assert_eq!(truncate("z".repeat(100), 5), "zzzzz");
}

#[cfg(unix)]
#[test]
fn config_files_refuse_links_and_fifos() {
    let f = crate::test_workspace::Fixture::new();
    let root = f.0.clone();
    std::fs::create_dir(root.join(".vade")).unwrap();
    let path = config::project_path(&root);
    assert_eq!(config::read_file(&path).unwrap(), value("{}"));
    // A FIFO in place of the file fails at once instead of blocking.
    assert!(std::process::Command::new("mkfifo").arg(&path).status().unwrap().success());
    let started = std::time::Instant::now();
    let error = config::read_file(&path).unwrap_err().to_string();
    assert!(error.contains("must be a regular file") && started.elapsed() < std::time::Duration::from_secs(2), "{error}");
    std::fs::remove_file(&path).unwrap();
    // A symlinked file or folder is refused for reads and writes.
    let outside = crate::test_workspace::Fixture::new();
    std::fs::write(outside.0.join("mcp.json"), r#"{"servers":{}}"#).unwrap();
    std::os::unix::fs::symlink(outside.0.join("mcp.json"), &path).unwrap();
    assert!(config::read_file(&path).unwrap_err().to_string().contains("is a symlink"));
    assert!(config::write_file(&path, &value("{}")).unwrap_err().to_string().contains("is a symlink"));
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(root.join(".vade")).unwrap();
    std::os::unix::fs::symlink(&outside.0, root.join(".vade")).unwrap();
    assert!(config::read_file(&path).unwrap_err().to_string().contains("is a symlink"));
    assert!(config::write_file(&path, &value(r#"{"servers":{}}"#)).is_err());
    assert_eq!(std::fs::read_to_string(outside.0.join("mcp.json")).unwrap(), r#"{"servers":{}}"#);
    assert_eq!(std::fs::read_dir(&outside.0).unwrap().count(), 1);
}

#[test]
fn deep_or_many_definitions_never_reach_the_chat_request() {
    // A schema nested past the limit is skipped with a note, not sent.
    let deep = format!(r#"{{"type":"object","properties":{{"a":{}}}}}"#, "{\"items\":".repeat(40) + "{}" + &"}".repeat(40));
    let listed = value(&format!(r#"[{{"name":"deep","inputSchema":{deep}}},{{"name":"flat"}}]"#));
    let mut notes = Vec::new();
    let tools = parse_tools("s", listed.as_array().unwrap(), &mut notes);
    assert_eq!(tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["flat"]);
    assert!(notes[0].contains("nests deeper than 32 levels"), "{notes:?}");
    // Many tools: later tools are left out once all definitions reach the value budget.
    let wide = format!(r#"{{"type":"object","properties":{{{}}}}}"#, (0..300).map(|i| format!(r#""p{i}":{{"type":"string"}}"#)).collect::<Vec<_>>().join(","));
    let listed: Vec<_> = (0..40).map(|i| value(&format!(r#"{{"name":"t{i}","inputSchema":{wide}}}"#))).collect();
    let mut servers: Vec<Server> = ["a", "b"].iter().map(|name| {
        let spec = config::parse_entry(name, &value(r#"{"command":"x"}"#), Source::User).unwrap();
        let tools = parse_tools(name, &listed[..20], &mut Vec::new());
        Server { spec, connection: None, tools, resources: Vec::new(), prompts: Vec::new(), error: None, restarted: false }
    }).collect();
    let mut notes = Vec::new();
    fit_budget(&mut servers, &mut notes);
    let kept: Vec<Value> = servers.iter().flat_map(|s| s.tools.iter().map(|t| t.definition(&s.spec.name))).collect();
    assert!(!kept.is_empty() && kept.len() < 40);
    assert!(kept.iter().map(|d| shape(d).1).sum::<usize>() <= MAX_DEFINITION_VALUES);
    assert!(notes.iter().any(|n| n.starts_with("MCP b: left out ")), "{notes:?}");
}

#[test]
fn large_definitions_stay_within_the_byte_budget_and_the_chat_starts() {
    // Each definition is near the 16 KiB limit and has few values; 3 x 128
    // of them pass every per-tool check but are 6 MiB together.
    let listed: Vec<_> = (0..128).map(|i| value(&format!(r#"{{"name":"t{i}","inputSchema":{{"type":"object","description":"{}"}}}}"#, "x".repeat(16000)))).collect();
    let mut servers: Vec<Server> = ["a", "b", "c"].iter().map(|name| {
        let spec = config::parse_entry(name, &value(r#"{"command":"x"}"#), Source::User).unwrap();
        let mut notes = Vec::new();
        let tools = parse_tools(name, &listed, &mut notes);
        assert_eq!((tools.len(), notes.len()), (128, 0), "{notes:?}");
        Server { spec, connection: None, tools, resources: Vec::new(), prompts: Vec::new(), error: None, restarted: false }
    }).collect();
    let mut notes = Vec::new();
    fit_budget(&mut servers, &mut notes);
    let kept: Vec<Value> = servers.iter().flat_map(|s| s.tools.iter().map(|t| t.definition(&s.spec.name))).collect();
    let bytes: usize = kept.iter().map(|d| json::stringify(d).unwrap().len()).sum();
    assert!(!kept.is_empty() && bytes <= MAX_DEFINITION_BYTES, "{bytes}");
    assert!(notes.iter().any(|n| n.starts_with("MCP a: left out ") && n.contains("512 KiB")), "{notes:?}");
    assert!(notes.iter().any(|n| n.starts_with("MCP c: left out 128 tools")), "{notes:?}");
    // The native tools and the kept MCP tools fit one chat request.
    let Value::Array(mut tools) = crate::coding_tools::definitions() else { unreachable!() };
    tools.extend(kept);
    let request = Value::Object(vec![
        ("tools".into(), Value::Array(tools)),
        ("messages".into(), value(r#"[{"role":"system","content":"rules"},{"role":"user","content":"hi"}]"#)),
    ]);
    crate::agent::Conversation::from_request(request, crate::agent::Limits::default()).unwrap();
}

#[test]
fn chat_home_files_and_env_names_stay_valid() {
    assert_eq!(config::user_path_in(Some(Path::new("/srv/vade"))).unwrap(), Path::new("/srv/vade/mcp.json"));
    assert!(config::user_path_in(Some(Path::new("relative"))).is_err());
    assert!(config::user_path_in(Some(Path::new("/srv/../vade"))).is_err());
    // An entry built in memory can repeat a key; json::parse never can.
    let twice = Value::Object(vec![
        ("command".into(), Value::String("srv".into())),
        ("env".into(), Value::Object(vec![("T".into(), Value::String("a".into())), ("T".into(), Value::String("b".into()))])),
    ]);
    assert!(config::parse_entry("s", &twice, Source::User).unwrap_err().contains("env sets T more than once"));
    // A file larger than the read limit is never written.
    let dir = std::env::temp_dir().join(format!("vade-mcp-big-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let big = Value::Object(vec![("note".into(), Value::String("x".repeat(config::MAX_CONFIG_BYTES as usize)))]);
    let error = config::write_file(&dir.join("mcp.json"), &big).unwrap_err();
    assert!(error.to_string().contains("larger than 1 MiB"), "{error}");
    assert!(!dir.join("mcp.json").exists());
    // The whole file must pass the read limits, though each part does alone.
    let path = dir.join("mcp.json");
    config::write_file(&path, &value(r#"{"servers":{}}"#)).unwrap();
    let old = std::fs::read(&path).unwrap();
    let mut nested = Value::Number("0".into());
    for _ in 0..64 {
        nested = Value::Array(vec![nested]);
    }
    let deep = Value::Object(vec![("trusted".into(), Value::Array(vec![Value::Object(vec![("entry".into(), nested)])]))]);
    let many = Value::Object(vec![
        ("note".into(), Value::Array(vec![Value::Number("0".into()); 99_990])),
        ("servers".into(), value(r#"{"a":{"command":"x","args":["1","2","3","4","5","6","7","8","9","10"]}}"#)),
    ]);
    for file in [deep, many] {
        let error = config::write_file(&path, &file).unwrap_err();
        assert!(error.to_string().contains("nesting or value limits"), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), old);
        config::read_file(&path).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn quoted_stderr_hides_overlapping_values_at_every_read_boundary() {
    let secrets = || vec![b"hunt".to_vec(), b"hunter2-secret".to_vec()];
    let stream = b"token hunter2-secret, then hunt alone, then hunter3 and hunter2-sec";
    // "hunter3" starts with the value "hunt"; the end may still become the long value.
    let expected = "token [env value], then [env value] alone, then [env value]er3 and";
    // Every split into two reads, and every split into three reads.
    for first in 0..=stream.len() {
        for second in first..=stream.len() {
            let mut tail = client::Redactor::new(secrets());
            for part in [&stream[..first], &stream[first..second], &stream[second..]] {
                tail.push(part);
            }
            assert_eq!(tail.text(), expected, "reads end at {first} and {second}");
        }
    }
    // A value that starts inside another match: the union of both is hidden.
    let secrets = || vec![b"prefix-hunt".to_vec(), b"hunter2-secret".to_vec()];
    let stream = b"token prefix-hunter2-secret end, prefix-hunter2 end";
    for first in 0..=stream.len() {
        for second in first..=stream.len() {
            let mut tail = client::Redactor::new(secrets());
            for part in [&stream[..first], &stream[first..second], &stream[second..]] {
                tail.push(part);
            }
            assert_eq!(tail.text(), "token [env value] end, [env value]er2 end", "reads end at {first} and {second}");
        }
    }
    // A value longer than the kept tail, cut anywhere, shows none of itself.
    let long: Vec<u8> = (0..300).flat_map(|i| format!("key line {i}\n").into_bytes()).collect();
    for size in [1, 7, 1024, 4096] {
        let mut tail = client::Redactor::new(vec![long.clone()]);
        let mut stream = b"start ".to_vec();
        stream.extend_from_slice(&long);
        stream.extend_from_slice(b" end");
        for part in stream.chunks(size) {
            tail.push(part);
        }
        assert_eq!(tail.text(), "start [env value] end", "reads of {size}");
    }
}
