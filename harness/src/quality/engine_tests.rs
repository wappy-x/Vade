//! Unit tests for the engine card: the set itself, every verdict with a pass
//! and a fail case, the grade rules, the server facts and the card text.
use super::engine::*;
use super::engine_cases::*;
use super::*;

/// One value the schema accepts: the first enum choice, else a fixed value
/// per type, with every property filled.
fn instance(schema: &Value) -> Value {
    if let Some(first) = schema.get("enum").and_then(Value::as_array).and_then(|c| c.first()) {
        return first.clone();
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("object") => Value::Object(match schema.get("properties") {
            Some(Value::Object(properties)) => {
                properties.iter().map(|(key, sub)| (key.clone(), instance(sub))).collect()
            }
            _ => Vec::new(),
        }),
        Some("array") => Value::Array(vec![schema.get("items").map_or(Value::Null, instance)]),
        Some("integer") => n(schema.get("minimum").and_then(|v| match v {
            Value::Number(t) => t.parse::<i64>().ok(),
            _ => None,
        }).unwrap_or(7)),
        Some("number") => n("2.5"),
        Some("boolean") => Value::Bool(true),
        _ => s("text"),
    }
}

fn reply(ids: &[u32], text: &str) -> Reply {
    Reply {
        text: text.into(),
        token_ids: ids.to_vec(),
        saw_done: true,
        finish_reason: Some("stop".into()),
        completion_tokens: Some(ids.len() as u64),
        prompt_tokens: Some(900),
        cached_tokens: Some(0),
        ttft_s: Some(0.05),
        draft: Some("auto".into()),
        ..Reply::default()
    }
}

#[test]
fn filler_is_deterministic_plain_text() {
    assert_eq!(filler(7, 40), filler(7, 40));
    assert_ne!(filler(7, 40), filler(8, 40));
    let text = filler(7, 400);
    assert!(!text.chars().any(|c| c.is_ascii_digit()), "filler holds no digits");
    for needle in &NEEDLES {
        let planted = needle_prompt(needle, 100, true);
        assert_eq!(planted.matches(needle.answer).count(), 1);
        assert!(!needle_prompt(needle, 100, false).contains(needle.answer));
        assert!(planted.ends_with(needle.question));
    }
}

#[test]
fn calibration_sizes_prompts_from_probe_counts() {
    // 64 sentences -> 900 tokens, 320 -> 3,972: 12 tokens per sentence, 132 frame.
    let cal = Calibration::from_probes([900, 3_972]).expect("calibration");
    assert!((cal.tokens_per_sentence - 12.0).abs() < 1e-9);
    assert!((cal.frame_tokens - 132.0).abs() < 1e-9);
    let sentences = cal.sentences_for(8_192);
    assert!(sentences as f64 * 12.0 + 132.0 >= 8_192.0 * 1.02, "margin kept: {sentences}");
    assert!(Calibration::from_probes([900, 900]).is_err());
    assert!(Calibration::from_probes([900, 100]).is_err());
}

#[test]
fn every_fixed_schema_and_tool_case_is_consistent() {
    assert_eq!(JSON_CASES.len() * 5, 50);
    assert_eq!(TOOL_CASES.len(), 20);
    for case in &JSON_CASES {
        let schema = json::parse(case.schema).expect("schema parses");
        let good = instance(&schema);
        assert_eq!(schema_failure(&schema, &good, "$"), Ok(()), "{}", case.name);
        let answer = json::stringify(&good).unwrap();
        assert_eq!(json_answer_failure(case.schema, &answer), None, "{}", case.name);
        let keys = match &schema.get("properties") {
            Some(Value::Object(properties)) => properties.len(),
            _ => 0,
        };
        assert_eq!(case.keys.matches(" (").count(), keys, "prompt names every key of {}", case.name);
    }
    for case in &JSON_CASES {
        assert!(!json_prompt(case, SCHEMA_ONLY_SOURCE).contains(case.keys));
        assert!(json_prompt(case, 0).contains(case.keys));
    }
    for case in &TOOL_CASES {
        let tool = TOOLS.iter().find(|tool| tool.name == case.name).expect("tool exists");
        let parameters = json::parse(tool.parameters).expect("parameters parse");
        let arguments = json::parse(case.arguments).expect("arguments parse");
        assert_eq!(schema_failure(&parameters, &arguments, "$"), Ok(()), "{}", case.id);
    }
    let ids: Vec<&str> = TOOL_CASES.iter().map(|case| case.id).collect();
    assert!(ids.iter().enumerate().all(|(at, id)| !ids[..at].contains(id)), "unique tool ids");
}

#[test]
fn json_answers_fail_on_form_and_schema() {
    let case = &JSON_CASES[0];
    assert_eq!(json_answer_failure(case.schema, r#" {"name":"Ada","age":34,"role":"pilot"} "#), None);
    let fails = [
        ("```json\n{\"name\":\"Ada\",\"age\":34,\"role\":\"pilot\"}\n```", "not one JSON value"),
        (r#"{"name":"Ada","age":34}"#, "misses key role"),
        (r#"{"name":"Ada","age":34,"role":"pilot","x":1}"#, "extra key x"),
        (r#"{"name":"Ada","age":"34","role":"pilot"}"#, "not of type integer"),
        (r#"{"name":"Ada","age":34.5,"role":"pilot"}"#, "not an integer"),
        (r#"{"name":"Ada","age":151,"role":"pilot"}"#, "out of range"),
        (r#"{"name":"Ada","name":"Bo","age":3,"role":"pilot"}"#, "not one JSON value"),
        (r#"{"name":"Ada","age":34,"role":"pilot"} trailing"#, "not one JSON value"),
    ];
    for (answer, want) in fails {
        let reason = json_answer_failure(case.schema, answer).expect(answer);
        assert!(reason.contains(want), "{answer}: {reason}");
    }
    let weather = &JSON_CASES[2];
    let reason = json_answer_failure(weather.schema, r#"{"city":"Oslo","temperature_c":-4,"condition":"fog"}"#);
    assert!(reason.is_some_and(|r| r.contains("enum")));
}

#[test]
fn tool_calls_must_match_name_and_arguments_exactly() {
    let case = &TOOL_CASES[4]; // convert_currency 250 USD -> EUR
    let call = |name: &str, arguments: &str| ToolCall { index: 0, id: None, name: name.into(), arguments: arguments.into() };
    let mut answer = Reply::default();
    answer.tool_calls = vec![call("convert_currency", r#"{ "to": "EUR", "amount": 250.0, "from": "USD" }"#)];
    assert_eq!(tool_call_failure(case, &answer), None, "key order, spacing and 250.0 are free");
    answer.tool_calls = vec![call("get_weather", r#"{"amount":250,"from":"USD","to":"EUR"}"#)];
    assert!(tool_call_failure(case, &answer).is_some_and(|r| r.contains("called get_weather")));
    answer.tool_calls = vec![call("convert_currency", r#"{"amount":250,"from":"usd","to":"EUR"}"#)];
    assert!(tool_call_failure(case, &answer).is_some_and(|r| r.contains("arguments")));
    answer.tool_calls = vec![call("convert_currency", r#"{"amount":250,"from":"USD","to":"EUR","note":"x"}"#)];
    assert!(tool_call_failure(case, &answer).is_some());
    answer.tool_calls = vec![call("convert_currency", r#"{"amount":250,"from":"USD""#)];
    assert!(tool_call_failure(case, &answer).is_some_and(|r| r.contains("do not parse")));
    answer.tool_calls.clear();
    answer.text = "I would convert it.".into();
    assert!(tool_call_failure(case, &answer).is_some_and(|r| r.contains("no tool call")));
    let good = call("convert_currency", case.arguments);
    answer.tool_calls = vec![good.clone(), ToolCall { index: 1, ..good }];
    assert!(tool_call_failure(case, &answer).is_some_and(|r| r.contains("2 tool calls")));
}

#[test]
fn streamed_tool_call_parts_join_by_index() {
    let mut answer = Reply::default();
    for data in [
        r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"set_timer","arguments":""}}]},"token_id":5}]}"#,
        r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"minutes\": 4, "}}]},"token_id":6}]}"#,
        r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"label\": \"tea\"}"}}]},"token_id":7}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
    ] {
        apply_sse_data(&mut answer, data, 0.1).expect("event");
    }
    assert_eq!(answer.tool_calls.len(), 1);
    assert_eq!(answer.tool_calls[0].id.as_deref(), Some("call-1"));
    assert_eq!(answer.tool_calls[0].name, "set_timer");
    assert_eq!(tool_call_failure(&TOOL_CASES[8], &answer), None);
    assert_eq!(answer.token_ids, vec![5, 6, 7]);
}

#[test]
fn pairs_need_complete_evidence_and_identical_ids() {
    let base = reply(&[1, 2, 3], "abc");
    assert_eq!(pair_failure(&base, &base.clone()), None);
    let other = reply(&[1, 9, 3], "abc");
    assert!(pair_failure(&base, &other).is_some_and(|r| r.contains("ids differ at token 1: 2 then 9")));
    let short = reply(&[1, 2], "ab");
    assert!(pair_failure(&base, &short).is_some_and(|r| r.contains("finish or count differs")));
    let mut cut = base.clone();
    cut.saw_done = false;
    assert!(pair_failure(&base, &cut).is_some_and(|r| r.contains("no terminal event")));
    let mut plain = base.clone();
    plain.draft = Some("plain".into());
    assert_eq!(pair_failure(&base, &plain), None, "drafting on and off must match; the mode may differ");
    let mut tool_end = base.clone();
    tool_end.finish_reason = Some("tool_calls".into());
    tool_end.completion_tokens = Some(4);
    assert_eq!(pair_failure(&tool_end, &tool_end.clone()), None, "a tool_calls finish counts the unstreamed end token");
    tool_end.finish_reason = Some("length".into());
    assert!(pair_failure(&tool_end, &tool_end.clone()).is_some_and(|r| r.contains("token id count mismatch")));
    let mut text = base.clone();
    text.text = "abd".into();
    assert!(pair_failure(&base, &text).is_some_and(|r| r.contains("texts differ")));
    assert!(pair_failure(&reply(&[], ""), &reply(&[], "")).is_some(), "no ids proves nothing");
}

#[test]
fn cache_items_need_a_cold_miss_and_a_warm_hit() {
    let cold = reply(&[1, 2, 3], "abc");
    let mut warm = cold.clone();
    warm.cached_tokens = Some(832);
    let item = |cold: Reply, warm: Reply| CacheItem { index: 0, cold: Some(cold), prime: None, warm: Some(warm), error: None };
    assert!(item(cold.clone(), warm.clone()).passed());
    let mut hot = cold.clone();
    hot.cached_tokens = Some(64);
    assert!(item(hot, warm.clone()).failure_reason().is_some_and(|r| r.contains("cold run read 64")));
    assert!(item(cold.clone(), cold.clone()).failure_reason().is_some_and(|r| r.contains("warm run reused no cached tokens")));
    let mut changed = warm.clone();
    changed.token_ids[2] = 4;
    assert!(item(cold.clone(), changed).failure_reason().is_some_and(|r| r.contains("ids differ at token 2")));
    let failed = CacheItem { index: 0, cold: Some(cold), prime: None, warm: None, error: Some("warm run failed: x".into()) };
    assert!(!failed.passed());
}

#[test]
fn concurrency_counts_only_overlapping_identical_copies() {
    let alone = reply(&[4, 5, 6], "def");
    let copy = |ids: &[u32], start_s: f64, end_s: f64| CopyRun { reply: Ok(reply(ids, "def")), start_s, end_s };
    let mut item = ConcurrencyItem {
        id: "x",
        alone: Ok(alone),
        copies: (0..CONCURRENT_COPIES).map(|i| copy(&[4, 5, 6], i as f64 * 0.01, 1.0)).collect(),
    };
    assert!(item.overlapped());
    assert_eq!(item.identical(), CONCURRENT_COPIES);
    item.copies[3] = copy(&[4, 5, 7], 0.03, 1.0);
    assert_eq!(item.identical(), CONCURRENT_COPIES - 1);
    assert!(item.copy_failure(3).is_some_and(|r| r.contains("ids differ at token 2")));
    item.copies[3] = CopyRun { reply: Err("upstream HTTP status line: HTTP/1.1 503".into()), start_s: 0.0, end_s: 0.1 };
    assert_eq!(item.identical(), CONCURRENT_COPIES - 1);
    // One after another: the last copy starts after the first one ends.
    let serial = ConcurrencyItem {
        id: "y",
        alone: Ok(reply(&[1], "a")),
        copies: (0..CONCURRENT_COPIES).map(|i| copy(&[1], i as f64, i as f64 + 0.5)).collect(),
    };
    assert!(!serial.overlapped());
    assert_eq!(serial.identical(), 0, "copies that never ran together prove nothing");
}

#[test]
fn grade_rules() {
    let full = |total| (total, total);
    let base = (full(2), full(50), full(20), full(20), full(24));
    let g = |(a, b, c, d, e): ((usize, usize), (usize, usize), (usize, usize), (usize, usize), (usize, usize))| grade(a, b, c, d, e);
    assert_eq!(g(base), Grade::APlus);
    assert_eq!(g((base.0, (49, 50), base.2, base.3, base.4)), Grade::A);
    assert_eq!(g((base.0, base.1, (19, 20), base.3, base.4)), Grade::A);
    assert_eq!(g((base.0, (49, 50), (19, 20), base.3, base.4)), Grade::B);
    assert_eq!(g((base.0, (48, 50), base.2, base.3, base.4)), Grade::C);
    assert_eq!(g((base.0, base.1, (18, 20), base.3, base.4)), Grade::C);
    assert_eq!(g(((1, 2), base.1, base.2, base.3, base.4)), Grade::C);
    assert_eq!(g((base.0, base.1, base.2, (19, 20), base.4)), Grade::C);
    assert_eq!(g((base.0, base.1, base.2, base.3, (23, 24))), Grade::C);
    assert_eq!(g(((0, 0), base.1, base.2, base.3, base.4)), Grade::C, "an empty check never passes");
    assert_eq!(Grade::APlus.label(), "A+");
}

#[test]
fn server_facts_come_from_the_vade_object() {
    let mut answer = reply(&[1], "a");
    answer.served_model = Some("served-model".into());
    answer.vade_receipt = Some(
        r#"{"draft":"auto","mtp_depth":3,"build":{"git_sha":"0123456789abcdef0123456789abcdef01234567","dirty":false,"source":"git"},"checkpoint":{"directory":"ckpt","sha256":"feedface"},"numerics":{"nvfp4":true,"mtp_depth":3,"world_size":1,"topology":"local"}}"#.into(),
    );
    let facts = ServerFacts::from_reply(&answer);
    assert_eq!(facts.git_sha.as_deref(), Some("0123456789abcdef0123456789abcdef01234567"));
    assert_eq!((facts.dirty, facts.nvfp4, facts.mtp_depth, facts.world_size), (Some(false), Some(true), Some(3), Some(1)));
    assert_eq!(facts.draft.as_deref(), Some("auto"));
    assert_eq!(facts.checkpoint_dir.as_deref(), Some("ckpt"));
    assert_eq!(ServerFacts::from_reply(&reply(&[1], "a")).git_sha, None);
}

#[test]
fn engine_set_flags_parse_and_refuse_quality_only_options() {
    let args = |extra: &[&str]| {
        let mut list: Vec<String> = ["--endpoint", "http://127.0.0.1:9", "--model", "m", "--json", "r.json"]
            .iter()
            .map(|a| a.to_string())
            .collect();
        list.extend(extra.iter().map(|a| a.to_string()));
        list
    };
    assert_eq!(Config::parse(args(&["--set", "engine-1"])).expect("engine set").set, CardSet::Engine);
    assert_eq!(Config::parse(args(&[])).expect("default").set, CardSet::Quality);
    assert_eq!(Config::parse(args(&["--set", "quality-set-1", "--only", "quality"])).expect("quality").set, CardSet::Quality);
    assert!(Config::parse(args(&["--set", "engine-2"])).is_err());
    for flag in [&["--only", "quality"][..], &["--no-code-run"], &["--server-log", "x.log"], &["--rustc", "rustc"]] {
        let mut extra = vec!["--set", "engine-1"];
        extra.extend_from_slice(flag);
        assert!(Config::parse(args(&extra)).is_err(), "{flag:?} refused with engine-1");
    }
}

fn sample_report() -> EngineReport {
    let config = Config { model: "mock-model".into(), set: CardSet::Engine, ..Config::default() };
    let good = reply(&[1, 2], "ok");
    EngineReport {
        config,
        server: ServerFacts { git_sha: Some("ab".repeat(20)), world_size: Some(1), nvfp4: Some(true), mtp_depth: Some(3), draft: Some("auto".into()), ..ServerFacts::default() },
        calibration: Calibration::from_probes([900, 3_972]),
        needles: NEEDLES
            .iter()
            .map(|needle| {
                let mut answer = reply(&[1], needle.answer);
                answer.prompt_tokens = Some(needle.target_tokens + 10);
                NeedleItem { needle, sentences: 10, prompt_chars: 100, reply: Some(answer), error: None, sizing: NeedleSizing::default() }
            })
            .collect(),
        json: JSON_CASES
            .iter()
            .flat_map(|case| (0..5).map(move |source| (case, source)))
            .map(|(case, source)| {
                let value = instance(&json::parse(case.schema).unwrap());
                JsonItem { case, source, reply: Some(reply(&[1], &json::stringify(&value).unwrap())), error: None }
            })
            .collect(),
        tools: TOOL_CASES
            .iter()
            .map(|case| {
                let mut answer = reply(&[1], "");
                answer.tool_calls = vec![ToolCall { index: 0, id: None, name: case.name.into(), arguments: case.arguments.into() }];
                ToolItem { case, reply: Some(answer), error: None }
            })
            .collect(),
        cache: (0..CACHE_PROMPTS)
            .map(|index| {
                let mut warm = good.clone();
                warm.cached_tokens = Some(832);
                CacheItem { index, cold: Some(good.clone()), prime: Some(warm.clone()), warm: Some(warm), error: None }
            })
            .collect(),
        concurrency: CONCURRENCY_PROMPTS
            .iter()
            .map(|(id, _)| ConcurrencyItem {
                id,
                alone: Ok(good.clone()),
                copies: (0..CONCURRENT_COPIES).map(|_| CopyRun { reply: Ok(good.clone()), start_s: 0.0, end_s: 1.0 }).collect(),
            })
            .collect(),
        elapsed_s: 12.5,
    }
}

#[test]
fn card_renders_in_width_with_grade_and_table() {
    let mut report = sample_report();
    assert_eq!(report.grade(), Grade::APlus);
    let card = report.card();
    let at = card.find("╯\n").expect("box then table") + "╯".len();
    let (boxed, table) = card.split_at(at);
    for line in boxed.lines() {
        assert_eq!(line.chars().count(), 74, "card line width: {line:?}");
    }
    assert!(boxed.contains("GRADE A+  ·  5/5 CHECKS PASS"), "{boxed}");
    assert!(boxed.contains(&"ab".repeat(20)));
    assert!(boxed.contains("concurrency per prompt: 8/8 8/8 8/8"));
    assert!(table.contains("| **A+** |"), "{table}");
    assert!(table.contains("| 5 | Concurrency invariance | PASS | 24/24 (8/8 8/8 8/8) |"), "{table}");
    let receipt = json::parse(&report.to_json().unwrap()).expect("receipt parses");
    assert_eq!(receipt.get("schema").and_then(Value::as_str), Some("vade.engine-card.v1"));
    assert_eq!(receipt.get("grade").and_then(Value::as_str), Some("A+"));

    report.json[3].reply.as_mut().unwrap().text = "not json".into();
    assert_eq!(report.grade(), Grade::A);
    let card = report.card();
    assert!(card.contains("GRADE A  ·  4/5 CHECKS PASS"), "{card}");
    assert!(card.contains("✗ json person-4: not one JSON value"), "{card}");
    report.cache[0].warm.as_mut().unwrap().token_ids[1] = 9;
    assert_eq!(report.grade(), Grade::C);
    assert!(report.card().contains("✗ cache prompt-1: ids differ at token 1: 2 then 9"));
}

#[test]
fn card_takes_server_text_by_characters_and_leaves_out_host_paths() {
    let mut report = sample_report();
    // A multi-byte character across byte 16 of a server-supplied hash.
    report.server.checkpoint_sha256 = Some(format!("{}é{}", "a".repeat(15), "b".repeat(40)));
    report.server.checkpoint_dir = Some("/home/someone/models/ckpt".into());
    let card = report.card();
    assert!(card.contains(&format!("Checkpoint sha256 {}é", "a".repeat(15))), "{card}");
    assert!(!card.contains("/home/someone"), "the card leaves out the checkpoint directory: {card}");
    let receipt = json::parse(&report.to_json().unwrap()).expect("receipt parses");
    let server = receipt.get("server").expect("server facts");
    assert_eq!(server.get("checkpoint_directory").and_then(Value::as_str), Some("/home/someone/models/ckpt"));
    assert!(report.markdown_table().starts_with("| Build sha |"), "{}", report.markdown_table());
}

/// A server that counts 132 frame tokens plus 12 per filler sentence (the
/// probes above) and admits a prompt when it and the answer fit `context`.
fn admit_with(context: u64, checks: &std::cell::Cell<usize>) -> impl FnMut(usize) -> Result<Admission, String> + '_ {
    move |sentences| {
        checks.set(checks.get() + 1);
        let tokens = 132 + 12 * sentences as u64;
        let input_limit = context - 1;
        let admitted = tokens <= input_limit && tokens + NEEDLE_ANSWER_TOKENS <= context;
        Ok(Admission { prompt_tokens: tokens, context_limit: context, input_limit, admitted, reason: (!admitted).then(|| "too long".into()) })
    }
}

#[test]
fn a_needle_beyond_the_server_context_is_not_run_before_sending() {
    let cal = Calibration::from_probes([900, 3_972]).expect("calibration");
    let checks = std::cell::Cell::new(0);
    let NeedleSize::NotRun { admission, reason, .. } = size_needle(32_768, &cal, admit_with(32_768, &checks)) else {
        panic!("a 32,768-token context cannot hold a 32,768-token prompt and its answer");
    };
    assert_eq!(checks.get(), 1, "one admission check decides it");
    assert_eq!(admission.ceiling(NEEDLE_ANSWER_TOKENS), 32_736);
    for text in ["at most 32736 prompt tokens", "context 32768", "input limit 32767", "at least 32768", "at least 32800 tokens", "vade up --context 40960"] {
        assert!(reason.contains(text), "{text}: {reason}");
    }
}

#[test]
fn a_needle_is_sized_to_the_room_the_server_admits() {
    let cal = Calibration::from_probes([900, 3_972]).expect("calibration");
    // A large context admits the calibrated size at the first check.
    let checks = std::cell::Cell::new(0);
    let wide = size_needle(32_768, &cal, admit_with(40_960, &checks));
    assert_eq!(checks.get(), 1);
    let NeedleSize::Send { sentences, admission: Some(admission), note: None } = wide else { panic!("{wide:?}") };
    assert_eq!(sentences, cal.sentences_for(32_768));
    assert!(admission.admitted && admission.prompt_tokens >= 32_768);
    // A context just above the target: the calibrated size (33,852 tokens)
    // is refused, so the filler steps down to a prompt from 32,768 to 32,968.
    let checks = std::cell::Cell::new(0);
    let tight = size_needle(32_768, &cal, admit_with(33_000, &checks));
    let NeedleSize::Send { admission: Some(admission), .. } = tight else { panic!("{tight:?}") };
    assert!(admission.admitted && (32_768..=32_968).contains(&admission.prompt_tokens), "{admission:?}");
    assert!(checks.get() <= NEEDLE_SIZE_CHECKS);
}

#[test]
fn needle_sizing_never_sends_an_unchecked_misfit_or_a_refused_prompt() {
    let cal = Calibration::from_probes([900, 3_972]).expect("calibration");
    // No admission route: the calibrated size is sent, and the card says so.
    let absent = size_needle(8_192, &cal, |_| Err("HTTP status 404".into()));
    let NeedleSize::Send { sentences, admission: None, note: Some(note) } = absent else { panic!("{absent:?}") };
    assert_eq!(sentences, cal.sentences_for(8_192));
    assert!(note.contains("HTTP status 404") && note.contains("sent unchecked"), "{note}");
    // A refusal inside the limits keeps the server's reason.
    let refused = size_needle(8_192, &cal, |sentences| {
        Ok(Admission { prompt_tokens: 132 + 12 * sentences as u64, context_limit: 65_536, input_limit: 65_535, admitted: false, reason: Some("prompt token outside vocabulary".into()) })
    });
    let NeedleSize::NotRun { reason, .. } = refused else { panic!("{refused:?}") };
    assert!(reason.contains("prompt token outside vocabulary"), "{reason}");
    // A count that never lands in range stops after the bounded checks.
    let checks = std::cell::Cell::new(0);
    let jumpy = size_needle(8_192, &cal, |sentences| {
        checks.set(checks.get() + 1);
        let tokens = if checks.get() % 2 == 1 { 9_000 } else { 8_000 } + sentences as u64 % 2;
        Ok(Admission { prompt_tokens: tokens, context_limit: 8_300, input_limit: 8_299, admitted: tokens + NEEDLE_ANSWER_TOKENS <= 8_300, reason: None })
    });
    let NeedleSize::NotRun { reason, .. } = jumpy else { panic!("{jumpy:?}") };
    assert_eq!(checks.get(), NEEDLE_SIZE_CHECKS);
    assert!(reason.contains(&format!("in {NEEDLE_SIZE_CHECKS} admission checks")), "{reason}");
}

#[test]
fn admission_replies_parse_strictly() {
    let reply = r#"{"object":"engine.admission","prompt_tokens":33800,"output_tokens":32,"context_limit":32768,"input_limit":32767,"output_limit":8192,"admitted":false,"reason":"Requested size does not fit."}"#;
    let admission = Admission::parse(reply).expect("admission");
    assert_eq!((admission.prompt_tokens, admission.context_limit, admission.input_limit, admission.admitted), (33_800, 32_768, 32_767, false));
    assert_eq!(admission.reason.as_deref(), Some("Requested size does not fit."));
    assert!(Admission::parse(r#"{"prompt_tokens":1.5,"context_limit":1,"input_limit":1,"admitted":true}"#).is_err());
    assert!(Admission::parse(r#"{"prompt_tokens":1,"context_limit":1,"input_limit":1}"#).is_err());
    assert!(Admission::parse("not json").is_err());
}

#[test]
fn a_not_run_needle_says_so_on_the_card_and_in_the_receipt() {
    let mut report = sample_report();
    report.needles[1].reply = None;
    report.needles[1].sizing = NeedleSizing {
        admission: Some(Admission { prompt_tokens: 33_800, context_limit: 32_768, input_limit: 32_767, admitted: false, reason: None }),
        not_run: Some("the server admits at most 32736 prompt tokens".into()),
        note: None,
    };
    assert_eq!(report.needle_score(), (1, 2));
    assert_eq!(report.grade(), Grade::C, "a not-run needle is never a pass");
    let card = report.card();
    assert!(card.contains("needle 32K  NOT RUN (refused before sending)"), "{card}");
    assert!(card.contains("NOT RUN: the server admits at most 32736 prompt tokens"), "{card}");
    let receipt = json::parse(&report.to_json().expect("receipt")).expect("receipt parses");
    let needle = &receipt.get("checks").and_then(Value::as_array).expect("checks")[0].get("items").and_then(Value::as_array).expect("items")[1];
    assert_eq!(needle.get("not_run").and_then(Value::as_str), Some("the server admits at most 32736 prompt tokens"));
    assert_eq!(needle.get("admission").and_then(|a| a.get("context_limit")), Some(&Value::Number("32768".into())));
}
