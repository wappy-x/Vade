//! Fixture tests for the quality card: golden answers for every fixed case,
//! failing answers, SSE parsing, draft receipt parsing and card rendering.
//! No GPU and no server are needed.
use super::*;

fn case(id: &str) -> &'static Case {
    CASES.iter().find(|case| case.id == id).expect("fixed case exists")
}

fn golden(id: &str) -> &'static str {
    match id {
        "code-clamp" => "```rust\npub fn clamp_i32(value: i32, low: i32, high: i32) -> i32 {\n    if low > high { return low; }\n    value.max(low).min(high)\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test] fn inside() { assert_eq!(clamp_i32(4, 0, 9), 4); }\n    #[test] fn above() { assert_eq!(clamp_i32(40, 0, 9), 9); }\n}\n```",
        "code-dedupe" => "```rust\npub fn dedupe_sorted(values: &[i32]) -> Vec<i32> {\n    let mut out: Vec<i32> = Vec::new();\n    for &value in values { if out.last() != Some(&value) { out.push(value); } }\n    out\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test] fn runs() { assert_eq!(dedupe_sorted(&[1, 1, 2]), vec![1, 2]); }\n    #[test] fn empty() { assert!(dedupe_sorted(&[]).is_empty()); }\n}\n```",
        "code-parse-list" => "```rust\npub fn parse_u32_list(text: &str) -> Result<Vec<u32>, String> {\n    if text.trim().is_empty() { return Ok(Vec::new()); }\n    text.split(',').map(|part| part.trim().parse::<u32>().map_err(|e| e.to_string())).collect()\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test] fn basic() { assert_eq!(parse_u32_list(\"1,2\").unwrap(), vec![1, 2]); }\n    #[test] fn bad() { assert!(parse_u32_list(\"x\").is_err()); }\n}\n```",
        "code-word-count" => "```rust\npub fn count_word(text: &str, word: &str) -> usize {\n    if word.is_empty() { return 0; }\n    text.split(|c: char| !c.is_ascii_alphanumeric()).filter(|part| part.eq_ignore_ascii_case(word)).count()\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test] fn basic() { assert_eq!(count_word(\"a cat\", \"cat\"), 1); }\n    #[test] fn case() { assert_eq!(count_word(\"Cat\", \"cat\"), 1); }\n}\n```",
        "json-order" => "{\"order_id\":\"A-1042\",\"items\":3,\"total_cents\":4599,\"priority\":\"high\",\"paid\":false}",
        "json-person" => "{\"name\":\"Marta Klein\",\"birth_year\":1988,\"city\":\"Porto\",\"hobbies\":[\"rowing\",\"chess\",\"baking\"]}",
        "json-classify" => "{\"tickets\":[{\"id\":1,\"label\":\"billing\"},{\"id\":2,\"label\":\"bug\"},{\"id\":3,\"label\":\"question\"},{\"id\":4,\"label\":\"bug\"}]}",
        "json-table" => "{\"rows\":4,\"sum\":514,\"max\":210,\"over_100\":[\"Ana\",\"Cy\"]}",
        "prose-open" => "A rain barrel stores roof water for dry days. A screened inlet keeps leaves out, and an overflow pipe sends extra water to a drain. Gardeners draw from the tap at the base to water beds, pots and young trees through a dry week.",
        "prose-list" => "- spring: sow peas and beans\n- summer: water beds at dawn\n- autumn: rake leaves into compost\n- winter: clean and oil tools",
        "prose-rewrite" => "During the 2049 maintenance window, the technician swapped the worn valve on line 17.",
        "prose-form" => "Location: north shelf\nCount: 26\nColor: blue",
        other => panic!("no golden answer for {other}"),
    }
}

#[test]
fn golden_answers_pass_every_text_check() {
    for case in CASES.iter().filter(|case| case.class != Class::Code) {
        let checks = score_rules(case, golden(case.id));
        let failed: Vec<_> = checks
            .iter()
            .filter(|check| check.run && !check.passed)
            .map(|check| format!("{} ({})", check.name, check.detail))
            .collect();
        assert!(failed.is_empty(), "{} golden failed: {failed:?}", case.id);
    }
}

fn require_rustc() {
    let mut probe = Command::new("rustc");
    probe.arg("--version");
    let probed = run_bounded(probe, Duration::from_secs(10));
    assert!(
        matches!(probed, Ok((true, _))),
        "rustc on PATH is required for the quality-card code tests: {probed:?}"
    );
}

#[test]
fn golden_code_passes_hidden_unit_tests() {
    require_rustc();
    for case in CASES.iter().filter(|case| case.class == Class::Code) {
        let checks = score_code(case, golden(case.id), "rustc", true, Duration::from_secs(60));
        let failed: Vec<_> = checks
            .iter()
            .filter(|check| !check.run || !check.passed)
            .map(|check| format!("{} run={} ({})", check.name, check.run, check.detail))
            .collect();
        assert!(failed.is_empty(), "{} golden failed: {failed:?}", case.id);
    }
}

#[test]
fn wrong_answers_fail_the_named_checks() {
    let json = score_rules(
        case("json-order"),
        "{\"order_id\":\"A-1042\",\"items\":4,\"total_cents\":4599,\"priority\":\"high\",\"paid\":true}",
    );
    assert!(json.iter().any(|c| c.name == "json field items exact" && !c.passed));
    assert!(json.iter().any(|c| c.name == "json field paid exact" && !c.passed));
    assert!(json.iter().any(|c| c.name == "json field order_id exact" && c.passed));

    let prose = score_rules(
        case("prose-open"),
        "A barrel holds water. It is a very simple tool for any small garden space here.",
    );
    assert!(prose.iter().any(|c| c.name.starts_with("starts with") && !c.passed));
    assert!(prose.iter().any(|c| c.name == "exactly 3 sentences" && !c.passed));
    assert!(prose.iter().any(|c| c.name == "contains \"overflow pipe\"" && !c.passed));

    let fenced = score_rules(case("json-table"), "```json\n{\"rows\":4,\"sum\":514,\"max\":210,\"over_100\":[\"Ana\",\"Cy\"]}\n```");
    assert!(
        fenced.iter().any(|c| c.name == "answer is one JSON document" && !c.passed),
        "a fence breaks the JSON-only form: {fenced:?}"
    );

    let extra_key = score_rules(
        case("json-table"),
        "{\"rows\":4,\"sum\":514,\"max\":210,\"over_100\":[\"Ana\",\"Cy\"],\"extra\":1}",
    );
    assert!(extra_key.iter().any(|c| c.name == "exact json keys" && !c.passed), "{extra_key:?}");

    let tickets = score_rules(
        case("json-classify"),
        "{\"tickets\":[{\"id\":1,\"label\":\"billing\"},{\"label\":\"bug\"},{\"id\":999,\"label\":\"question\"},{\"id\":\"wrong\",\"label\":\"bug\"}]}",
    );
    assert!(tickets.iter().any(|c| c.name == "json tickets[2].id exact" && !c.passed), "{tickets:?}");
    assert!(tickets.iter().any(|c| c.name == "json tickets[3].id exact" && !c.passed), "{tickets:?}");
    assert!(
        tickets.iter().any(|c| c.name == "exact json tickets item keys" && !c.passed),
        "{tickets:?}"
    );

    let list = score_rules(case("prose-list"), "- spring summer autumn winter\n- a\n- b\n- c");
    assert!(list.iter().any(|c| c.name == "line 2 contains \"summer\"" && !c.passed), "{list:?}");

    let form = score_rules(case("prose-form"), "Location: north shelf WRONG\nCount: 2600\nColor: blue");
    assert!(
        form.iter().any(|c| c.name == "answer equals the exact required text" && !c.passed),
        "{form:?}"
    );

    let broken = score_rules(case("json-table"), "not json at all");
    assert!(broken.iter().any(|c| c.name == "answer is one JSON document" && !c.passed));

    let code = score_code(case("code-clamp"), "Here is the function you asked for.", "rustc", false, Duration::from_secs(5));
    assert!(code.iter().any(|c| c.name == "one fenced code block" && !c.passed));
    assert!(code.iter().any(|c| c.name == "exact signature" && !c.passed));
}

#[test]
fn code_checks_stay_out_of_the_denominator_when_not_run() {
    let checks = score_code(case("code-clamp"), golden("code-clamp"), "rustc", false, Duration::from_secs(5));
    let hidden: Vec<_> = checks.iter().filter(|c| c.name.contains("hidden test")).collect();
    assert_eq!(hidden.len(), 5);
    assert!(hidden.iter().all(|c| !c.run));
    assert!(checks.iter().filter(|c| c.run).all(|c| c.passed));
}

#[test]
fn sse_events_fill_text_ids_usage_and_draft_receipt() {
    let mut reply = Reply::default();
    let events = [
        r#"{"id":"completion-7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}"#,
        r#"{"id":"completion-7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null,"token_id":11}]}"#,
        r#"{"id":"completion-7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null,"token_id":12}]}"#,
        r#"{"id":"completion-7","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22,"prompt_tokens_details":{"cached_tokens":0},"vade":{"prefill_ms":30.5,"decode_ms":40.25,"timing":"admitted_wall"}},"vade":{"sampling":{"temperature":0},"draft":"strict","mtp_depth":3,"thinking":false,"draft_used":true,"model_draft_used":true,"prefill_ms":31.0}}"#,
    ];
    for (index, event) in events.iter().enumerate() {
        assert!(!apply_sse_data(&mut reply, event, index as f64 * 0.1).unwrap());
    }
    assert!(apply_sse_data(&mut reply, "[DONE]", 0.4).unwrap());
    assert_eq!(reply.text, "Hello");
    assert_eq!(reply.token_ids, [11, 12]);
    assert_eq!(reply.completion_id, Some(7));
    assert_eq!(reply.finish_reason.as_deref(), Some("stop"));
    assert_eq!(reply.completion_tokens, Some(2));
    assert_eq!(reply.draft.as_deref(), Some("strict"));
    assert!(reply.saw_done);
    assert!(!reply.invalid_token_id);
    assert_eq!(reply.served_model.as_deref(), Some("m"));
    assert!(reply.vade_receipt.as_deref().is_some_and(|v| v.contains("strict")));
    assert_eq!(reply.draft_used, Some(true));
    assert_eq!(reply.decode_ms, Some(40.25));
    assert_eq!(reply.ttft_s, Some(0.1));
}

#[test]
fn draft_receipt_lines_parse_and_price_the_verify() {
    let log = "noise\ndraft id=7 cycles=10 rows=40 accepted=25 proposed=30 max_rows=4 target_ms=120.500 draft_ms=20.250 guards=0/0 host_bytes=10 bank_host_bytes=0\n\
2026-10-27T06:00:00 draft id=8 cycles=5 rows=20 accepted=5 proposed=15 max_rows=4 target_ms=59.500 draft_ms=9.750 guards=0/0 host_bytes=10 bank_host_bytes=0\nbroken draft id=x\n";
    let receipts = parse_draft_receipts(log);
    assert_eq!(receipts.len(), 2);
    assert_eq!(receipts[0].verify_ms_per_step(), Some(12.05));
    assert_eq!(receipts[0].accepted_per_step(), Some(2.5));
    let mut cost = VerifyCost::default();
    for receipt in &receipts {
        cost.add(*receipt);
    }
    assert_eq!(cost.cycles, 15);
    assert_eq!(cost.verify_ms_per_step(), Some(12.0));
    assert_eq!(cost.accepted_per_step(), Some(2.0));
}

#[test]
fn endpoints_normalize_and_refuse_https() {
    assert_eq!(parse_endpoint("http://127.0.0.1:8080").unwrap().path, "/v1/chat/completions");
    assert_eq!(parse_endpoint("http://127.0.0.1:8080/v1").unwrap().path, "/v1/chat/completions");
    assert_eq!(
        parse_endpoint("http://127.0.0.1:8080/v1/chat/completions").unwrap().path,
        "/v1/chat/completions"
    );
    assert!(parse_endpoint("https://127.0.0.1:8080").is_err());
    let config = Config::parse([
        "--endpoint".into(),
        "https://127.0.0.1:8080".into(),
        "--model".into(),
        "m".into(),
        "--json".into(),
        "out.json".into(),
    ]);
    assert!(config.is_err());
}

#[test]
fn request_bodies_pin_sampling_and_draft_arms() {
    let config = Config {
        endpoint: "http://127.0.0.1:8080".into(),
        model: "served-model".into(),
        json_path: "out.json".into(),
        ..Config::default()
    };
    let served = request_body(&config, case("json-order"), None).unwrap();
    let served = json::parse(&served).unwrap();
    assert!(served.get("draft").is_none());
    assert_eq!(served.get("cache"), Some(&Value::Bool(false)));
    assert_eq!(
        served.get("chat_template_kwargs").and_then(|v| v.get("enable_thinking")),
        Some(&Value::Bool(false))
    );
    let strict = request_body(&config, case("json-order"), Some("strict")).unwrap();
    assert_eq!(
        json::parse(&strict).unwrap().get("draft").and_then(Value::as_str),
        Some("strict")
    );
    let code = request_body(&config, case("code-clamp"), None).unwrap();
    assert!(code.contains("fenced rust code block"), "code tail appended");
    // GLM has no thinking-off form and its server refuses the field: the card asks low effort.
    let glm = Config { model: "GLM-5.3-Flash-FP4-local".into(), ..config };
    let glm = json::parse(&request_body(&glm, case("json-order"), None).unwrap()).unwrap();
    assert!(glm.get("chat_template_kwargs").is_none() && glm.get("enable_thinking").is_none());
    assert_eq!(glm.get("reasoning_effort").and_then(Value::as_str), Some("low"));
}

#[test]
fn engine_card_bodies_ask_a_glm_model_for_low_effort() {
    let config = |model: &str| Config { endpoint: "http://127.0.0.1:8080".into(), model: model.into(), json_path: "out.json".into(), ..Config::default() };
    let body = |model: &str| json::parse(&engine::engine_body(&config(model), Some("S"), "U", 8, Some(false), Vec::new()).unwrap()).unwrap();
    let qwen = body("qwen3.8-flash-next-fp8");
    assert_eq!(qwen.get("chat_template_kwargs").and_then(|v| v.get("enable_thinking")), Some(&Value::Bool(false)));
    assert!(qwen.get("reasoning_effort").is_none());
    let glm = body("glm-5.3-flash-fp4-local");
    assert!(glm.get("chat_template_kwargs").is_none() && glm.get("enable_thinking").is_none());
    assert_eq!(glm.get("reasoning_effort").and_then(Value::as_str), Some("low"));
}

#[test]
fn cards_render_scores_ids_and_verify_cost() {
    let config = Config {
        endpoint: "http://127.0.0.1:8080".into(),
        model: "served-model".into(),
        json_path: "out.json".into(),
        ..Config::default()
    };
    let quality = CASES
        .iter()
        .map(|case| {
            let mut reply = Reply {
                text: golden(case.id).into(),
                token_ids: vec![1, 2, 3],
                completion_tokens: Some(10),
                draft_used: Some(true),
                ttft_s: Some(0.1),
                total_s: 0.3,
                ..Reply::default()
            };
            reply.completion_id = Some(7);
            let checks = match case.scoring {
                Scoring::Code { .. } => {
                    score_code(case, &reply.text.clone(), "rustc", false, Duration::from_secs(5))
                }
                _ => score_rules(case, &reply.text.clone()),
            };
            QualityCase { case, reply, checks, error: None }
        })
        .collect::<Vec<_>>();
    let exactness = CASES
        .iter()
        .map(|case| {
            let arm = |draft: &str| Reply {
                text: "same".into(),
                token_ids: vec![1, 2],
                saw_done: true,
                completion_id: Some(7),
                finish_reason: Some("stop".into()),
                completion_tokens: Some(3),
                draft: Some(draft.into()),
                ttft_s: Some(0.1),
                total_s: 0.3,
                ..Reply::default()
            };
            ExactCase { case, strict: Some(arm("strict")), plain: Some(arm("plain")), error: None }
        })
        .collect::<Vec<_>>();
    let mut verify = VerifyCost::default();
    verify.add(DraftReceipt {
        id: 7,
        cycles: 10,
        rows: 40,
        accepted: 25,
        proposed: 30,
        target_ms: 120.5,
        draft_ms: 20.25,
    });
    let report = Report {
        config,
        quality,
        exactness,
        verify,
        verify_receipts: Vec::new(),
        verify_basis: "draft receipts matched by completion id".into(),
        elapsed_s: 12.5,
    };
    let cards = report.cards();
    assert!(cards.contains("ANSWER QUALITY RECEIPT"), "{cards}");
    assert!(cards.contains("IDS IDENTICAL 12/12"), "{cards}");
    assert!(cards.contains("draft modes: strict arm strict, plain arm plain"), "{cards}");
    assert!(cards.contains("token evidence complete:"), "{cards}");
    assert!(cards.contains("Verify ms per step 12.050"), "{cards}");
    assert!(cards.contains("accepted per step 2.500"), "{cards}");
    for line in cards.lines().filter(|line| line.starts_with('│') || line.starts_with('├')) {
        assert_eq!(line.chars().count(), 74, "card line width: {line:?}");
    }
    let receipt = report.to_json().unwrap();
    let parsed = json::parse(&receipt).unwrap();
    assert_eq!(
        parsed.get("schema").and_then(Value::as_str),
        Some("vade.quality-card.v1")
    );
    // The code checks here were built with code_run requested but not run,
    // so the derived receipt flag must say they did not run.
    assert_eq!(parsed.get("code_checks_run"), Some(&Value::Bool(false)));
    assert!(!report.code_checks_ran());
    assert!(report.successful());
}

fn evidence_arm(ids: Vec<u32>, draft: Option<&str>) -> Reply {
    Reply {
        text: "same".into(),
        token_ids: ids,
        saw_done: true,
        finish_reason: Some("stop".into()),
        completion_tokens: Some(3),
        draft: draft.map(str::to_owned),
        ttft_s: Some(0.1),
        total_s: 0.3,
        ..Reply::default()
    }
}

fn exact(strict: Reply, plain: Reply) -> ExactCase {
    ExactCase { case: case("json-order"), strict: Some(strict), plain: Some(plain), error: None }
}

fn failure(case: &ExactCase) -> String {
    case.failure_reason().expect("case must fail")
}

#[test]
fn exactness_fails_without_token_ids() {
    let bare = exact(
        evidence_arm(Vec::new(), Some("strict")),
        evidence_arm(Vec::new(), Some("plain")),
    );
    assert_eq!(bare.ids_identical(), Some(false));
    assert_eq!(bare.failure_reason().as_deref(), Some("no token ids"));
    assert!(!bare.passed());
}

#[test]
fn exactness_needs_complete_token_evidence_on_both_arms() {
    // Passing case: a stop reply may stream one fewer id than it counts
    // (the end-of-sequence token is counted, not streamed).
    let good = exact(
        evidence_arm(vec![1, 2], Some("strict")),
        evidence_arm(vec![1, 2], Some("plain")),
    );
    assert!(good.passed(), "{:?}", good.failure_reason());
    // Passing case with no allowance used: every counted id streamed.
    let mut strict = evidence_arm(vec![1, 2, 3], Some("strict"));
    strict.finish_reason = Some("length".into());
    let mut plain = evidence_arm(vec![1, 2, 3], Some("plain"));
    plain.finish_reason = Some("length".into());
    let full = exact(strict, plain);
    assert!(full.passed(), "{:?}", full.failure_reason());

    // Short list on a stop finish: one allowance does not cover two missing.
    let short = exact(evidence_arm(vec![1], Some("strict")), evidence_arm(vec![1], Some("plain")));
    assert!(failure(&short).contains("token id count mismatch"), "{}", failure(&short));
    assert!(!short.passed());

    // Short list on a length finish: a length stop streams its last token,
    // so one missing id is already incomplete.
    let mut strict = evidence_arm(vec![11], Some("strict"));
    strict.finish_reason = Some("length".into());
    strict.completion_tokens = Some(2);
    let mut plain = evidence_arm(vec![11], Some("plain"));
    plain.finish_reason = Some("length".into());
    plain.completion_tokens = Some(2);
    let length_short = exact(strict, plain);
    assert!(
        failure(&length_short).contains("token id count mismatch"),
        "{}",
        failure(&length_short)
    );
    assert!(!length_short.passed());

    // Missing completion count.
    let mut strict = evidence_arm(vec![1, 2], Some("strict"));
    strict.completion_tokens = None;
    let no_count = exact(strict, evidence_arm(vec![1, 2], Some("plain")));
    assert_eq!(failure(&no_count), "missing completion token count");
    assert!(!no_count.passed());

    // Missing finish reason.
    let mut strict = evidence_arm(vec![1, 2], Some("strict"));
    strict.finish_reason = None;
    let no_finish = exact(strict, evidence_arm(vec![1, 2], Some("plain")));
    assert_eq!(failure(&no_finish), "missing finish reason");
    assert!(!no_finish.passed());

    // No terminal event, even with ids, counts and modes in place.
    let mut strict = evidence_arm(vec![1, 2], Some("strict"));
    strict.saw_done = false;
    let no_terminal = exact(strict, evidence_arm(vec![1, 2], Some("plain")));
    assert_eq!(failure(&no_terminal), "no terminal event");
    assert!(!no_terminal.passed());

    // An invalid streamed id poisons the evidence even if lists compare equal.
    let mut strict = evidence_arm(vec![1, 2], Some("strict"));
    strict.invalid_token_id = true;
    let invalid = exact(strict, evidence_arm(vec![1, 2], Some("plain")));
    assert_eq!(failure(&invalid), "invalid token id");
    assert!(!invalid.passed());

    // Both arms complete alone but they disagree on finish and count.
    let mut plain = evidence_arm(vec![1, 2, 3], Some("plain"));
    plain.finish_reason = Some("length".into());
    let finish_split = exact(evidence_arm(vec![1, 2], Some("strict")), plain);
    assert!(
        failure(&finish_split).contains("finish reason mismatch"),
        "{}",
        failure(&finish_split)
    );
    assert!(!finish_split.passed());
    let mut plain = evidence_arm(vec![1, 2, 3], Some("plain"));
    plain.completion_tokens = Some(4);
    let count_split = exact(evidence_arm(vec![1, 2], Some("strict")), plain);
    assert!(
        failure(&count_split).contains("completion token count mismatch"),
        "{}",
        failure(&count_split)
    );
    assert!(!count_split.passed());
}

#[test]
fn invalid_streamed_token_ids_fail_the_exactness_cell() {
    fn streamed(token: &str, draft: &str) -> Reply {
        let mut reply = Reply::default();
        let chunk = format!(
            r#"{{"id":"completion-7","model":"m","choices":[{{"index":0,"delta":{{"content":"x"}},"finish_reason":null,"token_id":{token}}}]}}"#
        );
        let last = format!(
            r#"{{"id":"completion-7","model":"m","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":1,"completion_tokens":2}},"vade":{{"draft":"{draft}"}}}}"#
        );
        apply_sse_data(&mut reply, &chunk, 0.0).unwrap();
        apply_sse_data(&mut reply, &last, 0.1).unwrap();
        apply_sse_data(&mut reply, "[DONE]", 0.2).unwrap();
        reply
    }
    for token in ["1.1", "1.9", "-1", "4294967296", "4294967297", "\"11\""] {
        let cell = exact(streamed(token, "strict"), streamed(token, "plain"));
        assert_eq!(failure(&cell), "invalid token id", "token {token}");
        assert!(!cell.passed(), "token {token} must not pass");
    }
    // Fractional usage counts are never cast either.
    let mut reply = Reply::default();
    let bad_usage = r#"{"id":"completion-7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2.9}}"#;
    apply_sse_data(&mut reply, bad_usage, 0.0).unwrap();
    assert_eq!(reply.completion_tokens, None);
}

#[test]
fn exactness_fails_on_wrong_draft_modes() {
    let downgraded = ExactCase {
        case: case("json-order"),
        strict: Some(evidence_arm(vec![1, 2], Some("plain"))),
        plain: Some(evidence_arm(vec![1, 2], Some("plain"))),
        error: None,
    };
    assert_eq!(downgraded.ids_identical(), Some(true));
    assert_eq!(downgraded.draft_modes_ok(), Some(false));
    assert_eq!(
        downgraded.failure_reason().as_deref(),
        Some("draft mode mismatch: strict arm reported plain (want strict), plain arm reported plain (want plain)")
    );
    assert!(!downgraded.passed());
    let missing = ExactCase {
        case: case("json-order"),
        strict: Some(evidence_arm(vec![1, 2], None)),
        plain: Some(evidence_arm(vec![1, 2], Some("plain"))),
        error: None,
    };
    assert!(
        missing.failure_reason().is_some_and(|reason| reason.contains("reported missing")),
        "{:?}",
        missing.failure_reason()
    );
    assert!(!missing.passed());
    let good = ExactCase {
        case: case("json-order"),
        strict: Some(evidence_arm(vec![1, 2], Some("strict"))),
        plain: Some(evidence_arm(vec![1, 2], Some("plain"))),
        error: None,
    };
    assert!(good.passed());
}

#[test]
fn fractional_token_ids_are_never_converted_to_integers() {
    let mut reply = Reply::default();
    let event = r#"{"id":"completion-7","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null,"token_id":1.9}]}"#;
    apply_sse_data(&mut reply, event, 0.0).unwrap();
    assert!(reply.token_ids.is_empty(), "1.9 must not become 1: {:?}", reply.token_ids);
}

#[test]
fn median_averages_the_two_middle_values() {
    assert_eq!(median(vec![40.0, 10.0, 30.0, 20.0]), Some(25.0));
    assert_eq!(median(vec![30.0, 10.0, 20.0]), Some(20.0));
    assert_eq!(median(Vec::new()), None);
}

#[test]
fn bounded_run_ends_when_a_grandchild_holds_the_pipe() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid_path = std::env::temp_dir().join(format!("vade-quality-grandchild-{stamp}.pid"));
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(format!("sleep 30 & echo $! > {}; echo ready", pid_path.display()));
    let started = Instant::now();
    let result = run_bounded(command, Duration::from_millis(500));
    assert!(
        result.as_ref().is_err_and(|error| error.kind() == io::ErrorKind::TimedOut),
        "pipe held by grandchild must hit the deadline (sh on PATH is required): {result:?}"
    );
    assert!(started.elapsed() < Duration::from_secs(10), "must not wait for the grandchild");
    let mut pid = String::new();
    for _ in 0..100 {
        if let Ok(text) = std::fs::read_to_string(&pid_path) {
            pid = text.trim().to_owned();
            if !pid.is_empty() {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = std::fs::remove_file(&pid_path);
    assert!(!pid.is_empty(), "grandchild wrote its pid");
    let mut alive = true;
    for _ in 0..100 {
        let probe = Command::new("/bin/kill").arg("-0").arg(&pid).output();
        alive = matches!(probe, Ok(output) if output.status.success());
        if !alive {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!alive, "grandchild {pid} must stop with the timed-out check");
}

#[test]
fn failed_own_unit_tests_do_not_get_full_code_credit() {
    require_rustc();
    let answer = "```rust\npub fn clamp_i32(value: i32, low: i32, high: i32) -> i32 {\n    if low > high { return low; }\n    value.max(low).min(high)\n}\n#[cfg(test)]\nmod tests {\n    use super::*;\n    #[test] fn inside() { assert_eq!(clamp_i32(4, 0, 9), 999); }\n    #[test] fn above() { assert_eq!(clamp_i32(40, 0, 9), 9); }\n}\n```";
    let checks = score_code(case("code-clamp"), answer, "rustc", true, Duration::from_secs(60));
    assert!(
        checks.iter().filter(|c| c.name.contains("hidden test")).all(|c| c.run && c.passed),
        "hidden tests still pass: {checks:?}"
    );
    assert!(
        checks.iter().any(|c| c.name == "own unit tests pass" && c.run && !c.passed),
        "the failed own test must fail its check: {checks:?}"
    );
}

#[test]
fn verify_cost_needs_one_unique_receipt_per_completion_id() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let log_path = std::env::temp_dir().join(format!("vade-quality-verify-{stamp}.log"));
    let line = "draft id=7 cycles=10 rows=40 accepted=25 proposed=30 max_rows=4 target_ms=120.500 draft_ms=20.250 guards=0/0 host_bytes=1 bank_host_bytes=0\n";
    let mut strict = evidence_arm(vec![1, 2], Some("strict"));
    strict.completion_id = Some(7);
    let exactness = vec![exact(strict, evidence_arm(vec![1, 2], Some("plain")))];
    let config = Config {
        endpoint: "http://127.0.0.1:8080".into(),
        model: "served-model".into(),
        json_path: "out.json".into(),
        server_log: Some(log_path.clone()),
        ..Config::default()
    };
    std::fs::write(&log_path, line).unwrap();
    let (cost, matched, basis) = verify_cost(&config, &exactness, None);
    assert_eq!(cost.cycles, 10);
    assert_eq!(matched.len(), 1);
    assert_eq!(basis, "draft receipts matched by completion id");
    // A reused id in the log (two server runs) is ambiguous: no cost.
    std::fs::write(&log_path, format!("{line}{line}")).unwrap();
    let (cost, matched, basis) = verify_cost(&config, &exactness, None);
    assert_eq!(cost.receipts, 0);
    assert!(matched.is_empty());
    assert!(basis.contains("verify cost unavailable"), "{basis}");
    // An unrelated log is never summed as this run's cost.
    std::fs::write(&log_path, line.replace("id=7", "id=9")).unwrap();
    let (cost, matched, basis) = verify_cost(&config, &exactness, None);
    assert_eq!(cost.receipts, 0);
    assert!(matched.is_empty());
    assert!(basis.contains("verify cost unavailable"), "{basis}");
    // A missing log records the error instead of failing the run.
    let _ = std::fs::remove_file(&log_path);
    let (cost, _, basis) = verify_cost(&config, &exactness, None);
    assert_eq!(cost.receipts, 0);
    assert!(basis.contains("server log unreadable"), "{basis}");
}
