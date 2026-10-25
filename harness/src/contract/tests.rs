use super::*;

#[test]
fn response_speed_uses_decode_intervals_and_weighted_session_average() {
    let mut contract = Contract::default();
    for (first, last, tokens) in [(100, 300, 5), (200, 300, 2)] {
        contract.context.first_token_elapsed = Some(std::time::Duration::from_millis(first));
        contract.context.last_token_elapsed = Some(std::time::Duration::from_millis(last));
        let mut usage = UsageCollector::default();
        usage.observe_data(&format!(r#"{{"usage":{{"prompt_tokens":10,"completion_tokens":{tokens},"prompt_tokens_details":{{"cached_tokens":0}}}}}}"#)).unwrap();
        contract.completed(usage);
    }
    assert!(contract.context.turns[0].speed().contains("20.0 tok/s"));
    assert!(contract.context.turns[1].speed().contains("10.0 tok/s"));
    assert!(
        contract.context.turns[0]
            .speed()
            .contains("cached 0 tokens")
    );
    assert!(
        contract
            .context
            .session_speed
            .describe()
            .contains("TTFT 150.0 ms · output 16.7 tok/s")
    );
    assert!(
        contract
            .context
            .run_speed()
            .contains("Run speed: TTFT 150.0 ms · output 16.7 tok/s")
    );
    assert!(contract.context.run_speed().contains("2 model responses"));
    assert!(contract.context.reply_speed(std::time::Duration::ZERO, 32000, true)
        .contains("7 tokens written · write 17 tok/s · read ≈67 tok/s"));
    contract.reset_attempt();
    contract.completed(UsageCollector::default());
    assert!(
        contract
            .context
            .turns
            .back()
            .unwrap()
            .speed()
            .contains("TTFT unknown · output unknown")
    );
    assert!(
        contract
            .context
            .session_speed
            .describe()
            .contains("2 timed responses / 2 decode samples")
    );
    contract.begin_run();
    assert!(
        contract
            .context
            .run_speed()
            .contains("Run speed: TTFT unknown · output unknown")
    );
    assert!(contract.context.run_speed().contains("0 model responses"));
}

#[test]
fn usage_zero_missing_duplicates_and_conflicts_stay_distinct() {
    let event = |p, c| {
        format!(
            r#"{{"usage":{{"prompt_tokens":{p},"completion_tokens":3,"prompt_tokens_details":{{"cached_tokens":{c}}}}}}}"#
        )
    };
    let mut usage = UsageCollector::default();
    usage.observe(json::view(r#"{"usage":null}"#).unwrap());
    usage.observe(json::view(r#"{"choices":[]}"#).unwrap());
    assert_eq!(usage.finish(), (Usage::default(), false));
    let mut usage = UsageCollector::default();
    for _ in 0..3 {
        usage.observe(json::view(&event(100, 0)).unwrap());
    }
    assert_eq!(
        usage.finish(),
        (
            Usage {
                prompt_tokens: Some(100),
                completion_tokens: Some(3),
                cached_tokens: Some(0)
            },
            false
        )
    );
    for bad in [
        event(100, 101),
        event(100, 10),
        event(100, 20).replace("100", "-1"),
        event(100, 20).replace("100", "1e2"),
        r#"{"usage":true}"#.into(),
        r#"{"usage":{"prompt_tokens_details":42}}"#.into(),
    ] {
        let mut usage = UsageCollector::default();
        usage.observe(json::view(&event(100, 20)).unwrap());
        usage.observe(json::view(&bad).unwrap());
        // Even later valid telemetry cannot un-poison conflicting reports.
        usage.observe(json::view(&event(100, 20)).unwrap());
        assert_eq!(usage.finish(), (Usage::default(), true));
    }
}

#[test]
fn usage_prefilter_cannot_miss_escaped_keys_or_confuse_text_with_metadata() {
    let mut usage = UsageCollector::default();
    usage
        .observe_data(r#"{"choices":[{"delta":{"content":"the word usage"}}]}"#)
        .unwrap();
    assert_eq!(usage.finish(), (Usage::default(), false));
    let mut usage = UsageCollector::default();
    usage.observe_data(r#"{"\u0075sage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":7}}}"#).unwrap();
    let (usage, invalid) = usage.finish();
    assert!(!invalid);
    assert_eq!(usage.cached_tokens, Some(7));
}

#[test]
fn metadata_validation_is_bounded_and_unknown_features_stay_disabled() {
    let caps = include_str!("../../tests/fixtures/engine-capabilities-v1.json");
    let parsed = Capabilities::parse(caps).unwrap().unwrap();
    assert!(parsed.exact_admission && parsed.tools && parsed.prompt_cache_key);
    let mut context = Context { capabilities: Some(Capabilities { context_limit: 262_144, output_limit: 16_384, ..parsed.clone() }), ..Context::default() };
    assert_eq!(context.token_limit(None), 262_144); assert_eq!(context.token_limit(Some(131_072)), 131_072);
    context.capabilities = None; assert_eq!(context.token_limit(None), crate::context::DEFAULT_CONTEXT_TOKENS as u64);
    assert_eq!(
        parsed
            .identity
            .as_ref()
            .unwrap()
            .weights_fingerprint_kind
            .as_deref(),
        Some("safetensors_header_metadata")
    );
    assert_eq!(
        parsed.identity.unwrap().template_fingerprint,
        "sha256:mock-template"
    );
    for bad in [
        caps.replace("\"context_limit\": 4096", "\"context_limit\": 0"),
        caps.replace(
            "\"context_limit\": 4096",
            "\"context_limit\": 18446744073709551616",
        ),
        caps.replace("mock-template", "bad\\nidentity"),
        " ".repeat(MAX_METADATA + 1),
    ] {
        assert!(Capabilities::parse(&bad).is_err());
    }
    let absent = caps
        .replace("\"tools\": true,", "")
        .replace("\"thinking\": true,", "")
        .replace("\"prompt_cache_key\": \"scheduling_hint\",", "");
    let parsed = Capabilities::parse(&absent).unwrap().unwrap();
    assert!(!parsed.tools && !parsed.thinking && !parsed.prompt_cache_key);
}

#[test]
fn prefix_cache_decoration_preserves_caller_and_requires_capability() {
    let mut contract = Contract::default();
    assert_eq!(contract.decorate("{}").unwrap(), "{\"max_tokens\":8192}");
    contract.default_output = Some(65_536);
    assert_eq!(contract.decorate("{}").unwrap(), "{\"max_tokens\":16384}");
    contract.context.capabilities = Capabilities::parse(include_str!(
        "../../tests/fixtures/engine-capabilities-v1.json"
    ))
    .unwrap();
    contract
        .context
        .capabilities
        .as_mut()
        .unwrap()
        .prompt_cache_key = false;
    let explicit = r#"{"cache":false,"max_tokens":4,"enable_thinking":false}"#;
    assert_eq!(contract.decorate(explicit).unwrap(), explicit);
    contract.options.effort = Some("low");
    let effort = contract.decorate("{}").unwrap();
    assert!(effort.contains(r#""enable_thinking":true,"reasoning_effort":"low""#));
    contract.options.effort = None;
    contract.context.capabilities.as_mut().unwrap().prefix_cache = false;
    let decorated = contract.decorate("{}").unwrap();
    assert!(json::view(&decorated).unwrap().get("cache").is_none());
    contract.context.capabilities.as_mut().unwrap().prompt_cache_key = true; contract.affinity = Some("fixed".into());
    assert!(contract.decorate("{}").unwrap().contains(r#""cache":true,"prompt_cache_key":"fixed""#));
}

#[test]
fn cache_granularity_is_optional_positive_integer_metadata() {
    let caps = include_str!("../../tests/fixtures/engine-capabilities-v1.json");
    assert_eq!(
        Capabilities::parse(caps)
            .unwrap()
            .unwrap()
            .cache_granularity_tokens,
        None
    );
    for (value, expected) in [
        ("null", None),
        ("4096", Some(4096)),
        ("18446744073709551615", Some(u64::MAX)),
    ] {
        let body = caps.replacen('{', &format!("{{\"cache_granularity_tokens\":{value},"), 1);
        assert_eq!(
            Capabilities::parse(&body)
                .unwrap()
                .unwrap()
                .cache_granularity_tokens,
            expected
        );
    }
    for value in [
        "0",
        "-1",
        "1.5",
        "1e3",
        "true",
        "\"4096\"",
        "18446744073709551616",
    ] {
        let body = caps.replacen('{', &format!("{{\"cache_granularity_tokens\":{value},"), 1);
        assert!(Capabilities::parse(&body).is_err(), "{value}");
    }
}

#[test]
fn usage_retention_is_bounded_and_cumulative_counts_label_only_known_reports() {
    let mut contract = Contract::default();
    for i in 0..100 {
        let mut usage = UsageCollector::default();
        if i % 2 == 0 {
            usage.observe(json::view(r#"{"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":7}}}"#).unwrap());
        }
        contract.completed(usage);
    }
    assert_eq!(contract.context.turns.len(), 16);
    assert_eq!(contract.context.completed_responses, 100);
    assert_eq!(contract.context.responses_with_cached_usage, 50);
    assert_eq!(contract.context.reported_cached_tokens, 350);
    assert!(contract.context.describe().contains("50/100"));
    let mut restored = UsageCollector::default(); restored.observe_data(r#"{"usage":{"prompt_tokens":100,"completion_tokens":2},"vade":{"prefill_cache_restored_tokens":90,"prefill_ms":2}}"#).unwrap();
    assert_eq!(restored.finish().0.cached_tokens, Some(90));
    contract.begin_run();
    assert!(contract.context.turns.is_empty());
    assert_eq!(contract.context.reported_cached_tokens, 350);
}
