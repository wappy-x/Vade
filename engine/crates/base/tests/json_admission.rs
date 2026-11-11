//! Public behavior of the bounded JSON reader and the admission diagnostics.
use engine_base::{admission, json::{self, Limits, Value}};

fn parse(text: &str) -> Value { json::parse(text.as_bytes(), Limits::default()).unwrap() }

#[test]
fn values_parse_into_typed_accessors_with_exact_numbers() {
    let v = parse(" {\"shape\": [151936, 4096], \"dtype\": \"BF16\", \"offsets\": [0, 1244659712],\n\t\"ok\": true, \"none\": null, \"eps\": 1e-6, \"neg\": -0.5 } ");
    let o = v.object().unwrap();
    let shape = json::field(o, "shape").unwrap().array().unwrap().iter().map(|n| n.u64().unwrap()).collect::<Vec<_>>();
    assert_eq!(shape, [151936, 4096]);
    assert_eq!(json::field(o, "dtype").unwrap().string().unwrap(), "BF16");
    assert!(json::field(o, "ok").unwrap().boolean().unwrap());
    assert_eq!(json::field(o, "none").unwrap(), &Value::Null);
    assert_eq!(json::field(o, "eps").unwrap().f64().unwrap(), 1e-6);
    assert_eq!(json::field(o, "neg").unwrap().f64().unwrap(), -0.5);
    // Numbers keep their source spelling, so offsets never round through f64.
    assert_eq!(json::field(o, "eps").unwrap(), &Value::Number("1e-6".into()));
    assert_eq!(parse("9007199254740993").u64().unwrap(), 9007199254740993, "2^53 + 1 survives");
    assert!(json::field(o, "missing").unwrap_err().0.contains("missing field missing"));
    // Objects keep sorted keys, independent of input order.
    let keys = parse(r#"{"b":1,"a":2,"c":3}"#).object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(keys, ["a", "b", "c"]);
}

#[test]
fn accessors_refuse_the_wrong_type() {
    let v = parse(r#"[{}, [], "s", true, 1, null, -1, 1.5]"#);
    let a = v.array().unwrap();
    assert!(a[0].array().is_err() && a[1].object().is_err() && a[2].u64().is_err());
    assert!(a[3].string().is_err() && a[4].boolean().is_err() && a[5].f64().is_err());
    assert!(a[6].u64().is_err() && a[7].u64().is_err(), "u64 takes only nonnegative integers");
    assert_eq!(a[7].f64().unwrap(), 1.5);
    assert!(a[2].f64().is_err());
}

#[test]
fn string_escapes_decode_exactly() {
    assert_eq!(parse(r#""a\"b\\c\/d\b\f\n\r\t""#).string().unwrap(), "a\"b\\c/d\u{8}\u{c}\n\r\t");
    assert_eq!(parse(r#""\u00e9\u4E2D\ud83d\ude80""#).string().unwrap(), "é中🚀");
    assert_eq!(parse("\"raw é 中 🚀\"").string().unwrap(), "raw é 中 🚀");
    assert_eq!(parse(r#""\u0000""#).string().unwrap(), "\0");
}

#[test]
fn errors_name_the_byte_where_parsing_stopped() {
    for (text, message) in [
        ("[1 2]", "unexpected JSON character at byte 3"),
        ("{\"a\":1} x", "trailing JSON data at byte 8"),
        ("nul", "invalid JSON literal at byte 0"),
        ("[\"abc", "unterminated JSON string at byte 5"),
        ("{\"k\":1,\"k\":2}", "duplicate JSON object key"),
    ] {
        let error = json::parse(text.as_bytes(), Limits::default()).unwrap_err();
        assert!(error.to_string().contains(message), "{text}: {error}");
    }
}

#[test]
fn every_limit_is_enforced_independently() {
    // The top-level value is depth 0, so the default allows 33 nested arrays.
    let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
    assert!(json::parse(nested(33).as_bytes(), Limits::default()).is_ok());
    assert!(json::parse(nested(34).as_bytes(), Limits::default()).unwrap_err().0.contains("depth"));
    let tight = Limits { bytes: 64, depth: 4, nodes: 5, string_bytes: 4 };
    assert!(json::parse(b"[1,2,3,4]", tight).is_ok(), "five nodes");
    assert!(json::parse(b"[1,2,3,4,5]", tight).unwrap_err().0.contains("node limit"));
    assert!(json::parse(b"\"abcd\"", tight).is_ok());
    assert!(json::parse(b"\"abcde\"", tight).unwrap_err().0.contains("string limit"));
    // The string limit counts decoded UTF-8 bytes, also through escapes.
    assert!(json::parse(br#""\u00e9\u00e9\u00e9""#, tight).unwrap_err().0.contains("string limit"));
    assert!(json::parse(&[b' '; 65], tight).unwrap_err().0.contains("byte limit"));
}

#[test]
fn size_formatting_and_meminfo_reads_are_exact() {
    assert_eq!(admission::bytes(0), "0.00 GiB (0 bytes)");
    assert_eq!(admission::bytes(1 << 30), "1.00 GiB (1073741824 bytes)");
    assert_eq!(admission::bytes(66016133892), "61.48 GiB (66016133892 bytes)");
    let meminfo = "MemTotal:       127535024 kB\nMemFree:          3145728 kB\nMemAvailable:   115343360 kB\n";
    assert_eq!(admission::meminfo_bytes(meminfo, "MemTotal:").unwrap(), 127535024 * 1024);
    assert_eq!(admission::unified_available(meminfo, usize::MAX).unwrap(), 115343360 * 1024);
    assert_eq!(admission::unified_available(meminfo, 1 << 30).unwrap(), 1 << 30, "CUDA total caps the reading");
    assert!(admission::meminfo_bytes(meminfo, "Cached:").unwrap_err().contains("Cached:"));
}

/// One test owns the process-wide MemAvailable record, so no other test in
/// this binary races it.
#[test]
fn refusals_name_asked_available_gap_and_next_steps() {
    let required = 70u128 << 30;
    let available = 60usize << 30;
    let text = admission::refusal(required, available, "CONTEXT=32768", "the test source", admission::FLASH);
    let lines = text.lines().collect::<Vec<_>>();
    assert_eq!(lines[0], "Requested size does not fit.");
    assert_eq!(lines[1], "Asked: CONTEXT=32768; total 70.00 GiB (75161927680 bytes).");
    assert_eq!(lines[2], "Available: 60.00 GiB (64424509440 bytes) from the test source.");
    assert_eq!(lines[3], "Gap: 10.00 GiB (10737418240 bytes).");
    assert!(lines[4].starts_with("Next: 1. Lower CONTEXT / OUTPUT"), "{}", lines[4]);
    assert_eq!(lines.len(), 5, "no Memory line before any MemAvailable was read");
    // A refusal that fits reports a zero gap instead of wrapping.
    assert!(admission::refusal(1, 2, "x", "y", "z").contains("Gap: 0.00 GiB (0 bytes)."));
    admission::record_mem_available(3 << 30);
    let text = admission::allocation(required, available, "CONTEXT=32768");
    let last = text.lines().last().unwrap();
    assert!(last.starts_with("Memory: MemAvailable 3.00 GiB (3221225472 bytes); required 70.00 GiB"), "{last}");
    assert!(text.contains(admission::SERVE) && text.contains("configured allocation ceiling"));
    let tokens = admission::token_refusal(40000, 32768, "CONTEXT=40000", "the session ceiling", "CONTEXT");
    assert!(tokens.contains("Gap: 7232 tokens") && tokens.contains("cannot raise this token ceiling"), "{tokens}");
    assert!(!tokens.contains("Memory:"), "token refusals never carry the memory line");
}
