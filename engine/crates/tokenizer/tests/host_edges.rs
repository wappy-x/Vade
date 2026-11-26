//! Malformed local tokenizer files must fail before a chat starts.
use engine_tokenizer::{Limits, Tokenizer};
const FIXTURE: &str = include_str!("fixtures/host_edges/tokenizer.json");
fn refusal(from: &str, to: &str, reason: &str) {
    assert!(FIXTURE.contains(from), "fixture lost {from}");
    let text = FIXTURE.replacen(from, to, 1);
    let error = match Tokenizer::from_json(text.as_bytes(), Limits::default()) {
        Ok(_) => panic!("accepted {from} -> {to}"), Err(error) => error.to_string(),
    };
    assert!(error.contains(reason), "{from} -> {error}; expected {reason}");
}

#[test]
fn loader_rejects_missing_bytes_bad_ids_and_broken_merge_references() {
    let tokenizer = Tokenizer::from_json(FIXTURE.as_bytes(), Limits::default()).unwrap();
    assert_eq!(tokenizer.encode("ab").unwrap(), [256]);
    for (from, to, reason) in [
        (r#""!":33,"#, "", "missing byte token"),
        (r#""!":33"#, r#""!":4294967296"#, "token ID overflow"),
        (r#""!":33"#, r#""!":1000000"#, "oversized token ID"),
        (r#""merges":[["a","b"]]"#, r#""merges":[["zz","b"]]"#, "unknown left merge token"),
        (r#""merges":[["a","b"]]"#, r#""merges":[["a","c"]]"#, "unknown merged token"),
        (r#""merges":[["a","b"]]"#, r#""merges":[["a"]]"#, "invalid merge representation"),
        (r#""merges":[["a","b"]]"#, r#""merges":["ab"]"#, "invalid merge"),
        (r#""merges":[["a","b"]]"#, r#""merges":[null]"#, "invalid merge representation"),
    ] { refusal(from, to, reason); }
}

#[test]
fn loader_rejects_ambiguous_added_token_ids_text_and_flags() {
    for (from, to, reason) in [
        (r#""single_word":false"#, r#""single_word":true"#, "unsupported added token flags"),
        (r#""normalized":false"#, r#""normalized":true"#, "unsupported added token flags"),
        (r#""lstrip":false"#, r#""lstrip":true"#, "unsupported added token whitespace flags"),
        (r#""rstrip":false"#, r#""rstrip":true"#, "unsupported added token whitespace flags"),
        (r#""content":"<|endoftext|>""#, r#""content":"""#, "invalid added token text"),
        (r#""content":"<|im_start|>""#, r#""content":"<|endoftext|>""#, "invalid added token text"),
        (r#""id":151644"#, r#""id":151643"#, "duplicate or oversized added token ID"),
        (r#""id":151643"#, r#""id":97"#, "duplicate or oversized added token ID"),
        (r#""id":151643"#, r#""id":4294967296"#, "added token ID overflow"),
        (r#""id":151643"#, r#""id":1000000"#, "duplicate or oversized added token ID"),
    ] { refusal(from, to, reason); }
    refusal(r#""content":"<|endoftext|>""#, &format!(r#""content":"{}""#, "a".repeat(4097)), "invalid added token text");
}
