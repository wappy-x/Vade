//! Public loader and codec behaviour through a real-format tokenizer.json:
//! a byte-level BPE with the hybrid split pattern, four merges and the three
//! hybrid control tokens. No checkpoint is needed.
use engine_tokenizer::{Limits, Tokenizer};

const HYBRID_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CONTROLS: [(&str, u32); 3] = [("<|endoftext|>", 248044), ("<|im_start|>", 248045), ("<|im_end|>", 248046)];
// Merge ranks are list order: "he" and "ll" first, then "hell", then " w".
const MERGES: [(&str, &str); 4] = [("h", "e"), ("l", "l"), ("he", "ll"), ("Ġ", "w")];

/// The GPT-2 byte alphabet that byte-level BPE vocabularies are spelled in.
fn alphabet() -> Vec<char> {
    let mut extra = 256;
    (0..256u32).map(|b| {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) { char::from_u32(b).unwrap() }
        else { extra += 1; char::from_u32(extra - 1).unwrap() }
    }).collect()
}
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn bytelevel() -> &'static str { r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}"# }
/// Refusal cases below rewrite one field of this text at a time.
fn json(merges: &[(&str, &str)], controls: &[(&str, u32)]) -> String {
    let mut vocab = alphabet().iter().enumerate().map(|(id, c)| format!("{}:{id}", quote(&c.to_string()))).collect::<Vec<_>>();
    let mut next = 256;
    for (a, b) in merges {
        let joined = format!("{a}{b}");
        if !vocab.iter().any(|entry| entry.starts_with(&format!("{}:", quote(&joined)))) { vocab.push(format!("{}:{next}", quote(&joined))); next += 1; }
    }
    let merges = merges.iter().map(|(a, b)| format!("[{},{}]", quote(a), quote(b))).collect::<Vec<_>>().join(",");
    let added = controls.iter().map(|(text, id)| format!(r#"{{"id":{id},"content":{},"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#, quote(text)))
        .collect::<Vec<_>>().join(",");
    format!(r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{added}],"normalizer":{{"type":"NFC"}},"pre_tokenizer":{{"type":"Sequence","pretokenizers":[{{"type":"Split","pattern":{{"Regex":{}}},"behavior":"Isolated","invert":false}},{}]}},"post_processor":{},"decoder":{},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{{{}}},"merges":[{merges}]}}}}"#,
        quote(HYBRID_PATTERN), bytelevel(), bytelevel(), bytelevel(), vocab.join(","))
}
fn fixture() -> Tokenizer { Tokenizer::from_json(json(&MERGES, &CONTROLS).as_bytes(), Limits::default()).unwrap() }
fn refusal(text: &str) -> String {
    match Tokenizer::from_json(text.as_bytes(), Limits::default()) { Ok(_) => panic!("accepted a malformed tokenizer"), Err(error) => error.0 }
}

#[test]
fn loads_a_hybrid_byte_level_bpe_and_reports_its_controls() {
    let t = fixture();
    assert!(t.is_hybrid_text());
    assert!(!t.is_gemma4());
    assert_eq!(t.eos_token_id(), 248046);
    for (text, id) in CONTROLS {
        assert!(t.contains_id(id) && t.is_special(id), "{text}");
        assert_eq!(t.token_bytes(id).unwrap(), text.as_bytes());
    }
    assert_eq!(t.token_bytes(b'a' as u32).unwrap(), b"a");
    assert_eq!(t.token_bytes(32).unwrap(), b" ", "byte tokens are spelled in the byte alphabet and decode to raw bytes");
    assert_eq!(t.token_bytes(258).unwrap(), b"hell");
    assert!(!t.contains_id(260) && t.token_bytes(260).is_err());
    assert!(!t.is_special(b'a' as u32));
}

#[test]
fn merges_apply_in_rank_order_within_each_pre_token() {
    let t = fixture();
    let o = b'o' as u32;
    // "hello" -> he + ll -> hell, o; " world" keeps its leading space and merges " w".
    assert_eq!(t.encode("hello world").unwrap(), [258, o, 259, o, b'r' as u32, b'l' as u32, b'd' as u32]);
    // No merge crosses the pre-token boundary between "he" and " ll".
    assert_eq!(t.encode("he ll").unwrap(), [256, 32, 257]);
    assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
    // Loading the same file twice gives the same tokenizer and the same ids.
    assert!(t.shares_token_ids(&fixture()));
    assert_eq!(fixture().encode("hello hello").unwrap(), t.encode("hello hello").unwrap());
}

#[test]
fn decode_round_trips_ordinary_text_and_the_stream_decoder_agrees() {
    let t = fixture();
    for text in ["hello world", "  leading and trailing  ", "line\r\nbreaks\n\n", "naïve café 日本語 🚀", "quotes \" and \\ slash", "tabs\tand 1234567890"] {
        let ids = t.encode(text).unwrap();
        assert_eq!(t.decode(&ids, false).unwrap(), text);
        let mut stream = t.stream_decoder(false);
        let mut streamed = String::new();
        for &id in &ids { streamed.push_str(&stream.push(id).unwrap()); }
        streamed.push_str(&stream.finish());
        assert_eq!(streamed, text, "stream and whole decode must agree");
    }
    // A character split across byte tokens is held until it completes.
    let mut stream = t.stream_decoder(false);
    let rocket = "🚀".as_bytes();
    for &byte in &rocket[..3] { assert_eq!(stream.push(u32::from(byte)).unwrap(), ""); }
    assert_eq!(stream.push(u32::from(rocket[3])).unwrap(), "🚀");
    // An unfinished tail is flushed as one replacement character.
    assert_eq!(stream.push(u32::from(rocket[0])).unwrap(), "");
    assert_eq!(stream.finish(), "\u{fffd}");
    assert!(stream.push(260).is_err(), "unknown ids are errors, not silent gaps");
}

#[test]
fn encode_recognizes_controls_and_encode_ordinary_does_not() {
    let t = fixture();
    let text = "<|im_start|>hello<|im_end|>";
    assert_eq!(t.encode(text).unwrap(), [248045, 258, b'o' as u32, 248046]);
    let plain = t.encode_ordinary(text).unwrap();
    assert!(plain.iter().all(|&id| !t.is_special(id)), "untrusted text never becomes a control token: {plain:?}");
    assert_eq!(t.decode(&plain, false).unwrap(), text);
    let ids = t.encode(text).unwrap();
    assert_eq!(t.decode(&ids, false).unwrap(), text);
    assert_eq!(t.decode(&ids, true).unwrap(), "hello", "skip_special drops control tokens only");
}

#[test]
fn limits_bound_encode_and_decode() {
    let tight = Limits { max_input_bytes: 8, max_tokens: 3 };
    let t = Tokenizer::from_json(json(&MERGES, &CONTROLS).as_bytes(), tight).unwrap();
    assert!(t.encode("hello world").is_err(), "more than three tokens or eight bytes");
    assert_eq!(t.encode("hello").unwrap().len(), 2);
    assert!(t.decode(&[104, 104, 104, 104], false).is_err());
    let mut t = fixture();
    for zero in [Limits { max_input_bytes: 0, max_tokens: 1 }, Limits { max_input_bytes: 1, max_tokens: 0 }] {
        assert!(t.set_limits(zero).is_err());
        assert!(Tokenizer::from_json(json(&MERGES, &CONTROLS).as_bytes(), zero).is_err());
    }
    t.set_limits(tight).unwrap();
    assert!(t.encode("hello world").is_err());
}

#[test]
fn malformed_or_unsupported_files_are_refused_with_a_reason() {
    let good = json(&MERGES, &CONTROLS);
    for (from, to, reason) in [
        (r#""version":"1.0""#, r#""version":"2.0""#, "unsupported tokenizer version"),
        (r#""truncation":null"#, r#""truncation":{}"#, "truncation/padding"),
        (r#""normalizer":{"type":"NFC"}"#, r#""normalizer":{"type":"NFKC"}"#, "only NFC"),
        (r#""behavior":"Isolated""#, r#""behavior":"Removed""#, "unsupported split pattern"),
        (r#""type":"BPE""#, r#""type":"WordPiece""#, "only BPE"),
        (r#""dropout":null"#, r#""dropout":0.1"#, "dropout"),
        (r#""byte_fallback":false"#, r#""byte_fallback":true"#, "byte_fallback"),
    ] {
        assert!(good.contains(from), "fixture lost {from}");
        let error = refusal(&good.replacen(from, to, 1));
        assert!(error.contains(reason), "{from} -> {error}");
    }
    assert!(refusal(&json(&[("h", "q"), ("h", "q")], &CONTROLS)).contains("duplicate merge"));
    assert!(refusal(&json(&MERGES, &CONTROLS[..2])).contains("control token mismatch"));
    assert!(refusal(&json(&MERGES, &[CONTROLS[0], CONTROLS[1], ("<|im_end|>", 248047)])).contains("control token mismatch"));
    assert!(refusal(&good.replacen(r#""merges":["#, r#""merges":[["h","zz"],"#, 1)).contains("unknown right merge token"));
    assert!(refusal(&good.replacen(r#""!":33"#, r#""!":34"#, 1)).contains("duplicate or oversized token ID"));
    assert!(!refusal(&good[..good.len() - 1]).is_empty(), "truncated JSON");
    assert!(!refusal("").is_empty());
}

#[test]
fn shares_token_ids_compares_merges_and_vocabulary() {
    let t = fixture();
    let fewer = Tokenizer::from_json(json(&MERGES[..3], &CONTROLS).as_bytes(), Limits::default()).unwrap();
    assert!(!t.shares_token_ids(&fewer), "a missing merge changes ids");
    assert_ne!(fewer.encode("hello world").unwrap(), t.encode("hello world").unwrap());
}
