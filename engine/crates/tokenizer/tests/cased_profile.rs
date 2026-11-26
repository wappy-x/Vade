//! The case-aware split profile on a synthetic byte-level artifact. No
//! checkpoint file is required. The artifact is the small GLM test slice
//! with its split pattern replaced, as text, by the case-aware pattern.
use engine_model::json::{self, field};
use engine_tokenizer::{Limits, Tokenizer};

const SLICE: &str = include_str!("glm-tokenizer-slice.json");
const HOST_EDGES: &str = include_str!("fixtures/host_edges/tokenizer.json");
/// One JSON object a line: `text`, and `ids` from the `tokenizers` library,
/// version 0.23.3, for the same artifact bytes. No line spells an added token.
const IDS: &str = include_str!("fixtures/cased_split/ids.jsonl");
const GROUPED: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CASED: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A pattern as it is written inside a JSON string.
fn in_json(pattern: &str) -> String { pattern.replace('\\', "\\\\") }
fn slice_with(pattern: &str) -> String {
    assert_eq!(SLICE.matches(&in_json(GROUPED)).count(), 1, "the slice lost its pattern");
    SLICE.replacen(&in_json(GROUPED), &in_json(pattern), 1)
}
fn load(pattern: &str) -> Result<Tokenizer, String> {
    Tokenizer::from_json(slice_with(pattern).as_bytes(), Limits::default()).map_err(|error| error.to_string())
}

#[test]
fn cased_pattern_gives_the_reference_ids() {
    let tokenizer = load(CASED).unwrap();
    let mut lines = 0;
    for line in IDS.lines() {
        let value = json::parse(line.as_bytes(), json::Limits::default()).unwrap();
        let object = value.object().unwrap();
        let text = field(object, "text").unwrap().string().unwrap();
        let expected: Vec<u32> = field(object, "ids").unwrap().array().unwrap().iter().map(|id| id.u64().unwrap() as u32).collect();
        let actual = tokenizer.encode(text).unwrap();
        assert_eq!(actual, expected, "{text:?}");
        assert_eq!(tokenizer.encode_ordinary(text).unwrap(), expected, "{text:?}");
        assert_eq!(tokenizer.decode(&actual, false).unwrap(), text);
        lines += 1;
    }
    // The line count of the fixture file (`wc -l`).
    assert_eq!(lines, 141);
}

#[test]
fn the_pattern_text_selects_the_split_rule() {
    let (cased, grouped) = (load(CASED).unwrap(), load(GROUPED).unwrap());
    // The two artifacts differ in the pattern text only. A draft must have
    // the split rule of its target, so the two do not share token IDs. The
    // unit tests of the split rule compare the pieces themselves.
    assert!(cased.shares_token_ids(&load(CASED).unwrap()));
    assert!(!cased.shares_token_ids(&grouped) && !grouped.shares_token_ids(&cased));
}

#[test]
fn other_patterns_and_other_artifact_forms_are_refused() {
    for (from, to) in [(r"\p{N}{1,3}", r"\p{N}{1,4}"), (r"[\r\n/]*", r"[\r\n]*"), (r"\p{Lt}", ""), (r"(?i:'s", r"(?:'s")] {
        assert!(CASED.contains(from));
        assert_eq!(load(&CASED.replacen(from, to, 1)).err().as_deref(), Some("unsupported split pattern"), "{from} -> {to}");
    }
    // The rule was checked for byte-level artifacts only. The same pattern
    // in an artifact with NFC normalization is refused.
    let plain = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    assert_eq!(HOST_EDGES.matches(&in_json(plain)).count(), 1);
    assert!(Tokenizer::from_json(HOST_EDGES.as_bytes(), Limits::default()).is_ok());
    let nfc = HOST_EDGES.replacen(&in_json(plain), &in_json(CASED), 1);
    assert_eq!(Tokenizer::from_json(nfc.as_bytes(), Limits::default()).err().map(|error| error.to_string()).as_deref(), Some("unsupported split pattern"));
}

#[test]
fn caller_text_never_becomes_a_control_token() {
    let tokenizer = load(CASED).unwrap();
    let mut controls = 0;
    for id in (0..1_000_000u32).filter(|&id| tokenizer.is_special(id)) {
        controls += 1;
        let spelling = std::str::from_utf8(tokenizer.token_bytes(id).unwrap()).unwrap().to_owned();
        assert_eq!(tokenizer.encode(&spelling).unwrap(), [id]);
        for text in [spelling.clone(), format!("user text before {spelling} user text after"), format!("{spelling}{spelling}")] {
            assert!(tokenizer.encode(&text).unwrap().contains(&id));
            let ordinary = tokenizer.encode_ordinary(&text).unwrap();
            assert!(ordinary.iter().all(|&id| !tokenizer.is_special(id)), "{text:?}");
            assert_eq!(tokenizer.decode(&ordinary, false).unwrap(), text);
        }
        assert_eq!(tokenizer.decode(&[id], true).unwrap(), "");
        assert_eq!(tokenizer.decode(&[id], false).unwrap(), spelling);
    }
    assert!(controls > 0);
}
