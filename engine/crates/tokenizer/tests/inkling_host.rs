//! Host-only parity test for a tokenizer file with the case-aware split
//! pattern. The tokenizer file and the reference corpus are private. They
//! stay outside the tracked tree, in the ignored folder
//! `.local/cased-tokenizer/` at the repository root:
//!
//! - `tokenizer.json`: the checkpoint's file.
//! - `strings.jsonl`: one JSON object a line, with `category` and `text`.
//! - `reference_ids.jsonl`: one JSON object a line, with `ids` from the
//!   reference library for the string on the same line.
//!
//! The test reads no environment variable. It prints one `PARITY` line
//! with its counts.
use engine_model::json::{self, field, Value};
use engine_tokenizer::Tokenizer;

fn required_path(name: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/cased-tokenizer").join(name);
    assert!(path.is_file(), "put {name} in .local/cased-tokenizer/ for the ignored host test");
    path
}

fn parse_line(line: &str) -> Value {
    json::parse(line.as_bytes(), json::Limits { bytes: 1024 * 1024, nodes: 200_000, ..Default::default() }).unwrap()
}

#[test]
#[ignore = "requires a private tokenizer file and generated reference vectors"]
fn inkling_loader_matches_reference_and_guards_plain_control_text() {
    let tokenizer_path = required_path("tokenizer.json");
    let strings_path = required_path("strings.jsonl");
    let references_path = required_path("reference_ids.jsonl");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|error| panic!("tokenizer loader refused artifact: {error}"));

    let strings = std::fs::read_to_string(strings_path).unwrap();
    let references = std::fs::read_to_string(references_path).unwrap();
    let inputs = strings.lines().collect::<Vec<_>>();
    let expected = references.lines().collect::<Vec<_>>();
    assert_eq!(inputs.len(), expected.len());
    assert!(!inputs.is_empty());

    let mut differences = Vec::new();
    let (mut equal, mut decoded_back, mut ordinary_clean) = (0, 0, 0);
    let (mut control_records, mut control_guarded) = (0, 0);
    for (index, (&input, &reference)) in inputs.iter().zip(&expected).enumerate() {
        let input = parse_line(input);
        let input = input.object().unwrap();
        let text = field(input, "text").unwrap().string().unwrap();
        let category = field(input, "category").unwrap().string().unwrap();
        let reference = parse_line(reference);
        let reference_ids = field(reference.object().unwrap(), "ids").unwrap().array().unwrap().iter()
            .map(|value| u32::try_from(value.u64().unwrap()).unwrap()).collect::<Vec<_>>();

        let actual = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&actual, false).unwrap();
        if actual == reference_ids { equal += 1; }
        if decoded == text { decoded_back += 1; }
        if (actual != reference_ids || decoded != text) && differences.len() < 20 {
            differences.push(format!("record {index} ({category}) {text:?}: expected {reference_ids:?}, got {actual:?}, decoded {decoded:?}"));
        }

        // Caller text goes through `encode_ordinary`. It must give no
        // control token for any string, and it must decode back.
        let ordinary = tokenizer.encode_ordinary(text).unwrap();
        let clean = ordinary.iter().all(|&id| !tokenizer.is_special(id)) && tokenizer.decode(&ordinary, false).unwrap() == text;
        if clean { ordinary_clean += 1; }
        if category == "special_token_plain_text" {
            control_records += 1;
            // The reference turns the spelling into its control token.
            if clean && actual.iter().any(|&id| tokenizer.is_special(id)) { control_guarded += 1; }
        }
    }

    // Each control token of the file, by its own spelling.
    let (mut special_tokens, mut special_single_id, mut special_guarded) = (0, 0, 0);
    for id in (0..1_000_000u32).filter(|&id| tokenizer.is_special(id)) {
        special_tokens += 1;
        let spelling = std::str::from_utf8(tokenizer.token_bytes(id).unwrap()).unwrap();
        if tokenizer.encode(spelling).unwrap() == [id] { special_single_id += 1; }
        let ordinary = tokenizer.encode_ordinary(spelling).unwrap();
        if ordinary.iter().all(|&id| !tokenizer.is_special(id)) && tokenizer.decode(&ordinary, false).unwrap() == spelling
            && tokenizer.decode(&[id], true).unwrap().is_empty() { special_guarded += 1; }
    }

    let total = inputs.len();
    println!("PARITY strings={total} ids_equal={equal} decode_back={decoded_back} ordinary_without_control={ordinary_clean} control_records={control_records} control_records_guarded={control_guarded} control_tokens={special_tokens} control_single_id={special_single_id} control_spelling_guarded={special_guarded}");
    assert!(differences.is_empty(), "first differences:\n{}", differences.join("\n"));
    assert_eq!((equal, decoded_back, ordinary_clean), (total, total, total));
    assert_eq!(control_guarded, control_records);
    assert!(special_tokens > 0);
    assert_eq!((special_single_id, special_guarded), (special_tokens, special_tokens));
}
