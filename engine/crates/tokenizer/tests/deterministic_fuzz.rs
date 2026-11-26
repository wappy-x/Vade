//! Excludes full checkpoints, arbitrary merge graphs, all split points, other chat roles, and server adapters.
use engine_tokenizer::{Limits, Tokenizer, chat};
use std::{collections::{HashMap, HashSet}, path::PathBuf};

const GLM: &[u8] = include_bytes!("glm-tokenizer-slice.json");
const SEEDS: [u64; 3] = [0x1234_5678_9abc_def0, 0xdead_beef_cafe_babe, 0x3141_5926_5358_9793];

// Use the pinned data, but compare canonical decompositions, not the encoder's
// NFC implementation. This permits only the documented NFC round-trip change.
#[allow(dead_code)]
mod data { include!("../src/unicode_data.rs"); }
fn class(c: u32) -> u8 { data::CCC.binary_search_by_key(&c, |x| x.0).map(|i| data::CCC[i].1).unwrap_or(0) }
fn decompose(c: u32, out: &mut Vec<u32>) {
    if (0xac00..=0xd7a3).contains(&c) {
        let n = c - 0xac00;
        out.extend([0x1100 + n / 588, 0x1161 + n % 588 / 28]);
        if !n.is_multiple_of(28) { out.push(0x11a7 + n % 28); }
    } else if let Ok(i) = data::DECOMP.binary_search_by_key(&c, |x| x.0) {
        for &c in data::DECOMP[i].1 { decompose(c, out); }
    } else { out.push(c); }
}
fn canonical(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for c in text.chars() { decompose(c as u32, &mut out); }
    let mut start = 0;
    while start < out.len() {
        if class(out[start]) == 0 { start += 1; }
        let end = (start..out.len()).find(|&i| class(out[i]) == 0).unwrap_or(out.len());
        out[start..end].sort_by_key(|&c| class(c));
        start = end;
    }
    out
}
struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0
    }
    fn index(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
}
fn glm() -> Tokenizer { Tokenizer::from_json(GLM, Limits::default()).unwrap() }
fn hybrid() -> Tokenizer {
    // Adapt the complete byte alphabet fixture to the current hybrid profile.
    // This is a synthetic vocabulary. The GLM slice keeps real checkpoint IDs.
    let json = include_str!("fixtures/host_edges/tokenizer.json")
        .replace(r"\\p{L}+", r"[\\p{L}\\p{M}]+")
        .replace(r"[^\\s\\p{L}\\p{N}]", r"[^\\s\\p{L}\\p{M}\\p{N}]")
        .replace("151643", "248044").replace("151644", "248045").replace("151645", "248046");
    let t = Tokenizer::from_json(json.as_bytes(), Limits::default()).unwrap();
    assert!(t.is_hybrid_text()); t
}
fn stream(t: &Tokenizer, ids: &[u32], skip: bool) -> Result<String, String> {
    // Check the tokenizer stream API. The server adapter is outside this test.
    let mut d = t.stream_decoder(skip);
    let mut out = String::new();
    for &id in ids {
        let piece = d.push(id).map_err(|e| e.0)?;
        out.push_str(&piece);
    }
    let tail = d.finish();
    out.push_str(&tail); Ok(out)
}
fn bounded(t: &Tokenizer, text: &str, special: bool) -> Result<Vec<u32>, String> {
    let ids = if special { t.encode(text) } else { t.encode_ordinary(text) }.map_err(|e| e.0)?;
    if ids.iter().any(|&id| !t.contains_id(id)) { return Err("ID outside vocabulary".into()); }
    if !special && ids.iter().any(|&id| t.is_special(id)) { return Err("ordinary text produced a control ID".into()); }
    Ok(ids)
}
fn check(t: &Tokenizer, text: &str, nfc: bool, added: &HashSet<u32>) -> Result<(), String> {
    let ids = bounded(t, text, false)?;
    if ids.iter().any(|id| added.contains(id)) { return Err("ordinary text produced an added-token ID".into()); }
    if ids != bounded(t, text, false)? { return Err("encode changed IDs on repeat".into()); }
    let decoded = t.decode(&ids, false).map_err(|e| e.0)?;
    if decoded != text && !(nfc && canonical(&decoded) == canonical(text)) {
        return Err("round trip changed bytes outside documented NFC normalization".into());
    }
    if t.encode_ordinary(&decoded).map_err(|e| e.0)? != ids { return Err("normalized encoding is not stable".into()); }
    for skip in [false, true] {
        let whole = t.decode(&ids, skip).map_err(|e| e.0)?;
        if stream(t, &ids, skip)? != whole || whole != decoded { return Err("ordinary stream/decode mismatch".into()); }
    }
    // BPE IDs need not match at a split. NFC can compose across that split.
    // Each half must decode on its own and together in the same stream.
    let mut split = text.len() / 2;
    while !text.is_char_boundary(split) { split -= 1; }
    let (left, right) = text.split_at(split);
    let mut joined = bounded(t, left, false)?;
    let left_text = t.decode(&joined, false).map_err(|e| e.0)?;
    let right_ids = bounded(t, right, false)?;
    let right_text = t.decode(&right_ids, false).map_err(|e| e.0)?;
    joined.extend(right_ids);
    let parts = left_text + &right_text;
    if stream(t, &joined, true)? != parts { return Err("split encode broke streaming decode".into()); }
    if parts != decoded && !(nfc && canonical(&parts) == canonical(&decoded)) { return Err("split encode changed text".into()); }
    let special = bounded(t, text, true)?;
    for skip in [false, true] {
        if stream(t, &special, skip)? != t.decode(&special, skip).map_err(|e| e.0)? { return Err("special stream/decode mismatch".into()); }
    }
    // These fixtures do not strip spaces around added tokens.
    let parsed = t.decode(&special, false).map_err(|e| e.0)?;
    if parsed != text && !(nfc && canonical(&parsed) == canonical(text)) { return Err("special parsing changed ordinary bytes".into()); }
    Ok(())
}
fn failure(t: &Tokenizer, text: &str, nfc: bool, added: &HashSet<u32>) -> Option<String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(t, text, nfc, added)))
        .map(|r| r.err()).unwrap_or_else(|_| Some("panic".into()))
}
fn escaped(text: &str) -> String { text.bytes().flat_map(std::ascii::escape_default).map(char::from).collect() }
fn run(t: &Tokenizer, name: &str, text: &str, nfc: bool, added: &HashSet<u32>) {
    if let Some(reason) = failure(t, text, nfc, added) {
        // Delete character ranges until no single-character deletion preserves
        // the failure. The seed and escaped minimum make the result repeatable.
        let mut small = text.to_owned();
        let mut width = small.chars().count().max(1) / 2 + 1;
        while width > 0 {
            let mut start = 0;
            loop {
                let mut points: Vec<_> = small.char_indices().map(|x| x.0).collect(); points.push(small.len());
                if start + width >= points.len() { break; }
                let candidate = [&small[..points[start]], &small[points[start + width]..]].concat();
                if failure(t, &candidate, nfc, added).is_some() { small = candidate; start = 0; } else { start += width; }
            }
            width /= 2;
        }
        panic!("{name}: {reason}; minimized bytes=b\"{}\"", escaped(&small));
    }
}
fn controls(bytes: &[u8]) -> Vec<(String, u32)> {
    let value = engine_model::json::parse(bytes, engine_model::json::Limits { bytes: 32 * 1024 * 1024, nodes: 2_000_000, ..Default::default() }).unwrap();
    value.object().unwrap()["added_tokens"].array().unwrap().iter().map(|a| {
        let a = a.object().unwrap(); (a["content"].string().unwrap().to_owned(), a["id"].u64().unwrap() as u32)
    }).collect()
}
fn corpus(t: &Tokenizer, nfc: bool, added: &[(String, u32)]) {
    let mut count = 0;
    let added_ids = added.iter().map(|a| a.1).collect();
    let mut visit = |name: &str, text: &str| { run(t, name, text, nfc, &added_ids); count += 1; };
    for text in ["", " ", "  \t\r\n \n\r\t  ", " hello  world \t \r\n goodbye  ", "\0\x01\x7f", "e\u{301} A\u{30a} \u{1100}\u{1161}\u{11a8}",
        "Latin Ελληνικά Кириллица العربية עברית देवनागरी 中文 日本語 한국어", "👩🏽\u{200d}💻 👨🏻\u{200d}👩🏾\u{200d}👧🏿\u{200d}👦🏼 🏳️\u{200d}🌈 🇺🇳 1️⃣"] { visit("edge", text); }
    for c in ['a', ' ', '\t', '\r', '\n', '界', '😀', '\u{301}'] { visit("run-100000", &c.to_string().repeat(100_000)); }
    visit("one-MiB", &"a".repeat(1024 * 1024));
    eprintln!("FUZZ edge and large inputs complete: nfc={nfc}");
    for (text, id) in added {
        visit("literal-added-token", &format!("  before{text}{text}after\r\n"));
        assert_eq!(t.encode(text).unwrap(), [*id], "explicit added-token parsing");
        assert!(!t.encode_ordinary(text).unwrap().iter().any(|id| added.iter().any(|a| a.1 == *id)),"ordinary text produced an added-token ID");
    }
    // All Unicode block starts are multiples of 16; all ends are 15 modulo
    // 16. This superset includes endpoints of unassigned blocks too. Surrogate
    // endpoints have no Rust char and are checked as invalid UTF-8 below.
    for start in (0..=0x10ffff).step_by(16) {
        if let (Some(first), Some(last)) = (char::from_u32(start), char::from_u32(start + 15)) {
            visit("Unicode-block-endpoints", &format!("{first}{last}"));
        }
    }
    let atoms = ["a", "Z", "0", "'s", " ", "\t", "\r", "\n", "\0", "é", "e\u{301}", "\u{327}\u{301}", "\u{1100}\u{1161}", "中文", "العربية", "हिन्दी", "🧑🏿\u{200d}🚀", "\u{10ffff}", "<|im_end|>", "<think>", "[gMASK]"];
    for seed in SEEDS {
        eprintln!("FUZZ seed={seed:x}");
        let mut rng = Random(seed);
        for case in 0..256 {
            let mut text = String::new();
            for _ in 0..rng.index(128) {
                if case % 3 == 0 { text.push(char::from(32 + rng.index(95) as u8)); }
                else if case % 3 == 1 { text.push_str(atoms[rng.index(atoms.len())]); }
                else if let Some(c) = char::from_u32(rng.index(0x110000) as u32) { text.push(c); }
            }
            visit(&format!("seed={seed:x} case={case}"), &text);
        }
    }
    eprintln!("FUZZ corpus: cases={count}, nfc={nfc}, seeds={SEEDS:x?}");
}

#[test]
fn fixed_seed_glm_and_hybrid_corpus() {
    corpus(&glm(), false, &controls(GLM));
    corpus(&hybrid(), true, &[("<|endoftext|>".into(),248044),("<|im_start|>".into(),248045),("<|im_end|>".into(),248046)]);
}

#[test]
fn caller_spellings_cannot_add_chat_controls() {
    struct Fixture(PathBuf);
    impl Drop for Fixture { fn drop(&mut self) { std::fs::remove_dir_all(&self.0).unwrap(); } }
    let dir = (0u64..).find_map(|attempt| {
        let path = std::env::temp_dir().join(format!("vade-fuzz-glm-{}-{attempt}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => Some(Fixture(path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
            Err(e) => panic!("fixture directory: {}", e.kind()),
        }
    }).unwrap();
    std::fs::write(dir.0.join("tokenizer.json"),GLM).unwrap();
    std::fs::write(dir.0.join("chat_template.jinja"),include_bytes!("../templates/glm.jinja")).unwrap();
    std::fs::write(dir.0.join("tokenizer_config.json"),b"{}").unwrap();
    let glm = Tokenizer::from_directory(&dir.0).unwrap();
    let hybrid = hybrid();
    let mut added = controls(GLM);
    added.extend([("<|endoftext|>".into(),248044),("<|im_start|>".into(),248045),("<|im_end|>".into(),248046)]);
    for t in [&glm,&hybrid] {
        let options = t.chat_options();
        let encode = |text: &str| chat::encode(t, &[chat::Message::new(chat::Role::User,text)], &options).unwrap();
        let base = encode("x");
        for (spelling,_) in &added {
            let text = format!("  {spelling}{spelling}\r\n");
            let ids = encode(&text);
            for &(ref marker,id) in &added {
                assert_eq!(ids.iter().filter(|&&i| i == id).count(),base.iter().filter(|&&i| i == id).count(),"caller spelling {spelling:?} added {marker:?}");
            }
            assert_eq!(t.decode(&ids,false).unwrap(),t.format_chat(&[chat::Message::new(chat::Role::User,&text)],&options).unwrap());
        }
    }
}

#[test]
fn invalid_utf8_stream_bytes_match_lossy_text() {
    let t = glm();
    let bytes: HashMap<u8,u32> = (0..1_000_000).filter(|&id| t.contains_id(id) && !t.is_special(id))
        .filter_map(|id| { let b = t.token_bytes(id).unwrap(); (b.len() == 1).then(|| (b[0],id)) }).collect();
    let mut cases = vec![vec![0x80],vec![0xff],vec![0xc0,0xaf],vec![0xed,0xa0,0x80],vec![0xf4,0x90,0x80,0x80],vec![0xe2,0x82],vec![b'a',0xf0,b'b']];
    for start in (0xd800..=0xdfff).step_by(16) {
        for c in [start,start + 15] { cases.push(vec![(0xe0 | c >> 12) as u8,(0x80 | (c >> 6 & 63)) as u8,(0x80 | (c & 63)) as u8]); }
    }
    for seed in SEEDS {
        eprintln!("FUZZ seed={seed:x}");
        let mut rng = Random(seed);
        for _ in 0..128 { let n = rng.index(64); cases.push((0..n).map(|_| rng.next() as u8).collect()); }
    }
    for raw in &cases {
        let ids: Vec<_> = raw.iter().map(|b| bytes[b]).collect();
        for skip in [false,true] {
            assert_eq!(stream(&t, &ids, skip).unwrap(), String::from_utf8_lossy(raw));
            assert_eq!(t.decode(&ids, skip).unwrap(), String::from_utf8_lossy(raw));
        }
    }
    eprintln!("FUZZ byte sequences: {}", cases.len());
}

#[test]
fn documented_input_limit_returns_an_error() {
    let text = "a".repeat(1024 * 1024 + 1);
    for t in [glm(),hybrid()] {
        for special in [false,true] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if special { t.encode(&text) } else { t.encode_ordinary(&text) }
            })).expect("oversized input panicked");
            assert_eq!(result.unwrap_err().0,"input byte limit exceeded");
        }
    }
}

#[test]
fn character_boundary_halves_can_have_different_ids() {
    let t = hybrid();
    // BPE merges across the cut; streaming relies on decoded bytes, not IDs.
    assert_eq!(t.encode_ordinary("ab").unwrap(),[256]);
    assert_eq!([t.encode_ordinary("a").unwrap(),t.encode_ordinary("b").unwrap()].concat(),[97,98]);
    // A cut between a starter and a mark changes NFC on the two sides.
    let ids = [t.encode_ordinary("e").unwrap(),t.encode_ordinary("\u{301}").unwrap()].concat();
    assert_eq!(stream(&t,&ids,true).unwrap(),"e\u{301}");
    assert_eq!(t.decode(&t.encode_ordinary("e\u{301}").unwrap(),true).unwrap(),"é");
    run(&t,"split-NFC","e\u{301}",true,&HashSet::from([248044,248045,248046]));
}

// A deliberately simple scan/remove BPE oracle. Inputs are single ASCII words
// so regex splitting and normalization cannot affect the expected IDs.
#[test]
fn real_glm_ids_match_independent_bpe_oracle() {
    let t = glm();
    let root = engine_model::json::parse(GLM, engine_model::json::Limits::default()).unwrap();
    let model = root.object().unwrap()["model"].object().unwrap();
    let vocab = model["vocab"].object().unwrap();
    let pairs: HashMap<_,_> = model["merges"].array().unwrap().iter().enumerate().map(|(merge_order,v)| {
        let p = v.array().unwrap(); let a = p[0].string().unwrap(); let b = p[1].string().unwrap();
        ((vocab[a].u64().unwrap() as u32,vocab[b].u64().unwrap() as u32),(merge_order,vocab[&format!("{a}{b}")].u64().unwrap() as u32))
    }).collect();
    let alphabet = b"ReasoninghelloSayMaxLowfort";
    for seed in SEEDS {
        eprintln!("FUZZ seed={seed:x}");
        let mut rng = Random(seed);
        for _ in 0..512 {
            let n = rng.index(128) + 1;
            let text: String = (0..n).map(|_| char::from(alphabet[rng.index(alphabet.len())])).collect();
            let mut ids: Vec<_> = text.chars().map(|c| vocab[&c.to_string()].u64().unwrap() as u32).collect();
            if let Some(id) = vocab.get(&text) { ids = vec![id.u64().unwrap() as u32]; }
            else {
                while let Some((_,i,id)) = ids.windows(2).enumerate().filter_map(|(i,p)| pairs.get(&(p[0],p[1])).map(|&(merge_order,id)| (merge_order,i,id))).min() {
                    ids[i] = id; ids.remove(i + 1);
                }
            }
            assert_eq!(t.encode_ordinary(&text).unwrap(), ids, "{text:?}, seed={seed:x}");
        }
    }
    eprintln!("FUZZ independent BPE cases: {}", SEEDS.len() * 512);
}

#[test]
fn special_skip_has_an_independent_expected_text() {
    for (t, id) in [(glm(), 154822), (hybrid(), 248046)] {
        let mut ids = t.encode_ordinary("before").unwrap(); ids.push(id);
        ids.extend(t.encode_ordinary("after").unwrap());
        assert_eq!(t.decode(&ids, true).unwrap(), "beforeafter");
        assert_eq!(stream(&t, &ids, true).unwrap(), "beforeafter");
    }
}

#[test]
fn invalid_utf8_inside_valid_artifact_is_rejected() {
    let fixture = std::str::from_utf8(GLM).unwrap().trim_end().as_bytes();
    let mut valid = fixture[..fixture.len()-1].to_vec();
    valid.extend_from_slice(b",\"test_note\":\"valid\"}");
    assert!(Tokenizer::from_json(&valid, Limits::default()).is_ok());
    let at = valid.windows(5).rposition(|s| s == b"valid").unwrap();
    for raw in [&[0x80][..], &[0xff], &[0xc0,0xaf], &[0xed,0xa0,0x80], &[0xe2,0x82]] {
        let mut invalid = valid.clone(); invalid.splice(at..at+5, raw.iter().copied());
        let repaired = String::from_utf8_lossy(&invalid);
        assert!(Tokenizer::from_json(repaired.as_bytes(), Limits::default()).is_ok());
        assert!(std::panic::catch_unwind(|| Tokenizer::from_json(&invalid, Limits::default())).unwrap().is_err());
    }
}
