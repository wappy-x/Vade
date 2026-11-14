//! Inspect fixed pre-tokens in the public artifact, without the tokenizer crate.
//! Regex boundaries are supplied below. This is not a general text tokenizer.
#[path = "../../../../base/src/json.rs"]
pub mod json;
pub type Error = String;
pub type Result<T> = std::result::Result<T, Error>;
pub fn invalid(message: impl Into<String>) -> Error { message.into() }
use std::collections::{BTreeMap, HashMap};

fn main() {
    let path = std::env::args().nth(1).expect("usage: derive TOKENIZER_JSON");
    let bytes = std::fs::read(path).unwrap();
    let root = json::parse(&bytes, json::Limits { bytes:32<<20, nodes:2_000_000, string_bytes:1<<20, depth:24 }).unwrap();
    let model = root.object().unwrap()["model"].object().unwrap();
    assert!(model["ignore_merges"].boolean().unwrap());
    let vocab = model["vocab"].object().unwrap();
    let merge_order: HashMap<(String,String),usize> = model["merges"].array().unwrap().iter().enumerate().map(|(i,v)| {
        let (a,b) = if let json::Value::String(s) = v { s.split_once(' ').unwrap() }
            else { let parts = v.array().unwrap(); (parts[0].string().unwrap(),parts[1].string().unwrap()) };
        ((a.into(),b.into()),i)
    }).collect();
    let mut extra = 256;
    let alphabet: Vec<char> = (0..256u32).map(|b| {
        if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) { char::from_u32(b).unwrap() }
        else { extra += 1; char::from_u32(extra-1).unwrap() }
    }).collect();
    for pieces in [vec![], vec!["Hello"," world"], vec!["123","456","7"],
        vec!["雪"], vec!["é"], vec!["e","\u{301}"], vec!["日本語"], vec!["🚀"],
        vec!["\n"], vec!["\t"], vec![" "], vec!["Reasoning"," Effort",":"," Max"],
        vec!["go"], vec!["answer"], vec![" Low"], vec![" High"]] {
        let ids = pieces.iter().flat_map(|piece| {
            let key: String = piece.bytes().map(|b| alphabet[b as usize]).collect();
            merge(&key, vocab, &merge_order)
        }).collect::<Vec<_>>();
        println!("{pieces:?}: {ids:?}");
    }
}
fn merge(key: &str, vocab: &BTreeMap<String,json::Value>, merge_order: &HashMap<(String,String),usize>) -> Vec<u64> {
    if let Some(id) = vocab.get(key) { return vec![id.u64().unwrap()]; }
    let mut pieces = key.chars().map(|c| c.to_string()).collect::<Vec<_>>();
    // Scan all adjacent pairs for every merge. This reference has no heap,
    // linked nodes, split code, normalization code, or tokenizer calls.
    while let Some((at,_)) = pieces.windows(2).enumerate().filter_map(|(at,p)|
        merge_order.get(&(p[0].clone(),p[1].clone())).map(|&priority| (at,priority))).min_by_key(|&(at,priority)| (priority,at)) {
        let next = pieces.remove(at+1); pieces[at].push_str(&next);
    }
    pieces.iter().map(|p| vocab[p].u64().unwrap()).collect()
}
