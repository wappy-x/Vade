//! Embedding norm, last norm, head and logit cap against a saved reference trace and the real stored weights.
//! Ignored by default: the weight file and the trace are private. They stay
//! outside the tracked tree, in the ignored folder `.local/capped-reference/`
//! at the repository root:
//!
//! - `model/`: the model folder (`config.json` and one weight file). A link is enough.
//! - `trace/`: a trace folder with one `trace-index*.json` and the files it names.
//!
//! The test reads no environment variable. It reads the first run of the
//! index. The embedding norm and the last norm are checked at every position.
//! The head is checked at the last position, with the trace's own norm output
//! as input.
use engine_model::{capped_head::{embed_row, head_rows, last_norm, top_token}, json::{self, Value}, norm_cap::{SoftCap, round_bf16, weighted_norm},
    safetensors::{Dtype, Limits, TensorFile}};
use std::{fs::File, io::{Read, Seek, SeekFrom}, path::{Path, PathBuf}};

fn local(name: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.local/capped-reference").join(name);
    assert!(path.exists(), "put {name} in .local/capped-reference/ for the ignored host test");
    path
}
fn trace_index(folder: &Path) -> PathBuf {
    let mut found: Vec<PathBuf> = std::fs::read_dir(folder).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("trace-index") && p.extension().is_some_and(|e| e == "json")).collect();
    found.sort();
    found.into_iter().next().expect("a trace-index file in the trace folder")
}
fn floats(path: &Path, first: usize, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(first as u64 * 4)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}
fn bf16_rows(file: &TensorFile, name: &str, first: usize, count: usize) -> Vec<f32> {
    assert_eq!(file.header().tensors[name].dtype, Dtype::Bf16, "{name}");
    let mut bytes = vec![0u8; count * 2];
    file.read_range(name, first as u64 * 2, &mut bytes).unwrap();
    bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect()
}
fn shape(entry: &Value, name: &str) -> (PathBuf, Vec<usize>) {
    let file = json::field(json::field(entry.object().unwrap(), "files").unwrap().object().unwrap(), name).unwrap().object().unwrap();
    (PathBuf::from(file["file"].string().unwrap()), file["shape"].array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect())
}
fn equal(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count() }

#[test]
#[ignore = "requires the real weight file and a reference trace outside the repository"]
fn embedding_last_norm_and_head_match_the_reference_trace() {
    let (model, folder) = (local("model"), local("trace"));
    let config = json::parse(&std::fs::read(model.join("config.json")).unwrap(), json::Limits::default()).unwrap();
    let text = config.object().unwrap()["text_config"].object().unwrap();
    let (epsilon, used_rows, cap) = (text["rms_norm_eps"].f64().unwrap() as f32, text["unpadded_vocab_size"].u64().unwrap() as usize,
        text["final_logit_softcapping"].f64().unwrap());
    let soft = SoftCap::new(cap, "logit cap").unwrap();
    let weights = std::fs::read_dir(&model).unwrap().map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with("model-") && p.extension().is_some_and(|e| e == "safetensors")).expect("weight file");
    let file = TensorFile::open(&weights, Limits::default()).unwrap();

    let index = json::parse(&std::fs::read(trace_index(&folder)).unwrap(), json::Limits::default()).unwrap();
    let entry = &index.object().unwrap()["files"].array().unwrap()[0];
    let (state_file, states) = shape(entry, "states");
    let (logit_file, logit_shape) = shape(entry, "logits");
    let (count, tokens, hidden) = (states[0], states[1], states[2]);
    assert_eq!(logit_shape, [tokens, used_rows]);
    let ids: Vec<usize> = entry.object().unwrap()["input_ids"].array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect();
    assert_eq!(ids.len(), tokens);
    let state = |index: usize, token: usize| floats(&folder.join(&state_file), (index * tokens + token) * hidden, hidden);

    // Embedding norm: state 0 is the BF16 output of the embedding with its norm.
    let embed_norm = bf16_rows(&file, "model.llm.embed_norm.weight", 0, hidden);
    let last_weight = bf16_rows(&file, "model.llm.norm.weight", 0, hidden);
    let (mut embed_equal, mut norm_equal, mut norm_gap) = (0usize, 0usize, 0.0f32);
    for (token, &id) in ids.iter().enumerate() {
        let row = bf16_rows(&file, "model.llm.embed.weight", id * hidden, hidden);
        embed_equal += equal(&embed_row(&row, &embed_norm, epsilon).unwrap(), &state(0, token));
        // Last norm: state `count - 2` is its FP32 input, state `count - 1` its FP32 output before the BF16 cast.
        let (input, want) = (state(count - 2, token), state(count - 1, token));
        let exact = weighted_norm(&input, &last_weight, epsilon).unwrap();
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        norm_gap = norm_gap.max(exact.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale);
        let rounded: Vec<f32> = want.iter().map(|&v| round_bf16(v)).collect();
        norm_equal += equal(&last_norm(&input, &last_weight, epsilon).unwrap(), &rounded);
    }
    println!("EMBED positions {tokens}: values equal {embed_equal}/{}", tokens * hidden);
    println!("LAST NORM positions {tokens}: BF16 values equal {norm_equal}/{}; largest FP32 gap {norm_gap:e} of the largest output", tokens * hidden);

    // Head and logit cap at one position, from the trace's own norm output.
    let token = tokens - 1;
    let y: Vec<f32> = state(count - 1, token).iter().map(|&v| round_bf16(v)).collect();
    let want = floats(&folder.join(&logit_file), token * used_rows, used_rows);
    let stored_rows = file.header().tensors["model.llm.unembed.weight"].shape[0] as usize;
    assert!(stored_rows >= used_rows);
    let mut before = Vec::with_capacity(used_rows);
    let mut first = 0usize;
    while first < used_rows {
        let rows = (used_rows - first).min(4096);
        let chunk = bf16_rows(&file, "model.llm.unembed.weight", first * hidden, rows * hidden);
        before.extend(head_rows(&chunk, hidden, rows, &y).unwrap());
        first += rows;
    }
    let got: Vec<f32> = before.iter().map(|&v| soft.apply(v)).collect();
    let same = equal(&got, &want);
    // The trace holds capped FP32 logits. A trace value within 4 FP32 steps of the FP64 cap of this
    // reference's BF16 logit comes from the same BF16 logit: a BF16 step is thousands of FP32 steps.
    let same_before = before.iter().zip(&want).filter(|&(&b, &w)| {
        let exact = cap * ((b as f64) / cap).tanh();
        ((w as f64) - exact).abs() <= 4.0 * ((f32::from_bits(w.abs().to_bits() + 1) - w.abs()) as f64) }).count();
    let gap = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let (mine, theirs) = (top_token(&got).unwrap(), top_token(&want).unwrap());
    let ties = want.iter().filter(|&&v| v == want[theirs]).count();
    println!("HEAD position {token}: BF16 logits before the cap equal {same_before}/{used_rows}; capped FP32 logits equal {same}/{used_rows}; largest gap {gap:e}; top token {mine} (trace {theirs}, {ties} at the trace's top value); top logit {} (trace {})", got[mine], want[theirs]);
    assert!(embed_equal * 100 >= tokens * hidden * 99, "fewer than 99 in 100 embedding values equal the trace");
    assert!(norm_equal * 100 >= tokens * hidden * 99, "fewer than 99 in 100 last norm values equal the trace");
    assert!(same_before * 10 >= used_rows * 9, "fewer than 9 in 10 BF16 logits equal the trace");
    assert_eq!(mine, theirs, "the top token differs from the trace");
}
