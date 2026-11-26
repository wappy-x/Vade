//! Routed expert block against a saved reference trace and the real stored weights.
//! Ignored by default: the weight file and the trace are private. They stay
//! outside the tracked tree, in the ignored folder `.local/capped-reference/`
//! at the repository root:
//!
//! - `model/`: the model folder (`config.json` and one weight file). A link is enough.
//! - `trace/`: a trace folder with one `trace-index*.json` and the files it names.
//!
//! The test reads no environment variable. It checks the last token of the
//! first run of the index at every layer visit. For each visit it prints one
//! line: the largest router logit difference, the picked sets, the largest
//! router weight difference, and the count of block outputs that equal the
//! trace bit for bit.
use engine_model::{capped_moe::{CappedActivation, Expert, Routing, fold}, expert_widen::{Stored, StoredMatrix},
    json::{self, Value}, safetensors::{Dtype, Limits, TensorFile}};
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
fn tensor_bytes(file: &TensorFile, name: &str, first: usize, count: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; count];
    file.read_range(name, first as u64, &mut bytes).unwrap();
    bytes
}
fn tensor_f32(file: &TensorFile, name: &str, first: usize, count: usize) -> Vec<f32> {
    let tensor = &file.header().tensors[name];
    match tensor.dtype {
        Dtype::F32 => tensor_bytes(file, name, first * 4, count * 4).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        Dtype::Bf16 => tensor_bytes(file, name, first * 2, count * 2).as_chunks::<2>().0.iter().map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16)).collect(),
        other => panic!("{name} has type {other:?}"),
    }
}
/// One stored matrix of one expert, read from the file. The format follows the stored type.
fn stored(file: &TensorFile, base: &str, expert: usize, hold: &mut (Vec<u8>, Vec<u8>)) -> (usize, usize, Option<(f32, f32)>) {
    let weight = &file.header().tensors[base];
    let scale = &file.header().tensors[&format!("{base}_scale")];
    let rows = weight.shape[1] as usize;
    let (each, each_scale) = ((weight.shape[1] * weight.shape[2]) as usize, (scale.shape[1] * scale.shape[2]) as usize);
    hold.0 = tensor_bytes(file, base, expert * each, each);
    hold.1 = tensor_bytes(file, &format!("{base}_scale"), expert * each_scale, each_scale);
    match weight.dtype {
        Dtype::U8 => {
            let amax = tensor_f32(file, &format!("{base}_amax"), expert, 1)[0];
            let denominator = tensor_f32(file, &format!("{base}_outer_scale_denominator"), 0, 1)[0];
            (rows, weight.shape[2] as usize * 2, Some((amax, denominator)))
        }
        Dtype::F8E4M3 => (rows, weight.shape[2] as usize, None),
        other => panic!("{base} has type {other:?}"),
    }
}
fn matrix<'a>(shape: (usize, usize, Option<(f32, f32)>), hold: &'a (Vec<u8>, Vec<u8>)) -> StoredMatrix<'a> {
    let data = match shape.2 {
        Some((amax, denominator)) => Stored::Packed4 { codes: &hold.0, scales: &hold.1, amax, denominator },
        None => Stored::Float8 { values: &hold.0, scales: &hold.1 },
    };
    StoredMatrix { rows: shape.0, columns: shape.1, data }
}
fn shape(entry: &Value, name: &str) -> (PathBuf, Vec<usize>) {
    let file = json::field(json::field(entry.object().unwrap(), "files").unwrap().object().unwrap(), name).unwrap().object().unwrap();
    (PathBuf::from(file["file"].string().unwrap()), file["shape"].array().unwrap().iter().map(|v| v.u64().unwrap() as usize).collect())
}

#[test]
#[ignore = "requires the real weight file and a reference trace outside the repository"]
fn routed_expert_block_matches_the_reference_trace() {
    let (model, folder) = (local("model"), local("trace"));
    let config = json::parse(&std::fs::read(model.join("config.json")).unwrap(), json::Limits::default()).unwrap();
    let text = config.object().unwrap()["text_config"].object().unwrap();
    let routing = Routing::new(text["n_routed_experts"].u64().unwrap() as usize, text["num_experts_per_tok"].u64().unwrap() as usize,
        text["route_scale"].f64().unwrap()).unwrap();
    let activation = CappedActivation::new(text["mlp_gate_softcap"].f64().unwrap(), text["mlp_up_softcap"].f64().unwrap()).unwrap();
    let weights = std::fs::read_dir(&model).unwrap().map(|e| e.unwrap().path())
        .find(|p| p.file_name().unwrap().to_str().unwrap().starts_with("model-") && p.extension().is_some_and(|e| e == "safetensors")).expect("weight file");
    let file = TensorFile::open(&weights, Limits::default()).unwrap();

    let index = json::parse(&std::fs::read(trace_index(&folder)).unwrap(), json::Limits::default()).unwrap();
    let entry = &index.object().unwrap()["files"].array().unwrap()[0];
    let (io_file, io) = shape(entry, "expert-io");
    let (logit_file, _) = shape(entry, "router-logits");
    let (weight_file, _) = shape(entry, "router-weights");
    let (pick_file, picks) = shape(entry, "router-picks");
    let (visits, tokens, hidden, selected, experts) = (io[0], io[2], io[3], picks[2], routing.experts());
    assert_eq!((io[1], picks[0], picks[1], selected), (2, visits, tokens, routing.selected()));
    let (token, range) = (tokens - 1, (0, visits - 1));
    let layers: Vec<usize> = entry.object().unwrap()["visits"].array().unwrap().iter()
        .map(|v| v.object().unwrap()["layer_module"].u64().unwrap() as usize).collect();

    let (mut equal_all, mut total, mut picks_equal, mut worst) = (0usize, 0usize, 0usize, 0.0f32);
    for (visit, &layer) in layers.iter().enumerate().take(range.1 + 1).skip(range.0) {
        let base = format!("model.llm.layers.{layer}.mlp");
        let z = floats(&folder.join(&io_file), (visit * 2 * tokens + token) * hidden, hidden);
        let want = floats(&folder.join(&io_file), ((visit * 2 + 1) * tokens + token) * hidden, hidden);
        let ref_logits = floats(&folder.join(&logit_file), (visit * tokens + token) * experts, experts);
        let ref_weights = floats(&folder.join(&weight_file), (visit * tokens + token) * selected, selected);
        let ref_picks: Vec<usize> = floats(&folder.join(&pick_file), (visit * tokens + token) * selected, selected).iter().map(|v| v.to_bits() as usize).collect();

        // Router: this reference's logits against the trace, then the pick and the weights from the trace's logits.
        let gate = tensor_f32(&file, &format!("{base}.gate.weight"), 0, experts * hidden);
        let bias = tensor_f32(&file, &format!("{base}.gate.bias"), 0, experts);
        let logits = routing.logits(&gate, &z).unwrap();
        let logit_gap = logits.iter().zip(&ref_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let route = routing.route(&ref_logits, &bias).unwrap();
        let own = routing.route(&logits, &bias).unwrap();
        let (mut mine, mut theirs, mut from_own) = (route.ids.clone(), ref_picks.clone(), own.ids.clone());
        mine.sort_unstable(); theirs.sort_unstable(); from_own.sort_unstable();
        let weight_gap = route.ids.iter().zip(&route.weights).map(|(id, w)| ref_picks.iter().position(|p| p == id)
            .map_or(f32::INFINITY, |slot| (w - ref_weights[slot]).abs())).fold(0.0f32, f32::max);

        // Expert block: stored experts through the seam, the picks and weights of this reference, its fold order.
        let mut outputs = Vec::new();
        for &id in &route.ids {
            let (mut hold13, mut hold2) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
            let shape13 = stored(&file, &format!("{base}.experts.w13_weight"), id, &mut hold13);
            let shape2 = stored(&file, &format!("{base}.experts.w2_weight"), id, &mut hold2);
            let expert = Expert::from_stored(hidden, shape2.1, &matrix(shape13, &hold13), &matrix(shape2, &hold2)).unwrap();
            outputs.push(expert.run(&z, activation).unwrap());
        }
        let got = fold(&route.weights, &outputs).unwrap();
        let equal = got.iter().zip(&want).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let gap = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale;
        println!("CASE visit {visit} layer {layer} token {token}: logit gap {logit_gap:e}; picks equal {} (own logits {}); weight gap {weight_gap:e}; block equal {equal}/{hidden}; largest gap {gap:e} of the largest output",
            mine == theirs, from_own == theirs);
        equal_all += equal; total += hidden; picks_equal += usize::from(mine == theirs); worst = worst.max(gap);
    }
    let cases = range.1 - range.0 + 1;
    println!("SUMMARY visits {cases} token {token}: picks equal {picks_equal}/{cases}; block values equal {equal_all}/{total}; largest gap {worst:e}");
    assert_eq!(picks_equal, cases, "a picked set differs from the trace");
    assert!(equal_all * 10 >= total * 9, "fewer than 9 in 10 block outputs equal the trace");
}
