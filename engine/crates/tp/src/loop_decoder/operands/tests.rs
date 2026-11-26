//! Host tests of the weights by name and of the load, on a made-up checkpoint folder with a payload.
//! The folder takes the config, the storage sidecar and the header of the tensor-index fixture of the model crate.
//! Its payload is made here, so each expected byte is worked out in this file.
use super::{*, load::{self, Store, UNIT_BYTES}};
use engine_model::{flash::nvfp4::f32_to_bf16, json, plan::{inkling_index::{StoredRole, Stream}, loop_load::{self, LoadForm, PartKind}, loop_widen}, shapes::{Place, Proj}};
use std::{collections::BTreeSet, path::PathBuf, sync::atomic::{AtomicU64, Ordering}};

const CONFIG: &str = include_str!("../../../../model/tests/fixtures/inkling-index/config.json");
const SIDECAR: &str = include_str!("../../../../model/tests/fixtures/inkling-index/hf_quant_config.json");
const HEADER: &str = include_str!("../../../../model/tests/fixtures/inkling-index/header.json");
const SHARD: &str = "model-00001-of-00001.safetensors";
const P: &str = "model.llm.";
const DIVISOR: f32 = 2688.0;

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Folder(PathBuf);
impl Drop for Folder { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

/// A value of the payload at byte `at` of a tensor, by what the tensor is. Each value is valid for its type
/// and differs from byte to byte, so a read at a wrong offset gives other bytes.
fn payload_byte(name: &str, dtype: &str, at: usize) -> u8 {
    let mix = (at.wrapping_mul(2_654_435_761) >> 7) as u8 ^ name.len() as u8;
    match (dtype, name.ends_with("_scale")) {
        // Block scale of a 4-bit stack: an 8-bit float from 0.25 to under 4.
        ("F8_E4M3", true) => 0x28 + mix % 0x20,
        // Block scale of an 8-bit stack: a power of two from 2^-16 to 2^-9.
        ("U8", true) => 111 + mix % 8,
        // An 8-bit weight: every pattern but the two that are not numbers.
        ("F8_E4M3", false) => if mix & 0x7f == 0x7f { mix ^ 1 } else { mix },
        _ => mix,
    }
}
fn payload(name: &str, dtype: &str, bytes: usize) -> Vec<u8> {
    if name.ends_with("_outer_scale_denominator") { return DIVISOR.to_le_bytes().to_vec(); }
    // The scale of one expert: a positive 32-bit value that is not a BF16 value.
    if name.ends_with("_amax") { return (0..bytes / 4).flat_map(|i| (0.1f32 + 0.0537 * i as f32).to_le_bytes()).collect(); }
    (0..bytes).map(|at| payload_byte(name, dtype, at)).collect()
}
/// (name, type, start, end) of every tensor of the fixture header.
fn stored() -> Vec<(String, String, usize, usize)> {
    let doc = json::parse(HEADER.as_bytes(), json::Limits::default()).unwrap();
    doc.object().unwrap().iter().map(|(name, v)| { let o = v.object().unwrap(); let at = o["data_offsets"].array().unwrap();
        (name.clone(), o["dtype"].string().unwrap().to_owned(), at[0].u64().unwrap() as usize, at[1].u64().unwrap() as usize) }).collect()
}
/// The checkpoint folder and the bytes of its data section.
fn folder() -> (Folder, Vec<u8>) {
    let root = std::env::temp_dir().join(format!("vade-loop-operands-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&root).unwrap();
    let stored = stored();
    let mut data = vec![0u8; stored.iter().map(|t| t.3).max().unwrap()];
    for (name, dtype, start, end) in &stored { data[*start..*end].copy_from_slice(&payload(name, dtype, end - start)); }
    let mut file = (HEADER.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(HEADER.as_bytes()); file.extend_from_slice(&data);
    std::fs::write(root.join(SHARD), file).unwrap();
    std::fs::write(root.join("config.json"), CONFIG).unwrap();
    std::fs::write(root.join("hf_quant_config.json"), SIDECAR).unwrap();
    let map = stored.iter().map(|t| format!("{:?}:{SHARD:?}", t.0)).collect::<Vec<_>>().join(",");
    std::fs::write(root.join("model.safetensors.index.json"), format!(r#"{{"metadata":{{"total_size":{}}},"weight_map":{{{map}}}}}"#, data.len())).unwrap();
    (Folder(root), data)
}
fn slice<'a>(data: &'a [u8], name: &str) -> &'a [u8] {
    let (_, _, start, end) = stored().into_iter().find(|t| t.0 == name).unwrap_or_else(|| panic!("{name}")); &data[start..end]
}

/// Host bytes as the owner of the weights. `flip` changes one bit of one write, to prove that the read-back check can fail.
#[derive(Default)]
struct HostStore { created: usize, writes: usize, flip: Option<usize> }
impl Store for HostStore {
    type Value = Vec<u8>;
    fn create(&mut self, words: usize) -> Result<Vec<u8>> { self.created += 1; Ok(vec![0xa5; words * 4]) }
    fn write(&mut self, value: &mut Vec<u8>, word: usize, bytes: &[u8]) -> Result<()> {
        if !bytes.len().is_multiple_of(4) { return Err("test store: a write must hold whole words".into()); }
        value[word * 4..word * 4 + bytes.len()].copy_from_slice(bytes);
        if self.flip == Some(self.writes) { value[word * 4] ^= 1; }
        self.writes += 1; Ok(())
    }
    fn read(&mut self, value: &Vec<u8>, word: usize, out: &mut [u8]) -> Result<()> { out.copy_from_slice(&value[word * 4..word * 4 + out.len()]); Ok(()) }
}

/// One matrix of a widened stack, value by value, with no table: the second derivation of the widen step.
fn widened(data: &[u8], base: &str, expert: usize, rows: usize, columns: usize, four_bit: bool, unmix: bool) -> Vec<u8> {
    let (codes, scales) = (slice(data, base), slice(data, &format!("{base}_scale")));
    let factor = four_bit.then(|| {
        let amax = slice(data, &format!("{base}_amax"));
        let amax = f32::from_le_bytes(amax[expert * 4..expert * 4 + 4].try_into().unwrap());
        loop_widen::expert_factor(amax, f32::from_le_bytes(slice(data, &format!("{base}_outer_scale_denominator")).try_into().unwrap())).unwrap()
    });
    let (code_stride, scale_stride) = if four_bit { (columns / 2, columns / 16) } else { (columns, columns / 32) };
    let (codes, scales) = (&codes[expert * rows * code_stride..], &scales[expert * rows * scale_stride..]);
    let mut out = Vec::with_capacity(rows * columns * 2);
    for row in 0..rows {
        // Widened rows: every even stored row (the gate rows), then every odd stored row (the up rows).
        let from = if !unmix { row } else if row < rows / 2 { 2 * row } else { 2 * (row - rows / 2) + 1 };
        for column in 0..columns {
            let value = match factor {
                Some(factor) => { let byte = codes[from * code_stride + column / 2];
                    loop_widen::nvfp4_weight(if column % 2 == 0 { byte & 15 } else { byte >> 4 }, scales[from * scale_stride + column / 16], factor) }
                None => loop_widen::mxfp8_weight(codes[from * code_stride + column], scales[from * scale_stride + column / 32]),
            };
            out.extend_from_slice(&f32_to_bf16(value).to_le_bytes());
        }
    }
    out
}

#[test]
fn every_weight_of_every_visit_resolves_and_each_tensor_has_one_place() {
    let (folder, _) = folder();
    let checkpoint = loop_load::read_folder(&folder.0).unwrap();
    let d = &checkpoint.description;
    let operands = Operands::new(&checkpoint.load, |_| Ok(())).unwrap();
    assert_eq!(operands.len(), checkpoint.load.tensors.len());
    // Every visit of the run order finds every weight of its layer.
    assert_eq!(d.visits.len(), 16);
    for visit in &d.visits {
        let weights = operands.visit(d, visit).unwrap();
        let layer = &d.layers[visit.layer as usize];
        assert_eq!(weights.layer as u64, visit.layer);
        assert_eq!(weights.gates.is_some(), layer.message_gates);
        assert_eq!(matches!(weights.input_norms, InputNorms::Separate { .. }), layer.attention.separate_inputs);
        assert_eq!(matches!(weights.key_conv, ConvOperands::Token { .. }), layer.attention.window.is_none());
        assert_eq!(weights.mix_attention.is_some(), visit.layer > 0);
        if let Some(mix) = weights.mix_attention { assert_eq!(mix.tensor.shape.first().copied().filter(|_| mix.tensor.shape.len() == 2).unwrap_or(1), layer.attention_queries); }
        assert_eq!(weights.gate_up.tensor.stack().unwrap(), [7, 128, 96]);
        assert_eq!(weights.down.tensor.stack().unwrap(), [7, 96, 64]);
    }
    // The weights of the model and of the 8 layers are all the tensors of the load plan, each one time.
    let model = operands.model(d).unwrap();
    let mut seen = [model.embedding, model.embedding_norm, model.head, model.final_norm, model.mix_output, model.message_queries.unwrap(), model.message_kv_queries.unwrap()]
        .map(|r| r.tensor.name.clone()).to_vec();
    for layer in 0..d.layers.len() as u64 { seen.extend(operands.layer(d, layer).unwrap().all().into_iter().map(|r| r.tensor.name.clone())); }
    assert_eq!(seen.len(), operands.len());
    assert_eq!(seen.iter().cloned().collect::<BTreeSet<_>>(), checkpoint.load.tensors.keys().cloned().collect::<BTreeSet<_>>());
    // A role resolves to its stored name. The block-mix query of layer 3 is stored under index 2.
    assert_eq!(operands.layer(d, 3).unwrap().mix_attention.unwrap().tensor.name, format!("{P}attn_res.attn_queries.2"));
    assert_eq!(operands.weight(Place::Layer(5), StoredRole::ConvStatic(Stream::Value)).unwrap().tensor.name, format!("{P}layers.5.attn.v_sconv.static_w"));
    assert_eq!(model.head.tensor.form, LoadForm::LeadingRows);
    assert_eq!(model.head.words().unwrap(), 300 * 96 * 2 / 4);
    assert_eq!(operands.named(&format!("{P}embed.weight")).unwrap().tensor.name, model.embedding.tensor.name);
}

#[test]
fn a_missing_weight_is_a_refusal_that_names_the_layer_and_the_role() {
    let (folder, _) = folder();
    let checkpoint = loop_load::read_folder(&folder.0).unwrap();
    let d = &checkpoint.description;
    let without = |name: String| { let mut plan = checkpoint.load.clone(); assert!(plan.tensors.remove(&name).is_some(), "{name}"); Operands::new(&plan, |_| Ok(())).unwrap() };
    let operands = without(format!("{P}layers.4.mlp.gate.bias"));
    assert_eq!(operands.layer(d, 4).unwrap_err(), "loop operands: layer 4 has no RouterBias weight: the checkpoint does not hold it");
    assert_eq!(operands.visit(d, &d.visits[4]).unwrap_err(), "loop operands: layer 4 has no RouterBias weight: the checkpoint does not hold it");
    assert!(operands.layer(d, 3).is_ok());
    let operands = without(format!("{P}layers.2.controllers.k.gain_weight"));
    assert_eq!(operands.layer(d, 2).unwrap_err(), "loop operands: layer 2 has no Gate { gate: Key, part: Gain } weight: the checkpoint does not hold it");
    // Without the widened stack the packed twin still holds the experts: a product that asks for the widened matrix
    // is told so. Without both, the weight is missing.
    let operands = without(format!("{P}layers.6.mlp.experts.w2_weight"));
    assert_eq!(operands.expert(6, Proj::Down, 0).unwrap_err(), "loop operands: layer 6 holds its Down experts packed only (the widened stack is not loaded): this product reads the packed stack, and the compute gave no packed product for it");
    let mut plan = checkpoint.load.clone();
    for name in [format!("{P}layers.6.mlp.experts.w2_weight"), format!("{P}layers.6.mlp.experts.w2_weight{}", loop_load::PACKED_NAME)] { assert!(plan.tensors.remove(&name).is_some(), "{name}"); }
    let operands = Operands::new(&plan, |_| Ok(())).unwrap();
    assert_eq!(operands.expert(6, Proj::Down, 0).unwrap_err(), "loop operands: layer 6 has no Expert { proj: Down, part: Codes } weight: the checkpoint does not hold it");
    let operands = without(format!("{P}unembed.weight"));
    assert_eq!(operands.model(d).unwrap_err(), "loop operands: the model has no Head weight: the checkpoint does not hold it");
    let operands = Operands::new(&checkpoint.load, |_| Ok(())).unwrap();
    assert_eq!(operands.named("model.llm.absent").unwrap_err(), "loop operands: the checkpoint holds no weight named model.llm.absent");
    assert_eq!(operands.layer(d, 8).unwrap_err(), "loop operands: the model has no layer 8");
    assert_eq!(operands.expert(0, Proj::GateUp, 7).unwrap_err(), "loop operands: layer 0 has 7 experts and no expert 7");
    // A failed value stops the build of the store.
    assert_eq!(Operands::<()>::new(&checkpoint.load, |_| Err("no room".into())).unwrap_err(), "no room");
}

#[test]
fn a_load_gives_the_file_bytes_and_the_widened_experts() {
    let (folder, data) = folder();
    let checkpoint = loop_load::read_folder(&folder.0).unwrap();
    let d = &checkpoint.description;
    let mut store = HostStore::default();
    let loaded = load::load(&checkpoint, &folder.0, &mut store, 3).unwrap();
    let operands = &loaded.operands;
    assert_eq!((store.created, loaded.report.tensors), (checkpoint.load.tensors.len(), checkpoint.load.tensors.len()));
    assert_eq!((loaded.report.device_bytes, loaded.report.file_bytes), (checkpoint.load.bytes.device, checkpoint.load.bytes.file));
    for resident in operands.all() {
        let t = &resident.tensor;
        assert_eq!(resident.value.len() as u64, t.device_bytes().unwrap(), "{}", t.name);
        match t.form {
            // The stored bytes, unchanged.
            LoadForm::Stored => assert_eq!(resident.value, slice(&data, &t.name), "{}", t.name),
            // The first rows of the head, and nothing of the padded rows.
            LoadForm::LeadingRows => assert_eq!(resident.value, slice(&data, &t.name)[..300 * 96 * 2]),
            // The widened experts are compared below, and the packed units in `operands/packed_tests.rs`.
            LoadForm::WidenedExperts { .. } | LoadForm::PackedExperts { .. } => {}
        }
    }
    // Every expert of every stack, against the value-by-value form.
    let mut matrices = 0;
    for layer in 0..d.layers.len() as u32 {
        for (proj, stem, rows, columns) in [(Proj::GateUp, "w13", 128, 96), (Proj::Down, "w2", 96, 64)] {
            for expert in 0..7 {
                let part = operands.expert(layer, proj, expert).unwrap();
                assert_eq!((part.rows, part.columns, part.words.len()), (rows as u64, columns as u64, rows * columns / 2));
                let want = widened(&data, &format!("{P}layers.{layer}.mlp.experts.{stem}_weight"), expert as usize, rows, columns, layer < 6, proj == Proj::GateUp);
                assert_eq!(&part.value[part.words.start * 4..part.words.end * 4], want.as_slice(), "layer {layer} {stem} expert {expert}");
                matrices += 1;
            }
        }
    }
    assert_eq!(matrices, 8 * 2 * 7);
    // The widened values are not all zero and the experts of a stack differ.
    let (first, second) = (operands.expert(0, Proj::GateUp, 0).unwrap(), operands.expert(0, Proj::GateUp, 1).unwrap());
    assert_ne!(first.value[first.words.start * 4..first.words.end * 4], second.value[second.words.start * 4..second.words.end * 4]);
    // Units: one for each matrix of a stack, widened and packed, and one for each tensor that is smaller than a unit.
    let plain = checkpoint.load.tensors.values().filter(|t| t.stack().is_none()).count();
    assert!(checkpoint.load.tensors.values().all(|t| t.bytes as usize <= UNIT_BYTES));
    assert_eq!((loaded.units.len(), loaded.report.units, store.writes), (plain + 2 * matrices, plain + 2 * matrices, plain + 2 * matrices));
    assert_eq!(loaded.units.iter().map(|u| u.bytes as u64).sum::<u64>(), checkpoint.load.bytes.device);
    for unit in &loaded.units {
        let resident = operands.named(&unit.name).unwrap();
        assert_eq!(unit.sha256, loop_load::digest(&resident.value[unit.word * 4..unit.word * 4 + unit.bytes]), "{}", unit.name);
        assert_eq!(unit.matrix.is_some(), resident.tensor.part(PartKind::Codes).is_some());
    }
    // The read-back check passes, and two loads give the same hashes.
    let verified = load::verify(&loaded, &mut store, 2).unwrap();
    assert_eq!((verified.units, verified.bytes, verified.different.len()), (loaded.units.len(), checkpoint.load.bytes.device, 0));
    let again = load::load(&checkpoint, &folder.0, &mut HostStore::default(), 1).unwrap();
    assert_eq!((again.fingerprint(), &again.units), (loaded.fingerprint(), &loaded.units));
    assert_eq!(load::unit_lines(&loaded.units).lines().count(), loaded.units.len());
}

#[test]
fn a_changed_byte_on_the_owner_fails_the_read_back_check() {
    let (folder, _) = folder();
    let checkpoint = loop_load::read_folder(&folder.0).unwrap();
    let mut store = HostStore { flip: Some(40), ..Default::default() };
    let loaded = load::load(&checkpoint, &folder.0, &mut store, 1).unwrap();
    let verified = load::verify(&loaded, &mut store, 2).unwrap();
    // One worker made the units in plan order, so write 40 is unit 40.
    let unit = &loaded.units[40];
    assert_eq!(verified.different, [format!("{} matrix {} word {}", unit.name, unit.matrix.map_or("-".into(), |m| m.to_string()), unit.word)]);
}

#[test]
fn a_bad_file_or_a_bad_scale_stops_the_load_with_its_name() {
    let (folder, data) = folder();
    let checkpoint = loop_load::read_folder(&folder.0).unwrap();
    // A file that ends in the middle of its data: the read of a text tensor fails and names the file.
    let path = folder.0.join(SHARD);
    let whole = std::fs::read(&path).unwrap();
    std::fs::write(&path, &whole[..whole.len() - data.len() / 2]).unwrap();
    let error = load::load(&checkpoint, &folder.0, &mut HostStore::default(), 2).unwrap_err();
    assert!(error.starts_with("loop load: read of ") && error.contains(SHARD), "{error}");
    // A block scale that is not a number: the widen step refuses it and the error names the stack and the matrix.
    let name = format!("{P}layers.1.mlp.experts.w2_weight_scale");
    let (_, _, start, _) = stored().into_iter().find(|t| t.0 == name).unwrap();
    let mut bad = whole.clone();
    let at = whole.len() - data.len() + start + 2 * 96 * 4;
    bad[at] = 0x7f;
    std::fs::write(&path, bad).unwrap();
    let error = load::load(&checkpoint, &folder.0, &mut HostStore::default(), 2).unwrap_err();
    assert_eq!(error, format!("loop load: {P}layers.1.mlp.experts.w2_weight matrix 2: expert widen: a 4-bit block scale gives a weight that is not finite"));
}
