//! The loaded weights against the made-up checkpoints of the driver tests. The stored tensors of a fixture are
//! written as a checkpoint folder and loaded into host bytes. Then every weight name must give the words, or
//! the product, that the fixture gives for it: the fixture reads the same stored bytes by its own code.
use super::*;
use crate::loop_decoder::{fixture::Fixture, mix::support::Stand, operands::load::{self, Store}};
use engine_model::{flash::nvfp4::bf16_to_f32, plan::loop_load::LoopLoadPlan, safetensors::{Index, Limits},
    shapes::{Ffn, HEADER_BYTES, MAX_SHARDS, MAX_TENSORS, MAX_TENSOR_RANK}};
use std::{path::PathBuf, sync::atomic::{AtomicU64, Ordering}};

const SHARD: &str = "model-00001-of-00001.safetensors";
static NEXT: AtomicU64 = AtomicU64::new(0);
pub(super) struct Folder(pub(super) PathBuf);
impl Drop for Folder { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

/// Host bytes as the owner of the weights.
struct Bytes;
impl Store for Bytes {
    type Value = Vec<u8>;
    fn create(&mut self, words: usize) -> Result<Vec<u8>> { Ok(vec![0xa5; words * 4]) }
    fn write(&mut self, value: &mut Vec<u8>, word: usize, bytes: &[u8]) -> Result<()> { value[word * 4..word * 4 + bytes.len()].copy_from_slice(bytes); Ok(()) }
    fn read(&mut self, value: &Vec<u8>, word: usize, out: &mut [u8]) -> Result<()> { out.copy_from_slice(&value[word * 4..word * 4 + out.len()]); Ok(()) }
}
/// The product on host bytes: the BF16 values of the word range, widened, through the plain chain of the compute.
impl ResidentProduct<Vec<u8>> for Stand {
    fn resident_product(&self, weight: &Vec<u8>, words: Range<usize>, rows: usize, columns: usize, x: &Vec<f32>) -> Result<Vec<f32>> {
        let matrix = weight[words.start * 4..words.end * 4].as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect::<Vec<_>>();
        assert_eq!(matrix.len(), rows * columns);
        chain_product(self, &matrix, x)
    }
}
/// The stored tensors of the fixture as one weight file with its index, and the checkpoint that the load reads.
pub(super) fn folder(fixture: &Fixture) -> (Folder, LoopCheckpoint) {
    let root = std::env::temp_dir().join(format!("vade-loop-weights-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir(&root).unwrap();
    let (mut rows, mut data, mut map) = (Vec::new(), Vec::new(), Vec::new());
    for name in fixture.names() {
        let t = fixture.stored(name).unwrap();
        let dtype = match t.dtype { Dtype::Bf16 => "BF16", Dtype::F32 => "F32", Dtype::U8 => "U8", Dtype::F8E4M3 => "F8_E4M3", other => panic!("{other:?}") };
        rows.push(format!("{name:?}:{{\"dtype\":{dtype:?},\"shape\":{:?},\"data_offsets\":[{},{}]}}", t.shape, data.len(), data.len() + t.bytes.len()));
        data.extend_from_slice(&t.bytes);
        map.push(format!("{name:?}:{SHARD:?}"));
    }
    let mut header = format!("{{{}}}", rows.join(","));
    while !header.len().is_multiple_of(8) { header.push(' '); }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes()); file.extend_from_slice(&data);
    std::fs::write(root.join(SHARD), file).unwrap();
    let limits = Limits { header_bytes: HEADER_BYTES, tensors: MAX_TENSORS, rank: MAX_TENSOR_RANK, shards: MAX_SHARDS };
    let index = Index::parse(format!(r#"{{"metadata":{{"total_size":{}}},"weight_map":{{{}}}}}"#, data.len(), map.join(",")).as_bytes(), limits).unwrap();
    let shards = index.inspect_shards(&root, limits).unwrap();
    let bound = fixture.index.bind(&shards[SHARD].tensors, None).unwrap();
    let load = LoopLoadPlan::new(&fixture.description, &bound, &shards).unwrap();
    (Folder(root), LoopCheckpoint { description: fixture.description.clone(), bound, shards, load })
}
pub(super) fn bits(words: &[f32]) -> Vec<u32> { words.iter().map(|v| v.to_bits()).collect() }
/// Columns of a weight that is read through a product; 0 for a weight that is read as words.
fn columns(weight: Weight, hidden: usize, words: usize) -> usize {
    match weight {
        Weight::Head | Weight::Query(_) | Weight::Key(_) | Weight::Value(_) | Weight::Relative(_) | Weight::Sink(_) | Weight::OutputGate(_) | Weight::Router(_)
            | Weight::KeyTapProjection(_) | Weight::ValueTapProjection(_) | Weight::GateUp(..) => hidden,
        Weight::GateDown(..) => 2 * hidden,
        Weight::Output(_) | Weight::GateForget(..) | Weight::GateGain(..) | Weight::Down(..) => words / hidden,
        _ => 0,
    }
}
/// (weights read as words, products) that were compared.
fn equal_to_the_fixture(fixture: &Fixture) -> (usize, usize) {
    let (folder, checkpoint) = folder(fixture);
    let loaded = load::load(&checkpoint, &folder.0, &mut Bytes, 2).unwrap();
    let files = Files::open(&checkpoint, &folder.0).unwrap();
    let weights = LoadedWeights::new(&checkpoint, &files, &loaded.operands).unwrap();
    let (c, d, hidden) = (Stand::new(), &fixture.description, fixture.description.hidden as usize);
    let mut names = weights.names().collect::<Vec<_>>();
    for (i, layer) in d.layers.iter().enumerate() {
        let Ffn::Experts(e) = layer.ffn else { panic!("a fixture layer has routed experts") };
        for expert in 0..e.count as usize { names.extend([(Weight::GateUp(i, expert), false), (Weight::Down(i, expert), false)]); }
    }
    let (mut read, mut products) = (0, 0);
    for (weight, held_on_host) in names {
        let expected = <Fixture as Weights<Stand>>::words(fixture, weight).unwrap_or_else(|e| panic!("{weight:?}: {e}"));
        let columns = columns(weight, hidden, expected.len());
        if columns == 0 {
            assert!(held_on_host, "{weight:?}");
            assert_eq!(bits(<LoadedWeights<'_, Vec<u8>> as Weights<Stand>>::words(&weights, weight).unwrap()), bits(expected), "{weight:?}");
            read += 1;
            continue;
        }
        if !held_on_host { assert!(<LoadedWeights<'_, Vec<u8>> as Weights<Stand>>::words(&weights, weight).unwrap_err().contains("read through a product"), "{weight:?}"); }
        // An input with BF16 values of both signs, another for each weight.
        let x = (0..columns).map(|k| ((k * 37 + expected.len()) % 23) as f32 * 0.0625 - 0.6875).collect::<Vec<_>>();
        let got = weights.product(&c, weight, &x).unwrap_or_else(|e| panic!("{weight:?}: {e}"));
        assert_eq!(bits(&got), bits(&<Fixture as Weights<Stand>>::product(fixture, &c, weight, &x).unwrap()), "{weight:?}");
        assert_eq!(got.len(), expected.len() / columns, "{weight:?}");
        // Many input rows: the words of one product a row, in row order. One row is the product itself.
        let rows = (0..3 * columns).map(|k| ((k * 29 + expected.len()) % 31) as f32 * 0.0625 - 0.9375).collect::<Vec<_>>();
        let by_row = rows.chunks(columns).flat_map(|row| weights.product(&c, weight, &row.to_vec()).unwrap()).collect::<Vec<_>>();
        assert_eq!(bits(&weights.product_rows(&c, weight, &rows, 3).unwrap()), bits(&by_row), "{weight:?}");
        assert_eq!(bits(&weights.product_rows(&c, weight, &x, 1).unwrap()), bits(&got), "{weight:?}");
        assert!(weights.product_rows(&c, weight, &rows, 0).is_err() && weights.product_rows(&c, weight, &rows[1..].to_vec(), 3).is_err(), "{weight:?}");
        products += 1;
    }
    assert!(<LoadedWeights<'_, Vec<u8>> as Weights<Stand>>::words(&weights, Weight::MixAttention(0)).unwrap_err().contains("holds no weight"));
    assert!(weights.host_bytes() > 0);
    (read, products)
}
#[test]
fn every_weight_of_the_small_fixture_equals_the_fixture() {
    let (read, products) = equal_to_the_fixture(&Fixture::small());
    assert!(read > 0 && products > 0, "{read} {products}");
}
#[test]
fn every_weight_of_the_wide_fixture_equals_the_fixture() {
    let (read, products) = equal_to_the_fixture(&Fixture::wide());
    assert!(read > 0 && products > 0, "{read} {products}");
}
/// Host bytes that one value can share with views of its words: the owner of the weights with stacks.
struct Shared;
#[derive(Debug)]
struct Part { bytes: std::rc::Rc<std::cell::RefCell<Vec<u8>>>, first: usize, words: usize }
impl Store for Shared {
    type Value = Part;
    fn create(&mut self, words: usize) -> Result<Part> { Ok(Part { bytes: std::rc::Rc::new(std::cell::RefCell::new(vec![0xa5; words * 4])), first: 0, words }) }
    fn write(&mut self, value: &mut Part, word: usize, bytes: &[u8]) -> Result<()> {
        assert!(word * 4 + bytes.len() <= value.words * 4);
        let at = (value.first + word) * 4;value.bytes.borrow_mut()[at..at + bytes.len()].copy_from_slice(bytes); Ok(())
    }
    fn read(&mut self, value: &Part, word: usize, out: &mut [u8]) -> Result<()> { let at = (value.first + word) * 4; out.copy_from_slice(&value.bytes.borrow()[at..at + out.len()]); Ok(()) }
    fn views(&self) -> bool { true }
    fn view(&mut self, value: &Part, words: Range<usize>) -> Result<Part> {
        assert!(words.end <= value.words);
        Ok(Part { bytes: value.bytes.clone(), first: value.first + words.start, words: words.len() })
    }
}
impl ResidentProduct<Part> for Stand {
    fn resident_product(&self, weight: &Part, words: Range<usize>, rows: usize, columns: usize, x: &Vec<f32>) -> Result<Vec<f32>> {
        assert!(words.end <= weight.words);
        let bytes = weight.bytes.borrow();
        let matrix = bytes[(weight.first + words.start) * 4..(weight.first + words.end) * 4].as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect::<Vec<_>>();
        assert_eq!(matrix.len(), rows * columns);
        chain_product(self, &matrix, x)
    }
}
/// The load with stacks gives the units of the load without them, and the products of a stack, run as one product,
/// give the words of the fixture for each weight. (layers with a stack, products compared)
fn stacked_equal_to_the_fixture(fixture: &Fixture) -> (usize, usize) {
    let (folder, checkpoint) = folder(fixture);
    let plain = load::load(&checkpoint, &folder.0, &mut Bytes, 2).unwrap();
    let loaded = load::load(&checkpoint, &folder.0, &mut Shared, 2).unwrap();
    assert_eq!(loaded.fingerprint(), plain.fingerprint());
    assert_eq!(loaded.report.device_bytes, plain.report.device_bytes);
    let files = Files::open(&checkpoint, &folder.0).unwrap();
    let weights = LoadedWeights::new(&checkpoint, &files, &loaded.operands).unwrap();
    let (c, hidden) = (Stand::new(), fixture.description.hidden as usize);
    let x = (0..hidden).map(|k| ((k * 29 + 5) % 19) as f32 * 0.0625 - 0.5625).collect::<Vec<_>>();
    let mut compared = 0;
    let (mut taps, mut gates) = (0, 0);
    for i in 0..fixture.description.layers.len() {
        let (q, k, v) = ([Weight::Query(i), Weight::Relative(i), Weight::Sink(i), Weight::OutputGate(i)], [Weight::Key(i), Weight::KeyTapProjection(i)], [Weight::Value(i), Weight::ValueTapProjection(i)]);
        let stacked = weights.stacked.contains_key(&q[0]);
        let made = weights.stacked.contains_key(&k[1]);
        taps += usize::from(made);
        let six = [q[0], q[1], q[2], q[3], k[0], v[0]];
        // Groups that one product runs where the layer has a stack, and groups that are no run of a stack.
        let mut groups = vec![(&q[..], stacked), (&q[..2], stacked), (&six[..], stacked && !made)];
        if made { groups.extend([(&k[..], true), (&v[..], true)]); }
        for (group, run) in groups {
            assert_eq!(weights.run(group).unwrap().is_some(), run, "layer {i} {group:?}");
            let got = <LoadedWeights<'_, Part> as Weights<Stand>>::stack_products(&weights, &c, group, &x).unwrap();
            assert_eq!(got.len(), group.len());
            for (weight, got) in group.iter().zip(&got) {
                assert_eq!(bits(got), bits(&<Fixture as Weights<Stand>>::product(fixture, &c, *weight, &x).unwrap()), "{weight:?}");
                compared += 1;
            }
        }
        // Not one after another in this order: one product a weight.
        assert!(weights.run(&[q[1], q[0]]).unwrap().is_none() && weights.run(&[q[0], q[2]]).unwrap().is_none());
        // The forget and the gain matrix of each message gate: one product on an inner vector.
        for message in [Message::Query, Message::Key, Message::Value, Message::Expert] {
            let pair = [Weight::GateForget(i, message), Weight::GateGain(i, message)];
            if !weights.stacked.contains_key(&pair[0]) { continue; }
            gates += 1;
            let rank = weights.product_shape(pair[0]).unwrap().unwrap().1;
            let inner = (0..rank).map(|k| ((k * 13 + 3) % 11) as f32 * 0.125 - 0.625).collect::<Vec<_>>();
            assert!(weights.run(&pair).unwrap().is_some() && weights.run(&[pair[1], pair[0]]).unwrap().is_none());
            let got = <LoadedWeights<'_, Part> as Weights<Stand>>::stack_products(&weights, &c, &pair, &inner).unwrap();
            for (weight, got) in pair.iter().zip(&got) {
                assert_eq!(bits(got), bits(&<Fixture as Weights<Stand>>::product(fixture, &c, *weight, &inner).unwrap()), "{weight:?}");
                compared += 1;
            }
            // Many rows: one product gives each row's forget and gain side by side, the words of one product each.
            for rows in [2usize, 3, 7] {
                let x = (0..rows * rank).map(|k| ((k * 7 + 5) % 13) as f32 * 0.0625 - 0.375).collect::<Vec<_>>();
                let got = <LoadedWeights<'_, Part> as Weights<Stand>>::stack_product_rows(&weights, &c, &pair, &x, rows).unwrap().unwrap();
                let want = (0..rows).flat_map(|r| pair.iter().flat_map(|&weight| <Fixture as Weights<Stand>>::product(fixture, &c, weight, &x[r * rank..(r + 1) * rank].to_vec()).unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "{pair:?} rows {rows}");
                assert!(<LoadedWeights<'_, Part> as Weights<Stand>>::stack_product_rows(&weights, &c, &pair, &x[1..].to_vec(), rows).is_err());
                compared += 1;
            }
        }
    }
    assert!(taps > 0 && gates > 0, "a fixture layer makes taps from the token, and one reads messages: {taps} {gates}");
    (loaded.operands.stacks().len(), compared)
}
#[test]
fn stacked_products_of_the_small_fixture_equal_the_fixture() {
    let (stacks, compared) = stacked_equal_to_the_fixture(&Fixture::small());
    assert!(stacks > 0 && compared > 0, "{stacks} {compared}");
}
#[test]
fn stacked_products_of_the_wide_fixture_equal_the_fixture() {
    let (stacks, compared) = stacked_equal_to_the_fixture(&Fixture::wide());
    assert!(stacks > 0 && compared > 0, "{stacks} {compared}");
}
