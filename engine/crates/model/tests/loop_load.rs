//! Load plan, state plan and byte ledger of the looped text family, on the made-up fixture of the tensor index.
//! The expected numbers are worked out here from the fixture's config values and header, not from the code under test.
use engine_model::{json, plan::{description::LoopTextDescription, inkling_index::{BoundIndex, IndexInput},
    loop_load::{self, ExpertCodes, LoadForm, LoopLoadPlan, PartKind, device_extent},
    loop_state::{ConvRing, LoopLedger, LoopStatePlan, context_limit}, storage_groups::{ExpertRows, StorageGroups}},
    safetensors::{Dtype, Header, Limits}, shapes::Proj};
use std::{collections::BTreeMap, io::{self, Cursor, Read, Seek, SeekFrom}, path::PathBuf};

const CONFIG: &str = include_str!("fixtures/inkling-index/config.json");
const SIDECAR: &str = include_str!("fixtures/inkling-index/hf_quant_config.json");
const HEADER: &str = include_str!("fixtures/inkling-index/header.json");
const SHARD: &str = "model-00001-of-00001.safetensors";
const P: &str = "model.llm.";

// Sizes of the fixture config, written out. A window layer and a full layer have different sizes.
const HIDDEN: u64 = 96;
const HEAD_ROWS: u64 = 300;
const STORED_ROWS: u64 = 320;
const EXPERTS: u64 = 7;
const WIDTH: u64 = 64;
const LAYERS: u64 = 8;
const KERNEL: u64 = 5;
const WINDOW: u64 = 36;
const WINDOW_CHANNELS: u64 = 2 * 14;
const FULL_CHANNELS: u64 = 3 * 22;
const MAX_CONTEXT: u64 = 4096;
/// Run order of the fixture: layers 0 and 1, then layers 2 to 5 three times, then layers 6 and 7.
const VISITS: [(u64, u64); 16] = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 0), (7, 0)];
const BUCKET: u64 = 256;

// Only the length prefix and the header exist. A payload read fails the test.
struct HeaderOnly { metadata: Cursor<Vec<u8>>, file_bytes: u64 }
impl Read for HeaderOnly {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.metadata.position() >= self.metadata.get_ref().len() as u64 { return Err(io::Error::other("test attempted to read tensor payload")); }
        self.metadata.read(out)
    }
}
impl Seek for HeaderOnly {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from { SeekFrom::Start(n) => n as i128, SeekFrom::End(n) => self.file_bytes as i128 + n as i128,
            SeekFrom::Current(n) => self.metadata.position() as i128 + n as i128 };
        let position = u64::try_from(position).map_err(|_| io::Error::other("invalid test seek"))?;
        self.metadata.set_position(position); Ok(position)
    }
}
/// (name, type, shape, start, end) of every tensor of the fixture header, read with the JSON parser alone.
fn stored() -> Vec<(String, String, Vec<u64>, u64, u64)> {
    let doc = json::parse(HEADER.as_bytes(), json::Limits::default()).unwrap();
    doc.object().unwrap().iter().map(|(name, v)| {
        let o = v.object().unwrap(); let at = o["data_offsets"].array().unwrap();
        (name.clone(), o["dtype"].string().unwrap().to_owned(), o["shape"].array().unwrap().iter().map(|d| d.u64().unwrap()).collect(), at[0].u64().unwrap(), at[1].u64().unwrap())
    }).collect()
}
fn header() -> Header {
    let mut data = (HEADER.len() as u64).to_le_bytes().to_vec(); data.extend_from_slice(HEADER.as_bytes());
    let file_bytes = data.len() as u64 + stored().iter().map(|t| t.4).max().unwrap();
    Header::read(&mut HeaderOnly { metadata: Cursor::new(data), file_bytes }, Limits::default()).unwrap()
}
fn description() -> LoopTextDescription { LoopTextDescription::from_config(CONFIG.as_bytes()).unwrap() }
fn bound() -> BoundIndex {
    let groups = StorageGroups::from_json(SIDECAR.as_bytes()).unwrap();
    IndexInput::from_description(&description(), &groups).unwrap().bind(&header().tensors, None).unwrap()
}
fn plan() -> LoopLoadPlan { LoopLoadPlan::new(&description(), &bound(), &BTreeMap::from([(SHARD.to_owned(), header())])).unwrap() }
fn round4(n: u64) -> u64 { n.div_ceil(4) * 4 }

/// Device bytes of the weights by the names of the header: every text tensor that is not a part of an expert
/// stack and not the head keeps its bytes; the head keeps its used rows; each expert stack is BF16.
fn expected_weights() -> (u64, u64, u64) {
    let plain = stored().iter().filter(|t| t.0.starts_with(P) && !t.0.contains(".mlp.experts.") && t.0 != format!("{P}unembed.weight"))
        .map(|t| round4(t.4 - t.3)).sum::<u64>();
    let head = HEAD_ROWS * HIDDEN * 2;
    let experts = LAYERS * (EXPERTS * (2 * WIDTH) * HIDDEN * 2 + EXPERTS * HIDDEN * WIDTH * 2);
    (plain, head, experts)
}
/// Device bytes of the packed expert stacks: a unit an expert, each part of a unit on 16 bytes. The codes, the
/// block scales, and a table of 8,192 bytes in the 4-bit layers (0 to 5) and of 1,024 bytes in the 8-bit layers.
fn packed_plane_bytes(rows: u64, columns: u64, four_bit: bool) -> (u64, u64) {
    let rows = rows.next_multiple_of(32);
    let columns = columns.next_multiple_of(if four_bit { 64 } else { 128 });
    if four_bit { (rows * columns / 2, rows * columns / 16) }
    else { (rows * columns, rows * columns / 32) }
}

fn expected_packed() -> u64 {
    let unit = |rows: u64, columns: u64, four_bit: bool| {
        let (codes, scales) = packed_plane_bytes(rows, columns, four_bit);
        codes + scales + if four_bit { 8192 } else { 1024 }
    };
    EXPERTS * (6 * (unit(2 * WIDTH, HIDDEN, true) + unit(HIDDEN, WIDTH, true)) + 2 * (unit(2 * WIDTH, HIDDEN, false) + unit(HIDDEN, WIDTH, false)))
}
/// State bytes of one chat by the sizes of the config. `rows`: rows of a straight plane.
fn expected_state(rows: u64) -> (u64, u64, u64) {
    let (windows, fulls) = (VISITS.iter().filter(|v| v.0 % 2 == 0).count() as u64, VISITS.iter().filter(|v| v.0 % 2 == 1).count() as u64);
    let after = 2 * KERNEL * HIDDEN * 4;
    let fixed = windows * (2 * WINDOW * WINDOW_CHANNELS * 2 + 2 * KERNEL * WINDOW_CHANNELS * 4 + after) + fulls * (2 * KERNEL * FULL_CHANNELS * 4 + after);
    let planes = fulls * 2 * rows * FULL_CHANNELS * 2;
    // The stream and the restart point; one vector for each of the 4 blocks and one more; the 2 blocks of the loop; 4 messages for each of its 4 layers.
    let step = (2 + (LAYERS / 2 + 1) + 2 + 4 * 4) * HIDDEN * 4;
    (fixed, planes, step)
}

#[test]
fn every_text_tensor_has_one_device_form_and_one_file_range() {
    let (plan, header) = (plan(), header());
    let text = stored().into_iter().filter(|t| t.0.starts_with(P)).collect::<Vec<_>>();
    // Each text name is one whole part of one device tensor, at its own place in the file.
    let mut parts = BTreeMap::new();
    // A packed expert stack is made from the parts of its widened stack; it is counted apart, below.
    for t in plan.tensors.values().filter(|t| !t.form.is_packed()) { for p in &t.parts { assert!(parts.insert(p.name.clone(), (p, t)).is_none(), "{} is read twice", p.name); } }
    assert_eq!(parts.len(), text.len());
    for (name, dtype, shape, start, end) in &text {
        let (part, tensor) = parts[name];
        assert_eq!((part.shard.as_str(), part.offset, part.dtype), (SHARD, header.data_start + start, Dtype::parse(dtype).unwrap()), "{name}");
        if tensor.form == LoadForm::LeadingRows { assert_eq!((part.shape.clone(), part.bytes), (vec![HEAD_ROWS, HIDDEN], HEAD_ROWS * HIDDEN * 2)); }
        else { assert_eq!((&part.shape, part.bytes), (shape, end - start), "{name}"); }
    }
    // The image tower is not loaded.
    assert!(plan.tensors.keys().all(|name| name.starts_with(P)));
    assert_eq!(plan.bytes.unused, stored().iter().filter(|t| t.0.starts_with("model.visual.")).map(|t| t.4 - t.3).sum::<u64>());
    assert!(plan.bytes.unused > 0);
    // The head keeps its used rows. The embedding keeps every stored row.
    let head = plan.tensor(&format!("{P}unembed.weight")).unwrap();
    assert_eq!((head.form, head.shape.clone(), head.bytes, head.dtype), (LoadForm::LeadingRows, vec![HEAD_ROWS, HIDDEN], HEAD_ROWS * HIDDEN * 2, Dtype::Bf16));
    assert_eq!(plan.bytes.head_unused, (STORED_ROWS - HEAD_ROWS) * HIDDEN * 2);
    let embedding = plan.tensor(&format!("{P}embed.weight")).unwrap();
    assert_eq!((embedding.form, embedding.shape.clone()), (LoadForm::Stored, vec![STORED_ROWS, HIDDEN]));
    // A stored tensor keeps its type: the router bias is 32-bit and the router weight is BF16.
    assert_eq!(plan.tensor(&format!("{P}layers.3.mlp.gate.bias")).unwrap().dtype, Dtype::F32);
    assert_eq!(plan.tensor(&format!("{P}layers.3.mlp.gate.weight")).unwrap().dtype, Dtype::Bf16);
    let (plain, head_bytes, experts) = expected_weights();
    let packed = expected_packed();
    assert_eq!((plan.bytes.stored, plan.bytes.head, plan.bytes.experts, plan.bytes.packed, plan.bytes.device), (plain, head_bytes, experts, packed, plain + head_bytes + experts + packed));
    assert_eq!((plan.bytes.expert_stacks, plan.bytes.packed_stacks), (2 * LAYERS, 2 * LAYERS));
    assert_eq!(plan.bytes.file, text.iter().map(|t| t.4 - t.3).sum::<u64>() - plan.bytes.head_unused);
}

#[test]
fn expert_stacks_are_widened_and_gate_rows_are_unmixed() {
    let plan = plan();
    for layer in 0..LAYERS as u32 {
        // The fixture stores layers 0 to 5 in the 4-bit form and layers 6 and 7 in the 8-bit form.
        let codes = if layer < 6 { ExpertCodes::Nvfp4 } else { ExpertCodes::Mxfp8 };
        let gate_up = plan.experts(layer, Proj::GateUp).unwrap();
        assert_eq!(gate_up.name, format!("{P}layers.{layer}.mlp.experts.w13_weight"));
        assert_eq!(gate_up.form, LoadForm::WidenedExperts { codes, rows: Some(ExpertRows::Interleaved) });
        assert_eq!((gate_up.dtype, gate_up.stack().unwrap()), (Dtype::Bf16, [EXPERTS, 2 * WIDTH, HIDDEN]));
        assert_eq!((gate_up.bytes, gate_up.matrix_bytes().unwrap()), (EXPERTS * 2 * WIDTH * HIDDEN * 2, 2 * WIDTH * HIDDEN * 2));
        let down = plan.experts(layer, Proj::Down).unwrap();
        assert_eq!(down.form, LoadForm::WidenedExperts { codes, rows: None });
        assert_eq!(down.stack().unwrap(), [EXPERTS, HIDDEN, WIDTH]);
        for t in [gate_up, down] {
            let [_, rows, columns] = t.stack().unwrap();
            let kinds = t.parts.iter().map(|p| p.kind).collect::<Vec<_>>();
            let (code_bytes, scale_bytes) = if codes == ExpertCodes::Nvfp4 { (rows * columns / 2, rows * columns / 16) } else { (rows * columns, rows * columns / 32) };
            assert_eq!(kinds.len(), if codes == ExpertCodes::Nvfp4 { 4 } else { 2 });
            assert!(kinds.contains(&PartKind::Codes) && kinds.contains(&PartKind::BlockScale));
            // The parts of expert 3 start three matrices after the start of the part.
            let part = t.part(PartKind::Codes).unwrap();
            assert_eq!(t.matrix_part(PartKind::Codes, 3).unwrap(), (part.offset + 3 * code_bytes, code_bytes));
            assert_eq!(t.matrix_part(PartKind::BlockScale, 3).unwrap(), (t.part(PartKind::BlockScale).unwrap().offset + 3 * scale_bytes, scale_bytes));
            if codes == ExpertCodes::Nvfp4 {
                assert_eq!(t.matrix_part(PartKind::ExpertScale, 3).unwrap(), (t.part(PartKind::ExpertScale).unwrap().offset + 12, 4));
                // One divisor serves every expert of the stack.
                assert_eq!(t.matrix_part(PartKind::ScaleDivisor, 3).unwrap(), (t.part(PartKind::ScaleDivisor).unwrap().offset, 4));
            } else { assert!(t.matrix_part(PartKind::ExpertScale, 3).is_err()); }
            assert!(t.matrix_part(PartKind::Codes, EXPERTS).is_err());
        }
    }
    // A tensor that is not a stack has no matrix.
    assert!(plan.tensor(&format!("{P}embed.weight")).unwrap().matrix_bytes().is_err());
}

#[test]
fn a_packed_stack_stands_beside_each_widened_stack() {
    let plan = plan();
    for layer in 0..LAYERS as u32 {
        let codes = if layer < 6 { ExpertCodes::Nvfp4 } else { ExpertCodes::Mxfp8 };
        for (proj, order, rows, columns) in [(Proj::GateUp, Some(ExpertRows::Interleaved), 2 * WIDTH, HIDDEN), (Proj::Down, None, HIDDEN, WIDTH)] {
            let (widened, packed) = (plan.experts(layer, proj).unwrap(), plan.packed_experts(layer, proj).unwrap());
            // The name of the widened stack with the mark after it, the same role, place and file parts.
            assert_eq!(packed.name, format!("{}{}", widened.name, loop_load::PACKED_NAME));
            assert_eq!((packed.role, packed.place, &packed.parts), (widened.role, widened.place, &widened.parts));
            assert_eq!(packed.form, LoadForm::PackedExperts { codes, rows: order });
            assert_eq!(packed.stack().unwrap(), [EXPERTS, rows, columns]);
            let unit = packed.unit().unwrap();
            assert_eq!((unit.codes, unit.rows as u64, unit.columns as u64, unit.order), (codes, rows, columns, order));
            // One unit: the codes, the block scales and the table, each part on 16 bytes.
            let (code_plane, scale_plane) = packed_plane_bytes(rows, columns, codes == ExpertCodes::Nvfp4);
            let each = code_plane + scale_plane + if codes == ExpertCodes::Nvfp4 { 8192 } else { 1024 };
            assert_eq!((packed.matrix_bytes().unwrap(), packed.bytes, packed.device_bytes().unwrap()), (each, EXPERTS * each, EXPERTS * each));
            assert_eq!(unit.layout().unwrap().bytes as u64, each);
            // The parts of expert 3 are the parts of expert 3 of the widened stack.
            assert_eq!(packed.matrix_part(PartKind::Codes, 3).unwrap(), widened.matrix_part(PartKind::Codes, 3).unwrap());
            assert!(widened.unit().is_none() && !widened.form.is_packed() && packed.form.is_packed());
            // The widen step is for the widened stack, and the fill of a unit for the packed one.
            let (code_bytes, scale_bytes) = if codes == ExpertCodes::Nvfp4 { (rows * columns / 2, rows * columns / 16) } else { (rows * columns, rows * columns / 32) };
            let (stored, scales, scale) = (vec![0u8; code_bytes as usize], vec![0x78u8; scale_bytes as usize], (codes == ExpertCodes::Nvfp4).then_some(1.0));
            let mut out = vec![0xa5u8; each as usize];
            packed.fill_packed(&stored, &scales, scale, scale, &mut out).unwrap();
            assert!(out[..code_plane as usize].iter().all(|&code| code == 0));
            assert_eq!(out[code_plane as usize], 0x78);
            assert!(widened.fill_packed(&stored, &scales, scale, scale, &mut out).is_err());
            assert!(packed.widen(&stored, &scales, scale, scale, &mut vec![0u8; (rows * columns * 2) as usize]).is_err());
            assert!(packed.fill_packed(&stored, &scales, if scale.is_some() { None } else { Some(1.0) }, scale, &mut out).is_err());
        }
    }
    assert!(plan.packed_experts(LAYERS as u32, Proj::Down).is_err());
    assert_eq!(plan.tensors.values().filter(|t| t.form.is_packed()).count() as u64, 2 * LAYERS);
}

#[test]
fn a_stack_is_widened_from_its_stored_parts() {
    let plan = plan();
    // 4-bit, layer 0, gate and up: 128 rows of 96 columns. Code 2 is 1.0. Scale 0x38 is 1.0.
    let t = plan.experts(0, Proj::GateUp).unwrap();
    let (rows, columns) = (2 * WIDTH as usize, HIDDEN as usize);
    let mut codes = vec![0u8; rows * columns / 2];
    codes[columns / 2] = 2;                       // stored row 1, column 0: up row 0
    let scales = vec![0x38u8; rows * columns / 16];
    let mut out = vec![0u8; t.matrix_bytes().unwrap() as usize];
    t.widen(&codes, &scales, Some(1.0), Some(2.0), &mut out).unwrap();
    let value = |row: usize, column: usize| { let at = (row * columns + column) * 2; u16::from_le_bytes([out[at], out[at + 1]]) };
    // 1.0 * 1.0 * (1.0 / 2.0) = 0.5 = BF16 pattern 0x3f00, in the first up row: row `width` of the widened matrix.
    assert_eq!((value(WIDTH as usize, 0), value(1, 0), value(0, 0)), (0x3f00, 0, 0));
    // The scales of the other form are refused.
    assert!(t.widen(&codes, &scales, None, None, &mut out).is_err());
    // 8-bit, layer 6, down: 96 rows of 64 columns. Value 0x38 is 1.0. Scale 128 is 2.
    let t = plan.experts(6, Proj::Down).unwrap();
    let (rows, columns) = (HIDDEN as usize, WIDTH as usize);
    let mut values = vec![0u8; rows * columns]; values[columns + 1] = 0x38;
    let mut out = vec![0u8; t.matrix_bytes().unwrap() as usize];
    t.widen(&values, &vec![128u8; rows * columns / 32], None, None, &mut out).unwrap();
    let at = (columns + 1) * 2;
    assert_eq!(u16::from_le_bytes([out[at], out[at + 1]]), 0x4000);
    assert!(t.widen(&values, &vec![128u8; rows * columns / 32], Some(1.0), Some(1.0), &mut out).is_err());
}

#[test]
fn a_tensor_takes_whole_device_words() {
    assert_eq!([0, 1, 4, 5, 6, 4096, 4097].map(|n| device_extent(n).unwrap()), [0, 4, 4, 8, 8, 4096, 4100]);
    assert!(device_extent(u64::MAX).is_err());
    assert_eq!(loop_load::DEVICE_WORD, 4);
}

#[test]
fn state_bytes_of_each_visit() {
    let d = description();
    let state = LoopStatePlan::new(&d, 300, BUCKET).unwrap();
    assert_eq!((state.context, state.plane_rows, state.visits.len()), (300, 512, VISITS.len()));
    assert_eq!((state.window_visits(), state.full_visits()), (8, 8));
    for (v, &(layer, pass)) in state.visits.iter().zip(&VISITS) {
        assert_eq!((v.visit.layer, v.visit.pass), (layer, pass));
        let after = KERNEL * HIDDEN * 4;
        let (channels, rows) = if layer % 2 == 0 { (WINDOW_CHANNELS, WINDOW) } else { (FULL_CHANNELS, 512) };
        assert_eq!(v.window, (layer % 2 == 0).then_some(WINDOW), "layer {layer}");
        assert_eq!((v.rows, v.row_bytes, v.keys, v.values), (rows, channels * 2, rows * channels * 2, rows * channels * 2), "layer {layer}");
        assert_eq!(v.conv.map(|r| (r.ring, r.rows, r.channels, r.bytes)), [(ConvRing::Key, KERNEL, channels, KERNEL * channels * 4),
            (ConvRing::Value, KERNEL, channels, KERNEL * channels * 4), (ConvRing::AfterAttention, KERNEL, HIDDEN, after), (ConvRing::AfterExperts, KERNEL, HIDDEN, after)]);
        assert_eq!(v.bytes(), 2 * rows * channels * 2 + 2 * KERNEL * channels * 4 + 2 * after);
    }
    // The three passes of a looped layer have three states of the same size, in three slots.
    let slots = state.visits.iter().map(|v| v.visit.slot).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(slots.len(), VISITS.len());
    assert_eq!((state.step.stream_rows, state.step.stack_rows, state.step.carried_rows, state.step.message_rows), (2, 5, 2, 16));
    assert_eq!((state.step.row_bytes, state.step.bytes), (HIDDEN * 4, 25 * HIDDEN * 4));
    assert_eq!(state.plane_row_bytes().unwrap(), 8 * 2 * FULL_CHANNELS * 2);
}

#[test]
fn the_ledger_equals_the_formula_at_three_context_sizes() {
    let (d, plan) = (description(), plan());
    let (plain, head, experts) = expected_weights();
    for bucket in [BUCKET, 100, 1] {
        for context in [1, 300, MAX_CONTEXT] {
            let rows = context.div_ceil(bucket) * bucket;
            let state = LoopStatePlan::new(&d, context, bucket).unwrap();
            let ledger = LoopLedger::new(&plan.bytes, &state).unwrap();
            let (fixed, planes, step) = expected_state(rows);
            assert_eq!(state.plane_rows, rows);
            let weights = plain + head + experts + expected_packed();
            assert_eq!((ledger.context, ledger.weights, ledger.state_fixed, ledger.state_planes, ledger.step), (context, weights, fixed, planes, step), "context {context} bucket {bucket}");
            assert_eq!(ledger.total, weights + fixed + planes + step);
            assert_eq!((ledger.chat(), state.chat_bytes().unwrap()), (fixed + planes + step, fixed + planes + step));
            // The ledger is the sum of what the plans list, one entry at a time.
            assert_eq!(ledger.weights, plan.tensors.values().map(|t| t.device_bytes().unwrap()).sum::<u64>());
            assert_eq!(ledger.state_fixed + ledger.state_planes, state.visits.iter().map(|v| v.bytes()).sum::<u64>());
        }
    }
}

#[test]
fn a_state_that_does_not_fit_is_refused_with_the_memory_text() {
    let (d, plan) = (description(), plan());
    let weights = plan.bytes.device;
    let (fixed, _, step) = expected_state(0);
    let row = 8 * 2 * FULL_CHANNELS * 2;
    let base = weights + fixed + step;
    // Room for exactly three buckets of plane rows, and one byte less.
    let budget = base + 3 * BUCKET * row;
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, budget).unwrap(), 3 * BUCKET);
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, budget - 1).unwrap(), 2 * BUCKET);
    // The model's own limit caps the answer. Less than one bucket holds no token.
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, u64::MAX).unwrap(), MAX_CONTEXT);
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, base + BUCKET * row - 1).unwrap(), 0);
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, base - 1).unwrap(), 0);
    assert_eq!(context_limit(&d, &plan.bytes, BUCKET, 0).unwrap(), 0);

    let admit = |context: u64, budget: u64| LoopLedger::new(&plan.bytes, &LoopStatePlan::new(&d, context, BUCKET).unwrap()).unwrap()
        .admit(budget, context_limit(&d, &plan.bytes, BUCKET, budget).unwrap());
    // A context inside the third bucket fits. The first token of the fourth bucket does not.
    assert!(admit(3 * BUCKET, budget).is_ok());
    assert!(admit(2 * BUCKET + 1, budget).is_ok());
    let refusal = admit(3 * BUCKET + 1, budget).unwrap_err();
    let session = fixed + step + 4 * BUCKET * row;
    assert_eq!(refusal.to_string(), format!("CONTEXT=769; budget={budget}; resident={weights}; shared_workspace=0; session_pool={}; full_session={session}; reduce CONTEXT to 768 or less", budget - weights));
    assert_eq!((refusal.largest_context, refusal.session, refusal.resident), (768, session, weights));
    // A budget under the weights fits no context, and the text says so.
    let refusal = admit(1, weights - 1).unwrap_err();
    assert_eq!(refusal.to_string(), format!("CONTEXT=1; budget={}; resident={weights}; shared_workspace=0; session_pool=0; full_session={}; reduce CONTEXT: no context fits, the weights and the fixed state need more than the budget", weights - 1, fixed + step + BUCKET * row));
    // A context that the model does not take is refused before any byte is counted.
    for context in [0, MAX_CONTEXT + 1] {
        let error = LoopStatePlan::new(&d, context, BUCKET).unwrap_err().to_string();
        assert_eq!(error, format!("unsupported context: {context} tokens: the model takes 1 to {MAX_CONTEXT} tokens"));
    }
    assert!(LoopStatePlan::new(&d, 1, 0).is_err());
}

fn local(folder: &str) -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.local").join(folder) }

/// The real folder: the public model, or a link to it, in `.local/inkling-checkpoint` at the repository root.
/// `cargo test -p engine-model --test loop_load -- --ignored --nocapture real_folder_ledger`.
#[test]
#[ignore = "needs the public model folder in .local/inkling-checkpoint"]
fn real_folder_ledger() {
    let root = local("inkling-checkpoint");
    assert!(root.join("config.json").is_file(), "put the public model folder, or a link to it, in .local/inkling-checkpoint for the ignored host test");
    let started = std::time::Instant::now();
    let checkpoint = loop_load::read_folder(&root).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let (d, b) = (&checkpoint.description, checkpoint.load.bytes);
    let forms = |form: fn(&LoadForm) -> bool| checkpoint.load.tensors.values().filter(|t| form(&t.form)).count();
    println!("LOAD PLAN tensors={} stored={} head={} expert_stacks={} read_seconds={seconds:.4}", checkpoint.load.tensors.len(),
        forms(|f| *f == LoadForm::Stored), forms(|f| *f == LoadForm::LeadingRows), forms(|f| matches!(f, LoadForm::WidenedExperts { .. })));
    println!("LOAD BYTES stored_device={} head_device={} head_unused={} experts_device={} expert_sources={} device={} file_read={} image_tower_not_loaded={}",
        b.stored, b.head, b.head_unused, b.experts, b.expert_sources, b.device, b.file, b.unused);
    let file: u64 = checkpoint.shards.values().flat_map(|h| h.tensors.values()).map(|t| t.end - t.start).sum();
    assert_eq!(b.file + b.head_unused + b.unused, file);
    assert_eq!(b.device, checkpoint.load.tensors.values().map(|t| t.device_bytes().unwrap()).sum::<u64>());
    for context in [1024u64, 32_768, 131_072, d.max_positions] {
        let state = LoopStatePlan::new(d, context, BUCKET).unwrap();
        let ledger = LoopLedger::new(&b, &state).unwrap();
        println!("LEDGER context={context} plane_rows={} visits={} window_visits={} full_visits={} weights={} state_fixed={} state_planes={} step={} chat={} total={}",
            state.plane_rows, state.visits.len(), state.window_visits(), state.full_visits(), ledger.weights, ledger.state_fixed, ledger.state_planes, ledger.step, ledger.chat(), ledger.total);
        assert_eq!(ledger.total, ledger.weights + state.chat_bytes().unwrap());
    }
    for budget in [64u64 << 30, 100 << 30, 120 << 30] {
        println!("CONTEXT LIMIT budget={budget} context={}", context_limit(d, &b, BUCKET, budget).unwrap());
    }
}

/// The widened experts that the reference run dumped, with the stored bytes of the same experts, in
/// `.local/inkling-ref-dumps` (a link to the dump folder of the reference run; it is outside this repository).
/// `cargo test -p engine-model --test loop_load -- --ignored --nocapture widened_experts_equal_the_reference`.
#[test]
#[ignore = "needs the widened-expert dumps of the reference run in .local/inkling-ref-dumps"]
fn widened_experts_equal_the_reference() {
    use engine_model::plan::{loop_widen, storage_groups::{GATE_UP_ROWS, NVFP4_NIBBLES}};
    let root = local("inkling-ref-dumps");
    let index = std::fs::read(root.join("dump-experts-index.json")).expect("put a link to the dump folder in .local/inkling-ref-dumps");
    let doc = json::parse(&index, json::Limits { bytes: 1 << 20, ..Default::default() }).unwrap();
    let o = doc.object().unwrap();
    let numbers = |key: &str| o[key].array().unwrap().iter().map(|v| v.u64().unwrap()).collect::<Vec<_>>();
    let shapes: BTreeMap<String, Vec<u64>> = o["files"].array().unwrap().iter().map(|f| { let f = f.object().unwrap();
        (f["file"].string().unwrap().to_owned(), f["shape"].array().unwrap().iter().map(|d| d.u64().unwrap()).collect()) }).collect();
    let read = |name: &str| std::fs::read(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let float = |name: &str| f32::from_le_bytes(read(name).as_slice().try_into().unwrap());
    let (mut matrices, mut values, mut different) = (0u64, 0u64, 0u64);
    for layer in numbers("layers") { for expert in numbers("experts") {
        for (stored, loaded, order) in [("w13", "gate_up_proj", Some(GATE_UP_ROWS)), ("w2", "down_proj", None)] {
            let name = format!("loaded-L{layer}-E{expert}.{loaded}.bf16");
            let (rows, columns) = (shapes[&name][0] as usize, shapes[&name][1] as usize);
            let want = read(&name);
            let base = format!("stored-L{layer}-E{expert}.{stored}_weight");
            let (codes, scales) = (read(&format!("{base}.u8")), read(&format!("{base}_scale.u8")));
            let mut got = vec![0u8; rows * columns * 2];
            let four_bit = root.join(format!("{base}_amax.f32")).is_file();
            if four_bit {
                let factor = loop_widen::expert_factor(float(&format!("{base}_amax.f32")), float(&format!("{base}_outer_scale_denominator.f32"))).unwrap();
                loop_widen::widen_nvfp4(&codes, &scales, factor, rows, columns, order, NVFP4_NIBBLES, &mut got).unwrap();
            } else { loop_widen::widen_mxfp8(&codes, &scales, rows, columns, order, &mut got).unwrap(); }
            let unequal = got.as_chunks::<2>().0.iter().zip(want.as_chunks::<2>().0).filter(|(a, b)| a != b).count() as u64;
            println!("WIDEN layer={layer} expert={expert} matrix={loaded} form={} rows={rows} columns={columns} values={} different={unequal}",
                if four_bit { "4-bit" } else { "8-bit" }, rows * columns);
            assert_eq!(got.len(), want.len());
            matrices += 1; values += (rows * columns) as u64; different += unequal;
        }
    } }
    println!("WIDEN SUM matrices={matrices} values={values} different={different}");
    assert!(matrices > 0);
    assert_eq!(different, 0);
}
