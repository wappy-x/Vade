//! Tests on a small made-up model. Its config and storage sidecar are the
//! fixture of the tensor index (`tests/fixtures/inkling-index/`): 8 stored
//! layers, a loop over 4 of them with 3 passes (16 visits a token), width 96,
//! a window of 36 tokens, 7 experts with 2 picked, 320 stored head rows of
//! which 300 are used. Every size of the two layer kinds differs. The weights
//! are made here from a fixed generator. No value comes from a real model.
use super::*;
use crate::{plan::{inkling_index::{Expected, ExpertPart}, storage_groups::StorageGroups}, safetensors::Dtype, visit_loop::{Body, Driver}};
use std::collections::BTreeMap;

const CONFIG: &[u8] = include_bytes!("../../tests/fixtures/inkling-index/config.json");
const SIDECAR: &[u8] = include_bytes!("../../tests/fixtures/inkling-index/hf_quant_config.json");
const NO_LIMIT: Limits = Limits { expert_cache_bytes: usize::MAX };

/// A fixed generator (xorshift64*). The same seed gives the same model.
struct Dice(u64);
impl Dice {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12; self.0 ^= self.0 << 25; self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// A value in `low..high`.
    fn real(&mut self, low: f32, high: f32) -> f32 { low + (high - low) * ((self.next() >> 40) as f32 / (1u64 << 24) as f32) }
    fn below(&mut self, limit: u64) -> u64 { (self.next() >> 33) % limit }
}
/// Payload bytes for one expected tensor. The ranges keep every value finite.
fn payload(dice: &mut Dice, expected: &Expected) -> Vec<u8> {
    use StoredRole::*;
    let values = expected.shape.iter().product::<u64>() as usize;
    let range = match expected.role {
        Embedding => (-1.0, 1.0),
        EmbeddingNorm | FinalNorm | AttentionNorm | QueryInputNorm | KeyInputNorm | ValueInputNorm | ExpertNorm | QueryClip | RelativeClip => (0.75, 1.25),
        MixAttention | MixExperts | MixOutput | MessageQueries | MessageKvQueries => (-2.0, 2.0),
        ConvFixed(_) | ConvStatic(_) | ConvAfterAttention | ConvAfterExperts | RelativeTable => (-0.5, 0.5),
        Expert { part: ExpertPart::ExpertScale, .. } => (0.5, 1.0),
        Expert { part: ExpertPart::ScaleDivisor, .. } => (8.0, 8.0),
        _ => (-0.2, 0.2),
    };
    match (expected.dtype, expected.role) {
        // Packed 4-bit codes: any byte is two valid codes.
        (Dtype::U8, Expert { part: ExpertPart::Codes, .. }) => (0..values).map(|_| dice.below(256) as u8).collect(),
        // 8-bit float values: exponent field 4 to 7, so 0.125 to 1.875, either sign. Never the not-a-number code.
        (Dtype::F8E4M3, Expert { part: ExpertPart::Codes, .. }) => (0..values).map(|_| ((dice.below(2) << 7) | ((4 + dice.below(4)) << 3) | dice.below(8)) as u8).collect(),
        // 4-bit block scales: exponent field 5 to 7, so 0.25 to 1.875.
        (Dtype::F8E4M3, Expert { part: ExpertPart::BlockScale, .. }) => (0..values).map(|_| (((5 + dice.below(3)) << 3) | dice.below(8)) as u8).collect(),
        // 8-bit block exponents: 2^-3 to 2^0.
        (Dtype::U8, Expert { part: ExpertPart::BlockScale, .. }) => (0..values).map(|_| 124 + dice.below(4) as u8).collect(),
        (Dtype::Bf16, _) => (0..values).flat_map(|_| ((round_bf16(dice.real(range.0, range.1)).to_bits() >> 16) as u16).to_le_bytes()).collect(),
        (Dtype::F32, _) => (0..values).flat_map(|_| dice.real(range.0, range.1).to_le_bytes()).collect(),
        other => panic!("the fixture has no payload rule for {other:?}"),
    }
}
fn fixture() -> (LoopTextDescription, IndexInput, Memory) {
    let description = LoopTextDescription::from_config(CONFIG).unwrap();
    let index = IndexInput::from_description(&description, &StorageGroups::from_json(SIDECAR).unwrap()).unwrap();
    let mut dice = Dice(0x9e37_79b9_7f4a_7c15);
    let tensors = index.expected_main().unwrap().into_iter()
        .map(|(name, expected)| { let bytes = payload(&mut dice, &expected); (name, MemoryTensor { dtype: expected.dtype, shape: expected.shape, bytes }) }).collect();
    (description, index, Memory { tensors })
}
fn model(limits: Limits) -> Model {
    let (description, index, memory) = fixture();
    Model::new(&description, &index, Box::new(memory), limits).unwrap()
}
/// 40 token IDs under the stored row count. The text is longer than the window of 36.
fn text() -> Vec<u32> { let mut dice = Dice(7); (0..40).map(|_| dice.below(320) as u32).collect() }
/// Every word of a result, as bits.
fn words(outs: &[TokenOut]) -> Vec<u32> {
    let mut bits = Vec::new();
    for out in outs {
        for values in out.streams.iter().chain([&out.embedding, &out.end, &out.normed]).chain(out.logits.iter()) { bits.extend(values.iter().map(|v| v.to_bits())); }
        bits.extend(out.top);
    }
    bits
}
const EVERY: Want = Want { head: Head::Every, records: false };

#[test]
fn the_fixture_has_both_layer_kinds_both_expert_formats_and_a_loop() {
    let model = model(NO_LIMIT);
    let visits = model.schedule().visits();
    assert_eq!((visits.len(), model.hidden(), model.head_rows(), model.stored_rows), (16, 96, 300, 320));
    assert_eq!(visits.iter().map(|v| v.layer).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]);
    assert!(model.layers.iter().any(|l| matches!(l.attention.reach, Reach::Window { window: 36 })) && model.layers.iter().any(|l| matches!(l.attention.reach, Reach::Full { .. })));
    assert!(model.layers.iter().any(|l| matches!(l.k_conv, ConvData::Fixed(_))) && model.layers.iter().any(|l| matches!(l.k_conv, ConvData::TokenMade { .. })));
    let (description, index, _) = fixture();
    let stored: Vec<Dtype> = (0..description.layers.len()).map(|i| index.expected_main().unwrap()[&format!("model.llm.layers.{i}.mlp.experts.w2_weight")].dtype).collect();
    assert!(stored.contains(&Dtype::U8) && stored.contains(&Dtype::F8E4M3), "the fixture must hold packed 4-bit and 8-bit float experts: {stored:?}");
}

#[test]
fn a_text_gives_the_same_words_at_every_thread_count_and_every_cut() {
    let model = model(NO_LIMIT);
    let runner = model.runner().unwrap();
    let text = text();
    // One call, one thread.
    let mut whole = runner.chat().unwrap();
    let want = words(&runner.read(&mut whole, &text, EVERY, 1).unwrap());
    assert_eq!(whole.position(), text.len());
    // A window layer keeps 35 tokens for the next one. A full layer keeps all 40.
    let kept: Vec<usize> = whole.visits().iter().map(|v| v.attention.len()).collect();
    assert!(kept.contains(&35) && kept.contains(&40) && kept.iter().all(|&n| n == 35 || n == 40), "{kept:?}");
    // One call, 5 threads: more threads than cores change nothing.
    let mut threaded = runner.chat().unwrap();
    assert_eq!(words(&runner.read(&mut threaded, &text, EVERY, 5).unwrap()), want);
    assert_eq!(threaded, whole);
    // Three pieces with cuts at 1 and 37 (one side of the window edge each), and a fork at the second cut.
    let mut cut = runner.chat().unwrap();
    let mut got = runner.read(&mut cut, &text[..1], EVERY, 3).unwrap();
    got.extend(runner.read(&mut cut, &text[1..37], EVERY, 2).unwrap());
    let mut fork = cut.clone();
    got.extend(runner.read(&mut cut, &text[37..], EVERY, 16).unwrap());
    assert_eq!(words(&got), want);
    assert_eq!(cut, whole);
    // The fork goes on token by token.
    let tail: Vec<TokenOut> = text[37..].iter().flat_map(|&token| runner.read(&mut fork, &[token], EVERY, 1).unwrap()).collect();
    assert_eq!(words(&tail), words(&got[37..]));
    assert_eq!(fork, whole);
    // An empty read is valid and changes nothing.
    assert!(runner.read(&mut fork, &[], EVERY, 4).unwrap().is_empty());
    assert_eq!(fork, whole);
}

/// The two halves of a visit as the driver of `visit_loop` asks for them, one visit at a time on one thread.
struct Plain<'r, 'm> { runner: &'r Runner<'m>, states: BTreeMap<usize, VisitState> }
impl Body for Plain<'_, '_> {
    fn attention(&mut self, visit: &Visit, query: &[f32], key: &[f32], value: &[f32]) -> Result<Vec<f32>> {
        let (model, data) = (self.runner.model, &self.runner.model.layers[visit.layer]);
        let state = self.states.get_mut(&visit.slot).unwrap();
        let norm = |x: &[f32], which: usize| rounded_norm(x, &data.input_norms[which.min(data.input_norms.len() - 1)], model.epsilon);
        let out = self.runner.attention[visit.layer].step(&mut state.attention, &mut state.conv, &norm(query, 0)?, &norm(key, 1)?, &norm(value, 2)?)?;
        self.runner.after_attention[visit.layer].step(state.conv.get_mut(Slot::AfterAttention), &out)
    }
    fn experts(&mut self, visit: &Visit, input: &[f32]) -> Result<Vec<f32>> {
        let (model, data) = (self.runner.model, &self.runner.model.layers[visit.layer]);
        let z = rounded_norm(input, &data.expert_norm, model.epsilon)?;
        let (_, out) = capped_moe::moe_token(&data.routing, model.activation, &data.router, &data.router_bias, &z,
            |id| data.stacks.widen(model.source.as_ref(), model.hidden, data.width, id))?;
        self.runner.after_experts[visit.layer].step(self.states.get_mut(&visit.slot).unwrap().conv.get_mut(Slot::AfterExperts), &out)
    }
}

#[test]
fn a_read_equals_the_driver_of_the_visit_loop_token_by_token() {
    let model = model(NO_LIMIT);
    let runner = model.runner().unwrap();
    let layers = model.layers.iter().map(|l| visit_loop::LayerWeights { attention_queries: l.mix_attention.as_deref(), expert_query: &l.mix_experts,
        gates: l.gates.as_ref().map(|gates| gates.each_ref().map(|g| GateWeights { down: &g.down, forget: &g.forget, gain: &g.gain })) }).collect();
    let driver = Driver::new(model.config, visit_loop::Weights { layers, message_queries: &model.message_queries, output_query: &model.output_query }).unwrap();
    let fresh = runner.chat().unwrap();
    let mut body = Plain { runner: &runner, states: model.schedule().visits().iter().zip(fresh.visits()).map(|(visit, state)| (visit.slot, state.clone())).collect() };
    let text = &text()[..6];
    let mut chat = runner.chat().unwrap();
    let outs = runner.read(&mut chat, text, Want { head: Head::None, records: true }, 4).unwrap();
    for (&token, out) in text.iter().zip(&outs) {
        let token = driver.token(&model.embed(token).unwrap(), &mut body).unwrap();
        assert_eq!(token.streams.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(), out.streams.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(token.end.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), out.end.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert!(out.logits.is_none() && out.top.is_none() && out.records.len() == 16);
    }
    // The states that the two runs leave are the same too.
    for (visit, state) in model.schedule().visits().iter().zip(chat.visits()) { assert_eq!(&body.states[&visit.slot], state); }
    // A record holds the values inside its visit: the picked experts are distinct and their weights add up to the route scale.
    let record = &outs[5].records[9];
    let mut ids = record.route.ids.clone(); ids.sort_unstable(); ids.dedup();
    assert_eq!((ids.len(), record.router_logits.len(), record.expert_input.len()), (2, 7, 96));
    assert!((record.route.weights.iter().sum::<f32>() - 8.0).abs() < 1e-5);
}

#[test]
fn greedy_writes_the_top_token_of_each_position_and_never_a_padding_row() {
    let (description, index, mut memory) = fixture();
    // Rows 300 to 319 of the head are padding. Fill them with large numbers. A head that read them would write one of them.
    let head = memory.tensors.get_mut("model.llm.unembed.weight").unwrap();
    assert_eq!((head.dtype, &head.shape[..]), (Dtype::Bf16, &[320u64, 96][..]));
    for pair in head.bytes[300 * 96 * 2..].as_chunks_mut::<2>().0 { *pair = 0x4100u16.to_le_bytes(); }
    let model = Model::new(&description, &index, Box::new(memory), NO_LIMIT).unwrap();
    let runner = model.runner().unwrap();
    let prompt = &text()[..5];
    let mut chat = runner.chat().unwrap();
    let written = runner.greedy(&mut chat, prompt, 6, &[], 4).unwrap();
    assert_eq!((written.len(), chat.position()), (6, 5 + 5));
    assert!(written.iter().all(|&token| token < 300), "{written:?}");
    // The same tokens come from one read of the prompt and the written tokens: each top token is the next token.
    let mut all = prompt.to_vec(); all.extend(&written[..5]);
    let outs = runner.read(&mut runner.chat().unwrap(), &all, EVERY, 3).unwrap();
    assert_eq!(outs[4..].iter().map(|out| out.top.unwrap()).collect::<Vec<_>>(), written);
    for out in &outs {
        let logits = out.logits.as_ref().unwrap();
        assert_eq!(logits.len(), 300);
        assert_eq!(out.top.unwrap() as usize, capped_head::top_token(logits).unwrap());
        assert!(logits.iter().all(|v| v.abs() <= 30.0));
    }
    // The head on one thread and on 7 threads gives the same logits.
    assert_eq!(runner.logits(&outs[0].normed, 1).unwrap(), runner.logits(&outs[0].normed, 7).unwrap());
    // A stop token ends the run after it is written. No new token: the prompt is read and nothing is written.
    let stopped = runner.greedy(&mut runner.chat().unwrap(), prompt, 6, &[written[2]], 2).unwrap();
    let first_stop = written.iter().position(|&token| token == written[2]).unwrap();
    assert_eq!(stopped, written[..=first_stop]);
    let mut silent = runner.chat().unwrap();
    assert!(runner.greedy(&mut silent, prompt, 0, &[], 2).unwrap().is_empty() && silent.position() == 5);
    assert!(runner.greedy(&mut silent, &[], 3, &[], 2).is_err());
}

#[test]
fn the_expert_cache_changes_no_word() {
    let text = &text()[..8];
    let run = |limits: Limits| {
        let model = model(limits);
        let runner = model.runner().unwrap();
        let outs = runner.read(&mut runner.chat().unwrap(), text, EVERY, 4).unwrap();
        (words(&outs), model.cache_counts().unwrap())
    };
    let (kept, kept_counts) = run(NO_LIMIT);
    let (none, none_counts) = run(Limits { expert_cache_bytes: 0 });
    // One widened expert is 3 * 96 * 64 FP32 values. Room for two of them.
    let (two, two_counts) = run(Limits { expert_cache_bytes: 2 * 3 * 96 * 64 * 4 });
    assert_eq!(kept, none);
    assert_eq!(kept, two);
    // 8 tokens, 16 visits, 2 experts a visit: 256 uses. With no room every use widens.
    assert_eq!((none_counts.hits, none_counts.widened, none_counts.held, none_counts.held_bytes), (0, 256, 0, 0));
    assert_eq!(kept_counts.hits + kept_counts.widened, 256);
    // 8 stored layers of 7 experts: at most 56 are ever held.
    assert!(kept_counts.hits > 0 && kept_counts.held <= 56 && kept_counts.held_bytes == kept_counts.held * 3 * 96 * 64 * 4);
    assert!(two_counts.held <= 2 && two_counts.hits + two_counts.widened == 256);
}

#[test]
fn bad_inputs_are_refused_with_a_reason() {
    let (description, index, memory) = fixture();
    let load = |description: &LoopTextDescription, memory: Memory| Model::new(description, &index, Box::new(memory), NO_LIMIT).map(|_| ()).map_err(|e| e.to_string());
    assert!(load(&description, memory.clone()).is_ok());
    // A tensor that is absent, has another shape or another type is named.
    for name in ["model.llm.layers.3.attn.wq_du.weight", "model.llm.layers.6.mlp.experts.w2_weight_scale", "model.llm.attn_res.output_query"] {
        let mut absent = memory.clone();
        absent.tensors.remove(name).unwrap();
        assert!(load(&description, absent).unwrap_err().contains(name), "{name}");
        let mut reshaped = memory.clone();
        reshaped.tensors.get_mut(name).unwrap().shape.push(1);
        assert!(load(&description, reshaped).unwrap_err().contains(name), "{name}");
    }
    let mut retyped = memory.clone();
    retyped.tensors.get_mut("model.llm.layers.0.mlp.gate.bias").unwrap().dtype = Dtype::Bf16;
    assert!(load(&description, retyped).unwrap_err().contains("model.llm.layers.0.mlp.gate.bias"));
    // A value that is not finite is refused at load.
    let mut poisoned = memory.clone();
    poisoned.tensors.get_mut("model.llm.layers.2.mlp.gate.bias").unwrap().bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(load(&description, poisoned).unwrap_err().contains("not finite"));
    // A description whose layout fact or visit list differs from the host forms is refused.
    let mut flipped = description.clone();
    flipped.conv.taps = ConvTaps::Fixed { order: DescribedOrder::NewestFirst };
    assert!(load(&flipped, memory.clone()).unwrap_err().contains("tap order"));
    let mut moved = description.clone();
    moved.visits[7].slot += 1;
    assert!(load(&moved, memory.clone()).unwrap_err().contains("visit list"));
    let mut short = description.clone();
    short.max_positions = 3;
    let model = Model::new(&short, &index, Box::new(memory), NO_LIMIT).unwrap();
    let runner = model.runner().unwrap();
    let mut chat = runner.chat().unwrap();
    // A token outside the stored rows, and a text past the position limit. The chat stays usable after both.
    assert!(runner.read(&mut chat, &[5, 320], EVERY, 2).unwrap_err().to_string().contains("embedding table"));
    assert!(runner.read(&mut chat, &[5, 6, 7, 8], EVERY, 2).unwrap_err().to_string().contains("position limit"));
    assert_eq!(runner.read(&mut chat, &[5, 6, 7], EVERY, 2).unwrap().len(), 3);
    assert!(runner.read(&mut chat, &[8], EVERY, 2).is_err());
    // A visit outside the list and inputs that do not fit the visit are refused.
    assert!(runner.inputs(16, &[0.0; 96], &[]).is_err() && runner.inputs(3, &[0.0; 96], &[]).is_err());
}
