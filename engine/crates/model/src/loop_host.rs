//! Whole-model host form of a looped block-attention text decoder: token IDs
//! in, logits and the greedy next token out. It runs on the CPU from the
//! stored weights, in plain host arithmetic. It is slow on purpose. A device
//! run of the same model must give these words. See `loop_host/README.md`.
//!
//! This file holds no arithmetic of its own beyond the weighted norm call and
//! its one BF16 round. Every step is the host form of its operation:
//! `block_mix` and `visit_loop` (mixes, messages, gates, the visit list),
//! `sink_attention` and `short_conv` (attention and the four convolutions),
//! `capped_moe` and `expert_widen` (router, experts, fold), `capped_head`
//! (embedding norm, last norm, head, logit cap, top token). So the number
//! types, the rounding points and the sum orders are theirs.
//!
//! Source of the order in which the steps meet (read for the math, no code
//! copied): the public `transformers` release v5.24.0, commit
//! `eeee11099135592f1d03c2871fbd3d8132e5a4d7`, file
//! `models/inkling2_loop/modeling_inkling2_loop.py`. One visit: lines 876-910,
//! 937-997, 1022-1062 and 1091-1149. The token loop and the end: lines
//! 1350-1443 and 1814-1816.
//!
//! All sizes come from the model description and from the tensor shapes. No
//! model name selects anything. Threads change no word: a visit of one token
//! is a fixed function of its inputs and of the state that the token before
//! left, and each sum keeps its order. A prompt read in one call, in pieces
//! or token by token gives the same words. This module enables no model plan
//! and changes no serving path.
mod source;
pub use source::{CacheCounts, Folder, Memory, MemoryTensor, Source, open_folder};
use source::{ExpertCache, ExpertStacks, Names};
use crate::{Result, invalid, capped_head, capped_moe::{self, CappedActivation, GateUpRows, Route, Routing, STORED_GATE_UP_ROWS},
    expert_widen::LOW_NIBBLE_IS_EVEN_COLUMN, host_ops::router::RouterKind, norm_cap::{SoftCap, check_cap, round_bf16, weighted_norm},
    plan::{description::{ConvTaps, LoopLayerDescription, LoopTextDescription, TapOrder as DescribedOrder},
        inkling_index::{GatePart, IndexInput, MessageGate, StoredRole, Stream},
        storage_groups::{ExpertRows, GATE_UP_ROWS, NVFP4_NIBBLES, NibbleOrder}},
    shapes::{Ffn, Place, Proj}, short_conv::{self, FixedTaps, Slot, Taps, TokenTaps, VisitHistories},
    sink_attention::{self, Reach}, visit_loop::{self, GateWeights, Inputs, Schedule, Visit}};

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("loop host: {message}"))) }
}
fn size(value: u64) -> Result<usize> { usize::try_from(value).map_err(|_| invalid("loop host: a size exceeds usize")) }
/// The weighted norm of an FP32 vector, rounded to BF16 one time.
fn rounded_norm(x: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    let mut out = weighted_norm(x, weight, epsilon)?;
    for value in &mut out { *value = round_bf16(*value); }
    need(out.iter().all(|v| v.is_finite()), "a norm result is not finite in BF16")?;
    Ok(out)
}

/// Host resource limits. They change no word of a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Bytes of widened experts that the cache may hold.
    pub expert_cache_bytes: usize,
}

enum ConvData { Fixed(Vec<f32>), TokenMade { static_taps: Vec<f32>, projection: Vec<f32> } }
struct GateData { down: Vec<f32>, forget: Vec<f32>, gain: Vec<f32> }
/// The stored numbers of one layer, widened to FP32, and where its experts are.
struct LayerData {
    attention: sink_attention::Config,
    q_proj: Vec<f32>, q_scale: Vec<f32>, k_proj: Vec<f32>, v_proj: Vec<f32>, k_conv: ConvData, v_conv: ConvData,
    rel_proj: Vec<f32>, rel_scale: Vec<f32>, rel_table: Vec<f32>, sink_proj: Vec<f32>, gate_proj: Vec<f32>, o_proj: Vec<f32>,
    /// One norm weight for all three attention inputs, or one each for query, key and value.
    input_norms: Vec<Vec<f32>>,
    mix_attention: Option<Vec<f32>>, mix_experts: Vec<f32>,
    gates: Option<[GateData; visit_loop::MESSAGES]>,
    after_attention: Vec<f32>, after_experts: Vec<f32>, conv_taps: usize,
    expert_norm: Vec<f32>, router: Vec<f32>, router_bias: Vec<f32>,
    routing: Routing, width: usize, stacks: ExpertStacks,
}

/// One loaded model: its sizes, its dense weights and the place of its experts.
pub struct Model {
    source: Box<dyn Source>,
    hidden: usize, stored_rows: usize, head_rows: usize, max_positions: usize,
    epsilon: f32, logit_cap: SoftCap, activation: CappedActivation,
    config: visit_loop::Config, schedule: Schedule,
    embedding: (String, crate::safetensors::Dtype),
    embedding_norm: Vec<f32>, final_norm: Vec<f32>, head: Vec<f32>,
    output_query: Vec<f32>, message_queries: Vec<f32>,
    layers: Vec<LayerData>, experts: ExpertCache,
}

/// The layout facts have one name in the plan and one in the host form of
/// each operation. Refuse a build in which the two names of a fact differ.
fn layout_facts_agree(description: &LoopTextDescription) -> Result<()> {
    let same = |described: DescribedOrder, host: short_conv::TapOrder| matches!((described, host),
        (DescribedOrder::OldestFirst, short_conv::TapOrder::OldestFirst) | (DescribedOrder::NewestFirst, short_conv::TapOrder::NewestFirst));
    let check = |taps: ConvTaps| match taps {
        ConvTaps::Fixed { order } => same(order, short_conv::FIXED_TAP_ORDER),
        // `short_conv::projected_tap` is the index `head * taps + tap`: the one value of `TapIndex`.
        ConvTaps::FixedAndToken { order, index: _ } => same(order, short_conv::TOKEN_TAP_ORDER),
    };
    need(check(description.conv.taps) && description.layers.iter().all(|l| check(l.attention.key_value_conv.taps)),
        "the tap order of the description differs from the convolution host form")?;
    need(matches!((GATE_UP_ROWS, STORED_GATE_UP_ROWS), (ExpertRows::Interleaved, GateUpRows::Interleaved)),
        "the gate and up row order of the storage plan differs from the expert host form")?;
    need(matches!(NVFP4_NIBBLES, NibbleOrder::LowFirst) == LOW_NIBBLE_IS_EVEN_COLUMN,
        "the 4-bit code order of the storage plan differs from the expert decode")
}

impl Model {
    /// Load a model from its description, its tensor index and its stored
    /// tensors. The source can be a weight file or tensors held in memory.
    /// Dense tensors are read now. Experts are read when a token picks them.
    pub fn new(description: &LoopTextDescription, index: &IndexInput, source: Box<dyn Source>, limits: Limits) -> Result<Self> {
        let d = description;
        layout_facts_agree(d)?;
        let hidden = size(d.hidden)?;
        need(d.residual.normalize, "a block mix over vectors that are not normed is not supported")?;
        need(d.routing.kind == RouterKind::CorrectedSigmoid && d.routing.normalize_selected && d.routing.groups == 1
            && d.routing.selected_groups == 1 && !d.routing.shared_gate, "the router of the description is not the corrected sigmoid form")?;
        need(matches!(d.conv.taps, ConvTaps::Fixed { .. }), "the convolutions after attention and after the experts must have fixed taps")?;
        need(d.head_rows > 0 && d.head_rows <= d.vocab, "the head rows of the description are outside the stored rows")?;
        let config = visit_loop::Config { hidden_size: hidden, layers: d.layers.len(), block_layers: size(d.residual.block_layers)?,
            loop_start: size(d.core.start)?, loop_end: size(d.core.end)?, passes: size(d.core.passes)?,
            gate_width: size(d.core.gate_rank)?, rms_norm_eps: d.epsilon as f32 };
        let schedule = Schedule::new(config)?;
        // The description carries its own visit list. Both must name the same visits.
        need(schedule.visits().len() == d.visits.len() && schedule.visits().iter().zip(&d.visits).all(|(mine, theirs)|
            mine.layer as u64 == theirs.layer && mine.slot as u64 == theirs.slot && mine.messages.is_some() == theirs.messages
                && mine.boundary == d.layers[mine.layer].boundary), "the visit list of the description differs from the visit loop")?;
        let names = Names::new(index, source.as_ref())?;
        let model = Place::Model;
        let (stored_rows, head_rows) = (size(d.vocab)?, size(d.head_rows)?);
        let (embedding, expected) = names.need(StoredRole::Embedding, model)?;
        let embedding = (embedding.to_string(), expected.dtype);
        let head = names.first(StoredRole::Head, model, head_rows.checked_mul(hidden).ok_or_else(|| invalid("loop host: head size overflow"))?)?;
        let message_queries = if config.passes == 1 { Vec::new() } else {
            visit_loop::merge_message_queries(hidden, &names.all(StoredRole::MessageQueries, model)?, &names.all(StoredRole::MessageKvQueries, model)?)?
        };
        let layers = d.layers.iter().enumerate().map(|(i, l)| layer_data(d, &names, i, l)).collect::<Result<Vec<_>>>()?;
        let (embedding_norm, final_norm, output_query) = (names.all(StoredRole::EmbeddingNorm, model)?, names.all(StoredRole::FinalNorm, model)?, names.all(StoredRole::MixOutput, model)?);
        Ok(Self { hidden, stored_rows, head_rows, max_positions: size(d.max_positions)?, epsilon: d.epsilon as f32,
            logit_cap: SoftCap::new(d.caps.logits, "logit soft cap")?, activation: CappedActivation::new(d.caps.gate, d.caps.up)?,
            config, schedule, embedding, embedding_norm, final_norm, head, output_query,
            message_queries, layers, experts: ExpertCache::new(limits.expert_cache_bytes), source })
    }
    /// Load the model in the folder `root`: its config, its storage sidecar and its weight files.
    pub fn open(root: &std::path::Path, limits: Limits) -> Result<Self> {
        let (description, index, folder) = open_folder(root)?;
        Self::new(&description, &index, Box::new(folder), limits)
    }
    /// One widened expert of one stored layer, through the cache.
    pub fn expert(&self, layer: usize, expert: usize) -> Result<std::sync::Arc<capped_moe::Expert>> {
        let data = self.layers.get(layer).ok_or_else(|| invalid("loop host: layer is outside the stored layers"))?;
        let bytes = 3 * self.hidden * data.width * std::mem::size_of::<f32>();
        self.experts.get((layer, expert), bytes, || data.stacks.widen(self.source.as_ref(), self.hidden, data.width, expert))
    }
    pub fn hidden(&self) -> usize { self.hidden }
    /// Head rows that the model can write: the count of logits of one token.
    pub fn head_rows(&self) -> usize { self.head_rows }
    /// The visit list of one token, in run order.
    pub fn schedule(&self) -> &Schedule { &self.schedule }
    /// What the expert cache did so far.
    pub fn cache_counts(&self) -> Result<CacheCounts> { self.experts.counts() }
    /// The normed embedding of one token: the input of the first visit.
    pub fn embed(&self, token: u32) -> Result<Vec<f32>> {
        need((token as usize) < self.stored_rows, "a token is outside the embedding table")?;
        let row = source::floats(self.source.as_ref(), &self.embedding.0, self.embedding.1, token as usize * self.hidden, self.hidden)?;
        capped_head::embed_row(&row, &self.embedding_norm, self.epsilon)
    }
    /// Bind the host form of each operation to the weights. Do this once and keep the result.
    pub fn runner(&self) -> Result<Runner<'_>> {
        let (mut layers, mut attention, mut after_attention, mut after_experts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, l) in self.layers.iter().enumerate() {
            let gates = l.gates.as_ref().map(|gates| gates.each_ref().map(|g| GateWeights { down: &g.down, forget: &g.forget, gain: &g.gain }));
            layers.push(visit_loop::Layer::new(&self.config, i, visit_loop::LayerWeights { attention_queries: l.mix_attention.as_deref(), expert_query: &l.mix_experts, gates })?);
            let a = l.attention;
            let taps = |data| bound_taps(data, a, self.hidden);
            attention.push(sink_attention::Reference::new(a, sink_attention::Weights { q_proj: &l.q_proj, q_scale: &l.q_scale, k_proj: &l.k_proj,
                v_proj: &l.v_proj, k_conv: taps(&l.k_conv)?, v_conv: taps(&l.v_conv)?, rel_proj: &l.rel_proj, rel_scale: &l.rel_scale,
                rel_table: &l.rel_table, sink_proj: &l.sink_proj, gate_proj: &l.gate_proj, o_proj: &l.o_proj })?);
            after_attention.push(FixedTaps::new(self.hidden, l.conv_taps, &l.after_attention)?);
            after_experts.push(FixedTaps::new(self.hidden, l.conv_taps, &l.after_experts)?);
        }
        Ok(Runner { model: self, layers, attention, after_attention, after_experts })
    }
}

/// The key or value convolution of one layer, bound to its stored taps.
fn bound_taps(data: &ConvData, a: sink_attention::Config, hidden: usize) -> Result<Taps<'_>> {
    Ok(match data {
        ConvData::Fixed(weights) => Taps::Fixed(FixedTaps::new(a.kv_width(), a.conv_taps, weights)?),
        ConvData::TokenMade { static_taps, projection } => Taps::TokenMade(TokenTaps::new(a.kv_width(), a.num_kv_heads, a.conv_taps, hidden, static_taps, projection)?),
    })
}

fn layer_data(d: &LoopTextDescription, names: &Names<'_>, i: usize, l: &LoopLayerDescription) -> Result<LayerData> {
    use StoredRole::*;
    let place = Place::Layer(u32::try_from(i).map_err(|_| invalid("loop host: layer number exceeds its limit"))?);
    let (a, hidden) = (l.attention, size(d.hidden)?);
    let Ffn::Experts(shape) = l.ffn else { return Err(invalid("loop host: a layer with no routed experts is not supported")); };
    need(shape.shared_width == 0, "a shared expert is not supported")?;
    // One history set serves the four convolutions of a visit, so they share one tap count.
    need(a.key_value_conv.kernel == d.conv.kernel, "the key and value convolutions must have the tap count of the other two")?;
    let reach = match (a.window, a.position_scale) {
        (Some(window), None) => { need(a.relative.extent == window, "a window layer needs one relative column for each distance in its window")?; Reach::Window { window: size(window)? } }
        (None, Some(scale)) => Reach::Full { rel_extent: size(a.relative.extent)?, log_alpha: scale.alpha as f32, log_floor: scale.floor as f64 },
        _ => return Err(invalid("loop host: a layer must have a window, or the full reach with a position scale")),
    };
    let attention = sink_attention::Config { hidden_size: hidden, num_heads: size(a.query_heads)?, num_kv_heads: size(a.kv_heads)?,
        head_dim: size(a.head_dim)?, rel_dim: size(a.relative.dim)?, conv_taps: size(a.key_value_conv.kernel)?,
        qk_clip: check_cap(a.query_key_clip, "query and key clip")?, rel_clip: check_cap(a.relative.clip, "relative clip")?,
        rms_norm_eps: d.epsilon as f32, reach };
    attention.validate()?;
    let conv = |stream| -> Result<ConvData> { Ok(match a.key_value_conv.taps {
        ConvTaps::Fixed { .. } => ConvData::Fixed(names.all(ConvFixed(stream), place)?),
        ConvTaps::FixedAndToken { .. } => ConvData::TokenMade { static_taps: names.all(ConvStatic(stream), place)?, projection: names.all(ConvProjection(stream), place)? },
    }) };
    let input_norms = if a.separate_inputs {
        vec![names.all(QueryInputNorm, place)?, names.all(KeyInputNorm, place)?, names.all(ValueInputNorm, place)?]
    } else { vec![names.all(AttentionNorm, place)?] };
    let mix_attention = match l.attention_queries { 0 => None, _ => Some(names.all(MixAttention, place)?) };
    need(l.attention_queries == 0 || l.attention_queries == if a.separate_inputs { 3 } else { 1 }, "the block-mix query rows differ from the attention inputs of the layer")?;
    need(mix_attention.as_ref().is_none_or(|q| q.len() as u64 == l.attention_queries * d.hidden), "the block-mix query of a layer differs from its row count")?;
    let gate = |gate| -> Result<GateData> { Ok(GateData { down: names.all(Gate { gate, part: GatePart::Down }, place)?,
        forget: names.all(Gate { gate, part: GatePart::Forget }, place)?, gain: names.all(Gate { gate, part: GatePart::Gain }, place)? }) };
    let gates = if l.message_gates { Some([gate(MessageGate::Query)?, gate(MessageGate::Key)?, gate(MessageGate::Value)?, gate(MessageGate::Expert)?]) } else { None };
    let (experts, width) = (size(shape.count)?, size(shape.width)?);
    let stacks = ExpertStacks { gate_up: names.stack(place, Proj::GateUp)?, down: names.stack(place, Proj::Down)? };
    need(stacks.gate_up.experts() == experts && stacks.down.experts() == experts && stacks.gate_up.rows == 2 * width && stacks.gate_up.columns == hidden
        && stacks.down.rows == hidden && stacks.down.columns == width, "the stored expert stacks differ from the description")?;
    Ok(LayerData { attention, q_proj: names.all(Query, place)?, q_scale: names.all(QueryClip, place)?, k_proj: names.all(Key, place)?,
        v_proj: names.all(Value, place)?, k_conv: conv(Stream::Key)?, v_conv: conv(Stream::Value)?, rel_proj: names.all(RelativeProjection, place)?,
        rel_scale: names.all(RelativeClip, place)?, rel_table: names.all(RelativeTable, place)?, sink_proj: names.all(Sink, place)?,
        gate_proj: names.all(OutputGate, place)?, o_proj: names.all(Output, place)?, input_norms, mix_attention, mix_experts: names.all(MixExperts, place)?,
        gates, after_attention: names.all(ConvAfterAttention, place)?, after_experts: names.all(ConvAfterExperts, place)?, conv_taps: size(d.conv.kernel)?,
        expert_norm: names.all(ExpertNorm, place)?, router: names.all(Router, place)?, router_bias: names.all(RouterBias, place)?,
        routing: Routing::new(experts, size(shape.selected)?, d.routing.scale)?, width, stacks })
}

/// What one visit keeps for the next token: the keys and values of its
/// attention and the tails of its four convolutions.
#[derive(Debug, Clone, PartialEq)]
pub struct VisitState { pub attention: sink_attention::State, pub conv: VisitHistories }
/// The state of one chat: one [`VisitState`] for each visit, in run order.
/// A clone is a fork. A chat that met an error is refused from then on.
#[derive(Debug, Clone, PartialEq)]
pub struct Chat { visits: Vec<VisitState>, position: usize, broken: bool }
impl Chat {
    /// Tokens read so far. This is the position of the next token.
    pub fn position(&self) -> usize { self.position }
    /// The state of each visit, in run order.
    pub fn visits(&self) -> &[VisitState] { &self.visits }
}

/// Values inside one visit, for a check against a trace of another program.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// The three normed attention inputs: query, key, value. BF16 values.
    pub attention_inputs: [Vec<f32>; 3],
    /// Attention output, before its convolution.
    pub attention: Vec<f32>,
    /// Output of the convolution after attention.
    pub attended: Vec<f32>,
    /// The residual inside the block, FP32.
    pub residual: Vec<f32>,
    /// The normed input of the router and of the experts. BF16 values.
    pub expert_input: Vec<f32>,
    pub router_logits: Vec<f32>,
    /// The picked experts in fold order, with their weights.
    pub route: Route,
    /// The folded expert output, before its convolution.
    pub experts: Vec<f32>,
}
/// The stream after one visit and, when asked for, the values inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct VisitOut { pub stream: Vec<f32>, pub record: Option<Record> }

/// Which positions of a read get logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head { Every, Last, None }
/// What a read returns besides the chat state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Want { pub head: Head, pub records: bool }
/// The result of one token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenOut {
    /// The normed embedding: the input of the first visit.
    pub embedding: Vec<f32>,
    /// The stream after each visit, in run order.
    pub streams: Vec<Vec<f32>>,
    /// One record for each visit when the caller asked for records.
    pub records: Vec<Record>,
    /// The end mix: the input of the last norm, FP32.
    pub end: Vec<f32>,
    /// The output of the last norm. BF16 values.
    pub normed: Vec<f32>,
    /// Capped logits of the used head rows, when the head ran for this position.
    pub logits: Option<Vec<f32>>,
    /// The greedy next token: the largest logit, the lower token on a tie.
    pub top: Option<u32>,
}

/// The host forms of every operation, bound to the weights of one model.
pub struct Runner<'m> {
    model: &'m Model,
    layers: Vec<visit_loop::Layer<'m>>,
    attention: Vec<sink_attention::Reference<'m>>,
    after_attention: Vec<FixedTaps<'m>>,
    after_experts: Vec<FixedTaps<'m>>,
}
impl<'m> Runner<'m> {
    pub fn model(&self) -> &'m Model { self.model }
    /// A new chat: no keys, no values, every convolution tail zero.
    pub fn chat(&self) -> Result<Chat> {
        let visits = self.model.schedule.visits().iter().map(|visit| {
            let config = self.model.layers[visit.layer].attention;
            Ok(VisitState { attention: sink_attention::State::new(config)?, conv: config.histories()? })
        }).collect::<Result<Vec<_>>>()?;
        Ok(Chat { visits, position: 0, broken: false })
    }
    /// The inputs of visit `index` of one token, from the normed embedding and
    /// the streams after the earlier visits of that token.
    pub fn inputs<'a>(&self, index: usize, embedding: &'a [f32], streams: &'a [Vec<f32>]) -> Result<Inputs<'a>> {
        let visit = self.model.schedule.visits().get(index).ok_or_else(|| invalid("loop host: visit is outside the visit list"))?;
        let resolve = |source: &visit_loop::Source| -> Result<&'a [f32]> { match *source {
            visit_loop::Source::Embedding => Ok(embedding),
            visit_loop::Source::Visit(k) => streams.get(k).map(Vec::as_slice).ok_or_else(|| invalid("loop host: a visit reads a stream that is not made yet")),
        } };
        Ok(Inputs { stream: resolve(&visit.stream)?, blocks: visit.blocks.iter().map(resolve).collect::<Result<_>>()?,
            messages_from: visit.messages.as_ref().map(|m| m.from.iter().map(resolve).collect::<Result<_>>()).transpose()? })
    }
    /// Run visit `index` for the next token of `state`. An error can leave
    /// `state` half advanced: do not use it again. `threads` is the count of
    /// threads for the picked experts. It changes no word.
    pub fn visit(&self, index: usize, state: &mut VisitState, inputs: &Inputs<'_>, threads: usize, record: bool) -> Result<VisitOut> {
        let model = self.model;
        let entry: &Visit = model.schedule.visits().get(index).ok_or_else(|| invalid("loop host: visit is outside the visit list"))?;
        let (layer, data) = (&self.layers[entry.layer], &model.layers[entry.layer]);
        let messages = match (&entry.messages, &inputs.messages_from) {
            (Some(read), Some(from)) => Some(visit_loop::messages(&model.config, &model.message_queries, read.loop_layer, from)?),
            (None, None) => None,
            _ => return Err(invalid("loop host: the messages of a visit differ from its entry in the visit list")),
        };
        // Steps 2 to 4: the mix for attention, the message gates, the input norms.
        let [query, key, value] = layer.attention_inputs(&inputs.blocks, inputs.stream, messages.as_deref())?;
        let norm = |x: &[f32], which: usize| rounded_norm(x, &data.input_norms[which.min(data.input_norms.len() - 1)], model.epsilon);
        let normed = [norm(&query, 0)?, norm(&key, 1)?, norm(&value, 2)?];
        // Steps 5 to 7: attention, its convolution, the residual and the mix for the experts.
        let attention = self.attention[entry.layer].step(&mut state.attention, &mut state.conv, &normed[0], &normed[1], &normed[2])?;
        let attended = self.after_attention[entry.layer].step(state.conv.get_mut(Slot::AfterAttention), &attention)?;
        let (residual, input) = layer.expert_input(entry.boundary, &inputs.blocks, inputs.stream, &attended, messages.as_deref())?;
        // Steps 10 to 12: the norm, the router, the picked experts, their convolution, the output.
        let z = rounded_norm(&input, &data.expert_norm, model.epsilon)?;
        let logits = data.routing.logits(&data.router, &z)?;
        let route = data.routing.route(&logits, &data.router_bias)?;
        let outputs = self.expert_outputs(entry.layer, &z, &route, threads)?;
        let experts = capped_moe::fold(&route.weights, &outputs)?;
        let after = self.after_experts[entry.layer].step(state.conv.get_mut(Slot::AfterExperts), &experts)?;
        let stream = visit_loop::visit_output(&residual, &after)?;
        let record = record.then_some(Record { attention_inputs: normed, attention, attended, residual, expert_input: z, router_logits: logits, route, experts });
        Ok(VisitOut { stream, record })
    }
    /// The output of each picked expert, in pick order. Each expert is a pure
    /// function of its stored bytes and of `z`, so the thread count changes nothing.
    fn expert_outputs(&self, layer: usize, z: &[f32], route: &Route, threads: usize) -> Result<Vec<Vec<f32>>> {
        let model = self.model;
        let one = |id: usize| -> Result<Vec<f32>> { model.expert(layer, id)?.run(z, model.activation) };
        let workers = threads.clamp(1, route.ids.len().max(1));
        if workers == 1 { return route.ids.iter().map(|&id| one(id)).collect(); }
        let mut outputs: Vec<Option<Result<Vec<f32>>>> = route.ids.iter().map(|_| None).collect();
        let each = route.ids.len().div_ceil(workers);
        std::thread::scope(|scope| {
            for (ids, slots) in route.ids.chunks(each).zip(outputs.chunks_mut(each)) {
                let one = &one;
                scope.spawn(move || for (&id, slot) in ids.iter().zip(slots) { *slot = Some(one(id)); });
            }
        });
        outputs.into_iter().map(|out| out.unwrap_or_else(|| Err(invalid("loop host: an expert thread did not finish")))).collect()
    }
    /// The end of one token: the end mix, the last norm and, when asked for, the head.
    fn finish(&self, embedding: Vec<f32>, streams: Vec<Vec<f32>>, records: Vec<Record>, head: bool, threads: usize) -> Result<TokenOut> {
        let model = self.model;
        let end = model.schedule.end_mix(&model.output_query, &embedding, &streams)?;
        let normed = capped_head::last_norm(&end, &model.final_norm, model.epsilon)?;
        let (logits, top) = if head {
            let logits = self.logits(&normed, threads)?;
            let top = u32::try_from(capped_head::top_token(&logits)?).map_err(|_| invalid("loop host: token number exceeds its limit"))?;
            (Some(logits), Some(top))
        } else { (None, None) };
        Ok(TokenOut { embedding, streams, records, end, normed, logits, top })
    }
    /// Capped logits of the used head rows. Each logit is one chain over its own
    /// row, so rows can run on any number of threads.
    pub fn logits(&self, normed: &[f32], threads: usize) -> Result<Vec<f32>> {
        let model = self.model;
        let (rows, hidden) = (model.head_rows, model.hidden);
        let workers = threads.clamp(1, rows);
        if workers == 1 { return capped_head::head_logits(&model.head, hidden, rows, normed, model.logit_cap); }
        let each = rows.div_ceil(workers);
        let mut parts: Vec<Option<Result<Vec<f32>>>> = model.head.chunks(each * hidden).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (part, slot) in model.head.chunks(each * hidden).zip(parts.iter_mut()) {
                scope.spawn(move || *slot = Some(capped_head::head_logits(part, hidden, part.len() / hidden, normed, model.logit_cap)));
            }
        });
        let mut logits = Vec::with_capacity(rows);
        for part in parts { logits.extend(part.unwrap_or_else(|| Err(invalid("loop host: a head thread did not finish")))?); }
        Ok(logits)
    }
    /// Read `tokens` after the tokens that `chat` already holds. Visit `v` of
    /// token `t` needs the earlier visits of token `t` and visit `v` of token
    /// `t - 1`. So all visits with the same `v + t` are independent: they run
    /// side by side on up to `threads` threads. The words do not depend on
    /// `threads` or on how a text is cut into reads.
    pub fn read(&self, chat: &mut Chat, tokens: &[u32], want: Want, threads: usize) -> Result<Vec<TokenOut>> {
        need(!chat.broken, "this chat met an error before and cannot go on")?;
        let model = self.model;
        let (count, visits, threads) = (tokens.len(), model.schedule.visits().len(), threads.max(1));
        need(chat.visits.len() == visits, "the chat belongs to another model")?;
        need(chat.position.checked_add(count).is_some_and(|end| end <= model.max_positions), "the text is longer than the position limit of the model")?;
        if count == 0 { return Ok(Vec::new()); }
        let embeddings = tokens.iter().map(|&token| model.embed(token)).collect::<Result<Vec<_>>>()?;
        let mut streams: Vec<Vec<Vec<f32>>> = (0..count).map(|_| Vec::with_capacity(visits)).collect();
        let mut records: Vec<Vec<Record>> = (0..count).map(|_| Vec::new()).collect();
        chat.broken = true;
        for diagonal in 0..count + visits - 1 {
            let (first, last) = (diagonal.saturating_sub(count - 1), diagonal.min(visits - 1));
            // One cell for each visit of this diagonal: visit `first + i` of token `diagonal - first - i`.
            let mut cells: Vec<(usize, &mut VisitState, Option<Result<VisitOut>>)> = chat.visits[first..=last].iter_mut().enumerate()
                .map(|(i, state)| (first + i, state, None)).collect();
            let workers = threads.min(cells.len());
            let (each, inner) = (cells.len().div_ceil(workers), (threads / cells.len()).max(1));
            let run = |cell: &mut (usize, &mut VisitState, Option<Result<VisitOut>>)| {
                let token = diagonal - cell.0;
                cell.2 = Some(self.inputs(cell.0, &embeddings[token], &streams[token])
                    .and_then(|inputs| self.visit(cell.0, cell.1, &inputs, inner, want.records)));
            };
            if workers == 1 { cells.iter_mut().for_each(run); } else {
                std::thread::scope(|scope| { for part in cells.chunks_mut(each) { let run = &run; scope.spawn(move || part.iter_mut().for_each(run)); } });
            }
            let done: Vec<(usize, Option<Result<VisitOut>>)> = cells.into_iter().map(|(visit, _, out)| (visit, out)).collect();
            for (visit, out) in done {
                let out = out.unwrap_or_else(|| Err(invalid("loop host: a visit thread did not finish")))?;
                streams[diagonal - visit].push(out.stream);
                records[diagonal - visit].extend(out.record);
            }
        }
        let mut outs = Vec::with_capacity(count);
        for (token, ((embedding, streams), records)) in embeddings.into_iter().zip(streams).zip(records).enumerate() {
            let head = match want.head { Head::Every => true, Head::Last => token + 1 == count, Head::None => false };
            outs.push(self.finish(embedding, streams, records, head, threads)?);
        }
        chat.position += count;
        chat.broken = false;
        Ok(outs)
    }
    /// Greedy text: read `prompt`, then write up to `new_tokens` tokens, one a
    /// read. Each new token is the top token of the position before it. The
    /// run ends early after a token of `stop`. Returns the new tokens.
    pub fn greedy(&self, chat: &mut Chat, prompt: &[u32], new_tokens: usize, stop: &[u32], threads: usize) -> Result<Vec<u32>> {
        need(!prompt.is_empty(), "a greedy run needs a prompt of one token or more")?;
        let want = Want { head: Head::Last, records: false };
        let top = |outs: Vec<TokenOut>| outs.last().and_then(|out| out.top).ok_or_else(|| invalid("loop host: a read gave no top token"));
        let mut written = Vec::with_capacity(new_tokens);
        let mut next = if new_tokens == 0 { self.read(chat, prompt, Want { head: Head::None, records: false }, threads)?; return Ok(written) }
            else { top(self.read(chat, prompt, want, threads)?)? };
        loop {
            written.push(next);
            if written.len() == new_tokens || stop.contains(&next) { return Ok(written); }
            next = top(self.read(chat, &[next], want, threads)?)?;
        }
    }
}

#[cfg(test)]
mod tests;
