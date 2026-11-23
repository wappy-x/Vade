//! Host form of the visit driver of a looped decoder: the visit list, the
//! state that each pass starts from, the messages from the pass before and
//! the gates that mix them in. A visit is one run of one stored layer for one
//! token. The visit list is data made from config fields. No list is fixed here.
//! Source: Transformers v5.24.0, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`,
//! `models/inkling2_loop/`. In `configuration_inkling2_loop.py`: the loop
//! fields and their checks, lines 139-176. In `modeling_inkling2_loop.py`:
//! the message gate, lines 79-96. The messages, lines 145-173. The place of
//! the mixes and gates in a visit, lines 876-910, 937-997, 1022-1062 and
//! 1091-1149. The visit order and the restart, lines 1350-1443. In
//! `core_model_loading.py`: the stored message query layout, lines 823-849.
//! This file holds its own code and none of that source.
//! Arithmetic is f32 with two BF16 points: the inner product of a gate and the
//! stream after a visit. A product and a sum are separate f32 operations,
//! never a fused multiply-add. The sum orders are the engine's: a chain in
//! rising index from zero for a product, and for the norms inside a gate the
//! order of the weighted norm kernel. `tanh` and the BF16 round are those of
//! the shared `fixed_math` module, not the platform's. This module does not
//! enable a model plan or change serving. See `visit_loop/README.md`.
use crate::{Result, invalid, block_mix::{self, join_lanes}, fixed_math::{rn as bf16, tanh},
    json::{self, Value, field}, shapes::{CONFIG_BYTES, MAX_HIDDEN, MAX_LAYERS}};

/// Messages that a loop layer reads, in this order: query, key, value, expert.
pub const MESSAGES: usize = 4;
/// Ceiling on the visits of one token. A host resource limit, not a source rule.
pub const MAX_VISITS: usize = 65_536;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(invalid(format!("visit loop: {message}"))) }
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| invalid("visit loop: dimension product overflow"))
}
fn finite(x: &[f32]) -> Result<()> { need(x.iter().all(|v| v.is_finite()), "values must be finite") }
/// The sum of squares in the order of the engine's weighted norm kernel
/// (`norm` in `tp/src/causal_layer.rs`): chunks of 128 values with 32 lanes
/// each, the lanes joined, then the chunk sums in 32 lanes, joined again.
fn chunked_square_sum(x: &[f32]) -> f32 {
    let mut outer = [0.0f32; 32];
    for (chunk, values) in x.chunks(128).enumerate() {
        let mut lanes = [0.0f32; 32];
        for (i, &v) in values.iter().enumerate() { lanes[i % 32] += v * v; }
        outer[chunk % 32] += join_lanes(lanes);
    }
    join_lanes(outer)
}
/// The plain norm inside a gate, rounded to BF16: `rn(x * scale)` with
/// `scale = 1 / sqrt(sum / width + eps)`. The device runs the weighted norm
/// kernel here with no weight, so the sum of squares has that kernel's order.
fn gate_norm(x: &[f32], eps: f32) -> Result<Vec<f32>> {
    finite(x)?;
    let variance = chunked_square_sum(x) / x.len() as f32 + eps;
    need(variance.is_finite(), "gate norm exceeds f32 range")?;
    let scale = 1.0 / variance.sqrt();
    Ok(x.iter().map(|&v| bf16(v * scale)).collect())
}

/// Sizes of the loop. Each field comes from the text config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    /// `num_hidden_layers`: the count of stored layers.
    pub layers: usize,
    /// `n_layers_per_attn_res_block`: the first layer of each block is a boundary.
    pub block_layers: usize,
    /// `loop_config.core_start`: the first stored layer of the loop.
    pub loop_start: usize,
    /// `loop_config.core_end`: one after the last stored layer of the loop.
    pub loop_end: usize,
    /// `loop_config.num_passes`: how many times the loop layers run.
    pub passes: usize,
    /// `loop_config.gate_rank`: the inner width of a message gate.
    pub gate_width: usize,
    pub rms_norm_eps: f32,
}
impl Config {
    /// Read a text config or its enclosing `text_config` object. Every field
    /// is required: no default fills a missing one. A model name does not
    /// select behavior. A variable pass count is refused.
    pub fn from_json(input: &[u8]) -> Result<Self> {
        let doc = json::parse(input, json::Limits { bytes: CONFIG_BYTES, ..Default::default() })?;
        let root = doc.object()?;
        let o = root.get("text_config").map(Value::object).transpose()?.unwrap_or(root);
        let l = field(o, "loop_config")?.object()?;
        if let Some(v) = l.get("is_varloop") { need(!v.boolean()?, "a variable pass count is not supported")?; }
        let count = |value: &Value| -> Result<usize> {
            usize::try_from(value.u64()?).map_err(|_| invalid("visit loop: count exceeds usize"))
        };
        let config = Self { hidden_size: count(field(o, "hidden_size")?)?, layers: count(field(o, "num_hidden_layers")?)?,
            block_layers: count(field(o, "n_layers_per_attn_res_block")?)?, loop_start: count(field(l, "core_start")?)?,
            loop_end: count(field(l, "core_end")?)?, passes: count(field(l, "num_passes")?)?,
            gate_width: count(field(l, "gate_rank")?)?, rms_norm_eps: field(o, "rms_norm_eps")?.f64()? as f32 };
        config.validate()?;
        Ok(config)
    }
    /// The source checks the pass count, a loop start above zero and both
    /// loop edges on a block edge (lines 157-169). The other checks are
    /// limits of this host form.
    pub fn validate(&self) -> Result<()> {
        need(self.hidden_size > 0 && self.hidden_size as u64 <= MAX_HIDDEN, "hidden size is outside its limit")?;
        need(self.layers > 0 && self.layers as u64 <= MAX_LAYERS, "layer count is outside its limit")?;
        need(self.block_layers > 0, "a block must hold at least one layer")?;
        need(self.passes > 0, "the pass count must be at least 1")?;
        need(self.loop_start > 0, "the loop must start after the first layer")?;
        need(self.loop_start.is_multiple_of(self.block_layers) && self.loop_end.is_multiple_of(self.block_layers),
            "the loop must start and end on a block edge")?;
        need(self.loop_start < self.loop_end && self.loop_end <= self.layers, "the loop must lie inside the stored layers")?;
        need(self.gate_width > 0 && self.gate_width as u64 <= MAX_HIDDEN, "gate width is outside its limit")?;
        need(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "invalid RMS norm epsilon")?;
        let extra = mul(self.passes - 1, self.loop_layers())?;
        need(self.layers.checked_add(extra).is_some_and(|visits| visits <= MAX_VISITS), "visit count is outside its limit")?;
        let wide = mul(2, self.hidden_size)?;
        need(mul(mul(self.gate_width, wide)?, std::mem::size_of::<f32>())? <= isize::MAX as usize, "gate exceeds the address range")
    }
    /// Stored layers inside the loop.
    pub fn loop_layers(&self) -> usize { self.loop_end.saturating_sub(self.loop_start) }
    /// Layer visits for one token.
    pub fn visits(&self) -> usize { self.layers.saturating_add(self.passes.saturating_sub(1).saturating_mul(self.loop_layers())) }
}

/// A vector that a visit reads: the normed embedding of the token, or the
/// stream after an earlier visit of the same token. Visits count in run order
/// from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source { Embedding, Visit(usize) }

/// Where the messages of one visit come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageRead {
    /// The layer minus the loop start. It selects the message queries.
    pub loop_layer: usize,
    /// The block outputs of the pass before, oldest first. The last entry is
    /// the stream that the pass ended with.
    pub from: Vec<Source>,
}

/// One entry of the visit list: what the visit runs, reads and writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Visit {
    /// Stored layer. It selects the weights. Each pass uses the same weights.
    pub layer: usize,
    /// Pass of the loop, from 1. A layer outside the loop has pass 1.
    pub pass: usize,
    /// State slot for the keys, values and convolution tails that this visit
    /// writes. Each visit has its own slot. Pass 1 uses the layer number.
    /// Later passes follow the stored layers, in pass order (source line 1405).
    pub slot: usize,
    /// First layer of a block. Its input joins the block list. The input is
    /// not added to the stream.
    pub boundary: bool,
    /// The stream that the visit reads. The first loop layer of each later
    /// pass reads the stream that pass 1 started from.
    pub stream: Source,
    /// The finished block vectors before the visit, oldest first.
    pub blocks: Vec<Source>,
    /// Present from pass 2.
    pub messages: Option<MessageRead>,
}

/// The visit list of one token and the inputs of the end mix.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    config: Config,
    visits: Vec<Visit>,
    end_blocks: Vec<Source>,
    end_stream: Source,
}
/// What the driver hands to one visit.
#[derive(Debug, Clone)]
pub struct Inputs<'a> {
    pub stream: &'a [f32],
    pub blocks: Vec<&'a [f32]>,
    pub messages_from: Option<Vec<&'a [f32]>>,
}
impl Schedule {
    /// Make the visit list from the config. Layers before the loop run once.
    /// Pass 1 goes on from them. Each later pass restarts from the stream and
    /// the block list that pass 1 started from, and reads the block outputs
    /// of the pass before as messages. Layers after the loop go on from the
    /// last pass. A boundary layer puts its input on the block list.
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let c = config;
        let mut visits: Vec<Visit> = Vec::new();
        visits.try_reserve_exact(c.visits()).map_err(|_| invalid("visit loop: cannot reserve the visit list"))?;
        let (mut stream, mut blocks) = (Source::Embedding, Vec::new());
        let mut push = |stream: &mut Source, blocks: &mut Vec<Source>, layer: usize, pass: usize, messages: Option<MessageRead>| {
            let boundary = layer.is_multiple_of(c.block_layers);
            let slot = if pass == 1 { layer } else { c.layers + (pass - 2) * c.loop_layers() + (layer - c.loop_start) };
            let index = visits.len();
            visits.push(Visit { layer, pass, slot, boundary, stream: *stream, blocks: blocks.clone(), messages });
            if boundary { blocks.push(*stream); }
            *stream = Source::Visit(index);
        };
        for layer in 0..c.loop_start { push(&mut stream, &mut blocks, layer, 1, None); }
        let start = (stream, blocks.clone());
        for layer in c.loop_start..c.loop_end { push(&mut stream, &mut blocks, layer, 1, None); }
        for pass in 2..=c.passes {
            // The first loop layer is a boundary. It put the loop input on the
            // list. That entry is not a block output of the pass: skip it.
            let made = blocks.get(start.1.len() + 1..).ok_or_else(|| invalid("visit loop: the loop made no block list entry"))?;
            let from: Vec<Source> = made.iter().copied().chain([stream]).collect();
            (stream, blocks) = start.clone();
            for layer in c.loop_start..c.loop_end {
                let read = MessageRead { loop_layer: layer - c.loop_start, from: from.clone() };
                push(&mut stream, &mut blocks, layer, pass, Some(read));
            }
        }
        for layer in c.loop_end..c.layers { push(&mut stream, &mut blocks, layer, 1, None); }
        Ok(Self { config, visits, end_blocks: blocks, end_stream: stream })
    }
    pub fn config(&self) -> &Config { &self.config }
    /// The visits in run order.
    pub fn visits(&self) -> &[Visit] { &self.visits }
    /// The finished block vectors that the end mix reads, oldest first.
    pub fn end_blocks(&self) -> &[Source] { &self.end_blocks }
    /// The stream that the end mix reads.
    pub fn end_stream(&self) -> Source { self.end_stream }
    /// The largest count of vectors that one token must hold at one time: the
    /// inputs of the running visit and every vector that a later visit or the
    /// end mix still reads.
    pub fn peak_held(&self) -> usize {
        let end = self.visits.len();
        // Position 0 is the embedding. Position k + 1 is the stream after visit k.
        let position = |source: &Source| match *source { Source::Embedding => 0, Source::Visit(k) => k + 1 };
        let mut last = vec![0usize; end + 1];
        let mut read = |source: &Source, at: usize| { let p = position(source); last[p] = last[p].max(at); };
        for (at, visit) in self.visits.iter().enumerate() {
            read(&visit.stream, at);
            for source in &visit.blocks { read(source, at); }
            for source in visit.messages.iter().flat_map(|m| &m.from) { read(source, at); }
        }
        read(&self.end_stream, end);
        for source in &self.end_blocks { read(source, end); }
        // A vector at position p exists from visit p on. It is held up to its last reader.
        (0..=end).map(|at| (0..=at).filter(|&p| last[p] >= at).count()).max().unwrap_or(0)
    }
    /// Run the visits of one token in order. `visit` gets the entry and its
    /// inputs and returns the stream after the visit. The result holds the
    /// stream after each visit, in run order.
    pub fn run<F>(&self, embedding: &[f32], mut visit: F) -> Result<Vec<Vec<f32>>>
    where F: FnMut(&Visit, &Inputs<'_>) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        need(embedding.len() == hidden, "embedding shape differs from the hidden size")?;
        finite(embedding)?;
        let mut streams: Vec<Vec<f32>> = Vec::with_capacity(self.visits.len());
        for entry in &self.visits {
            let out = {
                let inputs = Inputs { stream: resolve(entry.stream, embedding, &streams),
                    blocks: entry.blocks.iter().map(|&s| resolve(s, embedding, &streams)).collect(),
                    messages_from: entry.messages.as_ref().map(|m| m.from.iter().map(|&s| resolve(s, embedding, &streams)).collect()) };
                visit(entry, &inputs)?
            };
            need(out.len() == hidden, "visit output shape differs from the hidden size")?;
            finite(&out)?;
            streams.push(out);
        }
        Ok(streams)
    }
    /// The mix before the last norm. `streams` is the result of `run`.
    pub fn end_mix(&self, output_query: &[f32], embedding: &[f32], streams: &[Vec<f32>]) -> Result<Vec<f32>> {
        need(streams.len() == self.visits.len() && embedding.len() == self.config.hidden_size, "end mix inputs differ from the visit list")?;
        let blocks: Vec<&[f32]> = self.end_blocks.iter().map(|&s| resolve(s, embedding, streams)).collect();
        block_mix::end_mix(self.config.hidden_size, output_query, &blocks, resolve(self.end_stream, embedding, streams))
    }
}
// A source always names an earlier visit: `Schedule::new` makes no other.
fn resolve<'a>(source: Source, embedding: &'a [f32], streams: &'a [Vec<f32>]) -> &'a [f32] {
    match source { Source::Embedding => embedding, Source::Visit(k) => &streams[k] }
}

/// The messages of one loop layer in the order query, key, value, expert.
/// `queries` is row-major `[loop layers][4][hidden]`. `from` holds the block
/// outputs of the pass before. Its last entry is the stream that the pass
/// ended with. A message is a block mix over those vectors (source lines
/// 145-173): a blend of the normed vectors. The result is f32.
pub fn messages(config: &Config, queries: &[f32], loop_layer: usize, from: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
    config.validate()?;
    let rows = mul(MESSAGES, config.hidden_size)?;
    need(loop_layer < config.loop_layers() && queries.len() == mul(config.loop_layers(), rows)?,
        "message query shape differs from the config")?;
    let (stream, blocks) = from.split_last().ok_or_else(|| invalid("visit loop: a message needs the pass before"))?;
    Ok(block_mix::mix(config.hidden_size, &queries[loop_layer * rows..(loop_layer + 1) * rows], blocks, stream)?.values)
}

/// Put the two stored message query tensors in the order that `messages`
/// reads. Each stored tensor is `[2 * loop layers][hidden]`. Rows `2j` and
/// `2j + 1` of the first are the query and the expert query of loop layer
/// `j`. Rows `2j` and `2j + 1` of the second are its key and value queries.
/// The result is `[loop layers][4][hidden]`: query, key, value, expert.
pub fn merge_message_queries(hidden: usize, query_expert: &[f32], key_value: &[f32]) -> Result<Vec<f32>> {
    let pair = mul(2, hidden)?;
    need(hidden > 0 && query_expert.len() == key_value.len() && query_expert.len().is_multiple_of(pair), "stored message query shape")?;
    let mut merged = Vec::with_capacity(mul(2, query_expert.len())?);
    for (qe, kv) in query_expert.chunks_exact(pair).zip(key_value.chunks_exact(pair)) {
        merged.extend_from_slice(&qe[..hidden]);
        merged.extend_from_slice(kv);
        merged.extend_from_slice(&qe[hidden..]);
    }
    Ok(merged)
}

/// Borrowed weights of one message gate.
#[derive(Debug, Clone, Copy)]
pub struct GateWeights<'a> {
    /// Row-major `[gate width][2 * hidden]`. BF16 values, widened to f32.
    pub down: &'a [f32],
    /// Row-major `[hidden][gate width]`, f32.
    pub forget: &'a [f32],
    /// Row-major `[hidden][gate width]`, f32.
    pub gain: &'a [f32],
}
/// One message gate: `(1 - forget) * value + gain * message` (lines 79-96).
#[derive(Debug, Clone, Copy)]
pub struct Gate<'a> { hidden: usize, width: usize, eps: f32, weights: GateWeights<'a> }
impl<'a> Gate<'a> {
    pub fn new(config: &Config, weights: GateWeights<'a>) -> Result<Self> {
        config.validate()?;
        let (hidden, width) = (config.hidden_size, config.gate_width);
        let cells = mul(hidden, width)?;
        need(weights.down.len() == mul(2, cells)? && weights.forget.len() == cells && weights.gain.len() == cells,
            "gate weight shape differs from the config")?;
        for values in [weights.down, weights.forget, weights.gain] { finite(values)?; }
        need(weights.down.iter().all(|&v| bf16(v).to_bits() == v.to_bits()), "the down weight must hold BF16 values")?;
        Ok(Self { hidden, width, eps: config.rms_norm_eps, weights })
    }
    /// Mix a message into a value. Order:
    /// 1. Plain norm of the value and of the message, in f32, with the config
    ///    epsilon and the sum order of the weighted norm kernel. Each number
    ///    is rounded to BF16. Put the two side by side.
    /// 2. Inner product with the down weight: add `weight * input` in f32 as
    ///    a chain in rising input order from zero. Round the sum to BF16 once.
    /// 3. Forget and gain: the f32 dot product with the inner vector as a
    ///    chain in rising order from zero, then the hyperbolic tangent.
    /// 4. `(1 - forget) * value + gain * message`, element by element, in
    ///    f32: two products, then one sum.
    ///
    /// The source does not give the order of the additions inside its BF16
    /// product. The orders above are the engine's own.
    pub fn apply(&self, value: &[f32], message: &[f32]) -> Result<Vec<f32>> {
        need(value.len() == self.hidden && message.len() == self.hidden, "gate input shape differs from the hidden size")?;
        let mut joined = gate_norm(value, self.eps)?;
        joined.extend(gate_norm(message, self.eps)?);
        let mut inner = Vec::with_capacity(self.width);
        for row in self.weights.down.chunks_exact(2 * self.hidden) {
            let mut sum = 0.0f32;
            for (weight, input) in row.iter().zip(&joined) { sum += weight * input; }
            need(sum.is_finite(), "gate inner product exceeds f32 range")?;
            inner.push(bf16(sum));
        }
        finite(&inner)?;
        let mut out = Vec::with_capacity(self.hidden);
        for d in 0..self.hidden {
            let (mut forget, mut gain) = (0.0f32, 0.0f32);
            for (k, g) in inner.iter().enumerate() {
                forget += self.weights.forget[d * self.width + k] * g;
                gain += self.weights.gain[d * self.width + k] * g;
            }
            out.push((1.0 - tanh(forget)) * value[d] + tanh(gain) * message[d]);
        }
        finite(&out)?;
        Ok(out)
    }
}

/// Borrowed mix queries and gates of one stored layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerWeights<'a> {
    /// Queries of the attention mix: one row, or three rows in the order
    /// query, key, value. The first stored layer has none.
    pub attention_queries: Option<&'a [f32]>,
    /// The one query of the expert mix.
    pub expert_query: &'a [f32],
    /// Gates of a loop layer in the order query, key, value, expert.
    pub gates: Option<[GateWeights<'a>; MESSAGES]>,
}
/// The stream arithmetic of one stored layer: the mixes, the gates and the
/// residual inside the block. The caller supplies attention and the experts.
#[derive(Debug, Clone)]
pub struct Layer<'a> {
    hidden: usize,
    attention_queries: Option<&'a [f32]>,
    expert_query: &'a [f32],
    gates: Option<[Gate<'a>; MESSAGES]>,
}
impl<'a> Layer<'a> {
    /// The first stored layer has no attention mix (source lines 862 and
    /// 1006). A layer has gates only inside the loop. A loop layer needs
    /// them when the config has more than one pass.
    pub fn new(config: &Config, layer: usize, weights: LayerWeights<'a>) -> Result<Self> {
        config.validate()?;
        let hidden = config.hidden_size;
        need(layer < config.layers, "layer is outside the stored layers")?;
        match weights.attention_queries {
            None => need(layer == 0, "only the first layer has no attention mix")?,
            Some(q) => {
                need(layer > 0, "the first layer has no attention mix")?;
                need(q.len() == hidden || q.len() == mul(3, hidden)?, "the attention mix takes one query or three")?;
                finite(q)?;
            }
        }
        need(weights.expert_query.len() == hidden, "the expert mix takes one query")?;
        finite(weights.expert_query)?;
        let inside = (config.loop_start..config.loop_end).contains(&layer);
        need(inside || weights.gates.is_none(), "a layer outside the loop has no gates")?;
        need(!inside || config.passes == 1 || weights.gates.is_some(), "a loop layer needs its gates")?;
        let gates = match weights.gates {
            Some(w) => Some([Gate::new(config, w[0])?, Gate::new(config, w[1])?, Gate::new(config, w[2])?, Gate::new(config, w[3])?]),
            None => None,
        };
        Ok(Self { hidden, attention_queries: weights.attention_queries, expert_query: weights.expert_query, gates })
    }
    fn gated<'m>(&'m self, messages: Option<&'m [Vec<f32>]>) -> Result<Option<(&'m [Gate<'a>; MESSAGES], &'m [Vec<f32>])>> {
        let Some(messages) = messages else { return Ok(None) };
        need(messages.len() == MESSAGES, "a loop layer reads four messages")?;
        let gates = self.gates.as_ref().ok_or_else(|| invalid("visit loop: messages reached a layer with no gates"))?;
        Ok(Some((gates, messages)))
    }
    /// The query, key and value inputs of the attention norms, in f32.
    /// One attention query gives one mix for all three. Three queries give
    /// one mix each. With no query the stream is used as it is. With
    /// messages, each input then goes through its own gate. With no
    /// messages the gates are skipped. They are not fed zeros.
    pub fn attention_inputs(&self, blocks: &[&[f32]], stream: &[f32], messages: Option<&[Vec<f32>]>) -> Result<[Vec<f32>; 3]> {
        need(stream.len() == self.hidden, "stream shape differs from the hidden size")?;
        let mut inputs = match self.attention_queries {
            None => { finite(stream)?; [stream.to_vec(), stream.to_vec(), stream.to_vec()] }
            Some(queries) => {
                let mut mixed = block_mix::mix(self.hidden, queries, blocks, stream)?.values;
                match mixed.len() {
                    1 => { let one = mixed.swap_remove(0); [one.clone(), one.clone(), one] }
                    _ => { let value = mixed.swap_remove(2); let key = mixed.swap_remove(1); [mixed.swap_remove(0), key, value] }
                }
            }
        };
        if let Some((gates, messages)) = self.gated(messages)? {
            for i in 0..3 { inputs[i] = gates[i].apply(&inputs[i], &messages[i])?; }
        }
        Ok(inputs)
    }
    /// The residual inside the block and the input of the expert norm.
    /// `attended` is the output of the convolution after attention. In a
    /// boundary layer the stream joins the block list and the residual is
    /// `attended` alone. In other layers the residual is `stream + attended`.
    /// The residual stays f32. It is not rounded. The expert input is the mix
    /// over the block list and the residual, then the expert gate when there
    /// are messages. Returns `(residual, expert input)`.
    pub fn expert_input(&self, boundary: bool, blocks: &[&[f32]], stream: &[f32], attended: &[f32], messages: Option<&[Vec<f32>]>) -> Result<(Vec<f32>, Vec<f32>)> {
        need(stream.len() == self.hidden && attended.len() == self.hidden, "residual shape differs from the hidden size")?;
        finite(stream)?;
        finite(attended)?;
        let residual: Vec<f32> = if boundary { attended.to_vec() } else { stream.iter().zip(attended).map(|(&keep, &add)| keep + add).collect() };
        finite(&residual)?;
        let mut list = blocks.to_vec();
        if boundary { list.push(stream); }
        let mut input = block_mix::mix(self.hidden, self.expert_query, &list, &residual)?.values.swap_remove(0);
        if let Some((gates, messages)) = self.gated(messages)? { input = gates[3].apply(&input, &messages[3])?; }
        Ok((residual, input))
    }
}

/// The stream after a visit: the f32 sum of the residual and the output of
/// the convolution after the experts, rounded to BF16 once.
pub fn visit_output(residual: &[f32], experts: &[f32]) -> Result<Vec<f32>> {
    need(residual.len() == experts.len(), "visit output shape mismatch")?;
    let out: Vec<f32> = residual.iter().zip(experts).map(|(&keep, &add)| bf16(keep + add)).collect();
    finite(&out)?;
    Ok(out)
}

/// The two parts of a visit that this module does not own.
pub trait Body {
    /// The norms, attention and the convolution after it. The inputs are
    /// f32 and are not normed yet. Returns the output of that convolution.
    fn attention(&mut self, visit: &Visit, query: &[f32], key: &[f32], value: &[f32]) -> Result<Vec<f32>>;
    /// The norm, the router, the experts and the convolution after them.
    /// The input is f32 and is not normed yet. Returns the output of that
    /// convolution.
    fn experts(&mut self, visit: &Visit, input: &[f32]) -> Result<Vec<f32>>;
}
/// Borrowed weights of the driver.
#[derive(Debug, Clone)]
pub struct Weights<'a> {
    /// One entry for each stored layer.
    pub layers: Vec<LayerWeights<'a>>,
    /// `[loop layers][4][hidden]`. Empty when the config has one pass.
    pub message_queries: &'a [f32],
    /// The one query of the end mix.
    pub output_query: &'a [f32],
}
/// The streams of one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The stream after each visit, in run order.
    pub streams: Vec<Vec<f32>>,
    /// The result of the end mix. The caller applies the last norm next.
    pub end: Vec<f32>,
}
/// Runs every visit of one token over a caller-supplied body.
#[derive(Debug, Clone)]
pub struct Driver<'a> {
    schedule: Schedule,
    layers: Vec<Layer<'a>>,
    message_queries: &'a [f32],
    output_query: &'a [f32],
}
impl<'a> Driver<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Result<Self> {
        let schedule = Schedule::new(config)?;
        need(weights.layers.len() == config.layers, "one weight entry for each stored layer is required")?;
        let queries = if config.passes == 1 { 0 } else { mul(config.loop_layers(), mul(MESSAGES, config.hidden_size)?)? };
        need(weights.message_queries.len() == queries, "message query shape differs from the config")?;
        need(weights.output_query.len() == config.hidden_size, "the end mix takes one query")?;
        finite(weights.message_queries)?;
        finite(weights.output_query)?;
        let layers = weights.layers.iter().enumerate().map(|(layer, w)| Layer::new(&config, layer, *w)).collect::<Result<Vec<_>>>()?;
        Ok(Self { schedule, layers, message_queries: weights.message_queries, output_query: weights.output_query })
    }
    pub fn schedule(&self) -> &Schedule { &self.schedule }
    /// The stream arithmetic of one stored layer.
    pub fn layer(&self, layer: usize) -> Option<&Layer<'a>> { self.layers.get(layer) }
    /// Run one token. The body keeps the keys, values and convolution tails
    /// of earlier tokens, by `Visit::slot`. `embedding` is the normed
    /// embedding of the token.
    pub fn token<B: Body>(&self, embedding: &[f32], body: &mut B) -> Result<Token> {
        let config = self.schedule.config;
        let streams = self.schedule.run(embedding, |visit, inputs| {
            let layer = &self.layers[visit.layer];
            let read = match (&visit.messages, &inputs.messages_from) {
                (Some(read), Some(from)) => Some(messages(&config, self.message_queries, read.loop_layer, from)?),
                _ => None,
            };
            let [query, key, value] = layer.attention_inputs(&inputs.blocks, inputs.stream, read.as_deref())?;
            let attended = body.attention(visit, &query, &key, &value)?;
            let (residual, input) = layer.expert_input(visit.boundary, &inputs.blocks, inputs.stream, &attended, read.as_deref())?;
            let experts = body.experts(visit, &input)?;
            visit_output(&residual, &experts)
        })?;
        let end = self.schedule.end_mix(self.output_query, embedding, &streams)?;
        Ok(Token { streams, end })
    }
}

#[cfg(test)]
mod tests;
