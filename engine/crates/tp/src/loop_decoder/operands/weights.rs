//! The weights of the driver (`mix::Weights`) on a loaded checkpoint.
//!
//! A matrix that the checkpoint stores as BF16 stays with the owner of the weights, and a product runs there on
//! the stored bytes (`ResidentProduct`). The picked expert of a stack is a word range of the stack. Its product
//! with one input row reads the packed stack where the compute has a packed product for the shape, and the widened
//! stack else; both give the same words. Every other
//! weight is small and is held as host words, read from the weight files one time, in the layout that its name
//! in `mix::Weight` gives. A matrix that is stored as 32-bit values, and a projection whose rows change order,
//! are host words too; their product is the plain chain product of the compute (`mix::chain_product`).
//!
//! Layouts made here from the stored form:
//! - Fixed taps by lag: `[channel, lag]`, and lag 0 meets the current token. The description says which stored
//!   tap that is.
//! - The projection of token-made taps: row `head * taps + lag`.
//! - The message queries: for each loop layer the query, key, value and expert row, from the two stored tensors.
use super::{ConvOperands, InputNorms, Operands, Resident, load::Files};
use crate::{Result, bounded_decoder::Compute, loop_decoder::mix::{Message, Weight, Weights, chain_product, check_runs, gathered_runs}};
use engine_model::{plan::{description::{ConvTaps, TapOrder}, loop_load::LoopCheckpoint, loop_packed_unit::UnitShape}, safetensors::Dtype, shapes::Proj};
use std::{collections::BTreeMap, ops::Range};

/// A compute that can run a product on a matrix that the owner of the weights holds as value `V`.
pub trait ResidentProduct<V>: Compute {
    /// `words` of `weight` hold `rows x columns` BF16 values, two a word, row-major. The result is `rows` 32-bit
    /// words: one chain a row over the columns in rising order from +0.0, each product rounded before its add,
    /// the sum not rounded.
    fn resident_product(&self, weight: &V, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value) -> Result<Self::Value>;
    /// `resident_product` for `inputs` rows of `x` (`[inputs][columns]`). The result is `[inputs][rows]`, row-major,
    /// and each word is the word of `resident_product` on its input row. The body here runs one product a row and
    /// joins them. A compute with a launch for many rows gives its own body with the same words.
    fn resident_product_rows(&self, weight: &V, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value, inputs: usize) -> Result<Self::Value> {
        if inputs == 0 || inputs.checked_mul(columns) != Some(self.words(x)) { return Err("resident product: the input differs from the input rows x the columns of the weight".into()); }
        if inputs == 1 { return self.resident_product(weight, words, rows, columns, x); }
        let outs = (0..inputs).map(|input| self.resident_product(weight, words.clone(), rows, columns, &self.range(x, input * columns, columns)?)).collect::<Result<Vec<_>>>()?;
        self.join(&outs.iter().collect::<Vec<_>>())
    }
    /// The products of the expert runs of a step of rows with no gather (`Weights::product_runs`). `runs` holds the
    /// matrix of each run in launch order: its value, its words, its rows and its count of launch rows; all runs have
    /// the same rows and `columns` columns. Launch row `i` reads row `input[i]` of `x` (row `i` without a table); row
    /// `j` of the result is launch row `output[j]` (row `j` without a table). The tables passed `mix::check_runs`.
    /// None: this compute has no such body for these runs, and the caller gathers. The body here gives None.
    fn resident_product_runs(&self, runs: &[(&V, Range<usize>, usize, usize)], x: &Self::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<Self::Value>> {
        let _ = (runs, x, columns, input, output);
        Ok(None)
    }
    /// `words` of `weight` hold the unit of one packed matrix (`loop_packed_unit`). The result is the result of
    /// `resident_product` on the same matrix widened to BF16, gate rows first, word for word. None: this compute
    /// has no packed product for this shape now, and the caller reads the widened matrix.
    fn packed_product(&self, _weight: &V, _words: Range<usize>, _unit: UnitShape, _x: &Self::Value) -> Result<Option<Self::Value>> { Ok(None) }
    /// The packed products of several matrices of one shape, each `units[e]` of `weight`, in one call: matrix `e` times
    /// `x` (`shared`) or times words `e * columns ..` of `x`. The result is the products back to back, each the words
    /// of `packed_product` on its matrix. None: no grouped product for this shape now; the caller takes one a matrix.
    fn packed_group(&self, _weight: &V, _units: &[Range<usize>], _unit: UnitShape, _x: &Self::Value, _shared: bool) -> Result<Option<Self::Value>> { Ok(None) }
    /// `packed_group` on the experts that the route picked, with their numbers read where the compute holds them:
    /// `stack` holds the units of all `experts` experts back to back, `ids` holds the picked numbers (whole 32-bit
    /// floats, in pick order). The host reads no number. None: no such product here; the caller reads the numbers.
    fn packed_picked(&self, _stack: &V, _experts: usize, _unit: UnitShape, _ids: &Self::Value, _x: &Self::Value, _shared: bool) -> Result<Option<Self::Value>> { Ok(None) }
}
#[cfg(target_os = "linux")]
impl<'b, 'd> ResidentProduct<spark_gpu_probe::driver::DeviceVector<'d>> for crate::bounded_decoder::Device<'b, 'd> {
    fn resident_product(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value) -> Result<Self::Value> {
        crate::bounded_decoder::Device::resident_product(self, weight, words, rows, columns, x)
    }
    fn resident_product_rows(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, rows: usize, columns: usize, x: &Self::Value, inputs: usize) -> Result<Self::Value> {
        crate::bounded_decoder::Device::resident_product_rows(self, weight, words, rows, columns, x, inputs)
    }
    fn resident_product_runs(&self, runs: &[(&spark_gpu_probe::driver::DeviceVector<'d>, Range<usize>, usize, usize)], x: &Self::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::resident_product_runs(self, runs, x, columns, input, output)
    }
    fn packed_product(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, words: Range<usize>, unit: UnitShape, x: &Self::Value) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_product(self, weight, words, unit, x)
    }
    fn packed_group(&self, weight: &spark_gpu_probe::driver::DeviceVector<'d>, units: &[Range<usize>], unit: UnitShape, x: &Self::Value, shared: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_group(self, weight, units, unit, x, shared)
    }
    fn packed_picked(&self, stack: &spark_gpu_probe::driver::DeviceVector<'d>, experts: usize, unit: UnitShape, ids: &Self::Value, x: &Self::Value, shared: bool) -> Result<Option<Self::Value>> {
        crate::bounded_decoder::Device::packed_picked(self, stack, experts, unit, ids, x, shared)
    }
}

enum Entry {
    /// Host words in the layout of the weight name. A matrix has its rows and columns.
    Host(Vec<f32>, Option<(usize, usize)>),
    /// A whole BF16 matrix of the owner of the weights, by the name of its tensor.
    Resident { name: String, rows: usize, columns: usize },
}
/// The weights of one loaded checkpoint for the driver. `V` is the value type of the owner of the weights.
pub struct LoadedWeights<'a, V> { operands: &'a Operands<V>, entries: BTreeMap<Weight, Entry>,
    /// A matrix that a stack of the operands holds: the stack and its words in the stack.
    stacked: BTreeMap<Weight, (usize, Range<usize>)> }

fn size(n: u64) -> Result<usize> { usize::try_from(n).map_err(|_| "loop weights: a size exceeds the address space".into()) }
/// Fixed taps in the layout of the arm. `row_is_tap`: the stored tensor is `[tap, channel]`; else `[channel, tap]`.
fn taps_by_lag(stored: &[f32], channels: usize, taps: usize, order: TapOrder, row_is_tap: bool) -> Result<Vec<f32>> {
    if channels.checked_mul(taps) != Some(stored.len()) { return Err("loop weights: stored taps differ from channels x taps".into()); }
    let mut out = vec![0.0; stored.len()];
    for channel in 0..channels {
        for lag in 0..taps {
            let tap = match order { TapOrder::OldestFirst => taps - 1 - lag, TapOrder::NewestFirst => lag };
            out[channel * taps + lag] = if row_is_tap { stored[tap * channels + channel] } else { stored[channel * taps + tap] };
        }
    }
    Ok(out)
}
fn order(taps: ConvTaps) -> TapOrder { match taps { ConvTaps::Fixed { order } | ConvTaps::FixedAndToken { order, .. } => order } }

impl<'a, V> LoadedWeights<'a, V> {
    /// Read the small weights from `files` and name every weight of the driver. `operands` are the loaded tensors.
    pub fn new(checkpoint: &LoopCheckpoint, files: &Files, operands: &'a Operands<V>) -> Result<Self> {
        let d = &checkpoint.description;
        let hidden = size(d.hidden)?;
        let mut entries = BTreeMap::new();
        let host = |r: &Resident<V>| files.words(&r.tensor);
        // A BF16 matrix stays with the owner of the weights. Any other matrix is small: host words.
        let matrix = |r: &Resident<V>| -> Result<Entry> {
            match (r.tensor.dtype, &r.tensor.shape[..]) {
                (Dtype::Bf16, &[rows, columns]) => Ok(Entry::Resident { name: r.tensor.name.clone(), rows: size(rows)?, columns: size(columns)? }),
                (_, &[rows, columns]) => Ok(Entry::Host(host(r)?, Some((size(rows)?, size(columns)?)))),
                _ => Err(format!("loop weights: {} is no matrix", r.tensor.name)),
            }
        };
        let m = operands.model(d)?;
        entries.insert(Weight::Embedding, Entry::Host(host(m.embedding)?, None));
        entries.insert(Weight::EmbeddingNorm, Entry::Host(host(m.embedding_norm)?, None));
        entries.insert(Weight::LastNorm, Entry::Host(host(m.final_norm)?, None));
        entries.insert(Weight::MixOutput, Entry::Host(host(m.mix_output)?, None));
        entries.insert(Weight::Head, matrix(m.head)?);
        if let (Some(first), Some(second)) = (m.message_queries, m.message_kv_queries) {
            // Rows `2j` and `2j + 1` of the first tensor: the query and the expert row of loop layer `j`. Of the
            // second: its key and its value row.
            let (first, second) = (host(first)?, host(second)?);
            if first.len() != second.len() || !first.len().is_multiple_of(2 * hidden) { return Err("loop weights: the two message query tensors differ from two rows a loop layer".into()); }
            let mut merged = Vec::with_capacity(2 * first.len());
            for (qe, kv) in first.chunks(2 * hidden).zip(second.chunks(2 * hidden)) {
                merged.extend_from_slice(&qe[..hidden]); merged.extend_from_slice(kv); merged.extend_from_slice(&qe[hidden..]);
            }
            entries.insert(Weight::Messages, Entry::Host(merged, None));
        }
        for (i, layer) in d.layers.iter().enumerate() {
            let l = operands.layer(d, i as u64)?;
            let a = layer.attention;
            for (weight, r) in [(Weight::Query(i), l.query), (Weight::Key(i), l.key), (Weight::Value(i), l.value), (Weight::Relative(i), l.relative_projection),
                (Weight::Sink(i), l.sink), (Weight::OutputGate(i), l.output_gate), (Weight::Output(i), l.output), (Weight::Router(i), l.router)] { entries.insert(weight, matrix(r)?); }
            for (weight, r) in [(Weight::QueryClip(i), l.query_clip), (Weight::RelativeClip(i), l.relative_clip), (Weight::RelativeTable(i), l.relative_table),
                (Weight::ExpertNorm(i), l.expert_norm), (Weight::RouterBias(i), l.router_bias), (Weight::MixExperts(i), l.mix_experts)] { entries.insert(weight, Entry::Host(host(r)?, None)); }
            if let Some(r) = l.mix_attention { entries.insert(Weight::MixAttention(i), Entry::Host(host(r)?, None)); }
            let (query, key, value) = match l.input_norms { InputNorms::One(n) => (n, n, n), InputNorms::Separate { query, key, value } => (query, key, value) };
            for (weight, r) in [(Weight::QueryNorm(i), query), (Weight::KeyNorm(i), key), (Weight::ValueNorm(i), value)] { entries.insert(weight, Entry::Host(host(r)?, None)); }
            let (channels, taps, heads) = (size(a.kv_heads * a.head_dim)?, size(a.key_value_conv.kernel)?, size(a.kv_heads)?);
            for (fixed_name, made_name, conv) in [(Weight::KeyTaps(i), Weight::KeyTapProjection(i), &l.key_conv), (Weight::ValueTaps(i), Weight::ValueTapProjection(i), &l.value_conv)] {
                match (conv, a.key_value_conv.taps) {
                    (ConvOperands::Fixed(r), ConvTaps::Fixed { order }) => { entries.insert(fixed_name, Entry::Host(taps_by_lag(&host(r)?, channels, taps, order, false)?, None)); }
                    (ConvOperands::Token { fixed, projection }, ConvTaps::FixedAndToken { order, .. }) => {
                        entries.insert(fixed_name, Entry::Host(taps_by_lag(&host(fixed)?, channels, taps, order, true)?, None));
                        // With the newest tap first a stored row `head * taps + tap` is row `head * taps + lag` already:
                        // a BF16 projection stays with the owner of the weights, as stored.
                        if order == TapOrder::NewestFirst && projection.tensor.dtype == Dtype::Bf16 {
                            if projection.tensor.shape != [(heads * taps) as u64, hidden as u64] { return Err(format!("loop weights: {} differs from heads x taps rows", projection.tensor.name)); }
                            entries.insert(made_name, matrix(projection)?);
                            continue;
                        }
                        // A stored row is `head * taps + tap`. The driver reads row `head * taps + lag`.
                        let stored = host(projection)?;
                        if stored.len() != heads * taps * hidden { return Err(format!("loop weights: {} differs from heads x taps rows", projection.tensor.name)); }
                        let mut by_lag = Vec::with_capacity(stored.len());
                        for head in 0..heads {
                            for lag in 0..taps {
                                let tap = match order { TapOrder::OldestFirst => taps - 1 - lag, TapOrder::NewestFirst => lag };
                                by_lag.extend_from_slice(&stored[(head * taps + tap) * hidden..(head * taps + tap + 1) * hidden]);
                            }
                        }
                        entries.insert(made_name, Entry::Host(by_lag, Some((heads * taps, hidden))));
                    }
                    _ => return Err(format!("loop weights: layer {i}: the stored taps differ from the taps of the description")),
                }
            }
            let stream_taps = size(d.conv.kernel)?;
            entries.insert(Weight::ConvAfterAttention(i), Entry::Host(taps_by_lag(&host(l.conv_after_attention)?, hidden, stream_taps, order(d.conv.taps), false)?, None));
            entries.insert(Weight::ConvAfterExperts(i), Entry::Host(taps_by_lag(&host(l.conv_after_experts)?, hidden, stream_taps, order(d.conv.taps), false)?, None));
            if let Some(g) = &l.gates {
                for (message, gate) in [(Message::Query, &g.query), (Message::Key, &g.key), (Message::Value, &g.value), (Message::Expert, &g.expert)] {
                    entries.insert(Weight::GateDown(i, message), matrix(gate.down)?);
                    entries.insert(Weight::GateForget(i, message), matrix(gate.forget)?);
                    entries.insert(Weight::GateGain(i, message), matrix(gate.gain)?);
                }
            }
        }
        let by_name = entries.iter().filter_map(|(w, e)| match e { Entry::Resident { name, .. } => Some((name.as_str(), *w)), Entry::Host(..) => None }).collect::<BTreeMap<_, _>>();
        let by_name = &by_name;
        let stacked = operands.stacks().iter().enumerate()
            .flat_map(|(i, stack)| stack.members.iter().filter_map(move |(name, words)| by_name.get(name.as_str()).map(|w| (*w, (i, words.clone()))))).collect();
        Ok(Self { operands, entries, stacked })
    }
    /// Where `weights` are two or more matrices that one stack holds one after another in this order, with the same
    /// columns: the stack value, the words of all of them, the rows of each and the columns. None otherwise.
    fn run(&self, weights: &[Weight]) -> Result<Option<(&V, Range<usize>, Vec<usize>, usize)>> {
        let mut found = Vec::with_capacity(weights.len());
        for weight in weights {
            let (Some((stack, words)), Some(Entry::Resident { rows, columns, .. })) = (self.stacked.get(weight), self.entries.get(weight)) else { return Ok(None) };
            found.push((*stack, words.clone(), *rows, *columns));
        }
        if found.len() < 2 || found.windows(2).any(|p| p[0].0 != p[1].0 || p[0].1.end != p[1].1.start || p[0].3 != p[1].3) { return Ok(None); }
        let ((stack, first, _, columns), last) = (&found[0], &found[found.len() - 1].1);
        let value = &self.operands.stacks().get(*stack).ok_or("loop weights: a stack is missing")?.value;
        Ok(Some((value, first.start..last.end, found.iter().map(|f| f.2).collect(), *columns)))
    }
    /// Every weight name but the experts, and true for a weight that is held as host words.
    pub fn names(&self) -> impl Iterator<Item = (Weight, bool)> + '_ { self.entries.iter().map(|(w, e)| (*w, matches!(e, Entry::Host(..)))) }
    /// The rows and columns of a weight that is read through a product. None for a weight that is read as words.
    pub fn product_shape(&self, weight: Weight) -> Result<Option<(usize, usize)>> {
        Ok(match (self.matrix(weight)?, self.entries.get(&weight)) {
            (Some((_, _, rows, columns)), _) => Some((rows, columns)),
            (None, Some(Entry::Host(_, shape))) => *shape,
            (None, _) => return Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        })
    }
    /// The value, the word range, the rows and the columns of a matrix that stays with the owner of the weights.
    /// None for a weight that is held as host words.
    pub fn matrix(&self, weight: Weight) -> Result<Option<(&V, Range<usize>, usize, usize)>> {
        let expert = |layer: usize, proj: Proj, expert: usize| -> Result<Option<(&V, Range<usize>, usize, usize)>> {
            let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
            let m = self.operands.expert(layer, proj, expert as u64)?;
            Ok(Some((m.value, m.words, size(m.rows)?, size(m.columns)?)))
        };
        match weight {
            Weight::GateUp(layer, e) => expert(layer, Proj::GateUp, e),
            Weight::Down(layer, e) => expert(layer, Proj::Down, e),
            _ => match self.entries.get(&weight) {
                Some(Entry::Resident { name, rows, columns }) => {
                    let words = rows.checked_mul(*columns).filter(|n| n.is_multiple_of(2)).ok_or_else(|| format!("loop weights: {name} does not fill whole pairs of BF16 values"))? / 2;
                    Ok(Some((&self.operands.named(name)?.value, 0..words, *rows, *columns)))
                }
                _ => Ok(None),
            },
        }
    }
    /// The host words of a weight that is read as words (`Weights::words`).
    pub fn host_words(&self, weight: Weight) -> Result<&[f32]> {
        match self.entries.get(&weight) {
            Some(Entry::Host(words, _)) => Ok(words),
            Some(Entry::Resident { .. }) => Err(format!("loop weights: {weight:?} stays with the owner of the weights: it is read through a product")),
            None if matches!(weight, Weight::GateUp(..) | Weight::Down(..)) => Err(format!("loop weights: {weight:?} stays with the owner of the weights: it is read through a product")),
            None => Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        }
    }
    /// Bytes of the host words. They are host memory beside the device bytes of the ledger.
    pub fn host_bytes(&self) -> usize { self.entries.values().map(|e| match e { Entry::Host(w, _) => std::mem::size_of_val(&w[..]), Entry::Resident { .. } => 0 }).sum() }
}
impl<C: ResidentProduct<V>, V> Weights<C> for LoadedWeights<'_, V> {
    fn words(&self, weight: Weight) -> Result<&[f32]> { self.host_words(weight) }
    fn product(&self, compute: &C, weight: Weight, x: &C::Value) -> Result<C::Value> {
        // An expert: the packed stack first. A compute without a packed product for the shape gives None.
        if let Weight::GateUp(layer, expert) | Weight::Down(layer, expert) = weight {
            let (layer, proj) = (u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?, if matches!(weight, Weight::Down(..)) { Proj::Down } else { Proj::GateUp });
            if let Some(packed) = self.operands.packed_expert(layer, proj, expert as u64)?
                && let Some(out) = compute.packed_product(packed.value, packed.words, packed.unit, x)? { return Ok(out); }
        }
        if let Some((value, words, rows, columns)) = self.matrix(weight)? { return compute.resident_product(value, words, rows, columns, x); }
        match self.entries.get(&weight) {
            Some(Entry::Host(words, _)) => chain_product(compute, words, x),
            _ => Err(format!("loop weights: the checkpoint holds no weight {weight:?}")),
        }
    }
    /// A matrix that stays with the owner of the weights goes to the compute with all its input rows. A matrix of
    /// host words runs one chain product a row.
    fn product_rows(&self, compute: &C, weight: Weight, x: &C::Value, rows: usize) -> Result<C::Value> {
        if rows == 0 || !compute.words(x).is_multiple_of(rows) { return Err("a product for rows holds no whole count of input rows".into()); }
        // One row is `product`: an expert reads its packed stack there.
        if rows == 1 { return self.product(compute, weight, x); }
        if let Some((value, words, matrix_rows, columns)) = self.matrix(weight)? { return compute.resident_product_rows(value, words, matrix_rows, columns, x, rows); }
        let columns = compute.words(x) / rows;
        let outs = (0..rows).map(|row| self.product(compute, weight, &compute.range(x, row * columns, columns)?)).collect::<Result<Vec<_>>>()?;
        compute.join(&outs.iter().collect::<Vec<_>>())
    }
    /// The expert runs go to the compute with no gather when every run is a matrix that stays with the owner of the
    /// weights, with one shape; any other case, or a compute with no such body, takes the body with gathers.
    fn product_runs(&self, compute: &C, runs: &[(Weight, usize)], x: &C::Value, columns: usize, input: Option<&[usize]>, output: Option<&[usize]>) -> Result<C::Value> {
        check_runs(compute.words(x), columns, runs, input, output)?;
        let mut matrices = Vec::with_capacity(runs.len());
        for &(weight, count) in runs {
            match self.matrix(weight)? {
                Some((value, words, rows, matrix_columns)) if matrix_columns == columns && matrices.first().is_none_or(|first: &(&V, Range<usize>, usize, usize)| first.2 == rows) => matrices.push((value, words, rows, count)),
                _ => return gathered_runs(self, compute, runs, x, columns, input, output),
            }
        }
        match compute.resident_product_runs(&matrices, x, columns, input, output)? {
            Some(value) => Ok(value),
            None => gathered_runs(self, compute, runs, x, columns, input, output),
        }
    }
    /// Matrices that a stack holds one after another run as one product over all their rows. A row is one chain over
    /// the same input, so each part holds the words of its own product.
    fn stack_products(&self, compute: &C, weights: &[Weight], x: &C::Value) -> Result<Vec<C::Value>> {
        let Some((value, words, rows, columns)) = self.run(weights)? else { return weights.iter().map(|&weight| self.product(compute, weight, x)).collect() };
        let y = compute.resident_product(value, words, rows.iter().sum(), columns, x)?;
        let mut start = 0;
        rows.iter().map(|&n| { let part = compute.range(&y, start, n); start += n; part }).collect()
    }
    fn stack_product_rows(&self, compute: &C, weights: &[Weight], x: &C::Value, rows: usize) -> Result<Option<C::Value>> {
        let Some((value, words, parts, columns)) = self.run(weights)? else { return Ok(None) };
        if rows == 0 || rows.checked_mul(columns) != Some(compute.words(x)) { return Err("loop weights: the input of a stacked product differs from its rows".into()); }
        compute.resident_product_rows(value, words, parts.iter().sum(), columns, x, rows).map(Some)
    }
    fn products(&self, compute: &C, weights: &[Weight], x: &C::Value, shared: bool) -> Result<Option<C::Value>> {
        // Experts of one projection of one layer, from one packed stack.
        let (layer, proj) = match weights.first() {
            Some(&Weight::GateUp(layer, _)) => (layer, Proj::GateUp),
            Some(&Weight::Down(layer, _)) => (layer, Proj::Down),
            _ => return Ok(None),
        };
        let layer_word = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let (mut value, mut units, mut shape): (Option<&V>, _, Option<UnitShape>) = (None, Vec::with_capacity(weights.len()), None);
        for &weight in weights {
            let expert = match (weight, proj) { (Weight::GateUp(l, e), Proj::GateUp) | (Weight::Down(l, e), Proj::Down) if l == layer => e, _ => return Ok(None) };
            let Some(packed) = self.operands.packed_expert(layer_word, proj, expert as u64)? else { return Ok(None); };
            if value.is_some_and(|v| !std::ptr::eq(v, packed.value)) || shape.is_some_and(|s| s != packed.unit) { return Ok(None); }
            (value, shape) = (Some(packed.value), Some(packed.unit));
            units.push(packed.words);
        }
        match (value, shape) { (Some(value), Some(unit)) => compute.packed_group(value, &units, unit, x, shared), _ => Ok(None) }
    }
    fn picked_products(&self, compute: &C, first: Weight, ids: &C::Value, x: &C::Value, shared: bool) -> Result<Option<C::Value>> {
        // The whole packed stack of the projection and layer of `first`; the compute reads the picked numbers.
        let (layer, proj) = match first { Weight::GateUp(layer, 0) => (layer, Proj::GateUp), Weight::Down(layer, 0) => (layer, Proj::Down), _ => return Ok(None) };
        let layer = u32::try_from(layer).map_err(|_| "loop weights: a layer number exceeds its range".to_owned())?;
        let Some((stack, experts, unit)) = self.operands.packed_experts(layer, proj)? else { return Ok(None); };
        compute.packed_picked(stack, experts, unit, ids, x, shared)
    }
}
#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "packed_tests.rs"]
mod packed_tests;
