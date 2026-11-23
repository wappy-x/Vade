//! State of one chat of the looped text family, for each layer visit, and the byte ledger of one device.
//! Host metadata only: this module allocates nothing and admits no execution.
//!
//! A stored layer in the loop range runs one time for each pass, and each visit keeps its own state
//! (reference: transformers v5.24.0, `modeling_inkling2_loop.py` lines 1354-1357 and 1405). So the
//! state is planned for each visit, not for each stored layer.
//!
//! Kept for the chat, for each visit:
//! - Keys and values. A row holds the finished numbers of one token: `kv_heads * head_dim` BF16
//!   values, two a 32-bit word. A window visit keeps a ring of `window` rows; the row of position
//!   `p` is `p mod window`. A visit that sees every earlier token keeps one row for each token of
//!   the context, in straight planes.
//! - Four convolution rings: the inputs of the key, the value, the after-attention and the
//!   after-experts convolution, `kernel` rows of 32-bit words each.
//!
//! Kept for one token step: the stream, the restart point of the loop, the normed block stack, the
//! normed block outputs of the pass before, and the messages of the running pass.
use super::{PlanError, Result, description::{LoopTextDescription, Visit}, loop_load::{LoadBytes, device_extent}};
use crate::safetensors::Dtype;
use std::fmt;

fn mul(a: u64, b: u64) -> Result<u64> { super::mul(a, b) }
fn add(a: u64, b: u64) -> Result<u64> { super::add(a, b) }
/// Bytes of one 32-bit device word.
const WORD: u64 = 4;
/// Message gates of one looped layer: the query, the key, the value and the expert input.
const MESSAGES_PER_LAYER: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvRing { Key, Value, AfterAttention, AfterExperts }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingBytes { pub ring: ConvRing, pub rows: u64, pub channels: u64, pub bytes: u64 }
/// The state of one layer visit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisitState {
    pub visit: Visit,
    /// `Some`: a ring of this many rows. `None`: straight planes of one row for each token.
    pub window: Option<u64>,
    /// Rows of the key store and of the value store.
    pub rows: u64,
    /// Bytes of one key row. A value row has the same size.
    pub row_bytes: u64,
    pub keys: u64,
    pub values: u64,
    pub conv: [RingBytes; 4],
}
impl VisitState {
    pub fn conv_bytes(&self) -> u64 { self.conv.iter().map(|r| r.bytes).sum() }
    pub fn bytes(&self) -> u64 { self.keys + self.values + self.conv_bytes() }
}
/// Values of one token step. Each row is `hidden` 32-bit words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepState {
    /// The stream and the saved restart point of the loop.
    pub stream_rows: u64,
    /// The normed vector of each finished block, of the embedding and of the current stream.
    pub stack_rows: u64,
    /// The normed block outputs of the pass before.
    pub carried_rows: u64,
    /// The messages of the running pass.
    pub message_rows: u64,
    pub row_bytes: u64,
    pub bytes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopStatePlan {
    /// Tokens that one chat can hold.
    pub context: u64,
    /// The most rows of one step (the step row limit). A ring of a window visit holds this many rows less one
    /// more than its window, so a row of a step never reads a ring row that a later row of the step wrote over.
    pub step_rows: u64,
    /// Rows of a straight plane: the context, rounded up to whole launch buckets.
    pub plane_rows: u64,
    /// One entry for each visit, in run order.
    pub visits: Vec<VisitState>,
    pub step: StepState,
}
impl LoopStatePlan {
    /// `bucket`: a launch whose extent follows the context takes the grid of the next `bucket` rows, so a
    /// straight plane holds whole buckets. The caller gives the rule of its device.
    pub fn new(d: &LoopTextDescription, context: u64, bucket: u64) -> Result<Self> { Self::for_steps(d, context, bucket, 1) }
    /// The plan for steps of up to `step_rows` rows: the ring of a window visit holds `window + step_rows - 1` rows.
    /// One row a step is the plan of `new`.
    pub fn for_steps(d: &LoopTextDescription, context: u64, bucket: u64, step_rows: u64) -> Result<Self> {
        if step_rows == 0 { return Err(PlanError::Invalid { field: "step row limit", reason: "a step has no row" }); }
        if context == 0 || context > d.max_positions || context > u64::from(u32::MAX) {
            return Err(PlanError::Unsupported { field: "context", value: format!("{context} tokens: the model takes 1 to {} tokens", d.max_positions.min(u64::from(u32::MAX))) });
        }
        if bucket == 0 { return Err(PlanError::Invalid { field: "context bucket", reason: "a launch bucket has no rows" }); }
        let plane_rows = mul(context.div_ceil(bucket), bucket)?;
        let visits = d.visits.iter().map(|&visit| {
            let layer = usize::try_from(visit.layer).ok().and_then(|i| d.layers.get(i))
                .ok_or(PlanError::Invalid { field: "visits", reason: "a visit names a layer that is not stored" })?;
            let a = layer.attention;
            let channels = mul(a.kv_heads, a.head_dim)?;
            let row_bytes = mul(channels, Dtype::Bf16.bytes())?;
            if row_bytes == 0 || device_extent(row_bytes)? != row_bytes {
                return Err(PlanError::Invalid { field: "key and value row", reason: "a row of BF16 values does not fill whole device words" });
            }
            if a.window == Some(0) { return Err(PlanError::Invalid { field: "sliding_window_size", reason: "a window holds no token" }); }
            let rows = match a.window { Some(window) => add(window, step_rows - 1)?, None => plane_rows };
            let plane = mul(rows, row_bytes)?;
            let ring = |ring, rows: u64, channels: u64| Ok(RingBytes { ring, rows, channels, bytes: mul(mul(rows, channels)?, WORD)? });
            Ok(VisitState { visit, window: a.window, rows, row_bytes, keys: plane, values: plane, conv: [
                ring(ConvRing::Key, a.key_value_conv.kernel, channels)?, ring(ConvRing::Value, a.key_value_conv.kernel, channels)?,
                ring(ConvRing::AfterAttention, d.conv.kernel, d.hidden)?, ring(ConvRing::AfterExperts, d.conv.kernel, d.hidden)?] })
        }).collect::<Result<Vec<_>>>()?;
        let core_layers = d.core.end.checked_sub(d.core.start).ok_or(PlanError::Invalid { field: "loop_config", reason: "the loop range ends before it starts" })?;
        let stack_rows = add(d.blocks(), 1)?;
        let (carried_rows, message_rows) = (core_layers.div_ceil(d.residual.block_layers.max(1)), mul(MESSAGES_PER_LAYER, core_layers)?);
        let row_bytes = mul(d.hidden, WORD)?;
        let rows = add(add(add(2, stack_rows)?, carried_rows)?, message_rows)?;
        let step = StepState { stream_rows: 2, stack_rows, carried_rows, message_rows, row_bytes, bytes: mul(rows, row_bytes)? };
        let plan = Self { context, step_rows, plane_rows, visits, step };
        plan.fixed_bytes()?; plan.plane_bytes()?;
        Ok(plan)
    }
    fn sum(&self, of: impl Fn(&VisitState) -> u64) -> Result<u64> { self.visits.iter().try_fold(0, |n, v| add(n, of(v))) }
    /// Bytes that do not change with the context: the rings of the window visits and all convolution rings.
    pub fn fixed_bytes(&self) -> Result<u64> { self.sum(|v| v.conv_bytes() + if v.window.is_some() { v.keys + v.values } else { 0 }) }
    /// Bytes of the straight planes at this context.
    pub fn plane_bytes(&self) -> Result<u64> { self.sum(|v| if v.window.is_none() { v.keys + v.values } else { 0 }) }
    /// Bytes that one more plane row costs over all visits.
    pub fn plane_row_bytes(&self) -> Result<u64> { self.sum(|v| if v.window.is_none() { 2 * v.row_bytes } else { 0 }) }
    /// All state of one chat: the state of every visit and the values of one token step.
    pub fn chat_bytes(&self) -> Result<u64> { add(add(self.fixed_bytes()?, self.plane_bytes()?)?, self.step.bytes) }
    pub fn window_visits(&self) -> usize { self.visits.iter().filter(|v| v.window.is_some()).count() }
    pub fn full_visits(&self) -> usize { self.visits.len() - self.window_visits() }
}

/// The bytes that one device holds for the weights and one chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopLedger {
    pub context: u64,
    /// Device bytes of the weights (`LoadBytes::device`).
    pub weights: u64,
    pub state_fixed: u64,
    pub state_planes: u64,
    pub step: u64,
    /// All of the above.
    pub total: u64,
}
impl LoopLedger {
    pub fn new(load: &LoadBytes, state: &LoopStatePlan) -> Result<Self> {
        let (state_fixed, state_planes, step) = (state.fixed_bytes()?, state.plane_bytes()?, state.step.bytes);
        Ok(Self { context: state.context, weights: load.device, state_fixed, state_planes, step,
            total: add(add(add(load.device, state_fixed)?, state_planes)?, step)? })
    }
    /// The state of one chat.
    pub fn chat(&self) -> u64 { self.state_fixed + self.state_planes + self.step }
    /// Refuse a ledger that does not fit in `budget` bytes of device memory.
    pub fn admit(&self, budget: u64, largest: u64) -> std::result::Result<(), MemoryRefusal> {
        if self.total <= budget { return Ok(()); }
        Err(MemoryRefusal { context: self.context, budget, resident: self.weights, session: self.chat(), largest_context: largest })
    }
}
/// The largest context whose ledger fits in `budget`, in whole buckets, or 0 when no context fits.
/// A ledger is weights, fixed state and step values, plus one plane row for each token of a whole bucket.
pub fn context_limit(d: &LoopTextDescription, load: &LoadBytes, bucket: u64, budget: u64) -> Result<u64> {
    // The sizes of everything but the planes do not change with the context: plan the smallest context to read them.
    let smallest = LoopStatePlan::new(d, 1, bucket)?;
    let base = add(add(load.device, smallest.fixed_bytes()?)?, smallest.step.bytes)?;
    let row = smallest.plane_row_bytes()?;
    let top = d.max_positions.min(u64::from(u32::MAX));
    let Some(room) = budget.checked_sub(base) else { return Ok(0); };
    // With no straight plane the context costs nothing.
    let rows = room.checked_div(row).unwrap_or(u64::MAX);
    // A plane holds whole buckets. The model's largest context can end inside a bucket that still fits.
    Ok((rows / bucket).saturating_mul(bucket).min(top))
}
/// Why a context does not fit. The text has the fields of the memory refusal of the text service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRefusal { pub context: u64, pub budget: u64, pub resident: u64, pub session: u64, pub largest_context: u64 }
impl fmt::Display for MemoryRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CONTEXT={}; budget={}; resident={}; shared_workspace=0; session_pool={}; full_session={}; reduce CONTEXT",
            self.context, self.budget, self.resident, self.budget.saturating_sub(self.resident), self.session)?;
        if self.largest_context > 0 { write!(f, " to {} or less", self.largest_context) }
        else { write!(f, ": no context fits, the weights and the fixed state need more than the budget") }
    }
}
