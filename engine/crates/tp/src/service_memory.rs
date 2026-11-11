//! Device charges for ordered prompt and decode execution.
use engine_model::plan::{admission::CheckpointAdmission,layer::Topology,state::StatePlan};
use crate::{Result,bounded_decoder::{expanded_bytes,group_workspace_bytes_for,RESERVED_ROWS,PROJECTION_MEMO_BYTES}};

/// Committed and unpublished state coexist. Expanded prompt rows are copied only
/// while their prefix grows. After that point both states share the same owners.
pub fn session(checkpoint:&CheckpointAdmission,prompt:usize,tokens:usize)->Result<usize> {
    session_with_expanded(checkpoint,prompt,tokens,expanded_bytes(checkpoint,prompt))
}
pub fn session_for_sparse_heads(checkpoint:&CheckpointAdmission,topology:Topology,prompt:usize,tokens:usize)->Result<usize> {
    let held=crate::admission_plan::SessionChargePlan::for_sparse_heads(checkpoint.description(),topology,tokens)?.expanded_bytes(prompt);
    session_with_expanded(checkpoint,prompt,tokens,held)
}
#[allow(clippy::manual_clamp)] // clamp panics and treats NaN differently.
fn session_with_expanded(checkpoint:&CheckpointAdmission,prompt:usize,tokens:usize,held:usize)->Result<usize> {
    if tokens==0 || prompt>tokens {return Err("invalid service reservation extent".into());}
    let state=|n:usize|StatePlan::new(checkpoint.description(),Topology {rank:0,parts:1},n as u64,isize::MAX as usize)
        .map(|s|s.state_layout().bytes()).map_err(|e|e.to_string());
    let early=state(prompt.min(RESERVED_ROWS).max(1))?.checked_mul(2).and_then(|n|held.checked_mul(2).and_then(|h|n.checked_add(h))).ok_or("early session charge overflow")?;
    let late=state(tokens)?.checked_mul(2).and_then(|n|n.checked_add(held)).ok_or("late session charge overflow")?;
    let tail_rows=prompt.min(256).max(1);
    let tails=checkpoint.description().layers.iter().try_fold(0usize,|n,l|match l.mixer {engine_model::plan::description::TokenMixer::Kda(k)=>(k.heads as usize).checked_mul(k.head_dim as usize).and_then(|v|v.checked_mul(8)).and_then(|v|v.checked_mul(tail_rows)).and_then(|v|n.checked_add(v)),_=>Some(n)}).ok_or("state tail charge overflow")?;
    let cache=crate::admission_plan::append_reservation_bytes(checkpoint.description())?;
    early.max(late).checked_add(tails).and_then(|n|n.checked_add(usize::try_from(checkpoint.description().vocab).ok()?.checked_mul(16)?)).and_then(|n|n.checked_add(cache)).ok_or_else(||"session charge overflow".into())
}

/// Scalar decode and grouped prompt work do not execute together. Their queues
/// drain and spare buffers are released at a phase boundary. Immutable parameter
/// owners and descriptor storage retain the separate ancillary allowance.
///
/// This is the charge of ONE session whose caller clears the projection memo before prompt work
/// (`Device::clear_projection_memo`: rows, chunks and limit go). `Device::prepare` alone does not meet the rule: it
/// keeps memo chunks (MEMO_KEEP), the last prompt group seeds the memo, and beside other open sessions the whole memo
/// stays while a prompt runs. A service with several sessions charges the memo AND the prompt scratch
/// (`row_split::sparse_service_charges`), as flash-serve does.
pub fn workspace(checkpoint:&CheckpointAdmission,context:usize,staging:usize,group:usize)->Result<usize> {
    let state=StatePlan::new(checkpoint.description(),Topology {rank:0,parts:1},context as u64,isize::MAX as usize).map_err(|e|e.to_string())?;
    let scalar=state.scratch_bytes().checked_mul(2).ok_or("scalar scratch overflow")?;
// A state-prefix copy can coexist with its source before the stream drains.
// Charge every recurrent layer's matrix, even if several copies are deferred.
let copies=if group>256 {state_copy_bytes(checkpoint)?} else {0};
let prompt=group_workspace_bytes_for(checkpoint,context,group)?.checked_add(copies).ok_or("prompt charge overflow")?;
let decode=scalar.checked_add(PROJECTION_MEMO_BYTES).ok_or("decode memo charge overflow")?;
// Prompt entry clears the memo. Decode can use its full memo only after
// prompt scratch has drained. These two peaks therefore share one allowance.
decode.max(prompt).checked_add(staging).and_then(|n|n.checked_add(1<<30)).ok_or_else(||"workspace charge overflow".into())
}

/// Owned recurrent matrix prefixes can coexist with their queued copy sources.
pub fn state_copy_bytes(checkpoint:&CheckpointAdmission)->Result<usize> {
let mut copies=0usize;
for layer in &checkpoint.description().layers {if let engine_model::plan::description::TokenMixer::Kda(k)=layer.mixer {
    let bytes=(k.heads as usize).checked_mul(k.head_dim as usize).and_then(|n|n.checked_mul(k.head_dim as usize)).and_then(|n|n.checked_mul(4)).ok_or("state copy overflow")?;
    copies=copies.checked_add(bytes).ok_or("state copy total overflow")?;
}}
Ok(copies)
}
