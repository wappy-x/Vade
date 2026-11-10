//! Rank-ordered K segments and decoder-layer collective schedules.
//!
//! Fold rule: S = P0, then S = S + P_r for r = 1..N-1, FP32
//! round-to-nearest, an empty segment adding +0.0 in its position. The same
//! order holds whether partials cross the wire or are folded in one kernel.
//! Sources: seqpar/src/transport.rs:1340-1347; gpu/kernels/src/fp8_routing.rs:521;
//! gpu/kernels/src/fp8.rs:344-352 and 399-402. Expert contributions combine by
//! ascending ID, shared last (gpu/kernels/src/flash_moe.rs:180-213).
//! This module states the order and executes no arithmetic on partials.
use super::*;

pub(super) fn segments(plan: &SplitPlan, reduction: Reduction) -> Result<Vec<Range<u64>>> {
    (0..plan.ranks).map(|rank| {
        let o = plan.owner(rank)?;
        match reduction {
            Reduction::Attention => scaled(&o.query_heads, required(plan.facts.attention)?.head_dim),
            Reduction::Linear => scaled(&o.value_heads, required(plan.facts.linear)?.value_dim),
            Reduction::Shared => { required(plan.facts.expert.filter(|e| e.shared_width != 0))?; Ok(o.shared.clone()) }
            Reduction::Dense => { required(plan.facts.dense)?; Ok(o.dense.clone()) }
            Reduction::Expert(e) => scaled(&plan.expert_blocks(rank, e)?, FP8_BLOCK),
        }
    }).collect()
}
pub(super) fn collectives(shape: &ModelShape, q: &SplitRequest) -> Result<Vec<Vec<Collective>>> {
    shape.layers.iter().map(|layer| {
        if q.ranks == MIN_SPARKS { return Ok(Vec::new()); }
        let sum = Collective::Sum { elements: shape.hidden };
        let mut ops = vec![sum.clone()];
        match layer.ffn {
            Ffn::Dense { .. } => ops.push(sum),
            Ffn::Experts(e) if q.expert_rows => {
                ops.push(sum);
                ops.push(Collective::Gather { elements: mul(e.selected, e.width)? });
                ops.push(Collective::Gather { elements: shape.hidden });
            }
            Ffn::Experts(e) => ops.push(Collective::Sum { elements: mul(add(e.selected, 1)?, shape.hidden)? }),
        }
        Ok(ops)
    }).collect()
}
