//! Checked resident weight, cache, state and workspace ledgers.
use super::*;

fn bytes(t: &TensorEntry, selection: Selection) -> Result<u64> {
    let product = |dims: &[u64]| dims.iter().try_fold(t.dtype.bytes(), |a, &b| mul(a, b));
    let bounded = |r: &Range<u64>, bound| {
        if r.end > bound { return Err(SplitError::Shape { what: "selection exceeds tensor" }); }
        length(r)
    };
    match selection {
        Selection::All => product(&t.shape),
        Selection::Absent => Ok(0),
        Selection::Rows(ranges) => {
            let first = t.shape.first().ok_or(SplitError::Shape { what: "row tensor" })?;
            let rows = ranges.iter().try_fold(0, |a, r| add(a, bounded(r, *first)?))?;
            let stride = t.shape.iter().skip(1).try_fold(t.dtype.bytes(), |a, &b| mul(a, b))?;
            mul(rows, stride)
        }
        Selection::Columns(r) => {
            if t.shape.len() != 2 { return Err(SplitError::Shape { what: "column tensor rank" }); }
            let rows = t.shape.first().ok_or(SplitError::Shape { what: "column rows" })?;
            let columns = t.shape.get(1).ok_or(SplitError::Shape { what: "column width" })?;
            mul(mul(*rows, bounded(&r, *columns)?)?, t.dtype.bytes())
        }
    }
}
pub(super) fn ledgers(plan: &SplitPlan, shape: &ModelShape, q: &SplitRequest) -> Result<Vec<Ledger>> {
    (0..q.ranks).map(|rank| {
        let o = plan.owner(rank)?;
        let weights = shape.tensors.iter().try_fold(0, |total, t| add(total, bytes(t, plan.selection(rank, t)?)?))?;
        let (mut cache, mut state) = (0, 0);
        for layer in shape.layers.iter().chain(shape.draft.iter().filter(|_| q.draft)) {
            match layer.mixer {
                Mixer::Attention(a) => {
                    cache = add(cache, add(mul(mul(2, length(&o.kv_heads)?)?, a.head_dim)?, a.index.map_or(0, |i| i.dim))?)?;
                }
                Mixer::Linear(l) => {
                    let values = mul(length(&o.value_heads)?, l.value_dim)?;
                    let recurrent = mul(values, l.key_dim)?;
                    let conv = mul(add(mul(mul(2, length(&o.key_heads)?)?, l.key_dim)?, values)?, l.conv)?;
                    state = add(state, add(recurrent, conv)?)?;
                }
            }
        }
        let cache = mul(mul(mul(q.context, q.sequences)?, cache)?, q.cache_bytes)?;
        let state = mul(mul(q.sequences, state)?, q.state_bytes)?;
        let reserve = *q.reserve.get(index(rank)?).ok_or(SplitError::Budgets { ranks: q.ranks, given: q.reserve.len() })?;
        let available = *q.memory.get(index(rank)?).ok_or(SplitError::Budgets { ranks: q.ranks, given: q.memory.len() })?;
        let needed = add(add(add(weights, cache)?, state)?, reserve)?;
        if needed > available { return Err(SplitError::NoFit { rank, needed, available }); }
        Ok(Ledger { weights, cache, state, reserve })
    }).collect()
}
