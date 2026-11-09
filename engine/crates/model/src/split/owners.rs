//! Contiguous owners and role-based selections, including rotated expert blocks.
use super::*;
use crate::safetensors::Dtype;

fn uniform<T: Copy + Eq>(slot: &mut Option<T>, value: T) -> Result<()> {
    if slot.is_some_and(|old| old != value) {
        return Err(SplitError::Unsupported { what: "multiple shapes for one layer kind" });
    }
    *slot = Some(value);
    Ok(())
}
fn divisible(total: u64, unit: u64) -> Result<()> {
    if total == 0 || unit == 0 || total % unit != 0 { return Err(SplitError::Shape { what: "split divisibility" }); }
    Ok(())
}
pub(super) fn facts(shape: &ModelShape, expert_rows: bool) -> Result<Facts> {
    let mut f = Facts { lexical: shape.lexical, ..Facts::default() };
    if shape.hidden == 0 || shape.vocab == 0 { return Err(SplitError::Shape { what: "model dimensions" }); }
    if f.lexical.is_some_and(|l| l.part_rows == 0) { return Err(SplitError::Shape { what: "lexical rows require headers" }); }
    if expert_rows { divisible(shape.hidden, ROW_BLOCK)?; }
    for layer in shape.layers.iter().chain(&shape.draft) {
        match layer.mixer {
            Mixer::Attention(a) => {
                divisible(a.query_heads, a.kv_heads)?;
                if a.head_dim == 0 { return Err(SplitError::Shape { what: "attention heads" }); }
                uniform(&mut f.attention, a)?;
            }
            Mixer::Linear(l) => {
                divisible(l.value_heads, l.key_heads)?;
                if l.key_dim == 0 || l.value_dim == 0 {
                    return Err(SplitError::Shape { what: "linear heads" });
                }
                uniform(&mut f.linear, l)?;
            }
        }
        match layer.ffn {
            Ffn::Dense { width } => { divisible(width, BF16_GROUP)?; uniform(&mut f.dense, width)?; }
            Ffn::Experts(e) => {
                divisible(e.width, FP8_BLOCK)?;
                if e.shared_width != 0 { divisible(e.shared_width, BF16_GROUP)?; }
                if e.count == 0 || e.selected == 0 || e.selected > e.count { return Err(SplitError::Shape { what: "expert count" }); }
                if e.storage == ExpertStorage::Stacked { return Err(SplitError::Unsupported { what: "stacked experts" }); }
                uniform(&mut f.expert, e)?;
            }
        }
    }
    Ok(f)
}
pub(super) fn partition(total: u64, rank: u32, ranks: u32) -> Result<Range<u64>> {
    if !(MIN_SPARKS..=MAX_SPARKS).contains(&ranks) { return Err(SplitError::Ranks { ranks }); }
    if rank >= ranks { return Err(SplitError::Shape { what: "rank index" }); }
    let (r, n) = (u64::from(rank), u64::from(ranks));
    let (base, extra) = (total / n, total % n);
    let start = add(mul(r, base)?, r.min(extra))?;
    Ok(start..add(add(start, base)?, u64::from(r < extra))?)
}
fn grouped(r: &Range<u64>, group: u64) -> Result<Range<u64>> {
    if r.is_empty() { return Ok(0..0); }
    if group == 0 { return Err(SplitError::Shape { what: "head group" }); }
    Ok(r.start / group..add(r.end / group, u64::from(r.end % group != 0))?)
}
pub(super) fn owners(shape: &ModelShape, f: &Facts, rank: u32, q: &SplitRequest) -> Result<Owners> {
    let part = |total| partition(total, rank, q.ranks);
    let channels = |width| scaled(&part(width / BF16_GROUP)?, BF16_GROUP);
    let query_heads = part(f.attention.map_or(0, |a| a.query_heads))?;
    let value_heads = part(f.linear.map_or(0, |l| l.value_heads))?;
    Ok(Owners {
        kv_heads: grouped(&query_heads, f.attention.map_or(1, |a| a.query_heads / a.kv_heads))?,
        key_heads: grouped(&value_heads, f.linear.map_or(1, |l| l.value_heads / l.key_heads))?,
        query_heads, value_heads,
        shared: channels(f.expert.map_or(0, |e| e.shared_width))?,
        dense: channels(f.dense.map_or(0, |width| width))?,
        vocab: part(shape.vocab)?,
        lexical_rows: part(match f.lexical { Some(l) => mul(l.parts, l.part_rows)?, None => 0 })?,
        output_rows: if q.expert_rows { Some(scaled(&part(shape.hidden / ROW_BLOCK)?, ROW_BLOCK)?) } else { None },
    })
}
pub(super) fn expert_blocks(f: &Facts, rank: u32, ranks: u32, expert: u32) -> Result<Range<u64>> {
    partition(0, rank, ranks)?;
    let e = required(f.expert)?;
    if u64::from(expert) >= e.count { return Err(SplitError::Shape { what: "expert index" }); }
    let (r, n) = (u64::from(rank), u64::from(ranks));
    let blocks = e.width / FP8_BLOCK;
    let (base, extra, first) = (blocks / n, blocks % n, u64::from(expert) % n);
    let before_wrap = extra.min(n.checked_sub(first).ok_or(SplitError::Shape { what: "expert rotation" })?);
    let wrapped = extra.checked_sub(before_wrap).ok_or(SplitError::Shape { what: "expert rotation" })?;
    let end = add(first, before_wrap)?;
    let before = r.min(end).checked_sub(r.min(first)).ok_or(SplitError::Shape { what: "expert rotation" })?;
    let start = add(mul(r, base)?, add(before, r.min(wrapped))?)?;
    Ok(start..add(add(start, base)?, u64::from((first..end).contains(&r) || r < wrapped))?)
}
fn projection(proj: Proj, channels: Range<u64>) -> Result<Selection> {
    match proj {
        Proj::Gate | Proj::Up => Ok(Selection::Rows(vec![channels])),
        Proj::Down => Ok(Selection::Columns(channels)),
        Proj::GateUp => Err(SplitError::Unsupported { what: "fused gate/up" }),
    }
}
pub(super) fn selection(plan: &SplitPlan, rank: u32, t: &TensorEntry) -> Result<Selection> {
    let o = plan.owner(rank)?;
    let role = t.role;
    if role.place == Place::Unused || matches!(role.place, Place::Draft { .. }) && !plan.draft {
        return Ok(Selection::Absent);
    }
    if t.dtype == Dtype::U8 { return Err(SplitError::Unsupported { what: "packed FP4" }); }
    if role.store != Store::Weight && !matches!(role.part, Part::Expert { .. } | Part::Whole) {
        if role.store != Store::BlockScale || t.shape.len() != 2 || !matches!(role.part,
            Part::Embedding | Part::Head | Part::Shared(_) | Part::Dense(_) | Part::Query | Part::Key
            | Part::Value | Part::Output | Part::Index | Part::LinearQkv | Part::LinearZ
            | Part::LinearOut | Part::LinearHeads) {
            return Err(SplitError::Unsupported { what: "non-expert quantized storage" });
        }
        // Select in weight coordinates, then retain every intersecting scale block.
        // Uneven channel/vocab splits can require the same scale on two ranks.
        let mut weight = t.clone();
        weight.role.store = Store::Weight;
        weight.shape = t.shape.iter().map(|&d| mul(d, FP8_BLOCK)).collect::<Result<_>>()?;
        let blocks = |r: Range<u64>| if r.is_empty() { 0..0 } else { r.start / FP8_BLOCK..r.end.div_ceil(FP8_BLOCK) };
        return Ok(match selection(plan, rank, &weight)? {
            Selection::Rows(ranges) => Selection::Rows(ranges.into_iter().map(blocks).collect()),
            Selection::Columns(range) => Selection::Columns(blocks(range)),
            other => other,
        });
    }
    let rows = |r| Ok(Selection::Rows(vec![r]));
    match role.part {
        Part::Embedding | Part::Head => rows(o.vocab.clone()),
        Part::Expert { id, proj } => {
            let e = id.ok_or(SplitError::Unsupported { what: "stacked experts" })?;
            let blocks = plan.expert_blocks(rank, e)?;
            if role.store == Store::Scalar {
                return Ok(if blocks.is_empty() { Selection::Absent } else { Selection::All });
            }
            if plan.expert_rows && proj == Proj::Down && matches!(role.place, Place::Layer(_)) {
                let r = o.output_rows.as_ref().ok_or(SplitError::Shape { what: "output rows" })?;
                return rows(if role.store == Store::BlockScale { r.start / FP8_BLOCK..r.end / FP8_BLOCK } else { r.clone() });
            }
            projection(proj, scaled(&blocks, if role.store == Store::Weight { FP8_BLOCK } else { 1 })?)
        }
        Part::Shared(p) => { required(plan.facts.expert.filter(|e| e.shared_width != 0))?; projection(p, o.shared.clone()) }
        Part::Dense(p) => { required(plan.facts.dense)?; projection(p, o.dense.clone()) }
        Part::Lexical { part } => {
            let l = required(plan.facts.lexical)?;
            if u64::from(part) >= l.parts { return Err(SplitError::Shape { what: "lexical part" }); }
            let start = mul(u64::from(part), l.part_rows)?;
            let from = o.lexical_rows.start.max(start);
            let to = o.lexical_rows.end.min(add(start, l.part_rows)?);
            if from < to {
                rows(from.checked_sub(start).ok_or(SplitError::Shape { what: "lexical start" })?
                    ..to.checked_sub(start).ok_or(SplitError::Shape { what: "lexical end" })?)
            } else { Ok(Selection::Absent) }
        }
        Part::Query | Part::Key | Part::Value | Part::Output | Part::QueryNorm | Part::KeyNorm | Part::Index | Part::IndexNorm => {
            let a = required(plan.facts.attention)?;
            match role.part {
                Part::Query => rows(scaled(&o.query_heads, mul(a.head_dim, if a.gated { 2 } else { 1 })?)?),
                Part::Key | Part::Value => rows(scaled(&o.kv_heads, a.head_dim)?),
                Part::Output => Ok(Selection::Columns(scaled(&o.query_heads, a.head_dim)?)),
                Part::QueryNorm | Part::KeyNorm => rows(partition(a.head_dim, rank, plan.ranks)?),
                Part::Index => { let i = required(a.index)?; rows(0..mul(add(i.heads, i.kv_heads)?, i.dim)?) }
                _ => rows(partition(required(a.index)?.dim, rank, plan.ranks)?),
            }
        }
        Part::LinearQkv | Part::LinearConv | Part::LinearZ | Part::LinearOut | Part::LinearHeads | Part::LinearNorm => {
            let l = required(plan.facts.linear)?;
            let k = scaled(&o.key_heads, l.key_dim)?;
            let v = scaled(&o.value_heads, l.value_dim)?;
            match role.part {
                Part::LinearHeads => rows(o.value_heads.clone()),
                Part::LinearNorm => rows(partition(l.value_dim, rank, plan.ranks)?),
                Part::LinearZ => rows(v),
                Part::LinearOut => Ok(Selection::Columns(v)),
                _ => {
                    let offset = mul(l.key_heads, l.key_dim)?;
                    let twice = mul(2, offset)?;
                    Ok(Selection::Rows(vec![k.clone(), add(k.start, offset)?..add(k.end, offset)?,
                        add(v.start, twice)?..add(v.end, twice)?]))
                }
            }
        }
        Part::Whole => Ok(Selection::All),
    }
}
