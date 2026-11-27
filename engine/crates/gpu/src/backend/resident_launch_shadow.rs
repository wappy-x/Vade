use super::*;
/// A command that reads weights at the bus rate. A prefetch for a later target is not placed across one: it would run
/// beside it on the bus.
/// The row launches of 2 to 8 token rows (ROW SHADOW, `bus_shadow::rows`) read weights at the bus rate too; no one-row step
/// records them, so the one-row list above is unchanged.
fn boundary(kind:Kind)->bool {
    matches!(kind,Kind::Expert|Kind::Fp8Tile|Kind::Plain|Kind::PlainBf16|Kind::PlainPacked|Kind::PlainPackedGroup|Kind::PlainPackedPicked|Kind::PlainStaged|Kind::PlainStagedRows|Kind::PlainStagedFew|Kind::PlainStagedOne|Kind::PlainStagedNarrow|Kind::PlainPackedRows|Kind::PlainTcRows|Kind::PlainTcBig|Kind::PlainTcPanel64|Kind::PlainPackedTcRows|Kind::PrefillPackedTc|Kind::FlashWindowAttn|Kind::Mx|Kind::Mixed|Kind::NvStream|Kind::NvTile|Kind::NvQuarters|Kind::DeviceRoute|Kind::DeviceRouteTcRows|Kind::Fp4CohortT8|Kind::Prepare|Kind::Collapse|Kind::Combine)
        || matches!(kind,Kind::MxRows2|Kind::MxRows4|Kind::MxRows8|Kind::MxRows8w|Kind::MxRowsTile|Kind::MixedRows|Kind::FoldRows|Kind::FoldRowsF32
            |Kind::Fp8RowsSmall|Kind::Fp8RowsMulti|Kind::Fp8RowsFew|Kind::Fp8RowsShare|Kind::PlainRowsSmall)
        // LATENT ROWS: they replace latent_absorption_tile in a check pass, which ended the recorded piece, so no prefetch
        // was ever placed across one. They stay boundaries: the prefetch placement of a row pass is what it was.
        || matches!(kind,Kind::LatentRowsDense|Kind::LatentRowsAttention)
}
/// The BF16 forms of the mixed and plain batch launches read as the FP32 mixed launch does, so they are boundaries too.
/// They are named apart so that a prefetch they stop is counted (`shadow[0]`, print only).
fn batch_bf16(kind:Kind)->bool {matches!(kind,Kind::MixedBf16|Kind::PlainBatch)}
/// The take that gets the prefetch for a target recorded after the calls of `range`: the last take with no boundary
/// behind it. None: there is none, or a BF16 batch launch lies between (counted).
/// WAIT COVER (seat DRAFTR): with `plain`, a take of kind Take counts too. The rows path (2 to 8 token rows: the row
/// groups of several chats and the check rows of a drafting pass) receives most of its hand-offs with that kind, and
/// those waits ran with an idle memory bus. Only a row window (`bus_shadow::rows`) asks for it, so a one-row step
/// records what it recorded before.
/// FIRST TAKE: the wait for the messages of an exchange starts at the first of its takes (`bus_shadow::one_exchange`). The prefetch
/// goes in front of that FIRST take, not the last, so it runs during the whole wait; all the takes of the exchange then
/// copy beside it. On a pair a take is always alone, so its place is what it was. MEASURED 2026-11-21 on four Sparks:
/// 31.04 to 31.24 -> 30.02 to 30.04 ms a step, token IDs and logits equal. Prefetch only: no value can change with it.
fn take_for(recorder:&mut Recorder<'_>,range:std::ops::Range<usize>,plain:bool)->Option<usize> {
    let mut crossed=false;let start=range.start;
    for index in range.rev() {
        let kind=recorder.calls[index].grid.kind;
        if kind==Kind::TakeScatter || plain && kind==Kind::Take {if crossed {recorder.shadow[0]+=1;return None;}
            let mut first=index;
            while first>start && crate::bus_shadow::one_exchange(&recorder.calls[first-1].command,&recorder.calls[first].command) {first-=1;}
            return Some(first);}
        if boundary(kind) {return None;}
        crossed|=batch_bf16(kind);
    }None
}
fn install(recorder:&mut Recorder<'_>,index:usize,window:crate::bus_shadow::Window)->bool {
    let stage_base=recorder.stage.as_ref().map_or(0,|stage|stage.device_address()+STAGE_BASE as u64);
    // SHADOW-STREAM: the window is prefetched by a command of its own on a side branch, placed just before the take, so
    // the take does not last as long as its prefetch. Later commands keep their order; indices at or after the place move up.
    // With the side form on there is no tail form: a `more` window in a take's tail would make the take last as long as
    // the prefetch. So a piece with no room for one more command gets no prefetch, and a take has one side prefetch at most.
    if !recorder.side_off {
        if recorder.calls.len()+2>=crate::GraphAllowance::FLASH_MAX_NODES {recorder.shadow[1]+=1;return false;}
        if index>0 && recorder.calls[index-1].grid.kind==Kind::ShadowSide {recorder.shadow[2]+=1;return false;}
        let g=recorder.calls[index].grid;
        let side=if g.kind==Kind::Take {crate::bus_shadow::side_plain(&recorder.calls[index].command,window)}
            else {crate::bus_shadow::side(&recorder.calls[index].command,[g.blocks,g.rows,g.planes,g.threads,g.shared],window,stage_base,&recorder.staged)};
        if let Some(command)=side {
            recorder.calls.insert(index,Call {grid:Grid {kind:Kind::ShadowSide,blocks:crate::bus_shadow::SIDE_BLOCKS,rows:1,planes:1,threads:256,shared:0,lane:MAIN},command,copy_bytes:0});
            for bell in &mut recorder.bells {if bell.0>=index {bell.0+=1;}}
            for zero in &mut recorder.zeros {if zero.0>=index {zero.0+=1;}}
            for fill in &mut recorder.waiting {if fill.0>=index {fill.0+=1;}}
            recorder.side_placed+=1;crate::bus_shadow::record(window.ids!=0);return true;
        }
        return false;
    }
    let call=&mut recorder.calls[index];let grid=&mut call.grid;
    let mut dimensions=[grid.blocks,grid.rows,grid.planes,grid.threads,grid.shared];
    if !crate::bus_shadow::arm(&mut call.command,&mut dimensions,window,stage_base,&recorder.staged) {return false;}
    grid.rows=dimensions[1];crate::bus_shadow::record(window.ids!=0);true
}
pub(super) fn static_target(recorder:&mut Recorder<'_>,target:&Command) {
    let stage_base=recorder.stage.as_ref().map_or(0,|stage|stage.device_address()+STAGE_BASE as u64);
    let one=crate::bus_shadow::mx(target).or_else(||if recorder.side_off {None} else {crate::bus_shadow::more(target).or_else(||crate::bus_shadow::batch(target))});
    // A row window may sit beside a plain take (WAIT COVER); a one-row window keeps the scatter take.
    let (window,plain)=match one {Some(window)=>(window,false),None=>{
        if recorder.side_off {return;}
        let Some(window)=crate::bus_shadow::rows(target,stage_base,&recorder.staged) else {return;};(window,true)}};
    let all=0..recorder.calls.len();if let Some(index)=take_for(recorder,all,plain) {install(recorder,index,window);}
}
pub(super) fn routed_target(recorder:&mut Recorder<'_>,window:crate::bus_shadow::Window) {
    let Some(ready)=recorder.calls.iter().rposition(|call|crate::bus_shadow::prior_route(&call.command,window)) else {return;};
    let after=ready+1..recorder.calls.len();if let Some(index)=take_for(recorder,after,false) {install(recorder,index,window);}
}
