use super::*;
fn boundary(kind:Kind)->bool {matches!(kind,Kind::Expert|Kind::Plain|Kind::PlainBf16|Kind::Mx|Kind::Mixed|Kind::NvStream|Kind::NvTile|Kind::DeviceRoute|Kind::Prepare|Kind::Collapse|Kind::Combine)}
fn install(recorder:&mut Recorder<'_>,index:usize,window:crate::bus_shadow::Window)->bool {
    let stage_base=recorder.stage.as_ref().map_or(0,|stage|stage.device_address()+STAGE_BASE as u64);
    // SHADOW-STREAM: the window is prefetched by a command of its own on a side branch, placed just before the take, so
    // the take does not last as long as its prefetch. Later commands keep their order; indices at or after the place move up.
    if !recorder.side_off && recorder.calls.len()+2<crate::GraphAllowance::FLASH_MAX_NODES {
        let g=recorder.calls[index].grid;
        if let Some(command)=crate::bus_shadow::side(&recorder.calls[index].command,[g.blocks,g.rows,g.planes,g.threads,g.shared],window,stage_base,&recorder.staged) {
            recorder.calls.insert(index,Call {grid:Grid {kind:Kind::ShadowSide,blocks:crate::bus_shadow::BLOCKS,rows:1,planes:1,threads:256,shared:0},command,copy_bytes:0});
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
    let Some(window)=crate::bus_shadow::mx(target).or_else(||if recorder.side_off {None} else {crate::bus_shadow::more(target)}) else {return;};
    for index in (0..recorder.calls.len()).rev() {
        let kind=recorder.calls[index].grid.kind;
        if kind==Kind::TakeScatter {install(recorder,index,window);return;}
        if boundary(kind) {return;}
    }
}
pub(super) fn routed_target(recorder:&mut Recorder<'_>,window:crate::bus_shadow::Window) {
    let Some(ready)=recorder.calls.iter().rposition(|call|crate::bus_shadow::prior_route(&call.command,window)) else {return;};
    for index in (ready+1..recorder.calls.len()).rev() {
        let kind=recorder.calls[index].grid.kind;
        if kind==Kind::TakeScatter {install(recorder,index,window);return;}
        if boundary(kind) {return;}
    }
}
