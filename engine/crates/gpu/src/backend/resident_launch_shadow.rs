use super::*;
fn boundary(kind:Kind)->bool {matches!(kind,Kind::Expert|Kind::Plain|Kind::PlainBf16|Kind::Mx|Kind::Mixed|Kind::NvStream|Kind::NvTile|Kind::DeviceRoute|Kind::Prepare|Kind::Collapse|Kind::Combine)}
fn install(recorder:&mut Recorder<'_>,index:usize,window:crate::bus_shadow::Window)->bool {
    let stage_base=recorder.stage.as_ref().map_or(0,|stage|stage.device_address()+STAGE_BASE as u64);
    let call=&mut recorder.calls[index];let grid=&mut call.grid;
    let mut dimensions=[grid.blocks,grid.rows,grid.planes,grid.threads,grid.shared];
    if !crate::bus_shadow::arm(&mut call.command,&mut dimensions,window,stage_base,&recorder.staged) {return false;}
    grid.rows=dimensions[1];crate::bus_shadow::record(window.ids!=0);true
}
pub(super) fn static_target(recorder:&mut Recorder<'_>,target:&Command) {
    let Some(window)=crate::bus_shadow::mx(target) else {return;};
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
