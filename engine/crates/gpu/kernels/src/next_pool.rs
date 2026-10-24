//! Slot resolution happens at replay, before any real history is accessed.
use crate::{next_math::{SlotStep, StateSlot}, kernel_params::*};
use core::arch::nvptx::_block_idx_y;
#[inline(always)]
unsafe fn selected(p: SlotStep) -> Option<(&'static StateSlot,usize)> {
    let row=_block_idx_y() as usize;
    p.selected(row).map(|id| (&*(p.table as *const StateSlot).add(id),row))
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_slot_position(p: SlotStep) {
    let row=crate::index();
    if let Some(id)=p.selected(row) { (*(p.table as *mut StateSlot).add(id)).position+=1; }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_route_mask(ids: u64, weights: u64, p: SlotStep) {
    let i=crate::index();
    if i<p.rows as usize*10 && p.selected(i/10).is_none() {
        *(ids as *mut u32).add(i)=u32::MAX; *(weights as *mut f32).add(i)=0.0;
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_prepare_slot(mut p: GdnPrepare, step: SlotStep) {
    let Some((slot,row))=selected(step) else { return; };
    let hd=p.value_heads as usize*128; let width=(2*p.key_heads+p.value_heads) as usize*128;
    p.history=slot.layers[step.layer as usize][1] as _;
    p.x=p.x.add(row*width); p.a=p.a.add(row*p.value_heads as usize); p.b=p.b.add(row*p.value_heads as usize);
    p.q=p.q.add(row*hd); p.k=p.k.add(row*hd); p.v=p.v.add(row*hd);
    p.beta=p.beta.add(row*p.value_heads as usize); p.g=p.g.add(row*p.value_heads as usize);
    crate::flash_gdn::conv_prepare_tokens(p);
}
/// prepare_slot + history_slot in one launch, the same bits (non-order):
/// grid (sections / 4 = 20, rows) x 128 threads, one warp per qkv section.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_prepare_hist_slot(mut p: GdnPrepare, step: SlotStep) {
    // The chunk slot (a programmatic launch) starts its state-only prologue in this
    // grid's run; it waits (griddepcontrol) before reading q/k/v/beta/g.
    core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
    let Some((slot,row))=selected(step) else { return; };
    let hd=p.value_heads as usize*128; let width=(2*p.key_heads+p.value_heads) as usize*128;
    let history=slot.layers[step.layer as usize][1] as *mut f32;
    p.x=p.x.add(row*width); p.a=p.a.add(row*p.value_heads as usize); p.b=p.b.add(row*p.value_heads as usize);
    p.q=p.q.add(row*hd); p.k=p.k.add(row*hd); p.v=p.v.add(row*hd);
    p.beta=p.beta.add(row*p.value_heads as usize); p.g=p.g.add(row*p.value_heads as usize);
    let thread=crate::index();
    crate::flash_gdn::conv_prepare_section(p, history, thread/32, thread%32);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_scan_slot(mut p: crate::gdn_params::GdnStepBatch, step: SlotStep) {
    let Some((slot,row))=selected(step) else { return; };
    let hd=p.heads as usize*128; let offset=(row*hd*4) as u64;
    p.q+=offset; p.k+=offset; p.v+=offset; p.out+=offset;
    p.beta+=(row*p.heads as usize*4) as u64; p.g+=(row*p.heads as usize*4) as u64;
    p.rows=slot.layers[step.layer as usize].as_ptr() as u64;
    crate::flash_next::gdn_scan(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_history_slot(mut p: GdnHistory, step: SlotStep) {
    let Some((slot,row))=selected(step) else { return; };
    p.x=p.x.add(row*p.channels as usize); p.history=slot.layers[step.layer as usize][1] as _;
    crate::flash_gdn::history_token(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_gdn_gate_slot(mut p: GdnGate, step: SlotStep) {
    let row=_block_idx_y() as usize; let offset=row*p.heads as usize*128;
    if step.selected(row).is_none() {
        let thread=crate::index();
        if thread<p.heads as usize*32 { for i in 0..4 { *p.output.add(offset+thread/32*128+thread%32+i*32)=0.0; } }
        return;
    }
    p.x=p.x.add(offset); p.z=p.z.add(offset); p.output=p.output.add(offset);
    if p.precision==0 && p.packed==0 && p.sessions==0 && p.tokens==1 { crate::flash_gdn::gate_plain(p, crate::index()); }
    else { crate::flash_gdn::gate_token(p); }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_ple_slot(mut p: FlashPleConv, step: SlotStep) {
    let Some((slot,row))=selected(step) else { return; };
    p.history=slot.ple; p.input+=(row*p.width as usize*4) as u64;
    p.gated+=(row*p.width as usize*4) as u64; p.residual+=(row*p.width as usize*4) as u64;
    crate::flash_ops::ple_conv(p);
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_qsa_slot(templates: u64, output: u64, norm: u64, step: SlotStep) {
    let row=crate::index(); if row>=step.rows as usize { return; }
    let mut record=*(templates as *const QsaBatchRow).add(row);
    record.keys=0;
    if let Some(id)=step.selected(row) {
        let slot=&*(step.table as *const StateSlot).add(id); let layer=slot.layers[step.layer as usize];
        record.keys=layer[0]; record.values=layer[1]; record.raw_index=layer[2]; record.pooled=layer[2]+3*128*4;
        record.past=slot.position; record.norm=norm;
    }
    *(output as *mut QsaBatchRow).add(row)=record;
}
