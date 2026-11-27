//! Scatter take entries, in their own module so the existing kernels keep their code.
use crate::{kernel_params::HybridCopyRow,resident_launch_abi::{Command,Kind},tp_mailbox::resident_tp_mailbox_take_scatter as take};
#[no_mangle]
pub extern "ptx-kernel" fn tp_mailbox_take_scatter(table:*const HybridCopyRow,seq_word:*const u32,seq:u64,error:*mut u32,limit_ns:u64) {take(table,seq_word,seq,error,limit_ns)}
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_take_scatter(table:*const Command,index:u64) {
    // SAFETY: The checked recorder writes five u64 words of a TakeScatter command in
    // its private aligned table. That table covers index and is uploaded before this
    // replay on the same stream. A command of another kind traps here. The body then
    // has exactly the extents of the direct entry above, which the host plan bounds.
    // Extra grid-y rows return after read-only prefetches. Grid-x and all original
    // rows are unchanged. Their metadata names retained checked weights, or a
    // retained immutable route table and initialized IDs read by an earlier route build.
    // The host excludes every receive target from that ID span. No tail reads the
    // receive slot or writes memory. The queue retains every owner through drain.
    unsafe {
        let c=&*table.add(index as usize);if c.kind!=Kind::TakeScatter as u32 {core::arch::nvptx::trap();}let words=c.payload.as_ptr().cast::<u64>();
        crate::bus_shadow::scatter_tail!(c);
        take(*words as _,*words.add(1) as _,*words.add(2),*words.add(3) as _,*words.add(4));
    }
}
