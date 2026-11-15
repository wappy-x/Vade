//! One launch of several same-shape causal commands. Grid x is one member's own block count.
//! Grid y selects that member. The body is the one-chat body, unchanged.
use crate::resident_launch_abi::{Command,Kind};
// SAFETY: The recorder writes this command only after `resident_causal_group::plan` accepts every member.
// The member table is 16-byte aligned in the staged device area and stays live until the queue drains.
// Grid x equals that member's own launch grid, so the one-chat body sees its own block index and grid size.
#[no_mangle]
pub extern "ptx-kernel" fn resident_launch_causal_group(table:*const Command,command_index:u64) {
    // SAFETY: The command index is inside the uploaded table. `count` member commands are initialized.
    // A block whose y is past count returns before it reads a member. Every thread of a block takes the
    // same branch, so a later barrier is reached by the whole block. Each body keeps the extent checks
    // of the one-chat entry, and shared memory stays inside the block.
    unsafe {
        let command=&*table.add(command_index as usize);
        if command.kind!=Kind::CausalGroup as u32 {core::arch::nvptx::trap();}
        let payload=command.payload.as_ptr();
        let members=core::ptr::read(payload.cast::<u64>());
        let count=core::ptr::read(payload.add(8).cast::<u32>());
        let member_kind=core::ptr::read(payload.add(12).cast::<u32>());
        let y=core::arch::nvptx::_block_idx_y();
        if y>=count {return;}
        let member=&*(members as *const Command).add(y as usize);
        if member.kind!=member_kind {core::arch::nvptx::trap();}
        if member_kind==Kind::CausalChain as u32 {
            let words=member.payload.as_ptr();
            crate::causal_fast::prepare_one(words.cast::<crate::causal_ops::Arguments>().read());
            core::arch::nvptx::_syncthreads();
            crate::causal_fast::step_columns_128(&*words.add(256).cast::<crate::causal_ops::Arguments>());
            core::arch::nvptx::_syncthreads();
            crate::causal_fast::gated_norm_one(words.add(512).cast::<crate::causal_ops::Arguments>().read());
        } else if member_kind==Kind::CausalFast as u32 {
            crate::causal_operation::resident_causal_fast(&*member.payload.as_ptr().cast::<crate::causal_ops::Arguments>());
        } else {core::arch::nvptx::trap();}
    }
}
