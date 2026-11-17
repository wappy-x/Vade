//! Block-FP8 expert rows for a SMALL token count (a draft verify pass, several chats in one step): 3 to 4 tokens of one part, by the
//! host's choice (`small_choice::group`). The word is the contract of `tc_f8` (`docs/tc-dense-contract.md`, section
//! Block FP8), made by the one body of `fp8_tc`: the same word as the decode step (`fp8_tile`), the prompt tile
//! (`fp8_rows_tile`), the generic body (`expert_group`) and the host reference. The three small entries
//! (`fp8_rows_small`, `fp8_rows_few`, `fp8_rows_multi`) run the same body with the same geometry; they stay three
//! entries because each is a recorded launch kind.
//!
//! A block is ONE tile of 16 weight rows and one tensor-core tile of at most 8 token rows of one part, on 4 to 32
//! warps (`fp8_tc::row_body`), so 2 to 8 rows read the weights once, as one row does.
//!
//! Launch: the same `GroupArguments` and table as `expert_group`; 4, 8, 16 or 32 whole warps a block
//! (`fp8_small::threads`); a grid of `parts * ceil(max_tokens / 8) * ceil(rows / 16)` blocks in one dimension
//! (`fp8_tc::linear`); dynamic shared `fp8_small::shared` bytes. All parts share rows and cols (the host checks).
use crate::expert_ops::GroupArguments;

#[export_name="fp8_rows_few"]
pub extern "ptx-kernel" fn entry_fp8_rows_few(launch:GroupArguments) {fp8_rows_few(launch)}
/// The body, shared by the direct entry and the recorded replay entry (`resident_launch_fp8_rows_few`).
#[inline(always)]
pub(crate) fn fp8_rows_few(launch:GroupArguments) {
    // SAFETY: Only the block index, the grid size, the block size and the first table entry's row count are read here
    // (the host keeps the table of `count` entries alive through stream completion; a launch with no part traps before
    // the read). Every other check of the launch, its parts and its operands, and every memory access, is in
    // `fp8_tc::group` and `fp8_tc::run` (their contracts). The body traps unless the grid is the product
    // `fp8_tc::linear` reads. Each (part, row, token) output word has exactly one writing thread.
    unsafe {
        use core::arch::nvptx::{_block_idx_x,_grid_dim_x};
        if launch.count==0 {core::arch::asm!("trap;");}
        let rows=(*(launch.table as *const crate::expert_ops::GroupEntry)).shape.rows as usize;
        let Some((part,tile,row_group))=crate::fp8_tc::linear(_block_idx_x() as usize,_grid_dim_x() as usize,launch.count as usize,rows,1) else {core::arch::asm!("trap;");return;};
        crate::fp8_tc::group_rows::<{crate::expert_ops::FP8_ROW_PREFETCH},{crate::expert_ops::FP8_ROW_STRIDE}>(launch,part,tile,row_group);
    }
}
