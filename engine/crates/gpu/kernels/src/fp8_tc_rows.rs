//! Bench entries of `fp8_tc::row_body`, the row body of block FP8 on tensor cores (the decode step and small row
//! groups): the prefetch one line ahead at three strides (once a line, twice, every step), each as a row-group
//! entry and as a decode entry of its own (the body without the other operations of `expert_operation`). The warp
//! count of a block is the launch's choice (4, 8 or 16 warps), so one entry times every warp count. The word is the
//! contract word of `tc_f8`, the same as every block FP8 entry.
//!
//! Row-group launch: the `GroupArguments` and table of `expert_group`; grid (ceil(rows / 16), parts *
//! ceil(max_tokens / 8)); dynamic shared `fp8_row_shared(warps, blocks, min(max_tokens, 8), false)` bytes. All parts
//! share rows and cols and have the same number of token tiles (the host checks).
//! Decode launch: the `LaunchArguments`, grid and shared bytes of the block FP8 form of `expert_operation`.
use crate::expert_ops::{GroupArguments, LaunchArguments};

macro_rules! rows_entry {($name:ident,$pf:expr,$ps:expr)=>{
    #[no_mangle]
    pub extern "ptx-kernel" fn $name(launch: GroupArguments) {
        // SAFETY: Only the block index and the grid size are read here. Every check of the launch, its parts and its
        // operands, and every memory access, is in `fp8_tc::group_rows` and `fp8_tc::row_body` (their contracts). The
        // body traps unless the grid's second dimension is a whole number of token tiles a part. Each
        // (part, row, token) output word has exactly one writing thread.
        unsafe {
            use core::arch::nvptx::{_block_idx_x, _block_idx_y, _grid_dim_y};
            let (count, y, tiles) = (launch.count, _block_idx_y() as u32, _grid_dim_y() as u32);
            if count == 0 { core::arch::asm!("trap;"); }
            let (part, tile) = if count == 1 { (0, y) } else {
                let per = tiles / count;
                if per == 0 || per * count != tiles { core::arch::asm!("trap;"); }
                (y / per, y % per)
            };
            crate::fp8_tc::group_rows::<{ $pf }, { $ps }>(launch, part as usize, tile as usize, _block_idx_x() as usize);
        }
    }
};}
rows_entry!(fp8_tc_rows_s4, 4, 4);
rows_entry!(fp8_tc_rows_s2, 4, 2);
rows_entry!(fp8_tc_rows_s1, 4, 1);

macro_rules! decode_entry {($name:ident,$pf:expr,$ps:expr)=>{
    /// The decode body of block FP8 alone: the launch of `expert_operation` for admitted Fp8Serial jobs.
    #[no_mangle]
    pub extern "ptx-kernel" fn $name(launch: LaunchArguments) { crate::fp8_tile::project::<{ $pf }, { $ps }, 1>(launch) }
};}
decode_entry!(fp8_tc_decode_s4, 4, 4);
decode_entry!(fp8_tc_decode_s2, 4, 2);
decode_entry!(fp8_tc_decode_s1, 4, 1);

macro_rules! decode_tiles_entry {($name:ident,$tiles:expr)=>{
    /// The decode body with several 16-row tiles a block (fewer, larger blocks for a matrix of many rows).
    #[no_mangle]
    pub extern "ptx-kernel" fn $name(launch: LaunchArguments) { crate::fp8_tile::project::<4, 4, { $tiles }>(launch) }
};}
decode_tiles_entry!(fp8_tc_decode_t2, 2);
decode_tiles_entry!(fp8_tc_decode_t4, 4);

macro_rules! tile_entry {($name:ident,$nf:expr,$token_first:expr)=>{
    /// The second form of the prompt tile (`fp8_tc::tile_body`), `$nf` tiles of 8 token rows a warp, prefetch one
    /// line ahead. Grid: (row groups, token tiles, parts), or (token tiles, row groups, parts) when the token tile
    /// is the fastest index. Dynamic shared `fp8_tc::tile_shared($nf)` bytes.
    #[no_mangle]
    pub extern "ptx-kernel" fn $name(launch: GroupArguments) {
        // SAFETY: Only the three block indices are read here. Every check of the launch, its parts and its operands,
        // and every memory access, is in `fp8_tc::group_tile` and `fp8_tc::tile_body` (their contracts).
        let (x, y, part) = unsafe { (core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_block_idx_y() as usize, core::arch::nvptx::_block_idx_z() as usize) };
        let (row_group, tile) = if $token_first { (y, x) } else { (x, y) };
        crate::fp8_tc::group_tile::<{ $nf }, 4>(launch, part, tile, row_group);
    }
};}
tile_entry!(fp8_tc_tile_n4, 4, false);
tile_entry!(fp8_tc_tile_n6, 6, false);
tile_entry!(fp8_tc_tile_n8, 8, false);
tile_entry!(fp8_tc_tile_n8t, 8, true);

macro_rules! tile3_entry {($name:ident,$nf:expr,$dec:expr,$park:expr)=>{
    /// A bench form of the third prompt tile (`fp8_tc::tile3_body`; the served form is `fp8_rows_tile3`), `$nf` tiles
    /// of 8 token rows a warp. `$dec`: 0 the tensor core decodes the weights, 1 the integer function. `$park`: where
    /// the waiting sums are kept. Grid: (row groups, token tiles, parts). Dynamic shared
    /// `fp8_tc::tile3_shared($nf, warps, $park)` bytes.
    #[no_mangle]
    pub extern "ptx-kernel" fn $name(launch: GroupArguments) {
        // SAFETY: Only the three block indices are read here. Every check of the launch, its parts and its operands,
        // and every memory access, is in `fp8_tc::group_tile3` and `fp8_tc::tile3_body` (their contracts).
        let (row_group, tile, part) = unsafe { (core::arch::nvptx::_block_idx_x() as usize, core::arch::nvptx::_block_idx_y() as usize, core::arch::nvptx::_block_idx_z() as usize) };
        crate::fp8_tc::group_tile3::<{ $nf }, { $dec }, { $park }, true>(launch, part, tile, row_group);
    }
};}
tile3_entry!(fp8_tc_t3_n4p, 4, 0, 1);
tile3_entry!(fp8_tc_t3_n4q, 4, 0, 2);
tile3_entry!(fp8_tc_t3_n4s, 4, 1, 0);
tile3_entry!(fp8_tc_t3_n6, 6, 0, 0);
tile3_entry!(fp8_tc_t3_n6p, 6, 0, 1);
tile3_entry!(fp8_tc_t3_n6q, 6, 0, 2);
tile3_entry!(fp8_tc_t3_n8p, 8, 0, 1);
tile3_entry!(fp8_tc_t3_n8q, 8, 0, 2);
