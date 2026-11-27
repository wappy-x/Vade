//! NVFP4 decode batch projection over COLUMN-TILE weights (`expert_ops::NV_TILE_ROWS`), with the zero-input skip.
//! Arithmetic and order are those of `nv_projection::row!` and `nv_tile`: every output row is ONE ascending FP32
//! chain over its columns from +0.0, each product `(value * activation) * fold` made by two separate `mul.rn.f32`
//! with `fold = (e4m3(scale) * activation_scale) * (global * activation)`, each step one separate `add.rn.f32`
//! (no fused multiply-add), and the same final BF16 round. Output words equal the row-major kernels on the same
//! logical matrix.
//! What changes is the storage and who reads it. A tile is 128 rows; its codes are stored column by column (64 bytes
//! a column: row 2k in the low nibble of byte k, row 2k+1 in the high nibble) and its scales block by block (128
//! bytes a 16-column block, one byte a row). A column whose quantized activation is +0.0 or -0.0 is NOT COPIED from
//! device memory. Its staged bytes are then stale, and its product is still +0 or -0: every code decodes to a finite
//! value, `finite * (+-0) = +-0`, `(+-0) * fold` is +-0 for a finite fold and the same NaN as the row-major kernels
//! for a non-finite fold, and the chain starts at +0.0, so a +-0 term never changes the sum bits.
//! Form: a block owns one tile of one job; a thread owns one row and does everything for it. A round is
//! NV_COLS_ROUND_BLOCKS 16-column blocks (4 KB of codes a block of threads): the live columns and the scales arrive by
//! asynchronous copies, NV_COLS_DEPTH rounds deep; the activations of the job are staged once. The arithmetic of a
//! block is written stage by stage (loads, decodes, products, then the 16 chain adds) with no branch.
//! A job's `tokens` field is its output stride in words (1 for a decode job; see `nv_small::column_jobs` for prompt jobs).
//! Launch: the `LaunchArguments` and job table of `expert_operation`; `rows/128 * count` blocks (job-major) of
//! NV_COLS_THREADS threads; dynamic shared `nv_cols_shared(cols)` bytes. Host admission: `expert_ops::nv_cols_admitted`
//! (a decode batch) or `nv_small::admitted_columns` with the checks of `Backend::expert_group` (a prompt group of few tokens).
use crate::expert_ops::{Arguments,LaunchArguments,Op};

use crate::expert_ops::{NV_COLS_DEPTH,NV_COLS_ROUND_BLOCKS,NV_COLS_THREADS,NV_TILE_ROWS};
const TILE_ROWS:usize=NV_TILE_ROWS as usize;
const THREADS:usize=NV_COLS_THREADS as usize;
const ROUND_BLOCKS:usize=NV_COLS_ROUND_BLOCKS as usize;
const DEPTH:usize=NV_COLS_DEPTH as usize;

/// The decode batch projection. Since the tensor-core rebase its words are CONTRACT (b)'s (`tc_contract`): the body is
/// `nv_cols_b::project`, with one chain a word for a first matrix and quarter chains for a second matrix
/// (`expert_ops::nv_chains` of the launch's shape). Same launch, storage, staging and zero-skip as before.
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols(launch:LaunchArguments) {project(launch)}
/// The rc.8 chain (the arithmetic described in this file's header). BENCH BASELINE ONLY since the rebase: no host
/// path launches it.
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_rc8(launch:LaunchArguments) {project_rc8(launch)}

#[inline(always)]
pub(crate) fn project(launch:LaunchArguments) {
    if crate::expert_ops::nv_chains(&launch.single.shape)==4 {crate::nv_cols_b::project::<4,false>(launch)} else {crate::nv_cols_b::project::<1,false>(launch)}
}
#[inline(always)]
pub(crate) fn project_rc8(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor before the launch, in one of two places. A decode batch:
    // `expert_ops::nv_cols_admitted` (column-tile NvProject jobs of one token; output stride 1). A prompt group of few
    // tokens: `nv_small::admitted_columns` and the span checks of `Backend::expert_group`, which builds one job a
    // (part, token) with `nv_small::column_jobs` (output stride = the token count of the part). Both give rows a
    // multiple of 128, cols a multiple of 64 and at most NV_TILE_MAX_COLS, 16-byte aligned weight and scale owners and a
    // 4-byte aligned input. The host keeps all owners alive through stream completion and fills the job table (count
    // entries of equal rows and cols). Each job's weight owner holds rows * cols/2 bytes and its scale owner
    // rows * cols/16 bytes in the column-tile layout; its input owner holds cols/16*17 FP32 words. Its output words
    // are `output + (row * tokens) * 4` for row < rows (`tokens` is the stride in words, at least 1), all inside one
    // live private allocation: `rows` words of its own for a decode job; for a prompt job its token's column of the
    // rows * tokens `[row][token]` words of its part. Weights, scales and inputs are immutable. Thread r of the block
    // of (job, tile) is the only writer of the job's output word (tile*128 + r) * tokens (one raw store before any
    // read of it). Jobs of different parts own disjoint spans. The jobs of one part share a span: their first words
    // differ by less than the stride, so their words interleave and never meet. Every
    // asynchronous copy reads 16 bytes inside one column or one scale block of the tile, or 4 bytes below cols/16*17
    // input words. Shared addresses stay inside the
    // layout below (checked against the dynamic size), which is the only shared allocation of this kernel, so it
    // starts at 0. Data copied by one thread is read by another only after the copier's own wait and a bar.sync that
    // every thread of the block executes the same number of times (the loop bound depends only on cols). A staged byte
    // of a column that was not copied is read as a value and never as an address. The body traps on any other launch.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        // One nibble to its FP32 value: the half-float bytes of the eight magnitudes, then the sign.
        macro_rules! decode {($nibble:expr)=>{{
            let value:f32;let nibble=$nibble as u32;
            let selector=(nibble&7)<<4;let sign=(nibble&8)<<12;
            asm!("{{ .reg .b16 lo,hi; .reg .b32 halves; prmt.b32 halves, 0x3e3c3800, 0x46444240, {selector}; or.b32 halves, halves, {sign}; mov.b32 {{lo,hi}}, halves; cvt.f32.f16 {value}, lo; }}",value=out(reg32)value,selector=in(reg32)selector,sign=in(reg32)sign,options(nomem,nostack));
            value
        }};}
        macro_rules! pending {($n:expr)=>{asm!("cp.async.wait_group {n};",n=const {$n})};}
        let tid=_thread_idx_x() as usize;
        let rows_launch=launch.single.shape.rows as usize;let cols_launch=launch.single.shape.cols as usize;
        if _block_dim_x() as usize!=THREADS || DEPTH<3 || launch.count==0 || launch.batch==0 || rows_launch==0 || rows_launch%TILE_ROWS!=0 || cols_launch==0 || cols_launch%(16*ROUND_BLOCKS)!=0 {asm!("trap;");}
        let per_job=rows_launch/TILE_ROWS;let block=_block_idx_x() as usize;
        let (job,tile)=(block/per_job,block%per_job);
        if job>=launch.count as usize {return;}
        let p:Arguments=(launch.batch as *const Arguments).add(job).read();
        let shape=p.shape;let cols=shape.cols as usize;
        if shape.op!=Op::NvProject || shape.tokens==0 || shape.block_rows!=NV_TILE_ROWS || shape.rows as usize!=rows_launch || cols!=cols_launch || cols>crate::expert_ops::NV_TILE_MAX_COLS as usize || p.input[0]&3!=0 || (p.weight|p.scales)&15!=0 {asm!("trap;");}
        let blocks=cols/16;let rounds=blocks/ROUND_BLOCKS;
        let input=p.input[0] as *const f32;
        let codes=(p.weight as *const u8).add(tile*cols*64);
        let scales=(p.scales as *const u8).add(tile*blocks*128);
        let global=mul!(shape.global,shape.activation);
        // Shared layout in bytes.
        let words=blocks*17;let act_at=0usize;let code_at=(words*4).div_ceil(16)*16;let scale_at=code_at+DEPTH*ROUND_BLOCKS*1024;let total=scale_at+DEPTH*ROUND_BLOCKS*128;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if (dynamic as usize)<total {asm!("trap;");}
        // Stage every activation word of this job once.
        let mut word=tid;
        while word<words {crate::ptx::copy4((act_at+word*4) as u32,input.add(word).cast());word+=THREADS;}
        crate::ptx::commit();
        asm!("cp.async.wait_all;");
        crate::ptx::sync();
        // Issue the copies of round r: a thread brings 16-byte pieces of live columns (piece = block, column, quarter)
        // and the first threads bring the scales. Always one group.
        // NO L2 PREFETCH HERE (measured, lane NV-GAP): a prefetch pulls a whole 128-byte line, that is the live column
        // AND its 64-byte neighbour. For a dead neighbour that is bus traffic the skip was made to save: with the
        // prefetch a zero column saved time only when its line-mate was zero too (10 us a launch lost at the real
        // zero shares). The copies bring only the bytes they name.
        let issue=|r:usize| {
            if r<rounds {
                let stage=r%DEPTH;let mut piece=tid;
                while piece<ROUND_BLOCKS*64 {
                    let (q,column,quarter)=(piece/64,(piece%64)/4,piece%4);let blk=r*ROUND_BLOCKS+q;
                    if crate::ptx::load((act_at+(blk*17+column)*4) as u32)!=0.0 {
                        crate::ptx::copy16((code_at+(stage*ROUND_BLOCKS+q)*1024+column*64+quarter*16) as u32,codes.add((blk*16+column)*64+quarter*16).cast());
                    }
                    piece+=THREADS;
                }
                if tid<ROUND_BLOCKS*8 {
                    let (q,part)=(tid/8,tid%8);
                    crate::ptx::copy16((scale_at+(stage*ROUND_BLOCKS+q)*128+part*16) as u32,scales.add((r*ROUND_BLOCKS+q)*128+part*16).cast());
                }
            }
            crate::ptx::commit();
        };
        let mut r=0;while r<DEPTH-1 {issue(r);r+=1;}
        let mut sum=0.0f32;let shift=4*(tid as u32&1);
        r=0;
        while r<rounds {
            // This thread's copies of round r are complete; after the barrier every thread's are, and every thread
            // has finished round r-1, so its stage may be written again.
            pending!(DEPTH-2);
            crate::ptx::sync();
            issue(r+DEPTH-1);
            let stage=r%DEPTH;
            // One block of sixteen columns, written stage by stage so that neighbouring instructions do not wait for
            // each other: all loads, all decodes, all products, then the sixteen adds of the chain in column order.
            // A zero column was not copied: its staged bytes are stale. Its product is still +0 or -0 (any decoded
            // value is finite and its activation is zero), so its add changes nothing. No test is needed here.
            macro_rules! block {($q:literal)=>{{
                let activation=(act_at+(r*ROUND_BLOCKS+$q)*17*4) as u32;
                let activation_scale=crate::ptx::load(activation+64);
                let scale=crate::ptx::load_u8((scale_at+(stage*ROUND_BLOCKS+$q)*128+tid) as u32);
                let bytes=(code_at+(stage*ROUND_BLOCKS+$q)*1024+tid/2) as u32;
                let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)=(crate::ptx::load(activation+0), crate::ptx::load(activation+4), crate::ptx::load(activation+8), crate::ptx::load(activation+12), crate::ptx::load(activation+16), crate::ptx::load(activation+20), crate::ptx::load(activation+24), crate::ptx::load(activation+28), crate::ptx::load(activation+32), crate::ptx::load(activation+36), crate::ptx::load(activation+40), crate::ptx::load(activation+44), crate::ptx::load(activation+48), crate::ptx::load(activation+52), crate::ptx::load(activation+56), crate::ptx::load(activation+60));
                let (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15)=(crate::ptx::load_u8(bytes+0), crate::ptx::load_u8(bytes+64), crate::ptx::load_u8(bytes+128), crate::ptx::load_u8(bytes+192), crate::ptx::load_u8(bytes+256), crate::ptx::load_u8(bytes+320), crate::ptx::load_u8(bytes+384), crate::ptx::load_u8(bytes+448), crate::ptx::load_u8(bytes+512), crate::ptx::load_u8(bytes+576), crate::ptx::load_u8(bytes+640), crate::ptx::load_u8(bytes+704), crate::ptx::load_u8(bytes+768), crate::ptx::load_u8(bytes+832), crate::ptx::load_u8(bytes+896), crate::ptx::load_u8(bytes+960));
                let fold=mul!(mul!(crate::expert_ops::e4m3(scale as u8),activation_scale),global);
                let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15)=(decode!((b0>>shift)&15), decode!((b1>>shift)&15), decode!((b2>>shift)&15), decode!((b3>>shift)&15), decode!((b4>>shift)&15), decode!((b5>>shift)&15), decode!((b6>>shift)&15), decode!((b7>>shift)&15), decode!((b8>>shift)&15), decode!((b9>>shift)&15), decode!((b10>>shift)&15), decode!((b11>>shift)&15), decode!((b12>>shift)&15), decode!((b13>>shift)&15), decode!((b14>>shift)&15), decode!((b15>>shift)&15));
                let (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15)=(mul!(mul!(v0,a0),fold), mul!(mul!(v1,a1),fold), mul!(mul!(v2,a2),fold), mul!(mul!(v3,a3),fold), mul!(mul!(v4,a4),fold), mul!(mul!(v5,a5),fold), mul!(mul!(v6,a6),fold), mul!(mul!(v7,a7),fold), mul!(mul!(v8,a8),fold), mul!(mul!(v9,a9),fold), mul!(mul!(v10,a10),fold), mul!(mul!(v11,a11),fold), mul!(mul!(v12,a12),fold), mul!(mul!(v13,a13),fold), mul!(mul!(v14,a14),fold), mul!(mul!(v15,a15),fold));
                add!(sum,p0);add!(sum,p1);add!(sum,p2);add!(sum,p3);add!(sum,p4);add!(sum,p5);add!(sum,p6);add!(sum,p7);add!(sum,p8);add!(sum,p9);add!(sum,p10);add!(sum,p11);add!(sum,p12);add!(sum,p13);add!(sum,p14);add!(sum,p15);
            }};}
            block!(0);
            block!(1);
            block!(2);
            block!(3);
            r+=1;
        }
        // A decode job has one token. A prompt job of a part with several tokens writes its own token column of the
        // part's [row][token] output: `output` then names that token's first word and the stride is the token count.
        *(p.output as *mut f32).add((tile*TILE_ROWS+tid)*shape.tokens as usize)=crate::expert_ops::round(sum);
    }
}
