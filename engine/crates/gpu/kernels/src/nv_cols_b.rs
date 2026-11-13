//! NVFP4 decode batch projection over COLUMN-TILE weights on CONTRACT (b): the launch, the storage, the weight
//! staging and the zero-input skip of `nv_cols`, with the word function of the tensor-core family (`tc_contract`).
//! A word (weight row, token): for each 64-column step in ascending column order acc = chain_step(acc, ..) (the
//! integer function of GB10's block-scaled NVFP4 tensor-core instruction: exact 16-column block sums, exact scale
//! products, terms and the old accumulator lined up on the largest exponent, cut at 35 fraction bits, cut toward zero
//! to FP32); word = bf16(fl(acc * fl(global * activation))). FIRST matrices (`nv_cols_b`): one chain from +0.0 over
//! all columns. SECOND matrices (`nv_cols_bq`): four chains, one a quarter of the columns, each from +0.0, joined
//! fl(fl(a0+a1)+fl(a2+a3)). The same words as `fp4_cohort` at any row count, so a token decoded alone equals the same
//! token in a prompt group or a batch.
//! A round of `nv_cols` is 4 blocks of 16 columns = one 64-column step, anchored at column 0. A thread owns one row.
//! THE BLOCK SUM IN INTEGERS (TC-DECODE-COST, 2026-11-11). `chain_step` takes, a 16-column block, the exact sum of
//! (weight half units * activation half units) and whether one of the 16 products is not zero. The activations of a
//! job are staged once as signed int8 half units and their magnitudes. Four weight codes become four bytes of half
//! units with one `prmt` for the positive codes and one for the negative codes, and `dp4a` adds four products in
//! int32 (exact: a block sum is at most 16 * 144 in magnitude). An exact integer sum does not depend on the order of
//! its terms, so the bytes are taken in the order the nibbles come out (columns 0, 4, 1, 5 | 2, 6, 3, 7 of each 8,
//! `byte_of`) and the activation bytes are staged in that order. The sum and the live bit are those of the FP32 form
//! this replaces (exact products, an exact 16-term FP32 sum, the OR of the product bits), so every word is unchanged.
//! A column whose activation is +-0 is not copied: its staged code byte is stale, but its activation byte and its
//! magnitude byte are 0, so it adds 0 to the block sum and to the live count whatever the stale code is.
//! Device history (a lane box, 2026-11-11, all bit-exact): an integer multiply a weight 1.03 to 1.11 x nv_cols; a product
//! table a column 1.04 to 1.19 x (still 1.03 to 1.07 x with the chain step removed); exact FP32 block sums 1.024 to
//! 1.045 x (the form of the epoch 3 step one bundle, 0.59 ms a pair step over the rc.8 chain).
use crate::expert_ops::{Arguments,LaunchArguments,Op};

use crate::expert_ops::{NV_COLS_DEPTH,NV_COLS_ROUND_BLOCKS,NV_COLS_THREADS,NV_TILE_ROWS};
const TILE_ROWS:usize=NV_TILE_ROWS as usize;
const THREADS:usize=NV_COLS_THREADS as usize;
const ROUND_BLOCKS:usize=NV_COLS_ROUND_BLOCKS as usize;
const DEPTH:usize=NV_COLS_DEPTH as usize;
/// The half units of the eight E2M1 magnitudes (0, 1, 2, 3, 4, 6, 8, 12) as the eight bytes `prmt` selects from.
const HALF_LOW:u32=0x0302_0100;
const HALF_HIGH:u32=0x0c08_0604;
/// The byte of column `column` (0..16) of a block in its four staged activation words: the order in which `project`
/// takes the nibbles of eight columns (0, 4, 1, 5 in the first word, 2, 6, 3, 7 in the second).
#[inline(always)]
const fn byte_of(column:usize)->usize {8*(column/8)+2*(column%4)+(column%8)/4}

/// FIRST matrices (expert gate and up): one chain a word.
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_b(launch:LaunchArguments) {project::<1,false>(launch)}
/// SECOND matrices (expert down): the quarter chain. cols must be a multiple of 256.
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_bq(launch:LaunchArguments) {project::<4,false>(launch)}
/// A SECOND matrix split by INPUT COLUMNS (the pair split): each job's matrix holds 4, 2 or 1 WHOLE column quarters of
/// its second matrix (one, two or four owners) and has one token. The output of a job is the RAW quarter accumulators
/// of every row, `[row][quarter]`: the same chains as `nv_cols_bq`, one a quarter from +0.0, with no join, no
/// epilogue and no round. The fold and the join come after (`nv_quarter_fold`, `tp_reduce_take`).
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_q4(launch:LaunchArguments) {project::<4,true>(launch)}
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_q2(launch:LaunchArguments) {project::<2,true>(launch)}
#[no_mangle]
pub extern "ptx-kernel" fn nv_cols_q1(launch:LaunchArguments) {project::<1,true>(launch)}

#[inline(always)]
pub(crate) fn project<const CHAINS:usize,const RAW:bool>(launch:LaunchArguments) {
    // SAFETY: The host checks every job descriptor before the launch, in one of four places. A decode batch:
    // `expert_ops::nv_cols_admitted` (column-tile NvProject jobs of one token; output stride 1). A prompt group of few
    // tokens: `nv_small::admitted_columns` and the span checks of `Backend::expert_group`, which builds one job a
    // (part, token) with `nv_small::column_jobs` (output stride = the token count of the part). The one-token device route
    // (`Backend::device_route_projection`, and its recorded replay `resident_launch_nv_tile`) and the multi-row device route
    // (`Backend::device_route_projection_rows`): `device_route::admitted` checks the whole table of one role (equal rows,
    // cols, layout and role, aligned owners) when the table is made; the job table of a launch is then written ON THE
    // DEVICE by `device_route_build` (one token) or `device_route_rows` (several rows, a phase's descriptors listed by
    // (expert, pair)), one whole `Arguments` a job copied from that checked table (its last word is the role), with every
    // row's expert IDs checked there (distinct inside a row, in range, a trap otherwise) and the input and output set to
    // that job's own pair-order spans of the route's buffers; the launch follows the build on the one ordered stream. All four give rows a
    // multiple of 128, cols a multiple of 64 and at most NV_TILE_MAX_COLS, 16-byte aligned weight and scale owners and a
    // 4-byte aligned input. The host keeps all owners alive through stream completion; the job table (count entries of
    // equal rows, cols and role) is filled before the launch, by the host or, on the device route, by `device_route_build` or `device_route_rows`. Each job's weight owner holds rows * cols/2 bytes and its scale owner
    // rows * cols/16 bytes in the column-tile layout; its input owner holds cols/16*17 FP32 words. Its output words
    // are `output + (row * tokens) * 4` for row < rows (`tokens` is the stride in words, at least 1), all inside one
    // live private allocation: `rows` words of its own for a decode job; for a prompt job its token's column of the
    // rows * tokens `[row][token]` words of its part. Weights, scales and inputs are immutable. Thread r of the block
    // of (job, tile) is the only writer of the job's output word (tile*128 + r) * tokens (one raw store before any
    // read of it). Jobs of different parts own disjoint spans. The jobs of one part share a span: their first words
    // differ by less than the stride, so their words interleave and never meet. Every
    // asynchronous copy reads 16 bytes inside one column or one scale block of the tile; the input words are read
    // with plain loads, word 17*block + k for block < cols/16 and k <= 16, below cols/16*17. Shared addresses stay
    // inside the layout below (checked against the dynamic size), which is the only shared allocation of this kernel,
    // so it starts at 0; every vector access is 16-byte aligned in it. Data stored or copied by one thread is read by
    // another only after the writer's own wait (for a copy) and a bar.sync that every thread of the block executes
    // the same number of times (the loop bounds depend only on cols). A staged byte of a column that was not copied
    // is read as a value and never as an address. The body traps on any other launch.
    // RAW (`nv_cols_q4` / `q2` / `q1`, checked by `Backend::nv_quarters`): every job is a one-token decode job of a
    // second matrix, and its output owner holds rows * CHAINS words: thread r of the block of (job, tile) is the only
    // writer of the CHAINS words from (tile*128 + r) * CHAINS.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add {($sum:ident,$value:expr)=>{{let value:f32=$value;asm!("add.rn.f32 {sum}, {sum}, {value};",sum=inout(reg32)$sum,value=in(reg32)value,options(nomem,nostack));}};}
        // Four E2M1 codes (the four low nibbles of the selector) to four bytes of half units: a positive code gives
        // its magnitude; a negative code gives 0 (its sign bit makes prmt copy the sign of the table byte, which is 0).
        macro_rules! halves {($selector:expr)=>{{let value:u32;asm!("prmt.b32 {value}, {low}, {high}, {selector};",value=out(reg32)value,low=in(reg32)HALF_LOW,high=in(reg32)HALF_HIGH,selector=in(reg32)$selector,options(pure,nomem,nostack));value}};}
        // c + the four products of the unsigned bytes of a and the signed bytes of b (int32, exact here).
        macro_rules! dot {($a:expr,$b:expr,$c:expr)=>{{let value:i32;asm!("dp4a.u32.s32 {value}, {a}, {b}, {c};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,options(pure,nomem,nostack));value}};}
        macro_rules! dot_unsigned {($a:expr,$b:expr,$c:expr)=>{{let value:u32;asm!("dp4a.u32.u32 {value}, {a}, {b}, {c};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,options(pure,nomem,nostack));value}};}
        macro_rules! store4 {($at:expr,$a:expr,$b:expr,$c:expr,$d:expr)=>{asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",p=in(reg32)$at,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,d=in(reg32)$d)};}
        macro_rules! store1 {($at:expr,$a:expr)=>{asm!("st.shared.b32 [{p}], {a};",p=in(reg32)$at,a=in(reg32)$a)};}
        macro_rules! load1 {($at:expr)=>{{let value:u32;asm!("ld.shared.b32 {value}, [{p}];",value=out(reg32)value,p=in(reg32)$at);value}};}
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
        // Every job of the launch must name this body's word function.
        if RAW {if !(CHAINS==1 || CHAINS==2 || CHAINS==4) || rounds%CHAINS!=0 || shape.role!=crate::expert_ops::NV_SECOND || shape.tokens!=1 {asm!("trap;");}}
        else if (CHAINS!=1 && CHAINS!=4) || rounds%CHAINS!=0 || crate::expert_ops::nv_chains(&shape) as usize!=CHAINS {asm!("trap;");}
        let per_chain=rounds/CHAINS;
        let input=p.input[0] as *const f32;
        let codes=(p.weight as *const u8).add(tile*cols*64);
        let scales=(p.scales as *const u8).add(tile*blocks*128);
        let global=mul!(shape.global,shape.activation);
        // Shared layout in bytes: the signed activation bytes (16 a block, in the order of `byte_of`), their
        // magnitudes (16 a block), the UE4M3 parts of each block's activation scale (one word a block), then
        // NV_COLS_DEPTH rounds of codes and scales. It is smaller than `expert_ops::nv_cols_shared` (the size of the
        // FP32 rows this form replaced), which is what the host passes.
        let act_at=0usize;let mag_at=blocks*16;let parts_at=blocks*32;let code_at=(blocks*36).div_ceil(16)*16;
        let scale_at=code_at+DEPTH*ROUND_BLOCKS*1024;let total=scale_at+DEPTH*ROUND_BLOCKS*128;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if (dynamic as usize)<total {asm!("trap;");}
        // Stage the activations of this job once: thread t takes blocks t, t + THREADS, ... A value of the quantizer is
        // +-0 or one of +-0.5 .. +-6, so twice it is an integer of magnitude at most 12, exactly.
        let mut staged=tid;
        while staged<blocks {
            let row=input.add(staged*17);
            macro_rules! half {($k:expr)=>{mul!(row.add($k).read(),2.0f32).to_int_unchecked::<i32>()}}
            let (h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15)=(half!(0),half!(1),half!(2),half!(3),half!(4),half!(5),half!(6),half!(7),half!(8),half!(9),half!(10),half!(11),half!(12),half!(13),half!(14),half!(15));
            let scale=row.add(16).read();
            macro_rules! signed {($a:expr,$b:expr,$c:expr,$d:expr)=>{($a as u32&0xff)|(($b as u32&0xff)<<8)|(($c as u32&0xff)<<16)|(($d as u32&0xff)<<24)}}
            macro_rules! magnitude {($a:expr,$b:expr,$c:expr,$d:expr)=>{$a.unsigned_abs()|($b.unsigned_abs()<<8)|($c.unsigned_abs()<<16)|($d.unsigned_abs()<<24)}}
            store4!((act_at+staged*16) as u32,signed!(h0,h4,h1,h5),signed!(h2,h6,h3,h7),signed!(h8,h12,h9,h13),signed!(h10,h14,h11,h15));
            store4!((mag_at+staged*16) as u32,magnitude!(h0,h4,h1,h5),magnitude!(h2,h6,h3,h7),magnitude!(h8,h12,h9,h13),magnitude!(h10,h14,h11,h15));
            store1!((parts_at+staged*4) as u32,crate::tc_contract::pack_parts(crate::tc_contract::ue4m3_parts_f32(scale.to_bits())));
            staged+=THREADS;
        }
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
                    if crate::ptx::load_u8((act_at+blk*16+byte_of(column)) as u32)!=0 {
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
        let mut acc=0u32;
        // This thread's nibble of a code byte: the low one for an even row, the high one for an odd row.
        let (shift,lift)=(4*(tid as u32&1),4-4*(tid as u32&1));
        // The finished chains of a quarter-chain word (unused for one chain).
        let mut part=[0u32;4];let (mut left,mut quarter)=(per_chain,0usize);
        r=0;
        while r<rounds {
            // This thread's copies of round r are complete; after the barrier every thread's are, and every thread
            // has finished round r-1, so its stage may be written again.
            pending!(DEPTH-2);
            crate::ptx::sync();
            issue(r+DEPTH-1);
            let stage=r%DEPTH;
            // One block of sixteen columns: the sixteen code bytes of this thread's row pair, packed four to a word,
            // the nibbles of this row in two words (columns 0, 4, 1, 5, 2, 6, 3, 7 of each 8, a byte each), then four
            // bytes of half units at a time for the positive and for the negative codes, and the products by dp4a.
            macro_rules! block {($q:literal)=>{{
                let blk=r*ROUND_BLOCKS+$q;
                let scale=crate::ptx::load_u8((scale_at+(stage*ROUND_BLOCKS+$q)*128+tid) as u32);
                let parts=load1!((parts_at+blk*4) as u32);
                let signed=crate::ptx::load4((act_at+blk*16) as u32);
                let magnitude=crate::ptx::load4((mag_at+blk*16) as u32);
                let bytes=(code_at+(stage*ROUND_BLOCKS+$q)*1024+tid/2) as u32;
                let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15)=(crate::ptx::load_u8(bytes+0), crate::ptx::load_u8(bytes+64), crate::ptx::load_u8(bytes+128), crate::ptx::load_u8(bytes+192), crate::ptx::load_u8(bytes+256), crate::ptx::load_u8(bytes+320), crate::ptx::load_u8(bytes+384), crate::ptx::load_u8(bytes+448), crate::ptx::load_u8(bytes+512), crate::ptx::load_u8(bytes+576), crate::ptx::load_u8(bytes+640), crate::ptx::load_u8(bytes+704), crate::ptx::load_u8(bytes+768), crate::ptx::load_u8(bytes+832), crate::ptx::load_u8(bytes+896), crate::ptx::load_u8(bytes+960));
                let (w0,w1,w2,w3)=(c0|(c1<<8)|(c2<<16)|(c3<<24),c4|(c5<<8)|(c6<<16)|(c7<<24),c8|(c9<<8)|(c10<<16)|(c11<<24),c12|(c13<<8)|(c14<<16)|(c15<<24));
                let (z0,z1)=(((w0>>shift)&0x0f0f_0f0f)|((w1<<lift)&0xf0f0_f0f0),((w2>>shift)&0x0f0f_0f0f)|((w3<<lift)&0xf0f0_f0f0));
                let (n0,n1)=(z0^0x8888_8888,z1^0x8888_8888);
                let (p0,p1,p2,p3)=(halves!(z0),halves!(z0>>16),halves!(z1),halves!(z1>>16));
                let (m0,m1,m2,m3)=(halves!(n0),halves!(n0>>16),halves!(n1),halves!(n1>>16));
                let positive=dot!(p3,signed[3],dot!(p2,signed[2],dot!(p1,signed[1],dot!(p0,signed[0],0i32))));
                let negative=dot!(m3,signed[3],dot!(m2,signed[2],dot!(m1,signed[1],dot!(m0,signed[0],0i32))));
                let live=dot_unsigned!(p3|m3,magnitude[3],dot_unsigned!(p2|m2,magnitude[2],dot_unsigned!(p1|m1,magnitude[1],dot_unsigned!(p0|m0,magnitude[0],0u32))));
                (positive-negative,live!=0,crate::tc_contract::ue4m3_parts(scale),crate::tc_contract::unpack_parts(parts))
            }};}
            let (b0,b1,b2,b3)=(block!(0),block!(1),block!(2),block!(3));
            acc=crate::tc_contract::chain_step(acc,[b0.0,b1.0,b2.0,b3.0],[b0.1,b1.1,b2.1,b3.1],[b0.2,b1.2,b2.2,b3.2],[b0.3,b1.3,b2.3,b3.3]);
            r+=1;
            // A quarter ends: keep its chain and start the next one from +0.0.
            // (A countdown, not a division a round.)
            if CHAINS!=1 {left-=1;if left==0 {part[quarter&3]=acc;acc=0;quarter+=1;left=per_chain;}}
        }
        if RAW {
            // The raw quarter accumulators of this row, in quarter order.
            let raw=(p.output as *mut u32).add((tile*TILE_ROWS+tid)*CHAINS);
            if CHAINS==1 {*raw=acc;} else {let mut q=0;while q<CHAINS {*raw.add(q)=part[q];q+=1;}}
            return;
        }
        if CHAINS==4 {
            let (mut low,high)=(f32::from_bits(part[0]),f32::from_bits(part[2]));let mut high=high;
            add!(low,f32::from_bits(part[1]));add!(high,f32::from_bits(part[3]));add!(low,high);
            acc=low.to_bits();
        }
        // A decode job has one token. A prompt job of a part with several tokens writes its own token column of the
        // part's [row][token] output: `output` then names that token's first word and the stride is the token count.
        *(p.output as *mut f32).add((tile*TILE_ROWS+tid)*shape.tokens as usize)=crate::expert_ops::round(mul!(f32::from_bits(acc),global));
    }
}
