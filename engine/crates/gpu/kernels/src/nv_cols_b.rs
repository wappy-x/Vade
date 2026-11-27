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
        use crate::canonical::{Acc,ZERO};
        macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! add_rn {($a:expr,$b:expr)=>{{let value:f32;asm!("add.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));value}};}
        macro_rules! store1 {($at:expr,$a:expr)=>{asm!("st.shared.b32 [{p}], {a};",p=in(reg32)$at,a=in(reg32)$a)};}
        macro_rules! zero16 {($at:expr)=>{asm!("st.shared.v4.b32 [{p}], {{{z}, {z}, {z}, {z}}};",p=in(reg32)$at,z=in(reg32)0u32)};}
        macro_rules! load1 {($at:expr)=>{{let value:u32;asm!("ld.shared.b32 {value}, [{p}];",value=out(reg32)value,p=in(reg32)$at);value}};}
        macro_rules! shfl_bfly {($v:expr,$d:expr)=>{{let o:u32;asm!("shfl.sync.bfly.b32 {o}, {v}, {d}, 31, -1;",o=out(reg32)o,v=in(reg32)$v,d=const {$d},options(pure,nomem,nostack));o}};}
        macro_rules! shfl_idx {($v:expr,$s:expr)=>{{let o:u32;asm!("shfl.sync.idx.b32 {o}, {v}, {s}, 31, -1;",o=out(reg32)o,v=in(reg32)$v,s=in(reg32)$s,options(pure,nomem,nostack));o}};}
        macro_rules! mma {($a:expr,$b:expr,$sa:expr,$sb:expr,$c:expr)=>{{
            let (a,b,sa,sb,c):([u32;4],[u32;2],u32,u32,Acc)=($a,$b,$sa,$sb,$c);
            let (d0,d1,d2,d3):(f32,f32,f32,f32);
            asm!(
                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
                 {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, \
                 {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
                d0=out(reg32)d0,d1=out(reg32)d1,d2=out(reg32)d2,d3=out(reg32)d3,
                a0=in(reg32)a[0],a1=in(reg32)a[1],a2=in(reg32)a[2],a3=in(reg32)a[3],
                b0=in(reg32)b[0],b1=in(reg32)b[1],
                c0=in(reg32)c.a,c1=in(reg32)c.b,c2=in(reg32)c.c,c3=in(reg32)c.d,
                sa=in(reg32)sa,sb=in(reg32)sb,options(nomem,nostack),
            );
            Acc {a:d0,b:d1,c:d2,d:d3}
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
        // Every job of the launch must name this body's word function.
        if RAW {if !(CHAINS==1 || CHAINS==2 || CHAINS==4) || rounds%CHAINS!=0 || shape.role!=crate::expert_ops::NV_SECOND || shape.tokens!=1 {asm!("trap;");}}
        else if (CHAINS!=1 && CHAINS!=4) || rounds%CHAINS!=0 || crate::expert_ops::nv_chains(&shape) as usize!=CHAINS {asm!("trap;");}
        let per_chain=rounds/CHAINS;
        let input=p.input[0] as *const f32;
        let codes=(p.weight as *const u8).add(tile*cols*64);
        let scales=(p.scales as *const u8).add(tile*blocks*128);
        let global=mul!(shape.global,shape.activation);
        // Shared layout in bytes: packed E2M1 activation codes (cols/2), packed UE4M3 activation scales (cols/16),
        // then NV_COLS_DEPTH rounds of XOR-swizzled column-tile codes (4 KB/round) and scales (512 B/round).
        let act_at=0usize;let act_scale_at=blocks*8;let code_at=(blocks*9).div_ceil(16)*16;
        let scale_at=code_at+DEPTH*ROUND_BLOCKS*1024;let total=scale_at+DEPTH*ROUND_BLOCKS*128;
        let dynamic:u32;asm!("mov.u32 {d}, %dynamic_smem_size;",d=out(reg32)dynamic,options(nomem,nostack));
        if (dynamic as usize)<total {asm!("trap;");}
        // Stage the quantized activations of this job once as E2M1 nibbles + UE4M3 scale bytes.
        let mut staged=tid;
        while staged<blocks {
            let row=input.add(staged*17);
            macro_rules! code {($k:expr)=>{crate::tc_contract::e2m1_code_f32(row.add($k).read().to_bits())}}
            let lo=code!(0)|(code!(1)<<4)|(code!(2)<<8)|(code!(3)<<12)|(code!(4)<<16)|(code!(5)<<20)|(code!(6)<<24)|(code!(7)<<28);
            let hi=code!(8)|(code!(9)<<4)|(code!(10)<<8)|(code!(11)<<12)|(code!(12)<<16)|(code!(13)<<20)|(code!(14)<<24)|(code!(15)<<28);
            store1!((act_at+staged*8) as u32,lo);
            store1!((act_at+staged*8+4) as u32,hi);
            let sc=crate::tc_contract::ue4m3_code_f32(row.add(16).read().to_bits());
            asm!("st.shared.u8 [{p}], {v};",p=in(reg32)(act_scale_at+staged) as u32,v=in(reg32)sc);
            staged+=THREADS;
        }
        crate::ptx::sync();
        // Issue the copies of round r with 4-way XOR swizzle and zero-column skip (both first and second matrices
        // have ~20-35% E2M1 zero-quantized activation columns whose weight reads can be skipped bit-identically).
        let issue=|r:usize| {
            if r<rounds {
                let stage=r%DEPTH;let mut piece=tid;
                while piece<ROUND_BLOCKS*64 {
                    let (col,q)=(piece/4,piece%4);let column=r*64+col;
                    let swz_q=q^((col>>1)&3);
                    let dst=(code_at+stage*4096+col*64+swz_q*16) as u32;
                    if ((load1!((act_at+(column/8)*4) as u32)>>(4*(column&7)))&7)==0 {zero16!(dst);}
                    else {crate::ptx::copy16(dst,codes.add(column*64+q*16).cast());}
                    piece+=THREADS;
                }
                if tid<ROUND_BLOCKS*8 {
                    crate::ptx::copy16((scale_at+stage*512+tid*16) as u32,scales.add(r*512+tid*16).cast());
                }
            }
            crate::ptx::commit();
        };
        let mut r=0;while r<DEPTH-1 {issue(r);r+=1;}
        let (warp,lane)=(tid/32,tid%32);
        let (g,t)=(lane/4,lane%4);
        let j=lane&7;
        let remap_src=(t*8+g) as u32;
        let mut acc=[ZERO;2];
        let (mut q0,mut q1,mut q2,mut q3)=([0.0f32;4],[0.0f32;4],[0.0f32;4],[0.0f32;4]);
        let (mut left,mut quarter)=(per_chain,0usize);
        // Load 8 rows x 32 columns of column-tile codes into registers, transpose 8x8 nibbles via 3 butterfly shuffles,
        // and remap to mma.m16n8k64 A-fragment lane order with zero extra SMEM and zero extra barriers.
        macro_rules! load_8x32 {($s_code:expr,$rb:expr,$col_base:expr)=>{{
            let c=$col_base+lane;let q=($rb)>>2;let sub=($rb)&3;let swz_q=q^((c>>1)&3);
            let mut v=load1!(($s_code+c*64+swz_q*16+sub*4) as u32);
            let p4=shfl_bfly!(v,4);
            v=if (j&4)==0 {(v&0x0000_ffff)|((p4&0x0000_ffff)<<16)} else {(p4>>16)|(v&0xffff_0000)};
            let p2=shfl_bfly!(v,2);
            v=if (j&2)==0 {(v&0x00ff_00ff)|((p2&0x00ff_00ff)<<8)} else {((p2>>8)&0x00ff_00ff)|(v&0xff00_ff00)};
            let p1=shfl_bfly!(v,1);
            v=if (j&1)==0 {(v&0x0f0f_0f0f)|((p1&0x0f0f_0f0f)<<4)} else {((p1>>4)&0x0f0f_0f0f)|(v&0xf0f0_f0f0)};
            shfl_idx!(v,remap_src)
        }};}
        macro_rules! load_sa {($s_scale:expr,$r0:expr)=>{{
            let rs=($s_scale+$r0+g+(t&1)*8) as u32;
            crate::ptx::load_u8(rs)|(crate::ptx::load_u8(rs+128)<<8)|(crate::ptx::load_u8(rs+256)<<16)|(crate::ptx::load_u8(rs+384)<<24)
        }};}
        r=0;
        while r<rounds {
            pending!(DEPTH-2);
            crate::ptx::sync();
            issue(r+DEPTH-1);
            let stage=r%DEPTH;
            let s_code=code_at+stage*4096;
            let s_scale=scale_at+stage*512;
            let (b0,b1,sb)=if g==0 {
                (load1!((act_at+(r*8+t)*4) as u32),load1!((act_at+(r*8+t+4)*4) as u32),load1!((act_scale_at+r*4) as u32))
            } else {(0u32,0u32,0u32)};
            let rb0=warp*4;
            let a0=[load_8x32!(s_code,rb0,0),load_8x32!(s_code,rb0+1,0),load_8x32!(s_code,rb0,32),load_8x32!(s_code,rb0+1,32)];
            let sa0=load_sa!(s_scale,warp*32);
            acc[0]=mma!(a0,[b0,b1],sa0,sb,acc[0]);
            let a1=[load_8x32!(s_code,rb0+2,0),load_8x32!(s_code,rb0+3,0),load_8x32!(s_code,rb0+2,32),load_8x32!(s_code,rb0+3,32)];
            let sa1=load_sa!(s_scale,warp*32+16);
            acc[1]=mma!(a1,[b0,b1],sa1,sb,acc[1]);
            r+=1;
            if CHAINS!=1 {
                left-=1;
                if left==0 {
                    let snap=[acc[0].a,acc[0].c,acc[1].a,acc[1].c];
                    match quarter {0=>q0=snap,1=>q1=snap,2=>q2=snap,_=>q3=snap}
                    acc=[ZERO;2];quarter+=1;left=per_chain;
                }
            }
        }
        if t!=0 {return;}
        let r0=tile*TILE_ROWS+warp*32+g;
        let out_rows=[r0,r0+8,r0+16,r0+24];
        if RAW {
            let base=p.output as *mut u32;
            let cur=[acc[0].a,acc[0].c,acc[1].a,acc[1].c];
            let mut e=0;
            while e<4 {
                let raw=base.add(out_rows[e]*CHAINS);
                if CHAINS==1 {*raw=cur[e].to_bits();}
                else {
                    *raw=q0[e].to_bits();*raw.add(1)=q1[e].to_bits();
                    if CHAINS==4 {*raw.add(2)=q2[e].to_bits();*raw.add(3)=q3[e].to_bits();}
                }
                e+=1;
            }
            return;
        }
        let vals=if CHAINS==4 {
            [add_rn!(add_rn!(q0[0],q1[0]),add_rn!(q2[0],q3[0])),add_rn!(add_rn!(q0[1],q1[1]),add_rn!(q2[1],q3[1])),
             add_rn!(add_rn!(q0[2],q1[2]),add_rn!(q2[2],q3[2])),add_rn!(add_rn!(q0[3],q1[3]),add_rn!(q2[3],q3[3]))]
        } else {[acc[0].a,acc[0].c,acc[1].a,acc[1].c]};
        let stride=shape.tokens as usize;let out=p.output as *mut f32;
        let mut e=0;
        while e<4 {*out.add(out_rows[e]*stride)=crate::expert_ops::round(mul!(vals[e],global));e+=1;}
    }
}
