//! MXFP8 rows for a SMALL token count (a draft verify pass), `mx_rows_tile` with the token tile as a parameter T = 2 or 4.
//! Same arithmetic and order as `mx_rows_tile`, `flash_next_mxfp8_bits` / `_pair` / `_m1` (`next_decode.rs` `row`): the output for
//! (weight row, token) is a fixed tree over the 32 lanes of a warp. Lane L owns columns `tile * 256 + L * 8 + J`, J = 0..7, of every
//! 256-column tile and keeps ONE chain `acc = fma(weight, activation, acc)` over its columns in tile order and J order, where
//! `weight = e4m3(code) * ue8m0(scale)`; the lane partials then combine with the `shfl_down` tree d = 16, 8, 4, 2, 1 and lane zero
//! stores the sum. What changes against `mx_rows_tile`: that kernel always makes 16 tokens of FMAs and 64 reduction trees for a
//! weight, so a call of 1 to 4 tokens pays the work of 16 (126 us flat at 4096 x 4096 against a 78 us byte floor). Here a lane makes
//! the FMAs and trees of T tokens only, and the shared activation stage holds T tokens.
//!
//! Launch: entries `mx_rows_small2/4` (T = 2, 4), the same `MxProjection` as the kernels above (one projection), grid
//! (ceil(rows / 16), ceil(tokens / T)), 128 threads, no dynamic shared memory. Same admission as `mx_rows_tile` (round 0, FP32
//! activations, cols % 32 == 0, weights and input 16-byte aligned, output 4-byte aligned).
use crate::next_params::MxProjection;

const TILE:usize=256;

#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small2(p:MxProjection) {project::<2,4096>(p)}
#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small4(p:MxProjection) {project::<4,8192>(p)}

/// `T` tokens a block; `SHARED` = 2 * T * 256 * 4 bytes (two stages of T token tiles).
fn project<const T:usize,const SHARED:usize>(p:MxProjection) {
    // SAFETY: The host checks the descriptor as MxLayer::push does (complete E4M3 rows and UE8M0 groups
    // in the retained arena, input rows for all tokens, a disjoint output of tokens x rows words) and
    // keeps every owner alive through stream completion. Weights, scales and input are immutable; each
    // (token, row) output word has exactly one writer (lane zero of the warp that owns the row). The
    // kernel traps unless round and input_bf16 are zero, cols is a multiple of 32, weights and input are
    // 16-byte aligned and the block has 128 threads. Rows past `rows` are never read or written; tokens
    // past `tokens` are never copied or written (T=4 with 3 tokens: the 4th token tile of a stage is not copied and its shared bytes are unwritten; they are read
    // only into the accumulator of the missing token, which has no path into another token's sum, and are never stored). Weight reads stay inside row * cols + column below cols
    // and scale reads inside row * cols / 32 + column / 32. Shared reads and writes stay inside the two
    // static stages. All 128 threads reach every barrier; loop bounds depend only on the descriptor.
    unsafe {
        use core::arch::asm;
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const OFF:usize=4;rep4!(w,{let $v:usize=OFF+w;$body});}}};}
        macro_rules! fma {($acc:expr,$a:expr,$b:expr)=>{{asm!("fma.rn.f32 {c}, {a}, {b}, {c};",c=inout(reg32)$acc,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));}};}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let (n,k,m)=(p.rows as usize,p.cols as usize,p.tokens as usize);
        if core::arch::nvptx::_block_dim_x()!=128 || p.round!=0 || p.input_bf16!=0 || k%32!=0 || (p.weights|p.input)&15!=0 {core::arch::asm!("trap;");}
        let (row_tile,token_tile)=(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize);
        if SHARED!=2*T*TILE*4 {core::arch::asm!("trap;");}
        if row_tile*16>=n || token_tile*T>=m {return;}
        let (warp,lane)=(tid/32,tid%32);
        let smem=crate::ptx::shared::<SHARED>();
        let stage_bytes=T*TILE*4;
        let tiles=k.div_ceil(TILE);
        let input=p.input as *const f32;
        let issue=|c:usize| {
            let stage=smem+((c&1)*stage_bytes) as u32;
            let mut item=tid;
            while item<T*64 {
                let (t,q)=(item/64,item%64);let token=token_tile*T+t;let column=c*TILE+q*4;
                if token<m && column<k {
                    // The chunk q holds columns q*4 .. q*4+3: lane q/2, first or second half of the lane's eight.
                    crate::ptx::copy16(stage+(t*1024+(q%2)*512+(q/2)*16) as u32,input.add(token*k+column).cast());
                }
                item+=128;
            }
        };
        let mut acc=[[0.0f32;T];4];
        let rows=[row_tile*16+warp*4,row_tile*16+warp*4+1,row_tile*16+warp*4+2,row_tile*16+warp*4+3];
        // The next tile's raw weight words and scale codes are loaded before the current tile is computed.
        let load_raw=|tile:usize|->([[u32;2];4],[u32;4]) {
            let at=tile*TILE+lane*8;
            let mut words=[[0u32;2];4];let mut codes=[0u32;4];
            if at<k {
                rep4!(i,{
                    let row=rows[i];
                    if row<n {
                        let (low,high):(u32,u32);
                        asm!("ld.global.v2.u32 {{{low},{high}}}, [{pointer}];",low=out(reg32)low,high=out(reg32)high,pointer=in(reg64)(p.weights as *const u8).add(row*k+at),options(nostack,readonly));
                        words[i]=[low,high];codes[i]=*(p.scales as *const u8).add(row*k/32+at/32) as u32;
                    }
                });
            }
            (words,codes)
        };
        issue(0);crate::ptx::commit();
        let (mut words,mut codes)=load_raw(0);
        let mut c=0;
        while c<tiles {
            let (next_words,next_codes)=if c+1<tiles {load_raw(c+1)} else {([[0u32;2];4],[0u32;4])};
            if c+1<tiles {issue(c+1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
            crate::ptx::sync();
            let at=c*TILE+lane*8;
            if at<k {
                let mut ww=[[0.0f32;8];4];
                rep4!(i,{
                    let code=codes[i];
                    let scale=f32::from_bits(if code==0 {0x00400000} else if code==255 {0x7fc00000} else {code<<23});
                    rep8!(j,{
                        let byte=((words[i][j/4]>>((j&3)*8))&255) as u8;
                        let x=(byte&127) as u32;
                        let magnitude=if x==127 {f32::from_bits(0x7fc00000)} else if x<8 {x as f32*0.001953125} else {f32::from_bits((x<<20)+(120<<23))};
                        let value=if byte&128!=0 {-magnitude} else {magnitude};
                        let product:f32;asm!("mul.rn.f32 {r}, {a}, {b};",r=out(reg32)product,a=in(reg32)value,b=in(reg32)scale,options(nomem,nostack));
                        ww[i][j]=product;
                    });
                });
                let lane_base=smem+((c&1)*stage_bytes+lane*16) as u32;
                let mut t=0;
                while t<T {
                    let first=crate::ptx::load4(lane_base+(t*1024) as u32);let second=crate::ptx::load4(lane_base+(t*1024+512) as u32);
                    let xx=[f32::from_bits(first[0]),f32::from_bits(first[1]),f32::from_bits(first[2]),f32::from_bits(first[3]),
                        f32::from_bits(second[0]),f32::from_bits(second[1]),f32::from_bits(second[2]),f32::from_bits(second[3])];
                    rep4!(i,{rep8!(j,{fma!(acc[i][t],ww[i][j],xx[j]);});});
                    t+=1;
                }
            }
            crate::ptx::sync();
            words=next_words;codes=next_codes;
            c+=1;
        }
        rep4!(i,{
            let mut t=0;
            while t<T {
                let mut v=acc[i][t];
                let mut d=16u32;
                while d!=0 {
                    let x:f32;
                    asm!("shfl.sync.down.b32 {x}, {v}, {d}, 31, -1;",x=out(reg32)x,v=in(reg32)v,d=in(reg32)d,options(nomem,nostack));
                    v+=x;d/=2;
                }
                let token=token_tile*T+t;
                if lane==0 && rows[i]<n && token<m {*(p.output as *mut f32).add(token*n+rows[i])=v;}
                t+=1;
            }
        });
    }
}

/// One-token MXFP8 decode body for the recorded entry `resident_launch_mx` (lane DENSE-DELIVERY, 2026-11-08). Same grid, same
/// projection pick and same arithmetic as `next_decode::resident_flash_next_mxfp8_m1` (`project::<true, true>` -> `row::<1, true, true>`):
/// block b runs the projection whose `ends` range holds b, warp w of the block owns row `(b - start) % ceil(rows / 4) * 4 + w`, lane L
/// owns columns `tile * 256 + L * 8 + J`, and keeps ONE chain `acc = fma(e4m3 * ue8m0, x, acc)` in tile order and J order; then the
/// `shfl_down` tree d = 16, 8, 4, 2, 1 and lane zero stores (BF16 rounded when `round`). Only the load schedule changes:
/// - m1 loads 16 weight bytes a lane (half unused) and waits for them before the next tile: 256 bytes in flight a warp. Here a lane
///   loads only its own 8 bytes and its scale byte (non-coherent, no L1 allocation), `DEPTH` tiles ahead of its FMAs.
/// - m1 also prefetches row r + 8 into L2 every tile. That is dropped: with 512 blocks in one wave row r + 8 is being read by another
///   warp at the same time, and in this body the prefetch cost 35% (kernel-bench --mx-wave, 2026-11-08: 142 against 224 GB/s).
/// No value, operand or order changes; `kernel-bench --mx-wave` compares every output byte with the m1 entry.
#[inline(always)]
pub(crate) fn decode<const DEPTH:usize>(layer:*const crate::next_params::MxLayer) {
    // SAFETY: The checked recorder replays the m1 entry's descriptor (MxLayer::push: complete E4M3 rows and UE8M0 groups, inputs and
    // disjoint outputs of rows words, all live through stream completion) with the m1 grid of ends[count] blocks of 128 threads; `layer`
    // is the aligned, initialized payload of the replayed command. Weight
    // reads stay at row * cols + column with column < cols (cols % 32 == 0, the 8 bytes of a lane never cross a row end), scale reads at
    // row * cols / 32 + column / 32, input reads at column .. column + 8 of token 0. Tiles at or past the last are never loaded. Each
    // output word has one writer (lane zero of the row's warp). No shared memory and no barrier; the shuffle runs in full warps (a warp
    // returns as a whole when its row is past `rows`).
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_idx_x,_thread_idx_x};
        macro_rules! eight {($element:ident,$body:block)=>{{{const $element:usize=0;$body}{const $element:usize=1;$body}{const $element:usize=2;$body}{const $element:usize=3;$body}
            {const $element:usize=4;$body}{const $element:usize=5;$body}{const $element:usize=6;$body}{const $element:usize=7;$body}}};}
        let block=_block_idx_x() as u32;
        let count=(*layer).count as usize;
        let mut index=0usize;
        eight!(PROJECTION,{if PROJECTION+1<count && block>=(*layer).ends[PROJECTION+1] {index+=1;}});
        let projection=(*layer).projections[index];
        let start=if index==0 {0} else {(*layer).ends[index]};
        let local=block-start;
        let quads=projection.rows.div_ceil(4);
        let row=(local%quads) as usize*4+_thread_idx_x() as usize/32;
        if row>=projection.rows as usize {return;}
        let lane=_thread_idx_x()&31;
        let columns=projection.cols;
        let lane_tiles=if lane*8<columns {(columns-lane*8).div_ceil(256)} else {0};
        let weight_pointer=(projection.weights as *const u8).add(row*columns as usize+lane as usize*8);
        let scale_pointer=(projection.scales as *const u8).add(row*(columns/32) as usize+lane as usize/4);
        macro_rules! chain {($bf16:literal,$round:literal,$input:expr)=>{{
            const BF16:bool=$bf16;
            const ROUND:bool=$round;
            let input_pointer=$input;
            let load=|tile:u32|->([u32;2],u32) {
                if tile<lane_tiles {
                    let (low,high,code):(u32,u32,u32);
                    asm!("ld.global.nc.L1::no_allocate.v2.u32 {{{low},{high}}}, [{pointer}];",low=out(reg32)low,high=out(reg32)high,pointer=in(reg64)weight_pointer.add(tile as usize*256),options(nostack,readonly));
                    asm!("ld.global.nc.L1::no_allocate.u8 {code}, [{pointer}];",code=out(reg32)code,pointer=in(reg64)scale_pointer.add(tile as usize*8),options(nostack,readonly));
                    ([low,high],code)
                } else {([0,0],0)}
            };
            let mut queue=[([0u32;2],0u32);DEPTH];
            let mut slot=0;
            while slot<DEPTH {queue[slot]=load(slot as u32);slot+=1;}
            let mut accumulator=0.0f32;
            let mut tile=0u32;
            while tile<lane_tiles {
                let (words,scale_code)=queue[0];
                let mut slot=0;
                while slot+1<DEPTH {queue[slot]=queue[slot+1];slot+=1;}
                queue[DEPTH-1]=load(tile+DEPTH as u32);
                let offset=tile as usize*256;
                let scale=f32::from_bits(if scale_code==0 {0x00400000} else if scale_code==255 {0x7fc00000} else {scale_code<<23});
                let mut inputs=[0.0f32;8];
                if BF16 {
                    let (word0,word1,word2,word3):(u32,u32,u32,u32);
                    asm!("ld.global.v4.u32 {{{word0},{word1},{word2},{word3}}}, [{pointer}];",word0=out(reg32)word0,word1=out(reg32)word1,word2=out(reg32)word2,word3=out(reg32)word3,pointer=in(reg64)input_pointer.add(offset*2),options(nostack,readonly));
                    let packed=[word0,word1,word2,word3];
                    eight!(ELEMENT,{inputs[ELEMENT]=f32::from_bits((packed[ELEMENT/2]>>((ELEMENT&1)*16))<<16);});
                } else {
                    let (word0,word1,word2,word3,word4,word5,word6,word7):(u32,u32,u32,u32,u32,u32,u32,u32);
                    asm!("ld.global.v4.u32 {{{word0},{word1},{word2},{word3}}}, [{pointer}];",word0=out(reg32)word0,word1=out(reg32)word1,word2=out(reg32)word2,word3=out(reg32)word3,pointer=in(reg64)input_pointer.add(offset*4),options(nostack,readonly));
                    asm!("ld.global.v4.u32 {{{word4},{word5},{word6},{word7}}}, [{pointer}];",word4=out(reg32)word4,word5=out(reg32)word5,word6=out(reg32)word6,word7=out(reg32)word7,pointer=in(reg64)input_pointer.add(offset*4+16),options(nostack,readonly));
                    let packed=[word0,word1,word2,word3,word4,word5,word6,word7];
                    eight!(ELEMENT,{inputs[ELEMENT]=if ROUND {crate::flash_gdn::rounded(f32::from_bits(packed[ELEMENT]))} else {f32::from_bits(packed[ELEMENT])};});
                }
                eight!(ELEMENT,{
                    if ELEMENT%2==0 {
                        let raw=(words[ELEMENT/4]>>((ELEMENT&3)*8)) as u16;
                        let (halves,first_value,second_value):(u32,f32,f32);
                        asm!("{{ .reg .b16 lo,hi; cvt.rn.f16x2.e4m3x2 {halves}, {raw}; mov.b32 {{lo,hi}}, {halves}; cvt.f32.f16 {first}, lo; cvt.f32.f16 {second}, hi; }}",
                            halves=out(reg32)halves,first=out(reg32)first_value,second=out(reg32)second_value,raw=in(reg16)raw,options(pure,nomem,nostack));
                        let _=halves;
                        let (first,second)=(first_value*scale,second_value*scale);
                        accumulator=core::f32::math::mul_add(first,inputs[ELEMENT],accumulator);
                        accumulator=core::f32::math::mul_add(second,inputs[ELEMENT+1],accumulator);
                    }
                });
                tile+=1;
            }
            accumulator
        }};}
        let partial=match (projection.input_bf16!=0,projection.round!=0) {
            (false,false)=>chain!(false,false,(projection.input as *const f32).add(lane as usize*8).cast::<u8>()),
            (false,true)=>chain!(false,true,(projection.input as *const f32).add(lane as usize*8).cast::<u8>()),
            (true,false)=>chain!(true,false,(projection.input as *const u16).add(lane as usize*8).cast::<u8>()),
            (true,true)=>chain!(true,true,(projection.input as *const u16).add(lane as usize*8).cast::<u8>()),
        };
        let mut sum=partial;
        let mut distance=16u32;
        while distance!=0 {
            let neighbor:f32;
            asm!("shfl.sync.down.b32 {neighbor}, {sum}, {distance}, 31, -1;",neighbor=out(reg32)neighbor,sum=in(reg32)sum,distance=in(reg32)distance,options(nomem,nostack));
            sum+=neighbor;distance/=2;
        }
        if lane==0 && projection.tokens>0 {
            *(projection.output as *mut f32).add(row)=if projection.round!=0 {crate::flash_gdn::rounded(sum)} else {sum};
        }
    }
}
