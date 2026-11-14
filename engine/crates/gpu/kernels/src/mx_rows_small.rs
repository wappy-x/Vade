//! MXFP8 rows for a SMALL token count (a draft verify pass, a batch of chats): 2 to 8 tokens of one projection.
//! Same arithmetic and order as `mx_rows_tile`, `flash_next_mxfp8_bits` / `_pair` / `_m1` (`next_decode.rs` `row`): the output for
//! (weight row, token) is a fixed tree over the 32 lanes of a warp. Lane L owns columns `tile * 256 + L * 8 + J`, J = 0..7, of every
//! 256-column tile and keeps ONE chain `acc = fma(weight, activation, acc)` over its columns in tile order and J order, where
//! `weight = e4m3(code) * ue8m0(scale)`; the lane partials then combine with the `shfl_down` tree d = 16, 8, 4, 2, 1 and lane zero
//! stores the sum.
//!
//! LOAD SCHEDULE (ROWS-MX, 2026-11-12): the one-token decode body's (`decode` below), for 2 to 8 tokens. Before, these entries
//! (and `mx_rows_tile` for 5 to 8 tokens) moved a whole block tile by tile: two block barriers every 256 columns held the block's
//! warps to the slowest of its weight rows, the next tile was asked for one tile ahead with loads that also filled L1, and the
//! activations went through a shared stage. Here a warp runs alone: no barrier, no shared memory; a lane loads only its own
//! 8 bytes of each of its rows and the row's scale byte (non-coherent, no L1 allocation) `DEPTH` tiles ahead of its FMAs, and
//! reads its 8 activations of each token straight from the input (cached loads).
//! MEASURED 2026-11-12. A pass of the model's 145 MXFP8 calls alone on one Spark (kernel-bench --mx-small time, two boxes),
//! replaced kernels against these: 2 tokens 14.4 to 14.5 ms against 11.8; 3 tokens 14.7 to 14.8 against 13.0; 4 tokens 14.2 to
//! 14.5 against 13.3 to 13.5; 5 tokens 14.6 to 14.9 against 13.9 to 14.0; 8 tokens 14.8 to 14.9 against 14.5 (the one-token
//! kernel: 12.6 to 12.9). The served batch step on a pair, with the row stage at 4 rows: 2 chats 76.50 against 75.43 ms,
//! 4 chats 106.60 against 102.97, 8 chats 161.71 against 159.98. The served step gains more at 8 tokens than the kernels alone
//! do; why is not measured.
//!
//! Launch: entries `mx_rows_small2` (T = 2, R = 2), `mx_rows_small4` (T = 4, R = 4), `mx_rows_small8` (T = 8, R = 2) and
//! `mx_rows_small8w` (T = 8, R = 4; the wide matrices), the same `MxProjection` as the kernels above (one projection), grid
//! (ceil(rows / (4 * R)), ceil(tokens / T)), 128 threads, no shared memory. Same admission as `mx_rows_tile` (round 0, FP32
//! activations, cols % 32 == 0, weights and input 16-byte aligned, output 4-byte aligned). `wave` is also the MXFP8 body of the
//! merged row launch (`next_decode.rs` `resident_projection_bf16_rows`, one weight row a warp).
use crate::next_params::MxProjection;

const TILE:usize=256;

#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small2(p:MxProjection) {rows::<2,2,2>(p)}
#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small4(p:MxProjection) {rows::<4,2,4>(p)}
#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small8(p:MxProjection) {rows::<8,2,2>(p)}
#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_small8w(p:MxProjection) {rows::<8,2,4>(p)}

/// One block: 4 warps of `R` weight rows each, `T` tokens.
fn rows<const T:usize,const DEPTH:usize,const R:usize>(p:MxProjection) {
    // SAFETY: Only thread and block indices are read here. The block-size trap holds the grid contract; a warp whose first row is
    // past `rows`, or whose token tile is past `tokens`, returns as a whole before `wave` reads anything.
    let (first,token)=unsafe {
        if core::arch::nvptx::_block_dim_x()!=128 {core::arch::asm!("trap;");}
        (core::arch::nvptx::_block_idx_x() as usize*(4*R)+core::arch::nvptx::_thread_idx_x() as usize/32*R,core::arch::nvptx::_block_idx_y() as usize*T)
    };
    if first>=p.rows as usize || token>=p.tokens as usize {return;}
    wave::<T,DEPTH,R>(p,first,token);
}

/// The `R` weight rows `first .. first + R` (R = 1, 2 or 4) of the calling warp for the tokens `token0 .. token0 + T` (T at most 8):
/// every lane of the warp calls it with the same arguments, `first < rows` and `token0 < tokens`. Lane zero stores the sums
/// `output[token * rows + row]`, not rounded. `DEPTH` tiles of a lane's own weight bytes are in flight ahead of its FMAs.
#[inline(always)]
pub(crate) fn wave<const T:usize,const DEPTH:usize,const R:usize>(p:MxProjection,first:usize,token0:usize) {
    // SAFETY: The host checks the descriptor as MxLayer::push does (complete E4M3 rows and UE8M0 groups in the retained arena,
    // input rows for all tokens, a disjoint output of tokens x rows words) and keeps every owner alive through stream completion.
    // Weights, scales and input are immutable (the non-coherent loads need that); each (token, row) output word has exactly one
    // writer (lane zero of the warp that owns the row). The body traps unless round and input_bf16 are zero, cols is a multiple
    // of 32 and weights and input are 16-byte aligned. Rows past `rows` are never read or written; tokens past `tokens` are never
    // read or written. Weight reads stay at row * cols + column with column + 8 <= cols (cols % 32 == 0, a lane's 8 bytes never
    // cross a row end), scale reads at row * cols / 32 + column / 32, input reads at token * cols + column .. column + 8. Tiles at
    // or past a lane's last are never loaded. The three lane bases are formed with wrapping arithmetic, because a lane with no
    // column (lane * 8 >= cols; a 32-column matrix has 28 of them) would point past the buffers: such a lane has `lane_tiles` zero
    // and never offsets or loads through its bases. A lane with a column has its bases inside the buffers (lane * 8 < cols,
    // lane / 4 < cols / 32, lane * 32 < cols * 4), so the guarded `add`s below stay inside one allocation. No shared memory and
    // no barrier; all 32 lanes of the warp reach the shuffle (the caller's contract: one call a warp with equal arguments, and
    // the loop bounds depend only on the descriptor).
    unsafe {
        use core::arch::asm;
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const OFF:usize=4;rep4!(w,{let $v:usize=OFF+w;$body});}}};}
        macro_rules! fma {($acc:expr,$a:expr,$b:expr)=>{{asm!("fma.rn.f32 {c}, {a}, {b}, {c};",c=inout(reg32)$acc,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));}};}
        let lane=core::arch::nvptx::_thread_idx_x() as usize%32;
        let (n,k,m)=(p.rows as usize,p.cols as usize,p.tokens as usize);
        if p.round!=0 || p.input_bf16!=0 || k%32!=0 || (p.weights|p.input)&15!=0 || T>8 || R>4 {core::arch::asm!("trap;");}
        let rows=[first,first+1,first+2,first+3];
        let tiles=k.div_ceil(TILE);
        let lane_tiles=if lane*8<k {(k-lane*8).div_ceil(TILE)} else {0};
        let weights=(p.weights as *const u8).wrapping_add(lane*8);
        let scales=(p.scales as *const u8).wrapping_add(lane/4);
        let input=(p.input as *const u8).wrapping_add(lane*32);
        let load=|tile:usize|->([[u32;2];4],[u32;4]) {
            let mut words=[[0u32;2];4];let mut codes=[0u32;4];
            if tile<lane_tiles {
                rep4!(i,{
                    if i<R && rows[i]<n {
                        let row=rows[i];
                        let (low,high,code):(u32,u32,u32);
                        asm!("ld.global.nc.L1::no_allocate.v2.u32 {{{low},{high}}}, [{pointer}];",low=out(reg32)low,high=out(reg32)high,pointer=in(reg64)weights.add(row*k+tile*TILE),options(nostack,readonly));
                        asm!("ld.global.nc.L1::no_allocate.u8 {code}, [{pointer}];",code=out(reg32)code,pointer=in(reg64)scales.add(row*(k/32)+tile*8),options(nostack,readonly));
                        words[i]=[low,high];codes[i]=code;
                    }
                });
            }
            (words,codes)
        };
        let mut queue=[([[0u32;2];4],[0u32;4]);DEPTH];
        let mut slot=0;
        while slot<DEPTH {queue[slot]=load(slot);slot+=1;}
        let mut acc=[[0.0f32;8];4];
        let mut c=0;
        while c<tiles {
            let (words,codes)=queue[0];
            let mut slot=0;
            while slot+1<DEPTH {queue[slot]=queue[slot+1];slot+=1;}
            queue[DEPTH-1]=load(c+DEPTH);
            if c<lane_tiles {
                let mut ww=[[0.0f32;8];4];
                rep4!(i,{
                    if i<R {
                        let code=codes[i];
                        let scale=f32::from_bits(if code==0 {0x00400000} else if code==255 {0x7fc00000} else {code<<23});
                        // Two E4M3 bytes at a time: the hardware conversion to FP16 is exact for every code (subnormals and
                        // both NaN codes included), FP16 to FP32 is exact, then one `mul.rn` by the scale (as `mx_rows_tile`).
                        rep4!(h,{
                            let pair=((words[i][h/2]>>((h&1)*16))&0xffff) as u16;
                            let (low,high):(f32,f32);
                            asm!("{{ .reg .b32 d; .reg .b16 l, u; .reg .f32 x, y; cvt.rn.f16x2.e4m3x2 d, {pair}; mov.b32 {{l,u}}, d; cvt.f32.f16 x, l; cvt.f32.f16 y, u; mul.rn.f32 {low}, x, {scale}; mul.rn.f32 {high}, y, {scale}; }}",
                                pair=in(reg16)pair,scale=in(reg32)scale,low=out(reg32)low,high=out(reg32)high,options(nomem,nostack));
                            ww[i][2*h]=low;ww[i][2*h+1]=high;
                        });
                    }
                });
                rep8!(t,{
                    if t<T && token0+t<m {
                        let pointer=input.add(((token0+t)*k+c*TILE)*4);
                        let (x0,x1,x2,x3,x4,x5,x6,x7):(u32,u32,u32,u32,u32,u32,u32,u32);
                        asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)x0,b=out(reg32)x1,c=out(reg32)x2,d=out(reg32)x3,pointer=in(reg64)pointer,options(nostack,readonly));
                        asm!("ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{pointer}];",a=out(reg32)x4,b=out(reg32)x5,c=out(reg32)x6,d=out(reg32)x7,pointer=in(reg64)pointer.add(16),options(nostack,readonly));
                        let xx=[f32::from_bits(x0),f32::from_bits(x1),f32::from_bits(x2),f32::from_bits(x3),f32::from_bits(x4),f32::from_bits(x5),f32::from_bits(x6),f32::from_bits(x7)];
                        rep4!(i,{if i<R {rep8!(j,{fma!(acc[i][t],ww[i][j],xx[j]);});}});
                    }
                });
            }
            c+=1;
        }
        rep4!(i,{
            if i<R {
                rep8!(t,{
                    if t<T {
                        let mut v=acc[i][t];
                        let mut d=16u32;
                        while d!=0 {
                            let x:f32;
                            asm!("shfl.sync.down.b32 {x}, {v}, {d}, 31, -1;",x=out(reg32)x,v=in(reg32)v,d=in(reg32)d,options(nomem,nostack));
                            v+=x;d/=2;
                        }
                        if lane==0 && rows[i]<n && token0+t<m {*(p.output as *mut f32).add((token0+t)*n+rows[i])=v;}
                    }
                });
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
pub(crate) fn decode<const DEPTH:usize,const MIXED:bool>(layer:*const crate::next_params::MxLayer) {
    // SAFETY: The checked recorder replays the m1 entry's descriptor (MxLayer::push: complete E4M3 rows and UE8M0 groups, inputs and
    // disjoint outputs of rows words, all live through stream completion) with the m1 grid of ends[count] blocks of 128 threads; `layer`
    // is the aligned, initialized payload of the replayed command. The three lane bases (weights, scales, input) are formed with
    // wrapping arithmetic, because a lane with no column (lane * 8 >= cols; a 32-column matrix has 28 of them) would point past the
    // buffers: such a lane has `lane_tiles` zero and never offsets or loads through its bases. A lane with a column has its bases
    // inside the buffers (lane * 8 < cols, lane / 4 < cols / 32), so the guarded `add`s below stay inside one allocation. Weight
    // reads stay at row * cols + column with column < cols (cols % 32 == 0, the 8 bytes of a lane never cross a row end), scale reads at
    // row * cols / 32 + column / 32, input reads at column .. column + 8 of token 0. Tiles at or past the last are never loaded. Each
    // output word has one writer (lane zero of the row's warp). No shared memory and no barrier; the shuffle runs in full warps (a warp
    // returns as a whole when its row is past `rows`). With MIXED, the payload is a ResidentProjectionBatch whose first
    // repr(C) field is this MxLayer. Its checked ends give plain descriptors ceil(rows/128) blocks and MX descriptors
    // ceil(rows/4) blocks. batch_block preserves the recorded plain-first bijection. Plain blocks call their unchanged
    // BF16 body and return before any MX read. Mixed MX outputs keep the extra BF16 round of that body's outer store.
    unsafe {
        use core::arch::asm;
        use core::arch::nvptx::{_block_idx_x,_thread_idx_x};
        macro_rules! eight {($element:ident,$body:block)=>{{{const $element:usize=0;$body}{const $element:usize=1;$body}{const $element:usize=2;$body}{const $element:usize=3;$body}
            {const $element:usize=4;$body}{const $element:usize=5;$body}{const $element:usize=6;$body}{const $element:usize=7;$body}}};}
        let block=_block_idx_x() as u32;
        let count=(*layer).count as usize;
        let (index,local)=if MIXED {
            let batch=&*layer.cast::<crate::next_params::ResidentProjectionBatch>();
            let (index,local)=crate::plain_admission::batch_block((*layer).ends,batch.plain,count,block);
            if batch.plain&(1<<index)!=0 {crate::next_decode::resident_flash_resident_projection_bf16(*batch);return;}
            (index,local)
        } else {
            let mut index=0usize;
            eight!(PROJECTION,{if PROJECTION+1<count && block>=(*layer).ends[PROJECTION+1] {index+=1;}});
            let start=if index==0 {0} else {(*layer).ends[index]};
            (index,block-start)
        };
        let projection=(*layer).projections[index];
        let quads=projection.rows.div_ceil(4);
        let row=(local%quads) as usize*4+_thread_idx_x() as usize/32;
        if row>=projection.rows as usize {return;}
        let lane=_thread_idx_x()&31;
        let columns=projection.cols;
        let lane_tiles=if lane*8<columns {(columns-lane*8).div_ceil(256)} else {0};
        let weight_pointer=(projection.weights as *const u8).wrapping_add(row*columns as usize+lane as usize*8);
        let scale_pointer=(projection.scales as *const u8).wrapping_add(row*(columns/32) as usize+lane as usize/4);
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
            (false,false)=>chain!(false,false,(projection.input as *const f32).wrapping_add(lane as usize*8).cast::<u8>()),
            (false,true)=>chain!(false,true,(projection.input as *const f32).wrapping_add(lane as usize*8).cast::<u8>()),
            (true,false)=>chain!(true,false,(projection.input as *const u16).wrapping_add(lane as usize*8).cast::<u8>()),
            (true,true)=>chain!(true,true,(projection.input as *const u16).wrapping_add(lane as usize*8).cast::<u8>()),
        };
        let mut sum=partial;
        let mut distance=16u32;
        while distance!=0 {
            let neighbor:f32;
            asm!("shfl.sync.down.b32 {neighbor}, {sum}, {distance}, 31, -1;",neighbor=out(reg32)neighbor,sum=in(reg32)sum,distance=in(reg32)distance,options(nomem,nostack));
            sum+=neighbor;distance/=2;
        }
        if lane==0 && projection.tokens>0 {
            let value=if projection.round!=0 {crate::flash_gdn::rounded(sum)} else {sum};
            *(projection.output as *mut f32).add(row)=if MIXED {crate::flash_gdn::rounded(value)} else {value};
        }
    }
}
