//! MXFP8 prompt rows in shared-memory tiles. Same arithmetic and order as `flash_next_mxfp8_bits` /
//! `_pair` (`next_decode.rs` `row`): the output for (weight row, token) is a fixed tree over the 32 lanes
//! of a warp. Lane L owns columns `tile * 256 + L * 8 + J`, J = 0..7, of every 256-column tile and keeps
//! ONE chain `acc = fma(weight, activation, acc)` over its columns in tile order and J order, where
//! `weight = e4m3(code) * ue8m0(scale)`; the lane partials then combine with the `shfl_down` tree
//! d = 16, 8, 4, 2, 1 and lane zero stores the sum. All 32 partials of an output stay separate here.
//!
//! A block of 12 warps takes 48 weight rows and 16 tokens. Each warp owns 4 rows, so one activation read
//! feeds 4 rows and one decoded weight feeds 16 tokens (64 accumulators a lane). Twelve warps share each
//! activation tile, so the activations come from L2 a third as often as with 4-warp blocks (the old shape was
//! L2 bound); the stamp caps the kernel at 160 registers (12 x 32 x 160 fits one block per SM). The activations of the
//! 16 tokens pass through shared memory one 256-column tile at a time (16-byte asynchronous copies,
//! swizzled so a lane's two 16-byte reads are conflict free, double buffered).
//!
//! Launch: entry `mx_rows_tile`, the same `MxProjection` as the kernels above (one projection), grid
//! (ceil(rows / 48), ceil(tokens / 16)), 384 threads, no dynamic shared memory. Needs round 0, FP32
//! activations, cols % 32 == 0, weights and input 16-byte aligned, output 4-byte aligned; else keep the
//! old kernels. Proven for 1 to 256 tokens in one launch (the old path launched groups of 16).
use crate::next_params::MxProjection;

const WARPS:usize=12;const TILE:usize=256;const TOKENS:usize=16;const STAGE:usize=TOKENS*TILE*4;const SHARED:usize=2*STAGE;

#[no_mangle]
pub extern "ptx-kernel" fn mx_rows_tile(p:MxProjection) {
    // SAFETY: The host checks the descriptor as MxLayer::push does (complete E4M3 rows and UE8M0 groups
    // in the retained arena, input rows for all tokens, a disjoint output of tokens x rows words) and
    // keeps every owner alive through stream completion. Weights, scales and input are immutable; each
    // (token, row) output word has exactly one writer (lane zero of the warp that owns the row). The
    // kernel traps unless round and input_bf16 are zero, cols is a multiple of 32, weights and input are
    // 16-byte aligned and the block has 384 threads. Rows past `rows` are never read or written; tokens
    // past `tokens` are never copied or written. Weight reads stay inside row * cols + column below cols
    // and scale reads inside row * cols / 32 + column / 32. Shared reads and writes stay inside the two
    // static stages. All 384 threads that pass the uniform early return reach every barrier; loop bounds depend only on the descriptor.
    unsafe {
        use core::arch::asm;
        macro_rules! rep4 {($v:ident,$body:block)=>{{{let $v:usize=0;$body}{let $v:usize=1;$body}{let $v:usize=2;$body}{let $v:usize=3;$body}}};}
        macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const OFF:usize=4;rep4!(w,{let $v:usize=OFF+w;$body});}}};}
        macro_rules! rep16 {($v:ident,$body:block)=>{{rep4!(hi,{rep4!(lo,{let $v:usize=hi*4+lo;$body});});}};}
        macro_rules! fma {($acc:expr,$a:expr,$b:expr)=>{{asm!("fma.rn.f32 {c}, {a}, {b}, {c};",c=inout(reg32)$acc,a=in(reg32)$a,b=in(reg32)$b,options(nomem,nostack));}};}
        let tid=core::arch::nvptx::_thread_idx_x() as usize;
        let (n,k,m)=(p.rows as usize,p.cols as usize,p.tokens as usize);
        if core::arch::nvptx::_block_dim_x()!=(WARPS*32) as u32 || p.round!=0 || p.input_bf16!=0 || k%32!=0 || (p.weights|p.input)&15!=0 {core::arch::asm!("trap;");}
        let (row_tile,token_tile)=(core::arch::nvptx::_block_idx_x() as usize,core::arch::nvptx::_block_idx_y() as usize);
        if row_tile*WARPS*4>=n || token_tile*TOKENS>=m {return;}
        let (warp,lane)=(tid/32,tid%32);
        let smem=crate::ptx::shared::<SHARED>();
        let tiles=k.div_ceil(TILE);
        let input=p.input as *const f32;
        let issue=|c:usize| {
            let stage=smem+((c&1)*STAGE) as u32;
            let mut item=tid;
            while item<TOKENS*64 {
                let (t,q)=(item/64,item%64);let token=token_tile*TOKENS+t;let column=c*TILE+q*4;
                if token<m && column<k {
                    // The chunk q holds columns q*4 .. q*4+3: lane q/2, first or second half of the lane's eight.
                    crate::ptx::copy16(stage+(t*1024+(q%2)*512+(q/2)*16) as u32,input.add(token*k+column).cast());
                }
                item+=WARPS*32;
            }
        };
        let mut acc=[[0.0f32;16];4];
        let first=row_tile*WARPS*4+warp*4;let rows=[first,first+1,first+2,first+3];
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
                    // Two E4M3 bytes at a time: the hardware conversion to FP16 is exact for every code (subnormals and
                    // both NaN codes included), FP16 to FP32 is exact, then one `mul.rn` by the scale as before.
                    rep4!(h,{
                        let pair=((words[i][h/2]>>((h&1)*16))&0xffff) as u16;
                        let (low,high):(f32,f32);
                        asm!("{{ .reg .b32 d; .reg .b16 l, u; .reg .f32 x, y; cvt.rn.f16x2.e4m3x2 d, {pair}; mov.b32 {{l,u}}, d; cvt.f32.f16 x, l; cvt.f32.f16 y, u; mul.rn.f32 {low}, x, {scale}; mul.rn.f32 {high}, y, {scale}; }}",
                            pair=in(reg16)pair,scale=in(reg32)scale,low=out(reg32)low,high=out(reg32)high,options(nomem,nostack));
                        ww[i][2*h]=low;ww[i][2*h+1]=high;
                    });
                });
                let lane_base=smem+((c&1)*STAGE+lane*16) as u32;
                rep16!(t,{
                    let first=crate::ptx::load4(lane_base+(t*1024) as u32);let second=crate::ptx::load4(lane_base+(t*1024+512) as u32);
                    let xx=[f32::from_bits(first[0]),f32::from_bits(first[1]),f32::from_bits(first[2]),f32::from_bits(first[3]),
                        f32::from_bits(second[0]),f32::from_bits(second[1]),f32::from_bits(second[2]),f32::from_bits(second[3])];
                    rep4!(i,{rep8!(j,{fma!(acc[i][t],ww[i][j],xx[j]);});});
                });
            }
            crate::ptx::sync();
            words=next_words;codes=next_codes;
            c+=1;
        }
        rep4!(i,{
            rep16!(t,{
                let mut v=acc[i][t];
                let mut d=16u32;
                while d!=0 {
                    let x:f32;
                    asm!("shfl.sync.down.b32 {x}, {v}, {d}, 31, -1;",x=out(reg32)x,v=in(reg32)v,d=in(reg32)d,options(nomem,nostack));
                    v+=x;d/=2;
                }
                let token=token_tile*TOKENS+t;
                if lane==0 && rows[i]<n && token<m {*(p.output as *mut f32).add(token*n+rows[i])=v;}
            });
        });
    }
}
