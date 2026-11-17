//! The third form of the latent absorption prompt tile: the word of `latent_absorption_tile`, bit for bit (contract:
//! `engine/docs/tc-dense-contract.md`, section "Plain BF16"; word function `gpu/src/absorb_tc.rs`), computed another
//! way. Device proof of the bits and the time: `kernel-bench --latent3`. The host launches it where its kernels take
//! the launch (`gpu/src/latent_absorb_rows.rs`, `third`).
//!
//! Why: in the first form every block stages the B operand again with single-word loads, and a warp asks for its A
//! operand one step before it uses it, so it waits for memory in every step (MEASURED, receipt
//! `tcbf-takeover-time-02`: the four operations run at 2,700 to 3,650 million multiply-adds a second).
//!
//! SCORES (`scores3`). A block is one token. Its B operand (the token's absorbed query, 32 heads x 512 features) is staged
//! ONCE, with 16-byte loads, as packed BF16 pairs. A warp then walks the token's selected rows in M tiles of 16 rows
//! (warp `w` takes tiles `w`, `w + warps`, ...). A range of the word is one stage of 8 steps: every address inside a
//! stage is a register plus a constant, the A operand is fetched one step ahead, and the lines of a later stage are
//! asked for ahead of time in a bench form (`PF`).
//! QUERY and PROJECT (`weights3`): a block is one head and 32 tokens, the same way with the weight rows as the A operand.
//! VALUES (`values3`): the two operands of the step change sides, see there.
use crate::{absorb_tc::{Form,Kind},canonical::{Acc,ZERO,add,madd}};
use core::arch::{asm,nvptx::{_block_dim_x,_block_idx_x,_thread_idx_x}};
/// Shared bytes of a block: 32 steps x 32 lanes x 4 N tiles x two packed registers.
const STAGE:usize=32*32*4*8;
/// Constant indices only, so every array element below is a register.
macro_rules! rep4 {($v:ident,$body:block)=>{{{const $v:usize=0;$body}{const $v:usize=1;$body}{const $v:usize=2;$body}{const $v:usize=3;$body}}};}
macro_rules! rep8 {($v:ident,$body:block)=>{{rep4!($v,$body);{const $v:usize=4;$body}{const $v:usize=5;$body}{const $v:usize=6;$body}{const $v:usize=7;$body}}};}
macro_rules! rep16 {($v:ident,$body:block)=>{{rep8!($v,$body);{const $v:usize=8;$body}{const $v:usize=9;$body}{const $v:usize=10;$body}{const $v:usize=11;$body}{const $v:usize=12;$body}{const $v:usize=13;$body}{const $v:usize=14;$body}{const $v:usize=15;$body}}};}
/// The high halves of two FP32 words as one packed BF16 pair, the first word low (the operand of the contract: the
/// high 16 bits of the word, no rounding).
macro_rules! high2 {($x:expr,$y:expr)=>{{
    let d:u32;
    asm!("prmt.b32 {d}, {x}, {y}, 0x7632;",d=out(reg32)d,x=in(reg32)$x,y=in(reg32)$y,options(pure,nomem,nostack));
    d
}};}
/// Two adjacent FP32 words at a register plus a constant, as one packed BF16 pair.
macro_rules! pair {($p:expr,$o:expr)=>{{
    let (x,y):(u32,u32);
    asm!("ld.global.nc.v2.u32 {{{x},{y}}}, [{p}+{o}];",x=out(reg32)x,y=out(reg32)y,p=in(reg64)$p,o=const $o,options(nostack,readonly));
    high2!(x,y)
}};}
/// The four A registers of step `$s` of the stage at the two row pointers (each already at the lane's first word).
macro_rules! fetch {($ra:expr,$rb:expr,$s:expr)=>{[pair!($ra,64*$s),pair!($rb,64*$s),pair!($ra,64*$s+32),pair!($rb,64*$s+32)]};}
/// The staged B registers of two N tiles.
macro_rules! quad {($p:expr,$o:expr)=>{{
    let (a,b,c,d):(u32,u32,u32,u32);
    asm!("ld.shared.v4.u32 {{{a},{b},{c},{d}}}, [{p}+{o}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg32)$p,o=const $o,options(nostack,readonly));
    [a,b,c,d]
}};}
macro_rules! prefetch {($pointer:expr)=>{asm!("prefetch.global.L2 [{pointer}];",pointer=in(reg64)$pointer,options(nostack,readonly));};}
macro_rules! mul {($a:expr,$b:expr)=>{{let value:f32;asm!("mul.rn.f32 {value}, {a}, {b};",value=out(reg32)value,a=in(reg32)$a,b=in(reg32)$b,options(pure,nomem,nostack));value}};}

/// Scores. `PF`: 0 no hints, 1 the next stage of the warp's rows (and the first stage of its next tile).
#[inline(always)]
fn scores3<const PF:usize,const PANEL:bool,const PACKED:bool,const PARALLEL:bool>(p:crate::prompt_ops::Arguments) {
    // SAFETY: the backend (`latent_absorb_rows::third`) and the bench validate every extent (`lengths`) and launch whole
    // warps; a launch this body does not take traps here as a whole block: another operation, another feature count
    // than 512, another head count than 32, no selection table, or an input that is not 16-byte aligned. A block
    // past the token count returns as a whole block before any read. The count word is a finite integer at most the
    // score capacity and a selected ID is a finite integer below the row bound, or the block traps before the read
    // that would use it. Global reads: the token's 32 x 512 B words; the token's selection row; 8-byte pairs inside a
    // selected cache row (a lane whose slot is past the count reads the row of the tile's first slot and drops the
    // value). A hint names an address inside a selected row or up to 64 bytes past it and reads nothing. Inputs are
    // immutable for the launch (the non-coherent loads need that). PACKED is bench-only: the cache has two high
    // halves per word, rows * 256 words, checked here before access and supplied by the bench's guarded allocation.
    // The 32,768 shared bytes belong to this block:
    // every staged word is written once before the one barrier, by the thread of its item, and read only after it.
    // All threads reach the barrier. Each (slot, head) output word of the token has one writer; blocks own disjoint
    // tokens. The output is private until the stream completes.
    unsafe {
        let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
        let threads=_block_dim_x() as usize;let warps=threads/32;
        if f.kind!=Kind::Scores || f.k!=512 || f.h!=32 || f.table==0 || warps==0 || threads%32!=0 || (p.input[0]|p.input[1])&15!=0 {asm!("trap;");}
        if PARALLEL && threads!=256 {asm!("trap;");}
        if PACKED && p.lengths[1]<(f.rows*256) as u64 {asm!("trap;");}
        let token=_block_idx_x() as usize;
        if token>=f.tokens {return;}
        let tid=_thread_idx_x() as usize;let (warp,lane)=(tid/32,tid%32);let (g,t)=(lane/4,lane%4);
        let (latent,out)=(p.input[1] as *const u8,p.output as *mut f32);
        let stage=crate::ptx::shared::<STAGE>();
        let partials=if PARALLEL {crate::ptx::shared::<16384>()} else {0};
        // Stage the token's B operand once. Item `i`: head `i / 128`, then 16 bytes: step `s`, quarter `q`. Its two
        // pairs go to lanes `(n % 8) * 4 + (2 q) % 4` and the next one, register `q / 2`, N tile `n / 8`.
        let b=(p.input[0] as *const u32).add(token*32*512);
        let mut item=tid;
        while item<4096 {
            let mut raw=[[0u32;4];4];
            crate::strip_warp::unroll!(4,j,{
                let i=item+j*threads;
                if i<4096 {
                    let (w0,w1,w2,w3):(u32,u32,u32,u32);
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,p=in(reg64)b.add(i*4),options(nostack,readonly));
                    raw[j]=[w0,w1,w2,w3];
                }
            });
            crate::strip_warp::unroll!(4,j,{
                let i=item+j*threads;
                if i<4096 {
                    let (n,s,q)=(i/128,i%128/4,i%4);
                    let at=stage+((((s*32+(n%8)*4+(2*q)%4)*4+n/8)*8+4*(q/2)) as u32);
                    let (first,second)=(high2!(raw[j][0],raw[j][1]),high2!(raw[j][2],raw[j][3]));
                    asm!("st.shared.u32 [{p}], {v};",p=in(reg32)at,v=in(reg32)first,options(nostack));
                    asm!("st.shared.u32 [{p}+32], {v};",p=in(reg32)at,v=in(reg32)second,options(nostack));
                }
            });
            item+=4*threads;
        }
        crate::ptx::sync();
        // The token's selection row: the IDs, the count last.
        let selected=(p.input[2] as *const f32).add(token*f.table);
        let raw=*selected.add(f.table-1);
        if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
        let count=raw as usize;
        // The lane's first word of the cache row of a live slot.
        let row=|slot:usize|->*const u8 {
            let raw=*selected.add(slot);
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>=f.rows {asm!("trap;");}
            latent.add(raw as usize*if PACKED {1024} else {2048}+t*if PACKED {4} else {8})
        };
        // The two row pointers of tile `mt` (a live tile): a slot past the count reads the tile's first row.
        let rows_of=|mt:usize|->(*const u8,*const u8) {
            let (sa,sb)=(mt*16+g,mt*16+g+8);
            (row(if sa<count {sa} else {mt*16}),row(if sb<count {sb} else {mt*16}))
        };
        let scale=1.0/crate::causal_ops::root(f.d as f32);
        macro_rules! fetch_a {($ra:expr,$rb:expr,$s:expr)=>{{
            if PACKED {
                let (a,b,c,d):(u32,u32,u32,u32);
                asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)a,p=in(reg64)$ra,o=const 32*$s,options(nostack,readonly));
                asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)b,p=in(reg64)$rb,o=const 32*$s,options(nostack,readonly));
                asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)c,p=in(reg64)$ra,o=const 32*$s+16,options(nostack,readonly));
                asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)d,p=in(reg64)$rb,o=const 32*$s+16,options(nostack,readonly));
                [a,b,c,d]
            } else {fetch!($ra,$rb,$s)}
        }};}
        if PARALLEL {
            // Two 16-row tiles advance together. The four warps of each tile own the four fixed ranges. All eight
            // warps execute the same number of rounds, including a masked last tile, so both barriers are uniform.
            // Each lane writes its own 64-byte partial span. Only range zero reads the four spans of its own lane,
            // joins them in the contract order, and writes that lane's disjoint output words. Shared total49,152.
            let (group,part)=(warp/4,warp%4);let rounds=f.c.div_ceil(16).div_ceil(2);
            let mut r=0usize;
            while r<rounds {
                let mt=r*2+group;let (sa,sb)=(mt*16+g,mt*16+g+8);let (la,lb)=(sa<count,sb<count);
                let mut acc=[ZERO;4];
                if mt*16<count {
                    let (ra,rb)=rows_of(mt);let bytes=if PACKED {256} else {512};
                    let (ra,rb)=(ra.wrapping_add(part*bytes),rb.wrapping_add(part*bytes));
                    let bq=stage+(part*8192+lane*32) as u32;
                    let mut panel=[[0u32;4];8];
                    rep8!(S,{panel[S]=fetch_a!(ra,rb,S);});
                    rep8!(S,{
                        let a=[if la {panel[S][0]} else {0},if lb {panel[S][1]} else {0},if la {panel[S][2]} else {0},if lb {panel[S][3]} else {0}];
                        let (low,high)=(quad!(bq,1024*S),quad!(bq,1024*S+16));
                        acc[0]=madd(a,[low[0],low[1]],acc[0]);acc[1]=madd(a,[low[2],low[3]],acc[1]);
                        acc[2]=madd(a,[high[0],high[1]],acc[2]);acc[3]=madd(a,[high[2],high[3]],acc[3]);
                    });
                }
                let at=partials+(((group*4+part)*32+lane)*64) as u32;
                rep4!(T,{
                    let v=acc[T];
                    asm!("st.shared.v4.f32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg32)at+(T*16) as u32,a=in(reg32)v.a,b=in(reg32)v.b,c=in(reg32)v.c,d=in(reg32)v.d,options(nostack));
                });
                crate::ptx::sync();
                if part==0 {
                    rep4!(T,{
                        macro_rules! partial {($p:expr)=>{{let q=quad!(at,2048*$p+16*T);Acc {a:f32::from_bits(q[0]),b:f32::from_bits(q[1]),c:f32::from_bits(q[2]),d:f32::from_bits(q[3])}}};}
                        let sum=add(add(acc[T],partial!(1)),add(partial!(2),partial!(3)));
                        let w=out.wrapping_add((token*32+T*8+2*t)*f.c+sa);
                        if sa<f.c {w.write(if la {mul!(sum.a,scale)} else {0.0});w.wrapping_add(f.c).write(if la {mul!(sum.b,scale)} else {0.0});}
                        if sb<f.c {w.wrapping_add(8).write(if lb {mul!(sum.c,scale)} else {0.0});w.wrapping_add(f.c+8).write(if lb {mul!(sum.d,scale)} else {0.0});}
                    });
                }
                crate::ptx::sync();r+=1;
            }
            return;
        }
        let tiles=f.c.div_ceil(16);
        let lead=120*t;
        let mut mt=warp;
        let (mut ra,mut rb)=if mt*16<count {rows_of(mt)} else {(latent,latent)};
        while mt<tiles {
            let (sa,sb)=(mt*16+g,mt*16+g+8);let (la,lb)=(sa<count,sb<count);
            let mut acc=[ZERO;4];
            let next=mt+warps;let more=next*16<count;
            let (na,nb)=if more {rows_of(next)} else {(ra,rb)};
            if mt*16<count {
                let (mut p0,mut p2)=([ZERO;4],[ZERO;4]);
                let mut bq=stage+(lane*32) as u32;
                let mut part=0usize;
                while part<4 {
                    if PF==1 {
                        if part<3 {prefetch!(ra.wrapping_add(512+lead));prefetch!(ra.wrapping_add(576+lead));prefetch!(rb.wrapping_add(512+lead));prefetch!(rb.wrapping_add(576+lead));}
                        else if more {prefetch!(na.wrapping_add(lead));prefetch!(na.wrapping_add(64+lead));prefetch!(nb.wrapping_add(lead));prefetch!(nb.wrapping_add(64+lead));}
                    }
                    // Ask for a complete range before consuming it. Constant indices keep the panel in registers.
                    let mut panel=[[0u32;4];8];
                    if PANEL {rep8!(S,{panel[S]=fetch_a!(ra,rb,S);});}
                    let mut now=if PANEL {panel[0]} else {fetch_a!(ra,rb,0)};
                    rep8!(S,{
                        let ahead=if S+1<8 {if PANEL {panel[(S+1)%8]} else {fetch_a!(ra,rb,(S+1)%8)}} else {[0u32;4]};
                        let a=[if la {now[0]} else {0},if lb {now[1]} else {0},if la {now[2]} else {0},if lb {now[3]} else {0}];
                        let (low,high)=(quad!(bq,1024*S),quad!(bq,1024*S+16));
                        acc[0]=madd(a,[low[0],low[1]],acc[0]);acc[1]=madd(a,[low[2],low[3]],acc[1]);
                        acc[2]=madd(a,[high[0],high[1]],acc[2]);acc[3]=madd(a,[high[2],high[3]],acc[3]);
                        now=ahead;
                    });
                    ra=ra.wrapping_add(if PACKED {256} else {512});rb=rb.wrapping_add(if PACKED {256} else {512});bq+=8192;
                    // Ranges 0 and 1 join into `p0`, range 2 waits in `p2`, range 3 joins the word.
                    if part==0 {rep4!(T,{p0[T]=acc[T];acc[T]=ZERO;});}
                    else if part==1 {rep4!(T,{p0[T]=add(p0[T],acc[T]);acc[T]=ZERO;});}
                    else if part==2 {rep4!(T,{p2[T]=acc[T];acc[T]=ZERO;});}
                    else {rep4!(T,{acc[T]=add(p0[T],add(p2[T],acc[T]));});}
                    part+=1;
                }
            }
            // The words of the lane: rows `sa` and `sb`, heads `T * 8 + 2 t` and the next one. A slot at or past the
            // count is +0.0.
            rep4!(T,{
                let w=out.wrapping_add((token*32+T*8+2*t)*f.c+sa);
                let sum:Acc=acc[T];
                if sa<f.c {w.write(if la {mul!(sum.a,scale)} else {0.0});w.wrapping_add(f.c).write(if la {mul!(sum.b,scale)} else {0.0});}
                if sb<f.c {w.wrapping_add(8).write(if lb {mul!(sum.c,scale)} else {0.0});w.wrapping_add(f.c+8).write(if lb {mul!(sum.d,scale)} else {0.0});}
            });
            (ra,rb)=(na,nb);
            mt=next;
        }
    }
}
/// Query and project (the two operations with a weight matrix). A block is one head and 32 tokens. Its B operand (the
/// 32 tokens' rows of this head) is staged ONCE with 16-byte loads; a warp then walks the weight rows in M tiles of 16
/// (warp `w` takes tiles `w`, `w + warps`, ...). The weights are the checkpoint's BF16 pairs, read as they lie, one
/// step ahead, at a register plus a constant. A range of the word is one stage: 4 steps for query (256 key
/// features), 8 for project (512 latent features).
#[inline(always)]
fn weights3<const PANEL:bool,const ASYNC:bool,const W64:bool>(p:crate::prompt_ops::Arguments) {
    // SAFETY: as `scores3`: the backend (`third`) and the bench validate every extent with the shape's `lengths` and
    // launch whole warps; a launch this body does not take traps here as a whole block (another operation or
    // shape than query with 256 key features and 512 latent features, or project with 512 latent features; a row
    // count that is not a multiple of 16; another head count than 32; an input or a weight address that is not
    // 16-byte aligned). A block past the last head returns as a whole block before any read. Global reads: the B
    // words of a token below `tokens` (a column past the last token is staged as +0.0 and its words are dropped);
    // 4-byte weight pairs of a row below the row count at a column below the term count; for project the count word
    // of a token below `tokens`. Inputs are immutable for the launch. The shared bytes belong to this block: every
    // staged word is written once before the one barrier and read only after it; all threads reach the barrier.
    // Each (token, head, row) output word has one writer. The output is private until the stream completes.
    // ASYNC is bench-only. Up to 12 whole warps each own two disjoint 512-byte weight tiles in 12,288 additional
    // shared bytes (45,056 total). A lane copies its aligned 16-byte span inside one live 16-row by 16-column tile.
    // The copy completes and every lane reaches the warp barrier before ldmatrix reads that tile. The next copy
    // writes the other tile; a tile is reused only after all lanes have read it. No block barrier is inside the walk.
    // W64 is bench-only and never combined with ASYNC. It stages 64 tokens in a disjoint dynamic shared region at
    // address zero, with no static shared allocation. The kernel checks the dynamic byte count before any access:
    // 64 * terms * 2 bytes. The bench passes that count and the driver checks its limit. Its stage has eight N tiles
    // instead of four; the same item-to-word map is a bijection. Output tokens past the group's end are discarded.
    unsafe {
        let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
        let threads=_block_dim_x() as usize;let warps=threads/32;
        let query=f.kind==Kind::Query;
        let (m_all,terms)=(f.m(),f.terms(0));
        if !(query && f.d==256 && f.k==512 || f.kind==Kind::Project && f.k==512) || m_all%16!=0 || f.h!=32 || warps==0 || threads%32!=0
            || (p.input[0]|p.weight)&15!=0 {asm!("trap;");}
        if ASYNC && warps>12 {asm!("trap;");}
        let ntiles=if W64 {8} else {4};let token_width=ntiles*8;
        if W64 {
            let dynamic:u32;asm!("mov.u32 {v}, %dynamic_smem_size;",v=out(reg32)dynamic,options(nomem,nostack));
            if (dynamic as usize)<token_width*terms*2 {asm!("trap;");}
        }
        let token_tiles=f.tokens.div_ceil(token_width);
        let block=_block_idx_x() as usize;let (head,n0)=(block/token_tiles,block%token_tiles*token_width);
        if head>=f.h {return;}
        let tid=_thread_idx_x() as usize;let (warp,lane)=(tid/32,tid%32);let (g,t)=(lane/4,lane%4);
        let out=p.output as *mut f32;
        // W64 has no static shared allocation: the checked dynamic region starts at shared address zero.
        let stage=if W64 {0} else {crate::ptx::shared::<STAGE>()};
        let a_stage=if ASYNC {crate::ptx::shared::<12288>()} else {0};
        // Stage the block's B operand once. Item `i`: column `i / quads` (token `n0 + column`), then 16 bytes: step
        // `s`, quarter `q`, as in `scores3` with the token in the place of the head.
        let quads=terms/4;
        let b=(p.input[0] as *const u32).add((n0*32+head)*terms);
        let mut item=tid;
        while item<token_width*quads {
            let mut raw=[[0u32;4];4];
            crate::strip_warp::unroll!(4,j,{
                let i=item+j*threads;
                if i<token_width*quads && n0+i/quads<f.tokens {
                    let (w0,w1,w2,w3):(u32,u32,u32,u32);
                    asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,
                        p=in(reg64)b.add(i/quads*32*terms+i%quads*4),options(nostack,readonly));
                    raw[j]=[w0,w1,w2,w3];
                }
            });
            crate::strip_warp::unroll!(4,j,{
                let i=item+j*threads;
                if i<token_width*quads {
                    let (n,s,q)=(i/quads,i%quads/4,i%4);
                    let at=stage+((((s*32+(n%8)*4+(2*q)%4)*ntiles+n/8)*8+4*(q/2)) as u32);
                    let (first,second)=(high2!(raw[j][0],raw[j][1]),high2!(raw[j][2],raw[j][3]));
                    asm!("st.shared.u32 [{p}], {v};",p=in(reg32)at,v=in(reg32)first,options(nostack));
                    asm!("st.shared.u32 [{p}+{next}], {v};",p=in(reg32)at,next=const if W64 {64} else {32},v=in(reg32)second,options(nostack));
                }
            });
            item+=4*threads;
        }
        crate::ptx::sync();
        // The lane's first weight pair of row `m`: the checkpoint's BF16 bits, column `2 t` low.
        let weights=p.weight as *const u8;
        let row=|m:usize|->*const u8 {weights.add((f.weight(head,m,0)+2*t)*2)};
        // The four A registers of step `$s` of the stage at the two row pointers.
        macro_rules! fetch_w {($ra:expr,$rb:expr,$s:expr)=>{{
            let (w0,w1,w2,w3):(u32,u32,u32,u32);
            asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)w0,p=in(reg64)$ra,o=const 32*$s,options(nostack,readonly));
            asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)w1,p=in(reg64)$rb,o=const 32*$s,options(nostack,readonly));
            asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)w2,p=in(reg64)$ra,o=const 32*$s+16,options(nostack,readonly));
            asm!("ld.global.nc.u32 {v}, [{p}+{o}];",v=out(reg32)w3,p=in(reg64)$rb,o=const 32*$s+16,options(nostack,readonly));
            [w0,w1,w2,w3]
        }};}
        let tiles=m_all/16;
        let mut mt=warp;
        while mt<tiles {
            let (ma,mb)=(mt*16+g,mt*16+g+8);
            let (mut ra,mut rb)=(row(ma),row(mb));
            let mut acc=[ZERO;8];let (mut p0,mut p2)=([ZERO;8],[ZERO;8]);
            macro_rules! weight_accumulators {($i:ident,$body:block)=>{rep8!($i,{if $i<ntiles {$body}});};}
            let mut bq=stage+(lane*ntiles*8) as u32;
            let mut part=0usize;
            macro_rules! copy_a {($step:expr)=>{{
                let step=$step;
                let at=a_stage+(warp*1024+(step%2)*512+lane*16) as u32;
                let source=weights.add(f.weight(head,mt*16+lane/2,step*16+(lane%2)*8)*2);
                crate::ptx::copy16_cached(at,source.cast());crate::ptx::commit();
            }};}
            if ASYNC {copy_a!(0usize);}
            while part<4 {
                // One stage of `$n` steps: `$rep` gives the constant step index.
                macro_rules! stage_steps {($rep:ident,$n:expr)=>{{
                    let mut panel=[[0u32;4];8];
                    if PANEL {$rep!(S,{panel[S]=fetch_w!(ra,rb,S);});}
                    let mut now=if ASYNC {[0u32;4]} else if PANEL {panel[0]} else {fetch_w!(ra,rb,0)};
                    $rep!(S,{
                        let ahead=if !ASYNC && S+1<$n {if PANEL {panel[(S+1)%$n]} else {fetch_w!(ra,rb,(S+1)%$n)}} else {[0u32;4]};
                        if ASYNC {
                            let step=part*$n+S;
                            crate::ptx::wait();crate::ptx::warp_sync();
                            if step+1<terms/16 {copy_a!(step+1);}
                            let at=a_stage+(warp*1024+(step%2)*512+(lane%16)*32+(lane/16)*16) as u32;
                            let (a,b,c,d):(u32,u32,u32,u32);
                            asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)a,b=out(reg32)b,c=out(reg32)c,d=out(reg32)d,p=in(reg32)at,options(nostack,readonly));
                            now=[a,b,c,d];
                        }
                        let mut b=[[0u32;4];4];
                        rep4!(J,{if J<(if W64 {4} else {2}) {b[J]=quad!(bq,(if W64 {2048} else {1024})*S+16*J);}});
                        weight_accumulators!(T,{acc[T]=madd(now,[b[T/2][2*(T%2)],b[T/2][2*(T%2)+1]],acc[T]);});
                        now=ahead;
                    });
                    ra=ra.wrapping_add(32*$n);rb=rb.wrapping_add(32*$n);bq+=(32*ntiles*8*$n) as u32;
                }};}
                if query {stage_steps!(rep4,4);} else {stage_steps!(rep8,8);}
                if part==0 {weight_accumulators!(T,{p0[T]=acc[T];acc[T]=ZERO;});}
                else if part==1 {weight_accumulators!(T,{p0[T]=add(p0[T],acc[T]);acc[T]=ZERO;});}
                else if part==2 {weight_accumulators!(T,{p2[T]=acc[T];acc[T]=ZERO;});}
                else {weight_accumulators!(T,{acc[T]=add(p0[T],add(p2[T],acc[T]));});}
                part+=1;
            }
            // The words of the lane: rows `ma` and `mb`, tokens `n0 + T * 8 + 2 t` and the next one. Project: a token
            // with no selected row gives +0.0.
            let round=crate::latent_absorb::round;
            weight_accumulators!(T,{
                let n=n0+T*8+2*t;let sum:Acc=acc[T];
                crate::strip_warp::unroll!(2,e,{
                    if n+e<f.tokens {
                        let empty=!query && f.table!=0 && {
                            let raw=*(p.input[1] as *const f32).add(f.count_word(n+e));
                            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
                            raw==0.0
                        };
                        let w=out.add(((n+e)*32+head)*m_all);
                        w.add(ma).write(if empty {0.0} else {round(if e==0 {sum.a} else {sum.b})});
                        w.add(mb).write(if empty {0.0} else {round(if e==0 {sum.c} else {sum.d})});
                    }
                });
            });
            mt+=warps;
        }
    }
}
#[no_mangle]
pub extern "ptx-kernel" fn latent_weights3(p:crate::prompt_ops::Arguments) {weights3::<false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_weights3_panel(p:crate::prompt_ops::Arguments) {weights3::<true,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_weights3_async(p:crate::prompt_ops::Arguments) {weights3::<false,true,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_weights3_tile64(p:crate::prompt_ops::Arguments) {weights3::<false,false,true>(p);}
/// Shared bytes of a values block: the A registers of 32 steps x 2 head tiles x 32 lanes, then the row offsets of the
/// chunk's 512 slots.
const VSTAGE:usize=32*2*32*16;const VIDS:usize=512*4;
/// A slot of a staged chunk past the token's count.
const NO_ROW:u32=u32::MAX;
/// Values, with the two operands of the tensor-core step in each other's place: the M side is the HEADS (the
/// probabilities, `[head][slot]`, staged a chunk of 32 steps at a time as the first form does) and the N side is the
/// FEATURES of the selected cache rows. The products of a step and their order are those of the word function
/// (`A[m][k] * B[k][n]` in ascending `k`); only the side of the instruction that holds each factor changes. Column
/// `c` of N tile `T` of a warp is feature `first + 4 c + T`, so a lane reads its four tiles of a cache row with ONE
/// 16-byte load, and the cache is never staged. A block is one token and 128 features: warp `w` owns head tile
/// `w % 2` and the 32 features from `32 (w / 2)`. `PF`: steps ahead for which the lines of the rows are asked (0: no
/// hints).
#[inline(always)]
fn values3<const PF:usize,const PANEL:bool,const PACKED:bool,const WIDE:bool,const PARK:bool,const FULL:bool>(p:crate::prompt_ops::Arguments) {
    // SAFETY: as `scores3`: the backend (`third`) and the bench validate every extent with the shape's `lengths`; a
    // launch this body does not take traps here as a whole block (another operation, feature count, head count or
    // block size, no selection table, a cache or an output that is not 16-byte aligned). A block past the last token
    // returns as a whole block before any read. The count word is a finite integer at most the score capacity and a
    // selected ID is a finite integer below the row bound, or the block traps before the read that would use it. A
    // row's byte offset is staged as a u32: the host declines a cache of 2,097,151 rows or more (`third`, as `launch`
    // does for the first form). Global reads: probability words of the token below its count; its selection row;
    // 16-byte pieces inside a selected cache row (a slot past the count reads row 0, which exists, and is masked to
    // +0.0). A hint names an address inside a cache row and reads nothing. Inputs are immutable for the launch. The
    // PACKED is bench-only: cache rows contain 256 packed high-half words rather than 512 FP32 words; that extent
    // is checked here and supplied by the bench's guarded allocation. The 34,816 shared bytes belong to this block:
    // a thread stores its own staged words before the barrier that the
    // reads follow, and a chunk is staged again only after the barrier that ends its steps. All 256 threads reach
    // every barrier: the chunk count comes from the one token's count. Each (head, feature) output word of the token
    // has one writer; blocks own disjoint (token, feature) sets. The output is private until the stream completes.
    unsafe {
        let Some(f)=Form::rows(p.shape) else {asm!("trap;");return;};
        if f.kind!=Kind::Values || f.k!=512 || f.h!=32 || f.table==0 || _block_dim_x()!=256 || (p.input[1]|p.output)&15!=0 {asm!("trap;");}
        if PACKED && p.lengths[1]<(f.rows*256) as u64 {asm!("trap;");}
        let block=_block_idx_x() as usize;let pieces=if FULL {1} else if WIDE {2} else {4};let (token,quarter)=(block/pieces,block%pieces);
        if token>=f.tokens {return;}
        let tid=_thread_idx_x() as usize;let (warp,lane)=(tid/32,tid%32);let (g,t)=(lane/4,lane%4);
        let features=if FULL {8} else {4};
        let (mtile,first_feature)=if FULL {(0,warp*64)} else if WIDE {(0,quarter*256+warp*32)} else {(warp&1,quarter*128+(warp>>1)*32)};
        let x=(p.input[0] as *const u32).add(token*32*f.c);
        // The lane's four words (N tiles 0 to 3 of column `g`) of cache row 0.
        let latent=(p.input[1] as *const u8).wrapping_add((first_feature+features*g)*if PACKED {2} else {4});
        let selected=(p.input[2] as *const f32).add(token*f.table);
        let raw=*selected.add(f.table-1);
        if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>f.c {asm!("trap;");}
        let count=raw as usize;let steps=count.div_ceil(16);
        let stage=crate::ptx::shared::<{VSTAGE+VIDS}>();let ids=stage+VSTAGE as u32;
        // The byte offset of the cache row of a slot, or NO_ROW past the count.
        let offset=|slot:usize|->u32 {
            if slot>=count {return NO_ROW;}
            let raw=*selected.add(slot);
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>=f.rows {asm!("trap;");}
            (raw as usize*if PACKED {1024} else {2048}) as u32
        };
        // The B registers of chunk step `cs` for the lane's four N tiles: rows `2 t`, `2 t + 1` (register 0) and
        // `2 t + 8`, `2 t + 9` (register 1) of the step, the lower row in the low half.
        macro_rules! load_b {($cs:expr)=>{{
            let low=crate::ptx::load2(ids+(($cs*16+2*t)*4) as u32);let high=crate::ptx::load2(ids+(($cs*16+2*t+8)*4) as u32);
            let mut words=[[0u32;4];4];
            crate::strip_warp::unroll!(4,r,{
                let at=if r<2 {low[r]} else {high[r-2]};
                if PACKED {
                    if FULL {
                        let (w0,w1,w2,w3):(u32,u32,u32,u32);
                        asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,
                            p=in(reg64)latent.wrapping_add(if at==NO_ROW {0} else {at as usize}),options(nostack,readonly));
                        words[r]=[w0,w1,w2,w3];
                    } else {
                    let (w0,w1):(u32,u32);
                    asm!("ld.global.nc.v2.u32 {{{a},{b}}}, [{p}];",a=out(reg32)w0,b=out(reg32)w1,
                        p=in(reg64)latent.wrapping_add(if at==NO_ROW {0} else {at as usize}),options(nostack,readonly));
                    words[r]=[w0,w1,0,0];
                    }
                } else {
                let (w0,w1,w2,w3):(u32,u32,u32,u32);
                asm!("ld.global.nc.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)w0,b=out(reg32)w1,c=out(reg32)w2,d=out(reg32)w3,
                    p=in(reg64)latent.wrapping_add(if at==NO_ROW {0} else {at as usize}),options(nostack,readonly));
                words[r]=[w0,w1,w2,w3];
                }
            });
            let m0=(if low[0]==NO_ROW {0} else {0xffffu32})|(if low[1]==NO_ROW {0} else {0xffff_0000u32});
            let m1=(if high[0]==NO_ROW {0} else {0xffffu32})|(if high[1]==NO_ROW {0} else {0xffff_0000u32});
            let mut b=[[0u32;2];8];
            rep8!(T,{if T<features {
                macro_rules! packed_pair {($x:expr,$y:expr)=>{{
                    let v:u32;
                    asm!("prmt.b32 {v}, {x}, {y}, {selector};",v=out(reg32)v,x=in(reg32)$x,y=in(reg32)$y,
                        selector=const if T%2==0 {0x5410} else {0x7632},options(pure,nomem,nostack));
                    v
                }};}
                b[T]=if PACKED {[packed_pair!(words[0][T/2],words[1][T/2])&m0,packed_pair!(words[2][T/2],words[3][T/2])&m1]}
                    // The unpacked forms have four feature tiles. FULL is instantiated only with PACKED.
                    else {[high2!(words[0][T%4],words[1][T%4])&m0,high2!(words[2][T%4],words[3][T%4])&m1]};
            }});
            b
        }};}
        let mut acc=[ZERO;16];let (mut p0,mut p2)=([ZERO;16],[ZERO;16]);
        // WIDE holds both 16-head tiles in the warp. Each gathered cache fragment serves both, so the grid needs
        // two feature blocks per token instead of four. Constant indices drop the unused half in the old form.
        macro_rules! accumulators {($i:ident,$body:block)=>{rep16!($i,{if $i<features*(if WIDE {2} else {1}) {$body}});};}
        let (mut part,mut s,mut first)=(0usize,0usize,0usize);
        // PARK uses each word's private output slot for the first joined half. The same thread writes and reads
        // that slot, using coherent loads (never .nc). No other block or lane owns that output word. The final
        // store replaces the temporary FP32 value with the ordinary rounded word before stream publication.
        let word_base=(p.output as *mut f32).add((token*32+mtile*16+g)*512+first_feature+2*features*t);
        macro_rules! save_half {($merge:expr)=>{{
            crate::strip_warp::unroll!(2,H,{
                if H==0 || WIDE {
                    crate::strip_warp::unroll!(2,N,{
                    if N==0 || FULL {crate::strip_warp::unroll!(4,R,{
                        let at=word_base.add(H*16*512+(R/2)*8*512+(R%2)*features+N*4);
                        let mut old=[0.0f32;4];
                        if $merge {asm!("ld.global.v4.f32 {{{a},{b},{c},{d}}}, [{p}];",a=out(reg32)old[0],b=out(reg32)old[1],c=out(reg32)old[2],d=out(reg32)old[3],p=in(reg64)at,options(nostack,readonly));}
                        let mut value=[0.0f32;4];
                        rep4!(T,{
                            let v=acc[H*features+N*4+T];let now=if R==0 {v.a} else if R==1 {v.b} else if R==2 {v.c} else {v.d};
                            value[T]=if $merge {old[T]+now} else {now};
                        });
                        asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg64)at,a=in(reg32)value[0],b=in(reg32)value[1],c=in(reg32)value[2],d=in(reg32)value[3],options(nostack));
                    });}
                    });
                }
            });
            accumulators!(T,{acc[T]=ZERO;});
        }};}
        macro_rules! close_part {()=>{{
            if PARK {
                if part==0 {save_half!(false);}
                else if part==1 {save_half!(true);}
                else if part==2 {accumulators!(T,{p2[T]=acc[T];acc[T]=ZERO;});}
                else {accumulators!(T,{
                    let at=word_base.add((T/features)*16*512+T%features);
                    let old=Acc {a:at.read(),b:at.add(features).read(),c:at.add(8*512).read(),d:at.add(8*512+features).read()};
                    acc[T]=add(old,add(p2[T],acc[T]));
                });}
            } else {
                if part==0 {accumulators!(T,{p0[T]=acc[T];acc[T]=ZERO;});}
                else if part==1 {accumulators!(T,{p0[T]=add(p0[T],acc[T]);acc[T]=ZERO;});}
                else if part==2 {accumulators!(T,{p2[T]=acc[T];acc[T]=ZERO;});}
                else {accumulators!(T,{acc[T]=add(p0[T],add(p2[T],acc[T]));});}
            }
        }};}
        while first<steps {
            // Stage the chunk's probabilities: in round `r` the block reads head `r`, this thread the slots `2 tid`
            // and `2 tid + 1` of the chunk (pair `tid % 8` of chunk step `tid / 8`). Four heads a round of loads.
            let (k,cs,pair)=(first*16+2*tid,tid/8,tid%8);
            let mut r=0usize;
            while r<32 {
                let mut raw=[[0u32;2];4];
                crate::strip_warp::unroll!(4,j,{
                    // The pointer of a slot is formed under the guard of its read: a slot past the count has none.
                    if k<count {raw[j][0]=*x.add((r+j)*f.c+k);}
                    if k+1<count {raw[j][1]=*x.add((r+j)*f.c+k+1);}
                });
                crate::strip_warp::unroll!(4,j,{
                    let n=r+j;
                    let at=stage+((((cs*2+n/16)*32+(n%8)*4+pair%4)*16+4*((n%16)/8+2*(pair/4))) as u32);
                    asm!("st.shared.u32 [{p}], {v};",p=in(reg32)at,v=in(reg32)high2!(raw[j][0],raw[j][1]),options(nostack));
                });
                r+=4;
            }
            // The cache row of each slot of the chunk, in the token's selection order. Both loads, then both checks.
            let (o0,o1)=(offset(first*16+tid),offset(first*16+tid+256));
            asm!("st.shared.u32 [{p}], {v};",p=in(reg32)ids+(tid*4) as u32,v=in(reg32)o0,options(nostack));
            asm!("st.shared.u32 [{p}], {v};",p=in(reg32)ids+((tid+256)*4) as u32,v=in(reg32)o1,options(nostack));
            crate::ptx::sync();
            let end=if first+32<steps {first+32} else {steps};
            // A panel is eight independently addressed cache steps, then eight ordered tensor steps. The final
            // panel is guarded before every load; no shared slot past the staged chunk is read.
            if PANEL {
                while s<end {
                    let mut panel=[[[0u32;2];8];8];
                    rep8!(S,{if s+S<end {panel[S]=load_b!(s+S-first);}});
                    rep8!(S,{
                        if s<end {
                            while part<3 && s>=crate::absorb_tc::range_end(steps,part) {
                                close_part!();
                                part+=1;
                            }
                            let a=quad!(stage+(((((s-first)*2+mtile)*32+lane)*16) as u32),0);
                            rep8!(T,{if T<features {acc[T]=madd(a,panel[S][T],acc[T]);}});
                            if WIDE {let a=quad!(stage+(((((s-first)*2+1)*32+lane)*16) as u32),0);rep8!(T,{if T<features {acc[T+features]=madd(a,panel[S][T],acc[T+features]);}});}
                            s+=1;
                        }
                    });
                }
            } else {
            let mut ahead=if s<end {load_b!(s-first)} else {[[0u32;2];8]};
            while s<end {
                while part<3 && s>=crate::absorb_tc::range_end(steps,part) {
                    close_part!();
                    part+=1;
                }
                let cs=s-first;let now=ahead;
                ahead=if s+1<end {load_b!(cs+1)} else {[[0u32;2];8]};
                if PF!=0 && g%4==0 && s+PF<end {
                    let low=crate::ptx::load2(ids+(((cs+PF)*16+2*t)*4) as u32);let high=crate::ptx::load2(ids+(((cs+PF)*16+2*t+8)*4) as u32);
                    crate::strip_warp::unroll!(4,r,{
                        let at=if r<2 {low[r]} else {high[r-2]};
                        if at!=NO_ROW {prefetch!(latent.wrapping_add(at as usize));}
                    });
                }
                let a=quad!(stage+((((cs*2+mtile)*32+lane)*16) as u32),0);
                rep8!(T,{if T<features {acc[T]=madd(a,now[T],acc[T]);}});
                if WIDE {let a=quad!(stage+((((cs*2+1)*32+lane)*16) as u32),0);rep8!(T,{if T<features {acc[T+features]=madd(a,now[T],acc[T+features]);}});}
                s+=1;
            }
            }
            // Every warp is done with the chunk before it is staged again.
            crate::ptx::sync();
            first+=32;
        }
        while part<4 {
            close_part!();
            part+=1;
        }
        // The lane's words: heads `g` and `g + 8` of its head tile, features `first + 8 t + T` (column 2 t) and
        // `first + 8 t + 4 + T` (column 2 t + 1): four 16-byte stores.
        let round=crate::latent_absorb::round;
        let w=word_base;
        macro_rules! store4 {($p:expr,$a:expr,$b:expr,$c:expr,$d:expr)=>{
            asm!("st.global.v4.f32 [{p}], {{{a},{b},{c},{d}}};",p=in(reg64)$p,a=in(reg32)$a,b=in(reg32)$b,c=in(reg32)$c,d=in(reg32)$d,options(nostack));
        };}
        if FULL {
            crate::strip_warp::unroll!(2,H,{
                crate::strip_warp::unroll!(2,N,{
                    let at=w.add(H*16*512+N*4);
                    store4!(at,round(acc[H*8+N*4].a),round(acc[H*8+N*4+1].a),round(acc[H*8+N*4+2].a),round(acc[H*8+N*4+3].a));
                    store4!(at.add(8),round(acc[H*8+N*4].b),round(acc[H*8+N*4+1].b),round(acc[H*8+N*4+2].b),round(acc[H*8+N*4+3].b));
                    store4!(at.add(8*512),round(acc[H*8+N*4].c),round(acc[H*8+N*4+1].c),round(acc[H*8+N*4+2].c),round(acc[H*8+N*4+3].c));
                    store4!(at.add(8*512+8),round(acc[H*8+N*4].d),round(acc[H*8+N*4+1].d),round(acc[H*8+N*4+2].d),round(acc[H*8+N*4+3].d));
                });
            });
        } else {
        store4!(w,round(acc[0].a),round(acc[1].a),round(acc[2].a),round(acc[3].a));
        store4!(w.add(4),round(acc[0].b),round(acc[1].b),round(acc[2].b),round(acc[3].b));
        store4!(w.add(8*512),round(acc[0].c),round(acc[1].c),round(acc[2].c),round(acc[3].c));
        store4!(w.add(8*512+4),round(acc[0].d),round(acc[1].d),round(acc[2].d),round(acc[3].d));
        if WIDE {
            let w=w.add(16*512);
            store4!(w,round(acc[4].a),round(acc[5].a),round(acc[6].a),round(acc[7].a));
            store4!(w.add(4),round(acc[4].b),round(acc[5].b),round(acc[6].b),round(acc[7].b));
            store4!(w.add(8*512),round(acc[4].c),round(acc[5].c),round(acc[6].c),round(acc[7].c));
            store4!(w.add(8*512+4),round(acc[4].d),round(acc[5].d),round(acc[6].d),round(acc[7].d));
        }
        }
    }
}
/// The served entries (`gpu/src/latent_absorb_rows.rs`, `third`): no hints. A cache that fits the GPU's second-level
/// cache is read faster without them (MEASURED, kernel bench runs `f8tc-lat3-1` and `f8tc-lat3-2`).
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3(p:crate::prompt_ops::Arguments) {scores3::<0,false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3(p:crate::prompt_ops::Arguments) {values3::<0,false,false,false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3_panel(p:crate::prompt_ops::Arguments) {scores3::<0,true,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_panel(p:crate::prompt_ops::Arguments) {values3::<0,true,false,false,false,false>(p);}
/// Bench-only packed-cache forms. No backend launches these entries.
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3_packed(p:crate::prompt_ops::Arguments) {scores3::<0,true,true,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_packed(p:crate::prompt_ops::Arguments) {values3::<0,true,true,false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3_packed_single(p:crate::prompt_ops::Arguments) {scores3::<0,false,true,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3_quarters(p:crate::prompt_ops::Arguments) {scores3::<0,true,true,true>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_packed_single(p:crate::prompt_ops::Arguments) {values3::<0,false,true,false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_wide(p:crate::prompt_ops::Arguments) {values3::<0,false,true,true,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_park(p:crate::prompt_ops::Arguments) {values3::<0,false,true,false,true,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_wide_park(p:crate::prompt_ops::Arguments) {values3::<0,false,true,true,true,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_full(p:crate::prompt_ops::Arguments) {values3::<0,false,true,true,false,true>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_full_park(p:crate::prompt_ops::Arguments) {values3::<0,false,true,true,true,true>(p);}
/// Bench forms with hints (the next stage; 4 steps ahead): ahead only where the cache is far larger than the GPU's
/// second-level cache (32,768 rows).
#[no_mangle]
pub extern "ptx-kernel" fn latent_scores3_p1(p:crate::prompt_ops::Arguments) {scores3::<1,false,false,false>(p);}
#[no_mangle]
pub extern "ptx-kernel" fn latent_values3_p4(p:crate::prompt_ops::Arguments) {values3::<4,false,false,false,false,false>(p);}
