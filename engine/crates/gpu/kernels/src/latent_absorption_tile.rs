//! Grouped latent absorption in shared-memory tiles. A tile changes loads, never a token's sum order.
//! A block owns 32 A rows and 64 B rows of one operation: up to 2,048 dot products over `terms` columns. A thread owns
//! 2 A rows x 4 B rows. The block walks the columns in chunks of four strips of SUM-ORDER v1, one 256-column period
//! at a time. A thread keeps one sum for each of its dot products and each of the four strips. Each sum takes its
//! columns in ascending order. The strips then join in the order of the fixed tree (distances 16, 8, 4, 2, 1):
//!   a[i] = s[i] + s[i+16], b[i] = a[i] + a[i+8], c[i] = b[i] + b[i+4], e[i] = c[i] + c[i+2], sum = e[0] + e[1].
//! Chunk g holds strips g, g+16, g+8, g+24 and so yields b[g]. The chunks run in the order 0, 4, 2, 6, 1, 5, 3, 7, so
//! three held values a dot product are enough. An empty strip is +0, as in the scalar form.
//! Two blocks run on an SM at once: the stamp caps the kernel at 128 registers (ptx_stamp.rs), and without a cap
//! one block fills the SM's registers. To fit, a thread keeps only the running chunk's 32 strip sums in registers.
//! The held values live in its local memory (read and written once a turn). A quad's operands arrive one load at a
//! time: one B row's four columns, or for values one column's four features. A thread's A row and B rows are the
//! same for every chunk, so their sources are found once, before the first chunk. None of this changes a sum:
//! every strip sum still takes the same rounded products in ascending column order.
//! | operation | outer | A rows  | B rows          | columns         |
//! | query     | head  | tokens  | latent features | key features    |
//! | scores    | token | heads   | selected slots  | latent features |
//! | values    | token | heads   | latent features | selected slots  |
//! | project   | head  | tokens  | value features  | latent features |
use crate::{latent_absorb_rows::{TILE_A,TILE_B,TILE_THREADS,tile_b_row},prompt_ops::{Arguments,Op}};
/// A staged row holds one chunk: 4 strips x 8 columns, then 4 words of padding that spread the rows over the banks.
/// Values stages its cache rows by column: 64 features and 8 words of padding a column, in the same B area.
const A:usize=TILE_A as usize;const B:usize=TILE_B as usize;const SK:usize=36;const VS:usize=72;const WORDS:usize=(A+B)*SK;const PERIOD:usize=256;
#[no_mangle]
pub extern "ptx-kernel" fn latent_absorption_tile(p:Arguments) {
    // SAFETY: prompt_operation validates latent_absorb_rows::lengths and launches latent_absorb_rows::launch:
    // 256 threads and one block for each outer index and 32 x 64 tile; any other grid traps or returns as a whole block.
    // A selected count is a finite integer at most `first`, and a selected ID is a finite integer below `rows`, or the
    // block traps before the read that would use it (scores reads its two rows' IDs once, before the first chunk;
    // values reads a column's ID when it issues that column). Every global read is below an extent that the host checked:
    // A and B rows below their counts, columns below `terms`, packed BF16 weights inside [heads,key+value,latent].
    // The 27,648 shared bytes belong to this block: two stage buffers. A thread stages only the quads it owns
    // (asynchronous copies of 16 or 8 aligned bytes or of single words, or widened BF16 stores), waits for the running
    // chunk's copies, widens its own packed quads in place, and reads staged words of other threads only after the
    // barrier that follows. A buffer is staged again only after the barrier that ends the products that read it.
    // A staged word outside the valid rows and columns is never used for an output. All 256 threads reach every
    // barrier: the chunk and period counts come from the shape, or from the one token's count that the whole block
    // shares. Each thread writes its own in-range outputs once. Threads and blocks own disjoint outputs. Inputs and
    // the private output are disjoint and stay live on the ordered stream. Products and sums are separate
    // round-to-nearest instructions on registers, so nothing contracts.
    unsafe {
        use core::arch::{asm,nvptx::{_thread_idx_x,_block_idx_x,_block_dim_x}};
        let s=p.shape;
        if _block_dim_x() as usize!=TILE_THREADS as usize || !matches!(s.op,Op::AbsorbQuery|Op::AbsorbScores|Op::AbsorbValues|Op::AbsorbProject) {asm!("trap;");}
        let (m,h,d,v,k,c,n)=(s.tokens as usize,s.heads as usize,s.dim as usize,s.taps as usize,s.cols as usize,s.first as usize,s.slots as usize);
        let thread=_thread_idx_x() as usize;let (ty,tx)=(thread/16,thread%16);let block=_block_idx_x() as usize;
        let dense=matches!(s.op,Op::AbsorbQuery|Op::AbsorbProject);let values=s.op==Op::AbsorbValues;
        let (na,nb)=match s.op {Op::AbsorbQuery=>(m,k),Op::AbsorbProject=>(m,v),Op::AbsorbScores=>(h,c),_=>(h,k)};
        let (ta,tb)=(na.div_ceil(A),nb.div_ceil(B));
        let (outer,a0,b0)=(block/(ta*tb),block/tb%ta*A,block%tb*B);
        if outer>=if dense {h} else {m} {return;}
        let count_of=|table:*const f32,token:usize|->usize {let raw=*table.add(token*(n+1)+n);
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>c {asm!("trap;");}raw as usize};
        let count=if dense {0} else {count_of(p.input[2] as *const f32,outer)};
        let selected=if dense {core::ptr::null()} else {(p.input[2] as *const f32).add(outer*(n+1))};
        let id=|slot:usize|->usize {let raw=*selected.add(slot);
            if !(raw>=0.0 && raw==raw as u32 as f32) || raw as usize>=s.rows as usize {asm!("trap;");}raw as usize};
        let terms=match s.op {Op::AbsorbQuery=>d,Op::AbsorbValues=>count,_=>k};
        let x=p.input[0] as *const f32;let latent=p.input[1] as *const f32;let w=p.weight as *const u16;
        let stage=crate::ptx::shared::<{2*WORDS*4}>();
        // Chunks run in the order 0, 4, 2, 6, 1, 5, 3, 7: the bits of the turn, reversed.
        let chunk_of=|turn:usize|->usize {(turn&1)<<2|turn&2|turn>>2};
        // Strip of a slot of a chunk: chunk, chunk+16, chunk+8, chunk+24.
        let strip=|turn:usize,slot:usize|->usize {chunk_of(turn)+(slot&1)*16+(slot>>1)*8};
        // A chunk of a period holds columns only while its lowest strip starts below `terms`.
        let live=|turn:usize,first:usize|->bool {turn<8 && first+8*chunk_of(turn)<terms};
        // The live chunk after this one: the next period of the turn, or the first period of the next turn with columns.
        let after=|turn:usize,first:usize|->(usize,usize) {
            if live(turn,first+PERIOD) {(turn,first+PERIOD)} else {let mut next=turn+1;while next<8 && !live(next,0) {next+=1;}(next,0)}
        };
        // Staged quad of a row: stage buffer, strip slot 0..3 of the chunk, quad 0..1 of the strip's eight columns.
        // All four factors are small (two buffers of 13,824 bytes), so the offset is formed in 32 bits.
        let place=|buffer:usize,row:usize,slot:usize,quad:usize|->u32 {stage+(buffer as u32)*((WORDS*4) as u32)+(row as u32)*((SK*4) as u32)+((slot*8+quad*4)*4) as u32};
        // Values: the features of one staged column. `at` is the column's place in the chunk, 0..31.
        let place_values=|buffer:usize,at:usize,feature:usize|->u32 {stage+((buffer*WORDS+A*SK+at*VS+feature)*4) as u32};
        // Up to four FP32 words by asynchronous copy: one 16-byte copy when the source allows it, else one copy a word.
        let copy=|to:u32,from:*const f32,valid:usize| {
            if valid==4 && (from as usize)&15==0 {crate::ptx::copy16_cached(to,from.cast());}
            else {let mut e=0;while e<valid {crate::ptx::copy4(to+(e*4) as u32,from.add(e).cast());e+=1;}}
        };
        // Four packed BF16 weights go to the first 8 bytes of their staged quad by one asynchronous copy, and are widened
        // in place to the exact FP32 words after the wait. A quad that the copy cannot take is widened from global words now.
        let packed=|from:*const u16,valid:usize|->bool {valid==4 && (from as usize)&7==0};
        let fetch=|to:u32,from:*const u16,valid:usize| {
            if packed(from,valid) {crate::ptx::copy8(to,from.cast());}
            else {let mut e=0;while e<valid {crate::ptx::store(to+(e*4) as u32,f32::from_bits((*from.add(e) as u32)<<16));e+=1;}}
        };
        let widen=|to:u32| {let [low,high]=crate::ptx::load2(to);
            crate::ptx::store(to,f32::from_bits(low<<16));crate::ptx::store(to+4,f32::from_bits(low&0xffff_0000));
            crate::ptx::store(to+8,f32::from_bits(high<<16));crate::ptx::store(to+12,f32::from_bits(high&0xffff_0000));
        };
        // The rows a thread stages are the same for every chunk: its A row (thread/8) and its two B rows (thread/8 and
        // 32 + thread/8). Column 0 of each is found here, once. Null: the row is outside the operation, or for scores
        // past the token's count, and nothing is staged for it. Values stages its B area by column, not by row.
        let a_source:*const f32={let a=a0+thread/8;if a>=na {core::ptr::null()} else {match s.op {
            Op::AbsorbQuery=>x.add((a*h+outer)*d),Op::AbsorbProject=>x.add((a*h+outer)*k),
            Op::AbsorbScores=>x.add((outer*h+a)*k),_=>x.add((outer*h+a)*c)}}};
        let b_source=|n:usize|->*const u8 {let b=b0+(thread+n*TILE_THREADS as usize)/8;
            if values || b>=nb {core::ptr::null()}
            else if dense {(if s.op==Op::AbsorbQuery {w.add(outer*(d+v)*k+b*d)} else {w.add((outer*(d+v)+d+b)*k)}).cast()}
            else if b<count {latent.add(id(b)*k).cast()} else {core::ptr::null()}};
        let b_sources=[b_source(0),b_source(1)];
        // Stage one chunk of one period: each thread issues one A quad and two B quads.
        let issue=|turn:usize,first:usize,buffer:usize| {
            let (row,slot,quad)=(thread/8,thread%8/2,thread%2);let column=first+8*strip(turn,slot)+quad*4;
            if !a_source.is_null() && column<terms {copy(place(buffer,row,slot,quad),a_source.add(column),(terms-column).min(4));}
            let mut item=thread;
            while item<B*8 {
                if values {
                    // One selected row's four features: 32 columns x 16 feature quads.
                    let (at,group)=(item/16,item%16);let (column,feature)=(first+8*strip(turn,at/8)+at%8,b0+group*4);
                    if column<terms && feature<nb {copy(place_values(buffer,at,group*4),latent.add(id(column)*k+feature),(nb-feature).min(4));}
                } else {
                    let (row,slot,quad)=(item/8,item%8/2,item%2);let column=first+8*strip(turn,slot)+quad*4;
                    let source=b_sources[item/TILE_THREADS as usize];
                    if !source.is_null() && column<terms {let valid=(terms-column).min(4);let to=place(buffer,A+row,slot,quad);
                        if dense {fetch(to,source.cast::<u16>().add(column),valid);} else {copy(to,source.cast::<f32>().add(column),valid);}}
                }
                item+=TILE_THREADS as usize;
            }
        };
        // After the wait: widen this thread's packed weight quads of the chunk in place.
        let finish=|turn:usize,first:usize,buffer:usize| {
            if dense {let mut item=thread;
                while item<B*8 {
                    let (row,slot,quad)=(item/8,item%8/2,item%2);let column=first+8*strip(turn,slot)+quad*4;
                    let source=b_sources[item/TILE_THREADS as usize];
                    if !source.is_null() && column<terms && packed(source.cast::<u16>().add(column),(terms-column).min(4)) {widen(place(buffer,A+row,slot,quad));}
                    item+=TILE_THREADS as usize;
                }
            }
        };
        let add=|x:f32,y:f32|->f32 {let sum:f32;asm!("add.rn.f32 {s}, {x}, {y};",s=out(reg32)sum,x=in(reg32)x,y=in(reg32)y,options(nomem,nostack));sum};
        // acc[strip slot][a][b]: the four strip sums of the running chunk. They are named by constant indices only, so
        // they stay in registers. hold[level][a][b]: joined values of the tree. A turn reads and writes them once, so
        // they live in the thread's local memory: the volatile accesses keep 24 values out of the registers.
        let mut acc=[[[0.0f32;4];2];4];let mut hold=[[[0.0f32;4];2];3];
        macro_rules! held {($l:expr,$i:expr,$j:expr)=>{core::ptr::read_volatile(&hold[$l][$i][$j])};}
        macro_rules! hold_set {($l:expr,$i:expr,$j:expr,$v:expr)=>{{let value=$v;core::ptr::write_volatile(&mut hold[$l][$i][$j],value);}};}
        macro_rules! pairs {($f:ident $(,$x:expr)*)=>{
            $f!(0,0 $(,$x)*);$f!(0,1 $(,$x)*);$f!(0,2 $(,$x)*);$f!(0,3 $(,$x)*);$f!(1,0 $(,$x)*);$f!(1,1 $(,$x)*);$f!(1,2 $(,$x)*);$f!(1,3 $(,$x)*);
        };}
        let mut at=if live(0,0) {(0,0)} else {(8,0)};let mut buffer=0;
        if at.0<8 {issue(at.0,at.1,buffer);crate::ptx::commit();}
        let mut turn=0;
        while turn<8 {
            while at.0==turn {
                let first=at.1;let next=after(at.0,at.1);
                // The next chunk's copies are issued before this chunk's products, so their wait overlaps the products.
                if next.0<8 {issue(next.0,next.1,buffer^1);crate::ptx::commit();crate::ptx::wait_one();} else {crate::ptx::wait();}
                finish(turn,first,buffer);
                crate::ptx::sync();
                // Four columns of one strip: 6 loads and up to 32 products. The two A quads are loaded first. Then one load
                // at a time brings a B row's four columns (values: one column's four features), so 12 operand words are
                // live at once, not 24. A strip sum takes its columns in ascending order either way.
                macro_rules! quad {($slot:expr,$quad:expr)=>{let column=first+8*strip(turn,$slot)+$quad*4;
                    if column<terms {
                        let a=[crate::ptx::load4(place(buffer,ty*2,$slot,$quad)),crate::ptx::load4(place(buffer,ty*2+1,$slot,$quad))];
                        // One product, rounded, then added to its own strip sum.
                        macro_rules! one {($i:expr,$j:expr,$x:expr,$y:expr)=>{let product:f32;
                            asm!("mul.rn.f32 {p}, {a}, {b};",p=out(reg32)product,a=in(reg32)f32::from_bits($x),b=in(reg32)f32::from_bits($y),options(nomem,nostack));
                            asm!("add.rn.f32 {s}, {s}, {p};",s=inout(reg32)acc[$slot][$i][$j],p=in(reg32)product,options(nomem,nostack));
                        };}
                        if values {
                            // One load holds this thread's four features (its B rows) of one column.
                            macro_rules! col {($e:expr)=>{if column+$e<terms {let f=crate::ptx::load4(place_values(buffer,$slot*8+$quad*4+$e,tx*4));
                                one!(0,0,a[0][$e],f[0]);one!(0,1,a[0][$e],f[1]);one!(0,2,a[0][$e],f[2]);one!(0,3,a[0][$e],f[3]);
                                one!(1,0,a[1][$e],f[0]);one!(1,1,a[1][$e],f[1]);one!(1,2,a[1][$e],f[2]);one!(1,3,a[1][$e],f[3]);}};}
                            col!(0);col!(1);col!(2);col!(3);
                        } else {
                            // One load holds four columns of one of this thread's B rows (tx, tx+16, tx+32, tx+48).
                            macro_rules! row {($j:expr)=>{let r=crate::ptx::load4(place(buffer,A+tx+16*$j,$slot,$quad));
                                one!(0,$j,a[0][0],r[0]);one!(1,$j,a[1][0],r[0]);
                                if column+1<terms {one!(0,$j,a[0][1],r[1]);one!(1,$j,a[1][1],r[1]);}
                                if column+2<terms {one!(0,$j,a[0][2],r[2]);one!(1,$j,a[1][2],r[2]);}
                                if column+3<terms {one!(0,$j,a[0][3],r[3]);one!(1,$j,a[1][3],r[3]);}};}
                            row!(0);row!(1);row!(2);row!(3);
                        }
                    }
                };}
                quad!(0,0);quad!(0,1);quad!(1,0);quad!(1,1);quad!(2,0);quad!(2,1);quad!(3,0);quad!(3,1);
                crate::ptx::sync();
                at=next;buffer^=1;
            }
            // Join the chunk's four strips to b[chunk], then join it into the tree. The order is fixed by the turn.
            // A turn with no columns joins four +0 sums, as the scalar form does.
            macro_rules! join {($i:expr,$j:expr)=>{
                let joined=add(add(acc[0][$i][$j],acc[1][$i][$j]),add(acc[2][$i][$j],acc[3][$i][$j]));
                acc[0][$i][$j]=0.0;acc[1][$i][$j]=0.0;acc[2][$i][$j]=0.0;acc[3][$i][$j]=0.0;
                match turn {
                    0=>hold_set!(0,$i,$j,joined),
                    1=>hold_set!(0,$i,$j,add(held!(0,$i,$j),joined)),
                    2=>hold_set!(1,$i,$j,joined),
                    3=>hold_set!(0,$i,$j,add(held!(0,$i,$j),add(held!(1,$i,$j),joined))),
                    4=>hold_set!(1,$i,$j,joined),
                    5=>hold_set!(1,$i,$j,add(held!(1,$i,$j),joined)),
                    6=>hold_set!(2,$i,$j,joined),
                    _=>hold_set!(0,$i,$j,add(held!(0,$i,$j),add(held!(1,$i,$j),add(held!(2,$i,$j),joined)))),
                }
            };}
            pairs!(join);
            turn+=1;
        }
        let out=p.output as *mut f32;
        macro_rules! emit {($i:expr,$j:expr)=>{let (a,b,sum)=(a0+ty*2+$i,b0+tile_b_row(values,tx as u32,$j) as usize,held!(0,$i,$j));
            if a<na && b<nb {match s.op {
                Op::AbsorbQuery=>out.add((a*h+outer)*k+b).write(crate::latent_absorb::round(sum)),
                Op::AbsorbProject=>{let live=count_of(p.input[1] as *const f32,a)!=0;
                    out.add((a*h+outer)*v+b).write(if live {crate::latent_absorb::round(sum)} else {0.0});},
                Op::AbsorbScores=>{let value=if b<count {let scale=1.0/crate::causal_ops::root(d as f32);let scaled:f32;
                        asm!("mul.rn.f32 {v}, {a}, {b};",v=out(reg32)scaled,a=in(reg32)sum,b=in(reg32)scale,options(nomem,nostack));
                        scaled} else {0.0};
                    out.add((outer*h+a)*c+b).write(value);},
                _=>out.add((outer*h+a)*k+b).write(crate::latent_absorb::round(sum)),
            }}
        };}
        pairs!(emit);
    }
}
