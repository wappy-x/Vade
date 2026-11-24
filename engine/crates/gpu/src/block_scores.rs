//! `WindowScores` at rows in an entry of its own (`resident_launch_block_scores`, kind `BLOCK_SCORES`): the prompt
//! read's launch of many query rows against one key plane. Safe shared code for the host and the GPU.
//!
//! The rows entry (`block_rows`) runs one thread a score: the 32 lanes of a warp read 32 key rows 2 KB apart. Here a
//! block of `THREADS` threads takes one key head (`g`, of `h / t`) and `ITEMS / t` query rows, `ITEMS` (query row,
//! head) items in all. It first rounds the queries of its items into shared words once (`query_at`), then walks the
//! key plane `TILE_KEYS` stored rows at a time: the block copies the tile into shared memory (rows padded by one
//! word, so the 32 lanes read 32 banks), then warp `w` takes the query rows `w, w + 4, ..` of the block, and lane `p`
//! runs the `t` score chains of that row's heads against tile row `p` together (`chains`: one key word read for all
//! `t` heads) and writes their words, so the 32 lanes write 32 neighbouring distances. Every chain is the chain of
//! `block_attention::unit` (the query word read as BF16, the low half then the high half of each packed key word, in
//! rising place, from +0.0), and the word is made from it as `unit` makes it (`score_word`), so the words are those
//! of `block_rows::unit` (host test).
//!
//! A stored row `p` of the plane is distance `(slot - p) mod rows` of a query row: as `p` runs over every stored row,
//! the distance runs over every distance once, so each output word is written once: a score when the distance is
//! live, else +0.0; a query row whose step words are no count gets NaN words, as in `unit`.
use crate::block_ops::{Op,Shape};

/// (query row, head) items of one block: `ITEMS / t` query rows of `t` heads (every `t` up to `TAPS_MAX` divides it).
/// A part of the launch layout, not a size of a model.
pub const ITEMS:usize=48;
/// Threads of one block: 4 warps.
pub const THREADS:usize=128;
/// Lanes of a warp, and the stored key rows of one tile: lane `p` takes tile row `p`.
pub const TILE_KEYS:usize=32;
/// The most query heads a key head may have here.
pub const TAPS_MAX:usize=4;
/// The most packed words of one key row of one key head (`d / 2`).
pub const HALF_MAX:usize=64;
/// Words of the shared key tile of the entry: `TILE_KEYS` rows of at most `HALF_MAX + 1` words (the padded row).
pub const TILE_WORDS:usize=TILE_KEYS*(HALF_MAX+1);
/// Words of the shared rounded queries of the entry: `ITEMS` heads of at most `2 * HALF_MAX` values.
pub const QUERY_WORDS:usize=ITEMS*2*HALF_MAX;
/// The shared input this entry needs: the key plane (input 1). The queries and the step words are each row's own;
/// the bias (input 2) is shared (`SHARED | BIAS_SHARED`) or each row's own (`SHARED`, as the decoder launches it).
pub const SHARED:u32=0b0010;
/// The mask bit of a bias shared by the rows.
pub const BIAS_SHARED:u32=0b0100;

/// The staged launch (`Some((blocks, threads))`) of a rows launch of `WindowScores` with `rows` query rows and the
/// shared mask `shared`: blocks = `h / t` key heads times `ceil(rows / (ITEMS / t))`, threads = `THREADS`.
/// `None` for any other launch: it stays on the rows entry.
#[inline]
pub fn stage(s:Shape,rows:usize,shared:u32)->Option<(usize,usize)> {
    if s.op!=Op::WindowScores || shared&!BIAS_SHARED!=SHARED {return None;}
    crate::block_rows::lengths(s,rows,shared)?;
    let (h,d,t)=(s.heads as usize,s.dim as usize,s.taps as usize);
    if t==0 || t>TAPS_MAX || !h.is_multiple_of(t) || d/2>HALF_MAX {return None;}
    Some(((h/t).checked_mul(rows.div_ceil(ITEMS/t))?,THREADS))
}
/// Query rows of one block: `ITEMS / t`.
#[inline(always)]
pub fn block_rows_of(s:Shape)->usize {ITEMS/s.taps as usize}
/// The block's key head and first query row.
#[inline(always)]
pub fn block_of(s:Shape,rows:usize,block:usize)->(usize,usize) {
    let per=block_rows_of(s);
    let chunks=rows.div_ceil(per);
    (block/chunks,block%chunks*per)
}
/// Where value `i` of the rounded query of item `item` lies in the shared query words.
#[inline(always)]
pub fn query_at(s:Shape,item:usize,i:usize)->usize {item*s.dim as usize+i}
/// The query row and head of item `item` of a block (`None` past the rows).
#[inline(always)]
pub fn item(s:Shape,rows:usize,group:usize,first:usize,item:usize)->Option<(usize,usize)> {
    let t=s.taps as usize;
    let row=first+item/t;
    (row<rows).then_some((row,group*t+item%t))
}
/// The tile place of word `j` of tile row `p` (rows of `half + 1` words).
#[inline(always)]
pub fn tile_at(half:usize,p:usize,j:usize)->usize {p*(half+1)+j}
/// The live count and slot of query row `row`, as `unit` reads its step words; `None` makes its words NaN.
#[inline(always)]
pub fn live(s:Shape,row:usize,steps:&[f32])->Option<(usize,usize)> {crate::block_attention::live_of(&steps[row*2..row*2+2],s.rows as usize)}
/// Word `j` of stored row `p` of key head `group` in the key plane: the word the tile holds at `p * half + j`.
#[inline(always)]
pub fn plane_word(s:Shape,group:usize,p:usize,j:usize)->usize {
    let (h,t,half)=(s.heads as usize,s.taps as usize,s.dim as usize/2);
    (p*(h/t)+group)*half+j
}
/// The `T` score chains of the `T` heads of one query row with one key row, together: `q(head, i)` is value `i` of the
/// ROUNDED query of head `head` (`fixed_math::rn`, as `query_at` holds it), `k(j)` the bits of packed word `j` of the
/// key row. Each chain is the chain of `unit`: from +0.0, the low then the high half of each word, in rising place.
#[inline(always)]
pub fn chains<const T:usize>(half:usize,q:&impl Fn(usize,usize)->f32,k:&impl Fn(usize)->u32)->[f32;T] {
    let mut sums=[0.0f32;T];
    for j in 0..half {
        let bits=k(j);
        let (lo,hi)=(f32::from_bits(bits<<16),f32::from_bits(bits&0xffff_0000));
        for (head,sum) in sums.iter_mut().enumerate() {*sum+=q(head,2*j)*lo;*sum+=q(head,2*j+1)*hi;}
    }
    sums
}
/// `chains` for `t` heads (1 to `TAPS_MAX`, as `stage` admits), in the first `t` words; NaN words for any other `t`.
#[inline(always)]
pub fn chains_of(t:usize,half:usize,q:&impl Fn(usize,usize)->f32,k:&impl Fn(usize)->u32)->[f32;TAPS_MAX] {
    let mut out=[f32::NAN;TAPS_MAX];
    match t {
        1=>out[..1].copy_from_slice(&chains::<1>(half,q,k)),
        2=>out[..2].copy_from_slice(&chains::<2>(half,q,k)),
        3=>out[..3].copy_from_slice(&chains::<3>(half,q,k)),
        4=>out=chains::<4>(half,q,k),
        _=>{},
    }
    out
}
/// The distance of stored row `p` from a query row at `slot`.
#[inline(always)]
pub fn distance(rows:usize,slot:usize,p:usize)->usize {if p<=slot {slot-p} else {slot+rows-p}}
/// The output word of head `head` at distance `dist` from its chain `sum`, as `unit` writes it (a live distance).
#[inline(always)]
pub fn score_word(s:Shape,head:usize,dist:usize,bias:&[f32],sum:f32)->f32 {
    use crate::fixed_math::rn;
    let k=s.cols as usize;
    let scaled=rn(rn(sum)*s.bound);
    rn(scaled+if dist<k {bias[head*k+dist]} else {0.0})
}
/// The bias words of query row `row`: the shared bias, or the row's own.
#[inline(always)]
pub fn bias_of(s:Shape,row:usize,shared:u32,bias:&[f32])->&[f32] {
    if shared&BIAS_SHARED!=0 {bias} else {let one=(s.heads*s.cols) as usize;&bias[row*one..(row+1)*one]}
}
/// Where output word (`row`, `head`, `dist`) lies: the units of a row are `heads * rows`, one word each.
#[inline(always)]
pub fn out_index(s:Shape,row:usize,head:usize,dist:usize)->usize {
    let (h,r)=(s.heads as usize,s.rows as usize);
    (row*h+head)*r+dist
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape(heads:u32,dim:u32,taps:u32,rows:u32,cols:u32)->Shape {Shape {op:Op::WindowScores,heads,dim,taps,rows,cols,epsilon:0.0,bound:0.088_388_35}}
    /// The bits of each word, every NaN as one word: an optimized build may swap the operands of an add, which changes
    /// which NaN payload survives (seat INKARM, release build on the ARM box), not whether the word is NaN.
    fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|if v.is_nan() {f32::NAN.to_bits()} else {v.to_bits()}).collect()}
    /// The staged entry, block by block and phase by phase as the GPU runs it: the rounded queries of the block's items,
    /// then each tile; warp `w` takes the block's query rows w, w + 4, ..; lane `p` the tile row `p`.
    fn staged(s:Shape,rows:usize,shared:u32,x:[&[f32];4])->Vec<f32> {
        let (blocks,threads)=stage(s,rows,shared).unwrap();
        let (half,r,width,d,t)=(s.dim as usize/2,s.rows as usize,(s.heads*s.dim) as usize,s.dim as usize,s.taps as usize);
        let mut out=vec![f32::from_bits(0x7fc0_dead);rows*s.heads as usize*r];
        let (mut tile,mut queries)=(vec![0.0f32;TILE_WORDS],vec![0.0f32;QUERY_WORDS]);
        let warps=threads/TILE_KEYS;
        for block in 0..blocks {
            let (group,first_row)=block_of(s,rows,block);
            for i in 0..ITEMS*d {
                let (it,c)=(i/d,i%d);
                queries[query_at(s,it,c)]=match item(s,rows,group,first_row,it) {Some((row,head))=>crate::fixed_math::rn(x[0][row*width+head*d+c]),None=>0.0};
            }
            let mut first=0;
            while first<r {
                let count=(r-first).min(TILE_KEYS);
                for i in 0..count*half {let (p,j)=(i/half,i%half);tile[tile_at(half,p,j)]=x[1][plane_word(s,group,first+p,j)];}
                for warp in 0..warps {
                    let mut local=warp;
                    while local<block_rows_of(s) {
                        let row=first_row+local;
                        if row<rows {
                            let state=live(s,row,x[3]);
                            for lane in 0..count {
                                match state {
                                    None=>for head in 0..t {out[out_index(s,row,group*t+head,first+lane)]=f32::NAN;},
                                    Some((n,slot))=>{
                                        let dist=distance(r,slot,first+lane);
                                        let sums=if dist<n {chains_of(t,half,&|head,i|queries[query_at(s,local*t+head,i)],&|j|tile[tile_at(half,lane,j)].to_bits())} else {[0.0;TAPS_MAX]};
                                        for head in 0..t {
                                            let word=if dist<n {score_word(s,group*t+head,dist,bias_of(s,row,shared,x[2]),sums[head])} else {0.0};
                                            out[out_index(s,row,group*t+head,dist)]=word;
                                        }
                                    }
                                }
                            }
                        }
                        local+=warps;
                    }
                }
                first+=count;
            }
        }
        out
    }
    fn by_units(s:Shape,rows:usize,shared:u32,x:[&[f32];4])->Vec<f32> {
        let units=(s.heads*s.rows) as usize;
        let mut out=vec![0.0f32;rows*units];
        for (index,word) in out.iter_mut().enumerate() {crate::block_rows::unit(s,rows,shared,index,x,&[],core::slice::from_mut(word));}
        out
    }
    #[test] fn window_scores_staged_at_rows_give_the_words_of_the_rows_unit() {
        let mut state=0x1234_5677u32;
        let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
        let mut checked=0;
        // (heads, dim, taps, key rows, bias extent, query rows): taps 1 to 4, rows that are no multiple of a tile or of
        // the rows of a block (ITEMS / t), the model's 24 heads of 128 at 3 taps; a window of one key row, one query
        // row and a head of two values (review REVSCORES, LOW 3).
        for (h,d,t,r,k,rows) in [(2u32,4u32,1u32,5u32,3u32,3usize),(4,16,2,33,40,40),(6,8,3,64,8,33),(8,32,4,70,70,65),(24,128,3,100,64,37),(3,128,3,40,20,2),(2,128,1,40,20,49),(1,2,1,1,1,1),(2,2,2,3,2,1)] {
            let s=shape(h,d,t,r,k);let (hu,du,tu,ru,ku)=(h as usize,d as usize,t as usize,r as usize,k as usize);
            for (case,shared) in (0..8).map(|c|(c%4,if c<4 {SHARED} else {SHARED|BIAS_SHARED})) {
                assert!(stage(s,rows,shared).is_some());
                let rnd=|v:u32,scale:f32|((v>>8) as f32/8_388_608.0-1.0)*scale;
                // A few odd words: signed zeros, infinities and NaN in the queries, the keys and the bias.
                let odd=|v:u32,x:f32|match v%89 {0=>-0.0,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>f32::NAN,_=>x};
                let queries=(0..rows*hu*du).map(|_|{let v=next();if v%97==0 {-0.0} else {odd(v>>3,rnd(v,2.0))}}).collect::<Vec<_>>();
                // Packed BF16 pairs; now and then a half that is an infinity or a NaN.
                let plane=(0..ru*(hu/tu)*du/2).map(|_|{let v=next();let w=(v&0x7fbf_7fbf)^0x3f00_3f00;f32::from_bits(match v>>24&63 {0=>w|0x7f80_0000,1=>w|0x0000_7fc0,_=>w})}).collect::<Vec<_>>();
                let bias=(0..if shared&BIAS_SHARED!=0 {1} else {rows}*hu*ku).map(|_|{let v=next();odd(v>>5,rnd(v,0.5))}).collect::<Vec<_>>();
                // Step words of each row: live count and slot; case 3 makes some rows no count (NaN words).
                let steps=(0..rows).flat_map(|row|{let v=next() as usize;let n=match case {0=>ru,1=>v%(ru+1),2=>(v%3).min(ru),_=>v%(ru+1)};
                    let slot=(v>>8)%ru;if case==3 && row%5==1 {[0.5f32,0.0]} else if case==3 && row%7==2 {[n as f32,ru as f32]} else {[n as f32,slot as f32]}}).collect::<Vec<_>>();
                let x=[&queries[..],&plane[..],&bias[..],&steps[..]];
                assert_eq!(bits(&staged(s,rows,shared,x)),bits(&by_units(s,rows,shared,x)),"{h} heads of {d}, taps {t}, {r} keys, {rows} rows, case {case}, mask {shared}");
                checked+=1;
            }
        }
        assert_eq!(checked,72);
    }
    #[test] fn window_scores_are_staged_only_in_the_launch_they_fit() {
        let s=shape(24,128,3,1024,1024);
        assert_eq!(stage(s,1024,SHARED),Some((8*64,128)));
        assert_eq!(stage(s,33,SHARED),Some((8*3,128)));
        // Another mask, another arm, too many taps, a key row too wide, a shape the arm refuses: the rows entry.
        assert_eq!(stage(s,1024,SHARED|BIAS_SHARED),Some((8*64,128)));
        assert_eq!(stage(s,1024,0b0100),None);
        assert_eq!(stage(s,1024,0b0011),None);
        assert_eq!(stage(s,1024,0b1010),None);
        assert_eq!(stage(s,1024,SHARED|crate::block_rows::LIST),None);
        assert_eq!(stage(Shape {op:Op::WindowValues,..s},1024,SHARED),None);
        assert_eq!(stage(shape(10,128,5,64,8),4,SHARED),None);
        assert_eq!(stage(shape(2,258,1,64,8),4,SHARED),None);
        // A head wider than 2 * HALF_MAX values: the rows entry.
        assert_eq!(stage(shape(3,256,3,40,20),2,SHARED),None);
        // Review REVSCORES3, LOW: the first head width past 2 * HALF_MAX (d = 130, half = 65) is refused; 128 is staged.
        assert_eq!(stage(shape(3,130,3,40,20),2,SHARED),None);
        assert!(stage(shape(3,128,3,40,20),2,SHARED).is_some());
        assert_eq!(stage(Shape {bound:0.0,..s},4,SHARED),None);
        assert_eq!(stage(s,0,SHARED),None);
        // More refusals (review REVSCORES, LOW 3): no shared key plane, no taps, heads no multiple of the taps, an odd
        // head width, no key rows or too many, too many query rows, the window form.
        assert_eq!(stage(s,4,0),None);
        assert_eq!(stage(Shape {taps:0,..s},4,SHARED),None);
        assert_eq!(stage(shape(10,128,4,64,8),4,SHARED),None);
        assert_eq!(stage(shape(2,5,1,64,8),4,SHARED),None);
        assert_eq!(stage(Shape {rows:0,..s},4,SHARED),None);
        assert_eq!(stage(Shape {rows:(1<<24)+1,..s},4,SHARED),None);
        assert_eq!(stage(s,crate::block_rows::ROWS_MAX+1,SHARED),None);
        assert_eq!(stage(s,4,SHARED|crate::block_rows::WINDOW),None);
        // Each item of each block of a launch is one (row, head) or past the rows, and every (row, head) is one item.
        let (rows,s)=(70,shape(6,8,3,16,4));
        let (blocks,_)=stage(s,rows,SHARED).unwrap();
        let mut seen=vec![0u8;rows*6];
        for block in 0..blocks {let (group,first)=block_of(s,rows,block);for it in 0..ITEMS {if let Some((row,head))=item(s,rows,group,first,it) {seen[row*6+head]+=1;}}}
        assert!(seen.iter().all(|&n|n==1));
    }
}
