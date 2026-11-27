//! `WindowValues` at rows in an entry of its own (`resident_launch_block_values`, kind `BLOCK_VALUES`): the prompt
//! read's launch of many query rows against one value plane. Safe shared code for the host and the GPU.
//!
//! The rows entry (`block_rows`) runs one thread a word, and each thread walks the whole view: the value rows of a
//! key head are read again for each query row and each of its query heads. Here a block of `THREADS` threads takes
//! one value head (`group`, of `h / t`), `CHANNELS` of its channels (a slice) and `QUERY_ROWS` query rows with the `t`
//! query heads of that value head. For each block of `block_attention::BLOCK` distances the block copies the stored
//! rows those query rows need into shared memory once (`TILE_ROWS` rows of `WORDS` packed words, padded by one word,
//! so 32 neighbouring rows sit in 32 banks); then each warp takes the query rows of the block in turn, and lane `l`
//! runs lane chain `l` of the sum (the distances `l`, `l + 32`, .. of the block, in rising order, from +0.0) for the
//! `t * CHANNELS` words of that query row. The 32 lane sums of a word are joined as `block_ops::join` joins them, and
//! the block sums are a chain in rising block number from +0.0 (`block_attention::distance_tree`); the word is made
//! from it as `unit` makes it (`word`). So the words are those of `block_rows::unit` (host test).
//!
//! The tile: tile row `k` of distance block `b` holds the stored row `(anchor - b * BLOCK - k) mod rows`, where the
//! anchor is the slot of the block's last query row. A query row at `slot` reads its distance `b * BLOCK + j` (stored
//! row `(slot - b * BLOCK - j) mod rows`) at tile row `offset + j` with `offset = (anchor - slot) mod rows`: for the
//! rows of one prompt chunk (slots one apart) the offset is below `QUERY_ROWS`, and the tile holds every row it reads.
//! A query row with another offset (step words that are not one chunk) reads the plane itself: the same words.
use crate::block_ops::{Op,Shape,LANES};
use crate::block_attention::BLOCK;

/// Query rows of one block. A part of the launch layout, not a size of a model.
pub const QUERY_ROWS:usize=32;
/// Threads of one block: 4 warps.
pub const THREADS:usize=128;
/// Channels of one block (a slice of the value head): each lane keeps `TAPS_MAX * CHANNELS` lane chains.
pub const CHANNELS:usize=16;
/// Packed words of one tile row (two channels a word).
pub const WORDS:usize=CHANNELS/2;
/// Words of one tile row in shared memory: `WORDS` and one word of padding.
pub const ROW_WORDS:usize=WORDS+1;
/// The most query heads a value head may have here.
pub const TAPS_MAX:usize=3;
/// Rows of the tile: a block of distances seen from `QUERY_ROWS` neighbouring slots.
pub const TILE_ROWS:usize=BLOCK+QUERY_ROWS-1;
/// Words of the tile.
pub const TILE_WORDS:usize=TILE_ROWS*ROW_WORDS;
/// Words of the block sums chain of a block: `QUERY_ROWS` rows, `TAPS_MAX` heads, `CHANNELS` channels.
pub const TOTAL_WORDS:usize=QUERY_ROWS*TAPS_MAX*CHANNELS;
/// The one input mask this entry takes: the value plane (input 1) shared, the weights, the gates and the step words
/// each row's own, as the decoder launches it.
pub const SHARED:u32=0b0010;
/// `add_terms` and `join_chains` are written out for these sizes.
const _:()=assert!(CHANNELS==16 && WORDS==8 && TAPS_MAX==3 && LANES==32);

/// The staged launch (`Some((blocks, threads))`) of a rows launch of `WindowValues` with `rows` query rows and the
/// shared mask `shared`: blocks = `ceil(rows / QUERY_ROWS)` chunks times `h / t` value heads times `d / CHANNELS`
/// slices, threads = `THREADS`. `None` for any other launch: it stays on the rows entry.
#[inline]
pub fn stage(s:Shape,rows:usize,shared:u32)->Option<(usize,usize)> {
    if s.op!=Op::WindowValues || shared!=SHARED {return None;}
    crate::block_rows::lengths(s,rows,shared)?;
    let (h,d,t)=(s.heads as usize,s.dim as usize,s.taps as usize);
    if t==0 || t>TAPS_MAX || !h.is_multiple_of(t) || d==0 || !d.is_multiple_of(CHANNELS) {return None;}
    let blocks=rows.div_ceil(QUERY_ROWS).checked_mul(h/t)?.checked_mul(d/CHANNELS)?;
    (blocks<=u32::MAX as usize).then_some((blocks,THREADS))
}
/// The block's value head, slice and first query row. The slices of one (chunk, value head) are neighbour blocks, so
/// they read the same weights at about the same time.
#[inline(always)]
pub fn block_of(s:Shape,block:usize)->(usize,usize,usize) {
    let (groups,slices)=((s.heads/s.taps) as usize,s.dim as usize/CHANNELS);
    let (rest,slice)=(block/slices,block%slices);
    (rest%groups,slice,rest/groups*QUERY_ROWS)
}
/// The live count and slot of query row `row`, as `unit` reads its step words; `None` makes its words NaN.
#[inline(always)]
pub fn live(s:Shape,row:usize,steps:&[f32])->Option<(usize,usize)> {crate::block_attention::live_of(&steps[row*2..row*2+2],s.rows as usize)}
/// The query rows `first..end` of a block of a launch of `rows` rows.
#[inline(always)]
pub fn end_of(rows:usize,first:usize)->usize {if rows-first>QUERY_ROWS {first+QUERY_ROWS} else {rows}}
/// The anchor of a block: the slot of its last query row, or 0 when that row has no count. Any anchor gives the same
/// words; this one puts the rows of one prompt chunk in the tile.
#[inline(always)]
pub fn anchor(s:Shape,end:usize,steps:&[f32])->usize {match live(s,end-1,steps) {Some((_,slot))=>slot,None=>0}}
/// The most live distances of the block's query rows: the blocks of distances the block walks.
#[inline(always)]
pub fn most_live(s:Shape,first:usize,end:usize,steps:&[f32])->usize {
    let mut most=0;
    for row in first..end {if let Some((n,_))=live(s,row,steps) {if n>most {most=n;}}}
    most
}
/// Tile rows the block copies for the distance block from `base`: the rows that a query row with an offset below
/// `QUERY_ROWS` reads (`most` = `most_live`, above `base`).
#[inline(always)]
pub fn tile_count(base:usize,most:usize)->usize {let left=most-base;QUERY_ROWS-1+if left>BLOCK {BLOCK} else {left}}
/// The stored row that tile row `k` holds for the distance block from `base`: `(anchor - base - k) mod rows`.
#[inline(always)]
pub fn tile_row(rows:usize,anchor:usize,base:usize,k:usize)->usize {
    let (r,back)=(rows as u32,((base+k) as u32)%rows as u32);
    ((anchor as u32+r-back)%r) as usize
}
/// The tile row offset of a query row at `slot`: its distance `base + j` is tile row `offset + j`.
#[inline(always)]
pub fn offset(rows:usize,anchor:usize,slot:usize)->usize {if anchor>=slot {anchor-slot} else {anchor+rows-slot}}
/// Whether a query row with this offset reads the tile (else it reads the plane).
#[inline(always)]
pub fn in_tile(offset:usize)->bool {offset<QUERY_ROWS}
/// The stored row of distance `dist` from a query row at `slot` (`dist` below the rows).
#[inline(always)]
pub fn stored_row(rows:usize,slot:usize,dist:usize)->usize {if dist<=slot {slot-dist} else {slot+rows-dist}}
/// Word `w` of the slice `slice` of stored row `p` of value head `group` in the plane.
#[inline(always)]
pub fn plane_word(s:Shape,group:usize,slice:usize,p:usize,w:usize)->usize {
    let (h,t,half)=(s.heads as usize,s.taps as usize,s.dim as usize/2);
    (p*(h/t)+group)*half+slice*WORDS+w
}
/// The shared place of word `w` of tile row `k`.
#[inline(always)]
pub fn tile_at(k:usize,w:usize)->usize {k*ROW_WORDS+w}
/// The weight of query head `head` at distance `dist` in the weights of query row `row`.
#[inline(always)]
pub fn weight_at(s:Shape,row:usize,head:usize,dist:usize)->usize {
    let (h,r)=(s.heads as usize,s.rows as usize);
    (row*h+head)*r+dist
}
/// Lane `lane`'s terms of one distance, added to its lane chains: `weight(i)` is the weight of the `i`-th query head
/// of the value head, `word(w)` the bits of packed word `w` of the slice of the distance's stored row. The term of
/// channel `c` is the weight times the value read as BF16 (the low half of the word for an even channel), as
/// `block_attention::unit` makes it; heads past `t` are not touched (their weights are not read).
/// Written out word by word on purpose: every chain is named by a fixed place, so the GPU keeps the 48 chains in
/// registers (with a loop over the words they went to the stack: 192 stack bytes, MEASURED slower).
#[inline(always)]
pub fn add_terms(chains:&mut [[f32;CHANNELS];TAPS_MAX],t:usize,weight:&impl Fn(usize)->f32,word:&impl Fn(usize)->u32) {
    let weights=[weight(0),if t>1 {weight(1)} else {0.0},if t>2 {weight(2)} else {0.0}];
    add_word(chains,t,&weights,0,word(0));
    add_word(chains,t,&weights,1,word(1));
    add_word(chains,t,&weights,2,word(2));
    add_word(chains,t,&weights,3,word(3));
    add_word(chains,t,&weights,4,word(4));
    add_word(chains,t,&weights,5,word(5));
    add_word(chains,t,&weights,6,word(6));
    add_word(chains,t,&weights,7,word(7));
}
/// The terms of packed word `w` (channels `2w` and `2w + 1`) for the heads below `t`.
#[inline(always)]
fn add_word(chains:&mut [[f32;CHANNELS];TAPS_MAX],t:usize,weights:&[f32;TAPS_MAX],w:usize,bits:u32) {
    let (low,high)=(f32::from_bits(bits<<16),f32::from_bits(bits&0xffff_0000));
    for (i,chain) in chains.iter_mut().enumerate() {
        if i<t {chain[2*w]+=weights[i]*low;chain[2*w+1]+=weights[i]*high;}
    }
}
/// The place of the block sums chain of (query row `q` of the block, head `i` of the value head, channel `c` of the
/// slice).
#[inline(always)]
pub fn total_at(q:usize,i:usize,c:usize)->usize {(q*TAPS_MAX+i)*CHANNELS+c}
/// Output word `o` of a block (below `(end - first) * t * CHANNELS`): its query row in the block, its head in the value
/// head and its channel in the slice; neighbouring `o` are neighbouring channels.
#[inline(always)]
pub fn output_of(t:usize,o:usize)->(usize,usize,usize) {let c=o%CHANNELS;let rest=o/CHANNELS;(rest/t,rest%t,c)}
/// Where the word of (query row `row`, query head `head`, channel `channel`) lies: the units of a row are `h * d`.
#[inline(always)]
pub fn out_index(s:Shape,row:usize,head:usize,channel:usize)->usize {
    let (h,d)=(s.heads as usize,s.dim as usize);
    (row*h+head)*d+channel
}
/// The output word from the chain of block sums `total` and the gate word, as `unit` writes it.
#[inline(always)]
pub fn word(total:f32,gate:f32)->f32 {
    use crate::fixed_math::{rn,sigmoid};
    rn(rn(total)*sigmoid(rn(gate)))
}
/// The join of the lane sums of one word as a warp makes it: `down(v, distance)` is the value `v` of the lane
/// `distance` above (the GPU's shuffle); for the distances 16, 8, 4, 2, 1 each lane adds it to its own, so lane 0
/// ends with `block_ops::join` of the 32 lane sums.
#[inline(always)]
pub fn join_down(value:f32,down:&impl Fn(f32,u32)->f32)->f32 {
    let mut v=value;
    v+=down(v,16);
    v+=down(v,8);
    v+=down(v,4);
    v+=down(v,2);
    v+=down(v,1);
    v
}
/// The joins of the lane chains of one query row (`join_down` of each), handed to `done(i, c, sum)` for head `i` below
/// `t` and channel `c`, head by head and channel by channel. Written out chain by chain on purpose (see `add_terms`).
#[inline(always)]
pub fn join_chains(chains:&[[f32;CHANNELS];TAPS_MAX],t:usize,down:&impl Fn(f32,u32)->f32,done:&mut impl FnMut(usize,usize,f32)) {
    if t>0 {join_row(&chains[0],0,down,done);}
    if t>1 {join_row(&chains[1],1,down,done);}
    if t>2 {join_row(&chains[2],2,down,done);}
}
#[inline(always)]
fn join_row(row:&[f32;CHANNELS],i:usize,down:&impl Fn(f32,u32)->f32,done:&mut impl FnMut(usize,usize,f32)) {
    done(i,0,join_down(row[0],down));
    done(i,1,join_down(row[1],down));
    done(i,2,join_down(row[2],down));
    done(i,3,join_down(row[3],down));
    done(i,4,join_down(row[4],down));
    done(i,5,join_down(row[5],down));
    done(i,6,join_down(row[6],down));
    done(i,7,join_down(row[7],down));
    done(i,8,join_down(row[8],down));
    done(i,9,join_down(row[9],down));
    done(i,10,join_down(row[10],down));
    done(i,11,join_down(row[11],down));
    done(i,12,join_down(row[12],down));
    done(i,13,join_down(row[13],down));
    done(i,14,join_down(row[14],down));
    done(i,15,join_down(row[15],down));
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape(heads:u32,dim:u32,taps:u32,rows:u32)->Shape {Shape {op:Op::WindowValues,heads,dim,taps,rows,cols:0,epsilon:0.0,bound:0.0}}
    /// The bits of each word, every NaN as one word: which NaN payload an add keeps is not fixed (an optimized host build
    /// may swap the operands of an add; the GPU makes its own NaN), so NaN words count as equal NaN, their payloads aside.
    fn bits(x:&[f32])->Vec<u32> {x.iter().map(|v|canon(*v)).collect()}
    fn canon(v:f32)->u32 {if v.is_nan() {0x7fc0_0000} else {v.to_bits()}}
    /// The staged entry, block by block and phase by phase as the GPU runs it: the tile of each distance block, then
    /// warp `w` takes the query rows w, w + 4, .. of the block; lane `l` its lane chains; the lanes joined as the
    /// shuffles join them; lane 0 adds the block sum to the chain in the block's shared words.
    fn staged(s:Shape,rows:usize,x:[&[f32];4])->Vec<f32> {
        let (blocks,threads)=stage(s,rows,SHARED).unwrap();
        let (h,d,t,r)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize);
        let mut out=vec![f32::from_bits(0x7fc0_dead);rows*h*d];
        let warps=threads/LANES;
        for block in 0..blocks {
            let (group,slice,first)=block_of(s,block);
            let end=end_of(rows,first);
            let anchor=anchor(s,end,x[3]);
            let most=most_live(s,first,end,x[3]);
            let mut tile=vec![f32::from_bits(0x7fa0_beef);TILE_WORDS];
            let mut totals=vec![0.0f32;TOTAL_WORDS];
            let mut base=0;
            while base<most {
                for i in 0..tile_count(base,most)*WORDS {let (k,w)=(i/WORDS,i%WORDS);tile[tile_at(k,w)]=x[1][plane_word(s,group,slice,tile_row(r,anchor,base,k),w)];}
                for warp in 0..warps {
                    let mut q=warp;
                    while first+q<end {
                        let row=first+q;
                        if let Some((n,slot))=live(s,row,x[3]).filter(|&(n,_)|base<n) {
                            let left=(n-base).min(BLOCK);
                            let at=offset(r,anchor,slot);
                            let mut lanes=vec![[[0.0f32;CHANNELS];TAPS_MAX];LANES];
                            for (lane,chains) in lanes.iter_mut().enumerate() {
                                let mut j=lane;
                                while j<left {
                                    let dist=base+j;
                                    let word=|w:usize|if in_tile(at) {tile[tile_at(at+j,w)].to_bits()} else {x[1][plane_word(s,group,slice,stored_row(r,slot,dist),w)].to_bits()};
                                    add_terms(chains,t,&|i|x[0][weight_at(s,row,group*t+i,dist)],&word);
                                    j+=LANES;
                                }
                            }
                            for i in 0..t {
                                for c in 0..CHANNELS {
                                    // The shuffle tree over the lanes: lane l gets lane l + distance (its own past 31).
                                    let mut v:Vec<f32>=lanes.iter().map(|chains|chains[i][c]).collect();
                                    let mut distance=LANES/2;
                                    while distance>0 {v=(0..LANES).map(|l|v[l]+if l+distance<LANES {v[l+distance]} else {v[l]}).collect();distance/=2;}
                                    let lane_sums:[f32;LANES]=core::array::from_fn(|l|lanes[l][i][c]);
                                    assert_eq!(canon(v[0]),canon(crate::block_ops::join(lane_sums)));
                                    totals[total_at(q,i,c)]+=v[0];
                                }
                            }
                        }
                        q+=warps;
                    }
                }
                base+=BLOCK;
            }
            for o in 0..(end-first)*t*CHANNELS {
                let (q,i,c)=output_of(t,o);
                let (row,head,channel)=(first+q,group*t+i,slice*CHANNELS+c);
                out[out_index(s,row,head,channel)]=match live(s,row,x[3]) {None=>f32::NAN,Some(_)=>word(totals[total_at(q,i,c)],x[2][out_index(s,row,head,channel)])};
            }
        }
        out
    }
    fn by_units(s:Shape,rows:usize,x:[&[f32];4])->Vec<f32> {
        let units=(s.heads*s.dim) as usize;
        let mut out=vec![0.0f32;rows*units];
        for (index,word) in out.iter_mut().enumerate() {crate::block_rows::unit(s,rows,SHARED,index,x,&[],core::slice::from_mut(word));}
        out
    }
    #[test] fn window_values_staged_at_rows_give_the_words_of_the_rows_unit() {
        let mut state=0x2545_f491u32;
        let mut next=||{state^=state<<13;state^=state>>17;state^=state<<5;state};
        let mut checked=0;
        // (heads, dim, taps, stored rows, query rows): taps 1 to 3, views below, at and above a block of distances
        // (one ring of 1,024 and a full view of 2,100), query rows that are no multiple of QUERY_ROWS, the model's
        // 24 heads of 128 at 3 taps, one stored row and one query row.
        for (h,d,t,r,rows) in [(2u32,16u32,1u32,5u32,3usize),(4,32,2,33,40),(6,16,3,64,33),(24,128,3,1024,37),(3,48,3,2100,70),(2,16,2,1,1),(1,16,1,1100,65)] {
            let s=shape(h,d,t,r);let (hu,du,tu,ru)=(h as usize,d as usize,t as usize,r as usize);
            for case in 0..5 {
                assert!(stage(s,rows,SHARED).is_some());
                let rnd=|v:u32,scale:f32|((v>>8) as f32/8_388_608.0-1.0)*scale;
                // A few odd words: signed zeros, infinities and NaN in the weights and the gates.
                let odd=|v:u32,x:f32|match v%89 {0=>-0.0,1=>f32::INFINITY,2=>f32::NEG_INFINITY,3=>f32::NAN,_=>x};
                let weights=(0..rows*hu*ru).map(|_|{let v=next();if v%97==0 {-0.0} else {odd(v>>3,rnd(v,0.5))}}).collect::<Vec<_>>();
                // Packed BF16 pairs; now and then a half that is an infinity or a NaN.
                let plane=(0..ru*(hu/tu)*du/2).map(|_|{let v=next();let w=(v&0x7fbf_7fbf)^0x3f00_3f00;f32::from_bits(match v>>24&63 {0=>w|0x7f80_0000,1=>w|0x0000_7fc0,_=>w})}).collect::<Vec<_>>();
                let gates=(0..rows*hu*du).map(|_|{let v=next();odd(v>>5,rnd(v,4.0))}).collect::<Vec<_>>();
                // Step words of each row. Case 0: one prompt chunk (slots one apart, ring), the live count of a read
                // from position `start`; case 1: random counts and slots (the plane path); case 2: few live distances;
                // case 3: some rows with no count (NaN words) or a slot past the rows; case 4: a chunk whose last row
                // has no count (anchor 0).
                let start=(next() as usize)%(3*ru);
                let steps=(0..rows).flat_map(|row|{let v=next() as usize;let position=start+row;
                    let chunk=[((position+1).min(ru)) as f32,(position%ru) as f32];
                    match case {0=>chunk,1=>{let n=v%(ru+1);[n as f32,((v>>8)%ru) as f32]},2=>[((v%3).min(ru)) as f32,((v>>8)%ru) as f32],
                        3=>if row%5==1 {[0.5,0.0]} else if row%7==2 {[1.0,ru as f32]} else {chunk},_=>if row+1==rows || row%32==31 {[f32::NAN,0.0]} else {chunk}}}).collect::<Vec<_>>();
                let x=[&weights[..],&plane[..],&gates[..],&steps[..]];
                assert_eq!(bits(&staged(s,rows,x)),bits(&by_units(s,rows,x)),"{h} heads of {d}, taps {t}, {r} stored rows, {rows} rows, case {case}");
                checked+=1;
            }
        }
        assert_eq!(checked,35);
    }
    #[test] fn the_written_out_terms_and_joins_name_each_chain_once() {
        // add_terms: chain (i, c) of a head below t gets weight(i) times value c of the row, once; heads past t stay.
        for t in 1..=TAPS_MAX {
            let mut chains=[[0.5f32;CHANNELS];TAPS_MAX];
            let words:[u32;WORDS]=core::array::from_fn(|w|(0x3f80_0000+((2*w as u32+1)<<16))|(2*w as u32+0x3f80));
            add_terms(&mut chains,t,&|i|(i+2) as f32,&|w|words[w]);
            for (i,chain) in chains.iter().enumerate() {
                for (c,&v) in chain.iter().enumerate() {
                    let bits=words[c/2];let value=f32::from_bits(if c%2==0 {bits<<16} else {bits&0xffff_0000});
                    assert_eq!(v.to_bits(),if i<t {(0.5+(i+2) as f32*value).to_bits()} else {0.5f32.to_bits()},"t {t} head {i} channel {c}");
                }
            }
            // join_chains with a shuffle that gives -0.0 (v + -0.0 = v): each chain (i, c) of a head below t once, in order.
            let mut seen=Vec::new();
            join_chains(&chains,t,&|_,_|-0.0,&mut |i,c,sum|seen.push((i,c,sum.to_bits())));
            let expected:Vec<(usize,usize,u32)>=(0..t).flat_map(|i|(0..CHANNELS).map(move |c|(i,c))).map(|(i,c)|(i,c,chains[i][c].to_bits())).collect();
            assert_eq!(seen,expected);
        }
    }
    #[test] fn window_values_are_staged_only_in_the_launch_they_fit() {
        let s=shape(24,128,3,8192);
        assert_eq!(stage(s,1024,SHARED),Some((32*8*8,THREADS)));
        assert_eq!(stage(s,643,SHARED),Some((21*8*8,THREADS)));
        // Another mask, another arm, too many taps, a slice that does not fit the dim, a shape the arm refuses.
        for mask in [0,0b0001,0b0011,0b0110,0b1010,SHARED|crate::block_rows::LIST,SHARED|crate::block_rows::WINDOW] {assert_eq!(stage(s,4,mask),None,"mask {mask}");}
        assert_eq!(stage(Shape {op:Op::WindowScores,..s},4,SHARED),None);
        assert_eq!(stage(shape(8,128,4,64),4,SHARED),None);
        assert_eq!(stage(shape(6,24,3,64),4,SHARED),None);
        assert_eq!(stage(shape(6,8,3,64),4,SHARED),None);
        assert_eq!(stage(Shape {taps:0,..s},4,SHARED),None);
        assert_eq!(stage(shape(10,128,3,64),4,SHARED),None);
        assert_eq!(stage(Shape {rows:0,..s},4,SHARED),None);
        assert_eq!(stage(Shape {rows:(1<<24)+1,..s},4,SHARED),None);
        assert_eq!(stage(Shape {cols:1,..s},4,SHARED),None);
        assert_eq!(stage(Shape {bound:1.0,..s},4,SHARED),None);
        // No heads, no dim, an epsilon (review REVVALUES, LOW 1): the arm refuses them (`block_attention::lengths`).
        assert_eq!(stage(Shape {heads:0,..s},4,SHARED),None);
        assert_eq!(stage(Shape {dim:0,..s},4,SHARED),None);
        assert_eq!(stage(Shape {epsilon:1.0e-6,..s},4,SHARED),None);
        assert_eq!(stage(s,0,SHARED),None);
        assert_eq!(stage(s,crate::block_rows::ROWS_MAX+1,SHARED),None);
        // Each output word of a launch is one word of one block, and each block's words are its own rows.
        let (rows,s)=(70,shape(6,32,3,16));
        let (blocks,_)=stage(s,rows,SHARED).unwrap();
        let mut seen=vec![0u8;rows*6*32];
        for block in 0..blocks {
            let (group,slice,first)=block_of(s,block);
            for o in 0..(end_of(rows,first)-first)*3*CHANNELS {let (q,i,c)=output_of(3,o);seen[out_index(s,first+q,group*3+i,slice*CHANNELS+c)]+=1;}
        }
        assert!(seen.iter().all(|&n|n==1));
        // The tile holds the stored row of each distance of a chunk's rows: tile row offset + j is stored row
        // (slot - base - j) mod rows.
        for (r,anchor,slot,base,j) in [(1024usize,5usize,1000usize,0usize,40usize),(8192,7200,7180,4096,1023),(7,3,1,0,6),(1100,0,1099,1024,75)] {
            let at=offset(r,anchor,slot);
            assert!(in_tile(at) || r<QUERY_ROWS);
            assert_eq!(tile_row(r,anchor,base,at+j),stored_row(r,slot,(base+j)%r));
        }
    }
}
