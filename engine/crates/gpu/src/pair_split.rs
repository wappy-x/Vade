//! The pair split (one sum exchange a sublayer): the shapes host and kernels share. Integer layout only.
//!
//! A SECOND matrix (expert down, shared down, dense down, an o projection) is split by INPUT COLUMNS: an owner holds
//! whole column quarters of every output row. One owner holds four quarters, two owners two each, four owners one
//! each. An owner computes the raw quarter accumulators of its quarters (`nv_cols_q4` / `q2` / `q1`: a chain a
//! quarter from +0.0, no join, no epilogue, no round), folds the routed experts inside each quarter
//! (`nv_quarter_fold`, the function of `tc_twin_fold`), and joins its own quarters. Two owners then exchange that
//! partial sum once (`tp_reduce_take`). The order of every sum is fixed by the column quarter and the expert id, so
//! one, two and four owners give the same bits.

/// `tp_reduce_take` modes (low bits of its last parameter; the high 32 bits hold the FP32 scale).
/// This owner's partial is the FIRST operand of the add (it is the lower owner).
pub const REDUCE_LOWER:u64=1;
/// Multiply the sum by the scale (the epilogue of a second matrix with no fold).
pub const REDUCE_SCALE:u64=2;
/// Round the result to BF16: the one BF16 round of the sublayer output.
pub const REDUCE_ROUND:u64=4;
/// Threads of a block of `tp_reduce_take` and `nv_quarter_fold`.
#[cfg(not(target_arch="nvptx64"))]
pub const THREADS:u32=256;
/// The most routed experts of one token row that `nv_quarter_fold` takes.
pub const FOLD_MAX_EXPERTS:u32=16;
/// `QuarterFold::flags`: round the joined sum to BF16 (one owner: the output is the finished word).
pub const FOLD_ROUND:u32=1;
/// `QuarterFold::flags`: NO fold (a second matrix with no experts: an o projection, a dense down). The output is the
/// join of the raw quarter accumulators `[row][quarter]` in `acc`; `ids`, `weights`, `scales` and `shared` are not read.
pub const FOLD_PLAIN:u32=2;
/// `QuarterFold::flags`: the operands that no expert slot indexes (`shared`; `acc` with FOLD_PLAIN) are stored
/// QUARTER-MAJOR, word `[quarter][row]`: the layout that a launch of one job a quarter writes (block FP8 jobs of the
/// role `FP8_QUARTER`, one MXFP8 projection a quarter). Without the flag they are `[row][quarter]` (`nv_cols_q*`).
pub const FOLD_QUARTER_MAJOR:u32=4;
/// `QuarterFold::flags`: `scales` holds one word for EVERY expert of the layer (`pool` words, in id order), and the
/// kernel picks `scales[id]` with the route's id of each slot. In the engine the ids are device values, so the host
/// cannot pick. Without the flag `scales` holds `experts` words in slot order.
pub const FOLD_SCALES_BY_ID:u32=8;
/// One launch of `nv_quarter_fold` for one token row.
/// `ids`, `weights`, `scales`: `experts` FP32 words each, in slot order: the routed expert ids (as the route writes
/// them), their route weights, and their epilogue scales fl(global * activation). `acc`: the raw quarter
/// accumulators `[slot][row][quarter]` (FP32 bits) of the experts' down matrices. `shared`: 0, or the shared
/// expert's raw quarter accumulators `[row][quarter]`. `output`: `rows` FP32 words.
#[repr(C)]
#[derive(Clone,Copy,Debug)]
pub struct QuarterFold {pub ids:u64,pub weights:u64,pub scales:u64,pub acc:u64,pub shared:u64,pub output:u64,pub experts:u32,pub rows:u32,pub quarters:u32,pub flags:u32,
    /// With FOLD_SCALES_BY_ID: the words of `scales` (the layer's experts). Not read otherwise.
    pub pool:u32}
/// Quarters an owner holds when `parts` equal owners split a second matrix: 4, 2 or 1.
#[cfg(not(target_arch="nvptx64"))]
pub const fn quarters_of(parts:usize)->Option<usize> {if parts==1 || parts==2 || parts==4 {Some(4/parts)} else {None}}
/// The entry that writes `quarters` raw quarter accumulators a row.
#[cfg(not(target_arch="nvptx64"))]
pub const fn quarters_entry(quarters:usize)->Option<&'static str> {
    match quarters {4=>Some("nv_cols_q4"),2=>Some("nv_cols_q2"),1=>Some("nv_cols_q1"),_=>None}
}
/// Blocks of a launch of `words` items, one item a thread.
#[cfg(not(target_arch="nvptx64"))]
pub const fn blocks(words:usize)->usize {words.div_ceil(THREADS as usize)}

#[cfg(all(test,not(target_arch="nvptx64")))]
mod tests {
    use super::*;
    #[test]
    fn an_owner_holds_whole_quarters() {
        assert_eq!([1,2,4].map(quarters_of),[Some(4),Some(2),Some(1)]);
        for parts in [0,3,5,6,8] {assert_eq!(quarters_of(parts),None);}
        assert_eq!([4,2,1].map(quarters_entry),[Some("nv_cols_q4"),Some("nv_cols_q2"),Some("nv_cols_q1")]);
        assert_eq!(quarters_entry(3),None);
        assert_eq!((blocks(1),blocks(256),blocks(257),blocks(4096)),(1,1,2,16));
        assert_eq!((core::mem::size_of::<QuarterFold>(),core::mem::offset_of!(QuarterFold,pool)),(72,64));
        // The flags are single bits of one word.
        assert_eq!(FOLD_ROUND|FOLD_PLAIN|FOLD_QUARTER_MAJOR|FOLD_SCALES_BY_ID,15);
    }
}
