//! Which prompt-row shapes take the data-movement, wrapped-operation, convolution and mHC kernels of lane P's reading rows
//! (`kernels/src/prompt_move.rs`, `prompt_fast.rs`, `mhc_project_tile.rs`), and their launch geometry. By shape only (and, for the
//! vector copy, 16-byte alignment); any other shape keeps `prompt_rows` or `mhc_boundary_project`.
//!
//! RULE (Astra, train 19): the admitted set is a subset of what the device sweep runs, tuple by tuple. Every predicate here names the
//! model's shape (structure values) and a dense token range (1..=GROUP_MAX tokens, the 256-row group of the service and its partial
//! last group); `swept.rs` lists exactly the tuples the sweeps (`kernel-bench --reading-probe list-*`) run, and a host test below
//! enumerates a universe around the predicates and fails when an admitted tuple is missing from that list.
use crate::prompt_ops::{Op,Shape};
use crate::causal_ops::{Op as Causal,Shape as CausalShape};

/// Threads per block of both movement kernels.
pub const MOVE_THREADS:u32=256;
/// Edge of the shared-memory tile of `prompt_place`: 32 output rows (slots) by 32 channels.
pub const TILE:u32=32;
/// Most tokens of a prompt group that take the reading entries; every count from 1 to this is swept.
pub const GROUP_MAX:usize=256;
/// Experts routed per token of the model.
pub const ROUTED:usize=8;

/// `PermuteRows` of the expert inputs: `tokens * ROUTED` rows of 2048 or 4096 words (the hidden width, or the middle width) copied
/// from `tokens` source rows. The vector copy moves four words at a time and needs 16-byte aligned source and output.
pub fn permute_admitted(s:Shape,source:u64,output:u64)->bool {
    let (rows,cols)=(s.rows as usize,s.cols as usize);
    s.op==Op::PermuteRows && matches!(cols,2048|4096) && rows%ROUTED==0 && (1..=GROUP_MAX).contains(&(rows/ROUTED)) && s.first as usize==rows/ROUTED
        && source&15==0 && output&15==0 && s.lengths().is_some()
}
/// One block per output row: the launch is `rows` blocks.
pub fn permute_blocks(s:Shape)->u32 {s.rows}
/// `ExpertPlaceTile` of the expert results on two ranks: `tokens * ROUTED` rows, local rows 1024 or 2048 (output rows of 2048 or
/// 4096 words), `slots = rows * dim`. Tiles are 32 channels wide. The table must come from `prompt_ops::place_table`.
pub fn place_admitted(s:Shape)->bool {
    let rows=s.rows as usize;
    s.op==Op::ExpertPlaceTile && matches!((s.dim,s.taps),(1024,2)|(2048,2)) && rows%ROUTED==0 && (1..=GROUP_MAX).contains(&(rows/ROUTED))
        && s.cols==s.dim*s.taps && s.slots as usize==rows*s.dim as usize && s.lengths().is_some()
}
/// Grid of the place tile: slot tiles on x, channel tiles on y.
pub fn place_grid(s:Shape)->(u32,u32) {(s.rows.div_ceil(TILE),s.cols/TILE)}

/// Threads of a `transpose_smem` block (a 32 x 32 shared-memory tile, 8 rows of the tile a pass).
pub const TRANSPOSE_THREADS:u32=256;
/// The matrices a prompt transposes: the exchange upload and the row transposes see the total row count of a rank-split weight, or the local rows
/// (half of it), and a token count as columns. These are the distinct row counts of the GLM checkpoint's two-dimensional weights and their halves that
/// are whole 4 x 4 tiles and at most 32,768 (`resident-decode-check plan-time`: MODEL MATRIX ROWS), so the set is the model's own. The vocabulary head
/// (154,880 rows, 77,440 local) and every other row count keep the older kernels.
pub const TRANSPOSE_ROWS:[usize;21]=[4,12,16,24,32,64,128,144,256,288,512,768,1024,1536,2048,4096,6144,8192,12288,16384,32768];
/// Token counts of a prompt group that are whole tiles: 4, 8, .. 256.
pub const TRANSPOSE_COLS_MAX:usize=256;
/// `Transpose` of one of those shapes with 16-byte aligned operands. Every admitted (rows, cols) PAIR is a tuple `swept.rs` lists and the device
/// sweep byte-compares (`kernel-bench --transpose-smem` walks `swept::transpose_pairs()`); a copy: the bits are the input's.
pub fn transpose_admitted(s:Shape,input:u64,output:u64)->bool {
    let (r,k)=(s.rows as usize,s.cols as usize);
    s.op==Op::Transpose && TRANSPOSE_ROWS.contains(&r) && k%4==0 && (4..=TRANSPOSE_COLS_MAX).contains(&k) && (input|output)&15==0 && s.lengths().is_some()
}
/// Grid of the transpose tile: column tiles on x, row tiles on y.
pub fn transpose_grid(s:Shape)->(u32,u32) {(s.cols.div_ceil(TILE),s.rows.div_ceil(TILE))}

/// True when the address `ahead` words past word `i` of a row of `wide` words is inside the row. The tile's prefetch hints are
/// only formed then, so no pointer is advanced past the last row's allocation.
pub const fn prefetch_inside(i:usize,ahead:usize,wide:usize)->bool {i+ahead<wide}
/// Tokens of one block of `mhc_project_tile` (blocks of MHC_TOKENS * 32 threads; the direct form needs no producer warps).
pub const MHC_TOKENS:usize=8;
pub const MHC_PRODUCER_WARPS:usize=0;
/// Fewest rows for the token-tiled mHC project: below it the one-block project is as fast or faster (device screen: 64 rows
/// 568 us against 681 us; 256 rows 1,725 us against 1,013 us), because eight tokens share a block and a chain.
pub const MHC_MIN_ROWS:u32=128;
/// The token-tiled mHC project (`kernels/src/mhc_project_tile.rs`) replaces the one-block project for the model's boundary (4 streams
/// of 4096) at 128 to GROUP_MAX rows. Any other valid shape keeps `mhc_boundary_project`.
pub fn mhc_tile_admitted(s:crate::mhc_params::Shape)->bool {
    s.validate().is_ok() && !s.split_project() && (MHC_MIN_ROWS..=GROUP_MAX as u32).contains(&s.rows) && s.streams==4 && s.hidden==4096
}
/// Blocks of the token-tiled project.
pub fn mhc_tile_blocks(rows:u32)->u32 {rows.div_ceil(MHC_TOKENS as u32)}
/// Tokens one thread of `prompt_conv` walks.
pub const CONV_RUN:usize=32;
/// Threads per block of the warp-per-unit and thread-per-element entries below.
pub const FAST_THREADS:u32=128;
/// Wrapped `Route` of the model: one group of 288 experts, 8 chosen, normalized scores (the scale is a value of the same expression).
fn route_admitted(i:CausalShape)->bool {
    i.heads==1 && i.dim==1 && i.taps==1 && i.rows==288 && i.cols==ROUTED as u32 && i.bound.is_finite() && i.bound>0.0
}
/// Wrapped `ExpertFold` of the model: 8 routed rows of 4096 plus the shared expert.
fn fold_admitted(i:CausalShape)->bool {i.rows==ROUTED as u32 && i.cols==4096 && i.taps==1}
/// Head counts of the recurrent prompt kernels (Prepare, GatedNorm, convolution rows): the model's 64 heads on one box, and the
/// 32 a rank owns when two ranks split them (`kda_heads`; the prompt path passes the local count). The kernels take any count;
/// each count here is device-swept (`swept.rs RECURRENT_HEADS`). Four ranks (16 heads) join after their own sweep.
pub const fn recurrent_heads(heads:u32)->bool {matches!(heads,32|64)}
/// The device version for a wrapped causal operation or the convolution rows of the model, by shape only
/// (`kernels/src/prompt_fast.rs`): KDA Prepare and GatedNorm of 32 or 64 heads of 128, Route, ExpertFold, and the 4-tap convolution
/// of 32 or 64 heads x 128 channels, for 1 to GROUP_MAX tokens. Float parameters (gate bound, epsilon, scale) are values of the same
/// expressions and are swept at three values each.
pub fn fast_kernel(s:Shape)->Option<&'static str> {
    s.lengths()?;
    if !(1..=GROUP_MAX).contains(&(s.tokens as usize)) {return None;}
    match s.op {
        Op::Wrapped=>{let i=s.inner;match i.op {
            Causal::Prepare if recurrent_heads(i.heads) && i.dim==128 && i.bound<0.0 && i.bound.is_finite()=>Some("prompt_prepare"),
            Causal::GatedNorm if recurrent_heads(i.heads) && i.dim==128 && i.epsilon>0.0 && i.epsilon.is_finite()=>Some("prompt_gated_norm"),
            Causal::Route if route_admitted(i)=>Some("prompt_route"),
            Causal::ExpertFold if fold_admitted(i)=>Some("prompt_fold"),
            _=>None}},
        Op::ConvRows if recurrent_heads(s.heads) && s.dim==128 && s.taps==4=>Some("prompt_conv"),
        _=>None,
    }
}
/// Blocks of `FAST_THREADS` threads for the kernel `fast_kernel` named.
pub fn fast_blocks(s:Shape)->u32 {
    let (m,i)=(s.tokens as usize,s.inner);
    let blocks=match s.op {
        Op::Wrapped=>match i.op {
            Causal::Prepare|Causal::GatedNorm=>(m*i.heads as usize).div_ceil(4),
            Causal::Route=>m.div_ceil(4),
            _=>(m*i.cols as usize).div_ceil(FAST_THREADS as usize),
        },
        _=>(3*s.heads as usize*s.dim as usize*m.div_ceil(CONV_RUN)).div_ceil(FAST_THREADS as usize),
    };
    blocks.max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swept;
    fn transpose(rows:u32,cols:u32)->Shape {let mut s=Shape::new(Op::Transpose,1).unwrap();s.rows=rows;s.cols=cols;s}
    /// The standing rule for `transpose_smem`, tuple by tuple (Astra, 10:37): the admitted set IS the list of pairs the device sweep walks. Neither
    /// direction is axis sampling: every admitted pair is on the list and every listed pair is admitted.
    #[test]
    fn the_admitted_transposes_are_exactly_the_swept_pairs() {
        let pairs:std::collections::BTreeSet<(usize,usize)>=swept::transpose_pairs().into_iter().collect();
        assert_eq!(pairs.len(),TRANSPOSE_ROWS.len()*(TRANSPOSE_COLS_MAX/4));
        let mut admitted=0usize;
        for r in 0..=32800u32 {for c in 0..=300u32 {if transpose_admitted(transpose(r,c),16,16) {admitted+=1;assert!(pairs.contains(&(r as usize,c as usize)),"admitted ({r},{c}) is not swept");}}}
        assert_eq!(admitted,pairs.len(),"every swept pair is admitted");
        for &(r,c) in &pairs {assert!(transpose_admitted(transpose(r as u32,c as u32),16,16),"swept ({r},{c}) is refused");}
        // Past the universe above: larger rows (the vocabulary head), larger token counts and the old caps are refused; the exchange shape is not.
        for r in [32804u32,40000,65536,65540,77440,154880,262144] {for c in [4u32,64,256] {assert!(!transpose_admitted(transpose(r,c),16,16),"({r},{c})");}}
        for r in [4u32,64,32768] {for c in [260u32,512,1024,1028] {assert!(!transpose_admitted(transpose(r,c),16,16),"({r},{c})");}}
        assert!(transpose_admitted(transpose(32768,256),16,16) && transpose_admitted(transpose(16384,256),16,16));
        // Not whole tiles, zero, misaligned operands.
        assert!(!transpose_admitted(transpose(6,64),16,16) && !transpose_admitted(transpose(64,6),16,16) && !transpose_admitted(transpose(0,64),16,16) && !transpose_admitted(transpose(64,0),16,16));
        assert!(!transpose_admitted(transpose(64,64),4,16) && !transpose_admitted(transpose(64,64),16,8));
        assert_eq!(transpose_grid(transpose(32768,256)),(8,1024));
        assert_eq!(transpose_grid(transpose(36,36)),(2,2));
    }
    fn permute(rows:u32,cols:u32)->Shape {let mut s=Shape::new(Op::PermuteRows,1).unwrap();s.rows=rows;s.cols=cols;s.first=rows/ROUTED as u32;s}
    fn place(tokens:u32,dim:u32,taps:u32)->Shape {let rows=tokens*ROUTED as u32;let mut s=Shape::new(Op::ExpertPlaceTile,1).unwrap();s.rows=rows;s.cols=dim*taps;s.dim=dim;s.taps=taps;s.slots=rows*dim;s}
    fn wrapped(op:Causal,heads:u32,dim:u32,taps:u32,rows:u32,cols:u32,eps:f32,bound:f32,tokens:usize)->Shape {Shape::wrapped(CausalShape {op,heads,dim,taps,rows,cols,epsilon:eps,bound},tokens).unwrap()}
    fn conv(tokens:usize,heads:u32,dim:u32,taps:u32)->Shape {let mut s=Shape::new(Op::ConvRows,tokens).unwrap();s.heads=heads;s.dim=dim;s.taps=taps;s}
    fn mhc(rows:u32,streams:u32,hidden:u32)->crate::mhc_params::Shape {crate::mhc_params::Shape {rows,streams,hidden,iterations:20,rms_eps:1e-6,eps:1e-6}}
    #[test]
    fn the_model_shapes_are_admitted_and_everything_near_them_keeps_the_generic_kernel() {
        assert!(permute_admitted(permute(2048,4096),256,512) && permute_admitted(permute(8,2048),0,0) && permute_admitted(permute(1000*0+2048,2048),0,0));
        assert!(!permute_admitted(permute(4,4096),0,0) && !permute_admitted(permute(2056,4096),0,0) && !permute_admitted(permute(2048,512),0,0) && !permute_admitted(permute(2048,4100),0,0));
        assert!(!permute_admitted(permute(2048,4096),260,512) && !permute_admitted(permute(2048,4096),256,516) && !permute_admitted(permute(0,4096),0,0));
        let mut wrong=permute(2048,4096);wrong.first=255;assert!(!permute_admitted(wrong,0,0));
        let mut other=permute(8,2048);other.op=Op::Transpose;assert!(!permute_admitted(other,0,0));
        assert!(place_admitted(place(256,1024,2)) && place_admitted(place(1,2048,2)) && place_admitted(place(100,1024,2)));
        assert!(!place_admitted(place(257,1024,2)) && !place_admitted(place(2,32,1)) && !place_admitted(place(2,1024,4)) && !place_admitted(place(0,1024,2)) && !place_admitted(place(2,1000,2)));
        let mut wrong=place(4,1024,2);wrong.slots+=1;assert!(!place_admitted(wrong));
        let mut old=place(8,1024,2);old.op=Op::ExpertPlace;assert!(!place_admitted(old));
        let mut odd=place(3,1024,2);odd.rows=4;odd.slots=4*1024;assert!(!place_admitted(odd));
        assert_eq!(place_grid(place(256,1024,2)),(64,64));assert_eq!(permute_blocks(permute(2048,4096)),2048);
    }
    #[test]
    fn prefetch_addresses_are_formed_only_inside_the_row() {
        // rows=128, streams=4, hidden=4096: wide 16384; the loop runs i = 0, 16, .., 16368 and the last hint that is formed is at 16368.
        let wide=16384;assert!(prefetch_inside(0,640,wide) && prefetch_inside(15743,640,wide) && !prefetch_inside(15744,640,wide) && !prefetch_inside(16368,640,wide));
        let formed=(0..wide).step_by(16).filter(|&i|prefetch_inside(i,640,wide)).map(|i|i+640).max().unwrap();assert_eq!(formed,16368);assert!(formed<wide);
        assert!(!prefetch_inside(0,640,640) && prefetch_inside(0,640,641) && !prefetch_inside(0,640,100));
    }
    #[test]
    fn the_mhc_tile_takes_the_models_boundary_from_128_rows() {
        assert!(mhc_tile_admitted(mhc(256,4,4096)) && mhc_tile_admitted(mhc(128,4,4096)) && mhc_tile_admitted(mhc(200,4,4096)));
        assert!(!mhc_tile_admitted(mhc(127,4,4096)) && !mhc_tile_admitted(mhc(257,4,4096)) && !mhc_tile_admitted(mhc(8,4,4096)) && !mhc_tile_admitted(mhc(1,4,4096)) && !mhc_tile_admitted(mhc(65536,4,4096)));
        assert!(!mhc_tile_admitted(mhc(256,3,4096)) && !mhc_tile_admitted(mhc(256,5,4096)) && !mhc_tile_admitted(mhc(256,4,2048)) && !mhc_tile_admitted(mhc(256,4,4095)) && !mhc_tile_admitted(mhc(256,0,4096)));
        assert_eq!(mhc_tile_blocks(256),32);assert_eq!(mhc_tile_blocks(129),17);assert_eq!(mhc_tile_blocks(128),16);
    }
    #[test]
    fn the_wrapped_and_convolution_shapes_of_the_model_take_their_kernels() {
        let prepare=wrapped(Causal::Prepare,64,128,0,0,0,0.0,-5.0,256);
        assert_eq!(fast_kernel(prepare),Some("prompt_prepare"));assert_eq!(fast_blocks(prepare),4096);
        let gated=wrapped(Causal::GatedNorm,64,128,0,0,0,1e-6,0.0,256);
        assert_eq!(fast_kernel(gated),Some("prompt_gated_norm"));assert_eq!(fast_blocks(gated),4096);
        // Two ranks split the 64 heads: each rank's 32 take the same kernels (half the blocks).
        let half=wrapped(Causal::Prepare,32,128,0,0,0,0.0,-5.0,256);assert_eq!(fast_kernel(half),Some("prompt_prepare"));assert_eq!(fast_blocks(half),2048);
        let half=wrapped(Causal::GatedNorm,32,128,0,0,0,1e-6,0.0,256);assert_eq!(fast_kernel(half),Some("prompt_gated_norm"));assert_eq!(fast_blocks(half),2048);
        assert_eq!(fast_kernel(conv(256,32,128,4)),Some("prompt_conv"));assert_eq!(fast_blocks(conv(256,32,128,4)),3*4096*8/128);
        for h in 0..=130u32 {assert_eq!(recurrent_heads(h),swept::RECURRENT_HEADS.contains(&(h as usize)),"heads {h}");}
        for no in [wrapped(Causal::Prepare,129,128,0,0,0,0.0,-5.0,2),wrapped(Causal::Prepare,6,6,0,0,0,0.0,-5.0,2),wrapped(Causal::Prepare,4,5,0,0,0,0.0,-5.0,16),wrapped(Causal::Prepare,16,128,0,0,0,0.0,-5.0,16),wrapped(Causal::Prepare,33,128,0,0,0,0.0,-5.0,16),wrapped(Causal::Prepare,64,128,0,0,0,0.0,-5.0,257),
            wrapped(Causal::GatedNorm,64,127,0,0,0,1e-6,0.0,16),wrapped(Causal::GatedNorm,64,128,0,0,0,1e-6,0.0,0+257)] {assert_eq!(fast_kernel(no),None);}
        let route=wrapped(Causal::Route,1,1,1,288,8,0.0,2.5,256);
        assert_eq!(fast_kernel(route),Some("prompt_route"));assert_eq!(fast_blocks(route),64);
        for no in [wrapped(Causal::Route,1,1,1,300,8,0.0,2.5,16),wrapped(Causal::Route,1,1,1,288,10,0.0,2.5,16),wrapped(Causal::Route,2,1,1,288,8,0.0,2.5,16),wrapped(Causal::Route,1,2,1,288,8,0.0,2.5,16),wrapped(Causal::Route,1,1,0,288,8,0.0,2.5,16),wrapped(Causal::Route,1,1,1,288,8,0.0,2.5,257)] {assert_eq!(fast_kernel(no),None);}
        let fold=wrapped(Causal::ExpertFold,0,0,1,8,4096,0.0,0.0,256);
        assert_eq!(fast_kernel(fold),Some("prompt_fold"));assert_eq!(fast_blocks(fold),8192);
        for no in [wrapped(Causal::ExpertFold,0,0,1,9,4096,0.0,0.0,16),wrapped(Causal::ExpertFold,0,0,1,8,4000,0.0,0.0,16),wrapped(Causal::ExpertFold,0,0,0,8,4096,0.0,0.0,16),wrapped(Causal::ExpertFold,0,0,1,4,4096,0.0,0.0,16)] {assert_eq!(fast_kernel(no),None);}
        assert_eq!(fast_kernel(conv(256,64,128,4)),Some("prompt_conv"));assert_eq!(fast_blocks(conv(256,64,128,4)),3*8192*8/128);
        for no in [conv(16,64,128,9),conv(16,64,128,0),conv(16,64,128,3),conv(16,16,128,4),conv(16,33,128,4),conv(16,64,127,4),conv(257,64,128,4)] {assert_eq!(fast_kernel(no),None);}
        assert_eq!(fast_kernel(wrapped(Causal::StreamMean,4,8,0,0,0,0.0,0.0,3)),None);
    }
    /// The standing rule: every tuple a predicate admits, over a universe that holds the model shapes, the previously admitted shapes
    /// and their neighbours, is a tuple the device sweep list (`swept.rs`, the lists `kernel-bench --reading-probe list-*` iterates) runs.
    #[test]
    fn every_admitted_tuple_is_in_the_device_sweep_list() {
        use std::collections::HashSet;
        let permute_list:HashSet<_>=swept::permute_cases().into_iter().collect();
        let place_list:HashSet<_>=swept::place_cases().into_iter().collect();
        let prepare_list:HashSet<_>=swept::prepare_cases().into_iter().collect();
        let route_list:HashSet<_>=swept::route_cases().into_iter().collect();
        let fold_list:HashSet<_>=swept::fold_cases().into_iter().collect();
        let conv_list:HashSet<_>=swept::conv_cases().into_iter().collect();
        let mhc_list:HashSet<_>=swept::mhc_cases().into_iter().collect();
        let mut seen=[0usize;8];
        for tokens in 0..=300usize {
            for cols in [4usize,8,512,1020,2047,2048,2049,4095,4096,4100,8192] {
                for rows in [tokens*ROUTED,tokens*ROUTED+1,tokens*ROUTED+4] {
                    let mut s=Shape::new(Op::PermuteRows,1).unwrap();s.rows=rows as u32;s.cols=cols as u32;s.first=tokens as u32;
                    if permute_admitted(s,0,0) {seen[0]+=1;assert!(permute_list.contains(&(rows,cols,tokens)),"permute {rows} {cols} {tokens}");}
                }
            }
            for (d,t) in [(1024usize,2usize),(2048,2),(32,1),(32,2),(512,4),(1024,1),(2048,1),(4096,1),(1024,4)] {
                for rows in [tokens*ROUTED,tokens*ROUTED+4] {
                    let mut s=Shape::new(Op::ExpertPlaceTile,1).unwrap();s.rows=rows as u32;s.dim=d as u32;s.taps=t as u32;s.cols=(d*t) as u32;s.slots=(rows*d) as u32;
                    if place_admitted(s) {seen[1]+=1;assert!(place_list.contains(&(rows/ROUTED,d,t)),"place {rows} {d} {t}");}
                }
            }
            for (h,d) in [(64usize,128usize),(32,128),(16,128),(33,128),(64,127),(128,128),(6,6),(4,5),(1,1),(64,129)] {
                let p=wrapped(Causal::Prepare,h as u32,d as u32,0,0,0,0.0,-5.0,tokens.max(1));
                if tokens>0 && fast_kernel(p)==Some("prompt_prepare") {seen[2]+=1;assert!(prepare_list.contains(&(tokens,h,d,true)),"prepare {tokens} {h} {d}");}
                let g=wrapped(Causal::GatedNorm,h as u32,d as u32,0,0,0,1e-6,0.0,tokens.max(1));
                if tokens>0 && fast_kernel(g)==Some("prompt_gated_norm") {seen[2]+=1;assert!(prepare_list.contains(&(tokens,h,d,false)),"gated {tokens} {h} {d}");}
            }
            for (r,k,t) in [(288usize,8usize,1usize),(288,8,0),(288,9,1),(256,8,1),(512,64,1),(64,8,1),(2,1,1)] {
                let s=wrapped(Causal::Route,1,1,t as u32,r as u32,k as u32,0.0,2.5,tokens.max(1));
                if tokens>0 && fast_kernel(s)==Some("prompt_route") {seen[3]+=1;assert!(route_list.contains(&(tokens,r,k,t)),"route {tokens} {r} {k} {t}");}
            }
            for (r,k,t) in [(8usize,4096usize,1usize),(8,4096,0),(8,4000,1),(7,4096,1),(1,1,1),(8,128,1),(4,4096,1)] {
                let s=wrapped(Causal::ExpertFold,0,0,t as u32,r as u32,k as u32,0.0,0.0,tokens.max(1));
                if tokens>0 && fast_kernel(s)==Some("prompt_fold") {seen[4]+=1;assert!(fold_list.contains(&(tokens,r,k,t)),"fold {tokens} {r} {k} {t}");}
            }
            for (h,d,t) in [(64usize,128usize,4usize),(32,128,4),(16,128,4),(33,128,4),(64,128,3),(8,128,4),(2,64,4),(1,32,1),(3,11,2),(64,127,4)] {
                if tokens>0 && fast_kernel(conv(tokens,h as u32,d as u32,t as u32))==Some("prompt_conv") {seen[5]+=1;assert!(conv_list.contains(&(tokens,h,d,t)),"conv {tokens} {h} {d} {t}");}
            }
            for (st,hd) in [(4usize,4096usize),(4,2048),(3,4096),(2,2),(1,4)] {
                if mhc_tile_admitted(mhc(tokens as u32,st as u32,hd as u32)) {seen[6]+=1;assert!(mhc_list.contains(&(tokens,st,hd)),"mhc {tokens} {st} {hd}");}
            }
        }
        assert_eq!((swept::GROUP_MAX,swept::MHC_MIN_ROWS,swept::ROUTED),(GROUP_MAX,MHC_MIN_ROWS as usize,ROUTED));
        // The test is not vacuous: every predicate admitted something, and each list is no larger than the admitted set plus the neighbours it also runs.
        assert!(seen[..7].iter().all(|&n|n>0),"{seen:?}");
        assert_eq!(permute_list.len(),2*GROUP_MAX);assert_eq!(place_list.len(),2*GROUP_MAX);assert_eq!(prepare_list.len(),2*swept::RECURRENT_HEADS.len()*GROUP_MAX);assert_eq!(conv_list.len(),swept::RECURRENT_HEADS.len()*GROUP_MAX);assert_eq!(mhc_list.len(),GROUP_MAX-127);
    }
}
