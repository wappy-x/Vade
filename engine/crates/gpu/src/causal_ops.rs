//! Fixed single-token KDA and dense operations. Shapes come from the layer recipe.
//! Host and device use these same operations and FP32 lane layouts.
use crate::next_math::{exp,bf16};
#[inline(always)]
pub(crate) fn round(x:f32)->f32 {f32::from_bits(bf16(x)<<16)}
#[inline(always)]
pub(crate) fn sigmoid(x:f32)->f32 {1.0/(1.0+exp(-x))}
#[inline(always)]
pub(crate) fn root(x:f32)->f32 {
    #[cfg(target_arch="nvptx64")] {crate::qsa_common::root(x)}
    #[cfg(not(target_arch="nvptx64"))] {x.sqrt()}
}
#[repr(u32)]
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum Op {Conv,Prepare,Step,GatedNorm,SwiGlu,Project,KeyNorm,Pool,PoolScores,Select,Gather,Scores,Softmax,Values,Route,ExpertFold,Nvfp4,StreamMean,Mxfp8Quantize,UnpackKv,StepColumns,StepLayout,ConvChannels,ConvLayout,ProjectRows,PoolChannels,PoolLayout,Mxfp8QuantizePositiveZero,ScoresSlots,ValuesChannels,ScoresIndirect,ValuesIndirect,ScoresTable,ValuesTable,StepPlanar,ConvParts,PackBf16,UnpackBf16,ScoresPacked,ValuesPacked,ProjectRowsBf16,AbsorbQuery,AbsorbScores,AbsorbValues,AbsorbProject}
#[repr(C)]
#[derive(Debug,Clone,Copy)]
pub struct Shape {pub op:Op,pub heads:u32,pub dim:u32,pub taps:u32,pub rows:u32,pub cols:u32,pub epsilon:f32,pub bound:f32}
impl Shape {
    pub fn lengths(self)->Option<([usize;4],usize,usize,usize)> {
        let (h,d,t,r,k)=(self.heads as usize,self.dim as usize,self.taps as usize,self.rows as usize,self.cols as usize);
        if !self.epsilon.is_finite() || !self.bound.is_finite() {return None;}
        let width=h.checked_mul(d)?;
        let (input,weight,units,words)=match self.op {
            Op::Conv if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?,3usize.checked_mul(t-1)?.checked_mul(width)?,0,0],3usize.checked_mul(width)?.checked_mul(t)?,3,width.checked_mul(t)?),
            Op::ConvChannels if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?,3usize.checked_mul(t-1)?.checked_mul(width)?,0,0],3usize.checked_mul(width)?.checked_mul(t)?,3usize.checked_mul(width)?,t),
            Op::ConvLayout if h>0 && d>0 && t>0 => ([3usize.checked_mul(width)?.checked_mul(t)?,0,0,0],0,3usize.checked_mul(width)?.checked_mul(t)?,1),
            Op::Prepare if h>0 && d>0 && d<=1024 && self.bound<0.0 => ([3usize.checked_mul(width)?,width,h,0],h.checked_add(width)?,h,4usize.checked_mul(d)?.checked_add(1)?),
            Op::Step if h>0 && d>0 && d<=1024 => ([h.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,h.checked_mul(d)?.checked_mul(d)?,0,0],0,h,d.checked_mul(d)?.checked_add(d)?),
            // Planar forms: one unit, the same inputs as Step and Conv (Conv with its q, k and v apart), the outputs of ALL heads or planes in
            // two contiguous runs. StepPlanar: every head's state (key-major, h*d*d words), then every head's output vector (h*d words).
            // ConvParts: the three convolution outputs (3*width words), then the three planes' shifted histories (3*(taps-1)*width words).
            Op::StepPlanar if h>0 && d>0 && d<=1024 => ([h.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,h.checked_mul(d)?.checked_mul(d)?,0,0],0,1,h.checked_mul(d.checked_mul(d)?.checked_add(d)?)?),
            Op::ConvParts if h>0 && d>0 && t>0 => ([width,width,width,3usize.checked_mul(t-1)?.checked_mul(width)?],3usize.checked_mul(width)?.checked_mul(t)?,1,3usize.checked_mul(width)?.checked_mul(t)?),
            Op::StepColumns if h>0 && d>0 && d<=1024 => ([h.checked_mul(4usize.checked_mul(d)?.checked_add(1)?)?,width.checked_mul(d)?,0,0],0,width,d.checked_add(1)?),
            Op::StepLayout if h>0 && d>0 && d<=1024 => ([width.checked_mul(d.checked_add(1)?)?,0,0,0],0,width.checked_mul(d.checked_add(1)?)?,1),
            Op::GatedNorm if h>0 && d>0 && d<=1024 && self.epsilon>0.0 => ([width,width,0,0],d,h,d),
            Op::SwiGlu if r>0 && self.bound>0.0 => ([r,r,0,0],0,r,1),
            Op::StreamMean if h>0 && d>0 => ([width,0,0,0],0,d,1),
            Op::Mxfp8Quantize|Op::Mxfp8QuantizePositiveZero if k>0 && k%32==0 && r<=2 => ([k,0,0,0],0,k,1),
            Op::UnpackKv if r>0 && h>0 && d>0 && k>0 && t<=r =>
                ([t.checked_mul(h)?.checked_mul(d.checked_add(k)?)?,0,0,0],0,r.checked_mul(h)?.checked_mul(d.checked_add(k)?)?,1),
            Op::Nvfp4 if r>0 && k>0 && k%16==0 => ([k,r.checked_mul(k/16)?,0,0],r.checked_mul(k)?,r,1),
            Op::Project if r>0 && k>0 => ([k,0,0,0],r.checked_mul(k)?,r,1),
            Op::ProjectRows if r>0 && k>0 && h>0 && h<=4 => ([h.checked_mul(k)?,0,0,0],r.checked_mul(k)?,r,h),
            // ProjectRows with BF16 weights packed two to a word: half the weight words, same widened operand a feature.
            Op::ProjectRowsBf16 if r>0 && k>0 && k%2==0 && h>0 && h<=4 => ([h.checked_mul(k)?,0,0,0],r.checked_mul(k)?/2,r,h),
            Op::ScoresIndirect if h>0 && d>0 && k>0 && r>0 && t<=r => ([h.checked_mul(d)?,t.checked_mul(2)?,0,0],0,h.checked_mul(r)?,1),
            Op::ValuesIndirect if h>0 && d>0 && k>0 && r>0 && t<=r => ([h.checked_mul(r)?,t.checked_mul(2)?,0,0],0,h.checked_mul(d)?,1),
            // Table form: one 16-byte entry (4 words) per live slot and plane; dim is the plane's own width.
            Op::ScoresTable if h>0 && d>0 && r>0 && t>0 && t<=r => ([h.checked_mul(d)?,t.checked_mul(4)?,0,0],0,h.checked_mul(r)?,1),
            Op::ValuesTable if h>0 && d>0 && r>0 && t>0 && t<=r => ([h.checked_mul(r)?,t.checked_mul(4)?,0,0],0,h.checked_mul(d)?,1),
            // Two BF16 patterns a word: feature 2i in the low half, 2i+1 in the high half. `rows` counts packed words.
            Op::PackBf16 if r>0 => ([r.checked_mul(2)?,0,0,0],0,r,1),
            Op::UnpackBf16 if r>0 => ([r,0,0,0],0,r.checked_mul(2)?,1),
            // Table form over packed rows: entry stride counts packed words a head. Scores: dim=key, cols=value. Values swapped.
            Op::ScoresPacked if h>0 && d>0 && k>0 && r>0 && t>0 && t<=r => ([h.checked_mul(d)?,t.checked_mul(4)?,0,0],0,h.checked_mul(r)?,1),
            Op::ValuesPacked if h>0 && d>0 && k>0 && r>0 && t>0 && t<=r => ([h.checked_mul(r)?,t.checked_mul(4)?,0,0],0,h.checked_mul(d)?,1),
            _ if crate::latent_absorb::is_absorb(self.op)=>crate::latent_absorb::lengths(self)?,
            _=>crate::latent_ops::lengths(self)?,
        };
        if input.into_iter().chain([weight,units.checked_mul(words)?]).any(|n|n>u32::MAX as usize || n>isize::MAX as usize/4) {return None;}
        Some((input,weight,units,words))
    }
}
/// Threads per block the warp-per-unit device versions need: four warps.
pub const FAST_THREADS:u32=128;
/// Most picks and most output words of the ExpertFold with a device version (`causal_fast::expert_fold`): the picks one thread
/// holds in registers (the model routes 8), and the largest swept word count (the model folds 4,096 words, 2,048 a rank of two).
pub const FOLD_PICKS:usize=8;
pub const FOLD_WORDS:usize=8192;
/// `dim` of the ExpertFold that rounds its row words (input 2) to BF16 as it reads them, with the cast of the separate round
/// launch (`bf16::rounded`, which keeps a NaN a NaN). One launch then gives the bytes of that round launch followed by the
/// fold. It is a mark, not a size: the fold alone never reads `dim`, and a decoder may leave any layer value there (the
/// generic step shape carries the mixer's head size), so every other `dim` is the fold alone on the generic unit, as before.
/// Only the device version reads the mark: the backend refuses it where the device version is not admitted, a wrapped
/// fold refuses it, and `reference` rounds the rows first.
pub const FOLD_ROUNDS:u32=u32::MAX;
/// Operations with a faster device version (`kernels/src/causal_fast.rs`), by shape only, and ONLY inside the envelope the
/// device sweeps cover (kernel-bench --causal-probe: every limit below is a swept value, with the next value outside swept too and
/// taking the generic path). Each version computes the same bytes as `unit` below. Anything else, and any other block size,
/// keeps the generic one-thread-per-unit path.
pub const fn fast_admitted(s:Shape,threads:u32)->bool {
    let (h,d,r,k)=(s.heads as usize,s.dim as usize,s.rows as usize,s.cols as usize);
    threads==FAST_THREADS && match s.op {
        // Heads and dims are structure, not counts: only the listed values were swept (kernel-bench --causal-probe, specs for
        // `bytes`), each with every pattern, so only those are admitted; the row and key counts are intervals whose edges were swept.
        Op::Prepare|Op::GatedNorm=>swept_heads_wide(h) && swept_dims_wide(d) && h*d<=140000 && (h<100 || matches!(d,1|128|1024)),
        Op::Step|Op::StepColumns|Op::StepLayout=>d==128 && matches!(h,1|2|3|17|63|64),
        Op::StepPlanar=>d==128 && matches!(h,1|2|3|17|32|63|64),
        // Conv writes its final layout in one launch (kernels/src/causal_small.rs) and KeyNorm stages its key in shared memory;
        // only the head, dim and tap values the device sweep covers are admitted.
        Op::Conv|Op::ConvParts=>(matches!(h,1|2|3|8|17|63|64|65) || matches!(s.op,Op::ConvParts) && h==32 && d==128 && s.taps==4)
            && matches!(d,1|3|32|127|128|129) && matches!(s.taps,2|3|4|5),
        Op::KeyNorm=>swept_dims_wide(d),
        // One warp per 32-word group; the column counts the device sweep covers (the model's 4,096 and 8,192, the dense 12,288, and small and odd multiples of 32).
        Op::Mxfp8Quantize|Op::Mxfp8QuantizePositiveZero=>r<=2 && (matches!(k,32|64|96|128|160|1024|2048|4096|8192|12288|16384) || quantize_rows(k)),
        Op::PoolScores=>matches!(d,1|2|31|32|33|127|128|255|256) && matches!(h,1|2|3|31|32|33|63|64|65|100|128) && r<=4096,
        Op::Softmax=>matches!(h,1|2|3|32|33|64|65|100|128) && r>=1 && r<=4096,
        Op::Gather=>matches!(d,1|3|32|33|512|1024) && r>=1 && r<=16384 && k>=1 && k<=4096,
        // Heads are not read by either version. Dims and taps are the swept lists of `swept.rs`, rows an interval with swept edges.
        Op::PoolChannels=>matches!(d,1|2|31|32|33|127|128|129) && matches!(s.taps,1|2|3|4|5|8|127|128|129) && r>=1 && r<=16384,
        // The fold of the routed experts with the round of its rows (dim FOLD_ROUNDS) only: the picks sit in one thread's
        // registers (1 to FOLD_PICKS, each swept), the words are an interval with swept edges, taps is the shared-row flag.
        // The fold alone (any other dim) keeps the generic unit.
        Op::ExpertFold=>s.dim==FOLD_ROUNDS && r>=1 && r<=FOLD_PICKS && k>=1 && k<=FOLD_WORDS && s.taps<=1,
        _=>false,
    }
}
/// Token rows quantized in one launch (`dynamic_projection::quantize_rows`, `mx_rows_launch`): up to `QUANTIZE_ROWS_MAX` rows
/// of 4,096, 8,192 or 12,288 words, so every count is a multiple of 4,096, above the single-row list (which stays as it
/// is: a launch that list admits is unchanged) and up to 1,024 rows of 12,288. A warp owns one 32-word group and reads
/// nothing outside it, so the count only sets how many groups the grid walks. Swept on the device at both ends, at 1,024
/// rows of each width and at odd row counts (kernel-bench --mx-quant: every word = the generic unit of the baseline image).
pub const QUANTIZE_ROWS_MAX:usize=1024;
#[allow(clippy::manual_is_multiple_of)]
pub const fn quantize_rows(k:usize)->bool {k%4096==0 && k>16384 && k<=QUANTIZE_ROWS_MAX*12288}
/// True when the launch of `s` is the warp version on token rows: it stores every word of its output (one lane a word, every
/// group has its warp: `geometry` gives one warp a group), so the backend does not clear the output first. The clear of
/// 1,024 rows of 4,096 words is 16.8 MB a projection. The single-row list keeps its clear: those launches do not change.
pub const fn quantize_rows_written(s:Shape)->bool {
    matches!(s.op,Op::Mxfp8Quantize|Op::Mxfp8QuantizePositiveZero) && quantize_rows(s.cols as usize) && fast_admitted(s,FAST_THREADS)
}
/// The head counts swept for Prepare and GatedNorm (every dim of `swept_dims_wide` with each, up to 140,000 words, and every dim
/// only at 1, 128 and 1,024 from 100 heads on).
const fn swept_heads_wide(h:usize)->bool {matches!(h,1|2|3|4|5|31|32|33|63|64|65|100|127|128)}
const fn swept_dims_wide(d:usize)->bool {matches!(d,1|2|5|31|32|33|64|100|127|128|129|255|1023|1024)}
/// Operations whose every unit writes every word of its output slice (checked word by word on the host reference in the
/// test below, and on the device by running the kernels on a poisoned output against the zero-filled baseline). The host
/// skips the zero fill of their private output: the zero words were always overwritten, and clearing megabytes per layer
/// (the 525,056-block clear of the unpacked KV rows is 1.35 ms a call) is pure bandwidth.
pub const fn writes_every_word(op:Op)->bool {
    matches!(op,Op::ConvParts|Op::StepPlanar|Op::Conv|Op::Prepare|Op::GatedNorm|Op::SwiGlu|Op::KeyNorm|Op::Step|Op::StepColumns|Op::StepLayout|Op::ConvChannels|Op::ConvLayout
        |Op::AbsorbQuery|Op::AbsorbScores|Op::AbsorbValues|Op::AbsorbProject|Op::UnpackKv|Op::PoolScores|Op::Gather|Op::Softmax|Op::ExpertFold|Op::Project|Op::PackBf16|Op::UnpackBf16)
}
/// Launch geometry (blocks, threads) of one causal operation. The default is one thread per unit; the operations with a
/// device version that works a warp or a block per unit get one warp/block per unit. The kernels loop over their units, so
/// any grid is correct; the geometry only decides speed.
pub fn geometry(s:Shape,units:usize)->(u32,u32) {
    let default=(units.div_ceil(128) as u32,128);
    // The ranked route counts in every thread; one expert a thread keeps each count to one pass over the experts.
    if crate::latent_ops::route_ranked(s) {return (1,crate::latent_ops::ROUTE_RANKED_THREADS);}
    if !fast_admitted(s,FAST_THREADS) {return default;}
    match s.op {
        Op::Prepare|Op::GatedNorm=>(units.div_ceil(4) as u32,FAST_THREADS),
        // The pool count grows with the context; a grid rounded up to 64 blocks changes every 256 rows, not every 4, so the
        // recorder's cached pieces (keyed by grid) are rebuilt about as rarely as before. Spare blocks find no unit and leave.
        Op::PoolScores=>(units.div_ceil(4).next_multiple_of(64) as u32,FAST_THREADS),
        Op::Step|Op::StepColumns=>(s.heads,FAST_THREADS),
        Op::StepPlanar=>(s.heads,FAST_THREADS),
        // One thread per (plane, channel); the kernel loops over its units, so the grid only decides speed.
        Op::Conv|Op::ConvParts=>((3*s.heads as usize*s.dim as usize).div_ceil(FAST_THREADS as usize) as u32,FAST_THREADS),
        // One block: the key's two serial chains are one unit.
        Op::KeyNorm=>(1,FAST_THREADS),
        // One warp per group of 32 words, four groups a block.
        Op::Mxfp8Quantize|Op::Mxfp8QuantizePositiveZero=>((s.cols as usize/32).div_ceil(4) as u32,FAST_THREADS),
        Op::StepLayout=>(s.heads.saturating_mul(17),FAST_THREADS),
        Op::Softmax=>(s.heads,FAST_THREADS),
        // Gather's unit count is the live selection size, which grows with the context until 2,051; round the grid as PoolScores does.
        Op::Gather=>(units.div_ceil(4).next_multiple_of(64) as u32,FAST_THREADS),
        // One warp a channel: 128 channels are 32 blocks (the generic version is one block). The grid does not depend on the rows.
        Op::PoolChannels=>(units.div_ceil(4) as u32,FAST_THREADS),
        _=>default,
    }
}
#[repr(C)]
#[derive(Clone,Copy)]
pub struct Arguments {pub input:[u64;4],pub weight:u64,pub output:u64,pub shape:Shape}
pub fn indirect_attention_proven(s:Shape)->bool {
    s.op==Op::ScoresIndirect && matches!(s.heads,32|64) && s.dim==256 && s.cols==256 && (1..=2051).contains(&s.rows) && s.taps>0 && s.taps<=s.rows && (s.taps==s.rows || s.rows==2051)
}
/// Shape-only family of the table form (`attention_table`): heads 1..=128, plane width 256, 1..=4096 rows of which
/// 1..=rows are live. Scores: dim = key width; Values: dim = value width. Launch: ScoresTable `heads*ceil(rows/128)`
/// blocks of 128 threads; ValuesTable `heads*dim/(32*ATTENTION_TABLE_VALUES_FEATURES)` blocks of 32 threads. The host validates every table entry.
pub const ATTENTION_TABLE_SCORES_THREADS:u32=128;
pub const ATTENTION_TABLE_VALUES_THREADS:u32=32;
/// Features a lane of the values kernel owns (a warp covers 32 times this many features of one head).
pub const ATTENTION_TABLE_VALUES_FEATURES:u32=1;
pub const PACKED_VALUES_FEATURES:u32=2;
/// Floats ahead of the key loads at which the scores kernel names the next 128-byte line to the L2 cache.
pub const ATTENTION_TABLE_SCORES_PREFETCH:u32=32;
pub fn attention_table_proven(s:Shape)->bool {
    (matches!(s.op,Op::ScoresTable|Op::ValuesTable) || matches!(s.op,Op::ScoresPacked|Op::ValuesPacked) && s.cols==256)
        && (1..=128).contains(&s.heads) && s.dim==256 && (1..=4096).contains(&s.rows) && (1..=s.rows).contains(&s.taps)
}
pub fn attention_table_blocks(s:Shape)->u32 {
    if matches!(s.op,Op::ScoresTable|Op::ScoresPacked) {s.heads*s.rows.div_ceil(ATTENTION_TABLE_SCORES_THREADS)}
    else if s.op==Op::ValuesPacked {s.heads*(s.dim/(32*PACKED_VALUES_FEATURES))}
    else {s.heads*(s.dim/(32*ATTENTION_TABLE_VALUES_FEATURES))}
}
/// Context rows in one grid bucket: a launch whose extent follows the context takes the grid of the next 256 rows.
pub const BUCKET_ROWS:u32=256;
/// Blocks of 128 threads for the absorbed scores of a decode step (`latent_absorption_scores`): one warp a unit, where a
/// unit is one latent row for four heads. The rows follow the context, so the grid is that of the next 256 rows: a
/// recorded graph piece is keyed by its grids, and this grid then changes as Gather's does, not at every token.
/// Exact: the kernel takes its rows from the shape, and a unit at or past `ceil(heads/4)*rows` names a head past the
/// last one and leaves before it reads or writes.
pub fn absorb_scores_blocks(s:Shape)->u32 {
    let rows=if s.rows==0 {1} else {s.rows};(s.heads.div_ceil(4)*rows.next_multiple_of(BUCKET_ROWS)).div_ceil(4)
}
/// Fixed 32-lane feature tree, independent of token count and topology.
fn square_sum(x:&[f32])->f32 {
    let mut lanes=[0.0f32;32];
    for (lane,sum) in lanes.iter_mut().enumerate() {let mut i=lane;while i<x.len() {*sum+=x[i]*x[i];i+=32;}}
    for distance in [16,8,4,2,1] {let old=lanes;for lane in 0..32-distance {lanes[lane]+=old[lane+distance];}}
    lanes[0]
}
/// Write one complete disjoint output unit. The caller validates all extents.
/// Convolution units are planes. Other KDA units are complete heads.
// This file is also built into the kernel module. `index%2==0` keeps the device code bytes; `is_multiple_of` moved them.
#[allow(clippy::manual_is_multiple_of)]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let (heads,d,taps)=(s.heads as usize,s.dim as usize,s.taps as usize);let width=heads*d;
    match s.op {
        Op::Conv=>{
            let plane=index;let history=(taps-1)*width;
            for ch in 0..width {
                let mut sum=0.0;
                for tap in 0..taps {let v=if tap+1==taps {x[0][plane*width+ch]} else {x[1][plane*history+tap*width+ch]};
                    sum+=v*w[(plane*width+ch)*taps+tap];}
                out[ch]=round(sum*sigmoid(sum));
                for tap in 0..taps-1 {out[width+tap*width+ch]=if tap+2==taps {x[0][plane*width+ch]} else {x[1][plane*history+(tap+1)*width+ch]};}
            }
        },
        Op::ConvChannels=>{
            let plane=index/width;let ch=index%width;let history=(taps-1)*width;let mut sum=0.0;
            for tap in 0..taps {let v=if tap+1==taps {x[0][plane*width+ch]} else {x[1][plane*history+tap*width+ch]};
                sum+=v*w[(plane*width+ch)*taps+tap];}
            out[0]=round(sum*sigmoid(sum));
            for tap in 0..taps-1 {out[tap+1]=if tap+2==taps {x[0][plane*width+ch]} else {x[1][plane*history+(tap+1)*width+ch]};}
        },
        Op::ConvLayout=>{
            let plane=index/(taps*width);let lane=index%(taps*width);let tap=lane/width;let ch=lane%width;
            out[0]=x[0][(plane*width+ch)*taps+tap];
        },
        Op::Prepare=>{
            let base=index*d;let q=&x[0][base..base+d];let k=&x[0][width+base..width+base+d];
            let qnorm=root(square_sum(q)+1e-6);let knorm=root(square_sum(k)+1e-6);let scale=1.0/root(d as f32);
            let rate=exp(w[index]);
            for j in 0..d {out[j]=(q[j]/qnorm)*scale;out[d+j]=k[j]/knorm;out[2*d+j]=x[0][2*width+base+j];
                out[3*d+j]=s.bound*sigmoid(rate*(x[1][base+j]+w[heads+base+j]));}
            out[4*d]=round(sigmoid(x[2][index]));
        },
        Op::Step=>{
            let p=&x[0][index*(4*d+1)..(index+1)*(4*d+1)];let state=&x[1][index*d*d..(index+1)*d*d];
            for key in 0..d {let decay=exp(p[3*d+key]);for value in 0..d {out[key*d+value]=state[key*d+value]*decay;}}
            for value in 0..d {
                let mut prediction=0.0;for key in 0..d {prediction+=out[key*d+value]*p[d+key];}
                let delta=(p[2*d+value]-prediction)*p[4*d];
                for key in 0..d {out[key*d+value]+=p[d+key]*delta;}
                let mut sum=0.0;for key in 0..d {sum+=out[key*d+value]*p[key];}out[d*d+value]=round(sum);
            }
        },
        Op::StepColumns=>{
            // Each value column is independent. Keep every key fold in the
            // original ascending order and round only the final output lane.
            let head=index/d;let value=index%d;let p=&x[0][head*(4*d+1)..(head+1)*(4*d+1)];
            let state=&x[1][head*d*d..(head+1)*d*d];
            for key in 0..d {let decay=exp(p[3*d+key]);out[key]=state[key*d+value]*decay;}
            let mut prediction=0.0;for key in 0..d {prediction+=out[key]*p[d+key];}
            let delta=(p[2*d+value]-prediction)*p[4*d];
            for key in 0..d {out[key]+=p[d+key]*delta;}
            let mut sum=0.0;for key in 0..d {sum+=out[key]*p[key];}out[d]=round(sum);
        },
        Op::StepLayout=>{
            // Restore the public per-head state/output layout by copying bits.
            let head=index/(d*d+d);let lane=index%(d*d+d);
            let (value,key)=if lane<d*d {(lane%d,lane/d)} else {(lane-d*d,d)};
            out[0]=x[0][(head*d+value)*(d+1)+key];
        },
        Op::GatedNorm=>{
            let base=index*d;let row=&x[0][base..base+d];let scale=1.0/root(square_sum(row)/d as f32+s.epsilon);
            for j in 0..d {out[j]=round(((row[j]*scale)*w[j])*sigmoid(x[1][base+j]));}
        },
        Op::SwiGlu=>{let gate=x[0][index].min(s.bound);let up=x[1][index].clamp(-s.bound,s.bound);
            out[0]=round(round(gate*sigmoid(gate))*up);},
        Op::StreamMean=>{let mut sum=0.0;for stream in 0..heads {sum+=x[0][stream*d+index];}out[0]=round(sum/heads as f32);},
        Op::Mxfp8Quantize=>{out[0]=crate::mxfp8_quantize::lane(x[0],index,s.rows);},
        Op::Mxfp8QuantizePositiveZero=>{out[0]=crate::mxfp8_quantize::positive_zero_lane(x[0],index,s.rows);},
        // Packing BF16-rounds each word and keeps its pattern: on an already rounded word bf16() is the identity
        // (fixed point), so pack then unpack returns the producer's exact bits. Feature 2i rides the low half.
        Op::PackBf16=>{out[0]=f32::from_bits(bf16(x[0][2*index])|(bf16(x[0][2*index+1])<<16));},
        Op::UnpackBf16=>{let bits=x[0][index/2].to_bits();out[0]=if index%2==0 {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)};},
        Op::UnpackKv=>{let values=s.cols as usize;let keys=s.rows as usize*heads*d;
            let (row,feature)=if index<keys {(index/d,index%d)} else {let i=index-keys;(i/values,d+i%values)};
            out[0]=if row<taps*heads {x[0][row*(d+values)+feature]} else {0.0};
        },
        Op::Nvfp4=>{let cols=s.cols as usize;let mut sum=0.0;for group in 0..cols/16 {let block=x[1][index*cols/16+group];
            for j in group*16..(group+1)*16 {sum+=(x[0][j]*w[index*cols+j])*block;}}out[0]=round(sum);},
        Op::Project=>{let cols=s.cols as usize;let mut sum=0.0;for j in 0..cols {sum+=x[0][j]*w[index*cols+j];}out[0]=sum;},
        Op::ProjectRows=>{let cols=s.cols as usize;let mut sum=[0.0f32;4];
            for j in 0..cols {let weight=w[index*cols+j];for token in 0..4 {if token<heads {sum[token]+=x[0][token*cols+j]*weight;}}}
            out[..heads].copy_from_slice(&sum[..heads]);},
        Op::ProjectRowsBf16=>{let cols=s.cols as usize;let mut sum=[0.0f32;4];
            for j in 0..cols {let bits=w[(index*cols+j)/2].to_bits();let weight=if j%2==0 {f32::from_bits(bits<<16)} else {f32::from_bits(bits&0xffff_0000)};
                for token in 0..4 {if token<heads {sum[token]+=x[0][token*cols+j]*weight;}}}
            out[..heads].copy_from_slice(&sum[..heads]);},
        #[cfg(not(target_arch="nvptx64"))]
        _ if crate::latent_absorb::is_absorb(s.op)=>crate::latent_absorb::unit(s,index,x,w,out,|a,b|a*b,|a,b|a+b),
        _=>crate::latent_ops::unit(s,index,x,w,out),
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn reference(s:Shape,x:[&[f32];4],w:&[f32])->Result<Vec<f32>,String> {
    if matches!(s.op,Op::ScoresIndirect|Op::ValuesIndirect|Op::ScoresTable|Op::ValuesTable|Op::ScoresPacked|Op::ValuesPacked) {return Err("indirect attention requires checked device row owners".into());}
    // The fold that rounds its rows (FOLD_ROUNDS) is the round launch followed by the fold; `unit` is the fold alone,
    // whatever else `dim` holds.
    if s.op==Op::ExpertFold && s.dim==FOLD_ROUNDS {
        let rounded=x[2].iter().map(|&v|crate::bf16::rounded(v)).collect::<Vec<_>>();
        return reference(Shape {dim:0,..s},[x[0],x[1],&rounded,x[3]],w);
    }
    let (lengths,weight,units,words)=s.lengths().ok_or("invalid causal operation shape")?;
    // Packed words are bit patterns, not floats: a packed word can read as NaN and still be exact. Skip its float check.
    let packed=matches!(s.op,Op::PackBf16|Op::UnpackBf16);
    let packed_weights=packed || matches!(s.op,Op::ProjectRowsBf16|Op::AbsorbQuery|Op::AbsorbProject);
    if x.iter().zip(lengths).any(|(v,n)|v.len()!=n) || w.len()!=weight
        || !packed && x.iter().flat_map(|v|v.iter()).any(|v|!v.is_finite())
        || !packed_weights && w.iter().any(|v|!v.is_finite()) {return Err("causal operation operand extent or value differs".into());}
    let mut out=vec![0.0;units*words];
    if matches!(s.op,Op::StepPlanar|Op::ConvParts) {planar_reference(s,x,w,&mut out);} else {for (i,row) in out.chunks_exact_mut(words).enumerate() {unit(s,i,x,w,row);}}
    if !packed && out.iter().any(|v|!v.is_finite()) {return Err("causal operation output is not finite".into());}Ok(out)
}
/// The planar forms by the per-head and per-plane units they replace: each head (plane) is computed by `unit` of Step (Conv)
/// and its words are placed in the two contiguous runs. These ops have no device generic version; only their fast entries exist.
#[cfg(not(target_arch="nvptx64"))]
fn planar_reference(s:Shape,x:[&[f32];4],w:&[f32],out:&mut [f32]) {
    let (heads,d,taps)=(s.heads as usize,s.dim as usize,s.taps as usize);
    match s.op {
        Op::StepPlanar=>{
            let mut one=vec![0.0f32;d*d+d];
            for head in 0..heads {
                unit(Shape {op:Op::Step,heads:1,..s},0,[&x[0][head*(4*d+1)..(head+1)*(4*d+1)],&x[1][head*d*d..(head+1)*d*d],&[],&[]],w,&mut one);
                out[head*d*d..(head+1)*d*d].copy_from_slice(&one[..d*d]);
                out[heads*d*d+head*d..heads*d*d+(head+1)*d].copy_from_slice(&one[d*d..]);
            }
        },
        _=>{
            let width=heads*d;let history=(taps-1)*width;let packed=[x[0],x[1],x[2]].concat();let mut one=vec![0.0f32;width*taps];
            for plane in 0..3 {
                unit(Shape {op:Op::Conv,..s},plane,[&packed,x[3],&[],&[]],w,&mut one);
                out[plane*width..(plane+1)*width].copy_from_slice(&one[..width]);
                out[3*width+plane*history..3*width+(plane+1)*history].copy_from_slice(&one[width..]);
            }
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn table_attention_family_blocks_and_lengths_are_fixed() {
        let scores=Shape {op:Op::ScoresTable,heads:64,dim:256,cols:256,rows:2051,taps:2051,epsilon:0.0,bound:0.0};
        let values=Shape {op:Op::ValuesTable,..scores};
        assert!(attention_table_proven(scores) && attention_table_proven(values));
        assert_eq!(attention_table_blocks(scores),64*17);assert_eq!(attention_table_blocks(values),64*8);
        assert_eq!(scores.lengths().unwrap(),([64*256,2051*4,0,0],0,64*2051,1));assert_eq!(values.lengths().unwrap(),([64*2051,2051*4,0,0],0,64*256,1));
        for rows in [1u32,2,127,128,129,1152,2051,4095,4096] {assert!(attention_table_proven(Shape {rows,taps:rows,..scores}));}
        assert!(attention_table_proven(Shape {rows:300,taps:1,..scores}) && attention_table_proven(Shape {heads:1,..scores}) && attention_table_proven(Shape {heads:128,..scores}));
        for no in [Shape {rows:0,taps:0,..scores},Shape {rows:4097,taps:4097,..scores},Shape {taps:0,..scores},Shape {rows:10,taps:11,..scores},Shape {heads:0,..scores},Shape {heads:129,..scores},
            Shape {dim:128,..scores},Shape {dim:255,..scores},Shape {op:Op::ScoresIndirect,..scores},Shape {op:Op::Scores,..scores},Shape {op:Op::ValuesIndirect,..scores}] {assert!(!attention_table_proven(no),"{no:?}");}
        assert!(reference(scores,[&[],&[],&[],&[]],&[]).is_err() && reference(values,[&[],&[],&[],&[]],&[]).is_err());
        assert_eq!(ATTENTION_TABLE_SCORES_THREADS,128);assert_eq!(ATTENTION_TABLE_VALUES_THREADS,32);assert_eq!(256%(32*ATTENTION_TABLE_VALUES_FEATURES),0);
    }
    #[test] fn pack_unpack_round_trips_every_bf16_rounded_word() {
        // Independently rounded words: patterns across signs, exponents and zeros, each already BF16.
        let mut input=Vec::new();
        for pattern in [0x0000u32,0x8000,0x3f80,0xbf80,0x0001,0x7f7f,0x0080,0xff7f,0x4049,0xc249,0x322b,0xb21d] {input.push(f32::from_bits(pattern<<16));}
        assert!(input.iter().all(|v|v.to_bits()&0xffff==0));
        let pack=Shape {op:Op::PackBf16,heads:0,dim:0,taps:0,rows:(input.len()/2) as u32,cols:0,epsilon:0.0,bound:0.0};
        let packed=reference(pack,[&input,&[],&[],&[]],&[]).unwrap();
        assert_eq!(packed.len(),input.len()/2);
        for (i,word) in packed.iter().enumerate() {assert_eq!(word.to_bits(),(input[2*i].to_bits()>>16)|(input[2*i+1].to_bits()&0xffff_0000),"word {i}");}
        let unpack=Shape {op:Op::UnpackBf16,rows:packed.len() as u32,..pack};
        let back=reference(unpack,[&packed,&[],&[],&[]],&[]).unwrap();
        assert_eq!(back.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),input.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        // Packing an unrounded word rounds it, the same bf16() the producers' round step applies.
        let plain=[1.1f32,-2.3];let one=reference(Shape {rows:1,..pack},[&plain,&[],&[],&[]],&[]).unwrap();
        assert_eq!(one[0].to_bits(),bf16(plain[0])|(bf16(plain[1])<<16));
        // The packed table family uses its own value width and rejects other widths.
        let scores=Shape {op:Op::ScoresPacked,heads:64,dim:256,cols:256,rows:2051,taps:2051,epsilon:0.0,bound:0.0};
        assert!(attention_table_proven(scores) && attention_table_proven(Shape {op:Op::ValuesPacked,..scores}));
        assert_eq!(PACKED_VALUES_FEATURES,2);
        assert_eq!(attention_table_blocks(scores),64*17);assert_eq!(attention_table_blocks(Shape {op:Op::ValuesPacked,..scores}),64*(256/(32*PACKED_VALUES_FEATURES)));
        assert!(!attention_table_proven(Shape {cols:128,..scores}));
        assert!(reference(scores,[&[],&[],&[],&[]],&[]).is_err());
    }
    #[test] fn project_rows_bf16_equals_the_scalar_chain_on_widened_weights() {
        // Shapes of the brief's sweep: kv_b, wq_b and f_b/g_b local rows and logical twins.
        for (rows,cols,tokens) in [(16384usize,512usize,2usize),(2048,1536,3),(4096,128,4),(8192,512,1),(1024,1536,2),(2048,128,3)] {
            let weight=(0..rows*cols).map(|i|{let hash=(i*2654435761usize)>>16;f32::from_bits((0x3f00|(hash as u32&0x7f)|(((hash>>7)&1) as u32)<<15)<<16)}).collect::<Vec<_>>();
            let packed=weight.as_chunks::<2>().0.iter().map(|p|f32::from_bits((p[0].to_bits()>>16)|(p[1].to_bits()&0xffff_0000))).collect::<Vec<_>>();
            let input=(0..tokens*cols).map(|i|((i%29) as f32-14.0)/32.0).collect::<Vec<_>>();
            let plain=Shape {op:Op::ProjectRows,heads:tokens as u32,dim:0,taps:0,rows:rows as u32,cols:cols as u32,epsilon:0.0,bound:0.0};
            let got=reference(plain,[&input,&[],&[],&[]],&weight).unwrap();
            let bf16_form=reference(Shape {op:Op::ProjectRowsBf16,..plain},[&input,&[],&[],&[]],&packed).unwrap();
            assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),bf16_form.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"{rows}x{cols}x{tokens}");
        }
    }
    #[test] fn indirect_dispatch_is_limited_to_the_device_swept_family() {
        let shape=Shape {op:Op::ScoresIndirect,heads:64,dim:256,cols:256,rows:1,taps:1,epsilon:0.0,bound:0.0};
        for heads in [32,64] {for rows in 1..=2051 {assert!(indirect_attention_proven(Shape {heads,rows,taps:rows,..shape}));}
        for taps in 1..=2051 {assert!(indirect_attention_proven(Shape {heads,rows:2051,taps,..shape}));}}
        for rows in [0,2052,u32::MAX] {assert!(!indirect_attention_proven(Shape {rows,taps:rows,..shape}));}
        assert!(!indirect_attention_proven(Shape {rows:2051,taps:2052,..shape}));
        for rows in 2..2051 {assert!(!indirect_attention_proven(Shape {rows,taps:rows-1,..shape}));}
        for wrong in [Shape {heads:7,..shape},Shape {dim:19,..shape},Shape {cols:255,..shape},Shape {taps:0,..shape},Shape {op:Op::ScoresSlots,..shape}] {assert!(!indirect_attention_proven(wrong));}
        assert!(reference(shape,[&[],&[],&[],&[]],&[]).is_err());
    }
    fn shape(op:Op)->Shape {Shape {op,heads:1,dim:2,taps:2,rows:2,cols:2,epsilon:1e-5,bound:-5.0}}
    #[test]
    fn kv_layout_keeps_slot_head_order_signed_zero_and_padding() {
        let s=Shape {op:Op::UnpackKv,heads:2,dim:2,taps:2,rows:3,cols:3,epsilon:0.0,bound:0.0};
        let input=[1.0,-0.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0];
        let got=reference(s,[&input,&[],&[],&[]],&[]).unwrap();
        let expected=[1.0f32,-0.0,6.0,7.0,11.0,12.0,16.0,17.0,0.0,0.0,0.0,0.0,
            3.0,4.0,5.0,8.0,9.0,10.0,13.0,14.0,15.0,18.0,19.0,20.0,0.0,0.0,0.0,0.0,0.0,0.0];
        assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(Shape {taps:4,..s}.lengths().is_none());
        assert_eq!(reference(Shape {taps:0,..s},[&[],&[],&[],&[]],&[]).unwrap(),vec![0.0;30]);
    }
    #[test]
    fn causal_convolution_keeps_oldest_first_history_and_bf16_outputs() {
        let s=shape(Op::Conv);let x=[1.0,2.0,3.0,4.0,5.0,6.0];let old=[7.0,8.0,9.0,10.0,11.0,12.0];
        let weights=[0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,1.0];
        let v=reference(s,[&x,&old,&[],&[]],&weights).unwrap();
        for plane in 0..3 {for j in 0..2 {let value=x[plane*2+j];assert_eq!(v[plane*4+j].to_bits(),round(value*sigmoid(value)).to_bits());assert_eq!(v[plane*4+2+j],value);}}
        assert!(reference(s,[&x,&old[..5],&[],&[]],&weights).is_err());
    }
    #[test]
    fn independent_convolution_channels_keep_history_and_output_bits() {
        for dim in [1,2,128] {for taps in [1,2,4] {let s=Shape {heads:3,dim,taps,..shape(Op::Conv)};let width=(s.heads*dim) as usize;let t=taps as usize;
            let x=(0..3*width).map(|i|(i as i32%19-9) as f32/8.0).collect::<Vec<_>>();
            let w=(0..3*width*t).map(|i|(i as i32%13-6) as f32/16.0).collect::<Vec<_>>();
            let mut old=(0..3*(t-1)*width).map(|i|if i%7==0 {-0.0} else {(i as i32%17-8) as f32/8.0}).collect::<Vec<_>>();
            for _ in 0..3 {
                let expected=reference(s,[&x,&old,&[],&[]],&w).unwrap();
                let channels=reference(Shape {op:Op::ConvChannels,..s},[&x,&old,&[],&[]],&w).unwrap();
                let got=reference(Shape {op:Op::ConvLayout,..s},[&channels,&[],&[],&[]],&[]).unwrap();
                assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                old=(0..3).flat_map(|p|got[p*t*width+width..(p+1)*t*width].iter().copied()).collect();
            }
        }}
    }
    #[test]
    fn recurrence_reads_immutable_state_and_rounds_only_its_output() {
        let s=shape(Op::Step);let p=[0.25,0.5,0.5,0.25,2.0,4.0,0.0,0.0,0.5];let old=[0.0;4];
        let next=reference(s,[&p,&old,&[],&[]],&[]).unwrap();assert_eq!(&next[..4],&[0.5,1.0,0.25,0.5]);
        assert_eq!(&next[4..],&[0.25,0.5]);assert_eq!(old,[0.0;4]);
        assert_eq!(next,reference(s,[&p,&old,&[],&[]],&[]).unwrap());
        assert_ne!(next,reference(s,[&p,&next[..4],&[],&[]],&[]).unwrap());
    }
    #[test]
    fn independent_recurrence_columns_keep_every_state_and_output_bit() {
        for d in [1,2,8,128] {let s=Shape {heads:3,dim:d,..shape(Op::Step)};let n=d as usize;
            let mut p=vec![0.0;3*(4*n+1)];let mut state=vec![0.0;3*n*n];
            for (i,v) in p.iter_mut().enumerate() {*v=(i as i32%17-8) as f32/32.0;}
            for (i,v) in state.iter_mut().enumerate() {*v=if i%7==0 {-0.0} else {(i as i32%29-14) as f32/64.0};}
            for _ in 0..3 {let expected=reference(s,[&p,&state,&[],&[]],&[]).unwrap();
                let columns=reference(Shape {op:Op::StepColumns,..s},[&p,&state,&[],&[]],&[]).unwrap();
                let got=reference(Shape {op:Op::StepLayout,..s},[&columns,&[],&[],&[]],&[]).unwrap();
                assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                state=(0..3).flat_map(|h|got[h*(n*n+n)..h*(n*n+n)+n*n].iter().copied()).collect();
            }
        }
        assert!(Shape {dim:1025,..shape(Op::StepColumns)}.lengths().is_none());
        assert!(Shape {heads:u32::MAX,..shape(Op::StepLayout)}.lengths().is_none());
    }
    #[test]
    fn sigmoid_beta_rounds_before_recurrence_and_invalid_shapes_refuse() {
        let s=shape(Op::Prepare);let v=reference(s,[&[1.0;6],&[0.0;2],&[0.1],&[]],&[0.0;3]).unwrap();
        assert_eq!(v[8].to_bits(),round(sigmoid(0.1)).to_bits());assert_ne!(v[8].to_bits(),sigmoid(0.1).to_bits());
        let mut bad=s;bad.dim=u32::MAX;assert!(bad.lengths().is_none());bad=s;bad.bound=0.0;assert!(bad.lengths().is_none());
        assert!(reference(s,[&[f32::NAN;6],&[0.0;2],&[0.1],&[]],&[0.0;3]).is_err());
    }
    #[test]
    fn grouped_plain_projection_keeps_each_ascending_feature_fold() {
        for tokens in 1..=4 {for cols in [1,17,128] {let s=Shape {op:Op::ProjectRows,heads:tokens,rows:7,cols,..shape(Op::Project)};
            let x=(0..tokens as usize*cols as usize).map(|i|if i%11==0 {-0.0} else {(i as i32%19-9) as f32/8.0}).collect::<Vec<_>>();
            let w=(0..7*cols as usize).map(|i|(i as i32%13-6) as f32/16.0).collect::<Vec<_>>();
            let got=reference(s,[&x,&[],&[],&[]],&w).unwrap();
            for token in 0..tokens as usize {let k=cols as usize;let expected=reference(Shape {op:Op::Project,..s},[&x[token*k..(token+1)*k],&[],&[],&[]],&w).unwrap();for row in 0..7 {assert_eq!(got[row*tokens as usize+token].to_bits(),expected[row].to_bits());}}
        }}
    }
    #[test]
    fn fast_paths_are_chosen_by_shape_and_block_size_inside_the_swept_envelope_only() {
        let s=|op,heads,dim,rows,cols|Shape {op,heads,dim,taps:2,rows,cols,epsilon:1e-5,bound:-5.0};
        for op in [Op::Prepare,Op::GatedNorm] {
            assert!(fast_admitted(s(op,64,128,0,0),128) && fast_admitted(s(op,128,1024,0,0),128) && fast_admitted(s(op,1,1,0,0),128));
            assert!(!fast_admitted(s(op,129,128,0,0),128) && !fast_admitted(s(op,64,1025,0,0),128) && !fast_admitted(s(op,64,0,0,0),128) && !fast_admitted(s(op,0,128,0,0),128));
            assert!(!fast_admitted(s(op,64,128,0,0),256) && !fast_admitted(s(op,64,128,0,0),64));
        }
        for op in [Op::Step,Op::StepColumns,Op::StepLayout] {
            assert!(fast_admitted(s(op,64,128,0,0),128) && fast_admitted(s(op,1,128,0,0),128));
            assert!(!fast_admitted(s(op,65,128,0,0),128) && !fast_admitted(s(op,64,127,0,0),128) && !fast_admitted(s(op,64,129,0,0),128) && !fast_admitted(s(op,64,128,0,0),32));
        }
        assert!(fast_admitted(s(Op::PoolScores,64,128,1040,0),128) && fast_admitted(s(Op::PoolScores,128,256,4096,0),128) && fast_admitted(s(Op::PoolScores,1,1,0,0),128));
        assert!(!fast_admitted(s(Op::PoolScores,129,128,10,0),128) && !fast_admitted(s(Op::PoolScores,64,257,10,0),128) && !fast_admitted(s(Op::PoolScores,64,128,4097,0),128) && !fast_admitted(s(Op::PoolScores,64,0,10,0),128));
        assert!(fast_admitted(s(Op::Softmax,64,0,2051,0),128) && fast_admitted(s(Op::Softmax,128,0,4096,0),128));
        assert!(!fast_admitted(s(Op::Softmax,129,0,10,0),128) && !fast_admitted(s(Op::Softmax,64,0,4097,0),128) && !fast_admitted(s(Op::Softmax,64,0,0,0),128) && !fast_admitted(s(Op::Softmax,64,0,10,0),64));
        assert!(fast_admitted(s(Op::Gather,0,512,4158,2051),128) && fast_admitted(s(Op::Gather,0,1024,16384,4096),128));
        assert!(!fast_admitted(s(Op::Gather,0,1025,10,10),128) && !fast_admitted(s(Op::Gather,0,512,16385,10),128) && !fast_admitted(s(Op::Gather,0,512,10,4097),128) && !fast_admitted(s(Op::Gather,0,0,10,10),128));
        // Review: heads and dims between the swept values keep the original entry too.
        for op in [Op::Prepare,Op::GatedNorm] {assert!(!fast_admitted(s(op,6,6,0,0),128) && !fast_admitted(s(op,64,130,0,0),128) && !fast_admitted(s(op,7,128,0,0),128) && !fast_admitted(s(op,100,64,0,0),128) && !fast_admitted(s(op,128,255,0,0),128));
            assert!(fast_admitted(s(op,100,128,0,0),128) && fast_admitted(s(op,5,1023,0,0),128) && fast_admitted(s(op,127,1,0,0),128) && !fast_admitted(s(op,128,1024,0,0),64));}
        for op in [Op::Step,Op::StepColumns,Op::StepLayout] {assert!(!fast_admitted(s(op,4,128,0,0),128) && !fast_admitted(s(op,32,128,0,0),128) && fast_admitted(s(op,17,128,0,0),128) && fast_admitted(s(op,63,128,0,0),128));}
        assert!(!fast_admitted(s(Op::PoolScores,30,128,10,0),128) && !fast_admitted(s(Op::PoolScores,32,129,10,0),128) && fast_admitted(s(Op::PoolScores,32,128,10,0),128) && fast_admitted(s(Op::PoolScores,100,255,4096,0),128));
        assert!(!fast_admitted(s(Op::Softmax,4,0,10,0),128) && !fast_admitted(s(Op::Softmax,63,0,10,0),128) && fast_admitted(s(Op::Softmax,100,0,10,0),128));
        assert!(!fast_admitted(s(Op::Gather,0,128,10,10),128) && !fast_admitted(s(Op::Gather,0,2,10,10),128) && fast_admitted(s(Op::Gather,0,3,10,10),128));
        // PoolChannels (READ-SPLIT): the swept dims and taps, rows 1 to 16,384, any heads; every other value keeps the original entry.
        {let p=|h:u32,d:u32,t:u32,r:u32|Shape {op:Op::PoolChannels,heads:h,dim:d,taps:t,rows:r,cols:0,epsilon:0.0,bound:0.0};
            assert!(fast_admitted(p(0,128,4,8192),128) && fast_admitted(p(64,128,128,16384),128) && fast_admitted(p(0,1,1,1),128));
            assert!(!fast_admitted(p(0,128,4,0),128) && !fast_admitted(p(0,128,4,16385),128) && !fast_admitted(p(0,130,4,10),128) && !fast_admitted(p(0,128,6,10),128)
                && !fast_admitted(p(0,128,0,10),128) && !fast_admitted(p(0,128,4,10),64));
            assert_eq!(geometry(p(0,128,4,8192),128),(32,128));
            // The standing rule: every admitted (dims, taps) is a swept one, and the swept rows hold both edges of the admitted interval.
            for d in 0..=300u32 {for t in 0..=300u32 {for r in [0u32,1,2,8192,16384,16385] {
                if fast_admitted(p(0,d,t,r),128) {
                    assert!(crate::swept::POOL_CHANNEL_DIMS.contains(&(d as usize)) && crate::swept::POOL_CHANNEL_TAPS.contains(&(t as usize)),"{d} {t} {r}");
                    assert!((1..=crate::swept::POOL_CHANNEL_ROWS_MAX as u32).contains(&r));
                    let rows=crate::swept::pool_channel_rows(t as usize);assert!(rows.contains(&1) && rows.contains(&crate::swept::POOL_CHANNEL_ROWS_MAX) && rows.contains(&(t as usize)) && rows.contains(&(t as usize+1)));
                }}}}
            for d in crate::swept::POOL_CHANNEL_DIMS {for t in crate::swept::POOL_CHANNEL_TAPS {assert!(fast_admitted(p(0,d as u32,t as u32,1),128) && fast_admitted(p(0,d as u32,t as u32,16384),128));}}
        }
        // The review's examples: each keeps the original entry.
        assert!(!fast_admitted(s(Op::PoolScores,1,4096,1,0),128) && !fast_admitted(s(Op::Softmax,1,0,65536,0),128) && !fast_admitted(s(Op::Gather,0,4096,1,1),128));
        for op in [Op::Route,Op::Project,Op::ProjectRows,Op::Pool,Op::Select,Op::ExpertFold,Op::UnpackKv,Op::SwiGlu] {assert!(!fast_admitted(s(op,64,128,128,128),128));}
        // ExpertFold with the round of its rows (dim FOLD_ROUNDS): 1 to 8 picks, 1 to 8,192 words, with or without the shared row; four warps only.
        // The fold alone keeps the generic unit whatever `dim` holds (0, or a layer's head size left there by the generic step shape).
        {let f=|picks:u32,words:u32,shared:u32,dim:u32|Shape {op:Op::ExpertFold,heads:0,dim,taps:shared,rows:picks,cols:words,epsilon:0.0,bound:0.0};
            let mark=FOLD_ROUNDS;
            for picks in 1..=8u32 {for words in [1u32,2,127,128,129,1024,2048,4096,8192] {for shared in 0..=1 {assert!(fast_admitted(f(picks,words,shared,mark),128),"{picks} {words} {shared}");
                for alone in [0u32,1,2,64,128,256,mark-1] {assert!(!fast_admitted(f(picks,words,shared,alone),128),"{picks} {words} {shared} {alone}");}}}}
            assert!(!fast_admitted(f(0,2048,1,mark),128) && !fast_admitted(f(9,2048,1,mark),128) && !fast_admitted(f(8,0,1,mark),128) && !fast_admitted(f(8,8193,1,mark),128)
                && !fast_admitted(f(8,2048,2,mark),128) && !fast_admitted(f(8,2048,1,mark),64) && !fast_admitted(f(8,2048,1,mark),256));
            // One thread a word, as the generic unit: the grid of the default geometry, with or without the mark.
            for dim in [0,128,mark] {assert_eq!(geometry(f(8,2048,1,dim),2048),(16,128));assert_eq!(geometry(f(8,4096,1,dim),4096),(32,128));assert_eq!(geometry(f(3,1,0,dim),1),(1,128));}
            assert_eq!((FOLD_PICKS,FOLD_WORDS,FOLD_ROUNDS),(8,8192,u32::MAX));
            // The mark changes no extent, and the host reference of the rounding fold is the fold of the rounded rows.
            assert_eq!(f(8,2048,1,mark).lengths(),f(8,2048,1,0).lengths());
            let ids=[3.0f32,0.0,7.0];let weights=[0.5f32,-0.25,1.5];
            let rows=[1.000_001f32,-2.5,3.000_3,1e-3,0.333_333_34,-7.123_456,65_535.9,-0.0,2.000_000_2,5.0,-1.5,0.1,9.75,-3.3,4.4];let shared=[0.25f32,-0.5,1.0,2.0,-4.0];
            let rounded=rows.map(crate::bf16::rounded);assert!(rounded.iter().zip(rows).any(|(a,b)|a.to_bits()!=b.to_bits()));
            let bits=|dim:u32,rows:&[f32]|reference(f(3,5,1,dim),[&ids,&weights,rows,&shared],&[]).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(mark,&rows),bits(0,&rounded));
            // Any other `dim` is the fold alone: a layer's head size in that field neither rounds the rows nor is refused.
            for alone in [1u32,2,128,256,mark-1] {assert_eq!(bits(alone,&rows),bits(0,&rows));assert_eq!(bits(alone,&rounded),bits(0,&rounded));}
            assert_ne!(bits(0,&rows),bits(0,&rounded),"the fixture must tell the two folds apart");
            // The two casts differ on a NaN (the fold's own cast does not keep it): the rounding fold must use the round launch's cast.
            let nan=f32::from_bits(0x7fff_ffff);assert_ne!(round(nan).to_bits(),crate::bf16::rounded(nan).to_bits());
        }
        // Mxfp8Quantize (one warp per group of 32): the swept column counts and planes 0 to 2 only.
        {let q=|op:Op,k:u32,plane:u32|Shape {op,heads:0,dim:0,taps:0,rows:plane,cols:k,epsilon:0.0,bound:0.0};
            for op in [Op::Mxfp8Quantize,Op::Mxfp8QuantizePositiveZero] {
                for k in [32u32,64,96,128,160,1024,2048,4096,8192,12288,16384] {for plane in 0..=2 {assert!(fast_admitted(q(op,k,plane),128),"{k} {plane}");}}
                for k in [0u32,16,31,33,192,224,2080,4128,8160,8224,16416,20000] {assert!(!fast_admitted(q(op,k,0),128),"{k}");}
                // Token rows in one launch: multiples of 4,096 words from 2 rows of 4,096 to 1,024 rows of 12,288, nothing else.
                for k in [5*4096u32,8*4096,257*4096,1024*4096,3*8192,1023*8192,1024*8192,2*12288,511*12288,1024*12288] {for plane in 0..=2 {
                    assert!(fast_admitted(q(op,k,plane),128) && quantize_rows_written(q(op,k,plane)),"{k} {plane}");}}
                for k in [16384+32u32,5*4096+32,18432,1024*4096+2048,1024*12288+32,1025*12288,1024*12288+4096,1<<24] {assert!(!fast_admitted(q(op,k,0),128) && !quantize_rows_written(q(op,k,0)),"{k}");}
                assert!(!fast_admitted(q(op,1024*4096,3),128) && !fast_admitted(q(op,1024*4096,0),64) && !quantize_rows_written(q(op,1024*4096,3)));
                // The single-row list keeps its clear before the launch: only the token-row family skips it.
                for k in [32u32,4096,8192,12288,16384] {assert!(fast_admitted(q(op,k,0),128) && !quantize_rows_written(q(op,k,0)),"{k}");}
                assert!(!writes_every_word(op));
                assert!(!fast_admitted(q(op,8192,3),128) && !fast_admitted(q(op,8192,0),64) && !fast_admitted(q(op,8192,0),256));
            }
        }
        // Conv (single launch to the final layout) and KeyNorm (block with a staged key): only the swept values.
        {let c=|h:u32,d:u32,t:u32|Shape {op:Op::Conv,heads:h,dim:d,taps:t,rows:0,cols:0,epsilon:1e-5,bound:0.0};
            assert!(fast_admitted(c(64,128,4),128) && fast_admitted(c(1,1,2),128) && fast_admitted(c(65,129,5),128) && fast_admitted(c(8,3,3),128));
            assert!(!fast_admitted(c(64,128,4),64) && !fast_admitted(c(64,128,4),256) && !fast_admitted(c(64,128,1),128) && !fast_admitted(c(64,128,6),128)
                && !fast_admitted(c(64,100,4),128) && !fast_admitted(c(4,128,4),128) && !fast_admitted(c(0,128,4),128) && !fast_admitted(c(64,0,4),128) && !fast_admitted(c(66,128,4),128));
            for h in 0..=70u32 {for d in 0..=140u32 {for t in 0..=8u32 {if fast_admitted(c(h,d,t),128) {assert!(h>0 && d>0 && (2..=5).contains(&t));}}}}
            let k=|d:u32|Shape {op:Op::KeyNorm,heads:0,dim:d,taps:0,rows:0,cols:0,epsilon:1e-6,bound:0.0};
            for d in [1u32,2,5,31,32,33,64,100,127,128,129,255,1023,1024] {assert!(fast_admitted(k(d),128),"{d}");assert!(!fast_admitted(k(d),64) && !fast_admitted(k(d),256));}
            for d in [0u32,3,4,6,34,126,130,256,512,1022,1025,2048] {assert!(!fast_admitted(k(d),128),"{d}");}
        }
    }
    #[test]
    fn geometry_covers_every_unit_and_keeps_the_default_elsewhere() {
        let s=|op,heads,dim|Shape {op,heads,dim,taps:2,rows:2,cols:2,epsilon:1e-5,bound:-5.0};
        assert_eq!(geometry(s(Op::Route,64,128),1),(1,128));
        assert_eq!(geometry(s(Op::StepColumns,64,127),8128),(64,128));
        assert_eq!(geometry(s(Op::StepColumns,64,128),8192),(64,128));
        assert_eq!(geometry(s(Op::Step,64,128),64),(64,128));
        assert_eq!(geometry(s(Op::StepLayout,64,128),64*128*129),(64*17,128));
        assert_eq!(geometry(s(Op::StepLayout,64,64),64*64*65),(2080,128));
        assert_eq!(geometry(Shape {taps:4,..s(Op::Conv,64,128)},3),(192,128));
        assert_eq!(geometry(Shape {taps:4,..s(Op::Conv,1,1)},3),(1,128));
        assert_eq!(geometry(Shape {taps:3,..s(Op::Conv,8,3)},3),(1,128));
        for (op,k,blocks) in [(Op::Mxfp8Quantize,32u32,1u32),(Op::Mxfp8QuantizePositiveZero,4096,32),(Op::Mxfp8QuantizePositiveZero,8192,64),(Op::Mxfp8Quantize,12288,96),(Op::Mxfp8Quantize,16384,128),(Op::Mxfp8Quantize,160,2)] {
            assert_eq!(geometry(Shape {op,heads:0,dim:0,taps:0,rows:0,cols:k,epsilon:0.0,bound:0.0},k as usize),(blocks,128));}
        assert_eq!(geometry(Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols:8224,epsilon:0.0,bound:0.0},8224),(65,128));
        assert_eq!(geometry(Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols:1024*4096,epsilon:0.0,bound:0.0},1024*4096),(32768,128));
        assert_eq!(geometry(Shape {op:Op::Mxfp8Quantize,heads:0,dim:0,taps:0,rows:0,cols:1024*12288,epsilon:0.0,bound:0.0},1024*12288),(98304,128));
        assert_eq!(geometry(s(Op::KeyNorm,0,128),1),(1,128));
        assert_eq!(geometry(s(Op::KeyNorm,0,130),1),(1,128));
        assert_eq!(geometry(s(Op::Prepare,64,128),64),(16,128));
        assert_eq!(geometry(s(Op::Prepare,3,128),3),(1,128));
        assert_eq!(geometry(s(Op::GatedNorm,5,128),5),(2,128));
        assert_eq!(geometry(s(Op::PoolScores,64,128),1),(64,128));
        assert_eq!(geometry(s(Op::PoolScores,64,128),1040),(320,128));
        assert_eq!(geometry(s(Op::PoolScores,64,128),1041),(320,128));
        assert_eq!(geometry(s(Op::Softmax,32,128),32),(32,128));
        assert_eq!(geometry(s(Op::Gather,0,512),2051),(576,128));
        assert_eq!(geometry(s(Op::Gather,0,512),15),(64,128));
        assert_eq!(geometry(s(Op::Gather,0,0),2051),(17,128));
    }
    #[test]
    fn planar_forms_equal_the_joined_units_and_write_every_word() {
        let fill=|n:usize,seed:usize|(0..n).map(|j|((j*7+seed*3) as i32%13-6) as f32/8.0+0.0625).collect::<Vec<f32>>();
        let sentinel=0x7fc0_dead_u32;
        // StepPlanar: the states of all heads, then the outputs of all heads, equal Step's per-head [state | output] re-placed.
        for (heads,dim) in [(1usize,2usize),(3,4),(2,5)] {
            let step=Shape {op:Op::Step,heads:heads as u32,dim:dim as u32,taps:0,rows:0,cols:0,epsilon:1e-5,bound:0.0};
            let planar=Shape {op:Op::StepPlanar,..step};
            let (l,_,units,words)=step.lengths().unwrap();let (pl,_,punits,pwords)=planar.lengths().unwrap();
            assert_eq!(l,pl);assert_eq!((punits,pwords),(1,units*words));assert!(writes_every_word(Op::StepPlanar));
            let x0=fill(l[0],1);let x1=fill(l[1],2);
            let per_head=reference(step,[&x0,&x1,&[],&[]],&[]).unwrap();let got=reference(planar,[&x0,&x1,&[],&[]],&[]).unwrap();
            let (d,h)=(dim,heads);let mut expected=vec![f32::from_bits(sentinel);h*(d*d+d)];
            for head in 0..h {
                expected[head*d*d..(head+1)*d*d].copy_from_slice(&per_head[head*(d*d+d)..head*(d*d+d)+d*d]);
                expected[h*d*d+head*d..h*d*d+(head+1)*d].copy_from_slice(&per_head[head*(d*d+d)+d*d..(head+1)*(d*d+d)]);
            }
            assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            assert!(got.iter().all(|v|v.to_bits()!=sentinel));
        }
        // ConvParts: the three convolution outputs, then the three histories, equal Conv's per-plane [output | history] re-placed.
        for (heads,dim,taps) in [(1usize,2usize,2usize),(2,3,4),(1,5,5)] {
            let conv=Shape {op:Op::Conv,heads:heads as u32,dim:dim as u32,taps:taps as u32,rows:0,cols:0,epsilon:1e-5,bound:0.0};
            let parts=Shape {op:Op::ConvParts,..conv};
            let (l,wn,_,_)=conv.lengths().unwrap();let (pl,pw,punits,pwords)=parts.lengths().unwrap();let width=heads*dim;
            assert_eq!(wn,pw);assert_eq!((punits,pwords),(1,3*width*taps));assert_eq!(pl,[width,width,width,l[1]]);assert!(writes_every_word(Op::ConvParts));
            let q=fill(width,1);let k=fill(width,2);let v=fill(width,3);let hist=fill(l[1],4);let weight=fill(wn,5);
            let packed=[q.clone(),k.clone(),v.clone()].concat();
            let per_plane=reference(conv,[&packed,&hist,&[],&[]],&weight).unwrap();let got=reference(parts,[&q,&k,&v,&hist],&weight).unwrap();
            let history=(taps-1)*width;let mut expected=vec![f32::from_bits(sentinel);3*width*taps];
            for plane in 0..3 {
                let row=&per_plane[plane*width*taps..(plane+1)*width*taps];
                expected[plane*width..(plane+1)*width].copy_from_slice(&row[..width]);
                expected[3*width+plane*history..3*width+(plane+1)*history].copy_from_slice(&row[width..]);
            }
            assert_eq!(got.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        // Admission and launch geometry: the swept values only, 128 threads, no generic launch.
        let sp=|h:u32,d:u32|Shape {op:Op::StepPlanar,heads:h,dim:d,taps:0,rows:0,cols:0,epsilon:1e-5,bound:0.0};
        for h in [1u32,2,3,17,32,63,64] {assert!(fast_admitted(sp(h,128),128));assert_eq!(geometry(sp(h,128),1),(h,128));}
        for (h,d) in [(4u32,128u32),(65,128),(64,127),(64,129),(0,128)] {assert!(!fast_admitted(sp(h,d),128));}
        assert!(!fast_admitted(sp(64,128),64) && !fast_admitted(sp(64,128),256));
        let cp=|h:u32,d:u32,t:u32|Shape {op:Op::ConvParts,heads:h,dim:d,taps:t,rows:0,cols:0,epsilon:1e-5,bound:0.0};
        assert!(fast_admitted(cp(64,128,4),128) && !fast_admitted(cp(64,128,6),128) && fast_admitted(cp(32,128,4),128) && !fast_admitted(cp(32,127,4),128) && !fast_admitted(cp(64,128,4),64));
        assert_eq!(geometry(cp(64,128,4),1),(192,128));
    }
    #[test]
    fn listed_operations_write_every_word_of_every_unit() {
        let s=|op,heads,dim,taps,rows,cols|Shape {op,heads,dim,taps,rows,cols,epsilon:1e-5,bound:if op==Op::Prepare {-5.0} else {2.0}};
        let shapes=[s(Op::Prepare,3,5,2,0,0),s(Op::GatedNorm,3,5,2,0,0),s(Op::SwiGlu,0,0,0,7,0),s(Op::KeyNorm,0,6,0,0,0),
            s(Op::Step,3,4,2,0,0),s(Op::StepColumns,3,4,2,0,0),s(Op::StepLayout,3,4,2,0,0),s(Op::Conv,3,4,3,0,0),s(Op::ConvChannels,3,4,3,0,0),s(Op::ConvLayout,3,4,3,0,0),
            s(Op::UnpackKv,2,3,2,4,5),s(Op::PoolScores,3,4,0,6,0),s(Op::Gather,0,5,0,9,6),s(Op::Softmax,3,5,0,7,0),s(Op::ExpertFold,0,0,1,4,6),s(Op::Project,0,0,0,5,7)];
        let mut checked=0;
        for shape in shapes {
            assert!(writes_every_word(shape.op));
            let (lengths,weights,units,words)=shape.lengths().expect("valid shape");
            let mut inputs:Vec<Vec<f32>>=lengths.iter().enumerate().map(|(i,&n)|(0..n).map(|j|((j*7+i*3) as i32%13-6) as f32/8.0+0.0625).collect()).collect();
            if shape.op==Op::Softmax {inputs[1]=vec![5.0];}
            if shape.op==Op::Gather {inputs[1]=(0..lengths[1]).map(|j|(j%9) as f32).collect();inputs[2]=vec![4.0];}
            if shape.op==Op::ExpertFold {inputs[0]=(0..lengths[0]).map(|j|j as f32).collect();}
            let weight:Vec<f32>=(0..weights).map(|j|((j*5) as i32%11-5) as f32/16.0+0.03125).collect();
            let sentinel=f32::from_bits(0x7fc0_dead);
            let mut out=vec![sentinel;units*words];
            let refs=[&inputs[0][..],&inputs[1][..],&inputs[2][..],&inputs[3][..]];
            for (i,row) in out.chunks_exact_mut(words).enumerate() {unit(shape,i,refs,&weight,row);}
            assert!(out.iter().all(|v|v.to_bits()!=0x7fc0_dead),"{:?} leaves a word unwritten",shape.op);
            checked+=1;
        }
        assert_eq!(checked,16);
        for op in [Op::Route,Op::Pool,Op::Select,Op::Scores,Op::Values] {assert!(!writes_every_word(op));}
    }
    #[test] fn absorbed_scores_take_the_grid_of_their_bucket_and_spare_units_name_no_head() {
        for heads in [1u32,4,5,16,32] {for rows in [1u32,3,4,255,256,257,2051,4096] {
            let s=Shape {op:Op::AbsorbScores,heads,dim:64,taps:64,rows,cols:128,epsilon:0.0,bound:0.0};
            let (blocks,exact)=(absorb_scores_blocks(s),(heads.div_ceil(4)*rows).div_ceil(4));
            assert!(blocks>=exact,"{heads} {rows}");assert_eq!(blocks,absorb_scores_blocks(Shape {rows:rows.next_multiple_of(BUCKET_ROWS),..s}));
            // The kernel's own index rule: unit = block*4 + warp, head0 = unit/rows*4. Every unit of a spare block, and every
            // spare unit of the last exact block, names a head at or past `heads`, where the kernel returns.
            for unit in heads.div_ceil(4)*rows..blocks*4 {assert!(unit/rows*4>=heads,"{heads} {rows} {unit}");}
            // Every real unit is launched.
            assert!(blocks*4>=heads.div_ceil(4)*rows);
        }}
        // One bucket for a whole 256 rows, as Gather.
        let s=Shape {op:Op::AbsorbScores,heads:32,dim:64,taps:64,rows:1,cols:128,epsilon:0.0,bound:0.0};
        assert_eq!(absorb_scores_blocks(s),absorb_scores_blocks(Shape {rows:256,..s}));assert!(absorb_scores_blocks(Shape {rows:257,..s})>absorb_scores_blocks(s));
    }
}
