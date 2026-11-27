//! Fixed-order latent absorption on the tensor cores: one word function for the four fold operations (query, scores,
//! values, project), in the prompt form (`prompt_ops`, a group of tokens) and in the step form (`causal_ops`, one token).
//! Contract: `engine/docs/tc-dense-contract.md`, section "Plain BF16". CPU twin: `tc_twin_bf16::fold`.
//!
//! Every operation is a matrix product `C[m][n] = sum over k of A[m][k] * B[k][n]` for one OUTER index:
//! | operation | outer | m               | n      | k (the sum)     | A                        | B             |
//! | query     | head  | latent features | tokens | key features    | key weights (BF16)       | query rows    |
//! | scores    | token | selected slots  | heads  | latent features | selected cache rows      | absorbed query|
//! | values    | token | latent features | heads  | selected slots  | selected cache rows      | probabilities |
//! | project   | head  | value features  | tokens | latent features | value weights (BF16)     | mixed latent  |
//! The word of `C[m][n]`: the operands are the high 16 bits of their FP32 words (the weights are BF16 patterns); the
//! terms fill 16-term steps of `mma.sync m16n8k16 bf16` in ascending `k`, the last step filled with +0.0; the steps are
//! cut into four ranges by the step count only; a range is one ascending chain from +0.0; the ranges join as
//! `fl(fl(p0 + p1) + fl(p2 + p3))`; then the operation's last step (one BF16 round, or one FP32 multiply for scores).
//! A word reads its own `m` row of A and its own `n` column of B. The block, the warp, the tile and the token count
//! choose only WHERE a word is computed.
use crate::{causal_ops::{Op as StepOp,Shape as StepShape},prompt_ops::{Op,Shape}};
/// Terms of one tensor-core step.
pub const STEP:usize=16;
/// A block: 8 warps, each one M tile of 16 rows, times 4 N tiles of 8 columns.
pub const WARPS:usize=8;pub const TILES:usize=4;pub const THREADS:u32=256;
pub const M_BLOCK:usize=WARPS*16;pub const N_BLOCK:usize=TILES*8;
/// Steps of one staged chunk of the B operand: 32 steps x 4 tiles x 32 lanes x two packed registers.
pub const CHUNK:usize=32;
pub const STAGE_BYTES:usize=CHUNK*TILES*32*8;
/// Values: the cache row of each of a chunk's `CHUNK * 16` slots, as a word offset.
pub const ID_BYTES:usize=CHUNK*STEP*4;
/// THE SMALL FORM of query and project in the prompt form (`kernels/src/latent_absorption_tile.rs`): up to `SMALL_TOKENS`
/// tokens, each one column of a block's one N tile, staged whole before the sum (`SMALL_STEP_BYTES` a step: 32 lanes x two
/// packed registers). A launch whose staged steps do not fit `STAGE_BYTES` keeps the tile's own body.
pub const SMALL_TOKENS:usize=8;
pub const SMALL_STEP_BYTES:usize=32*8;
/// A slot of a staged chunk past the token's count: its operand is +0.0.
pub const NO_ROW:u32=u32::MAX;
/// The step form of scores and values (one token): a block owns 2 M tiles (32 rows) and 32 columns of N. Warp `w` owns
/// M tile `w / 4` and range `w % 4` of the sum, so the four ranges of a word run side by side and meet in the join.
/// The B operand is staged `STEP_CHUNK` steps of each range at a time (4 ranges x 8 steps fill the stage).
pub const STEP_M:usize=32;pub const STEP_CHUNK:usize=CHUNK/4;
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
pub enum Kind {Query,Scores,Values,Project}
/// One launch. `c`: score words a token and head (prompt form: `first`; step form: `rows`). `rows`: the bound of a
/// selected row ID. `table`: words of one token's selection row (`slots + 1`, the count last), or 0 in the step form,
/// where slot `j` is gathered row `j` and the count is one word.
/// Packed u32 words per head for BF16 probability rows (`AbsorbSoftmaxPacked` -> `AbsorbValuesPackedProb`),
/// padded to a multiple of 4 words (16 bytes) so every head row is 16-byte aligned for vectorized stores/loads.
#[inline(always)]
pub const fn prob_words(c:usize)->usize {c.div_ceil(2).div_ceil(4)*4}
#[derive(Clone,Copy,Debug)]
pub struct Form {pub kind:Kind,pub tokens:usize,pub h:usize,pub d:usize,pub v:usize,pub k:usize,pub c:usize,pub rows:usize,pub table:usize,pub packed:bool,pub packed_prob:bool}
impl Form {
    #[inline(always)]
    pub fn rows(s:Shape)->Option<Self> {
        let kind=match s.op {Op::AbsorbQuery=>Kind::Query,Op::AbsorbScores|Op::AbsorbScoresPacked=>Kind::Scores,Op::AbsorbValues|Op::AbsorbValuesPacked|Op::AbsorbValuesPackedProb=>Kind::Values,Op::AbsorbProject=>Kind::Project,_=>return None};
        Some(Self {kind,tokens:s.tokens as usize,h:s.heads as usize,d:s.dim as usize,v:s.taps as usize,k:s.cols as usize,c:s.first as usize,rows:s.rows as usize,table:s.slots as usize+1,packed:matches!(s.op,Op::AbsorbScoresPacked|Op::AbsorbValuesPacked|Op::AbsorbValuesPackedProb),packed_prob:s.op==Op::AbsorbValuesPackedProb})
    }
    #[inline(always)]
    pub fn step(s:StepShape)->Option<Self> {
        let kind=match s.op {StepOp::AbsorbQuery=>Kind::Query,StepOp::AbsorbScores=>Kind::Scores,StepOp::AbsorbValues=>Kind::Values,StepOp::AbsorbProject=>Kind::Project,_=>return None};
        Some(Self {kind,tokens:1,h:s.heads as usize,d:s.dim as usize,v:s.taps as usize,k:s.cols as usize,c:s.rows as usize,rows:s.rows as usize,table:0,packed:false,packed_prob:false})
    }
    #[inline(always)]
    pub const fn attention(self)->bool {matches!(self.kind,Kind::Scores|Kind::Values)}
    #[inline(always)]
    pub const fn outer(self)->usize {if self.attention() {self.tokens} else {self.h}}
    #[inline(always)]
    pub const fn m(self)->usize {match self.kind {Kind::Query|Kind::Values=>self.k,Kind::Project=>self.v,Kind::Scores=>self.c}}
    #[inline(always)]
    pub const fn n(self)->usize {if self.attention() {self.h} else {self.tokens}}
    /// Blocks of `THREADS` threads: one for each outer index, 128 rows of M and 32 columns of N.
    #[inline(always)]
    pub const fn blocks(self)->usize {self.outer()*self.m().div_ceil(M_BLOCK)*self.n().div_ceil(N_BLOCK)}
    /// (outer, first M row, first N column) of a block. A block past `blocks()` names an outer index past the last.
    #[inline(always)]
    pub const fn block(self,block:usize)->(usize,usize,usize) {
        let (tm,tn)=(self.m().div_ceil(M_BLOCK),self.n().div_ceil(N_BLOCK));
        if tm==0 || tn==0 {return (usize::MAX,0,0);}
        (block/(tm*tn),block/tn%tm*M_BLOCK,block%tn*N_BLOCK)
    }
    /// The step form. Rows of M a block owns: 2 M tiles (32 rows, 4 ranges in parallel across 8 warps) for both
    /// attention and dense steps, maximizing SM wave efficiency on small head/row counts.
    #[inline(always)]
    pub const fn step_m(self)->usize {STEP_M}
    #[inline(always)]
    pub const fn step_blocks(self)->usize {self.outer()*self.m().div_ceil(self.step_m())*self.n().div_ceil(N_BLOCK)}
    /// (outer, first M row, first N column) of a block of the step form. A block past `step_blocks()` names an outer
    /// index past the last.
    #[inline(always)]
    pub const fn step_block(self,block:usize)->(usize,usize,usize) {
        let (tm,tn)=(self.m().div_ceil(self.step_m()),self.n().div_ceil(N_BLOCK));
        if tm==0 || tn==0 {return (usize::MAX,0,0);}
        (block/(tm*tn),block/tn%tm*self.step_m(),block%tn*N_BLOCK)
    }
    /// Terms of one sum. `count`: the token's selected rows (values).
    #[inline(always)]
    pub const fn terms(self,count:usize)->usize {match self.kind {Kind::Query=>self.d,Kind::Values=>count,_=>self.k}}
    /// The word of `B[k][n]` in the first input.
    #[inline(always)]
    pub const fn b(self,outer:usize,k:usize,n:usize)->usize {
        match self.kind {Kind::Query=>(n*self.h+outer)*self.d+k,Kind::Project=>(n*self.h+outer)*self.k+k,Kind::Scores=>(outer*self.h+n)*self.k+k,Kind::Values=>(outer*self.h+n)*self.c+k}
    }
    /// Query and project: the BF16 pattern of `A[m][k]` in the packed weights (`[heads][key rows + value rows][latent]`,
    /// the key rows transposed by the loader).
    #[inline(always)]
    pub const fn weight(self,outer:usize,m:usize,k:usize)->usize {
        if matches!(self.kind,Kind::Query) {outer*(self.d+self.v)*self.k+m*self.d+k} else {(outer*(self.d+self.v)+self.d+m)*self.k+k}
    }
    /// Scores and values: (selection slot, latent feature) of `A[m][k]`.
    #[inline(always)]
    pub const fn latent(self,m:usize,k:usize)->(usize,usize) {if matches!(self.kind,Kind::Scores) {(m,k)} else {(k,m)}}
    /// The output word of `C[m][n]`.
    #[inline(always)]
    pub const fn out(self,outer:usize,m:usize,n:usize)->usize {
        match self.kind {Kind::Query=>(n*self.h+outer)*self.k+m,Kind::Project=>(n*self.h+outer)*self.v+m,Kind::Scores=>(outer*self.h+n)*self.c+m,Kind::Values=>(outer*self.h+n)*self.k+m}
    }
    /// The count word of a token in a selection table (prompt form), or the one count word (step form).
    #[inline(always)]
    pub const fn count_word(self,token:usize)->usize {if self.table==0 {0} else {token*self.table+self.table-1}}
}
/// Steps of `terms` terms, and the steps of range `part` (the cut of `kernels/src/canonical.rs`).
#[inline(always)]
pub const fn steps(terms:usize)->usize {terms.div_ceil(STEP)}
#[inline(always)]
pub const fn range_end(steps:usize,part:usize)->usize {steps*(part+1)/4}
/// `mma.sync m16n8k16`: (row m, column k) of the low half of A register `reg` (0..4) of a lane; the high half is column k + 1.
#[inline(always)]
pub const fn a_place(lane:usize,reg:usize)->(usize,usize) {(lane/4+(reg&1)*8,(lane%4)*2+(reg>>1)*8)}
/// (row k, column n) of the low half of B register `reg` (0..2) of a lane; the high half is row k + 1.
#[inline(always)]
pub const fn b_place(lane:usize,reg:usize)->(usize,usize) {((lane%4)*2+reg*8,lane/4)}
/// (row m, column n) of accumulator register `reg` (0..4) of a lane.
#[inline(always)]
pub const fn c_place(lane:usize,reg:usize)->(usize,usize) {(lane/4+(reg>>1)*8,(lane%4)*2+(reg&1))}
/// Byte offset of a lane's two staged B registers of chunk step `step` and N tile `tile`.
#[inline(always)]
pub const fn stage_at(step:usize,tile:usize,lane:usize)->usize {((step*TILES+tile)*32+lane)*8}
/// The stored word of `C[m][n]` of one outer index: the twin's fold, then the operation's last step.
/// `x` is the first input, `latent` the cache rows, `selected` the selection table (prompt form) or the count word
/// (step form), `w` the packed weights. Project in the prompt form takes the selection table as `selected`.
#[cfg(not(target_arch="nvptx64"))]
pub fn word(f:Form,outer:usize,m:usize,n:usize,x:&[f32],latent:&[f32],selected:&[f32],w:&[f32])->f32 {
    use crate::tc_twin_bf16::{fold,high};
    let pattern=|i:usize|->u16 {let bits=w[i/2].to_bits();if i.is_multiple_of(2) {bits as u16} else {(bits>>16) as u16}};
    let count=if f.attention() {selected[f.count_word(outer)] as usize} else {0};
    let id=|slot:usize|if f.table==0 {slot} else {selected[outer*f.table+slot] as usize};
    if f.kind==Kind::Scores && m>=count {return 0.0;}
    if f.kind==Kind::Project && f.table!=0 && selected[f.count_word(n)]==0.0 {return 0.0;}
    let acc=f32::from_bits(fold(f.terms(count),|k|match f.kind {
        Kind::Query|Kind::Project=>pattern(f.weight(outer,m,k)),
        _=>{let (slot,feature)=f.latent(m,k);let i=id(slot)*f.k+feature;
            if f.packed {(latent[i/2].to_bits()>>(16*(i%2))) as u16} else {high(latent[i])}},
    },|k|if f.packed_prob {let w=x[(outer*f.h+n)*prob_words(f.c)+k/2].to_bits();(w>>(16*(k%2))) as u16} else {high(x[f.b(outer,k,n)])}));
    if f.kind==Kind::Scores {acc*(1.0/crate::causal_ops::root(f.d as f32))} else {crate::latent_absorb::round(acc)}
}
#[cfg(test)] mod tests;
