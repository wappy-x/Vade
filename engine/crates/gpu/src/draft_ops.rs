//! Dimensions of the draft lanes. Each operation defines all input, output and grid extents.
/// Maximum packed activation rows per single `Panel` or `Panels` launch.
pub const PANEL_INPUTS_MAX:usize=256;
/// Maximum concurrent chats per batched draft Markov step (`FoldPickBatch`, `MarkovPrepBatch`, `MarkovStepBatch`).
pub const DRAFT_BATCH_MAX:usize=32;
/// Power-of-2 warp slabs sharing one 16-row NVFP4 panel inside a 128-thread CTA (`1`, `2`, or `4`).
/// Keeps every panel's slabs in the same CTA so all 4 warps hit L1 on one DRAM pass over up to 64 inputs.
#[inline(always)]
pub const fn panel_slabs(inputs:usize)->usize {
    if inputs<=16 {1} else if inputs<=32 {2} else {4}
}
#[derive(Clone,Copy)]
pub enum DraftOp {
    /// Multiply lane-interleaved NVFP4 panels by up to `PANEL_INPUTS_MAX` activation rows.
    PackActivations(usize),
    /// Rounded residual followed by one finite-status triple per warp.
    CheckedSum(usize),
    HeadFold{rows:usize,hidden:usize,latent:usize,inputs:usize,epsilon:f32,bound:f32,confidence:bool},
    Panel {rows:usize, columns:usize, inputs:usize},
    /// Like `Panel`, gated by a 1-word device stop flag in input 2; appends a 1-word propagated stop flag after the output.
    PanelGated {rows:usize, columns:usize, inputs:usize},
    /// Two or three sibling panels of one input, stored back to back (rows in whole 64-row blocks, an unused
    /// third entry 0): one launch writes their outputs back to back, each as `Panel` writes it.
    Panels {rows:[usize;3], columns:usize, inputs:usize},
    /// Add the bias stored after the panel and the base row, then reduce each 16-token tile.
    /// When `gated`, input 1 appends a 1-word device stop flag and non-zero exits immediately with zero panel reads.
    FoldPick {rows:usize, columns:usize, epsilon:f32, bound:f32, confidence:bool, gated:bool},
    /// Batched `FoldPick` across `batch` chats (`1..=DRAFT_BATCH_MAX`) sharing one read of the fold panel and bias.
    FoldPickBatch {rows:usize, columns:usize, batch:usize, epsilon:f32, bound:f32, confidence:bool},
    /// Finish tile reduction; if `stop_above_sum > 0.0`, writes `1.0` to word 4 when invalid or `sum > stop_above_sum`.
    /// When `gated`, input 1 is a 1-word device stop flag; non-zero writes `[0, 0, 0, 0, 1]` and exits.
    FoldFinish {tiles:usize, confidence:bool, stop_above_sum:f32, gated:bool},
    /// Batched `FoldFinish` across `batch` chats (`1..=DRAFT_BATCH_MAX`), one 128-lane block per chat.
    FoldFinishBatch {batch:usize, tiles:usize, confidence:bool},
    /// First stage of a two-stage parallel reduction over 16-token fold tiles: `blocks` blocks of 128 lanes reduce
    /// `tiles` four-word records `[best, id, valid, sum]` into `blocks` block records for `FoldFinish`.
    FoldReduce {tiles:usize, blocks:usize, confidence:bool, gated:bool},
    /// Batched `FoldReduce` across `batch` chats (`1..=DRAFT_BATCH_MAX`), `blocks` blocks per chat.
    FoldReduceBatch {batch:usize, tiles:usize, blocks:usize, confidence:bool},
    /// Rounded elementwise sum (`rn(x + y)`) for the draft model; any non-finite sum or BF16 overflow writes NaN so
    /// downstream norm/pack/fold checks refuse the proposal without a per-add reduction kernel or host flag join.
    AddRound(usize),
    /// Combine batched `rnn_draft[rows, width]`, `rnn_feat[width]`, and `rnn_bias[width]` into `static_base[rows, width]`.
    MarkovStaticBase {rows:usize, width:usize},
    /// Combine step-major `rnn_draft[depth, batch, width]`, `rnn_feat[batch, width]`, and `rnn_bias[width]`.
    MarkovStaticBaseBatch {batch:usize, depth:usize, width:usize},
    /// Fused Markov step input preparation: unpacks the BF16 embedding row (`anchor` or device `prev_pick`) and packs
    /// `[state, embedding]` directly into `2 * latent / 64` NVFP4 activation tiles.
    MarkovPrep {latent:usize, vocab:usize, anchor:u32, from_pick:bool},
    /// Batched `MarkovPrep` across `batch` chats (`1..=DRAFT_BATCH_MAX`). Input 1 holds `[batch]` anchor IDs (when
    /// `!from_pick`) or `[batch, pick_stride]` device picks (when `from_pick`); input 2 holds `[batch, latent]` states.
    MarkovPrepBatch {batch:usize, latent:usize, vocab:usize, from_pick:bool, pick_stride:usize},
    /// Fused Markov recurrence + activation packing: adds `dyn_out + static_base`, runs the rounded GRU recurrence,
    /// and packs the resulting `latent` row into `latent / 64` NVFP4 activation tiles (`[new_state, packed_latent, (stop)]`).
    MarkovStep {latent:usize, has_state:bool, gated:bool},
    /// Batched `MarkovStep` across `batch` chats (`1..=DRAFT_BATCH_MAX`), writing `[new_states, packed_latents, scratch]`.
    MarkovStepBatch {batch:usize, latent:usize, has_state:bool},
    /// Fused register-only warp-tiled online-softmax attention for the draft model (`width == 128`): one warp per
    /// `(row, head, split)`, grouped by `(kv_head, split)` so warps in a block reuse K/V in L1. When `splits == 1`,
    /// writes the finalized BF16-rounded output `[rows, heads, 128]`; when `splits > 1`, writes raw FP32 partials
    /// `[rows, heads, splits, 132]` (`o[0..128], m, l, pad, pad`) for `FlashAttnMerge`.
    /// When `causal_base > 0`, applies 3-tier block-causal masking (`row + row_off < 8` attends to `0..causal_base`,
    /// `< 12` attends to `0..min(causal_base + 4, keys)`, else `0..keys`) so extending a draft block past 8 keeps
    /// block-1 (`rows 0..8`) identical to an 8-row block with zero attention dilution.
    FlashAttn {rows:usize, heads:usize, kv:usize, width:usize, keys:usize, splits:usize, causal_base:usize, row_off:usize},
    /// Merge `splits` online-softmax partials from `FlashAttn` into the finalized BF16-rounded output `[rows, heads, 128]`.
    FlashAttnMerge {rows:usize, heads:usize, width:usize, splits:usize},
    Gate(usize), Rope{rows:usize,heads:usize,width:usize},
    Scores{rows:usize,heads:usize,kv:usize,width:usize,keys:usize},
    Normalizer{queries:usize,keys:usize},
    Values{rows:usize,heads:usize,kv:usize,width:usize,keys:usize},
    Recurrent(usize), Mix{words:usize,taps:usize},Unpack(usize),
    /// The packed BF16 row `id` of a `[rows][width]` table, widened, with `id` read on the device from word 1 of a
    /// greedy pick `[best, id, flag]` (`Backend::enqueue_argmax_batch_device`). A pick that is not a whole number
    /// under `rows` with flag 1 gives zeros; the host refuses such a pick when it reads the picks.
    UnpackAt{width:usize,rows:usize},
    /// The logits of one Markov step of the draft head, `n` words: with `dot` the fold product, `bias` the fold bias
    /// and `base` the head row of the step, `cap(r(base + r(dot + bias)))` with `r` the BF16 round of `logits_bf16`
    /// (a word that is not finite is kept) and `cap(x) = tanh(rn(x) * epsilon) * bound`, the `Softcap` arm. The steps
    /// it replaces are two `bf16_sum`s and the arm, in that order, so each word is theirs. A sum that is not finite
    /// gives NaN, so the greedy pick of the step flags the row, as the finiteness check of those sums did.
    MarkovLogits{n:usize,epsilon:f32,bound:f32},
    /// Reduce each tile of 1024 logits, then reduce the tile winners.
    PickPartials(usize),PickFinish(usize),
    /// Partial softmax denominators of `n` logits against their greedy pick `[best, id, flag]`: one warp a slice of
    /// `chunk` logits sums `exp(logit - best)` and writes one word a slice (`n.div_ceil(chunk)` words; zeros when the
    /// flag is not 1). The draft reads the top-1 probability `1 / sum` of a step from them; it never picks a token.
    ProbPart{n:usize,chunk:usize},
}
impl DraftOp {
    pub fn layout(self)->Option<([usize;3],usize,usize,[u64;6])> {
        use DraftOp::*;
        let (lens,out,lanes,fields)=match self {
            CheckedSum(n) if n>0 && n<=1<<19=>{
                let warps=n.div_ceil(32);([n,n,0],n+warps*3,warps*32,[18,n,0,0,0,0])}
            HeadFold{rows,hidden,latent,inputs,epsilon,bound,confidence} if (1..=16).contains(&inputs) && epsilon>0.0 && bound>0.0=>{
                let packed=(hidden.checked_mul(inputs)?.checked_add(latent)?/64).checked_mul(18)?;
                let out=rows.checked_mul(inputs-1)?.checked_add(rows.div_ceil(16)*4)?;
                ([panel_words(rows,hidden)?,panel_words(rows,latent)?.checked_add(rows)?,packed],out,rows.next_multiple_of(16)*2,
                    [17,rows,hidden,latent.checked_mul(64)?+inputs*2+usize::from(confidence),epsilon.to_bits() as usize,bound.to_bits() as usize])}
            PackActivations(n) if n>0 && n.is_multiple_of(64)=>([n,0,0],n/64*18,n/16,[16,n/64,0,0,0,0]),
            Panel{rows,columns,inputs} if panel_words(rows,columns).is_some() && (1..=PANEL_INPUTS_MAX).contains(&inputs)=>{
                let slabs=panel_slabs(inputs);
                ([panel_words(rows,columns)?,(columns/64*18).checked_mul(inputs)?,0],rows.checked_mul(inputs)?,rows.next_multiple_of(16).checked_mul(2)?.checked_mul(slabs)?,[13,rows,columns,inputs,0,0])}
            PanelGated{rows,columns,inputs} if panel_words(rows,columns).is_some() && (1..=16).contains(&inputs)=>{
                ([panel_words(rows,columns)?,(columns/64*18).checked_mul(inputs)?,1],rows.checked_mul(inputs)?.checked_add(1)?,rows.next_multiple_of(16)*2,[13,rows,columns,inputs,1,0])}
            Panels{rows,columns,inputs} if (1..=PANEL_INPUTS_MAX).contains(&inputs) && rows[0]>0 && rows[1]>0 && rows.iter().all(|r|r.is_multiple_of(64))=>{
                let mut words=0usize;
                for r in rows.into_iter().filter(|&r|r>0) {words=words.checked_add(panel_words(r,columns)?)?;}
                let total=rows[0].checked_add(rows[1])?.checked_add(rows[2])?;
                let slabs=panel_slabs(inputs);
                ([words,(columns/64*18).checked_mul(inputs)?,0],total.checked_mul(inputs)?,total.checked_mul(2)?.checked_mul(slabs)?,[19,rows[0],columns,inputs,rows[1],rows[2]])}
            FoldPick{rows,columns,epsilon,bound,confidence,gated} if epsilon>0.0 && bound>0.0=>{
                let flags=usize::from(confidence)|(usize::from(gated)<<1);
                ([panel_words(rows,columns)?.checked_add(rows)?,columns/64*18+usize::from(gated),rows],rows.div_ceil(16)*4,rows.next_multiple_of(16)*2,[14,rows,columns,flags,epsilon.to_bits() as usize,bound.to_bits() as usize])}
            FoldPickBatch{rows,columns,batch,epsilon,bound,confidence} if (1..=DRAFT_BATCH_MAX).contains(&batch) && epsilon>0.0 && bound>0.0=>{
                let flags=usize::from(confidence)|((batch-1)<<2);
                let slabs=batch.div_ceil(8);
                ([panel_words(rows,columns)?.checked_add(rows)?,(columns/64*18).checked_mul(batch)?,rows.checked_mul(batch)?],
                    batch.checked_mul(rows.div_ceil(16))?.checked_mul(4)?,rows.next_multiple_of(16).checked_mul(2)?.checked_mul(slabs)?,[14,rows,columns,flags,epsilon.to_bits() as usize,bound.to_bits() as usize])}
            FoldFinish{tiles,confidence,stop_above_sum,gated} if tiles>0 && tiles<=(1<<24)/16 && stop_above_sum>=0.0=>{
                // One block: the pick, four warp picks, then the exponential of each tile when it sums.
                ([tiles*4,usize::from(gated),0],20+if confidence {tiles} else {0},128,[15,tiles,usize::from(confidence),stop_above_sum.to_bits() as usize,usize::from(gated),0])}
            FoldFinishBatch{batch,tiles,confidence} if (1..=DRAFT_BATCH_MAX).contains(&batch) && tiles>0 && tiles<=(1<<24)/16=>{
                let stride=20+if confidence {tiles} else {0};
                ([batch.checked_mul(tiles)?.checked_mul(4)?,0,0],batch.checked_mul(stride)?,batch.checked_mul(128)?,[15,tiles,usize::from(confidence),0,0,batch])}
            FoldReduce{tiles,blocks,confidence,gated} if tiles>0 && tiles<=(1<<24)/16 && blocks>0 && blocks<=128 && blocks<=tiles=>{
                ([tiles*4,usize::from(gated),0],blocks*4,blocks*128,[22,tiles,blocks,usize::from(confidence),usize::from(gated),0])}
            FoldReduceBatch{batch,tiles,blocks,confidence} if (1..=DRAFT_BATCH_MAX).contains(&batch) && tiles>0 && tiles<=(1<<24)/16 && blocks>0 && blocks<=128 && blocks<=tiles=>{
                ([batch.checked_mul(tiles)?.checked_mul(4)?,0,0],batch.checked_mul(blocks)?.checked_mul(4)?,batch.checked_mul(blocks)?.checked_mul(128)?,[22,tiles,blocks,usize::from(confidence),0,batch])}
            AddRound(n) if n>0 && n<=1<<24=>{
                ([n,n,0],n,n.div_ceil(4),[23,n,0,0,0,0])}
            MarkovStaticBase{rows,width} if rows>0 && width>0 && rows.checked_mul(width).is_some_and(|n|n<=1<<24)=>{
                let n=rows*width;([n,width,width],n,n.div_ceil(4),[24,rows,width,0,0,0])}
            MarkovStaticBaseBatch{batch,depth,width} if (1..=DRAFT_BATCH_MAX).contains(&batch) && depth>0 && width>0 && depth.checked_mul(batch).and_then(|n|n.checked_mul(width)).is_some_and(|n|n<=1<<24)=>{
                let n=depth*batch*width;([n,batch*width,width],n,n.div_ceil(4),[24,depth,width,batch,0,0])}
            MarkovPrep{latent,vocab,anchor,from_pick} if latent>0 && latent.is_multiple_of(64) && vocab>0 && vocab<1<<24 && (from_pick || (anchor as usize)<vocab)=>{
                let w1=vocab.checked_mul(latent)?/2;
                let lens=if from_pick {[w1,5,latent]} else {[w1,0,0]};
                (lens,(2*latent/64)*18,(2*latent)/16,[25,latent,vocab,anchor as usize,usize::from(from_pick),0])}
            MarkovPrepBatch{batch,latent,vocab,from_pick,pick_stride} if (1..=DRAFT_BATCH_MAX).contains(&batch) && latent>0 && latent.is_multiple_of(64) && vocab>0 && vocab<1<<24 && (!from_pick || pick_stride>=3)=>{
                let w1=vocab.checked_mul(latent)?/2;
                let lens=if from_pick {[w1,batch.checked_mul(pick_stride)?,batch.checked_mul(latent)?]} else {[w1,batch,0]};
                (lens,batch.checked_mul((2*latent/64)*18)?,batch.checked_mul((2*latent)/16)?,[25,latent,vocab,pick_stride,usize::from(from_pick),batch])}
            MarkovStep{latent,has_state,gated} if latent>0 && latent.is_multiple_of(64)=>{
                let r3=latent.checked_mul(3)?;
                let g=usize::from(gated);
                let lens=[r3+g,r3,if has_state {latent} else {0}];
                (lens,2*latent+(latent/64)*18+g,latent.div_ceil(256)*128,[26,latent,usize::from(has_state),g,0,0])}
            MarkovStepBatch{batch,latent,has_state} if (1..=DRAFT_BATCH_MAX).contains(&batch) && latent>0 && latent.is_multiple_of(64)=>{
                let r3=batch.checked_mul(latent)?.checked_mul(3)?;
                let lens=[r3,r3,if has_state {batch.checked_mul(latent)?} else {0}];
                (lens,batch.checked_mul(2*latent+(latent/64)*18)?,batch.checked_mul(latent.div_ceil(256))?.checked_mul(128)?,[26,latent,usize::from(has_state),0,batch,0])}
            FlashAttn{rows,heads,kv,width,keys,splits,causal_base,row_off} if width==128 && kv>0 && heads>0 && heads%kv==0 && rows>0 && rows<65536 && keys>0 && (1..=32).contains(&splits) && row_off<256 && causal_base<=keys && causal_base<=u32::MAX as usize=>{
                let queries=rows.checked_mul(heads)?;let q=queries.checked_mul(width)?;
                let k=keys.checked_mul(kv)?.checked_mul(width)?;
                let out=if splits==1 {q} else {queries.checked_mul(splits)?.checked_mul(132)?};
                ([q,k,k],out,queries.checked_mul(splits)?.checked_mul(32)?,[20,(1.0/(width as f32).sqrt()).to_bits() as usize,heads,kv,(causal_base<<32)|(row_off<<24)|(splits<<16)|rows,keys])}
            FlashAttnMerge{rows,heads,width,splits} if width==128 && rows>0 && heads>0 && (2..=32).contains(&splits)=>{
                let queries=rows.checked_mul(heads)?;let parts=queries.checked_mul(splits)?.checked_mul(132)?;
                ([parts,0,0],queries.checked_mul(width)?,queries.checked_mul(32)?,[21,heads,rows,splits,0,0])}
            Gate(n)=>([n,n,0],n,n,[0,0,0,0,0,0]),
            Rope{rows,heads,width} if width>0 && width%2==0 && heads>0=>{
                let trig=rows.checked_mul(width)?;let n=trig.checked_mul(heads)?;
                ([n,trig,0],n,n,[1,width,heads,0,0,0])}
            Scores{rows,heads,kv,width,keys}|Values{rows,heads,kv,width,keys} if kv>0 && heads>0 && heads%kv==0 && width>0 && keys>0=>{
                let queries=rows.checked_mul(heads)?;let q=queries.checked_mul(width)?;
                let k=keys.checked_mul(kv)?.checked_mul(width)?;let scores=queries.checked_mul(keys)?;
                if matches!(self,Scores{..}) {([q,k,0],scores,scores,[2,(1.0/(width as f32).sqrt()).to_bits() as usize,heads,kv,width,keys])}
                else {([scores,k,0],q,q,[4,0,heads,kv,width,keys])}}
            Normalizer{queries,keys} if keys>0=>([queries.checked_mul(keys)?,0,0],queries.checked_mul(keys)?,queries.checked_mul(32)?,[3,0,0,0,0,keys]),
            Recurrent(r)=>([r.checked_mul(3)?,r,0],r.checked_mul(2)?,r.checked_mul(2)?,[5,r,0,0,0,0]),
            Mix{words,taps} if taps>0=>([words.checked_mul(taps)?,taps,0],words,words,[6,0,taps,0,0,0]),
            Unpack(n) if n%2==0=>([n/2,0,0],n,n,[7,0,0,0,0,0]),
            UnpackAt{width,rows} if width%2==0 && rows>0 && rows<1<<24=>([rows.checked_mul(width)?/2,3,0],width,width,[8,rows,0,0,0,0]),
            MarkovLogits{n,epsilon,bound} if epsilon>0.0 && bound>0.0=>([n,n,n],n,n,[9,epsilon.to_bits() as usize,bound.to_bits() as usize,0,0,0]),
            PickPartials(width) if width>0 && width<=1<<24=>{
                let tiles=width.div_ceil(1024);([width,0,0],tiles*3,tiles*32,[10,width,0,0,0,0])}
            PickFinish(tiles) if tiles>0 && tiles<=(1<<24)/1024=>([tiles*3,0,0],3,32,[11,tiles,0,0,0,0]),
            ProbPart{n,chunk} if n>0 && chunk>0 && chunk%32==0=>{
                let parts=n.div_ceil(chunk);([n,3,0],parts,parts.checked_mul(32)?,[12,n,chunk,0,0,0])}
            _=>return None,
        };
        (lanes>0 && out<=u32::MAX as usize && lens.iter().all(|&n|n<=u32::MAX as usize)).then_some((lens,out,lanes,fields.map(|v|v as u64)))
    }
}

pub fn panel_words(rows:usize,columns:usize)->Option<usize> {
    if rows==0 || rows>1<<24 || columns==0 || !columns.is_multiple_of(64) {return None;}
    let elements=rows.next_multiple_of(64).checked_mul(columns)?;
    (elements/8).checked_add(elements/64)?.checked_add(2048).filter(|&words|words<=u32::MAX as usize)
}
