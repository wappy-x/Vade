//! Dimensions of the draft lanes. Each operation defines all input, output and grid extents.
/// Maximum packed activation rows per single `Panel` or `Panels` launch.
pub const PANEL_INPUTS_MAX:usize=256;
/// Maximum concurrent chats per batched draft Markov step (`FoldPickBatch`, `MarkovPrepBatch`, `MarkovStepBatch`).
pub const DRAFT_BATCH_MAX:usize=32;
/// Rows (all sibling panels together) of a small matrix: its panels get one warp for each 8 or 16 inputs, so every
/// warp makes one pass over its panel. One warp a panel gives such a matrix at most 512 warps, under half of one
/// wave of resident warps on GB10 (48 SMs x 6 blocks x 4 warps = 1,152), so its few warps' latency sets its time
/// (measured: a warp that walks the inputs in 4 waves stalls on its loads 86 percent of its cycles). The CTAs that
/// share a panel read its weights at the same time, from L2 after the first read. The larger matrices (`gate_up`,
/// the vocabulary head) already fill a wave and read from DRAM: their warps stay in the panel's CTA.
pub const PANEL_SPLIT_ROWS_MAX:usize=8192;
/// Inputs of a small matrix up to which a warp takes 8 inputs (the pipelined 1..=8-input product); above it a warp
/// takes 16 inputs (two MMA tiles), which halves the CTAs of the launch.
pub const PANEL_NARROW_INPUTS_MAX:usize=64;
/// How the warps of a launch share a 16-row panel of `inputs` activation rows (`Panel`, `PanelGated`, `Panels`).
/// The kernel reads it from the bits above the input count of the command; the host sizes the grid with it.
#[derive(Clone,Copy,PartialEq,Eq,Debug)]
pub struct PanelShape {
    /// Shift of the warps a panel (`1 << shift` warps, a power of 2, so a panel's warps are neighbours in the grid).
    pub shift:u32,
    /// Warps of 8 inputs (the pipelined 1..=8-input product, one pass); else warps of 16 inputs (two MMA tiles).
    pub narrow:bool,
}
impl PanelShape {
    pub const fn warps(self)->usize {1<<self.shift}
    /// The bits the command carries above the input count.
    pub const fn code(self)->usize {(self.shift as usize)|((self.narrow as usize)<<3)}
}
/// - `1..=8` inputs: one warp, the pipelined 1..=8-input product.
/// - a small matrix (at most `PANEL_SPLIT_ROWS_MAX` rows): one warp for each 8 inputs up to
///   `PANEL_NARROW_INPUTS_MAX` inputs (2, 4 or 8 warps a panel), one warp for each 16 inputs above (8 or 16 warps).
/// - a larger matrix: warps of 16 inputs, up to 4 warps a panel (one CTA) that walk the inputs in waves of 64.
#[inline(always)]
pub const fn panel_shape(rows:usize,inputs:usize)->PanelShape {
    if inputs<=8 {return PanelShape{shift:0,narrow:true};}
    if rows<=PANEL_SPLIT_ROWS_MAX {
        let narrow=inputs<=PANEL_NARROW_INPUTS_MAX;
        let warps=inputs.div_ceil(if narrow {8} else {16});
        return PanelShape{shift:warps.next_power_of_two().trailing_zeros(),narrow};
    }
    PanelShape{shift:if inputs<=16 {0} else if inputs<=32 {1} else {2},narrow:false}
}
/// Warps a 16-row panel in the launch grid.
#[inline(always)]
pub const fn panel_slabs(rows:usize,inputs:usize)->usize {panel_shape(rows,inputs).warps()}
#[derive(Clone,Copy)]
pub enum DraftOp {
    /// Multiply lane-interleaved NVFP4 panels by up to `PANEL_INPUTS_MAX` activation rows.
    PackActivations(usize),
    /// Rounded residual followed by one finite-status triple per warp.
    CheckedSum(usize),
    HeadFold{rows:usize,hidden:usize,latent:usize,inputs:usize,epsilon:f32,bound:f32,confidence:bool,bf16_base:bool},
    Panel {rows:usize, columns:usize, inputs:usize},
    /// Like `Panel`, writing packed BF16 pairs (`inputs * rows.div_ceil(2)` words) streamed with `L2::evict_first`.
    PanelBf16 {rows:usize, columns:usize, inputs:usize},
    /// Like `Panel`, gated by a 1-word device stop flag in input 2; appends a 1-word propagated stop flag after the output.
    PanelGated {rows:usize, columns:usize, inputs:usize},
    /// Two or three sibling panels of one input, stored back to back (rows in whole 64-row blocks, an unused
    /// third entry 0): one launch writes their outputs back to back, each as `Panel` writes it.
    Panels {rows:[usize;3], columns:usize, inputs:usize},
    /// Add the bias stored after the panel and the base row, then reduce each 16-token tile.
    /// When `gated`, input 1 appends a 1-word device stop flag and non-zero exits immediately with zero panel reads.
    /// When `bf16_base`, input 2 holds packed BF16 base logits (`rows.div_ceil(2)` words per row).
    FoldPick {rows:usize, columns:usize, epsilon:f32, bound:f32, confidence:bool, gated:bool, bf16_base:bool},
    /// Batched `FoldPick` across `batch` chats (`1..=DRAFT_BATCH_MAX`) sharing one read of the fold panel and bias.
    FoldPickBatch {rows:usize, columns:usize, batch:usize, epsilon:f32, bound:f32, confidence:bool, bf16_base:bool},
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
    /// Like `Rope`, writing packed BF16 pairs (`rows * heads * (width / 2)` words, `width % 4 == 0`).
    RopeBf16{rows:usize,heads:usize,width:usize},
    /// Pack `n` BF16-rounded FP32 words into `n / 2` packed BF16 `u32` words (`n % 2 == 0`).
    PackBf16(usize),
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
            HeadFold{rows,hidden,latent,inputs,epsilon,bound,confidence,bf16_base} if (1..=16).contains(&inputs) && latent<(1<<24) && epsilon>0.0 && bound>0.0=>{
                let packed=(hidden.checked_mul(inputs)?.checked_add(latent)?/64).checked_mul(18)?;
                let base_row=if bf16_base {rows.div_ceil(2)} else {rows};
                let out=base_row.checked_mul(inputs-1)?.checked_add(rows.div_ceil(16)*4)?;
                let c=latent.checked_mul(64)?+inputs*2+usize::from(confidence)+(usize::from(bf16_base)<<31);
                ([panel_words(rows,hidden)?,panel_words(rows,latent)?.checked_add(rows)?,packed],out,rows.next_multiple_of(16)*2,
                    [17,rows,hidden,c,epsilon.to_bits() as usize,bound.to_bits() as usize])}
            PackActivations(n) if n>0 && n.is_multiple_of(64)=>([n,0,0],n/64*18,n/16,[16,n/64,0,0,0,0]),
            Panel{rows,columns,inputs} if panel_words(rows,columns).is_some() && (1..=PANEL_INPUTS_MAX).contains(&inputs)=>{
                let shape=panel_shape(rows,inputs);
                ([panel_words(rows,columns)?,(columns/64*18).checked_mul(inputs)?,0],rows.checked_mul(inputs)?,rows.next_multiple_of(16).checked_mul(2)?.checked_mul(shape.warps())?,[13,rows,columns,inputs|shape.code()<<16,0,0])}
            PanelBf16{rows,columns,inputs} if panel_words(rows,columns).is_some() && (1..=PANEL_INPUTS_MAX).contains(&inputs)=>{
                let shape=panel_shape(rows,inputs);
                ([panel_words(rows,columns)?,(columns/64*18).checked_mul(inputs)?,0],rows.div_ceil(2).checked_mul(inputs)?,rows.next_multiple_of(16).checked_mul(2)?.checked_mul(shape.warps())?,[28,rows,columns,inputs|shape.code()<<16,0,0])}
            PanelGated{rows,columns,inputs} if panel_words(rows,columns).is_some() && (1..=16).contains(&inputs)=>{
                let shape=panel_shape(rows,inputs);
                ([panel_words(rows,columns)?,(columns/64*18).checked_mul(inputs)?,1],rows.checked_mul(inputs)?.checked_add(1)?,rows.next_multiple_of(16)*2*shape.warps(),[13,rows,columns,inputs|shape.code()<<16,1,0])}
            Panels{rows,columns,inputs} if (1..=PANEL_INPUTS_MAX).contains(&inputs) && rows[0]>0 && rows[1]>0 && rows.iter().all(|r|r.is_multiple_of(64))=>{
                let mut words=0usize;
                for r in rows.into_iter().filter(|&r|r>0) {words=words.checked_add(panel_words(r,columns)?)?;}
                let total=rows[0].checked_add(rows[1])?.checked_add(rows[2])?;
                let shape=panel_shape(total,inputs);
                ([words,(columns/64*18).checked_mul(inputs)?,0],total.checked_mul(inputs)?,total.checked_mul(2)?.checked_mul(shape.warps())?,[19,rows[0],columns,inputs|shape.code()<<16,rows[1],rows[2]])}
            FoldPick{rows,columns,epsilon,bound,confidence,gated,bf16_base} if epsilon>0.0 && bound>0.0=>{
                let flags=usize::from(confidence)|(usize::from(gated)<<1)|(usize::from(bf16_base)<<8);
                let base_words=if bf16_base {rows.div_ceil(2)} else {rows};
                ([panel_words(rows,columns)?.checked_add(rows)?,columns/64*18+usize::from(gated),base_words],rows.div_ceil(16)*4,rows.next_multiple_of(16)*2,[14,rows,columns,flags,epsilon.to_bits() as usize,bound.to_bits() as usize])}
            FoldPickBatch{rows,columns,batch,epsilon,bound,confidence,bf16_base} if (1..=DRAFT_BATCH_MAX).contains(&batch) && epsilon>0.0 && bound>0.0=>{
                let flags=usize::from(confidence)|((batch-1)<<2)|(usize::from(bf16_base)<<8);
                let slabs=batch.div_ceil(8);
                let base_words=(if bf16_base {rows.div_ceil(2)} else {rows}).checked_mul(batch)?;
                ([panel_words(rows,columns)?.checked_add(rows)?,(columns/64*18).checked_mul(batch)?,base_words],
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
            FlashAttn{rows,heads,kv,width,keys,splits,causal_base,row_off} if width==128 && kv>0 && heads>0 && heads%kv==0 && rows>0 && rows<65536 && keys>0 && (1..=255).contains(&splits) && row_off<256 && causal_base<=keys && causal_base<=u32::MAX as usize=>{
                let queries=rows.checked_mul(heads)?;let q=queries.checked_mul(width)?;
                let k=keys.checked_mul(kv)?.checked_mul(width)?;
                let out=if splits==1 {q} else {queries.checked_mul(splits)?.checked_mul(132)?};
                ([q,k,k],out,queries.checked_mul(splits)?.checked_mul(32)?,[20,(1.0/(width as f32).sqrt()).to_bits() as usize,heads,kv,(causal_base<<32)|(row_off<<24)|(splits<<16)|rows,keys])}
            FlashAttnMerge{rows,heads,width,splits} if width==128 && rows>0 && heads>0 && splits>=2=>{
                let queries=rows.checked_mul(heads)?;let parts=queries.checked_mul(splits)?.checked_mul(132)?;
                ([parts,0,0],queries.checked_mul(width)?,queries.checked_mul(32)?,[21,heads,rows,splits,0,0])}
            Gate(n) if n>0 && n<=1<<24=>([n,n,0],n,n.div_ceil(4),[0,n,0,0,0,0]),
            Rope{rows,heads,width} if width>0 && width%2==0 && heads>0=>{
                let trig=rows.checked_mul(width)?;let n=trig.checked_mul(heads)?;
                ([n,trig,0],n,n,[1,width,heads,0,0,0])}
            RopeBf16{rows,heads,width} if width>0 && width%4==0 && heads>0=>{
                let trig=rows.checked_mul(width)?;let n=trig.checked_mul(heads)?;let half=n/2;
                ([n,trig,0],half,half,[29,width,heads,0,0,0])}
            PackBf16(n) if n>0 && n%2==0 && n<=1<<24=>([n,0,0],n/2,n.div_ceil(4),[30,n,0,0,0,0]),
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

#[cfg(test)]
mod tests {
    use super::*;
    /// The kernel (`draft_panel.rs`, `run`) reads the warps a panel and their width from the command. A warp of 8
    /// inputs makes one pass, so its launch must hold a warp for each 8 inputs; a small matrix gets one pass at
    /// every width; a larger matrix keeps at most 4 warps a panel. Every launch grid must hold all of those warps.
    #[test]
    fn panel_grids_hold_every_warp_slab() {
        for inputs in 1..=PANEL_INPUTS_MAX {
            for (rows,small) in [(100usize,true),(PANEL_SPLIT_ROWS_MAX,true),(PANEL_SPLIT_ROWS_MAX+64,false)] {
                let shape=panel_shape(rows,inputs);
                let warps=if inputs<=8 {1} else if small {
                    if inputs<=PANEL_NARROW_INPUTS_MAX {inputs.div_ceil(8).next_power_of_two()} else {inputs.div_ceil(16).next_power_of_two()}
                } else if inputs<=16 {1} else if inputs<=32 {2} else {4};
                assert_eq!((shape.warps(),shape.narrow),(warps,inputs<=8 || (small && inputs<=PANEL_NARROW_INPUTS_MAX)),"rows={rows} inputs={inputs}");
                assert!(!shape.narrow || shape.warps()*8>=inputs,"a warp of 8 inputs makes one pass: rows={rows} inputs={inputs}");
                assert!(!small || shape.warps()*if shape.narrow {8} else {16}>=inputs,"a small matrix gets one pass: rows={rows} inputs={inputs}");
                assert!(shape.shift<8 && shape.code()<16,"the shape is 4 bits above the input count");
                assert_eq!(panel_slabs(rows,inputs),warps);
                let (_,_,lanes,fields)=DraftOp::Panel{rows,columns:128,inputs}.layout().unwrap();
                assert_eq!(lanes,rows.next_multiple_of(16)*2*warps,"Panel rows={rows} inputs={inputs}");
                assert_eq!(fields[3],(inputs|shape.code()<<16) as u64,"Panel command rows={rows} inputs={inputs}");
                if inputs<=16 {
                    let (_,_,gated,gated_fields)=DraftOp::PanelGated{rows,columns:128,inputs}.layout().unwrap();
                    assert_eq!((gated,gated_fields[3]),(lanes,fields[3]),"PanelGated rows={rows} inputs={inputs}");
                }
            }
            // Siblings take the shape of their rows together: 4096 + 4096 is a small matrix, 4096 + 4160 is not.
            for sib in [[4096usize,4096,0],[4096,4160,0]] {
                let total=sib.iter().sum::<usize>();
                let shape=panel_shape(total,inputs);
                let (_,_,lanes,fields)=DraftOp::Panels{rows:sib,columns:128,inputs}.layout().unwrap();
                assert_eq!(lanes,total*2*shape.warps(),"Panels {sib:?} inputs={inputs}");
                assert_eq!(fields[3],(inputs|shape.code()<<16) as u64,"Panels command {sib:?} inputs={inputs}");
            }
        }
    }
    #[test]
    fn bf16_base_and_vectorized_draft_layouts_match_extents() {
        for rows in [79usize, 201_024] {
            for inputs in [1usize, 8, 16, 64, 224] {
                let (fp_l, fp_o, fp_lanes, fp_f) = DraftOp::Panel { rows, columns: 2304, inputs }.layout().unwrap();
                let (bf_l, bf_o, bf_lanes, bf_f) = DraftOp::PanelBf16 { rows, columns: 2304, inputs }.layout().unwrap();
                assert_eq!((fp_l, fp_lanes, fp_f[3], fp_f[4]), (bf_l, bf_lanes, bf_f[3], bf_f[4]));
                assert_eq!((fp_o, bf_o, bf_f[0]), (rows * inputs, rows.div_ceil(2) * inputs, 28));
            }
            for batch in [1usize, 8, 32] {
                let (fp_lens, _, fp_lanes, fp_f) = DraftOp::FoldPickBatch {
                    rows, columns: 256, batch, epsilon: 1.0 / 30.0, bound: 30.0, confidence: true, bf16_base: false,
                }.layout().unwrap();
                let (bf_lens, _, bf_lanes, bf_f) = DraftOp::FoldPickBatch {
                    rows, columns: 256, batch, epsilon: 1.0 / 30.0, bound: 30.0, confidence: true, bf16_base: true,
                }.layout().unwrap();
                assert_eq!(fp_lanes, bf_lanes);
                assert_eq!(fp_lens[2], rows * batch);
                assert_eq!(bf_lens[2], rows.div_ceil(2) * batch);
                assert_eq!(bf_f[3] ^ fp_f[3], 1 << 8);
            }
        }
        assert!(DraftOp::PanelBf16 { rows: 64, columns: 2304, inputs: PANEL_INPUTS_MAX + 1 }.layout().is_none());
        let (gl, go, glanes, gf) = DraftOp::Gate(19).layout().unwrap();
        assert_eq!((gl, go, glanes, gf[0]), ([19, 19, 0], 19, 5, 0));
        let (rl, ro, rlanes, rf) = DraftOp::RopeBf16 { rows: 7, heads: 8, width: 128 }.layout().unwrap();
        assert_eq!((rl, ro, rlanes, rf[0]), ([7 * 8 * 128, 7 * 128, 0], 7 * 8 * 64, 7 * 8 * 64, 29));
        let (pl, po, planes, pf) = DraftOp::PackBf16(18).layout().unwrap();
        assert_eq!((pl, po, planes, pf[0]), ([18, 0, 0], 9, 5, 30));
    }
}
