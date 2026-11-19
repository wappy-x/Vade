//! Layer-major prompt groups. All rows of a group finish one layer before the
//! next layer starts, so one weight read serves many rows. Each row keeps its
//! scalar arithmetic. Recurrent state and latent history stay in token order.
use super::*;
use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
#[path="rows/pools.rs"] mod pools;
#[path="rows/seed.rs"] mod seed;
#[path="rows/shared.rs"] mod shared; #[path="rows/recurrent.rs"] mod recurrent;
#[path="rows/wide_charge.rs"] mod wide_charge;
use wide_charge::{wide_phase_group,wide_phase_workspace};
#[cfg(test)] pub(crate) use wide_charge::wide_cut_layers;
pub use shared::SharedRows;
/// ROWS FOLD: what ONE layer of a row pass holds between its routed down step and that step's fold. It lies on the
/// caller's stack and ends with the layer, so no step reads another layer's row and nothing is allocated for it.
pub(super) struct RowsFold<V> {
    /// The routed down step waits: the fold step runs both (`Decoder::rows_fold_waits` said so at the down step).
    waits:bool,
    /// This rank's own rows of the shared projection, when the compute kept them here for that fold (`rows_fold_shared`).
    shared:Option<V>,
    /// GROUP FOLD: the pass is a prompt group (`layer_rows` without kept rows). Only such a pass asks `folds_groups`.
    group:bool,
    /// GROUP FOLD: the down step that waits is the grouped form of a prompt group (`rows_group_fold_waits`).
    grouped:bool,
    /// LOOKAHEAD: this rank's own rows of the shared projection, made behind the routed gate and up of a prompt group
    /// (`rows_ahead`). The shared projection's step hands them to the fold (`rows_fold_shared`).
    ahead:Option<V>,
    /// ROWS OVERLAP: the outputs of a stage whose gather is held (`rows_project_open`), as (job index, value name).
    held:Vec<(usize,String)>,
}
impl<V> Default for RowsFold<V> {fn default()->Self {Self {waits:false,shared:None,group:false,grouped:false,ahead:None,held:Vec::new()}}}
impl<V> RowsFold<V> {
    /// The fold state of one layer of a prompt group.
    fn of_group()->Self {Self {group:true,..Self::default()}}
}
/// ROWS FOLD: the steps of a routed layer's fold, by index. They follow from the layer's program alone, which has the
/// same steps at every context (the batch step relies on that too), so a decoder finds them once a layer
/// (`Decoder::fold_links`) and a row pass reads three numbers.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub(super) struct FoldLinks {
    /// The routed down step whose only reader is the fold (`fold_of`).
    down:usize,
    /// The fold step (ExpertSum).
    fold:usize,
    /// The shared projection, when the fold is its only reader: its step, and whether its rows are rounded (it feeds no route).
    shared:Option<(usize,bool)>,
}
impl FoldLinks {
    /// The links of a layer's program. None: no routed down step whose fold is its only reader.
    pub(super) fn of(plan:&LayerPlan)->Option<Self> {
        let down=plan.steps.iter().position(|step|step.kind==Kind::ExpertDown)?;
        let (fold,sum)=fold_of(down,plan)?;
        let shared=sum.inputs.get(3).and_then(|name| {
            let at=plan.steps[..fold].iter().rposition(|step|step.kind==Kind::Project && step.weights.len()==1 && step.inputs.len()==1 && step.outputs.len()==1 && step.outputs[0].name==*name)?;
            let readers=plan.steps.iter().filter(|step|step.inputs.iter().any(|input|input==name)).count();
            if at<=down || readers!=1 || plan.returns.iter().any(|kept|kept==name) {return None;}
            let router=plan.steps.iter().any(|step|step.kind==Kind::Route && step.inputs.contains(name));
            Some((at,!router))
        });
        Some(Self {down,fold,shared})
    }
}
/// Largest group one prefill call accepts. A fixed size, so memory is known at admission.
/// Measured on the real server: 1040 tokens read in 46 s at this size and 54 s at 1024.
pub const PROMPT_GROUP:usize=256;
/// Largest group a server admits. The server's group comes from its admitted memory.
pub const PROMPT_GROUP_MAX:usize=1024;
/// Groups above `PROMPT_GROUP_MAX` that only a prompt lease funds (`row_split::prompt_lease`), largest first. No startup
/// workspace holds them: a lone chat borrows their bytes from the unused part of the session pool for one prompt call and
/// returns them after it, so the pool, the cache cap and the admitted context are those of the startup group. A group
/// reads every routed expert once (about 0.3 s of a 1,024-row group does not grow with its rows), so 4,096 rows pay that
/// read once where four groups of 1,024 pay it four times. A leased call cuts its tokens into whole groups of these
/// sizes and a rest of at most `PROMPT_GROUP_MAX` rows (`lease_groups`). A group above `PROMPT_GROUP_MAX` passes every
/// layer as runs of `PROMPT_GROUP_MAX` rows (`Decoder::layer_wide`), so no kernel sees more rows than a startup group
/// gives it; only the routed expert product launches hold the parts of every run (WIDE COHORT).
/// THREE RUNS (2026-11-17): 3,072 rows stand between the two older sizes. The lease charges a wide group the whole
/// workspace of a startup group of its size, and a lone chat's spare pool covers that for three runs where it does
/// not for four (the 8,192-token read: 8.42 GB for 3,072 rows and 12.64 GB for 4,096 against 9.86 GB to lend).
/// FOUR RUNS: 4,096 rows are charged by the two phases of their wide pass where that charge applies
/// (`wide_phase_workspace`), which the spare pool of that read covers; where it does not apply they keep the whole charge.
pub const PROMPT_GROUP_LEASE:[usize;3]=[4096,3072,2048];
const _:()=assert!(PROMPT_GROUP_LEASE[0]<=PROMPT_GROUP_CHECK && PROMPT_GROUP_LEASE[2]>PROMPT_GROUP_MAX
    && PROMPT_GROUP_LEASE[0]>PROMPT_GROUP_LEASE[1] && PROMPT_GROUP_LEASE[1]>PROMPT_GROUP_LEASE[2]
    && PROMPT_GROUP_LEASE[0].is_multiple_of(PROMPT_GROUP_MAX) && PROMPT_GROUP_LEASE[1].is_multiple_of(PROMPT_GROUP_MAX)
    && PROMPT_GROUP_LEASE[2].is_multiple_of(PROMPT_GROUP_MAX));
/// The groups of one prompt call of `tokens` rows whose largest group is `rows`: up to `PROMPT_GROUP_MAX` rows the call
/// is cut as before, in runs of `rows`. Above it only whole groups of a `PROMPT_GROUP_LEASE` size (at most `rows`) are
/// taken, largest first, and the rest runs in groups of at most `PROMPT_GROUP_MAX` rows. The row bits do not depend on
/// the cut; the cut keeps every group at a row count the kernels' swept lists hold.
pub fn lease_groups(tokens:usize,rows:usize)->Vec<usize> {
    let mut out=Vec::new();let mut left=tokens;
    if rows==0 {return out;}
    if rows>PROMPT_GROUP_MAX {for size in PROMPT_GROUP_LEASE {if size<=rows {while left>=size {out.push(size);left-=size;}}}}
    let small=rows.min(PROMPT_GROUP_MAX);
    while left>0 {let n=left.min(small);out.push(n);left-=n;}
    out
}
/// Largest group the row pass itself accepts. Above `PROMPT_GROUP_MAX` only the check tool asks for it
/// (`--row-groups`): a premise size, not charged by any admission ledger.
pub const PROMPT_GROUP_CHECK:usize=4096;
/// Group size while other chats wait: about six seconds of their time.
pub const PROMPT_SHARED:usize=64;
/// Most rows of a group whose MXFP8 projections of one input run as one merged row launch (ROW STAGE). The entry
/// `flash_resident_projection_bf16_rows` takes 2 to 8 rows and the stage sends it all of them. Until ROWS-MX
/// (2026-11-12) the stage stopped at 3: the 4-row merged launch read q, k and v of a layer in 404 us where the one-row
/// launch takes 260 (one Spark, kernel-bench --rows-p1), and lost to one small launch a matrix. With the row body on the
/// one-token load schedule (`mx_rows_small::wave`) the same 4-row launch takes 287 us, and on a pair the 4-chat step read
/// 102.97 ms with the stage at 4 rows against 104.81 with it at 3 (main: 106.60). A larger group keeps one launch chain a
/// matrix.
/// STAGE EIGHT (seat DRAFTR, 2026-11-17): 8, was 4. The entry takes 2 to 8 rows. Measured on a served 8-chat step (run
/// draftr-c3-ct1): the single calls of the stage's matrices cost 15.5 ms a step (145 + 45 + 143 launches) where the merged
/// form of a 4-chat step cost 12.5 ms; from 5 rows a pass paid that difference.
pub const ROW_STAGE_ROWS:usize=8;
/// Expanded latent rows each chat holds, all charged at admission. Past this context
/// a group expands only the rows its tokens selected, in memory that lasts for one layer.
pub const RESERVED_ROWS:usize=1152;
/// Rows of room behind the held rows of a decoder with a proposer. Admission charges them.
pub const HELD_ROOM:usize=128;
/// Room the vector pool leaves under its cap at each decode token and after each prompt
/// group. At the least cap of 2,000,000,000 B the pool then keeps at most 926,258,176 B of
/// spares, inside the 1 GiB the shared workspace adds to the larger of its decode and prompt peaks.
pub const POOL_ROOM:usize=1<<30;
/// Byte cap of the decode reserve: the device vectors that decode steps use (two copies of each recurrent state,
/// the per-layer vectors, the caches they grow), kept through prompts and from one session to the next, so a
/// session's first steps do not allocate them again. Spare storage, released before any allocation is refused.
pub const DECODE_RESERVE:usize=1<<30;
/// Room the end of a prompt leaves in the vector pool beyond POOL_ROOM. The first decode steps still return the
/// prompt's own states and caches to the pool (about 280 MB at 4,158 tokens: 34 recurrent states of 4.2 MB, the
/// latent caches that an append moves); with this room the token-start trim then frees nothing. At the least cap
/// the pool keeps at most 120,951,808 B of the prompt's spares for the next prompt.
pub const DECODE_RETURNS:usize=3<<28;
/// Most bytes of slab chunks the projection memo keeps when a session starts. Its rows go; this storage stays, so
/// the first long-context steps of the next session fill it without a driver call. Inside PROJECTION_MEMO_BYTES.
pub const MEMO_KEEP:usize=1<<29;
/// A prompt group that starts a session (position 0) and runs as a group trims the vector
/// pool's spares to POOL_ROOM first. What the previous session freed just before (its caches,
/// expanded rows and decode spares) can fill the pool to its cap; a full pool refuses the
/// group's own returns, so every layer would allocate again. Trimming leaves POOL_ROOM of
/// headroom for those returns and keeps the rest: the previous session's decode spares stay
/// for this session's first decode steps, which would otherwise buy their whole working set
/// back. Clearing all spares here made the first decode step of every served session
/// allocate again (measured on pair 1: 2,005 buffers and 279 MB at 15 tokens of context,
/// 512 buffers and 1,038 MB at 4,158, in the first step's own pool counts). Trimming only
/// frees free memory, and the admission charge already funds POOL_ROOM of spares.
pub fn trims_spares_first(rows:usize,position:u64)->bool {rows>=PROMPT_SHARED && position==0}
/// The vector pool's byte cap while a prompt group of `rows` rows runs, or None for a group that keeps the pool's own
/// cap. The least cap (2 GB) dates from groups of at most PROMPT_GROUP rows. A group's buffers grow with its rows, so
/// a larger group gets the least cap once for each PROMPT_GROUP rows: the same number of spare buffers as before
/// (4 GB at 512 rows, 8 GB at 1024). Under the least cap one 8,192-token prompt in 1024-row groups made 87.7 GB of
/// new device buffers and spent 3.3 s in the allocator (measured on the model, both ranks): a full pool refused the
/// group's own returns, and a 1024-row layer's planes did not fit beside its other buffers. The cap lasts until the
/// group's end trims the pool (`rows_keep`), so between groups, in decode and at rest the pool holds what it held.
/// Spares are free storage inside the device budget, which releases them before it refuses any allocation.
pub fn group_pool_cap(rows:usize)->Option<usize> {(rows>PROMPT_GROUP).then(||spark_gpu_probe::VECTOR_POOL_LEAST_CAP.saturating_mul(rows.div_ceil(PROMPT_GROUP)))}
/// The most the projection memo holds, charged once per server at admission. One kept
/// row of this model is 65,536 bytes packed plus its key, and a token can select up to
/// 2,051 rows in each of 11 latent layers (1.48 GB), so the memo holds about two tokens' rows.
pub const PROJECTION_MEMO_BYTES:usize=crate::projection_memo::FULL;
/// Largest union of selected rows whose attention scores run in tiles, computing all pairs.
/// Past it, computing all pairs costs more than reading only the selected ones.
pub const SCORE_MAP_ROWS:usize=16384;
/// Which prompt rows pass through the vocabulary head.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum HeadRows {All,Last,From(usize)}
/// What a group keeps until its verdict: for each recurrent layer the inputs of its two state
/// producers. The state after a prefix of the rows then costs two kernels a layer, and no
/// row is computed twice. About 130 KB a row and layer for this model, inside the group's
/// workspace, released at the verdict.
pub struct OpenRows<V> {first:u64,rows:usize,layers:Vec<Option<KeptRows<V>>>,hidden:[V;2]}
struct KeptRows<V> {conv:[V;3],history:String,step:V,decay:V,state:String}
/// Bytes for each latent layer's expanded rows at full context. Rows are packed two
/// BF16 patterns a word, so each feature costs two bytes, not four.
pub fn expanded_bytes(c:&CheckpointAdmission,context:usize)->usize {
    c.description().layers.iter().map(|l|match l.mixer {TokenMixer::LatentAttention(a)=>(a.heads*(a.key_dim+a.value_dim)) as usize*2,_=>0}).sum::<usize>().saturating_mul(context.min(RESERVED_ROWS))
}
/// Device bytes one prompt group can hold at once, charged once per server because
/// only one execution runs at a time: all rows' fixed buffers in the largest layer,
/// their attention buffers at this context, one layer's selected rows, and the exchange.
pub fn group_workspace_bytes(c:&CheckpointAdmission,context:usize)->Result<usize> {group_workspace_bytes_for(c,context,PROMPT_GROUP)}
/// Words of the five planes of one absorbed prompt group: the absorbed query and the mixed latent (`latent` words a
/// head a token), scores and probabilities (`live` words: one a selected row), and the projected output (`value` words).
pub fn absorbed_group_words(group:usize,heads:usize,latent:usize,value:usize,live:usize)->Option<usize> {
    group.checked_mul(heads)?.checked_mul(latent.checked_mul(2)?.checked_add(live.checked_mul(2)?)?.checked_add(value)?)
}
/// A chat group also owns its joined latent, selection table and local query beside the five planes.
pub fn absorbed_chat_words(chats:usize,heads:usize,latent:usize,key:usize,value:usize,slots:usize)->Option<usize> {
    absorbed_group_words(chats,heads,latent,value,slots)?
        .checked_add(chats.checked_mul(slots)?.checked_mul(latent)?)?
        .checked_add(chats.checked_mul(slots.checked_add(1)?)?)?
        .checked_add(chats.checked_mul(heads)?.checked_mul(key)?)
}
/// The same charge for a group of this many rows.
pub fn group_workspace_bytes_for(c:&CheckpointAdmission,context:usize,group:usize)->Result<usize> {group_workspace_bytes_with(c,context,group,LatentRows::Expanded)}
/// Which attention form the latent layers of a loaded server took. It decides whether a prompt group can hold expanded rows.
#[derive(Clone,Copy,Debug,PartialEq,Eq,Default)]
pub enum LatentRows {
    /// Some latent layer expands the rows its tokens selected: the charge every server had before this type.
    #[default] Expanded,
    /// Every latent layer of the checkpoint took the absorbed form on every rank. A prompt group and a decode step of
    /// such a layer leave the expansion branch before they expand or hold a row (`Kind::ExpandKv` here and in
    /// `bounded_decoder.rs`), so no group holds rows of one layer for the whole context. `draft`: a predictor layer is
    /// loaded and is NOT absorbed. It reads no prompt group (`rows/seed.rs` runs its cache half only) and proposes one
    /// row at a time, so what one such row can hold stays charged (`expanded_row_words`).
    Absorbed {draft:bool},
}
/// Words one decode row of an unabsorbed latent layer can hold at once in its widest form (`bounded_decoder.rs`,
/// `Kind::ExpandKv`, the fallback after the memo): for each of its `slots` selected rows the packed expansion twice (the
/// pieces and their reserved join) and the plain keys and values twice (the unpacked join and the planes cut from it),
/// the gathered latent, and one chunk of PROMPT_GROUP rows between its projection and its pack. `row`: words of one
/// expanded row of every head. The projection memo has its own charge.
pub fn expanded_row_words(row:usize,latent:usize,slots:usize)->Option<usize> {
    slots.checked_mul(row.checked_mul(3)?.checked_add(latent)?)?.checked_add(PROMPT_GROUP.checked_mul(row)?)
}
/// `group_workspace_bytes_for` for a server whose latent layers took `rows`: the whole-group charge
/// (`whole_group_workspace`), or for a group above the largest whole-charged lease size the charge of the two phases
/// of its wide pass (`wide_phase_group`, `wide_phase_workspace`).
pub fn group_workspace_bytes_with(c:&CheckpointAdmission,context:usize,group:usize,rows:LatentRows)->Result<usize> {
    if wide_phase_group(group,rows) {wide_phase_workspace(c,context,group)} else {whole_group_workspace(c,context,group,rows)}
}
/// The charge of a group of `group` rows that passes each layer whole.
fn whole_group_workspace(c:&CheckpointAdmission,context:usize,group:usize,rows:LatentRows)->Result<usize> {
    let d=c.description();let (mut per_row,mut selected,mut gather,mut route_epochs)=(0usize,0usize,0usize,0usize);
    for layer in 0..d.layers.len() {
        // Slot and context buffers are charged below from the group's own shapes.
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
        let mut words=0usize;let mut widest=0usize;
        for spec in plan.steps.iter().flat_map(|step|&step.outputs) {
            if matches!(spec.name.as_str(),"keys"|"values"|"scores"|"probabilities"|"selected_latent"|"selected"|"selected_count"|"next_state"|"next_history"|"next_latent_cache"|"next_index_keys"|"next_index_gates"|"pool_keys"|"pool_scores") {continue;}
            let elements=spec.elements().map_err(|e|e.to_string())?;words=words.checked_add(elements).ok_or("prompt workspace overflow")?;widest=widest.max(elements);
        }
        if let engine_model::shapes::Ffn::Experts(e)=d.layers[layer].ffn {
            // Fund every routed layer's complete immutable metadata epoch,
            // even when an ordered pass queue retains them all until its drain.
            // Weight descriptor templates are allocated before RESIDENT READY.
            let mut left=group;let mut sets=Vec::new();
            while left>0 {let rows=left.min(spark_gpu_probe::prompt_route::MAX_ROWS as usize);left-=rows;
                sets.push(spark_gpu_probe::prompt_route::Shape {rows:rows as u32,picks:e.selected as u32,experts:e.count as u32});}
            if let Some(bytes)=spark_gpu_probe::prompt_route::epoch_workspace_bytes(&sets) {route_epochs=route_epochs.checked_add(bytes).ok_or("prompt route epoch charge overflow")?;}
            // Quantized activations: seventeen words for sixteen features, for gate, up and down.
            let packed=(e.selected as usize).checked_mul(((d.hidden as usize/16)*17).max(e.width as usize/16*17)).and_then(|n|n.checked_mul(2)).ok_or("prompt workspace overflow")?;
            // Permuted inputs, and the route's complete scratch for each token.
            let route=(e.selected as usize*2+e.count as usize*2+d.routing.groups as usize*2).checked_add(e.selected as usize*d.hidden as usize).ok_or("prompt workspace overflow")?;
            words=words.checked_add(packed).and_then(|n|n.checked_add(route)).ok_or("prompt workspace overflow")?;
        }
        if let TokenMixer::LatentAttention(a)=d.layers[layer].mixer {
            let i=a.indexer;let slots=(i.token_budget+if i.select_tail {i.pool-1} else {0}) as usize;let live=slots.min(context);
            let latent=a.kv_rank as usize;
            let row=(a.heads*(a.key_dim+a.value_dim)) as usize;
            // Two buffers per head, the selection and its validation status, three words per entry of the Scores table,
            // and the pair map, which holds one word per token for each row of a small union.
            words=words.checked_add(live.checked_mul(a.heads as usize*2).ok_or("prompt workspace overflow")?).and_then(|n|n.checked_add(slots+2+live*3+context.min(SCORE_MAP_ROWS))).ok_or("prompt workspace overflow")?;
            // Past the persistent limit, a layer holds only rows its tokens selected.
            // The gathered latents of those rows live beside them, with one chunk being copied.
            // The rows are packed (row/2 words); the gathered latents stay full words. One chunk
            // also sits unpacked between its projection and its pack, hence one group's plain rows.
            let union=context.min(group.checked_mul(live).ok_or("prompt workspace overflow")?).checked_add(group).and_then(|rows|rows.checked_mul(row/2+a.kv_rank as usize)).and_then(|n|n.checked_add(group.checked_mul(row)?)).ok_or("prompt workspace overflow")?;
            match rows {
                LatentRows::Expanded=>selected=selected.max(union),
                // The predictor layer has the geometry of the latent layers (`TextDescription::with_draft_layer`). One decode
                // row of it, and never more than the union this charge funded before: no charge of any server grows.
                LatentRows::Absorbed {draft:true}=>selected=selected.max(union.min(expanded_row_words(row,a.kv_rank as usize,live).ok_or("prompt workspace overflow")?)),
                LatentRows::Absorbed {draft:false}=>{},
            }
            // Absorbed attention replaces the expanded rows with five planes that a group holds at once.
            // Charge the larger alternative, with the same head count as the expanded rows above.
            // A large absorbed group can keep one lossless packed cache mirror beside its five planes. Charge it
            // only for the packed tile's latent width. This is conservative when a rank's head count or alignment
            // makes it keep the original cache; memory admission does not choose a different execution by topology.
            let mirror=if group>=17 && latent==512 {context.checked_mul(latent).map(|n|n/2).ok_or("packed latent cache charge overflow")?} else {0};
            selected=selected.max(absorbed_group_words(group,a.heads as usize,a.kv_rank as usize,a.value_dim as usize,live).and_then(|n|n.checked_add(mirror)).ok_or("prompt workspace overflow")?);
            // Chat steps and prompt groups are ordered. Their larger attention peak funds both forms.
            selected=selected.max(absorbed_chat_words(super::BATCH_ROWS,a.heads as usize,a.kv_rank as usize,a.key_dim as usize,a.value_dim as usize,live).ok_or("chat workspace overflow")?);
        }
        // A group past PROMPT_GROUP also charges what the 256-row charge leaves to its margin (235 MB at 1,024 rows and a
        // 32,768 context): the layer's input rows, the two live planes and the partials of the open mHC boundary pass, and a
        // latent layer's pool scores (one word for each complete pool).
        if group>PROMPT_GROUP {
            let (streams,hidden)=(d.residual.streams as usize,d.hidden as usize);
            words=streams.checked_mul(hidden).and_then(|n|n.checked_mul(3)).and_then(|n|n.checked_add(hidden.div_ceil(128))).and_then(|n|words.checked_add(n)).ok_or("prompt workspace overflow")?;
            if let TokenMixer::LatentAttention(a)=d.layers[layer].mixer {words=words.checked_add(context/(a.indexer.pool as usize).max(1)).ok_or("prompt workspace overflow")?;}
        }
        per_row=per_row.max(words);gather=gather.max(widest);
    }
    // Each rank's half plus the peer's half, with the mapped allocation's growth margin.
    let exchange=gather.checked_mul(group).and_then(|n|n.checked_mul(3)).map(|n|n/2).ok_or("prompt workspace overflow")?;
    per_row.checked_mul(group).and_then(|n|n.checked_add(selected)).and_then(|n|n.checked_add(exchange)).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_add(route_epochs)).ok_or_else(||"prompt workspace overflow".into())
}
/// Partition a selected union into retained prefix rows and new rows. The row
/// order may change; each query keeps its original selected order through `map`.
fn expansion_rows(selected:&[f32],slots:usize,total:usize,held:usize)->Result<(Vec<f32>,Vec<u32>,usize)> {
    let words=slots.checked_add(1).ok_or("prompt selection width overflow")?;
    need(held<=total && selected.len().is_multiple_of(words),"prompt selected geometry differs")?;
    let mut map=vec![u32::MAX;total];let mut rows=Vec::new();
    for token in selected.chunks_exact(words) {
        let n=token[slots];need(n.is_finite() && n>=0.0 && n.fract()==0.0 && n<=slots as f32,"prompt selected count is invalid")?;
        for &id in &token[..n as usize] {
            need(id.is_finite() && id>=0.0 && id.fract()==0.0 && id<total as f32,"prompt selected row is outside its context")?;
            let id=id as usize;if map[id]==u32::MAX {map[id]=0;rows.push(id as f32);}
        }
    }
    need(!rows.is_empty(),"prompt selection is empty")?;
    rows.sort_by_key(|&id|usize::from(id as usize>=held));
    let cached=rows.iter().take_while(|&&id|(id as usize)<held).count();
    // Retained IDs are pure copies. Ascending order makes every adjacent run one copy.
    rows[..cached].sort_unstable_by_key(|&id|id as usize);
    for (at,&id) in rows.iter().enumerate() {map[id as usize]=number(at as u64)?;}
    Ok((rows,map,cached))
}
/// Each token takes rows no earlier token took, and its own new row, so the union of a
/// group reaches every row of its context.
fn force_distinct_rows(selected:&mut [f32],slots:usize,first:usize) {
    for (token,row) in selected.chunks_exact_mut(slots+1).enumerate() {
        let count=slots.min(first+1+token);
        for (slot,id) in row[..count-1].iter_mut().enumerate() {*id=((token*count+slot)%(first+1)) as f32;}
        row[count-1]=(first+token) as f32;row[slots]=count as f32;
    }
}
/// The memo bytes a memory proof must hold as ballast: the device's memo limit, and only when it equals the full
/// allowance PROJECTION_MEMO_BYTES, which admission charges while no startup memo bound is set. A smaller limit would
/// prove less than such a server can reach. A server with memo admission charges its smaller locked bound
/// (`row_split::Factory::with_memo_bound`); this check does not cover that case.
pub fn full_memo_ballast(limit:usize)->Result<usize> {
    if limit!=PROJECTION_MEMO_BYTES {return Err(format!("memo limit {limit} differs from the charged bound {PROJECTION_MEMO_BYTES}"));}
    Ok(limit)
}
/// `[rows,tokens]`: the token's slot for that row, or -1. Empty when a token selects a row
/// twice, because one row then has two slots and only the table can name them.
fn score_map(selected:&[f32],slots:usize,rows:usize,m:usize)->Vec<f32> {
    let mut map=vec![-1.0f32;rows*m];
    for (token,row) in selected.chunks_exact(slots+1).enumerate() {
        for (slot,&id) in row[..row[slots] as usize].iter().enumerate() {
            let Some(place)=map.get_mut(id as usize*m+token) else {return Vec::new();};
            if *place!=-1.0 {return Vec::new();}
            *place=slot as f32;
        }
    }
    map
}
/// WIDE GROUP: the cut of a routed layer's program (`Decoder::wide_cut`), from the program alone.
pub(super) fn wide_cut_of(plan:&LayerPlan)->Option<(usize,String)> {
    let cut=plan.steps.iter().rposition(|step|step.kind==Kind::MhcPrepare)?;let (head,tail)=plan.steps.split_at(cut);
    // Every step of the second half is one `rows_stateless` runs: none reads or makes history.
    if !tail.iter().all(|step|matches!(step.kind,Kind::MhcPrepare|Kind::Collapse|Kind::Norm|Kind::Project|Kind::Route|Kind::ExpertProject|Kind::SwiGlu|Kind::ExpertDown|Kind::ExpertSum|Kind::Combine)) {return None;}
    let made=|steps:&[engine_model::plan::layer::Step],name:&str|steps.iter().any(|step|step.outputs.iter().any(|output|output.name==name));
    // It reads ONE earlier value, and only its mHC boundary reads it; it makes the output, and the first half does not.
    let mut carry:Option<&str>=None;
    for step in tail {for input in &step.inputs {if !made(tail,input) {
        if !matches!(step.kind,Kind::MhcPrepare|Kind::Collapse|Kind::Combine) || carry.is_some_and(|name|name!=input) {return None;}
        carry=Some(input);
    }}}
    let carry=carry?;
    // Every state the layer returns is made by the first half.
    if !made(head,carry) || !made(tail,"output") || made(head,"output") || plan.returns.iter().any(|name|name!="output" && !made(head,name)) {return None;}
    Some((cut,carry.to_owned()))
}
/// The host tables of one grouped expert step (`Decoder::expert_tables`): the (token, pick) rows of each expert; each
/// expert's (ID, first row, rows) in ascending expert order; the source row of each grouped row; and three numbers
/// (earlier rows, this expert's rows, row index) for each row in grouped order and in (token, pick) order.
struct ExpertTables {groups:BTreeMap<usize,Vec<usize>>,spans:Vec<(usize,usize,usize)>,order:Vec<f32>,grouped:Vec<f32>,placed:Vec<f32>}
/// The (matrix, rows) list of a grouped expert step for the experts of `spans`.
fn expert_names<'s>(step:&'s engine_model::plan::layer::Step,spans:&[(usize,usize,usize)])->Vec<(&'s str,usize)> {spans.iter().map(|&(id,_,count)|(step.weights[id].name.as_str(),count)).collect()}
fn row_shape(op:Row,m:usize)->Result<RowShape> {RowShape::new(op,m).ok_or_else(||"prompt row count exceeds ABI".into())}
/// One chat's side of a row pass: its rows (`m` of them from position `first`), its state and held rows, and what its
/// history steps keep for the steps after them. The one-chat pass has one (`layer_rows`); a shared pass has one a chat
/// (rows/shared.rs), so both run the same history steps (`Decoder::rows_history`).
struct RowsChat<'a,V> {first:u64,m:usize,state:&'a BTreeMap<String,V>,expanded:Option<&'a Expanded<V>>,
    /// The pass keeps the inputs of the recurrent state producers (`OpenRows`), and holds the rows it expands.
    keep:bool,persist:bool,
    kept_conv:Option<([V;3],String)>,kept_step:Option<(V,V,String)>,
    /// The expanded rows this pass reads, and how many; the selection of each row, on the host and on the device.
    cache:Option<V>,cache_rows:u64,selected:Vec<f32>,selected_device:Option<V>,selection_checked:bool}
impl<'f,C:Compute> Decoder<'f,C> {
    /// Test and admission control. Bits do not depend on this limit.
    pub fn set_expanded_limit(&self,rows:usize) {self.expanded_limit.set(rows);}
    /// A chat's state at a parked prompt: a new handle of every kept plane (`share_state`), and the expanded rows built
    /// again from the kept `latent_cache` rows, as many rows a layer as the prompt held when it was parked. The expansion
    /// of one latent row depends only on its bits, so the words equal the ones that were dropped. One prompt group of
    /// rows is expanded at a time, into storage with room for the rest, so every group is appended where it is and
    /// a layer's rows are allocated once (`held::extend`). On a device, as in a prompt group (`rows_pass`): launch
    /// recording is paused, so the rebuild captures no graph piece, and each group's own buffers are freed when the
    /// group is done (`rows_keep`), so the rebuild holds one group of scratch beside the rows, not every group's.
    #[cold] #[inline(never)]
    pub fn resume_state(&self,parked:&Parked<C::Value>)->Result<State<C::Value>> {
        let was=self.compute.rows_unrecorded(true)?;let result=self.resume_rows(parked);
        // Recording resumes also when the rebuild fails. The rebuild's own error is the one reported.
        let restored=self.compute.rows_unrecorded(was);let value=result?;restored?;Ok(value)
    }
    #[cold] #[inline(never)]
    fn resume_rows(&self,parked:&Parked<C::Value>)->Result<State<C::Value>> {
        let b=&self.compute;let d=self.checkpoint.description();let mut state=share_state(b,&parked.state)?;
        for (&layer,&(held,layout)) in &parked.held {
            // Only a prompt layer of the checkpoint holds rows a prompt group reads. Any other owner goes on without
            // held rows, which is exact: every row it needs is expanded when it is selected.
            let Some(TokenMixer::LatentAttention(a))=d.layers.get(layer).map(|spec|spec.mixer) else {continue;};
            if held==0 {continue;}
            need(layout==self.sparse_layout(layer)?,"parked expanded layout differs")?;
            need(held as u64<=state.position && held<=self.expanded_limit.get(),"parked expanded rows exceed the prompt")?;
            let plan={let mut schedules=self.schedules.borrow_mut();schedules.get_mut(layer).ok_or("parked layer is outside the recipe")?.at(state.position).cloned().map_err(|e|e.to_string())?};
            let name=&plan.steps.iter().find(|step|matches!(step.kind,Kind::ExpandKv)).ok_or("parked layer has no expansion")?.weights[0].name;
            let rank=a.kv_rank as usize;let row_words=layout.row_words();let packed_words=row_words/2;
            let latents=state.layers[layer].get("latent_cache").ok_or("parked prompt has no latent rows")?;
            // With a proposer the held rows grow with the decoded rows: the same room behind them as a prompt group leaves.
            let spare=if self.draft.is_some() {HELD_ROOM.min(self.expanded_limit.get()-held)} else {0};
            let mut cache:Option<C::Value>=None;let mut at=0;
            while at<held {let count=(held-at).min(PROMPT_GROUP);
                let result=(|| {let part=self.expand_owned_rows(name,&b.range(latents,at*rank,count*rank)?,count,layout)?;let part=self.pack_expanded(&part,count*row_words)?;
                    cache=Some(b.rows_extend(cache.as_ref(),at,&part,packed_words,(held-at-count+spare)*packed_words)?);Ok::<(),String>(())})();
                // The group's projection and its packed form are dropped. Inside a work scope a device only lists a
                // dropped buffer; this drain frees them, and the next group takes the same sizes back from the pool.
                // Drain on error too, after the rows this failed rebuild built are dropped: the drain then frees them
                // with the group's scratch. Keep the rebuild error if the drain also fails.
                if result.is_err() {cache=None;state.expanded.clear();}
                let drained=b.rows_keep();result?;drained?;at+=count;}
            state.expanded.insert(layer,Expanded(std::rc::Rc::new(cache.ok_or("parked prompt rebuilt no row")?),held,layout));
        }
        Ok(state)
    }
    /// Memory test only: tokens select distinct rows, so a group's union reaches its whole
    /// context. The logits then mean nothing. No server calls this.
    pub fn set_forced_union(&self) {self.forced_union.set(Some(0));}
    /// The next prompt group is the last of its prompt: of the rows it expands past the held rows, those its last
    /// token selected are kept in the projection memo. The first decode token selects nearly the same rows;
    /// without this it expands each of them again (about 14,000 rows at 4,158 tokens of context) into a memo
    /// that is empty at a session start.
    /// A kept row is the pure expansion the decode step would make: the same bits, by the same projection.
    pub fn set_memo_seed(&self,on:bool) {self.memo_seed.set(on);}
    /// The largest union of selected rows a group expanded while the test above was on.
    pub fn forced_union(&self)->Option<usize> {self.forced_union.get()}
    /// One chat's prompt rows in one group. No committed state changes until
    /// every layer and the head succeed. Equal bits to `step` for each row.
    pub fn prefill_rows(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows)->Result<(Vec<Vec<f32>>,State<C::Value>)> {
        self.rows_pass(tokens,state,head,0,None).map(|(logits,next,_)|(logits,next))
    }
    /// The same group, which also keeps what `prefix_state` needs.
    pub fn prefill_rows_open(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows)->Result<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)> {
        let (logits,next,open)=self.rows_pass(tokens,state,head,2,None)?;Ok((logits,next,open.ok_or("prompt group kept no rows")?))
    }
    /// The same group for a greedy caller: each row from `first` on answers with the ID its head picks, taken where
    /// the logits are (`Compute::rows_greedy`, the pick of the batch step). No logits come to the host. The same
    /// state and the same kept rows as `prefill_rows_open` with `HeadRows::From(first)`, and the ID of each row is
    /// the best of that pass's logits (the lower ID of two equal values).
    pub fn prefill_rows_open_greedy(&self,tokens:&[u32],state:&State<C::Value>,first:usize)->Result<(Vec<u32>,State<C::Value>,OpenRows<C::Value>)> {
        let mut ids=Vec::new();let (_,next,open)=self.rows_pass(tokens,state,HeadRows::From(first),2,Some(&mut ids))?;
        need(first<tokens.len() && ids.len()==tokens.len()-first,"prompt greedy row count differs")?;Ok((ids,next,open.ok_or("prompt group kept no rows")?))
    }
    /// The same group, which keeps only each row's last hidden state (`open_hidden`), for a
    /// proposer's cache of prompt positions. `prefix_state` does not apply to it.
    pub fn prefill_rows_hidden(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows)->Result<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)> {
        let (logits,next,open)=self.rows_pass(tokens,state,head,1,None)?;Ok((logits,next,open.ok_or("prompt group kept no hidden rows")?))
    }
    /// The state after the first `rows` rows of a group, from the group's own results: appended
    /// caches are prefixes of the group's caches, and each recurrent layer runs its two state
    /// producers again on the kept inputs of those rows. Equal bits to a group of only those rows.
    pub fn prefix_state(&self,open:&OpenRows<C::Value>,before:&State<C::Value>,full:&State<C::Value>,rows:usize)->Result<State<C::Value>> {
        let d=self.checkpoint.description();let b=&self.compute;
        need(rows>0 && rows<open.rows && before.position==open.first && full.position==open.first+open.rows as u64 && open.layers.len()==d.layers.len() && full.layers.len()==d.layers.len() && before.layers.len()==d.layers.len(),"prefix state differs from its group")?;
        let last=open.first as usize+rows;let mut layers=Vec::new();
        for layer in 0..d.layers.len() {let mut next=BTreeMap::new();let whole=&full.layers[layer];
            let part=|name:&str,words:usize|->Result<C::Value> {b.range(whole.get(name).ok_or("prefix state has no cache")?,0,words)};
            match d.layers[layer].mixer {
                TokenMixer::Kda(k)=>{let kept=open.layers[layer].as_ref().ok_or("prefix state has no kept rows")?;let old=&before.layers[layer];
                    let mut s=row_shape(Row::HistoryRows,rows)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;
                    let (lengths,_,_,_)=s.lengths().ok_or("invalid prefix history shape")?;
                    let x=[b.range(&kept.conv[0],0,lengths[0])?,b.range(&kept.conv[1],0,lengths[1])?,b.range(&kept.conv[2],0,lengths[2])?];
                    next.insert("kda.history".to_owned(),b.prompt(s,[Some(&x[0]),Some(&x[1]),Some(&x[2]),old.get(&kept.history),None],&[])?);
                    let mut s=row_shape(Row::StepRows,rows)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;
                    let (lengths,_,_,_)=s.lengths().ok_or("invalid prefix step shape")?;
                    let columns=b.prompt(s,[Some(&b.range(&kept.step,0,lengths[0])?),old.get(&kept.state),Some(&b.range(&kept.decay,0,lengths[2])?),None,None],&[])?;
                    next.insert("kda.state".to_owned(),b.range(&columns,0,lengths[1])?);},
                TokenMixer::LatentAttention(a)=>{let width=a.indexer.dim as usize;
                    next.insert("latent_cache".to_owned(),part("latent_cache",last*a.kv_rank as usize)?);
                    next.insert("index_keys".to_owned(),part("index_keys",last*width)?);next.insert("index_gates".to_owned(),part("index_gates",last*width)?);
                    next.insert("pool_keys".to_owned(),part("pool_keys",(last/a.indexer.pool as usize)*width)?);},
            }
            need(next.len()==whole.len(),"prefix state names differ")?;layers.push(next);
        }
        // Held rows are the first rows of one cache, so a prefix holds the first of the same rows.
        let mut expanded=BTreeMap::new();
        for (&layer,Expanded(cache,held,layout)) in &full.expanded {
            let TokenMixer::LatentAttention(_)=d.layers[layer].mixer else {return Err("held rows need a latent layer".into());};
            need(*layout==self.sparse_layout(layer)?,"expanded prefix layout differs")?;
            if *held<=last {expanded.insert(layer,Expanded(cache.clone(),*held,*layout));}
            else {expanded.insert(layer,Expanded(std::rc::Rc::new(b.range(cache,0,last*layout.row_words()/2)?),last,*layout));}
        }
        Ok(State {position:last as u64,layers,expanded})
    }
    /// The last hidden state of one row of an open group, before and after the final norm.
    /// The last `keep` rows of a prompt group's open rows, in storage of their own. A group larger than the predictor's
    /// trail must not hold its whole hidden planes while the chat waits: a range of them would be a view that keeps both planes
    /// (32,768 B a row). Rows a verdict still needs (kept layer inputs) are left as they are.
    pub fn open_tail(&self,open:OpenRows<C::Value>,keep:usize)->Result<OpenRows<C::Value>> {
        if keep==0 || open.rows<=keep || open.layers.iter().any(|kept|kept.is_some()) {return Ok(open);}
        let hidden=self.checkpoint.description().hidden as usize;let skip=open.rows-keep;
        let own=|plane:&C::Value|self.compute.rows_append(None,0,&self.compute.range(plane,skip*hidden,keep*hidden)?,hidden);
        let hidden_rows=[own(&open.hidden[0])?,own(&open.hidden[1])?];
        Ok(OpenRows {first:open.first+skip as u64,rows:keep,layers:open.layers,hidden:hidden_rows})
    }
    pub fn open_hidden(&self,open:&OpenRows<C::Value>,row:usize)->Result<[C::Value;2]> {
        let h=self.checkpoint.description().hidden as usize;need(row<open.rows,"open row is outside its group")?;
        Ok([self.compute.range(&open.hidden[0],row*h,h)?,self.compute.range(&open.hidden[1],row*h,h)?])
    }
    /// `keep`: 0 nothing, 1 the rows' hidden states, 2 also the recurrent layers' kept inputs.
    /// Prompt groups keep launch recording paused (PROVED-3): a new row count or chunk position then
    /// captures no graph pieces. Verification rows repeat a small bounded set of shapes, so they keep
    /// the caller's recording state and let resident pieces replay. Both forms enqueue the same commands
    /// in the same order. A prompt group's recording state is restored also when the group fails.
    /// Warm multi-row device route tables, descriptor scratch, and placement tables across all target routed layers
    /// for verify passes up to `max_rows` rows.
    pub fn warm_verify_routes(&self,max_rows:usize)->Result<()> {
        if !self.predictor_enabled() {return Ok(());}
        let max_rows=max_rows.min(VERIFY_ROWS).min(self.capacity as usize);if max_rows<2 {return Ok(());}
        let d=self.checkpoint.description();
        let mut schedules=self.schedules.borrow_mut();
        for layer in 0..d.layers.len() {
            let plan=schedules.get_mut(layer).ok_or("verify warm layer is outside the recipe")?.at(1).cloned().map_err(|e|e.to_string())?;
            let Some(picks)=plan.steps.iter().find(|s|s.kind==Kind::Route).and_then(|s|s.outputs.first()).and_then(|o|o.shape.first()).map(|&k|k as usize) else {continue;};
            for (index,step) in plan.steps.iter().enumerate() {
                if step.kind==Kind::ExpertDown {self.compute.warm_routed_rows(std::slice::from_ref(step),picks,max_rows)?;}
                else if step.id=="expert_gate" && plan.steps.get(index+1).is_some_and(|up|up.id=="expert_up" && up.kind==Kind::ExpertProject) {
                    self.compute.warm_routed_rows(&plan.steps[index..index+2],picks,max_rows)?;
                }
            }
        }
        self.verify_routes.set(self.verify_routes.get().max(max_rows));
        Ok(())
    }
    /// `picked`: the head rows answer with their IDs, pushed there, and the logits stay empty (`prefill_rows_open_greedy`).
    fn rows_pass(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows,keep:u8,picked:Option<&mut Vec<u32>>)->Result<(Vec<Vec<f32>>,State<C::Value>,Option<OpenRows<C::Value>>)> {
        // A check is already inside the resident queue opened by `Compute::work`. Unlike prompt
        // groups, its 2..=8-row shapes recur every cycle and can amortize retained pieces.
        let paused=if keep==2 {None} else {Some(self.compute.rows_unrecorded(true)?)};
        self.compute.rows_verify(keep==2 && self.predictor_enabled() && (2..=self.verify_routes.get()).contains(&tokens.len()));
        let result=self.rows_pass_body(tokens,state,head,keep,picked);
        self.compute.rows_verify(false);
        // A failed large group must end its extra spare capacity too. Keep the
        // original error if the device cannot synchronize for this cleanup.
        if result.is_err() && group_pool_cap(tokens.len()).is_some() {let _=self.compute.rows_keep();}
        let restored=paused.map_or(Ok(()),|was|self.compute.rows_unrecorded(was).map(|_|()));let value=result?;restored?;Ok(value)
    }
    fn rows_pass_body(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows,keep:u8,mut picked:Option<&mut Vec<u32>>)->Result<(Vec<Vec<f32>>,State<C::Value>,Option<OpenRows<C::Value>>)> {
        let c=self.checkpoint;let d=c.description();let m=tokens.len();
        need(m>0 && m<=PROMPT_GROUP_CHECK && state.layers.len()==d.layers.len(),"prompt group differs")?;
        need(state.position.checked_add(m as u64).is_some_and(|n|n<=self.capacity && n<=u32::MAX as u64),"prompt group exceeds capacity")?;
        need(tokens.iter().all(|&t|u64::from(t)<d.vocab),"prompt token exceeds vocabulary")?;
        if trims_spares_first(m,state.position) {self.compute.rows_keep()?;}
        let streams=d.residual.streams as usize;let hidden=d.hidden as usize;
        let embed=|tokens:&[u32]|->Result<C::Value> {match self.compute.rows_embedding(tokens,streams)? {
            Some(value)=>Ok(value),
            None=>{let rows=tokens.iter().map(|&t|self.embedding(t)).collect::<Result<Vec<_>>>()?;self.compute.join(&rows.iter().collect::<Vec<_>>())},
        }};
        // WIDE GROUP: a prompt group above PROMPT_GROUP_MAX rows passes every layer as runs of at most that many rows
        // (`layer_wide`), and the residual rows of each run stay a value of their own from the embedding to the last
        // layer: a run's rows are never copied out of or into the group's rows between two layers.
        if m>PROMPT_GROUP_MAX && keep<2 {
            let mut runs=tokens.chunks(PROMPT_GROUP_MAX).map(|run|Ok((run.len(),embed(run)?))).collect::<Result<Vec<_>>>()?;
            let mut layers=Vec::new();let mut expanded=BTreeMap::new();
            for layer in 0..d.layers.len() {
                let cut=self.wide_cut(layer,state.position+m as u64)?;
                let (outputs,next,held)=self.layer_wide(layer,state,runs,cut.as_ref().map(|(cut,carry)|(*cut,carry.as_str())))?;
                runs=outputs;layers.push(next);
                if let Some(held)=held {expanded.insert(layer,held);}
            }
            // The stream mean and the final norm of each run (a row depends on that row alone), then one value for the head.
            let mut s=shape(Op::StreamMean);s.heads=number(d.residual.streams)?;s.dim=number(d.hidden)?;let name="model.language_model.norm.weight";
            let (mut means,mut norms)=(Vec::with_capacity(runs.len()),Vec::with_capacity(runs.len()));
            for (n,residual) in runs {
                let mean=self.compute.prompt(RowShape::wrapped(s,n).ok_or("prompt row count exceeds ABI")?,[Some(&residual),None,None,None,None],&[])?;drop(residual);
                norms.push(self.compute.rows_norm(name,norm_shape(c,name,hidden)?,&mean,n,&self.parameters[name])?);means.push(mean);
            }
            let normalized=self.compute.join(&norms.iter().collect::<Vec<_>>())?;drop(norms);
            let logits=self.head_rows(&normalized,m,head,picked.as_deref_mut())?;
            if m>=PROMPT_SHARED {self.compute.rows_keep()?;}
            let open=if keep>0 {let mean=self.compute.join(&means.iter().collect::<Vec<_>>())?;
                Some(OpenRows {first:state.position,rows:m,layers:(0..d.layers.len()).map(|_|None).collect(),hidden:[mean,self.compute.range(&normalized,0,m*hidden)?]})} else {None};
            return Ok((logits,State {position:state.position+m as u64,layers,expanded},open));
        }
        let mut residual=embed(tokens)?;
        let mut layers=Vec::new();let mut expanded=BTreeMap::new();let mut kept_layers=Vec::new();
        for layer in 0..d.layers.len() {
            let (output,next,cache,kept)=self.layer_rows(layer,state.position,m,residual,&state.layers[layer],state.expanded.get(&layer).filter(|v|v.1<=state.position as usize && v.1<=self.expanded_limit.get()),keep==2,None)?;
            residual=output;layers.push(next);kept_layers.push(kept);
            // A group past the held limit adds no rows, but the rows held before it stay:
            // they are the first rows of the same cache and admission charges them for the whole chat.
            if let Some(cache)=cache {expanded.insert(layer,Expanded(std::rc::Rc::new(cache),state.position as usize+m,self.sparse_layout(layer)?));}
            else if let Some(held)=state.expanded.get(&layer) {expanded.insert(layer,held.clone());}
        }
        let mut s=shape(Op::StreamMean);s.heads=number(d.residual.streams)?;s.dim=number(d.hidden)?;
        let mean=self.compute.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,[Some(&residual),None,None,None,None],&[])?;
        let name="model.language_model.norm.weight";
        let normalized=self.compute.rows_norm(name,norm_shape(c,name,hidden)?,&mean,m,&self.parameters[name])?;
        let logits=self.head_rows(&normalized,m,head,picked)?;
        // Keep the group's spares, trimmed to the pool's normal room, instead of releasing them:
        // the next group, or the next session's first group, reuses the same buffer sizes.
        if m>=PROMPT_SHARED {self.compute.rows_keep()?;}
        // Each row's last hidden state, before and after the final norm, for a proposer.
        let open=if keep>0 {Some(OpenRows {first:state.position,rows:m,layers:kept_layers,hidden:[self.compute.range(&mean,0,m*hidden)?,self.compute.range(&normalized,0,m*hidden)?]})} else {None};
        Ok((logits,State {position:state.position+m as u64,layers,expanded},open))
    }
    /// The logits of the head rows of a group of `m` rows, from its rows after the final norm.
    fn head_rows(&self,normalized:&C::Value,m:usize,head:HeadRows,mut picked:Option<&mut Vec<u32>>)->Result<Vec<Vec<f32>>> {
        let d=self.checkpoint.description();let hidden=d.hidden as usize;
        let vocab=d.vocab as usize;let mut logits=Vec::new();
        let first=match head {HeadRows::Last=>m-1,HeadRows::All=>0,HeadRows::From(row)=>{need(row<m,"prompt head row is outside its group")?;row}};
        // Each head row has its own launch group, so the logits stay bounded.
        let mut at=first;while at<m {let count=(m-at).min(32);
            let rows=self.compute.range(normalized,at*hidden,count*hidden)?;
            let projected=self.project_group("lm_head.weight",&rows,count,true)?;
            // A greedy caller: each row's ID is picked where its logits are, and three words a row are read, not the
            // row. A backend that cannot pick there reads the rows and picks on the host by the same rule.
            if let Some(ids)=picked.as_deref_mut() && let Some(best)=self.compute.rows_greedy(&projected,count)? {
                need(best.len()==count && best.iter().all(|&id|u64::from(id)<d.vocab),"prompt greedy rows differ")?;ids.extend(best);at+=count;continue;}
            let values=self.compute.read(&projected)?;
            need(values.len()==count*vocab && values.iter().all(|v|v.is_finite()),"prompt logits differ")?;
            match picked.as_deref_mut() {
                Some(ids)=>for row in values.chunks_exact(vocab) {ids.push(greedy_logits(row,vocab)?);},
                None=>logits.extend(values.chunks_exact(vocab).map(<[f32]>::to_vec)),
            }at+=count;}
        Ok(logits)
    }
    /// Token-major `[m,rows]` from token-major `[m,cols]`, gathered on every rank.
    pub(super) fn project_group(&self,name:&str,input:&C::Value,m:usize,round:bool)->Result<C::Value> {
        if let Some(value)=self.compute.rows_project(self.checkpoint,name,input,m,round)? {return Ok(value);}
        let view=self.checkpoint.headers().weights().get(name).ok_or("missing prompt matrix")?;let cols=view.logical().shape[1] as usize;
        let mut rows=Vec::new();for token in 0..m {rows.push(self.project(name,&self.compute.range(input,token*cols,cols)?,round)?);}
        self.compute.join(&rows.iter().collect::<Vec<_>>())
    }
    /// WIDE GROUP: where a layer's program is cut for a group above `PROMPT_GROUP_MAX` rows: the index of the second
    /// half's first step (the last mHC prepare, `ffn.prepare`) and the one value that half reads from the steps before it
    /// (`after_attention`, one residual row a token). The mixer half before the cut holds every step that reads or makes
    /// history (convolution, recurrence, caches, pools, selection, attention); the half after it (the routed experts, or
    /// a dense block) is stateless: each of its rows depends on that row alone. None: the program is not of this form (a
    /// step of the second half reads history, or reads another earlier value), and the layer runs whole.
    fn wide_cut(&self,layer:usize,last:u64)->Result<Option<(usize,String)>> {
        let mut schedules=self.schedules.borrow_mut();
        let plan=schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(last).map_err(|e|e.to_string())?;
        Ok(wide_cut_of(plan))
    }
    /// WIDE GROUP: one layer of a prompt group above `PROMPT_GROUP_MAX` rows, given and returned as its runs of at most
    /// `PROMPT_GROUP_MAX` rows in token order: (rows, residual rows) of each run.
    /// `cut` (`wide_cut`): the layer's program has a mixer half and a stateless routed half. The mixer half runs a run
    /// at a time, each with the state and the held rows after the run before it: exactly what consecutive groups of
    /// those sizes do, so every kernel of that half sees a row count a served group already gives it. The routed half
    /// then runs for the SAME runs, a step at a time for every run (each row of that half depends on that row alone, so
    /// a run's steps are those of a group of its size), and only the routed expert products of all runs share ONE
    /// launch a list (`experts_wide`, WIDE COHORT): each expert's weights are then read from memory once for the whole
    /// group, which is the cost a group pays once. No cut: the whole layer runs a run at a time, as consecutive groups do.
    /// The bits are those of the smaller groups. A run's rows go from layer to layer as its own value, so what a run's
    /// last step of a layer leaves for the first step of the next (a residual pass that is finished with the next
    /// prepare) is what a group of its size leaves. Returns each run's output rows, the layer's next state, and its
    /// held rows when a run made or kept some.
    #[allow(clippy::type_complexity)] // one private call.
    fn layer_wide(&self,layer:usize,state:&State<C::Value>,runs:Vec<(usize,C::Value)>,cut:Option<(usize,&str)>)->Result<(Vec<(usize,C::Value)>,BTreeMap<String,C::Value>,Option<Expanded<C::Value>>)> {
        let m=runs.iter().map(|(n,_)|*n).sum::<usize>();
        if cut.is_some() {spark_gpu_probe::prompt_launches::record("wide_group_layer");}
        let mut held=state.expanded.get(&layer).cloned();let mut before:Option<BTreeMap<String,C::Value>>=None;
        // Each run: its rows and what its mixer half hands over (the layer's output when the layer has no cut).
        let mut made:Vec<(usize,C::Value)>=Vec::with_capacity(runs.len());let mut at=0usize;
        for (n,rows) in runs {
            let first=state.position+at as u64;
            let owner=held.as_ref().filter(|v|v.1<=first as usize && v.1<=self.expanded_limit.get());
            let (middle,next,cache,_)=self.layer_rows(layer,first,n,rows,before.as_ref().unwrap_or(&state.layers[layer]),owner,false,cut)?;
            // As between two groups: a run that made held rows replaces them, any other run keeps what was held.
            if let Some(cache)=cache {held=Some(Expanded(std::rc::Rc::new(cache),first as usize+n,self.sparse_layout(layer)?));}
            made.push((n,middle));before=Some(next);at+=n;
            // WIDE PHASE: the charge of a group above the whole-charged sizes holds ONE run in a mixer half
            // (`wide_phase_workspace`). The run's own values are dropped (`layer_rows` returned) and so is the state
            // before it (the assignment above); drain, so none of them is charged when the next run starts. The
            // carries in `made`, the new state and the held rows stay owned here.
            if m>PROMPT_GROUP_LEASE[1] {self.compute.rows_phase_end()?;}
        }
        let next=before.ok_or("wide prompt group has no rows")?;
        let Some((cut,carry))=cut else {return Ok((made,next,held));};
        // The routed half: the values of each run (first the one value its mixer half handed over).
        let mut runs:Vec<(usize,BTreeMap<String,C::Value>)>=made.into_iter().map(|(n,middle)|{let mut values=BTreeMap::new();values.insert(carry.to_owned(),middle);(n,values)}).collect();
        let last=state.position+m as u64;
        let plan={let mut schedules=self.schedules.borrow_mut();schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(last).cloned().map_err(|e|e.to_string())?};
        let mut passes:Vec<BTreeMap<String,C::Pass>>=runs.iter().map(|_|BTreeMap::new()).collect();
        let mut folds:Vec<RowsFold<C::Value>>=runs.iter().map(|_|RowsFold::of_group()).collect();
        // Steps a run has already done with an earlier step (`rows_stateless`: Some(n)).
        let mut skips=vec![0usize;runs.len()];
        for index in cut..plan.steps.len() {
            // WIDE COHORT: the routed expert products of every run in one launch a list, when every run is at this step.
            if skips.iter().all(|&skip|skip==0) && let Some(done)=self.experts_wide(layer,index,&plan,&mut runs,&mut folds)? {skips.fill(done);continue;}
            for (slot,(n,values)) in runs.iter_mut().enumerate() {
                if skips[slot]>0 {skips[slot]-=1;continue;}
                skips[slot]=self.rows_stateless(layer,index,&plan,*n,values,&mut passes[slot],&mut folds[slot])?.ok_or_else(||format!("{}: wide prompt group step reads history",plan.steps[index].id))?;
            }
        }
        let outputs=runs.into_iter().map(|(n,mut values)|Ok((n,values.remove("output").ok_or("prompt layer has no output")?))).collect::<Result<Vec<_>>>()?;
        // WIDE PHASE: the layer's end. Every value of the routed half but each run's output is dropped here (the
        // maps above, the passes, the folds); drain, so the next layer's first mixer run starts with none charged.
        if m>PROMPT_GROUP_LEASE[1] {drop(passes);drop(folds);self.compute.rows_phase_end()?;}
        Ok((outputs,next,held))
    }
    /// The fold shape sent to both the middle's producer and its sole reader.
    fn expert_fold_shape(&self,plan:&LayerPlan,step:&engine_model::plan::layer::Step)->Result<Shape> {
        let route=plan.steps.iter().find(|s|s.kind==Kind::Route).ok_or("prompt expert fold has no route")?;
        let mut s=shape(Op::ExpertFold);s.rows=number(route.outputs[0].shape[0])?;s.cols=number(self.checkpoint.description().hidden)?;s.taps=u32::from(step.inputs.len()==4);Ok(s)
    }
    /// WIDE COHORT: the grouped expert step `index` for every run of a wide prompt group at once, when it is one of the
    /// two that launch the routed expert products: the gate step (with up and their SwiGLU: `rows_experts_gathered_wide`)
    /// or the fold step whose grouped down step waited for it (`rows_experts_fold_wide`). Each run gets the tables,
    /// the lists and the values that `experts_group` gives its own call, so each run's result is the result of that
    /// call; the compute shares only the product launches. `runs`: (rows, values) of each run. `folds`: each run's fold state.
    /// Some(n): the step is done for every run, and the next n steps with it. None, before any launch: the step is not
    /// one of the two, or a run or the compute does not take the shared form; each run then takes the step on its own.
    fn experts_wide(&self,layer:usize,index:usize,plan:&LayerPlan,runs:&mut [(usize,BTreeMap<String,C::Value>)],folds:&mut [RowsFold<C::Value>])->Result<Option<usize>> {
        let b=&self.compute;let step=&plan.steps[index];
        if runs.len()<2 || runs.iter().any(|(n,_)|b.routes_rows(*n)) {return Ok(None);}
        match step.kind {
            Kind::ExpertProject if step.id=="expert_gate"=>{
                let (Some(up),Some(swiglu))=(plan.steps.get(index+1),plan.steps.get(index+2)) else {return Ok(None);};
                if up.id!="expert_up" || up.kind!=Kind::ExpertProject || swiglu.kind!=Kind::SwiGlu || runs.iter().any(|(_,values)|!values.contains_key(&step.inputs[0]) || !values.contains_key(&step.inputs[1])) {return Ok(None);}
                let limit=self.checkpoint.description().swiglu_limit as f32;
                if let Some(links)=self.fold_links(layer,plan).filter(|links|plan.steps[links.down].kind==Kind::ExpertDown) {
                    let shared=links.shared.and_then(|(at,_)|plan.steps[at].weights.first()).map(|w|w.name.as_str());
                    let calls=runs.iter().map(|(m,v)|(*m,&v[&step.inputs[0]],&v[&step.inputs[1]])).collect::<Vec<_>>();
                    let steps=[step.clone(),up.clone(),plan.steps[links.down].clone()];
                    if let Some(made)=b.rows_experts_device_wide(&steps,limit,&calls,(self.expert_fold_shape(plan,&plan.steps[links.fold])?,shared))? {
                        need(made.len()==runs.len(),"device expert result count differs")?;
                        for ((_,values),middle) in runs.iter_mut().zip(made) {values.insert("expert_middle_grouped".to_owned(),middle);}return Ok(Some(2));
                    }
                }
                let tables=runs.iter().map(|(n,values)|self.expert_tables(plan,step,*n,values)).collect::<Result<Vec<_>>>()?;
                let lists=tables.iter().map(|tables|(expert_names(step,&tables.spans),expert_names(up,&tables.spans))).collect::<Vec<_>>();
                // PACKED MIDDLE: when every run's grouped down step will wait for its fold (`rows_group_fold_waits`: a group
                // that folds, the fold's own down step), the fold's product launch is the only reader of the middle
                // (`expert_middle_grouped` is read by nothing else, see the fold branch below), so the compute may keep
                // it in that launch's own form. The down list of a run names the same experts and counts as its gate list.
                let down=self.fold_links(layer,plan).filter(|links|runs.iter().all(|(n,_)|b.folds_groups(*n)) && plan.steps.get(links.down).is_some_and(|down|down.kind==Kind::ExpertDown))
                    .map(|links|(links,tables.iter().map(|tables|expert_names(&plan.steps[links.down],&tables.spans)).collect::<Vec<_>>()));
                let made={
                    let sets=runs.iter().zip(&tables).zip(&lists).map(|(((_,values),tables),(gate,up))|(gate.as_slice(),up.as_slice(),limit,&values[&step.inputs[0]],tables.order.as_slice(),tables.grouped.as_slice())).collect::<Vec<_>>();
                    match &down {
                        Some((links,down))=>{
                            let shared=links.shared.and_then(|(at,_)|plan.steps[at].weights.first()).map(|weight|weight.name.as_str());
                            b.rows_experts_gathered_wide_for(&sets,&down.iter().map(Vec::as_slice).collect::<Vec<_>>(),(self.expert_fold_shape(plan,&plan.steps[links.fold])?,shared))?},
                        None=>b.rows_experts_gathered_wide(&sets)?}};
                let Some(made)=made else {return Ok(None);};
                need(made.len()==runs.len(),"wide expert cohort result count differs")?;
                for ((_,values),middle) in runs.iter_mut().zip(made) {values.insert("expert_middle_grouped".to_owned(),middle);}
                Ok(Some(2))
            },
            Kind::ExpertSum=>{
                // Every run's grouped down step waited for this fold (`rows_stateless`, GROUP FOLD), with its middle in the grouped order.
                if !folds.iter().all(|fold|fold.waits && fold.grouped) || step.inputs.len()<3 {return Ok(None);}
                let Some(links)=self.fold_links(layer,plan).filter(|links|links.fold==index) else {return Ok(None);};
                let down=&plan.steps[links.down];
                // As `experts_group` refuses it: a fold with a shared row that is not this rank's own rows.
                if step.inputs.len()>=4 && folds.iter().any(|fold|fold.shared.is_none()) {return Ok(None);}
                if runs.iter().any(|(_,values)|!values.contains_key("expert_middle_grouped") || !values.contains_key(&down.inputs[1]) || step.inputs.iter().take(2).any(|name|!values.contains_key(name))) {return Ok(None);}
                let s=self.expert_fold_shape(plan,step)?;
                let calls=runs.iter().zip(folds.iter()).map(|((m,v),f)|(*m,&v["expert_middle_grouped"],&v[&step.inputs[0]],&v[&step.inputs[1]],step.inputs.get(3).and(f.shared.as_ref()))).collect::<Vec<_>>();
                if let Some(made)=b.rows_experts_fold_device_wide(down,&calls,s)? {
                    need(made.len()==runs.len(),"device fold result count differs")?;
                    for (((_,v),f),value) in runs.iter_mut().zip(folds.iter_mut()).zip(made) {f.waits=false;f.grouped=false;f.shared=None;v.remove("expert_middle_grouped");v.insert(step.outputs[0].name.clone(),value);}return Ok(Some(0));
                }
                let tables=runs.iter().map(|(n,values)|self.expert_tables(plan,down,*n,values)).collect::<Result<Vec<_>>>()?;
                let lists=tables.iter().map(|tables|expert_names(down,&tables.spans)).collect::<Vec<_>>();
                let made={
                    let sets=runs.iter().zip(folds.iter()).zip(&tables).zip(&lists).map(|((((n,values),fold),tables),list)|(list.as_slice(),&values["expert_middle_grouped"],tables.placed.as_slice(),
                        &values[&step.inputs[0]],&values[&step.inputs[1]],step.inputs.get(3).and(fold.shared.as_ref()),s,*n)).collect::<Vec<_>>();
                    b.rows_experts_fold_wide(&sets)?};
                let Some(made)=made else {return Ok(None);};
                need(made.len()==runs.len(),"wide expert cohort result count differs")?;
                for (((_,values),fold),value) in runs.iter_mut().zip(folds.iter_mut()).zip(made) {
                    fold.waits=false;fold.grouped=false;fold.shared=None;values.remove("expert_middle_grouped");
                    values.insert(step.outputs[0].name.clone(),value);
                }
                Ok(Some(0))
            },
            _=>Ok(None),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn layer_rows(&self,layer:usize,first:u64,m:usize,residual:C::Value,state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,keep:bool,cut:Option<(usize,&str)>)->Result<(C::Value,BTreeMap<String,C::Value>,Option<C::Value>,Option<KeptRows<C::Value>>)> {
        let b=&self.compute;let last=first+m as u64;
        let mut chat=self.rows_chat(layer,first,m,state,expanded,keep)?;
        let planning=std::time::Instant::now();
        let plan={let mut schedules=self.schedules.borrow_mut();schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(last).cloned().map_err(|e|e.to_string())?};
        b.plan_cost("planning",planning.elapsed().as_nanos());
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        // GROUP FOLD: a prompt group (no kept rows) may fold a routed layer's down rows before their exchange.
        let mut passes=BTreeMap::new();let mut fold=if keep {RowsFold::default()} else {RowsFold::of_group()};let mut skip=0;
        for index in 0..cut.map_or(plan.steps.len(),|cut|cut.0) {
            if skip>0 {skip-=1;continue;}
            if let Some(done)=self.rows_stateless(layer,index,&plan,m,&mut values,&mut passes,&mut fold)? {skip=done;continue;}
            skip=self.rows_history(layer,index,&plan,&mut chat,&mut values)?;
        }
        need(skip==0,"prompt layer cut is inside a stage")?;
        let output=values.remove(cut.map_or("output",|cut|cut.1)).ok_or("prompt layer has no output")?;
        let (next,cache,kept)=self.rows_chat_end(layer,chat,&mut values)?;
        Ok((output,next,cache,kept))
    }
    /// One chat's side of a row pass of `m` rows that start at position `first` (`RowsChat`).
    fn rows_chat<'a>(&self,layer:usize,first:u64,m:usize,state:&'a BTreeMap<String,C::Value>,expanded:Option<&'a Expanded<C::Value>>,keep:bool)->Result<RowsChat<'a,C::Value>> {
        if let Some(owner)=expanded {need(owner.2==self.sparse_layout(layer)?,"expanded prompt owner layout differs")?;}
        let last=first+m as u64;
        // With a proposer a pass follows a few steps, each time. Its held rows then follow the decoded rows
        // (admission charges them by the request's tokens), so no pass expands a decoded row twice.
        let persist=last as usize<=self.expanded_limit.get() && (first==0 || expanded.is_some_and(|v|v.1==first as usize || self.draft.is_some() && v.1<first as usize));
        Ok(RowsChat {first,m,state,expanded,keep,persist,kept_conv:None,kept_step:None,cache:None,cache_rows:last,selected:Vec::new(),selected_device:None,selection_checked:false})
    }
    /// What a chat's layer leaves: its next state (taken out of `values`, where its history steps put it), its new
    /// held rows when the pass holds them, and the kept inputs of a recurrent layer.
    #[allow(clippy::type_complexity)] // one private result: the three things a layer leaves a chat.
    fn rows_chat_end(&self,layer:usize,chat:RowsChat<'_,C::Value>,values:&mut BTreeMap<String,C::Value>)->Result<(BTreeMap<String,C::Value>,Option<C::Value>,Option<KeptRows<C::Value>>)> {
        let d=self.checkpoint.description();
        let names=match d.layers[layer].mixer {
            TokenMixer::Kda(_)=>vec![("kda.history","next_history"),("kda.state","next_state")],
            // The complete pools ride with the state, as in the step path.
            TokenMixer::LatentAttention(_)=>vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")],
        };
        let mut next=BTreeMap::new();
        for (input,name) in names {next.insert(input.into(),values.remove(name).ok_or("prompt layer has no next state")?);}
        let kept=match (chat.kept_conv,chat.kept_step) {(Some((conv,history)),Some((step,decay,state)))=>Some(KeptRows {conv,history,step,decay,state}),_=>None};
        need(!chat.keep || kept.is_some()==matches!(d.layers[layer].mixer,TokenMixer::Kda(_)),"recurrent layer kept no rows")?;
        Ok((next,if chat.persist {chat.cache} else {None},kept))
    }
    /// One step of a layer that reads ONE chat's history, for that chat's rows of a row pass: `values` holds the chat's
    /// own rows of every value the step reads, and gets what the step makes. The one-chat pass gives its own value map
    /// (`layer_rows`); a shared pass gives each chat's map in turn (rows/shared.rs). Returns how many of the next steps
    /// are done with this one.
    fn rows_history(&self,layer:usize,index:usize,plan:&LayerPlan,chat:&mut RowsChat<'_,C::Value>,values:&mut BTreeMap<String,C::Value>)->Result<usize> {
        let d=self.checkpoint.description();let b=&self.compute;let step=&plan.steps[index];
        let (first,m,state,expanded,keep,persist)=(chat.first,chat.m,chat.state,chat.expanded,chat.keep,chat.persist);let last=first+m as u64;
        // A check pass (it keeps what a prefix needs) of few rows behind earlier rows appends to its caches in place, with the plain step's room behind them (see CacheAppend below).
        let in_place=keep && first>0 && m<=VERIFY_ROWS;
        let room=|width:usize|->Result<usize> {(self.capacity.saturating_sub(last).min(HELD_ROOM as u64) as usize).checked_mul(width).ok_or_else(||"cache room overflow".to_owned())};
            match step.kind {
                Kind::Conv=>{if let Some(done)=self.rows_recurrent_span(layer,index,plan,chat,values)? {return Ok(done);}let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("convolution requires a recurrent layer".into());};
                    let mut s=row_shape(Row::ConvRows,m)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;
                    let x=[Some(&values[&step.inputs[0]]),Some(&values[&step.inputs[1]]),Some(&values[&step.inputs[2]]),state.get(&step.inputs[3]),None];
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                    let (convolved,history)=self.conv_rows(s,x,&weight)?;
                    if keep {let (lengths,_,_,_)=s.lengths().ok_or("invalid convolution row shape")?;
                        chat.kept_conv=Some(([b.range(&values[&step.inputs[0]],0,lengths[0])?,b.range(&values[&step.inputs[1]],0,lengths[1])?,b.range(&values[&step.inputs[2]],0,lengths[2])?],step.inputs[3].clone()));}
                    values.insert(step.outputs[0].name.clone(),convolved);values.insert(step.outputs[1].name.clone(),history);},
                Kind::KdaStep=>{let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("recurrence requires a recurrent layer".into());};
                    let mut s=row_shape(Row::StepRows,m)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;
                    // Decay does not depend on the recurrent state. Only the update runs in token order.
                    let decay=b.prompt(RowShape {op:Row::StepDecay,..s},[Some(&values[&step.inputs[0]]),None,None,None,None],&[])?;
                    let columns=b.prompt(s,[Some(&values[&step.inputs[0]]),state.get(&step.inputs[1]),Some(&decay),None,None],&[])?;
                    if keep {let (lengths,_,_,_)=s.lengths().ok_or("invalid recurrence row shape")?;
                        chat.kept_step=Some((b.range(&values[&step.inputs[0]],0,lengths[0])?,b.range(&decay,0,lengths[2])?,step.inputs[1].clone()));}
                    let (h,d)=(self.kda_layout(layer)?.local_heads(),k.head_dim as usize);let output=b.range(&columns,h*d*d,m*h*d)?;let next=b.range(&columns,0,h*d*d)?;
                    values.insert(step.outputs[0].name.clone(),output);values.insert(step.outputs[1].name.clone(),next);},
                Kind::CacheAppend=>{let old=state.get(&step.inputs[0]).ok_or("prompt cache state is absent")?;let new=&values[&step.inputs[1]];
                    // A CHECK PASS APPENDS WHERE THE CACHE IS, as the plain step does (`Compute::cache_append`). A join buys a buffer that grows with the position in every pass and copies the whole cache into it,
                    // and the next plain step then copies the cache once more into storage with room. A refused row leaves nothing to undo: the state before the pass keeps its own shorter view, the kept prefix is a
                    // view of the first rows of the same storage (`prefix_state`), and an append writes only past the end of every live view (`held::append_in_place`), so the next pass writes over the refused rows;
                    // while anyone still reads them, the append copies as before. The words are those of the join. A prompt group joins as before.
                    let value=if in_place {let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("cache append requires a latent layer".into());};
                        b.cache_append(old,new,first as usize,a.kv_rank as usize,room(a.kv_rank as usize)?)?} else {b.join(&[old,new])?};
                    values.insert(step.outputs[0].name.clone(),value);},
                Kind::Pool=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("pooling requires a latent layer".into());};
                    // A pool depends only on its own complete rows, so the last
                    // token's pools hold each earlier token's pools as a prefix.
                    let width=a.indexer.dim as usize;let total=last as usize;
                    let mut s=shape(Op::Pool);s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.rows=number(last)?;s.epsilon=d.epsilon as f32;
                    // Every completed earlier pool is kept. Only newly completed pools need a
                    // fold, with the same scalar arithmetic: one launch a pool in a small
                    // sequential pass, one launch over their window in a larger group.
                    // A larger group that starts inside the first pool has no earlier pool to keep: it
                    // takes the full fold below (its rows are the whole history), so no state value is a
                    // view that holds the window fold's scratch.
                    if first>0 && (m<=VERIFY_ROWS || first>=a.indexer.pool)
                        && let Some(prior)=state.get(&step.outputs[2].name) {
                            let old_keys=state.get(&step.inputs[0]).ok_or("prompt index keys are absent")?;
                            let old_gates=state.get(&step.inputs[1]).ok_or("prompt index gates are absent")?;
                            let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                            let x=[old_keys,old_gates,&values[&step.inputs[2]],&values[&step.inputs[3]]];
                            // A check pass appends its key and gate rows where they are too (see CacheAppend above).
                            let result=if in_place {let room=room(width)?;pools::folds(b,s,first as usize,m,[b.cache_append(x[0],x[2],first as usize,width,room)?,b.cache_append(x[1],x[3],first as usize,width,room)?],prior,&weight)?}
                                else if m<=VERIFY_ROWS {pools::append(b,s,first as usize,m,x,prior,&weight)?} else {pools::append_window(b,s,first as usize,m,x,prior,&weight)?};
                            for (spec,value) in step.outputs.iter().zip(result) {values.insert(spec.name.clone(),value);}
                            return Ok(0);
                        }
                    let prefix=|old:&C::Value,new:&C::Value|{let body=b.range(new,0,(m-1)*width)?;b.join(&[old,&body])};
                    let keys=prefix(state.get(&step.inputs[0]).ok_or("prompt index keys are absent")?,&values[&step.inputs[2]])?;
                    let gates=prefix(state.get(&step.inputs[1]).ok_or("prompt index gates are absent")?,&values[&step.inputs[3]])?;
                    let key=b.range(&values[&step.inputs[2]],(m-1)*width,width)?;let gate=b.range(&values[&step.inputs[3]],(m-1)*width,width)?;
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                    let result=b.operation(s,[Some(&keys),Some(&gates),Some(&key),Some(&gate)],&weight)?;
                    let len=total*width;values.insert(step.outputs[0].name.clone(),b.range(&result,0,len)?);values.insert(step.outputs[1].name.clone(),b.range(&result,len,len)?);
                    values.insert(step.outputs[2].name.clone(),b.range(&result,2*len,(total/a.indexer.pool as usize)*width)?);},
                Kind::ScorePools=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("pool scores require a latent layer".into());};
                    let pools=last/a.indexer.pool;
                    let value=if pools==0 {b.zeros(0)?} else {let mut s=row_shape(Row::PoolScoresRows,m)?;s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.rows=number(pools)?;s.first=number(first+1)?;
                        b.prompt(s,[Some(&values[&step.inputs[0]]),Some(&values[&step.inputs[1]]),Some(&values[&step.inputs[2]]),None,None],&[])?};
                    values.insert(step.outputs[0].name.clone(),value);},
                Kind::Select=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("selection requires a latent layer".into());};
                    let i=a.indexer;let slots=i.token_budget+if i.select_tail {i.pool-1} else {0};let pools=last/i.pool;
                    let mut s=row_shape(Row::SelectRows,m)?;s.dim=number(i.pool)?;s.taps=number(i.token_budget)?;s.rows=number(pools)?;s.cols=u32::from(i.select_tail);s.first=number(first+1)?;s.slots=number(slots)?;
                    let empty;let scores=if pools==0 {empty=b.zeros(0)?;&empty} else {&values[&step.inputs[0]]};
                    let value=b.prompt(s,[Some(scores),None,None,None,None],&[])?;
                    let words=slots as usize+1;
                    // Absorbed attention consumes the device table directly. Read only one
                    // validation status per token, retaining every count and causal-ID check.
                    let absorbed=plan.steps.iter().find(|step|step.kind==Kind::ExpandKv)
                        .is_some_and(|step|b.has_latent_absorption(&step.weights[0].name));
                    if m>VERIFY_ROWS && absorbed && self.forced_union.get().is_none() {
                        let mut check=row_shape(Row::CheckSelectionRows,m)?;check.slots=number(slots)?;check.first=number(first+1)?;
                        let status=b.prompt(check,[Some(&value),None,None,None,None],&[])?;
                        let status=b.read(&status)?;
                        need(status.len()==m && status.iter().all(|&v|v.to_bits()==0),"absorbed prompt selection is invalid")?;
                        spark_gpu_probe::prompt_launches::record("prompt_selection_checked");
                        chat.selected_device=Some(b.range(&value,0,m*words)?);chat.selection_checked=true;
                        values.insert(step.outputs[0].name.clone(),value);return Ok(0);
                    }
                    // For small row groups below the pool budget, every token keeps all complete pools, so
                    // IDs and counts are known on the host while `select_known_rows` defers NaN count checks
                    // to the step's final readback.
                    if m<=VERIFY_ROWS
                        && let Some(mut known)=spark_gpu_probe::select_admission::known_selection_rows(s)
                        && b.select_known_rows(&value,slots as usize,&known)?
                    {
                        let host=std::time::Instant::now();
                        if self.forced_union.get().is_some() {force_distinct_rows(&mut known,slots as usize,first as usize);}
                        b.plan_cost("prompt_selection",host.elapsed().as_nanos());
                        let uploaded=b.upload_known(&known)?;
                        chat.selected_device=Some(b.range(&uploaded,0,known.len())?);
                        chat.selected=known;
                        values.insert(step.outputs[0].name.clone(),uploaded);
                    } else {
                        let read=b.read(&value)?;need(read.len()==m*words,"prompt selection extent differs")?;
                        let host=std::time::Instant::now();
                        for (token,row) in read.chunks_exact(words).enumerate() {need(row[words-1]>=0.0 && row[words-1] as u64<=slots.min(first+1+token as u64),"prompt selection count is invalid")?;}
                        chat.selected=read;if self.forced_union.get().is_some() {force_distinct_rows(&mut chat.selected,slots as usize,first as usize);}
                        b.plan_cost("prompt_selection",host.elapsed().as_nanos());
                        // Keep the selection that was just checked. Its device words are unchanged;
                        // only a forced test union needs the modified host table uploaded below.
                        chat.selected_device=if self.forced_union.get().is_none() {Some(b.range(&value,0,chat.selected.len())?)} else {None};
                        values.insert(step.outputs[0].name.clone(),value);
                    }},
                Kind::Gather=>{},
                Kind::ExpandKv=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("KV expansion requires a latent recipe".into());};
                    if b.has_latent_absorption(&step.weights[0].name) {
                        let (query_name,output_name)=indirect::attention_tail(&plan,index).ok_or("absorbed prompt tail is not private")?;
                        let layout=self.sparse_layout(layer)?;let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}) as usize;
                        if !chat.selection_checked {need(chat.selected.len()==m*(slots+1),"absorbed prompt selection extent differs")?;}
                        let queries=self.local_query(&values[query_name],m,layout)?;
                        let host=std::time::Instant::now();
                        for (token,row) in chat.selected.chunks_exact(slots+1).enumerate() {
                            let count=row[slots];need(count.is_finite() && count>=0.0 && count.fract()==0.0 && count<=slots.min(first as usize+token+1) as f32,"absorbed prompt count differs")?;
                            need(row[..count as usize].iter().all(|&id|id.is_finite() && id>=0.0 && id.fract()==0.0 && id<(first as usize+token+1) as f32),"absorbed prompt row exceeds its context")?;
                        }
                        b.plan_cost("prompt_selection",host.elapsed().as_nanos());
                        let uploaded;
                        let selected_value=match chat.selected_device.as_ref() {
                            Some(v)=>v,
                            None=>{uploaded=b.upload(&chat.selected)?;&uploaded},
                        };
                        let mut s=row_shape(Row::AbsorbQuery,m)?;
                        s.heads=number(layout.local_heads() as u64)?;s.dim=number(a.key_dim)?;s.taps=number(a.value_dim)?;
                        s.cols=number(a.kv_rank)?;s.rows=number(last)?;s.slots=number(slots as u64)?;s.first=number(slots.min(last as usize) as u64)?;
                        let local=b.latent_absorption_rows(&step.weights[0].name,s,&queries,&values["next_latent_cache"],selected_value)?;
                        values.insert(output_name.to_owned(),b.gather_attended(&local,m,layout)?);return Ok(3);
                    }
                    // The expansion of one latent row depends only on its bits.
                    // Each row is expanded once and every later token reuses it.
                    let name=&step.weights[0].name;let rank=a.kv_rank as usize;
                    let layout=self.sparse_layout(layer)?;let row_words=layout.row_words();
                    need(a.key_dim%2==0 && a.value_dim%2==0,"packed expanded rows need even key and value dims")?;let packed_words=row_words/2;
                    if persist {
                        let held=expanded.map_or(0,|v|v.1);let count=last as usize-held;
                        let new=if count==m {self.expand_owned_rows(name,&values["kv_latent"],m,layout)?} else {
                            // Rows that steps decoded since the last group: expanded here once, then held.
                            let latents=&values["next_latent_cache"];let mut parts=Vec::new();let mut at=0;
                            while at<count {let n=(count-at).min(PROMPT_GROUP);parts.push(self.expand_owned_rows(name,&b.range(latents,(held+at)*rank,n*rank)?,n,layout)?);at+=n;}
                            if parts.len()==1 {parts.remove(0)} else {b.join(&parts.iter().collect::<Vec<_>>())?}};
                        // Held rows are stored packed; the rounded projection is scratch once packed.
                        let new=self.pack_expanded(&new,count*row_words)?;
                        // With a proposer every pass appends: room for later rows, inside the held limit, so no pass copies the held rows.
                        let room=if self.draft.is_some() {HELD_ROOM.min(self.expanded_limit.get()-last as usize)*packed_words} else {0};
                        chat.cache=Some(b.rows_extend(expanded.map(|v|&*v.0),held,&new,packed_words,room)?);return Ok(0);
                    }
                    // Only rows some token selected. Selected rows are pure, so they
                    // need no order among themselves. New IDs point into this matrix.
                    let i=a.indexer;let slots=(i.token_budget+if i.select_tail {i.pool-1} else {0}) as usize;let words=slots+1;
                    // Reuse only the rows the caller already holds. The prompt does not
                    // extend the retained prefix or publish the temporary selected union.
                    let held=expanded.map(|v|v.1).unwrap_or(0);
                    let host=std::time::Instant::now();
                    let (rows,chosen,cached)=expansion_rows(&chat.selected,slots,last as usize,held)?;
                    if let Some(most)=self.forced_union.get() {self.forced_union.set(Some(most.max(rows.len())));}
                    b.plan_cost("prompt_selection",host.elapsed().as_nanos());
                    let mut cache_values=b.rows_reserve(rows.len()*packed_words)?;
                    // A range is a view on the retained allocation. Copy each contiguous
                    // run directly: no table, permutation output, or extra chunk is held.
                    let mut at=0;while at<cached {
                        let id=rows[at] as usize;let mut end=at+1;
                        while end<cached && rows[end] as usize==id+end-at {end+=1;}
                        let prefix=expanded.ok_or("prompt retained prefix is absent")?;
                        let part=b.range(&prefix.0,id*packed_words,(end-at)*packed_words)?;
                        b.rows_fill(&mut cache_values,at*packed_words,&part,(end-at)*packed_words)?;at=end;
                    }
                    // What the memo is seeded with: the rows the LAST token of the prompt selected. The first decode token
                    // selects nearly the same rows. The whole union of a 256-token group is most of a long prompt's rows.
                    let seed=if self.memo_seed.get() {
                        let row=&chat.selected[(m-1)*words..m*words];let mut ids=row[..(row[slots] as usize).min(slots)].iter().map(|&v|v as u32).collect::<Vec<_>>();
                        ids.sort_unstable();Some(ids)
                    } else {None};
                    if cached<rows.len() {
                        let table=b.upload(&rows[cached..])?;let mut p=row_shape(Row::PermuteRows,1)?;
                        p.rows=number((rows.len()-cached) as u64)?;p.cols=number(a.kv_rank)?;p.first=number(last)?;
                        let latents=b.prompt(p,[Some(&values["next_latent_cache"]),Some(&table),None,None,None],&[])?;
                        while at<rows.len() {let count=(rows.len()-at).min(PROMPT_GROUP);
                            let part=self.expand_owned_rows(name,&b.range(&latents,(at-cached)*rank,count*rank)?,count,layout)?;
                            let part=self.pack_expanded(&part,count*row_words)?;
                            b.rows_fill(&mut cache_values,at*packed_words,&part,count*packed_words)?;
                            // The identity of a row is that of its latent row in the cache the decode steps gather from.
                            if let Some(seed)=&seed {
                                let source=&values["next_latent_cache"];let picked=(0..count).filter(|&k|seed.binary_search(&(rows[at+k] as u32)).is_ok()).collect::<Vec<_>>();
                                let keys=picked.iter().map(|&k|b.range(source,rows[at+k] as usize*rank,rank)).collect::<Result<Vec<_>>>()?;
                                let kept=picked.iter().map(|&k|b.range(&part,k*packed_words,packed_words)).collect::<Result<Vec<_>>>()?;
                                if !picked.is_empty() {b.remember_projections(name,true,&keys.iter().zip(&kept).collect::<Vec<_>>())?;}
                            }
                            at+=count;}
                    }
                    chat.cache_rows=rows.len() as u64;chat.cache=Some(cache_values);
                    let host=std::time::Instant::now();
                    for row in chat.selected.chunks_exact_mut(words) {let count=row[slots] as usize;for v in &mut row[..count] {*v=chosen[*v as usize] as f32;}}
                    b.plan_cost("prompt_selection",host.elapsed().as_nanos());
                    values.insert("selected".to_owned(),b.upload(&chat.selected)?);},
                Kind::Scores|Kind::Softmax|Kind::Values=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("attention requires a latent layer".into());};
                    let i=a.indexer;let slots=i.token_budget+if i.select_tail {i.pool-1} else {0};
                    let mut s=row_shape(match step.kind {Kind::Scores=>Row::ScoresRowsPacked,Kind::Softmax=>Row::SoftmaxRows,_=>Row::ValuesRowsPacked},m)?;
                    let layout=self.sparse_layout(layer)?;
                    s.heads=number(layout.local_heads() as u64)?;s.dim=number(a.key_dim)?;s.taps=number(a.value_dim)?;s.rows=number(chat.cache_rows)?;s.cols=number(slots.min(last))?;s.slots=number(slots)?;
                    // Order score chains by row, so tokens reading one key share its lines.
                    let mut table=Vec::new();let mut map=Vec::new();
                    if step.kind==Kind::Scores {let count=slots as usize;
                        // Tiles compute all pairs, so a small union takes a map from pair to slot.
                        if chat.cache_rows as usize<=SCORE_MAP_ROWS {map=score_map(&chat.selected,count,chat.cache_rows as usize,m);}
                        if map.is_empty() {
                            for (token,row) in chat.selected.chunks_exact(count+1).enumerate() {for (slot,&id) in row[..row[count] as usize].iter().enumerate() {table.push((id as u32,token as u32,slot as u32));}}
                            table.sort_unstable();s.first=number(table.len() as u64)?;
                        } else {s.op=Row::ScoresMapPacked;}
                    }
                    let order=if !map.is_empty() {Some(b.upload(&map)?)} else if table.is_empty() {None} else {Some(b.upload(&table.iter().flat_map(|e|[e.0 as f32,e.1 as f32,e.2 as f32]).collect::<Vec<_>>())?)};
                    let selected_value=values.get("selected").ok_or("prompt selection is absent")?;let e=chat.cache.as_ref().ok_or("prompt expanded rows are absent")?;
                    let value=match step.kind {
                        Kind::Scores=>{let query=self.local_query(&values[&step.inputs[0]],m,layout)?;b.prompt(s,[Some(&query),Some(e),Some(selected_value),order.as_ref(),None],&[])?},
                        Kind::Softmax=>b.prompt(s,[Some(&values[&step.inputs[0]]),Some(selected_value),None,None,None],&[])?,
                        _=>b.prompt(s,[Some(&values[&step.inputs[0]]),Some(e),Some(selected_value),None,None],&[])?,
                    };
                    let value=if step.kind==Kind::Values {b.gather_attended(&value,m,layout)?} else {value};
                    values.insert(step.outputs[0].name.clone(),value);},
                _=>return Err(format!("{}: prompt row operation is unsupported",step.id)),
            }
        Ok(0)
    }
    /// One step of a layer for `m` rows, when the step reads no history of a chat: each row's
    /// result then depends on that row alone. The rows may be consecutive tokens of one chat (a
    /// prompt group) or one token of each of several chats (a batch step). None: the step reads
    /// history. Some(n): the step is done, and the next n steps are done with it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn rows_stateless(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,passes:&mut BTreeMap<String,C::Pass>,fold:&mut RowsFold<C::Value>)->Result<Option<usize>> {
        let c=self.checkpoint;let d=c.description();let b=&self.compute;let step=&plan.steps[index];let mut skip=0;
        // ROWS OVERLAP: a step that reads a held output records the held gather first.
        if step.inputs.iter().any(|name|fold.held.iter().any(|(_,held)|held==name)) {self.rows_close(values,fold)?;}
        match step.kind {
            Kind::MhcPrepare|Kind::Combine if layer==d.layers.len()=>return self.draft_batch_residual(index,plan,m,values).map(Some),
            Kind::MhcPrepare=>{let key=(layer,index);let p=self.blocks.get(&key).ok_or("resident residual parameters are absent")?;
                let (pass,norm)=b.rows_begin(key,p,&values[&step.inputs[0]],m)?;
                passes.insert(step.id.split('.').next().unwrap().to_owned(),pass);values.insert(plan.steps[index+2].outputs[0].name.clone(),norm);skip=2;},
            Kind::Combine=>{let side=step.id.split('.').next().unwrap();let pass=passes.remove(side).ok_or("prompt residual pass is absent")?;
                let value=b.rows_finish(pass,&values[&step.inputs[2]],m)?;values.insert(step.outputs[0].name.clone(),value);},
            // A projection that an earlier stage of this layer made is done. Else the stage that starts here makes it; the
            // shared row of a fold whose down step waits stays on its rank (`rows_fold_shared`; a pass that does not fold
            // pays the one flag `fold.waits` here).
            // LOCAL QUERY: the query rows of a prompt group's latent layer stay on their rank (`rows_query_local`; a pass
            // that is not a prompt group pays the one flag `fold.group` here).
            Kind::Project=>{if !values.contains_key(&step.outputs[0].name) && !fold.held.iter().any(|(_,held)|*held==step.outputs[0].name) && !(fold.waits && self.rows_fold_shared(layer,index,plan,m,values,fold)?)
                && !(fold.group && self.rows_query_local(layer,index,plan,m,values)?) && !self.rows_project_open(index,plan,m,values,fold)? {self.rows_project_stage(index,plan,m,values)?;}},
            Kind::Norm=>{let name=&step.weights[0].name;let width=step.outputs[0].elements().map_err(|e|e.to_string())?;
                let value=b.rows_norm(name,norm_shape(c,name,width)?,&values[&step.inputs[0]],m,self.parameter(name)?)?;values.insert(step.outputs[0].name.clone(),value);},
            // LOOKAHEAD: a SwiGLU that `rows_ahead` made behind the routed gate and up is done.
            Kind::SwiGlu=>{if !values.contains_key(&step.outputs[0].name) {let value=self.rows_swiglu(step,m,values)?;values.insert(step.outputs[0].name.clone(),value);}},
            Kind::KdaPrepare|Kind::GatedNorm=>{let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("recurrent operation requires a recurrent layer".into());};
                let mut s=shape(if step.kind==Kind::KdaPrepare {Op::Prepare} else {Op::GatedNorm});s.epsilon=d.epsilon as f32;
                s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;if step.kind==Kind::KdaPrepare {s.bound=k.gate_lower_bound as f32;}
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                let mut x=[None;5];for (slot,name) in step.inputs.iter().enumerate() {x[slot]=Some(&values[name]);}
                let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,x,&weight)?;
                let value=if step.kind==Kind::GatedNorm {b.gather_attended(&value,m,self.kda_layout(layer)?)?} else {value};values.insert(step.outputs[0].name.clone(),value);},
            Kind::KeyNorm=>{let TokenMixer::LatentAttention(a)=self.mixer(layer)? else {return Err("index key requires a latent layer".into());};
                let mut s=shape(Op::KeyNorm);s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.epsilon=plan.settings.index_key_epsilon;
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(self.parameter(&w.name)?);}
                let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,[Some(&values[&step.inputs[0]]),None,None,None,None],&weight)?;values.insert(step.outputs[0].name.clone(),value);},
            Kind::Route=>{let mut s=shape(Op::Route);s.heads=number(d.routing.groups)?;s.dim=number(d.routing.selected_groups)?;s.rows=number(step.weights[0].shape[0])?;
                s.cols=number(step.outputs[0].shape[0])?;s.taps=u32::from(d.routing.normalize_selected);s.bound=d.routing.scale as f32;s.epsilon=d.epsilon as f32;
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(self.parameter(&w.name)?);}
                let (_,_,_,words)=s.lengths().ok_or("invalid prompt route shape")?;
                let rows=RowShape::wrapped(s,m).map(|mut r|{r.first=u32::from(spark_gpu_probe::prompt_admission::planes_admitted(r));r}).ok_or("prompt row count exceeds ABI")?;
                let result=b.prompt(rows,[Some(&values[&step.inputs[0]]),None,None,None,None],&weight)?;let k=s.cols as usize;
                {
                    // IDs and weights: the first 2k words of each route row, joined in order; or with ROUTE PLANES (P8) two
                    // token-major planes the router wrote at the start, taken as two views. No host read either way.
                    need(b.words(&result)==m*words,"prompt route extent differs")?;
                    for (output,first) in [(0,0),(1,k)] {
                        let parts=if rows.first!=0 {vec![b.range(&result,first*m,m*k)?]} else {(0..m).map(|token|b.range(&result,token*words+first,k)).collect::<Result<Vec<_>>>()?};
                        values.insert(step.outputs[output].name.clone(),if rows.first!=0 {parts.into_iter().next().ok_or("route plane is absent")?} else {b.join(&parts.iter().collect::<Vec<_>>())?});
                    }
                }},
            // ROWS FOLD: a routed down step that its fold can take (`rows_fold_waits`) runs at the fold step, below.
            Kind::ExpertDown if self.rows_fold_waits(layer,index,plan,m,values)=>{fold.waits=true;fold.grouped=false;fold.shared=None;},
            // GROUP FOLD: the grouped down step of a prompt group that its fold can take (`rows_group_fold_waits`) runs at
            // the fold step, below, with the fold before the exchange.
            Kind::ExpertDown if fold.group && self.rows_group_fold_waits(layer,index,plan,m,values)=>{fold.waits=true;fold.grouped=true;fold.shared=None;},
            Kind::ExpertProject|Kind::ExpertDown=>{skip=self.experts_group(index,plan,m,values,None,Some((layer,&mut *fold)))?;if step.kind==Kind::ExpertProject && !fold.held.is_empty() {self.rows_close(values,fold)?;}},
            Kind::ExpertSum=>{let route=plan.steps.iter().find(|s|s.kind==Kind::Route).ok_or("prompt expert fold has no route")?;
                let mut s=shape(Op::ExpertFold);s.rows=number(route.outputs[0].shape[0])?;s.cols=number(d.hidden)?;s.taps=u32::from(step.inputs.len()==4);
                // ROWS FOLD: the down step of this layer waited (`fold.waits`; a pass that does not fold pays that one flag).
                // The compute runs the down step and this fold as one stage: each rank folds its own rows of every token and
                // ONE exchange carries the folded rows (one row a token; the unfolded rows are one a (token, pick)). The fold
                // is row by row, so the words are the words of the fold after the gather (`Decoder::expert_fold` for one
                // row). A compute that does not take the stage gets the down step now, as it was, and the fold below.
                if fold.waits {
                    fold.waits=false;let grouped=std::mem::take(&mut fold.grouped);
                    // The shared row: this rank's own rows when `rows_fold_shared` kept them here, else the gathered rows.
                    let kept=fold.shared.take();
                    let links=self.fold_links(layer,plan).filter(|links|links.fold==index).ok_or("prompt expert fold has no waiting down step")?;
                    let down=&plan.steps[links.down];
                    let folded=if grouped {
                        // GROUP FOLD: the grouped down step with this fold (`experts_group`): each rank places and folds its
                        // own rows, and ONE exchange carries the folded rows. The fold is a chain a word, so the words are the
                        // words of the fold after the gather. Done: the fold's output is here.
                        self.experts_group(links.down,plan,m,values,Some((index,s,kept.as_ref())),None)?;
                        values.contains_key(&step.outputs[0].name)
                    } else {
                        let input=|name:&String|values.get(name).ok_or_else(||format!("{}: prompt expert fold input is absent",step.id));
                        let shared=match (step.inputs.get(3),kept.as_ref()) {(Some(_),Some(local))=>Some(local),(Some(name),None)=>Some(input(name)?),(None,_)=>None};
                        let folded=b.rows_routed_fold(down,input(&step.inputs[0])?,input(&down.inputs[0])?,input(&step.inputs[1])?,shared,s,m)?;
                        if let Some(value)=folded {values.insert(step.outputs[0].name.clone(),value);true} else {false}};
                    if folded {return Ok(Some(0));}
                    // The stage is refused. The fold below reads whole rows: a shared row kept on its rank is made again,
                    // with its exchange; then the down step, as it was (the grouped form made it when its stage was refused).
                    if kept.is_some() {
                        let (shared,_)=links.shared.ok_or("prompt expert fold has no shared projection")?;
                        self.rows_project_stage(shared,plan,m,values)?;
                    }
                    if !grouped {self.experts_group(links.down,plan,m,values,None,None)?;}
                }
                let mut x=[None;5];for (slot,name) in step.inputs.iter().enumerate() {x[slot]=Some(&values[name]);}
                let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,x,&[])?;values.insert(step.outputs[0].name.clone(),value);},
            _=>return Ok(None),
        }
        Ok(Some(skip))
    }
    /// The convolution rows of a group and the history after its last token. A group past the reading kernels' swept token range
    /// (`prompt_admission::GROUP_MAX`) runs them in runs of that many tokens, each with the history after the run before it, as
    /// consecutive groups do: the device has its convolution kernel for runs of that size, and one launch for all tokens would
    /// take the slow generic kernel. A token reads only the rows before it, through the history, so the bits are equal.
    /// ONE LAUNCH: a run of a swept big size (512, 768 or 1,024 tokens, `prompt_admission::conv_whole`) has that kernel for
    /// all its tokens, so it takes one launch and one history, and no join copies the convolved plane.
    fn conv_rows(&self,s:RowShape,x:[Option<&C::Value>;5],weight:&[f32])->Result<(C::Value,C::Value)> {
        let b=&self.compute;let (m,run_rows)=(s.tokens as usize,spark_gpu_probe::prompt_admission::GROUP_MAX);
        if m<=run_rows || spark_gpu_probe::prompt_admission::conv_whole(s) {return Ok((b.prompt(s,x,weight)?,b.prompt(RowShape {op:Row::HistoryRows,..s},x,&[])?));}
        let width=(s.heads as usize).checked_mul(s.dim as usize).ok_or("convolution row width overflow")?;
        let planes=[x[0].ok_or("convolution rows are absent")?,x[1].ok_or("convolution rows are absent")?,x[2].ok_or("convolution rows are absent")?];
        let mut history:Option<C::Value>=None;let mut parts=Vec::with_capacity(m.div_ceil(run_rows));let mut first=0;
        while first<m {
            let n=(m-first).min(run_rows);let run=RowShape {tokens:number(n as u64)?,..s};
            let rows=[b.range(planes[0],first*width,n*width)?,b.range(planes[1],first*width,n*width)?,b.range(planes[2],first*width,n*width)?];
            let input=[Some(&rows[0]),Some(&rows[1]),Some(&rows[2]),history.as_ref().or(x[3]),None];
            parts.push(b.prompt(run,input,weight)?);
            history=Some(b.prompt(RowShape {op:Row::HistoryRows,..run},input,&[])?);first+=n;
        }
        Ok((b.join(&parts.iter().collect::<Vec<_>>())?,history.ok_or("convolution group is empty")?))
    }
    /// Every projection of the layer that is ready from `index` on (its rows are here, its result
    /// is not) is made now, as one stage. A projection is a pure function of its rows, so it may
    /// run before the steps between. A compute that can (`Compute::rows_stage`) gives the stage
    /// one exchange; else each projection takes its own call. The step at `index` is always ready.
    fn rows_project_stage(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>)->Result<()> {
        let ready=plan.steps.iter().enumerate().skip(index).filter(|(_,step)|step.kind==Kind::Project && step.weights.len()==1 && step.inputs.len()==1 && step.outputs.len()==1
            && values.contains_key(&step.inputs[0]) && !values.contains_key(&step.outputs[0].name)).map(|(at,_)|at).collect::<Vec<_>>();
        need(ready.first()==Some(&index),"prompt projection input is absent")?;
        let outputs={
            let jobs=ready.iter().map(|&at| {let step=&plan.steps[at];
                let router=plan.steps.iter().any(|s|s.kind==Kind::Route && s.inputs.contains(&step.outputs[0].name));
                (step.weights[0].name.as_str(),&values[&step.inputs[0]],!router)}).collect::<Vec<_>>();
            match if jobs.len()>1 {self.compute.rows_stage(self.checkpoint,&jobs,m)?} else {None} {
                Some(outputs)=>outputs,
                None=>jobs.iter().map(|(name,rows,round)|self.project_group(name,rows,m,*round)).collect::<Result<Vec<_>>>()?,
            }};
        need(outputs.len()==ready.len(),"row stage output count differs")?;
        for (at,value) in ready.into_iter().zip(outputs) {values.insert(plan.steps[at].outputs[0].name.clone(),value);}
        Ok(())
    }
    /// ROWS OVERLAP: in a check pass, the stage that holds the router gate opens with the shared expert's block-FP8 gather
    /// held (`Compute::rows_stage_open`): the router's output is here at once, and the held outputs wait in `fold.held` until
    /// the routed gate and up step (or their first reader) records the gather (`rows_close`). Between them the router chain
    /// and the routed gate and up are recorded, beside the block-FP8 launches. False, before any work: not offered.
    fn rows_project_open(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,fold:&mut RowsFold<C::Value>)->Result<bool> {
        let ready=plan.steps.iter().enumerate().skip(index).filter(|(_,step)|step.kind==Kind::Project && step.weights.len()==1 && step.inputs.len()==1 && step.outputs.len()==1
            && values.contains_key(&step.inputs[0]) && !values.contains_key(&step.outputs[0].name)).map(|(at,_)|at).collect::<Vec<_>>();
        let router=|at:usize|plan.steps.iter().any(|s|s.kind==Kind::Route && s.inputs.contains(&plan.steps[at].outputs[0].name));
        if ready.len()<2 || ready.first()!=Some(&index) || !fold.held.is_empty() || !ready.iter().any(|&at|router(at)) {return Ok(false);}
        let jobs=ready.iter().map(|&at|(plan.steps[at].weights[0].name.as_str(),&values[&plan.steps[at].inputs[0]],!router(at))).collect::<Vec<_>>();
        let Some(outputs)=self.compute.rows_stage_open(self.checkpoint,&jobs,m)? else {return Ok(false);};
        need(outputs.len()==ready.len(),"row stage output count differs")?;
        for (job,(at,value)) in ready.into_iter().zip(outputs).enumerate() {
            let name=plan.steps[at].outputs[0].name.clone();match value {Some(value)=>{values.insert(name,value);},None=>fold.held.push((job,name))}
        }Ok(true)
    }
    /// ROWS OVERLAP: the held gather (`Compute::rows_stage_close`); each held output takes its step's name.
    fn rows_close(&self,values:&mut BTreeMap<String,C::Value>,fold:&mut RowsFold<C::Value>)->Result<()> {
        let held=std::mem::take(&mut fold.held);let made=self.compute.rows_stage_close()?;need(made.len()==held.len(),"held row stage output count differs")?;
        for (job,value) in made {let (_,name)=held.iter().find(|(at,_)|*at==job).ok_or("held row stage output is unknown")?;values.insert(name.clone(),value);}
        Ok(())
    }
    /// LOCAL QUERY. A latent layer's query projection is split by heads, and every reader of the query on a rank takes
    /// only that rank's heads (`local_query`: the score step, or the absorbed tail in its place). So a prompt group needs
    /// no exchange of it: each rank keeps the rows it made (`Compute::rows_project_owned`), which are its rows of the
    /// gathered value, word for word. Measured on a pair: 11 exchanges of 32 MiB a rank in a group of 1,024 rows. True:
    /// the projection `index` is that query, its rows are here in the local form, and the layer makes one exchange less.
    /// False, before any work: another projection, a reader that is not a score step's query, a rank that owns every
    /// head, or a compute that does not keep the rows; the stage then makes the projection with its exchange.
    fn rows_query_local(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>)->Result<bool> {
        let step=&plan.steps[index];
        // Every reader is a score step that reads it as its query, and nothing returns it or routes by it.
        if !matches!(self.mixer(layer)?,TokenMixer::LatentAttention(_)) || !crate::sparse_heads::query_stays_local(plan,index) {return Ok(false);}
        let name=&step.outputs[0].name;
        let layout=self.sparse_layout(layer)?;let rows=layout.query_rows();let width=layout.total_heads()*layout.key_dim();
        if rows.len()==width || step.outputs[0].elements().map_err(|e|e.to_string())?!=width {return Ok(false);}
        let Some(input)=values.get(&step.inputs[0]) else {return Ok(false);};
        let Some(local)=self.compute.rows_project_owned(self.checkpoint,&step.weights[0].name,input,m,true,rows.clone())? else {return Ok(false);};
        need(self.compute.words(&local)==m*rows.len(),"local query rows extent differs")?;
        values.insert(name.clone(),local);Ok(true)
    }
    /// ROWS FOLD: the fold steps of `layer`, found once (`FoldLinks::of`) and kept: a layer's program has the same steps at
    /// every context. The two kinds are read back, so a program that differs is found again and never trusted.
    fn fold_links(&self,layer:usize,plan:&LayerPlan)->Option<FoldLinks> {
        let Some(slot)=self.fold_links.get(layer) else {return FoldLinks::of(plan);};
        let known=match slot.get() {Some(known)=>known,None=>{let found=FoldLinks::of(plan);slot.set(Some(found));found}};
        match known {
            Some(links) if plan.steps.get(links.down).is_some_and(|step|step.kind==Kind::ExpertDown) && plan.steps.get(links.fold).is_some_and(|step|step.kind==Kind::ExpertSum)=>Some(links),
            Some(_)=>FoldLinks::of(plan),
            None=>None,
        }
    }
    /// ROWS FOLD, the shared row. The fold of a routed layer reads, for each rank, that rank's own rows of the shared
    /// expert's down rows, so a fold that runs before the exchange needs no exchange of them. Called when the layer's
    /// down step waits. True when the projection `index` is the last input of that fold with the fold as its only
    /// reader (`FoldLinks`), and the compute kept its rows on their rank (`Compute::rows_project_local`): they are then
    /// held in `fold.shared` for that fold, and the layer makes one exchange less. False, before any work: the stage
    /// makes the projection with its exchange.
    fn rows_fold_shared(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&BTreeMap<String,C::Value>,fold:&mut RowsFold<C::Value>)->Result<bool> {
        let Some((at,round))=self.fold_links(layer,plan).and_then(|links|links.shared) else {return Ok(false);};
        if at!=index {return Ok(false);}
        // LOOKAHEAD: the rows are here already (`rows_ahead` made them, by the call below, behind the routed gate and up).
        if let Some(local)=fold.ahead.take() {fold.shared=Some(local);return Ok(true);}
        let step=&plan.steps[index];
        let Some(input)=values.get(&step.inputs[0]) else {return Ok(false);};
        let Some(local)=self.compute.rows_project_local(self.checkpoint,&step.weights[0].name,input,m,round)? else {return Ok(false);};
        fold.shared=Some(local);Ok(true)
    }
    /// The SwiGLU step `step` of `m` rows: one launch over all of them.
    fn rows_swiglu(&self,step:&engine_model::plan::layer::Step,m:usize,values:&BTreeMap<String,C::Value>)->Result<C::Value> {
        let elements=step.outputs[0].elements().map_err(|e|e.to_string())?;let mut s=shape(Op::SwiGlu);
        s.rows=number(elements as u64*m as u64)?;s.bound=self.checkpoint.description().swiglu_limit as f32;
        self.compute.operation(s,[Some(&values[&step.inputs[0]]),Some(&values[&step.inputs[1]]),None,None],&[])
    }
    /// LOOKAHEAD: what a prompt group queues behind its routed gate and up (`gate` is their first step) while their
    /// exchange is open (`Compute::rows_experts_open`), so that it runs while the link carries their rows: the two later
    /// steps of the layer that read nothing of that exchange and make no exchange themselves. They are the shared
    /// expert's SwiGLU and this rank's own rows of its down projection (the call of `rows_fold_shared`; the layer's fold
    /// reads only this rank's rows of it). Their inputs are complete here: the stage before the route gathered the
    /// shared gate and up. Each is the launch its own step makes, on the same words, so its result has the same bits;
    /// only its place in the queue is earlier. The SwiGLU's value takes its step's name, and that step finds it done.
    /// The local rows wait in `fold.ahead` for the shared projection's step. Nothing is queued when the layer's fold
    /// has no shared row of this form, when the SwiGLU's inputs are not here, or when the compute keeps no rows local
    /// (the SwiGLU alone is then made ahead, and the projection keeps its stage and its exchange).
    fn rows_ahead(&self,layer:usize,gate:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,fold:&mut RowsFold<C::Value>)->Result<()> {
        let Some((at,round))=self.fold_links(layer,plan).and_then(|links|links.shared) else {return Ok(());};
        let project=&plan.steps[at];
        let Some(swiglu)=plan.steps.get(gate+3..at).and_then(|steps|steps.iter().find(|step|step.kind==Kind::SwiGlu && step.inputs.len()==2 && step.outputs.len()==1 && step.outputs[0].name==project.inputs[0])) else {return Ok(());};
        if values.contains_key(&swiglu.outputs[0].name) || !swiglu.inputs.iter().all(|name|values.contains_key(name)) {return Ok(());}
        let middle=self.rows_swiglu(swiglu,m,values)?;
        let local=self.compute.rows_project_local(self.checkpoint,&project.weights[0].name,&middle,m,round)?;
        values.insert(swiglu.outputs[0].name.clone(),middle);fold.ahead=local;Ok(())
    }
    /// LOOKAHEAD: the work behind an open gate-and-up exchange (`rows_ahead`), then its close (`Compute::rows_experts_close`):
    /// the middle takes its name and the gate, up and SwiGLU steps are done. The close always follows the open, also
    /// when `rows_ahead` failed; `rows_ahead`'s own error is then the one reported.
    fn experts_ahead(&self,layer:usize,gate:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,fold:&mut RowsFold<C::Value>)->Result<usize> {
        let queued=self.rows_ahead(layer,gate,plan,m,values,fold);
        let middle=self.compute.rows_experts_close(queued.is_err());
        queued?;values.insert("expert_middle_grouped".to_owned(),middle?);Ok(2)
    }
    /// ROWS FOLD: true when the routed down step `index` of this row pass waits for its fold (`Compute::rows_routed_fold`,
    /// called at the fold step in `rows_stateless`): the compute folds the rows of this pass, the fold is the only
    /// reader of the down rows (`FoldLinks`), and the middle is here in (token, pick) order (the routed gate and up made
    /// it; a middle the grouped pass kept in its own order stays with that pass). Nothing between the two steps reads
    /// the down rows or the fold's output, so running the down step later changes no value. A pass that does not fold
    /// (a prompt group, a batch step in the grouped form) pays the one flag `folds_rows` here.
    fn rows_fold_waits(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&BTreeMap<String,C::Value>)->bool {
        self.compute.folds_rows(m) && self.fold_links(layer,plan).is_some_and(|links|links.down==index) && values.contains_key(&plan.steps[index].inputs[0])
    }
    /// GROUP FOLD: true when the grouped down step `index` of a prompt group waits for its fold (`Compute::rows_experts_fold`,
    /// called from `experts_group` at the fold step): the compute folds the rows of a group, the fold is the only reader of
    /// the down rows (`FoldLinks`), and the middle is here in the grouped order the gate and up steps kept
    /// (`expert_middle_grouped`). Nothing between the two steps reads the down rows or the fold's output, so running the
    /// down step later changes no value. A group whose compute does not fold pays the one flag `folds_groups` here.
    fn rows_group_fold_waits(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&BTreeMap<String,C::Value>)->bool {
        self.compute.folds_groups(m) && self.fold_links(layer,plan).is_some_and(|links|links.down==index) && values.contains_key("expert_middle_grouped")
    }
    /// ROUTE ROWS: the layer's routed experts with the picks left on the device (`Compute::rows_routed`): one job a
    /// (token, pick), in that order, so nothing is grouped, permuted or placed back. Gate, up and their SwiGLU are one
    /// call; its result takes the SwiGLU step's own name, in the (token, pick) order the down step reads. None: the
    /// compute does not take this step, and `experts_group` reads the route (the IDs are then read from the device).
    fn experts_routed(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>)->Result<Option<usize>> {
        let b=&self.compute;let step=&plan.steps[index];
        if step.kind==Kind::ExpertDown {
            // The middle in (token, pick) order: the routed gate and up made it, or the plain SwiGLU step did. A middle that
            // the grouped pass kept in its own order (`expert_middle_grouped`) stays with that pass.
            let Some(middle)=values.get(&step.inputs[0]) else {return Ok(None);};
            let Some(value)=b.rows_routed(std::slice::from_ref(step),None,&values[&step.inputs[1]],middle,m)? else {return Ok(None);};
            values.insert(step.outputs[0].name.clone(),value);return Ok(Some(0));
        }
        if step.id!="expert_gate" {return Ok(None);}
        let (Some(up),Some(swiglu))=(plan.steps.get(index+1),plan.steps.get(index+2)) else {return Ok(None);};
        if up.id!="expert_up" || up.kind!=Kind::ExpertProject || up.inputs!=step.inputs || swiglu.kind!=Kind::SwiGlu || swiglu.outputs.len()!=1 {return Ok(None);}
        let bound=self.checkpoint.description().swiglu_limit as f32;
        let Some(middle)=b.rows_routed(&plan.steps[index..index+2],Some(bound),&values[&step.inputs[1]],&values[&step.inputs[0]],m)? else {return Ok(None);};
        values.insert(swiglu.outputs[0].name.clone(),middle);Ok(Some(2))
    }
    /// The host tables of the grouped expert step `step` for `m` rows: the picks are read from the device, and the rows
    /// of one expert become contiguous, in ascending expert order. The statements `experts_group` always ran first.
    fn expert_tables(&self,plan:&LayerPlan,step:&engine_model::plan::layer::Step,m:usize,values:&BTreeMap<String,C::Value>)->Result<ExpertTables> {
        let b=&self.compute;
        spark_gpu_probe::prompt_launches::record("prompt_route_host");
        let ids=b.read(&values[&step.inputs[1]])?;let k=step.outputs[0].shape[0] as usize;need(ids.len()==m*k,"prompt route count differs")?;
        let host=std::time::Instant::now();
        let mut groups:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for (token,row) in ids.chunks_exact(k).enumerate() {for (slot,&id) in row.iter().enumerate() {
            need(id>=0.0 && id.fract()==0.0 && (id as usize)<step.weights.len(),"prompt expert ID is outside the recipe")?;
            need(!row[..slot].contains(&id),"prompt route repeats an expert")?;groups.entry(id as usize).or_default().push(token*k+slot);
        }}
        if step.id=="expert_gate" {super::batch::note_expert_union(plan.layer,m*k,groups.len());b.note_batch_experts(m*k,groups.len());}
        let down=step.kind==Kind::ExpertDown;
        // Rows of one expert become contiguous, in ascending expert order.
        let mut order=Vec::with_capacity(m*k);let mut spans=Vec::with_capacity(groups.len());
        for (&id,rows) in &groups {spans.push((id,order.len(),rows.len()));order.extend(rows.iter().map(|&row|if down {row} else {row/k} as f32));}
        // Three numbers per output row: earlier rows, this expert's rows, row index.
        let mut grouped=Vec::with_capacity(3*m*k);let mut placed=vec![0.0f32;3*m*k];
        for &(id,at,count) in &spans {for (position,&row) in groups[&id].iter().enumerate() {
            grouped.extend_from_slice(&[at as f32,count as f32,position as f32]);placed[3*row..3*row+3].copy_from_slice(&[at as f32,count as f32,position as f32]);}}
        b.plan_cost("prompt_expert_tables",host.elapsed().as_nanos());
        Ok(ExpertTables {groups,spans,order,grouped,placed})
    }
    /// Rows choosing one expert share one weight pass. Host routes only place
    /// rows. Every output row returns to its own (token,slot) unchanged.
    /// GROUP FOLD: `fold` names the fold of a down step that waited for it (its step, its shape, and this rank's rows of
    /// the shared row when the compute kept them): the compute then gets the down step and the fold as one stage
    /// (`Compute::rows_experts_fold`) and the fold's output is the result. A compute that refuses the stage gets the down
    /// step as it was; the caller sees no fold output and runs the fold.
    /// LOOKAHEAD: `ahead` is the layer and its fold state, given for the layer's own gate and up steps: see the gate branch.
    #[allow(clippy::type_complexity)] // one private call; the tuple is the whole fold.
    fn experts_group(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,fold:Option<(usize,Shape,Option<&C::Value>)>,ahead:Option<(usize,&mut RowsFold<C::Value>)>)->Result<usize> {
        let b=&self.compute;let step=&plan.steps[index];
        if b.routes_rows(m) && let Some(done)=self.experts_routed(index,plan,m,values)? {return Ok(done);}
        // Both native stages precede the legacy host table read. A private
        // middle must be consumed here or fail; it never enters that fallback.
        // Only a pass whose down step waits for its fold (GROUP FOLD: `fold.group`) consumes it: a pass that keeps
        // rows (`prefill_rows_open`) runs its down step alone, which reads whole rows, so its gate stays legacy.
        if step.id=="expert_gate" && ahead.as_ref().is_some_and(|(_,state)|state.group) && let Some(links)=self.fold_links(plan.layer,plan).filter(|links|plan.steps[links.down].kind==Kind::ExpertDown) {
            if let (Some(up),Some(swiglu))=(plan.steps.get(index+1),plan.steps.get(index+2)) && up.id=="expert_up" && swiglu.kind==Kind::SwiGlu {
                let shared=links.shared.and_then(|(at,_)|plan.steps[at].weights.first()).map(|w|w.name.as_str());
                let steps=[step.clone(),up.clone(),plan.steps[links.down].clone()];
                if let Some(mut made)=b.rows_experts_device_wide(&steps,self.checkpoint.description().swiglu_limit as f32,&[(m,&values[&step.inputs[0]],&values[&step.inputs[1]])],(self.expert_fold_shape(plan,&plan.steps[links.fold])?,shared))? {
                    need(made.len()==1,"device expert result count differs")?;values.insert("expert_middle_grouped".to_owned(),made.pop().unwrap());return Ok(2);
                }
            }
        }
        if step.kind==Kind::ExpertDown && let Some((sum,shape,kept))=fold && let Some(middle)=values.get("expert_middle_grouped") {
            let sum=&plan.steps[sum];
            if let Some(mut made)=b.rows_experts_fold_device_wide(step,&[(m,middle,&values[&sum.inputs[0]],&values[&sum.inputs[1]],sum.inputs.get(3).and(kept))],shape)? {
                need(made.len()==1,"device fold result count differs")?;values.remove("expert_middle_grouped");values.insert(sum.outputs[0].name.clone(),made.pop().unwrap());return Ok(0);
            }
        }
        let ExpertTables {groups,spans,order,grouped,placed}=self.expert_tables(plan,step,m,values)?;let k=step.outputs[0].shape[0] as usize;
        let width=step.weights[0].shape[1] as usize;let out_width=step.weights[0].shape[0] as usize;let down=step.kind==Kind::ExpertDown;
        let permute=|source:&C::Value,table:&[f32],rows:usize,cols:usize,from:usize|->Result<C::Value> {
            let table=b.upload(table)?;let mut s=row_shape(Row::PermuteRows,1)?;s.rows=number(rows as u64)?;s.cols=number(cols as u64)?;s.first=number(from as u64)?;
            b.prompt(s,[Some(source),Some(&table),None,None,None],&[])};
        use expert_names as names;
        if !down && step.id=="expert_gate" {
            let up=&plan.steps[index+1];let swiglu=&plan.steps[index+2];need(up.id=="expert_up" && swiglu.kind==Kind::SwiGlu,"prompt expert program differs")?;
            let (gates,ups,bound)=(names(step,&spans),names(up,&spans),self.checkpoint.description().swiglu_limit as f32);
            // LOOKAHEAD: a prompt group whose compute leaves the link step of this exchange open (the device, when it
            // folds a group's down rows first: GROUP FOLD) queues the shared expert's two steps behind it (`rows_ahead`)
            // and closes it (`experts_ahead`). Those steps then run while the link carries the middle.
            let mut ahead=ahead.filter(|(_,state)|state.group);
            // GATHERED ROWS under LOOKAHEAD: the open exchange on rows the compute reads through `order`; no copy is made.
            if let Some((layer,state))=ahead.as_mut() && b.rows_experts_open_gathered(&gates,(&ups,bound),&values[&step.inputs[0]],&order,&grouped)? {
                return self.experts_ahead(*layer,index,plan,m,values,state);
            }
            // GATHERED ROWS: the compute reads each token's row through `order`; the copy below is not made.
            if let Some(middle)=b.rows_experts_gathered(&gates,Some((&ups,bound)),&values[&step.inputs[0]],&order,&grouped)? {
                values.insert("expert_middle_grouped".to_owned(),middle);return Ok(2);
            }
            let source=permute(&values[&step.inputs[0]],&order,m*k,width,m)?;
            // LOOKAHEAD on the copied rows (a compute that does not take the gathered operands). `source` lives until
            // this call returns, so past the close.
            if let Some((layer,state))=ahead.as_mut() && b.rows_experts_open(&gates,(&ups,bound),&source,&grouped)? {
                return self.experts_ahead(*layer,index,plan,m,values,state);
            }
            if let Some(middle)=b.rows_experts(&gates,Some((&ups,bound)),&source,&grouped)? {
                values.insert("expert_middle_grouped".to_owned(),middle);return Ok(2);
            }
        }
        let held=values.remove("expert_middle_grouped");
        let source=match held {Some(v)=>v,None=>permute(&values[&step.inputs[0]],&order,m*k,width,if down {m*k} else {m})?};
        if down&& let Some((sum,shape,kept))=fold {
            let sum=&plan.steps[sum];need(sum.kind==Kind::ExpertSum && sum.inputs.len()>=3,"prompt expert fold step differs")?;
            let folded={
                let input=|name:&String|values.get(name).ok_or_else(||format!("{}: prompt expert fold input is absent",sum.id));
                // The fold reads this rank's own rows of the shared row. Gathered rows are not that form: the stage is refused below.
                match (sum.inputs.get(3),kept) {
                    (Some(_),None)=>None,
                    (shared,kept)=>b.rows_experts_fold(&names(step,&spans),&source,&placed,input(&sum.inputs[0])?,input(&sum.inputs[1])?,shared.and(kept),shape,m)?,
                }};
            if let Some(value)=folded {values.insert(sum.outputs[0].name.clone(),value);return Ok(0);}
        }
        if down&& let Some(value)=b.rows_experts(&names(step,&spans),None,&source,&placed)? {values.insert(step.outputs[0].name.clone(),value);return Ok(0);}
        let mut parts=Vec::with_capacity(spans.len());
        for &(id,at,count) in &spans {
            let rows=b.range(&source,at*width,count*width)?;
            parts.push(self.project_group(&step.weights[id].name,&rows,count,true)?);
        }
        let joined=b.join(&parts.iter().collect::<Vec<_>>())?;drop(parts);
        // Invert the placement: output (token,slot) reads its row in the joined matrix.
        let host=std::time::Instant::now();
        let mut inverse=vec![0.0f32;m*k];let mut at=0usize;for rows in groups.values() {for &row in rows {inverse[row]=at as f32;at+=1;}}
        b.plan_cost("prompt_expert_tables",host.elapsed().as_nanos());
        let value=permute(&joined,&inverse,m*k,out_width,m*k)?;values.insert(step.outputs[0].name.clone(),value);Ok(0)
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn expansion_partition_keeps_every_selected_row_in_its_query_order() {
        for total in [1usize,2,5,8,1152,1280,7680] {for held in [0,1,total/2,total] {
            let slots=total.min(7);let mut input=Vec::new();
            for token in 0..9 {for at in 0..slots {input.push(((token*3+slots-at)%total) as f32);}input.push(slots as f32);}
            let (rows,map,cached)=expansion_rows(&input,slots,total,held).unwrap();
            assert!(rows[..cached].iter().all(|&id|(id as usize)<held));
            assert!(rows[cached..].iter().all(|&id|id as usize>=held));
            for query in input.chunks_exact(slots+1) {for &id in &query[..slots] {assert_eq!(rows[map[id as usize] as usize].to_bits(),id.to_bits());}}
            let unique=input.chunks_exact(slots+1).flat_map(|v|v[..slots].iter().map(|&x|x as usize)).collect::<std::collections::BTreeSet<_>>();
            assert_eq!(rows.len(),unique.len());
        }}
    }
    #[test] fn expansion_partition_rejects_invalid_counts_and_row_ids() {
        for bad in [vec![0.0,-1.0],vec![0.0,1.5],vec![0.0,f32::NAN],vec![1.0,1.0],vec![-1.0,1.0],vec![f32::INFINITY,1.0],vec![0.5,1.0]] {
            assert!(expansion_rows(&bad,1,1,0).is_err(),"{bad:?}");
        }
        assert!(expansion_rows(&[0.0],1,1,0).is_err());assert!(expansion_rows(&[0.0,0.0],1,1,0).is_err());
        assert!(expansion_rows(&[0.0,1.0],1,1,2).is_err());
    }
}
#[cfg(test)]
mod forced_union_tests {
    use super::*;
    #[test]
    fn forced_rows_reach_the_whole_context_with_valid_counts() {
        let (slots,first,group)=(2051usize,32512usize,256usize);let words=slots+1;let mut selected=vec![0.0;group*words];
        force_distinct_rows(&mut selected,slots,first);
        let mut seen=std::collections::BTreeSet::new();
        for (token,row) in selected.chunks_exact(words).enumerate() {
            let count=row[slots] as usize;assert_eq!(count,slots);
            let ids=row[..count].iter().map(|&v|v as usize).collect::<std::collections::BTreeSet<_>>();
            assert_eq!(ids.len(),count);assert!(ids.iter().all(|&id|id<=first+token));seen.extend(ids);
        }
        assert_eq!(seen.len(),first+group);
    }
    #[test]
    fn memo_ballast_must_equal_the_charged_bound() {
        assert_eq!(full_memo_ballast(PROJECTION_MEMO_BYTES).unwrap(),PROJECTION_MEMO_BYTES);
        for limit in [0,1,PROJECTION_MEMO_BYTES-1,PROJECTION_MEMO_BYTES+1] {assert!(full_memo_ballast(limit).is_err());}
    }
    #[test]
    fn score_map_names_each_slot_and_refuses_a_repeat() {
        // Two tokens, three slots at most, four rows.
        let selected=[2.0,0.0,0.0,2.0,3.0,1.0,0.0,2.0];let map=score_map(&selected,3,4,2);
        assert_eq!(map,[1.0,-1.0,-1.0,1.0,0.0,-1.0,-1.0,0.0]);
        assert!(score_map(&[1.0,1.0,0.0,2.0],3,4,1).is_empty());
        assert!(score_map(&[9.0,0.0,0.0,1.0],3,4,1).is_empty());
    }
    #[test]
    fn forced_rows_fit_a_young_context() {
        let (slots,first)=(2051usize,3usize);let mut selected=vec![0.0;2*(slots+1)];force_distinct_rows(&mut selected,slots,first);
        assert_eq!(selected[slots] as usize,4);assert_eq!(selected[2*slots+1] as usize,5);
        assert_eq!(selected[3] as usize,3);assert_eq!(selected[slots+1+4] as usize,4);
    }
}
