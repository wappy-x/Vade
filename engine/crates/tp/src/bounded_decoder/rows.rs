//! Layer-major prompt groups. All rows of a group finish one layer before the
//! next layer starts, so one weight read serves many rows. Each row keeps its
//! scalar arithmetic. Recurrent state and latent history stay in token order.
use super::*;
use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
#[path="rows/pools.rs"] mod pools;
/// Largest group one prefill call accepts. A fixed size, so memory is known at admission.
/// Measured on the real server: 1040 tokens read in 46 s at this size and 54 s at 1024.
pub const PROMPT_GROUP:usize=256;
/// Largest group prefill accepts. The server's group comes from its admitted memory.
pub const PROMPT_GROUP_MAX:usize=1024;
/// Group size while other chats wait: about six seconds of their time.
pub const PROMPT_SHARED:usize=64;
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
/// The most the projection memo holds, charged once per server at admission. One kept
/// row of this model is 65,536 bytes packed plus its key, and a token can select up to
/// 2,051 rows in each of 11 latent layers (1.48 GB), so the memo holds about two tokens' rows.
pub const PROJECTION_MEMO_BYTES:usize=3<<30;
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
/// The same charge for a group of this many rows.
pub fn group_workspace_bytes_for(c:&CheckpointAdmission,context:usize,group:usize)->Result<usize> {
    let d=c.description();let (mut per_row,mut selected,mut gather)=(0usize,0usize,0usize);
    for layer in 0..d.layers.len() {
        // Slot and context buffers are charged below from the group's own shapes.
        let plan=LayerPlan::new(d,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
        let mut words=0usize;let mut widest=0usize;
        for spec in plan.steps.iter().flat_map(|step|&step.outputs) {
            if matches!(spec.name.as_str(),"keys"|"values"|"scores"|"probabilities"|"selected_latent"|"selected"|"selected_count"|"next_state"|"next_history"|"next_latent_cache"|"next_index_keys"|"next_index_gates"|"pool_keys"|"pool_scores") {continue;}
            let elements=spec.elements().map_err(|e|e.to_string())?;words=words.checked_add(elements).ok_or("prompt workspace overflow")?;widest=widest.max(elements);
        }
        if let engine_model::shapes::Ffn::Experts(e)=d.layers[layer].ffn {
            // Quantized activations: seventeen words for sixteen features, for gate, up and down.
            let packed=(e.selected as usize).checked_mul(((d.hidden as usize/16)*17).max(e.width as usize/16*17)).and_then(|n|n.checked_mul(2)).ok_or("prompt workspace overflow")?;
            // Permuted inputs, and the route's complete scratch for each token.
            let route=(e.selected as usize*2+e.count as usize*2+d.routing.groups as usize*2).checked_add(e.selected as usize*d.hidden as usize).ok_or("prompt workspace overflow")?;
            words=words.checked_add(packed).and_then(|n|n.checked_add(route)).ok_or("prompt workspace overflow")?;
        }
        if let TokenMixer::LatentAttention(a)=d.layers[layer].mixer {
            let i=a.indexer;let slots=(i.token_budget+if i.select_tail {i.pool-1} else {0}) as usize;let live=slots.min(context);
            let row=(a.heads*(a.key_dim+a.value_dim)) as usize;
            // Two buffers per head, the selection, three words per entry of the Scores table,
            // and the pair map, which holds one word per token for each row of a small union.
            words=words.checked_add(live.checked_mul(a.heads as usize*2).ok_or("prompt workspace overflow")?).and_then(|n|n.checked_add(slots+1+live*3+context.min(SCORE_MAP_ROWS))).ok_or("prompt workspace overflow")?;
            // Past the persistent limit, a layer holds only rows its tokens selected.
            // The gathered latents of those rows live beside them, with one chunk being copied.
            // The rows are packed (row/2 words); the gathered latents stay full words. One chunk
            // also sits unpacked between its projection and its pack, hence one group's plain rows.
            selected=selected.max(context.min(group.checked_mul(live).ok_or("prompt workspace overflow")?).checked_add(group).and_then(|rows|rows.checked_mul(row/2+a.kv_rank as usize)).and_then(|n|n.checked_add(group.checked_mul(row)?)).ok_or("prompt workspace overflow")?);
            // Absorbed attention replaces the expanded rows with five planes that a group holds at once.
            // Charge the larger alternative, with the same head count as the expanded rows above.
            selected=selected.max(absorbed_group_words(group,a.heads as usize,a.kv_rank as usize,a.value_dim as usize,live).ok_or("prompt workspace overflow")?);
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
    per_row.checked_mul(group).and_then(|n|n.checked_add(selected)).and_then(|n|n.checked_add(exchange)).and_then(|n|n.checked_mul(4)).ok_or_else(||"prompt workspace overflow".into())
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
/// The memo bytes a memory proof must hold as ballast: the device's memo limit, and only when
/// it equals what admission charges. A smaller limit would prove less than a server can reach.
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
fn row_shape(op:Row,m:usize)->Result<RowShape> {RowShape::new(op,m).ok_or_else(||"prompt row count exceeds ABI".into())}
impl<'f,C:Compute> Decoder<'f,C> {
    /// Test and admission control. Bits do not depend on this limit.
    pub fn set_expanded_limit(&self,rows:usize) {self.expanded_limit.set(rows);}
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
        self.rows_pass(tokens,state,head,0).map(|(logits,next,_)|(logits,next))
    }
    /// The same group, which also keeps what `prefix_state` needs.
    pub fn prefill_rows_open(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows)->Result<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)> {
        let (logits,next,open)=self.rows_pass(tokens,state,head,2)?;Ok((logits,next,open.ok_or("prompt group kept no rows")?))
    }
    /// The same group, which keeps only each row's last hidden state (`open_hidden`), for a
    /// proposer's cache of prompt positions. `prefix_state` does not apply to it.
    pub fn prefill_rows_hidden(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows)->Result<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)> {
        let (logits,next,open)=self.rows_pass(tokens,state,head,1)?;Ok((logits,next,open.ok_or("prompt group kept no hidden rows")?))
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
    /// One group with launch recording paused (PROVED-3): every rows op launches directly, and a new
    /// row count or chunk position captures no graph pieces. Same commands, same order, same bits
    /// (measured on the device wire: logits and every state plane equal). Recording resumes after,
    /// also when the group fails.
    fn rows_pass(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows,keep:u8)->Result<(Vec<Vec<f32>>,State<C::Value>,Option<OpenRows<C::Value>>)> {
        let was=self.compute.rows_unrecorded(true)?;let result=self.rows_pass_body(tokens,state,head,keep);
        let restored=self.compute.rows_unrecorded(was);let value=result?;restored?;Ok(value)
    }
    fn rows_pass_body(&self,tokens:&[u32],state:&State<C::Value>,head:HeadRows,keep:u8)->Result<(Vec<Vec<f32>>,State<C::Value>,Option<OpenRows<C::Value>>)> {
        let c=self.checkpoint;let d=c.description();let m=tokens.len();
        need(m>0 && m<=PROMPT_GROUP_MAX && state.layers.len()==d.layers.len(),"prompt group differs")?;
        need(state.position.checked_add(m as u64).is_some_and(|n|n<=self.capacity && n<=u32::MAX as u64),"prompt group exceeds capacity")?;
        need(tokens.iter().all(|&t|u64::from(t)<d.vocab),"prompt token exceeds vocabulary")?;
        if trims_spares_first(m,state.position) {self.compute.rows_keep()?;}
        let streams=d.residual.streams as usize;let hidden=d.hidden as usize;
        let mut residual=match self.compute.rows_embedding(tokens,streams)? {
            Some(value)=>value,
            None=>{let rows=tokens.iter().map(|&t|self.embedding(t)).collect::<Result<Vec<_>>>()?;self.compute.join(&rows.iter().collect::<Vec<_>>())?},
        };
        let mut layers=Vec::new();let mut expanded=BTreeMap::new();let mut kept_layers=Vec::new();
        for layer in 0..d.layers.len() {
            let (output,next,cache,kept)=self.layer_rows(layer,state.position,m,residual,&state.layers[layer],state.expanded.get(&layer).filter(|v|v.1<=state.position as usize && v.1<=self.expanded_limit.get()),keep==2)?;
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
        let vocab=d.vocab as usize;let mut logits=Vec::new();
        let first=match head {HeadRows::Last=>m-1,HeadRows::All=>0,HeadRows::From(row)=>{need(row<m,"prompt head row is outside its group")?;row}};
        // Each head row has its own launch group, so the logits stay bounded.
        let mut at=first;while at<m {let count=(m-at).min(32);
            let rows=self.compute.range(&normalized,at*hidden,count*hidden)?;
            let values=self.compute.read(&self.project_group("lm_head.weight",&rows,count,true)?)?;
            need(values.len()==count*vocab && values.iter().all(|v|v.is_finite()),"prompt logits differ")?;
            logits.extend(values.chunks_exact(vocab).map(<[f32]>::to_vec));at+=count;}
        // Keep the group's spares, trimmed to the pool's normal room, instead of releasing them:
        // the next group, or the next session's first group, reuses the same buffer sizes.
        if m>=PROMPT_SHARED {self.compute.rows_keep()?;}
        // Each row's last hidden state, before and after the final norm, for a proposer.
        let open=if keep>0 {Some(OpenRows {first:state.position,rows:m,layers:kept_layers,hidden:[self.compute.range(&mean,0,m*hidden)?,self.compute.range(&normalized,0,m*hidden)?]})} else {None};
        Ok((logits,State {position:state.position+m as u64,layers,expanded},open))
    }
    /// Token-major `[m,rows]` from token-major `[m,cols]`, gathered on every rank.
    pub(super) fn project_group(&self,name:&str,input:&C::Value,m:usize,round:bool)->Result<C::Value> {
        if let Some(value)=self.compute.rows_project(self.checkpoint,name,input,m,round)? {return Ok(value);}
        let view=self.checkpoint.headers().weights().get(name).ok_or("missing prompt matrix")?;let cols=view.logical().shape[1] as usize;
        let mut rows=Vec::new();for token in 0..m {rows.push(self.project(name,&self.compute.range(input,token*cols,cols)?,round)?);}
        self.compute.join(&rows.iter().collect::<Vec<_>>())
    }
    fn layer_rows(&self,layer:usize,first:u64,m:usize,residual:C::Value,state:&BTreeMap<String,C::Value>,expanded:Option<&Expanded<C::Value>>,keep:bool)->Result<(C::Value,BTreeMap<String,C::Value>,Option<C::Value>,Option<KeptRows<C::Value>>)> {
        if let Some(owner)=expanded {need(owner.2==self.sparse_layout(layer)?,"expanded prompt owner layout differs")?;}
        let c=self.checkpoint;let d=c.description();let b=&self.compute;let last=first+m as u64;
        let planning=std::time::Instant::now();
        let plan={let mut schedules=self.schedules.borrow_mut();schedules.get_mut(layer).ok_or("prompt layer is outside the recipe")?.at(last).cloned().map_err(|e|e.to_string())?};
        b.plan_cost(planning.elapsed().as_nanos());
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        let mut kept_conv=None;let mut kept_step=None;
        let mut passes=BTreeMap::new();let mut skip=0;let mut cache=None;let mut cache_rows=last;let mut selected=Vec::new();
        // With a proposer a pass follows a few steps, each time. Its held rows then follow the decoded rows
        // (admission charges them by the request's tokens), so no pass expands a decoded row twice.
        let persist=last as usize<=self.expanded_limit.get() && (first==0 || expanded.is_some_and(|v|v.1==first as usize || self.draft.is_some() && v.1<first as usize));
        for index in 0..plan.steps.len() {
            if skip>0 {skip-=1;continue;}
            let step=&plan.steps[index];
            if let Some(done)=self.rows_stateless(layer,index,&plan,m,&mut values,&mut passes)? {skip=done;continue;}
            match step.kind {
                Kind::Conv=>{let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("convolution requires a recurrent layer".into());};
                    let mut s=row_shape(Row::ConvRows,m)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;
                    let x=[Some(&values[&step.inputs[0]]),Some(&values[&step.inputs[1]]),Some(&values[&step.inputs[2]]),state.get(&step.inputs[3]),None];
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                    let (convolved,history)=self.conv_rows(s,x,&weight)?;
                    if keep {let (lengths,_,_,_)=s.lengths().ok_or("invalid convolution row shape")?;
                        kept_conv=Some(([b.range(&values[&step.inputs[0]],0,lengths[0])?,b.range(&values[&step.inputs[1]],0,lengths[1])?,b.range(&values[&step.inputs[2]],0,lengths[2])?],step.inputs[3].clone()));}
                    values.insert(step.outputs[0].name.clone(),convolved);values.insert(step.outputs[1].name.clone(),history);},
                Kind::KdaStep=>{let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("recurrence requires a recurrent layer".into());};
                    let mut s=row_shape(Row::StepRows,m)?;s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;
                    // Decay does not depend on the recurrent state. Only the update runs in token order.
                    let decay=b.prompt(RowShape {op:Row::StepDecay,..s},[Some(&values[&step.inputs[0]]),None,None,None,None],&[])?;
                    let columns=b.prompt(s,[Some(&values[&step.inputs[0]]),state.get(&step.inputs[1]),Some(&decay),None,None],&[])?;
                    if keep {let (lengths,_,_,_)=s.lengths().ok_or("invalid recurrence row shape")?;
                        kept_step=Some((b.range(&values[&step.inputs[0]],0,lengths[0])?,b.range(&decay,0,lengths[2])?,step.inputs[1].clone()));}
                    let (h,d)=(self.kda_layout(layer)?.local_heads(),k.head_dim as usize);let output=b.range(&columns,h*d*d,m*h*d)?;let next=b.range(&columns,0,h*d*d)?;
                    values.insert(step.outputs[0].name.clone(),output);values.insert(step.outputs[1].name.clone(),next);},
                Kind::CacheAppend=>{let old=state.get(&step.inputs[0]).ok_or("prompt cache state is absent")?;
                    let value=b.join(&[old,&values[&step.inputs[1]]])?;values.insert(step.outputs[0].name.clone(),value);},
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
                            let result=if m<=VERIFY_ROWS {pools::append(b,s,first as usize,m,x,prior,&weight)?} else {pools::append_window(b,s,first as usize,m,x,prior,&weight)?};
                            for (spec,value) in step.outputs.iter().zip(result) {values.insert(spec.name.clone(),value);}
                            continue;
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
                    let words=slots as usize+1;let read=b.read(&value)?;need(read.len()==m*words,"prompt selection extent differs")?;
                    for (token,row) in read.chunks_exact(words).enumerate() {need(row[words-1]>=0.0 && row[words-1] as u64<=slots.min(first+1+token as u64),"prompt selection count is invalid")?;}
                    selected=read;if self.forced_union.get().is_some() {force_distinct_rows(&mut selected,slots as usize,first as usize);}
                    values.insert(step.outputs[0].name.clone(),value);},
                Kind::Gather=>{},
                Kind::ExpandKv=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("KV expansion requires a latent recipe".into());};
                    if b.has_latent_absorption(&step.weights[0].name) {
                        let (query_name,output_name)=indirect::attention_tail(&plan,index).ok_or("absorbed prompt tail is not private")?;
                        let layout=self.sparse_layout(layer)?;let slots=(a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0}) as usize;
                        need(selected.len()==m*(slots+1),"absorbed prompt selection extent differs")?;
                        let queries=self.local_query(&values[query_name],m,layout)?;
                        for (token,row) in selected.chunks_exact(slots+1).enumerate() {
                            let count=row[slots];need(count.is_finite() && count>=0.0 && count.fract()==0.0 && count<=slots.min(first as usize+token+1) as f32,"absorbed prompt count differs")?;
                            need(row[..count as usize].iter().all(|&id|id.is_finite() && id>=0.0 && id.fract()==0.0 && id<(first as usize+token+1) as f32),"absorbed prompt row exceeds its context")?;
                        }
                        let selected=b.upload(&selected)?;let mut s=row_shape(Row::AbsorbQuery,m)?;
                        s.heads=number(layout.local_heads() as u64)?;s.dim=number(a.key_dim)?;s.taps=number(a.value_dim)?;
                        s.cols=number(a.kv_rank)?;s.rows=number(last)?;s.slots=number(slots as u64)?;s.first=number(slots.min(last as usize) as u64)?;
                        let local=b.latent_absorption_rows(&step.weights[0].name,s,&queries,&values["next_latent_cache"],&selected)?;
                        values.insert(output_name.to_owned(),b.gather_attended(&local,m,layout)?);skip=3;continue;
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
                        cache=Some(b.rows_extend(expanded.map(|v|&*v.0),held,&new,packed_words,room)?);continue;
                    }
                    // Only rows some token selected. Selected rows are pure, so they
                    // need no order among themselves. New IDs point into this matrix.
                    let i=a.indexer;let slots=(i.token_budget+if i.select_tail {i.pool-1} else {0}) as usize;let words=slots+1;
                    // Reuse only the rows the caller already holds. The prompt does not
                    // extend the retained prefix or publish the temporary selected union.
                    let held=expanded.map(|v|v.1).unwrap_or(0);
                    let (rows,chosen,cached)=expansion_rows(&selected,slots,last as usize,held)?;
                    if let Some(most)=self.forced_union.get() {self.forced_union.set(Some(most.max(rows.len())));}
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
                        let row=&selected[(m-1)*words..m*words];let mut ids=row[..(row[slots] as usize).min(slots)].iter().map(|&v|v as u32).collect::<Vec<_>>();
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
                    cache_rows=rows.len() as u64;cache=Some(cache_values);
                    for row in selected.chunks_exact_mut(words) {let count=row[slots] as usize;for v in &mut row[..count] {*v=chosen[*v as usize] as f32;}}
                    values.insert("selected".to_owned(),b.upload(&selected)?);},
                Kind::Scores|Kind::Softmax|Kind::Values=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("attention requires a latent layer".into());};
                    let i=a.indexer;let slots=i.token_budget+if i.select_tail {i.pool-1} else {0};
                    let mut s=row_shape(match step.kind {Kind::Scores=>Row::ScoresRowsPacked,Kind::Softmax=>Row::SoftmaxRows,_=>Row::ValuesRowsPacked},m)?;
                    let layout=self.sparse_layout(layer)?;
                    s.heads=number(layout.local_heads() as u64)?;s.dim=number(a.key_dim)?;s.taps=number(a.value_dim)?;s.rows=number(cache_rows)?;s.cols=number(slots.min(last))?;s.slots=number(slots)?;
                    // Order score chains by row, so tokens reading one key share its lines.
                    let mut table=Vec::new();let mut map=Vec::new();
                    if step.kind==Kind::Scores {let count=slots as usize;
                        // Tiles compute all pairs, so a small union takes a map from pair to slot.
                        if cache_rows as usize<=SCORE_MAP_ROWS {map=score_map(&selected,count,cache_rows as usize,m);}
                        if map.is_empty() {
                            for (token,row) in selected.chunks_exact(count+1).enumerate() {for (slot,&id) in row[..row[count] as usize].iter().enumerate() {table.push((id as u32,token as u32,slot as u32));}}
                            table.sort_unstable();s.first=number(table.len() as u64)?;
                        } else {s.op=Row::ScoresMapPacked;}
                    }
                    let order=if !map.is_empty() {Some(b.upload(&map)?)} else if table.is_empty() {None} else {Some(b.upload(&table.iter().flat_map(|e|[e.0 as f32,e.1 as f32,e.2 as f32]).collect::<Vec<_>>())?)};
                    let selected=values.get("selected").ok_or("prompt selection is absent")?;let e=cache.as_ref().ok_or("prompt expanded rows are absent")?;
                    let value=match step.kind {
                        Kind::Scores=>{let query=self.local_query(&values[&step.inputs[0]],m,layout)?;b.prompt(s,[Some(&query),Some(e),Some(selected),order.as_ref(),None],&[])?},
                        Kind::Softmax=>b.prompt(s,[Some(&values[&step.inputs[0]]),Some(selected),None,None,None],&[])?,
                        _=>b.prompt(s,[Some(&values[&step.inputs[0]]),Some(e),Some(selected),None,None],&[])?,
                    };
                    let value=if step.kind==Kind::Values {b.gather_attended(&value,m,layout)?} else {value};
                    values.insert(step.outputs[0].name.clone(),value);},
                _=>return Err(format!("{}: prompt row operation is unsupported",step.id)),
            }
        }
        let names=match d.layers[layer].mixer {
            TokenMixer::Kda(_)=>vec![("kda.history","next_history"),("kda.state","next_state")],
            // The complete pools ride with the state, as in the step path.
            TokenMixer::LatentAttention(_)=>vec![("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")],
        };
        let output=values.remove("output").ok_or("prompt layer has no output")?;let mut next=BTreeMap::new();
        for (input,name) in names {next.insert(input.into(),values.remove(name).ok_or("prompt layer has no next state")?);}
        let kept=match (kept_conv,kept_step) {(Some((conv,history)),Some((step,decay,state)))=>Some(KeptRows {conv,history,step,decay,state}),_=>None};
        need(!keep || kept.is_some()==matches!(d.layers[layer].mixer,TokenMixer::Kda(_)),"recurrent layer kept no rows")?;
        Ok((output,next,if persist {cache} else {None},kept))
    }
    /// One step of a layer for `m` rows, when the step reads no history of a chat: each row's
    /// result then depends on that row alone. The rows may be consecutive tokens of one chat (a
    /// prompt group) or one token of each of several chats (a batch step). None: the step reads
    /// history. Some(n): the step is done, and the next n steps are done with it.
    pub(super) fn rows_stateless(&self,layer:usize,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>,passes:&mut BTreeMap<String,C::Pass>)->Result<Option<usize>> {
        let c=self.checkpoint;let d=c.description();let b=&self.compute;let step=&plan.steps[index];let mut skip=0;
        match step.kind {
            Kind::MhcPrepare=>{let key=(layer,index);let p=self.blocks.get(&key).ok_or("resident residual parameters are absent")?;
                let (pass,norm)=b.rows_begin(key,p,&values[&step.inputs[0]],m)?;
                passes.insert(step.id.split('.').next().unwrap().to_owned(),pass);values.insert(plan.steps[index+2].outputs[0].name.clone(),norm);skip=2;},
            Kind::Combine=>{let side=step.id.split('.').next().unwrap();let pass=passes.remove(side).ok_or("prompt residual pass is absent")?;
                let value=b.rows_finish(pass,&values[&step.inputs[2]],m)?;values.insert(step.outputs[0].name.clone(),value);},
            // A projection that an earlier stage of this layer made is done. Else the stage that starts here makes it.
            Kind::Project=>{if !values.contains_key(&step.outputs[0].name) {self.rows_project_stage(index,plan,m,values)?;}},
            Kind::Norm=>{let name=&step.weights[0].name;let width=step.outputs[0].elements().map_err(|e|e.to_string())?;
                let value=b.rows_norm(name,norm_shape(c,name,width)?,&values[&step.inputs[0]],m,&self.parameters[name])?;values.insert(step.outputs[0].name.clone(),value);},
            Kind::SwiGlu=>{let elements=step.outputs[0].elements().map_err(|e|e.to_string())?;let mut s=shape(Op::SwiGlu);
                s.rows=number(elements as u64*m as u64)?;s.bound=d.swiglu_limit as f32;
                let value=b.operation(s,[Some(&values[&step.inputs[0]]),Some(&values[&step.inputs[1]]),None,None],&[])?;values.insert(step.outputs[0].name.clone(),value);},
            Kind::KdaPrepare|Kind::GatedNorm=>{let TokenMixer::Kda(k)=d.layers[layer].mixer else {return Err("recurrent operation requires a recurrent layer".into());};
                let mut s=shape(if step.kind==Kind::KdaPrepare {Op::Prepare} else {Op::GatedNorm});s.epsilon=d.epsilon as f32;
                s.heads=number(self.kda_layout(layer)?.local_heads() as u64)?;s.dim=number(k.head_dim)?;s.taps=number(k.conv)?;if step.kind==Kind::KdaPrepare {s.bound=k.gate_lower_bound as f32;}
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                let mut x=[None;5];for (slot,name) in step.inputs.iter().enumerate() {x[slot]=Some(&values[name]);}
                let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,x,&weight)?;
                let value=if step.kind==Kind::GatedNorm {b.gather_attended(&value,m,self.kda_layout(layer)?)?} else {value};values.insert(step.outputs[0].name.clone(),value);},
            Kind::KeyNorm=>{let TokenMixer::LatentAttention(a)=d.layers[layer].mixer else {return Err("index key requires a latent layer".into());};
                let mut s=shape(Op::KeyNorm);s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.epsilon=plan.settings.index_key_epsilon;
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,[Some(&values[&step.inputs[0]]),None,None,None,None],&weight)?;values.insert(step.outputs[0].name.clone(),value);},
            Kind::Route=>{let mut s=shape(Op::Route);s.heads=number(d.routing.groups)?;s.dim=number(d.routing.selected_groups)?;s.rows=number(step.weights[0].shape[0])?;
                s.cols=number(step.outputs[0].shape[0])?;s.taps=u32::from(d.routing.normalize_selected);s.bound=d.routing.scale as f32;s.epsilon=d.epsilon as f32;
                let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(&self.parameters[&w.name]);}
                let (_,_,_,words)=s.lengths().ok_or("invalid prompt route shape")?;
                let result=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,[Some(&values[&step.inputs[0]]),None,None,None,None],&weight)?;
                let routes=b.read(&result)?;need(routes.len()==m*words,"prompt route extent differs")?;let k=s.cols as usize;
                let mut ids=Vec::with_capacity(m*k);let mut chosen=Vec::with_capacity(m*k);
                for row in routes.chunks_exact(words) {ids.extend_from_slice(&row[..k]);chosen.extend_from_slice(&row[k..2*k]);}
                values.insert(step.outputs[0].name.clone(),b.upload_known(&ids)?);values.insert(step.outputs[1].name.clone(),b.upload(&chosen)?);},
            Kind::ExpertProject|Kind::ExpertDown=>{skip=self.experts_group(index,plan,m,values)?;},
            Kind::ExpertSum=>{let route=plan.steps.iter().find(|s|s.kind==Kind::Route).ok_or("prompt expert fold has no route")?;
                let mut s=shape(Op::ExpertFold);s.rows=number(route.outputs[0].shape[0])?;s.cols=number(d.hidden)?;s.taps=u32::from(step.inputs.len()==4);
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
    fn conv_rows(&self,s:RowShape,x:[Option<&C::Value>;5],weight:&[f32])->Result<(C::Value,C::Value)> {
        let b=&self.compute;let (m,run_rows)=(s.tokens as usize,spark_gpu_probe::prompt_admission::GROUP_MAX);
        if m<=run_rows {return Ok((b.prompt(s,x,weight)?,b.prompt(RowShape {op:Row::HistoryRows,..s},x,&[])?));}
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
    /// Rows choosing one expert share one weight pass. Host routes only place
    /// rows. Every output row returns to its own (token,slot) unchanged.
    fn experts_group(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>)->Result<usize> {
        let b=&self.compute;let step=&plan.steps[index];
        let ids=b.read(&values[&step.inputs[1]])?;let k=step.outputs[0].shape[0] as usize;need(ids.len()==m*k,"prompt route count differs")?;
        let mut groups:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
        for (token,row) in ids.chunks_exact(k).enumerate() {for (slot,&id) in row.iter().enumerate() {
            need(id>=0.0 && id.fract()==0.0 && (id as usize)<step.weights.len(),"prompt expert ID is outside the recipe")?;
            need(!row[..slot].contains(&id),"prompt route repeats an expert")?;groups.entry(id as usize).or_default().push(token*k+slot);
        }}
        if step.id=="expert_gate" {super::batch::note_expert_union(plan.layer,m*k,groups.len());}
        let width=step.weights[0].shape[1] as usize;let out_width=step.weights[0].shape[0] as usize;let down=step.kind==Kind::ExpertDown;
        // Rows of one expert become contiguous, in ascending expert order.
        let mut order=Vec::with_capacity(m*k);let mut spans=Vec::with_capacity(groups.len());
        for (&id,rows) in &groups {spans.push((id,order.len(),rows.len()));order.extend(rows.iter().map(|&row|if down {row} else {row/k} as f32));}
        let permute=|source:&C::Value,table:&[f32],rows:usize,cols:usize,from:usize|->Result<C::Value> {
            let table=b.upload(table)?;let mut s=row_shape(Row::PermuteRows,1)?;s.rows=number(rows as u64)?;s.cols=number(cols as u64)?;s.first=number(from as u64)?;
            b.prompt(s,[Some(source),Some(&table),None,None,None],&[])};
        // Three numbers per output row: earlier rows, this expert's rows, row index.
        let mut grouped=Vec::with_capacity(3*m*k);let mut placed=vec![0.0f32;3*m*k];
        for &(id,at,count) in &spans {for (position,&row) in groups[&id].iter().enumerate() {
            grouped.extend_from_slice(&[at as f32,count as f32,position as f32]);placed[3*row..3*row+3].copy_from_slice(&[at as f32,count as f32,position as f32]);}}
        fn names<'s>(step:&'s engine_model::plan::layer::Step,spans:&[(usize,usize,usize)])->Vec<(&'s str,usize)> {spans.iter().map(|&(id,_,count)|(step.weights[id].name.as_str(),count)).collect()}
        if !down && step.id=="expert_gate" {
            let up=&plan.steps[index+1];let swiglu=&plan.steps[index+2];need(up.id=="expert_up" && swiglu.kind==Kind::SwiGlu,"prompt expert program differs")?;
            let source=permute(&values[&step.inputs[0]],&order,m*k,width,m)?;
            if let Some(middle)=b.rows_experts(&names(step,&spans),Some((&names(up,&spans),self.checkpoint.description().swiglu_limit as f32)),&source,&grouped)? {
                values.insert("expert_middle_grouped".to_owned(),middle);return Ok(2);
            }
        }
        let held=values.remove("expert_middle_grouped");
        let source=match held {Some(v)=>v,None=>permute(&values[&step.inputs[0]],&order,m*k,width,if down {m*k} else {m})?};
        if down&& let Some(value)=b.rows_experts(&names(step,&spans),None,&source,&placed)? {values.insert(step.outputs[0].name.clone(),value);return Ok(0);}
        let mut parts=Vec::with_capacity(spans.len());
        for &(id,at,count) in &spans {
            let rows=b.range(&source,at*width,count*width)?;
            parts.push(self.project_group(&step.weights[id].name,&rows,count,true)?);
        }
        let joined=b.join(&parts.iter().collect::<Vec<_>>())?;drop(parts);
        // Invert the placement: output (token,slot) reads its row in the joined matrix.
        let mut inverse=vec![0.0f32;m*k];let mut at=0usize;for rows in groups.values() {for &row in rows {inverse[row]=at as f32;at+=1;}}
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
