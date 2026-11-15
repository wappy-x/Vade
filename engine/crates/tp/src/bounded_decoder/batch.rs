//! One decode step for several chats. A step that reads no history of a chat runs once for
//! all the chats, on the prompt-row forms: one weight read serves every chat. A step that reads
//! one chat's history (its recurrent state, its caches, its attention) runs once a chat, as the
//! plain step runs it. Each chat's bits equal its own `step`, whatever the other chats are.
use super::*;
use spark_gpu_probe::prompt_ops::Shape as RowShape;
use std::sync::atomic::{AtomicU64,Ordering};
/// Most chats in one batch step.
pub const BATCH_ROWS:usize=8;
/// Fewest chats a server sends as one batch step when three or more chats are open. Measured on the
/// real model (two Sparks, served, 2026-11-12): a batch of 3 costs 95.3 ms, of 4 109.3 ms, of 7 158.5 ms
/// a step, each under that many plain steps of 42.6 ms. Bits do not depend on this number.
pub const BATCH_MIN_ROWS:usize=3;
/// Two chats that are alone on the server ride one batch step (`ROUTED_CHATS` below). Measured on the
/// real model (two Sparks, served, 2026-11-12): two chats in turns take 88.4 ms a round, one batch step
/// of the two takes 77.8 ms. With a third chat open the floor stays `BATCH_MIN_ROWS`: batch steps of two
/// while a third chat read its prompt left every later step of 3 to 7 chats 4 to 6 ms slower (loads A1,
/// A2, B1; the same with the routed path off; cause not found). Bits do not depend on this number.
pub const BATCH_PAIR_ROWS:usize=2;
/// The latent-absorb chats sweep on GB10 (2026-11-12, ASTRA-ROWS sweep.log) first wins beyond both ranges at five rows.
/// Smaller groups keep the per-chat launches. This floor does not change the arithmetic of a row.
pub const CHAT_ROWS_MIN:usize=5;
fn chat_rows_admitted(rows:usize)->bool {(CHAT_ROWS_MIN..=BATCH_ROWS).contains(&rows)}
static CHAT_ATTENTION_LAYERS:AtomicU64=AtomicU64::new(0);
/// Completed grouped attention layers, for the served path check.
pub fn chat_attention_layers()->u64 {CHAT_ATTENTION_LAYERS.load(Ordering::Relaxed)}
type ChatTable<V>=Option<(usize,Vec<u32>,V)>;
/// Counts are part of the key: layers can share a stride but use different selection rules.
fn chat_table<'a,C:Compute>(b:&C,stride:usize,counts:impl Iterator<Item=f32>+Clone,table:&'a mut ChatTable<C::Value>)->Result<&'a C::Value> {
    if table.as_ref().is_none_or(|(kept,live,_)|*kept!=stride || !live.iter().copied().eq(counts.clone().map(f32::to_bits))) {
        let live=counts.collect::<Vec<_>>();let n=live.len();
        need(n.checked_mul(stride).is_some_and(|rows|rows<1<<24),"chat rows exceed exact selection words")?;
        let mut words=vec![0.0f32;n*(stride+1)];
        for (chat,(row,&count)) in words.chunks_exact_mut(stride+1).zip(&live).enumerate() {
            for (slot,id) in row[..stride].iter_mut().enumerate() {*id=(chat*stride+slot) as f32;}row[stride]=count;
        }
        *table=Some((stride,live.into_iter().map(f32::to_bits).collect(),b.upload(&words)?));
    }
    Ok(&table.as_ref().ok_or("chat selection table is absent")?.2)
}
/// Layers the expert counts below can name.
const COUNTED_LAYERS:usize=64;
/// Diagnostic: for each layer, the routed experts the rows of its row passes picked, and how
/// many different experts that was (the experts the pass read). Print only.
static EXPERT_PICKS:[[AtomicU64;2];COUNTED_LAYERS]=[const {[AtomicU64::new(0),AtomicU64::new(0)]};COUNTED_LAYERS];
pub(super) fn note_expert_union(layer:usize,picks:usize,distinct:usize) {
    if let Some(counts)=EXPERT_PICKS.get(layer) {counts[0].fetch_add(picks as u64,Ordering::Relaxed);counts[1].fetch_add(distinct as u64,Ordering::Relaxed);}
}
/// For each layer: (picks, different experts) of every row pass so far.
pub fn expert_union_counts()->Vec<[u64;2]> {EXPERT_PICKS.iter().map(|counts|[counts[0].load(Ordering::Relaxed),counts[1].load(Ordering::Relaxed)]).collect()}
/// ROUTED CHATS: most chats of a batch step whose routed experts run once a chat (`Decoder::routed_span`). Measured on
/// the real model (two Sparks, 2026-11-12). Check tool, ms a step, grouped row forms against once a chat: 2 chats 84.25
/// against 77.52, 3 chats 105.22 against 100.77, 4 chats 123.77 against 126.10, 8 chats 190.69 against 220.45. In the
/// server the grouped forms' waits cost less than in the tool: 3 chats 95.24 against 99.57 ms a step, so only a batch
/// of 2 takes this path (77.94 ms a step). Once a chat pays an exchange and a weight read for each chat. Bits do not
/// depend on this number.
pub const ROUTED_CHATS:usize=2;
/// Steps that read one chat's history. Every other step of a layer depends on its row alone.
fn reads_history(kind:Kind)->bool {
    matches!(kind,Kind::Conv|Kind::KdaStep|Kind::CacheAppend|Kind::Pool|Kind::ScorePools|Kind::Select|Kind::Gather|Kind::ExpandKv|Kind::Scores|Kind::Softmax|Kind::Values)
}
impl<'f,C:Compute> Decoder<'f,C> {
    /// One token for each chat: `rows[i]` is a chat's input token and its state. Result `i` is
    /// that chat's logits, its greedy ID and its next state, with the bits of `step` on that
    /// chat alone. The chats may stand at different positions. No given state changes.
    pub fn step_batch(&self,rows:&[(u32,&State<C::Value>)])->Result<Vec<(Vec<f32>,u32,State<C::Value>)>> {self.step_batch_head(rows,false)}
    /// The same step when no chat needs its logits (every chat is greedy): each chat's ID is picked on the backend that holds
    /// the logits (`Compute::rows_greedy`), as the plain greedy step picks it, and no logits cross to the host. The logits of
    /// a result are then empty. The same ID and the same next state as `step_batch`.
    pub fn step_batch_greedy(&self,rows:&[(u32,&State<C::Value>)])->Result<Vec<(Vec<f32>,u32,State<C::Value>)>> {self.step_batch_head(rows,true)}
    fn step_batch_head(&self,rows:&[(u32,&State<C::Value>)],greedy:bool)->Result<Vec<(Vec<f32>,u32,State<C::Value>)>> {
        let c=self.checkpoint;let d=c.description();let n=rows.len();
        need(n>0 && n<=BATCH_ROWS,"batch step row count differs")?;
        for (token,state) in rows {
            need(state.position<self.capacity && state.layers.len()==d.layers.len(),"batch step state context differs")?;
            need(u64::from(*token)<d.vocab,"batch step token exceeds vocabulary")?;
        }
        let tokens=rows.iter().map(|row|row.0).collect::<Vec<_>>();let streams=d.residual.streams as usize;let hidden=d.hidden as usize;
        // What one chat's step makes anew and frees on this rank: the state and the history of each recurrent layer's own heads.
        let mut state_bytes=0usize;
        for (layer,spec) in d.layers.iter().enumerate() {
            if let TokenMixer::Kda(k)=spec.mixer {
                let (heads,dim)=(self.kda_layout(layer)?.local_heads(),k.head_dim as usize);
                state_bytes+=(heads*dim*dim+3*(k.conv as usize).saturating_sub(1)*heads*dim)*4;
            }
        }
        self.compute.batch_rows(n,state_bytes)?;
        let mut residual=match self.compute.rows_embedding(&tokens,streams)? {
            Some(value)=>value,
            None=>{let each=tokens.iter().map(|&token|self.embedding(token)).collect::<Result<Vec<_>>>()?;self.compute.join(&each.iter().collect::<Vec<_>>())?},
        };
        let contexts=rows.iter().map(|row|row.1.position+1).collect::<Vec<_>>();
        let mut layers=(0..n).map(|_|Vec::with_capacity(d.layers.len())).collect::<Vec<_>>();
        // The selection of CHAT ROWS follows from the contexts alone: one upload serves every latent layer of the step.
        let mut table=None;
        for layer in 0..d.layers.len() {
            let states=rows.iter().map(|row|&row.1.layers[layer]).collect::<Vec<_>>();
            let held=rows.iter().map(|row|row.1.expanded.get(&layer)).collect::<Vec<_>>();
            let (output,next)=self.layer_batch(layer,&contexts,residual,&states,&held,&mut table)?;
            residual=output;for (chat,state) in layers.iter_mut().zip(next) {chat.push(state);}
        }
        let mut s=shape(Op::StreamMean);s.heads=number(d.residual.streams)?;s.dim=number(d.hidden)?;
        let mean=self.compute.prompt(RowShape::wrapped(s,n).ok_or("batch row count exceeds ABI")?,[Some(&residual),None,None,None,None],&[])?;
        let name="model.language_model.norm.weight";
        let normalized=self.compute.rows_norm(name,norm_shape(c,name,hidden)?,&mean,n,&self.parameters[name])?;
        let vocab=d.vocab as usize;let projected=self.project_group("lm_head.weight",&normalized,n,true)?;
        if greedy && let Some(best)=self.compute.rows_greedy(&projected,n)? {
            need(best.len()==n && best.iter().all(|&id|u64::from(id)<d.vocab),"batch step greedy IDs differ")?;
            return Ok(best.into_iter().zip(rows).zip(layers).map(|((best,(_,state)),layers)|(Vec::new(),best,State {position:state.position+1,layers,expanded:state.expanded.clone()})).collect());
        }
        let values=self.compute.read(&projected)?;
        need(values.len()==n*vocab && values.iter().all(|value|value.is_finite()),"batch step logits differ")?;
        let mut results=Vec::with_capacity(n);
        for ((logits,(_,state)),layers) in values.chunks_exact(vocab).zip(rows).zip(layers) {
            results.push((logits.to_vec(),greedy_logits(logits,vocab)?,State {position:state.position+1,layers,expanded:state.expanded.clone()}));
        }
        Ok(results)
    }
    /// ROUTED CHATS. The end of the routed block that starts at step `index`, when `index` is the layer's route and the
    /// batch takes this path: the route, the picked experts' gate and up, their SwiGLU, their down, the shared expert's
    /// SwiGLU and down, and the expert fold then run once a chat, as steps of the plain decode step (`layer_steps` with a
    /// span), exactly as the history steps do. The route stays on the device there (`routed_experts`, `routed_fold`), so
    /// the layer has no route read, no table upload and no permute launch: the row forms read every row's route to the
    /// host each layer to group the rows by expert, and host and device then took turns 47 times a step. The cost: an
    /// expert's weights are read once a chat that picked it, not once for all the chats that did. The dense projections
    /// before the route (the router's logits, the shared expert's gate and up) stay one shared stage. Bits: each chat
    /// runs the steps of its own `step`.
    fn routed_span(&self,index:usize,plan:&LayerPlan,n:usize)->Option<usize> {
        // A device compute needs its device exchange for the route to stay there; the host twin reads for free.
        if plan.steps[index].kind!=Kind::Route || n>ROUTED_CHATS || self.compute.resident_weights_ready() && !self.compute.folds_routed() {return None;}
        plan.steps[index..].iter().position(|step|step.kind==Kind::ExpertSum).map(|at|index+at+1)
    }
    /// One layer of a batch step. `residual` holds one row a chat, chat-major. Returns the
    /// layer's output in the same form and each chat's next state of this layer.
    fn layer_batch(&self,layer:usize,contexts:&[u64],residual:C::Value,states:&[&BTreeMap<String,C::Value>],held:&[Option<&Expanded<C::Value>>],
        table:&mut ChatTable<C::Value>)->Result<(C::Value,Vec<BTreeMap<String,C::Value>>)> {
        let b=&self.compute;let n=contexts.len();let planning=std::time::Instant::now();
        // Every chat's program has the same steps. Only shapes that follow the context differ, and
        // only steps that read history have such shapes. So the shared steps read one plan a layer,
        // made once from the first context seen (a copy each step cost 4 ms a step on the real model).
        let kept=self.batch_plans.borrow().get(&layer).cloned();
        let plan=match kept {Some(plan)=>plan,None=>{
            let made=Rc::new(self.schedules.borrow_mut().get_mut(layer).ok_or("batch layer is outside the recipe")?.at(contexts[0]).cloned().map_err(|e|e.to_string())?);
            self.batch_plans.borrow_mut().insert(layer,made.clone());made}};
        b.plan_cost("planning",planning.elapsed().as_nanos());
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        let mut chats=(0..n).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        let mut passes=BTreeMap::new();let mut fold=super::rows::RowsFold::default();let mut skip=0;
        for index in 0..plan.steps.len() {
            if skip>0 {skip-=1;continue;}
            let step=&plan.steps[index];let routed=self.routed_span(index,&plan,n);
            if routed.is_none() && !reads_history(step.kind) {
                // What each chat made alone becomes one chat-major value for the shared step.
                for name in &step.inputs {
                    if values.contains_key(name) || !chats.iter().all(|chat|chat.contains_key(name)) {continue;}
                    let joined=b.join(&chats.iter().map(|chat|&chat[name]).collect::<Vec<_>>())?;values.insert(name.clone(),joined);
                }
                skip=self.rows_stateless(layer,index,&plan,n,&mut values,&mut passes,&mut fold)?.ok_or_else(||format!("{}: batch step operation is unsupported",step.id))?;continue;
            }
            let end=routed.unwrap_or_else(||index+plan.steps[index..].iter().take_while(|step|reads_history(step.kind)).count());
            skip=end-index-1;
            // CHAT ROWS takes a history span of five or more chats; a routed span (two chats) runs once a chat.
            if routed.is_none() && self.chat_rows(layer,index..end,&plan,contexts,states,held,&mut values,&mut chats,table)? {continue;}
            for (chat,own) in chats.iter_mut().enumerate() {
                self.chat_inputs((chat,n),&plan.steps[index..end],&values,own)?;
                self.layer_steps(layer,contexts[chat],&[],states[chat],held[chat],false,false,Some((index..end,own)))?;
            }
        }
        let output=values.remove("output").ok_or("batch layer has no output")?;
        let names:&[(&str,&str)]=match self.mixer(layer)? {
            TokenMixer::Kda(_)=>&[("kda.history","next_history"),("kda.state","next_state")],
            TokenMixer::LatentAttention(_)=>&[("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")],
        };
        let mut next=Vec::with_capacity(n);
        for chat in &mut chats {
            let mut state=BTreeMap::new();
            for (input,name) in names {state.insert((*input).to_owned(),chat.remove(*name).ok_or("batch layer has no next state")?);}
            next.push(state);
        }
        Ok((output,next))
    }
    /// CHAT ROWS. A latent layer's history span that ends in a Gather and an absorbed attention tail (PoolScores, Select,
    /// Gather, ExpandKv, Scores, Softmax, Values). Each chat runs its own steps up to the Gather as before. Its Gather
    /// writes at its own place of one latent value. One row group of the verify pass's kernels runs the attention of all
    /// chats (each row's sums in SUM-ORDER v1, the bits of one chat's step), and one gather of the attended rows serves
    /// all chats. True: the span ran and its output is a shared value. False: the span is not of that form.
    #[allow(clippy::too_many_arguments)]
    fn chat_rows(&self,layer:usize,span:std::ops::Range<usize>,plan:&LayerPlan,contexts:&[u64],states:&[&BTreeMap<String,C::Value>],
        held:&[Option<&Expanded<C::Value>>],values:&mut BTreeMap<String,C::Value>,chats:&mut [BTreeMap<String,C::Value>],
        table:&mut ChatTable<C::Value>)->Result<bool> {
        let b=&self.compute;let n=chats.len();let d=self.checkpoint.description();
        if !chat_rows_admitted(n) {return Ok(false);}
        let TokenMixer::LatentAttention(a)=self.mixer(layer)? else {return Ok(false);};
        let Some(at)=plan.steps[span.clone()].iter().position(|step|step.kind==Kind::Gather).map(|at|span.start+at) else {return Ok(false);};
        let absorbed=plan.steps.get(at+1).is_some_and(|next|next.kind==Kind::ExpandKv && !next.weights.is_empty() && b.has_latent_absorption(&next.weights[0].name));
        let tail=indirect::attention_tail(plan,at+1).filter(|_|absorbed && at+5==span.end);
        let (Some((_,output)),Some(whole_query))=(tail,tail.and_then(|(query,_)|values.get(query))) else {return Ok(false);};
        let (gather,name)=(&plan.steps[at],&plan.steps[at+1].weights[0].name);let layout=self.sparse_layout(layer)?;
        need(gather.inputs.len()==3 && plan.steps[at+1].inputs.first()==gather.outputs.first().map(|o|&o.name),"chat rows gather differs")?;
        // Complete query rows on every rank (as the one-chat path reads them): any other form keeps the chats apart.
        if b.words(whole_query)!=n*layout.total_heads()*layout.key_dim() {return Ok(false);}
        let queries=self.local_query(whole_query,n,layout)?;
        // Each chat's own steps before the Gather: its pool scores and its selection.
        for (chat,own) in chats.iter_mut().enumerate() {
            self.chat_inputs((chat,n),&plan.steps[span.start..at],values,own)?;
            self.layer_steps(layer,contexts[chat],&[],states[chat],held[chat],false,false,Some((span.start..at,own)))?;
        }
        // The layer's row bound, as the decode step's Gather reserves it: one size at every context.
        let stride=gather.outputs[0].shape[0].min(self.capacity) as usize;
        need(n.checked_mul(stride).is_some_and(|rows|rows<1<<24),"chat rows exceed exact selection words")?;
        let mut shapes=Vec::with_capacity(n);let mut picked=Vec::with_capacity(n);
        for (own,&context) in chats.iter().zip(contexts) {
            let mut s=shape(Op::Gather);s.epsilon=d.epsilon as f32;s.heads=number(a.indexer.heads)?;s.taps=number(a.indexer.pool)?;
            s.rows=number(context)?;s.dim=number(a.kv_rank)?;s.cols=number(gather.outputs[0].shape[0].min(context))?;
            let live=selected_count(a.indexer.pool,a.indexer.token_budget,a.indexer.select_tail,context);
            need(live<=u64::from(s.cols),"absorbed selection count differs")?;
            picked.push(b.range(own.get(&gather.inputs[1]).ok_or("chat selection is absent")?,0,s.cols as usize)?);shapes.push((s,live as f32));
        }
        let mut gathers=Vec::with_capacity(n);
        for ((own,&(s,live)),picked) in chats.iter().zip(&shapes).zip(&picked) {
            let input=|name:&String|own.get(name).ok_or_else(||format!("chat gather input {name} is absent"));
            gathers.push((s,[Some(input(&gather.inputs[0])?),Some(picked),Some(input(&gather.inputs[2])?),None],live));
        }
        // Chat i's selection names rows i x stride onward of the joined latent, and its count is the decode step's.
        let selected=chat_table(b,stride,shapes.iter().map(|&(_,live)|live),table)?;
        let mut s=RowShape::new(spark_gpu_probe::prompt_ops::Op::AbsorbQuery,n).ok_or("chat rows exceed the row ABI")?;
        s.heads=number(layout.local_heads() as u64)?;s.dim=number(a.key_dim)?;s.taps=number(a.value_dim)?;s.cols=number(a.kv_rank)?;
        s.rows=number((n*stride) as u64)?;s.first=number(stride as u64)?;s.slots=number(stride as u64)?;
        let Some(local)=b.latent_absorption_chats(name,s,&gathers,stride,&queries,selected)? else {
            // No row group here: each chat runs the rest of the span alone.
            drop(gathers);
            for (chat,own) in chats.iter_mut().enumerate() {
                self.chat_inputs((chat,n),&plan.steps[at..span.end],values,own)?;
                self.layer_steps(layer,contexts[chat],&[],states[chat],held[chat],false,false,Some((at..span.end,own)))?;
            }
            return Ok(true);
        };
        values.insert(output.to_owned(),b.gather_attended(&local,n,layout)?);CHAT_ATTENTION_LAYERS.fetch_add(1,Ordering::Relaxed);Ok(true)
    }
    /// Chat `chat` of `n`: its row of each shared value that `steps` read.
    fn chat_inputs(&self,(chat,n):(usize,usize),steps:&[engine_model::plan::layer::Step],values:&BTreeMap<String,C::Value>,own:&mut BTreeMap<String,C::Value>)->Result<()> {
        let b=&self.compute;
        for name in steps.iter().flat_map(|step|&step.inputs) {
            if own.contains_key(name) {continue;}
            let Some(whole)=values.get(name) else {continue;};
            // One row a chat, as the value is on this rank: a value of a recurrent layer holds this rank's own heads,
            // so its row is not the plan's full width.
            let words=b.words(whole);need(words%n==0,"batch step shared value is not one row a chat")?;let width=words/n;
            own.insert(name.clone(),b.range(whole,chat*width,width)?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_attention_floor_keeps_small_groups_on_the_per_chat_path() {
        for rows in 0..=4 {assert!(!chat_rows_admitted(rows));}
        for rows in 5..=8 {assert!(chat_rows_admitted(rows));}
        assert!(!chat_rows_admitted(9));
    }
    #[test]
    fn chat_table_changes_when_a_layer_keeps_different_counts() {
        let mut table=None;
        let first=chat_table(&Host,3,[3.0,1.0].into_iter(),&mut table).unwrap().clone();
        assert_eq!(first,vec![0.0,1.0,2.0,3.0,3.0,4.0,5.0,1.0]);
        let next=chat_table(&Host,3,[1.0,2.0].into_iter(),&mut table).unwrap().clone();
        assert_eq!(next,vec![0.0,1.0,2.0,1.0,3.0,4.0,5.0,2.0]);
        assert_eq!(*chat_table(&Host,3,[3.0,1.0].into_iter(),&mut table).unwrap(),first);
        assert_eq!(*chat_table(&Host,2,[2.0].into_iter(),&mut table).unwrap(),vec![0.0,1.0,2.0]);
        let first=selected_count(4,8,true,13) as f32;let next=selected_count(2,10,true,13) as f32;
        assert_eq!((first,next),(9.0,11.0));
        for count in [first,next,first] {
            let value=chat_table(&Host,11,[count;BATCH_ROWS].into_iter(),&mut table).unwrap();
            for (chat,row) in value.chunks_exact(12).enumerate() {assert_eq!(row[0],(chat*11) as f32);assert_eq!(row[11],count);}
        }
    }
}
