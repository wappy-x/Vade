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
        for layer in 0..d.layers.len() {
            let states=rows.iter().map(|row|&row.1.layers[layer]).collect::<Vec<_>>();
            let held=rows.iter().map(|row|row.1.expanded.get(&layer)).collect::<Vec<_>>();
            let (output,next)=self.layer_batch(layer,&contexts,residual,&states,&held)?;
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
    fn layer_batch(&self,layer:usize,contexts:&[u64],residual:C::Value,states:&[&BTreeMap<String,C::Value>],held:&[Option<&Expanded<C::Value>>])->Result<(C::Value,Vec<BTreeMap<String,C::Value>>)> {
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
            for (chat,own) in chats.iter_mut().enumerate() {
                // The chat's row of each shared value these steps read.
                for name in plan.steps[index..end].iter().flat_map(|step|&step.inputs) {
                    if own.contains_key(name) {continue;}
                    let Some(whole)=values.get(name) else {continue;};
                    // One row a chat, as the value is on this rank: a value of a recurrent layer holds this rank's own heads,
                    // so its row is not the plan's full width.
                    let words=b.words(whole);need(words%n==0,"batch step shared value is not one row a chat")?;let width=words/n;
                    own.insert(name.clone(),b.range(whole,chat*width,width)?);
                }
                self.layer_steps(layer,contexts[chat],&[],states[chat],held[chat],false,false,Some((index..end,own)))?;
            }
            skip=end-index-1;
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
}
