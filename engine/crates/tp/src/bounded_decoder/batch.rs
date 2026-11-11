//! One decode step for several chats. A step that reads no history of a chat runs once for
//! all the chats, on the prompt-row forms: one weight read serves every chat. A step that reads
//! one chat's history (its recurrent state, its caches, its attention) runs once a chat, as the
//! plain step runs it. Each chat's bits equal its own `step`, whatever the other chats are.
use super::*;
use spark_gpu_probe::prompt_ops::Shape as RowShape;
use std::sync::atomic::{AtomicU64,Ordering};
/// Most chats in one batch step.
pub const BATCH_ROWS:usize=8;
/// Fewest chats a server sends as one batch step. Measured on the real model (two Sparks,
/// 2026-11-10): one plain step is 52.8 ms; a batch of 2 costs 107 to 122 ms, so two chats are
/// faster in turns; a batch of 4 costs 135 to 164 ms and of 8 about 204 ms, so more chats gain.
/// Bits do not depend on this number. Lower it when the batch of 2 costs under two plain steps.
pub const BATCH_MIN_ROWS:usize=3;
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
/// Steps that read one chat's history. Every other step of a layer depends on its row alone.
fn reads_history(kind:Kind)->bool {
    matches!(kind,Kind::Conv|Kind::KdaStep|Kind::CacheAppend|Kind::Pool|Kind::ScorePools|Kind::Select|Kind::Gather|Kind::ExpandKv|Kind::Scores|Kind::Softmax|Kind::Values)
}
impl<'f,C:Compute> Decoder<'f,C> {
    /// One token for each chat: `rows[i]` is a chat's input token and its state. Result `i` is
    /// that chat's logits, its greedy ID and its next state, with the bits of `step` on that
    /// chat alone. The chats may stand at different positions. No given state changes.
    pub fn step_batch(&self,rows:&[(u32,&State<C::Value>)])->Result<Vec<(Vec<f32>,u32,State<C::Value>)>> {
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
        let vocab=d.vocab as usize;let values=self.compute.read(&self.project_group("lm_head.weight",&normalized,n,true)?)?;
        need(values.len()==n*vocab && values.iter().all(|value|value.is_finite()),"batch step logits differ")?;
        let mut results=Vec::with_capacity(n);
        for ((logits,(_,state)),layers) in values.chunks_exact(vocab).zip(rows).zip(layers) {
            results.push((logits.to_vec(),greedy_logits(logits,vocab)?,State {position:state.position+1,layers,expanded:state.expanded.clone()}));
        }
        Ok(results)
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
        b.plan_cost(planning.elapsed().as_nanos());
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        let mut chats=(0..n).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        let mut passes=BTreeMap::new();let mut skip=0;
        for index in 0..plan.steps.len() {
            if skip>0 {skip-=1;continue;}
            let step=&plan.steps[index];
            if !reads_history(step.kind) {
                // What each chat made alone becomes one chat-major value for the shared step.
                for name in &step.inputs {
                    if values.contains_key(name) || !chats.iter().all(|chat|chat.contains_key(name)) {continue;}
                    let joined=b.join(&chats.iter().map(|chat|&chat[name]).collect::<Vec<_>>())?;values.insert(name.clone(),joined);
                }
                skip=self.rows_stateless(layer,index,&plan,n,&mut values,&mut passes)?.ok_or_else(||format!("{}: batch step operation is unsupported",step.id))?;continue;
            }
            let end=index+plan.steps[index..].iter().take_while(|step|reads_history(step.kind)).count();
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
