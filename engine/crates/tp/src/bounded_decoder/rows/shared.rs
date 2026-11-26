//! SHARED PASS: one row pass for the rows of several chats (the check rows of the drafts of two chats in ONE pass).
//!
//! A row pass has two kinds of steps. A step that reads no history of a chat (`rows_stateless`: the mHC boundary, the
//! norms, every projection, the route, the experts and their fold) gives each row a result that depends on that row
//! alone, so it runs ONCE on the rows of all chats, and one read of a weight serves them all. A step that reads one
//! chat's history (its recurrent state, its caches, its attention: `rows_history`) runs once a chat, on that chat's own
//! rows, at that chat's own position, with the very calls of that chat's own pass.
//!
//! So each chat gets the bits of its own pass (`Decoder::prefill_rows_open`): its logits, its next state, and its own
//! open rows, from which `prefix_state` builds the state after any prefix of ITS rows. One chat may keep every row of
//! the pass while another keeps one.
use super::*;
use super::super::batch::reads_history;
/// One chat of a shared pass: its rows in order (rows it carried first), its committed state, and the first of its
/// rows that goes through the head.
pub struct SharedRows<'a,V> {pub tokens:&'a [u32],pub state:&'a State<V>,pub head:usize}
/// What a layer leaves one chat: its next state, its new held rows, the kept inputs of a recurrent layer.
type ChatLayer<V>=(BTreeMap<String,V>,Option<V>,Option<KeptRows<V>>);
impl<'f,C:Compute> Decoder<'f,C> {
    /// One pass for the rows of several chats. Result `i` is chat `i`'s: the logits of its rows from `head` on, its
    /// state after its rows, and its open rows, each with the bits of that chat's own `prefill_rows_open`. No given
    /// state changes. At most VERIFY_ROWS rows in all: the routed lanes of a check pass are warmed for that many.
    #[allow(clippy::type_complexity)] // one result a chat: what `prefill_rows_open` gives.
    pub fn prefill_rows_shared(&self,chats:&[SharedRows<'_,C::Value>])->Result<Vec<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)>> {
        let total=chats.iter().map(|chat|chat.tokens.len()).sum::<usize>();
        // As a check pass of one chat: recording stays as the caller has it, and the routed lanes take the rows.
        self.compute.rows_verify(self.predictor_enabled() && (2..=self.verify_routes.get()).contains(&total));
        let result=self.rows_shared(chats,total);
        self.compute.rows_verify(false);result
    }
    #[allow(clippy::type_complexity)]
    fn rows_shared(&self,chats:&[SharedRows<'_,C::Value>],total:usize)->Result<Vec<(Vec<Vec<f32>>,State<C::Value>,OpenRows<C::Value>)>> {
        let c=self.checkpoint;let d=c.description();let n=chats.len();
        need((2..=super::super::BATCH_ROWS).contains(&n) && total<=super::super::VERIFY_ROWS,"shared pass exceeds its bound")?;
        for chat in chats {
            let m=chat.tokens.len();
            need(m>0 && chat.head<m && chat.state.layers.len()==d.layers.len(),"shared pass chat differs")?;
            need(chat.state.position.checked_add(m as u64).is_some_and(|last|last<=self.capacity && last<=u32::MAX as u64),"shared pass exceeds capacity")?;
            need(chat.tokens.iter().all(|&token|u64::from(token)<d.vocab),"shared pass token exceeds vocabulary")?;
        }
        let streams=d.residual.streams as usize;let hidden=d.hidden as usize;
        let tokens=chats.iter().flat_map(|chat|chat.tokens.iter().copied()).collect::<Vec<_>>();
        let mut residual=match self.compute.rows_embedding(&tokens,streams)? {
            Some(value)=>value,
            None=>{let rows=tokens.iter().map(|&token|self.embedding(token)).collect::<Result<Vec<_>>>()?;self.compute.join(&rows.iter().collect::<Vec<_>>())?},
        };
        let mut layers=(0..n).map(|_|Vec::with_capacity(d.layers.len())).collect::<Vec<_>>();
        let mut expanded=(0..n).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        let mut kept=(0..n).map(|_|Vec::with_capacity(d.layers.len())).collect::<Vec<_>>();
        for layer in 0..d.layers.len() {
            let (output,each)=self.layer_rows_shared(layer,chats,total,residual)?;residual=output;
            for (at,(next,cache,rows)) in each.into_iter().enumerate() {
                let state=chats[at].state;layers[at].push(next);kept[at].push(rows);
                // As the one-chat pass: new held rows, or the rows held before the pass.
                if let Some(cache)=cache {expanded[at].insert(layer,Expanded(std::rc::Rc::new(cache),state.position as usize+chats[at].tokens.len(),self.sparse_layout(layer)?));}
                else if let Some(held)=state.expanded.get(&layer) {expanded[at].insert(layer,held.clone());}
            }
        }
        let mut s=shape(Op::StreamMean);s.heads=number(d.residual.streams)?;s.dim=number(d.hidden)?;
        let mean=self.compute.prompt(RowShape::wrapped(s,total).ok_or("shared pass row count exceeds ABI")?,[Some(&residual),None,None,None,None],&[])?;
        let name="model.language_model.norm.weight";
        let normalized=self.compute.rows_norm(name,norm_shape(c,name,hidden)?,&mean,total,&self.parameters[name])?;
        // The head rows of every chat are one launch group: one read of the head for all of them.
        let count=chats.iter().map(|chat|chat.tokens.len()-chat.head).sum::<usize>();
        let head=if count==total {self.compute.range(&normalized,0,total*hidden)?} else {
            let mut parts=Vec::with_capacity(n);let mut first=0;
            for chat in chats {let m=chat.tokens.len();parts.push(self.compute.range(&normalized,(first+chat.head)*hidden,(m-chat.head)*hidden)?);first+=m;}
            self.compute.join(&parts.iter().collect::<Vec<_>>())?};
        let vocab=d.vocab as usize;let logits=self.compute.read(&self.project_group("lm_head.weight",&head,count,true)?)?;
        need(logits.len()==count*vocab && logits.iter().all(|value|value.is_finite()),"shared pass logits differ")?;
        let mut rows=logits.chunks_exact(vocab);let mut first=0;let mut results=Vec::with_capacity(n);
        for (((chat,layers),expanded),kept) in chats.iter().zip(layers).zip(expanded).zip(kept) {
            let m=chat.tokens.len();let own=rows.by_ref().take(m-chat.head).map(<[f32]>::to_vec).collect::<Vec<_>>();
            let open=OpenRows {first:chat.state.position,rows:m,layers:kept,hidden:[self.compute.range(&mean,first*hidden,m*hidden)?,self.compute.range(&normalized,first*hidden,m*hidden)?]};
            results.push((own,State {position:chat.state.position+m as u64,layers,expanded},open));first+=m;
        }
        Ok(results)
    }
    /// One layer of a shared pass. `residual` holds the rows of all chats, chat after chat. Returns the layer's output
    /// in the same form and what the layer leaves each chat.
    fn layer_rows_shared(&self,layer:usize,chats:&[SharedRows<'_,C::Value>],total:usize,residual:C::Value)->Result<(C::Value,Vec<ChatLayer<C::Value>>)> {
        let b=&self.compute;let n=chats.len();let planning=std::time::Instant::now();
        // Every chat's program has the same steps. Only shapes that follow the context differ, and only steps that read
        // history have such shapes (the batch step stands on the same fact). So each chat's history steps read its own
        // plan, and the shared steps read the first chat's.
        let mut sides=Vec::with_capacity(n);let mut plans=Vec::with_capacity(n);
        for chat in chats {
            let (first,m)=(chat.state.position,chat.tokens.len());
            let held=chat.state.expanded.get(&layer).filter(|v|v.1<=first as usize && v.1<=self.expanded_limit.get());
            sides.push(self.rows_chat(layer,first,m,&chat.state.layers[layer],held,true)?);
            plans.push({let mut schedules=self.schedules.borrow_mut();schedules.get_mut(layer).ok_or("shared pass layer is outside the recipe")?.at(first+m as u64).cloned().map_err(|e|e.to_string())?});
        }
        b.plan_cost("planning",planning.elapsed().as_nanos());
        let plan=&plans[0];
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),residual);
        let mut own=(0..n).map(|_|BTreeMap::new()).collect::<Vec<_>>();
        let mut passes=BTreeMap::new();let mut fold=RowsFold::default();let mut skip=0;
        for index in 0..plan.steps.len() {
            if skip>0 {skip-=1;continue;}
            let step=&plan.steps[index];
            if !reads_history(step.kind) {
                // What each chat made alone becomes one value for the shared step, chat after chat.
                for name in &step.inputs {
                    if values.contains_key(name) || !own.iter().all(|chat|chat.contains_key(name)) {continue;}
                    let joined=b.join(&own.iter().map(|chat|&chat[name]).collect::<Vec<_>>())?;values.insert(name.clone(),joined);
                }
                skip=self.rows_stateless(layer,index,plan,total,&mut values,&mut passes,&mut fold)?.ok_or_else(||format!("{}: shared pass operation is unsupported",step.id))?;continue;
            }
            let end=index+plan.steps[index..].iter().take_while(|step|reads_history(step.kind)).count();
            let mut first=0;
            for ((side,plan),own) in sides.iter_mut().zip(&plans).zip(own.iter_mut()) {
                // This chat's rows of each shared value that its history steps read.
                for name in plan.steps[index..end].iter().flat_map(|step|&step.inputs) {
                    if own.contains_key(name) {continue;}
                    let Some(whole)=values.get(name) else {continue;};
                    let words=b.words(whole);need(words.is_multiple_of(total),"shared pass value is not one row a token")?;let width=words/total;
                    own.insert(name.clone(),b.range(whole,first*width,side.m*width)?);
                }
                let mut at=index;while at<end {at+=1+self.rows_history(layer,at,plan,side,own)?;}
                need(at==end,"shared pass history span differs")?;first+=side.m;
            }
            skip=end-index-1;
        }
        let output=values.remove("output").ok_or("shared pass layer has no output")?;
        let each=sides.into_iter().zip(own.iter_mut()).map(|(side,own)|self.rows_chat_end(layer,side,own)).collect::<Result<Vec<_>>>()?;
        Ok((output,each))
    }
}
