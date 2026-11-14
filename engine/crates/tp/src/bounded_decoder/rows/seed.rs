//! The predictor layer's cache of the rows that wait for it, brought up in ONE pass. The serial form is a depth-0
//! `propose_after` for each row (proposer.rs): its own chain of launches and one exchange a projection, 0.40 to 0.51 ms a
//! row on the device (pair three, 2026-11-11: 33 to 54 ms for a prompt of 83 to 106 rows). Here the cache half of the
//! layer (the steps whose outputs a later position reads) runs once for all the rows, with the prompt-row kernels: one
//! launch a step and one exchange a projection, whatever the row count. The caches are those of the serial form.
//! Proposals only choose which tokens a verify pass tries: no kept token depends on this file.
use super::*;
use super::super::proposer::PredictorState;
/// The four caches of the predictor layer: the state names and the plan outputs that make them.
const NAMES:[(&str,&str);4]=[("latent_cache","next_latent_cache"),("index_keys","next_index_keys"),("index_gates","next_index_gates"),("pool_keys","pool_keys")];
impl<'f,C:Compute> Decoder<'f,C> {
    /// `rows` rows of an open group from row `first`: the hidden plane a proposer reads, as one value.
    pub fn open_hidden_rows(&self,open:&OpenRows<C::Value>,first:usize,rows:usize)->Result<C::Value> {
        let h=self.checkpoint.description().hidden as usize;need(first.checked_add(rows).is_some_and(|end|end<=open.rows),"open rows are outside their group")?;
        self.compute.range(&open.hidden[PROPOSER_HIDDEN],first*h,rows*h)
    }
    /// Bring the predictor layer's cache up by `tokens.len()` positions in one pass. Row `r` fuses token `r` with row
    /// `r` of `hidden` (token-major: the target's last hidden state of the position before that token), as a depth-0
    /// `propose_after` does for one row. `seen` is the cache so far; None starts at position zero. Launch recording is
    /// paused as for a prompt group (`rows_pass`): the row count differs with every request.
    pub fn seed_predictor(&self,hidden:&C::Value,tokens:&[u32],seen:Option<&PredictorState<C::Value>>)->Result<PredictorState<C::Value>> {
        let was=self.compute.rows_unrecorded(true)?;let result=self.seed_rows(hidden,tokens,seen);
        let restored=self.compute.rows_unrecorded(was);let value=result?;restored?;Ok(value)
    }
    fn seed_rows(&self,hidden:&C::Value,tokens:&[u32],seen:Option<&PredictorState<C::Value>>)->Result<PredictorState<C::Value>> {
        let draft=self.draft.as_ref().ok_or("predictor layer is not enabled")?.clone();let c=self.checkpoint;let d=c.description();let b=&self.compute;
        let (h,m,layer)=(d.hidden as usize,tokens.len(),d.layers.len());let first=seen.map_or(0,PredictorState::rows);
        need(m>0 && m<=PROMPT_GROUP_MAX && tokens.iter().all(|&t|u64::from(t)<d.vocab) && b.words(hidden)>=m*h,"predictor seed rows or tokens differ")?;
        let last=first.checked_add(m as u64).filter(|&n|n<=self.capacity && n<=u32::MAX as u64).ok_or("predictor seed exceeds capacity")?;
        let TokenMixer::LatentAttention(a)=self.mixer(layer)? else {return Err("predictor seed requires a latent layer".into());};
        let plan={let mut schedule=self.draft_schedule.borrow_mut();schedule.as_mut().ok_or("predictor layer is not enabled")?.at(last).cloned().map_err(|e|e.to_string())?};
        let prefix=format!("model.language_model.layers.{layer}");
        let norm=|part:&str,x:&C::Value|->Result<C::Value> {let name=format!("{prefix}.{part}.weight");
            b.rows_norm(&name,norm_shape(c,&name,h)?,x,m,draft.parameters.get(&name).ok_or("predictor norm is absent")?)};
        // The fused input of each row, as in `propose_run`: [enorm(embedding), hnorm(hidden)]. One embedding a token, as
        // there: `rows_embedding` also starts a prompt group's row workspace and pool trim, which belong to the target's groups.
        let embedded={let rows=tokens.iter().map(|&t|b.range(&self.embedding(t)?,0,h)).collect::<Result<Vec<_>>>()?;b.join(&rows.iter().collect::<Vec<_>>())?};
        let (embedded,hidden)=(norm("enorm",&embedded)?,norm("hnorm",&b.range(hidden,0,m*h)?)?);
        let mut parts=Vec::with_capacity(2*m);for row in 0..m {parts.push(b.range(&embedded,row*h,h)?);parts.push(b.range(&hidden,row*h,h)?);}
        let fused=b.join(&parts.iter().collect::<Vec<_>>())?;drop(parts);
        let mut values=BTreeMap::new();values.insert("residual".to_owned(),self.project_group(&format!("{prefix}.eh_proj.weight"),&fused,m,true)?);
        // The earlier positions' caches, cut to the words their positions fill.
        let (rank,width,pool)=(a.kv_rank as usize,a.indexer.dim as usize,a.indexer.pool as usize);let at=first as usize;
        let mut state=BTreeMap::new();
        for (name,words) in [("latent_cache",at*rank),("index_keys",at*width),("index_gates",at*width),("pool_keys",(at/pool)*width)] {
            let value=match seen.and_then(|s|s.layers().get(name)) {Some(value)=>b.range(value,0,words)?,None if words==0=>b.zeros(0)?,None=>return Err("predictor seed cache is absent".into())};
            state.insert(name,value);}
        for index in seed_steps(&plan)? {
            let step=&plan.steps[index];
            match step.kind {
                // One residual stream: the prepare is the norm alone (see layer_steps).
                Kind::MhcPrepare=>{let target=&plan.steps[index+2];let name=&target.weights[0].name;
                    let value=b.rows_norm(name,norm_shape(c,name,h)?,&values[&step.inputs[0]],m,self.parameter(name)?)?;values.insert(target.outputs[0].name.clone(),value);},
                Kind::Project=>{let router=plan.steps.iter().any(|s|s.kind==Kind::Route && s.inputs.contains(&step.outputs[0].name));
                    let value=self.project_group(&step.weights[0].name,&values[&step.inputs[0]],m,!router)?;values.insert(step.outputs[0].name.clone(),value);},
                Kind::Norm=>{let name=&step.weights[0].name;let wide=step.outputs[0].elements().map_err(|e|e.to_string())?;
                    let value=b.rows_norm(name,norm_shape(c,name,wide)?,&values[&step.inputs[0]],m,self.parameter(name)?)?;values.insert(step.outputs[0].name.clone(),value);},
                Kind::CacheAppend=>{let value=b.join(&[&state["latent_cache"],&values[&step.inputs[1]]])?;values.insert(step.outputs[0].name.clone(),value);},
                Kind::KeyNorm=>{let mut s=shape(Op::KeyNorm);s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.epsilon=plan.settings.index_key_epsilon;
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(self.parameter(&w.name)?);}
                    let value=b.prompt(RowShape::wrapped(s,m).ok_or("prompt row count exceeds ABI")?,[Some(&values[&step.inputs[0]]),None,None,None,None],&weight)?;values.insert(step.outputs[0].name.clone(),value);},
                // As a prompt group's pool step (`layer_rows`), on the predictor's own caches: every completed earlier
                // pool is kept, and only the pools these rows complete are folded.
                Kind::Pool=>{let mut s=shape(Op::Pool);s.heads=number(a.indexer.heads)?;s.dim=number(a.indexer.dim)?;s.taps=number(a.indexer.pool)?;s.rows=number(last)?;s.epsilon=d.epsilon as f32;
                    let mut weight=Vec::new();for w in &step.weights {weight.extend_from_slice(self.parameter(&w.name)?);}
                    let (keys,gates)=(&values[&step.inputs[2]],&values[&step.inputs[3]]);
                    if first>0 && (m<=VERIFY_ROWS || first>=a.indexer.pool) {
                        let x=[&state["index_keys"],&state["index_gates"],keys,gates];
                        let result=if m<=VERIFY_ROWS {pools::append(b,s,at,m,x,&state["pool_keys"],&weight)?} else {pools::append_window(b,s,at,m,x,&state["pool_keys"],&weight)?};
                        for (spec,value) in step.outputs.iter().zip(result) {values.insert(spec.name.clone(),value);}continue;
                    }
                    let joined=|old:&C::Value,new:&C::Value|{let body=b.range(new,0,(m-1)*width)?;b.join(&[old,&body])};
                    let (all_keys,all_gates)=(joined(&state["index_keys"],keys)?,joined(&state["index_gates"],gates)?);
                    let (key,gate)=(b.range(keys,(m-1)*width,width)?,b.range(gates,(m-1)*width,width)?);
                    let result=b.operation(s,[Some(&all_keys),Some(&all_gates),Some(&key),Some(&gate)],&weight)?;let len=last as usize*width;
                    values.insert(step.outputs[0].name.clone(),b.range(&result,0,len)?);values.insert(step.outputs[1].name.clone(),b.range(&result,len,len)?);
                    values.insert(step.outputs[2].name.clone(),b.range(&result,2*len,(last as usize/pool)*width)?);},
                kind=>return Err(format!("{}: predictor seed step {kind:?} is unsupported",step.id)),
            }
        }
        let mut layers=BTreeMap::new();for (name,output) in NAMES {layers.insert(name.to_owned(),values.remove(output).ok_or("predictor seed made no cache")?);}
        Ok(PredictorState::seeded(last,layers))
    }
}
/// The steps of the predictor layer's plan that its caches depend on, in plan order. A prepare stands for itself and
/// the two steps it covers. The rest of the layer (attention, experts) and the projections only they read are left out.
fn seed_steps(plan:&LayerPlan)->Result<Vec<usize>> {
    let mut units=Vec::new();let mut index=0;
    while index<plan.steps.len() {let step=&plan.steps[index];
        if step.kind==Kind::MhcPrepare {let target=plan.steps.get(index+2).ok_or("predictor prepare is incomplete")?;units.push((index,step.inputs.clone(),target.outputs.iter().map(|o|o.name.clone()).collect::<Vec<_>>()));index+=3;continue;}
        units.push((index,step.inputs.clone(),step.outputs.iter().map(|o|o.name.clone()).collect()));index+=1;}
    let mut wanted=NAMES.iter().map(|(_,output)|(*output).to_owned()).collect::<std::collections::BTreeSet<_>>();let mut kept=Vec::new();
    let end=units.iter().rposition(|unit|unit.2.iter().any(|name|wanted.contains(name))).ok_or("predictor plan makes no cache")?;
    for (index,inputs,outputs) in units[..=end].iter().rev() {
        if outputs.iter().any(|name|wanted.contains(name)) {kept.push(*index);wanted.extend(inputs.iter().cloned());}
    }
    kept.reverse();
    for (name,output) in NAMES {need(plan.steps.iter().any(|s|s.outputs.iter().any(|o|o.name==output)),name)?;}
    Ok(kept)
}
