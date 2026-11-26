//! A proposal step batches the embedding, projections and norms of the active chats at one depth.
//! The layer uses their independent histories. The head returns each chat's pick and probability
//! before that chat can start its next word.
use super::*;
use super::proposer::PredictorState;
/// One chat's chain for `propose_chains`: what `propose_after_mass` takes.
pub struct ChainStart<'a,V> {
    /// The target's last hidden state of the row before `token`.
    pub hidden:&'a V,
    pub token:u32,
    /// At least 1: a chain of depth 0 (the cache only) is not a proposal.
    pub depth:usize,
    /// The chat's predictor cache so far; None starts at position zero.
    pub seen:Option<&'a PredictorState<V>>,
    /// The chain ends once the product of its probabilities falls under this (CHAIN_STOP or CHAIN_STOP_NAMED).
    pub stop:Option<f32>,
}
/// The first joined head pass of this process has had its picks set beside the picks of each row alone.
static ONE_PASS_CHECKED:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
/// A chat's chain while it runs.
struct Run<V> {hidden:V,token:u32,depth:usize,first:u64,stop:Option<f32>,empty:BTreeMap<String,V>,after:Option<PredictorState<V>>,
    chain:Option<BTreeMap<String,V>>,proposals:Vec<u32>,mass:Vec<Option<f32>>,chance:f32,live:bool}
struct Word<'a,V> {token:u32,hidden:&'a V,context:u64,prior:&'a BTreeMap<String,V>}
impl<'f,C:Compute> Decoder<'f,C> {
    /// Result `i`: chat `i`'s proposals, their probabilities and its cache after the first step: those of
    /// `propose_after_mass(chains[i].hidden, chains[i].token, chains[i].depth, chains[i].seen, chains[i].stop)`.
    #[allow(clippy::type_complexity)] // each chat's proposals, their probabilities and its cache.
    pub fn propose_chains(&self,chains:&[ChainStart<'_,C::Value>])->Result<Vec<(Vec<u32>,Vec<Option<f32>>,PredictorState<C::Value>)>> {
        let draft=self.draft.as_ref().ok_or("predictor layer is not enabled")?;let c=self.checkpoint;let d=c.description();let b=&self.compute;
        let h=d.hidden as usize;let layer=d.layers.len();let prefix=format!("model.language_model.layers.{layer}");
        let norm=|part:&str,x:&C::Value|->Result<C::Value> {let name=format!("{prefix}.{part}.weight");
            b.norm(&name,norm_shape(c,&name,h)?,x,draft.parameters.get(&name).ok_or("predictor norm is absent")?)};
        let mut runs=Vec::with_capacity(chains.len());
        for chain in chains {
            need((1..=spark_engine::draft::PREDICTOR_DEPTH_MAX).contains(&chain.depth) && u64::from(chain.token)<d.vocab,"proposal depth or token differs")?;
            // The empty cache is read only by the first step of a chain that has no earlier position.
            let mut empty=BTreeMap::new();
            if chain.seen.is_none() {for (name,elements) in &draft.empty {empty.insert(name.clone(),b.zeros(self.state_words(layer,*elements)?)?);}}
            let first=chain.seen.map_or(0,PredictorState::rows);need(first+chain.depth as u64<=self.capacity,"predictor cache exceeds capacity")?;
            runs.push(Run {hidden:b.range(chain.hidden,0,h)?,token:chain.token,depth:chain.depth,first,stop:chain.stop,empty,after:None,chain:None,
                proposals:Vec::new(),mass:Vec::new(),chance:1.0,live:true});
        }
        let most=runs.iter().map(|run|run.depth).max().unwrap_or(0);
        for step in 0..most {
            let mut active=Vec::new();
            let mut words=Vec::new();
            for (at,(run,chain)) in runs.iter().zip(chains).enumerate().filter(|(_,(run,_))|run.live && step<run.depth) {
                let prior=match (step,&run.after,&run.chain) {
                    (0,_,_)=>chain.seen.map_or(&run.empty,PredictorState::layers),
                    (1,Some(after),_)=>after.layers(),
                    (_,_,Some(chain))=>chain,
                    _=>return Err("proposal chain lost its cache".into()),
                };
                active.push(at);
                words.push(Word {token:run.token,hidden:&run.hidden,context:run.first+step as u64+1,prior});
            }
            if active.is_empty() {break;}
            let (headed,states)=if let [word]=words.as_slice() {
                let embedded=b.range(&self.embedding(word.token)?,0,h)?;
                let input=b.join(&[&norm("enorm",&embedded)?,&norm("hnorm",word.hidden)?])?;
                let x=self.project(&format!("{prefix}.eh_proj.weight"),&input,true)?;
                let (mut outputs,next)=self.layer_steps(layer,word.context,&[&x],word.prior,None,false,false,None)?;
                (norm("shared_head.norm",&outputs.remove(0))?,vec![next])
            } else {self.propose_batch_word(&words)?};
            let n=active.len();
            let rows=(0..n).map(|i|b.range(&headed,i*h,h)).collect::<Result<Vec<_>>>()?;
            let picked=if n>1 {b.greedy_projection_mass_rows(c,"lm_head.weight",&headed,n)?} else {None};
            let picked=match picked {Some(picked)=>{need(picked.len()==n,"proposal rows pick count differs")?;
                    // Once a process (every rank alike, so the exchanges pair up): the picks of each row alone beside
                    // the joined ones, in the log.
                    if !ONE_PASS_CHECKED.swap(true,std::sync::atomic::Ordering::Relaxed) {
                        let alone=rows.iter().map(|row|b.greedy_projection_mass(c,"lm_head.weight",row)).collect::<Result<Vec<_>>>()?;
                        let ids=alone.iter().zip(&picked).filter(|(a,j)|a.0==j.0).count();let bits=alone.iter().zip(&picked).filter(|(a,j)|a.0==j.0 && a.1.map(f32::to_bits)==j.1.map(f32::to_bits)).count();
                        eprintln!("SERVED ONE PASS CHECK rows={n} same_ids={ids} same_bits={bits}");
                    }
                    picked},
                None=>rows.iter().map(|row|b.greedy_projection_mass(c,"lm_head.weight",row)).collect::<Result<Vec<_>>>()?};
            for (((at,headed),next),(best,probability)) in active.into_iter().zip(rows).zip(states).zip(picked) {
                need(u64::from(best)<d.vocab,"proposal ID exceeds vocabulary")?;let run=&mut runs[at];
                if step==0 {run.after=Some(PredictorState::seeded(run.first+1,next));} else {run.chain=Some(next);}
                run.mass.push(probability);run.token=best;run.proposals.push(best);
                if let (Some(under),Some(p))=(run.stop,probability) {run.chance*=p;if run.chance<under {run.live=false;}}
                run.hidden=headed;
            }
        }
        runs.into_iter().map(|run|Ok((run.proposals,run.mass,run.after.ok_or("proposal made no step")?))).collect()
    }

    fn propose_batch_word(&self,words:&[Word<'_,C::Value>])->Result<(C::Value,Vec<BTreeMap<String,C::Value>>)> {
        let c=self.checkpoint;let d=c.description();let b=&self.compute;
        let h=d.hidden as usize;let n=words.len();
        let prefix=format!("model.language_model.layers.{}",d.layers.len());
        let norm=|part:&str,x:&C::Value|->Result<C::Value> {
            let name=format!("{prefix}.{part}.weight");
            b.rows_norm(&name,norm_shape(c,&name,h)?,x,n,self.parameter(&name)?)
        };
        let tokens=words.iter().map(|word|word.token).collect::<Vec<_>>();
        let embedded=match b.draft_embeddings(&tokens)? {
            Some(rows)=>rows,
            None=>{
                let each=tokens.iter().map(|&token|b.range(&self.embedding(token)?,0,h)).collect::<Result<Vec<_>>>()?;
                b.join(&each.iter().collect::<Vec<_>>())?
            }
        };
        let hidden=b.join(&words.iter().map(|word|word.hidden).collect::<Vec<_>>())?;
        let (embedded,hidden)=(norm("enorm",&embedded)?,norm("hnorm",&hidden)?);
        let mut parts=Vec::with_capacity(2*n);
        for row in 0..n {
            parts.push(b.range(&embedded,row*h,h)?);
            parts.push(b.range(&hidden,row*h,h)?);
        }
        let fused=b.join(&parts.iter().collect::<Vec<_>>())?;
        let x=self.project_group(&format!("{prefix}.eh_proj.weight"),&fused,n,true)?;
        let contexts=words.iter().map(|word|word.context).collect::<Vec<_>>();
        let states=words.iter().map(|word|word.prior).collect::<Vec<_>>();
        let (output,next)=self.draft_layer_batch(&contexts,x,&states)?;
        Ok((norm("shared_head.norm",&output)?,next))
    }

    /// The appended layer's one-stream residual uses the same norm and BF16 add as its scalar plan.
    pub(super) fn draft_batch_residual(&self,index:usize,plan:&LayerPlan,m:usize,values:&mut BTreeMap<String,C::Value>)->Result<usize> {
        let step=&plan.steps[index];
        if step.kind==Kind::Combine {
            let sum=self.bf16_sum(&values[&step.inputs[0]],&values[&step.inputs[2]])?;
            values.insert(step.outputs[0].name.clone(),sum);
            return Ok(0);
        }
        let target=&plan.steps[index+2];
        let h=self.checkpoint.description().hidden as usize;
        let name=&target.weights[0].name;
        let norm=self.compute.rows_norm(name,norm_shape(self.checkpoint,name,h)?,&values[&step.inputs[0]],m,self.parameter(name)?)?;
        values.insert(target.outputs[0].name.clone(),norm);
        Ok(2)
    }
}
