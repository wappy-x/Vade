use super::*;
use crate::projection_stage::{self,Job};
struct Collect(std::rc::Rc<std::cell::Cell<bool>>);
impl Drop for Collect {fn drop(&mut self) {self.0.set(false);}}
pub fn gather_stage<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&[DeviceVector<'d>],rows:&[Rows])->Result<Vec<DeviceVector<'d>>> {
    gather_stage_rounds(b,mailbox,sequence,topology,local,&vec![false;local.len()],rows)
}
/// `rounds[i]`: part i still needs its BF16 round, which the gather does inside its joined put.
pub fn gather_stage_rounds<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&[DeviceVector<'d>],rounds:&[bool],rows:&[Rows])->Result<Vec<DeviceVector<'d>>> {
    let words=projection_stage::checked_rows(rows)?;
    if local.is_empty() || local.len()!=rows.len() || local.len()!=rounds.len() || local.iter().zip(rows).any(|(value,row)|value.len()!=row.count) {return Err("projection stage local row extent differs".into());}
    if words.checked_mul(4).is_none_or(|bytes|bytes>mailbox.slot_bytes()) {return Err("projection stage payload exceeds mailbox slot".into());}
    gather_device(b,mailbox,sequence,topology,&local.iter().zip(rounds).map(|(value,round)|(value,*round)).collect::<Vec<_>>(),rows)
}
/// Keep complete local heads; exchange only outputs that have global consumers.
pub fn gather_stage_selected<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,output:Vec<DeviceVector<'d>>,rounds:Vec<bool>,rows:Vec<Rows>,owned:Vec<bool>)->Result<Vec<DeviceVector<'d>>> {
    if output.len()!=rounds.len() || output.len()!=rows.len() || output.iter().zip(&rows).any(|(v,r)|v.len()!=r.count) {return Err("selected stage extent differs".into());}
        if owned.len()!=output.len() {return Err("recurrent stage ownership count differs".into());}
        let mut kept=Vec::new();let mut sent=Vec::new();let mut send_rounds=Vec::new();let mut send_rows=Vec::new();
        for (((value,round),row),local) in output.into_iter().zip(rounds).zip(rows).zip(owned) {
            if local {kept.push(Some(if round {b.round_bf16_lanes(&value)?.into_vector()} else {value}));}
            else {kept.push(None);sent.push(value);send_rounds.push(round);send_rows.push(row);}
        }
        let gathered=if sent.is_empty() {Vec::new()} else {gather_stage_rounds(b,mailbox,sequence,topology,&sent,&send_rounds,&send_rows)?};
        let mut gathered=gathered.into_iter();let values=kept.into_iter().map(|value|value.or_else(||gathered.next()).ok_or("recurrent stage output is absent".into())).collect::<Result<Vec<_>>>()?;
    Ok(values)
}
struct Defer(std::rc::Rc<std::cell::Cell<bool>>);
impl Drop for Defer {fn drop(&mut self) {self.0.set(false);}}
impl<'b,'d> Weights<'b,'d> {
    pub fn prepare_kda(&mut self,c:&CheckpointAdmission)->Result<()> {
        let draft=c.description().with_draft_layer().map_err(|e|e.to_string())?;
        let description=draft.as_ref().unwrap_or(c.description());
        for (layer,spec) in description.layers.iter().enumerate() {
            if !matches!(spec.mixer,engine_model::plan::description::TokenMixer::Kda(_)) {continue;}
            let layout=crate::kda_heads::layout(spec.mixer,self.topology)?;
            let plan=LayerPlan::new(description,layer,1,Topology {rank:0,parts:1}).map_err(|e|e.to_string())?;
            for name in crate::kda_heads::projections(&plan,layout) {
                let Some(matrix)=self.matrices.get(&name) else {continue;};
                let rows=matrix.rows;let heads=layout.heads();
                if rows.total%layout.total_heads()!=0 {return Err("recurrent projection head extent differs".into());}
                let width=rows.total/layout.total_heads();
                if rows.start!=heads.start*width || rows.count!=layout.local_heads()*width {return Err("recurrent projection ownership differs".into());}
                self.kda_local.insert(name);
            }
        }Ok(())
    }
    pub fn sparse_topology(&self)->Topology {self.topology}
    pub fn apply_owned(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<DeviceVector<'d>> {
        self.owned_projection(name,b,input,tokens,layout,tokens==1)
    }
    pub fn apply_owned_rows(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<DeviceVector<'d>> {
        self.owned_projection(name,b,input,tokens,layout,false)
    }
    fn owned_projection(&mut self,name:&str,b:&Backend<'d>,input:&DeviceVector<'_>,tokens:usize,layout:crate::sparse_heads::Layout,scalar:bool)->Result<DeviceVector<'d>> {
        let matrix=self.matrices.get(name).ok_or("owned expansion matrix is absent")?;
        let owned=layout.projection_rows();
        if layout.topology()!=self.topology || matrix.rows.start!=owned.start || matrix.rows.count!=owned.len() || matrix.rows.total!=layout.global_row_words() {
            return Err("owned expansion matrix interval differs".into());
        }
        if self.collecting.get() {return Err("owned expansion entered a projection stage".into());}
        self.collecting.set(true);let guard=Collect(self.collecting.clone());
        let output=if scalar {self.apply(name,b,input,true)} else {self.apply_matrix(name,b,input,tokens,true)};
        drop(guard);output
    }
    pub fn gather_attended(&mut self,b:&Backend<'d>,local:&DeviceVector<'d>,tokens:usize,layout:crate::sparse_heads::Layout)->Result<DeviceVector<'d>> {
        use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
        let width=layout.local_heads()*layout.value_dim();let total=layout.total_heads()*layout.value_dim();
        if tokens==0 || layout.topology()!=self.topology || local.len()!=tokens.checked_mul(width).ok_or("attended extent overflow")? {
            return Err("attended gather extent differs".into());
        }
        if width==total {let mut output=b.vector(local.len())?;b.copy_prefix(local,&mut output,local.len())?;return Ok(output);}
        if let Some(mailbox)=self.mailbox.filter(|mailbox|local.len()<=mailbox.slot_bytes()/4) {
            let row=Rows {start:layout.attended_rows().start,count:width,total,split:true};
            let output=gather_device(b,mailbox,&mut self.sequence,self.topology,&[(local,false)],&vec![row;tokens])?;
            return b.resident_join(&output.iter().map(|value|(value,value.len())).collect::<Vec<_>>(),&mut None,0);
        }
        let transpose=|input:&DeviceVector<'_>,rows:usize,cols:usize,finite:u32|->Result<DeviceVector<'d>> {
            let mut shape=RowShape::new(Row::Transpose,1).ok_or("attended transpose shape")?;
            shape.rows=rows.try_into().map_err(|_|"attended rows exceed ABI")?;shape.cols=cols.try_into().map_err(|_|"attended columns exceed ABI")?;shape.first=finite;
            b.prompt_operation(shape,[Some(input),None,None,None,None],None)
        };
        let source=transpose(local,tokens,width,0)?;self.gather_mapped(b,&source)?;
        transpose(self.exchange.as_ref().ok_or("attended exchange is absent")?,total,tokens,1)
    }
    fn stage_rows(&self,jobs:&[Job<'_,DeviceVector<'d>>])->Result<Option<(Vec<Rows>,Vec<usize>)>> {
        if self.collecting.get() {return Err("nested projection stage".into());}
        let Some(mailbox)=self.mailbox else {return Ok(None);};
        let mut rows=Vec::new();let mut counts=Vec::new();
        for job in jobs {
            match job {
                Job::Project {input,..}=>{
                    if input.is_empty() || input.len()>9 {return Ok(None);}
                    for (name,value) in input {
                        let Some(matrix)=self.matrices.get(*name) else {return Ok(None);};
                        if value.len()!=matrix.cols || value.is_segmented() || input.len()>1 && !matches!(matrix.storage,Storage::Mx(_)|Storage::Plain {..}|Storage::Packed {shape:Shape {op:Op::NvProject|Op::Fp8Serial,..},..}) {return Ok(None);}
                        rows.push(matrix.rows);counts.push(1);
                    }
                },
                Job::Routed {steps,ids,input}=>{
                    let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(None);};
                    if !(1..=2).contains(&steps.len()) || !self.routes.get(&first.name).is_some_and(Option::is_some) {return Ok(None);}
                    let Some(matrix)=self.matrices.get(&first.name) else {return Ok(None);};
                    let width=if steps[0].kind==Kind::ExpertDown {matrix.cols.checked_mul(ids.len())} else {Some(matrix.cols)};
                    if ids.len()!=8 || width!=Some(input.len()) || ids.is_segmented() || input.is_segmented() {return Ok(None);}
                    rows.extend(std::iter::repeat_n(matrix.rows,steps.len()*ids.len()));counts.extend(std::iter::repeat_n(ids.len(),steps.len()));
                },
                Job::Gated {name,gate,up,shape}=>{
                    let Some(matrix)=self.matrices.get(*name) else {return Ok(None);};
                    if shape.op!=spark_gpu_probe::causal_ops::Op::SwiGlu || gate.len()!=matrix.cols || up.len()!=matrix.cols || shape.rows as usize!=matrix.cols
                        || gate.is_segmented() || up.is_segmented() {return Ok(None);}
                    rows.push(matrix.rows);counts.push(1);
                },
            }
        }
        let Ok(words)=projection_stage::checked_rows(&rows) else {return Ok(None);};
        if projection_stage::checked_messages(counts.len(),words,mailbox.slot_bytes()).is_err() {return Ok(None);}
        Ok(Some((rows,counts)))
    }
    /// One offered stage. None: the stage is refused and its steps run one at a time; with a device exchange attached
    /// that is counted (projection_stage::Counts::refused), so a run can prove no stage was refused.
    pub fn apply_stage(&mut self,b:&Backend<'d>,jobs:&[Job<'_,DeviceVector<'d>>])->Result<Option<Vec<Vec<DeviceVector<'d>>>>> {
        let result=self.offered_stage(b,jobs)?;
        if result.is_none() && self.mailbox.is_some() {projection_stage::refused();}
        Ok(result)
    }
    fn offered_stage(&mut self,b:&Backend<'d>,jobs:&[Job<'_,DeviceVector<'d>>])->Result<Option<Vec<Vec<DeviceVector<'d>>>>> {
        // A routed job's table is made here at the layer's first step, as apply_routed would make it. Without it the first
        // decode step refused the expert gate/up stage, so the shared expert's down ran after the routed down and LOCAL
        // FOLD found no shared row (one unfolded down a routed layer, once a process). Now step one forms every later step's stages.
        if self.mailbox.is_some() && !self.collecting.get() {
            for job in jobs {
                if let Job::Routed {steps,ids,..}=job && super::route_device::routed_shape(steps,ids.len()) {self.route_group(b,steps,ids.len())?;}
            }
        }
        let Some((rows,counts))=self.stage_rows(jobs)? else {return Ok(None);};
        self.collecting.set(true);let guard=Collect(self.collecting.clone());
        // Each job leaves one flag an output: true when a split output still owes its BF16 round, which the stage gather does.
        // A whole (not split) output is rounded at once, so a later job of the stage can read it.
        self.rounds.clear();self.deferring.set(true);let defer=Defer(self.deferring.clone());
        let mut output=Vec::new();
        for job in jobs {
            match job {
                Job::Project {input,round}=>{
                    if input.len()==1 {output.push(self.apply(input[0].0,b,input[0].1,*round)?);}
                    else {let Some(values)=self.apply_batch(b,input,*round)? else {return Ok(None);};output.extend(values);}
                },
                Job::Routed {steps,ids,input}=>{
                    let Some(values)=self.apply_routed(b,steps,ids,input)? else {return Ok(None);};output.extend(values);
                },
                Job::Gated {name,gate,up,shape}=>{
                    let value=b.causal_operation(*shape,[Some(gate),Some(up),None,None],None)?;
                    output.push(self.apply(name,b,&value,true)?);
                },
            }
        }
        drop(guard);drop(defer);let rounds=std::mem::take(&mut self.rounds);
        if output.len()!=rows.len() || output.len()!=rounds.len() || output.iter().zip(&rows).any(|(value,row)|value.len()!=row.count) || rounds.iter().zip(&rows).any(|(round,row)|*round && !row.split) {return Err("projection stage local row extent differs".into());}
        let owned=jobs.iter().flat_map(|job|match job {
            Job::Project {input,..}=>input.iter().map(|(name,_)|self.kda_local.contains(*name)).collect::<Vec<_>>(),
            Job::Routed {steps,ids,..}=>vec![false;steps.len()*ids.len()],Job::Gated {..}=>vec![false],
        }).collect::<Vec<_>>();
        let words=output.iter().zip(&owned).filter(|(_,local)|!**local).map(|(v,_)|v.len()).sum();
        let values=gather_stage_selected(b,self.mailbox.ok_or("projection stage mailbox is absent")?,&mut self.sequence,self.topology,output,rounds,rows,owned)?;
        projection_stage::accepted(jobs.len(),words);
        let mut values=values.into_iter();let result=counts.into_iter().map(|count|values.by_ref().take(count).collect::<Vec<_>>()).collect::<Vec<_>>();
        if values.next().is_some() || result.iter().any(Vec::is_empty) {return Err("projection stage result extent differs".into());}Ok(Some(result))
    }
}
