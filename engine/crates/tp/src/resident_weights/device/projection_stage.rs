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
    /// Several `apply_matrix` calls of one row group as a stage: the split outputs share exchanges.
    /// Two forms, each with the words of the single calls:
    /// - a small group on the device mailbox: each token of each projection is one part of a stage
    ///   gather (the plain step's), so a projection's rows come back token-major with no copy after;
    /// - any other group on the host link: the local blocks lie in order in the exchange buffer, ONE
    ///   link step moves them, one copy puts each projection's rank blocks side by side, and the
    ///   transpose that makes them token-major refuses a word that is not finite, as the single
    ///   call's does.
    ///
    /// `widest`: the most words a token may have in one host-link exchange, over all ranks (one
    /// residual row: what admission charges the exchange buffer for).
    ///
    /// None, before any work: the caller makes the calls one by one (no exchange owner, or no two
    /// split outputs that fit one gather).
    pub fn apply_matrix_stage(&mut self,b:&Backend<'d>,jobs:&[(&str,&DeviceVector<'_>,bool)],m:usize,widest:usize)->Result<Option<Vec<DeviceVector<'d>>>> {
        /// Most parts of one mailbox stage gather (the bound its device test proves).
        const PARTS:usize=32;
        if self.collecting.get() || jobs.len()<2 || m==0 {return Ok(None);}
        let mailbox=self.mailbox.filter(|_|self.small.get() && m<=PARTS);
        if mailbox.is_none() && self.ring.is_none() {return Ok(None);}
        let slot=mailbox.map_or(0,|mailbox|mailbox.slot_bytes()/4);
        // The gathers: runs of split jobs, each with its local words and its rows over all ranks.
        let mut gathers:Vec<(Vec<usize>,usize,usize)>=Vec::new();let mut split=0;
        for (job,(name,input,_)) in jobs.iter().enumerate() {
            let matrix=self.matrices.get(*name).ok_or_else(||format!("projection is not resident: {name}"))?;
            if input.len()!=m.checked_mul(matrix.cols).ok_or("prompt input overflow")? || input.is_segmented() {return Err("resident prompt feature extent differs".into());}
            // A matrix of the KDA head split is not exchanged here (its rows are this rank's heads).
            if !matrix.rows.split || self.kda_local.contains(*name) {continue;}
            let words=matrix.rows.count.checked_mul(m).ok_or("row stage extent overflow")?;let total=matrix.rows.total;
            if if mailbox.is_some() {words>slot} else {total>widest} {return Ok(None);}
            split+=1;
            match gathers.last_mut().filter(|held|if mailbox.is_some() {(held.0.len()+1)*m<=PARTS && held.1+words<=slot} else {held.2+total<=widest}) {
                Some(held)=>{held.0.push(job);held.1+=words;held.2+=total;},
                None=>gathers.push((vec![job],words,total)),
            }
        }
        // Small plain matrices of a group of few rows run as the plain step runs them: in its merged
        // launch, one job for each (matrix, row). A matrix this small is read again for each row at
        // little cost, and the sum chains of all jobs then run side by side. The row kernel makes
        // each word as one serial chain of `cols` terms, one launch a matrix, one after the other
        // (measured on the real model: 13 ms a step for 267 such calls in a one-row batch against
        // 2.6 ms in the plain step).
        const FEW:usize=8;const SMALL:usize=8<<20;
        let small=|round:bool|jobs.iter().enumerate().filter(|(_,(name,_,flag))|m<=FEW && *flag==round && self.matrices[*name].cols.is_multiple_of(32)
            && matches!(&self.matrices[*name].storage,Storage::Plain {weight,..} if weight.len()*4<=SMALL)).map(|(job,_)|job).collect::<Vec<_>>();
        let merged=[small(false),small(true)];
        // Nothing to share and nothing to merge: the single calls are the same work.
        let fp8=|job:usize|matches!(&self.matrices[jobs[job].0].storage,Storage::Packed {shape,..} if shape.op!=Op::NvProject);
        let pairs=(0..jobs.len()).any(|a|(a+1..jobs.len()).any(|b|fp8(a) && fp8(b)
            && jobs[a].1.device_address()==jobs[b].1.device_address() && jobs[a].1.len()==jobs[b].1.len() && self.matrices[jobs[a].0].cols==self.matrices[jobs[b].0].cols));
        if (split<2 || gathers.len()==split) && merged.iter().all(Vec::is_empty) && !pairs {return Ok(None);}
        // Proof for logs: this stage shares at least one exchange (PROMPT KERNEL COUNTS, SERVED PROMPT COUNTS).
        if split>=2 && gathers.len()!=split {spark_gpu_probe::prompt_launches::record("row_stage");}
        // What a merged launch made for a job: its local rows, and whether they are [local rows,m].
        let mut made:Vec<Option<(DeviceVector<'d>,bool)>>=(0..jobs.len()).map(|_|None).collect();
        for (round,list) in [false,true].into_iter().zip(&merged) {
            for launch in list.chunks((9/m).max(1)) {
                let stamp=self.cost.start(b);
                let views=launch.iter().map(|&job| {let cols=self.matrices[jobs[job].0].cols;
                    (0..m).map(|token|jobs[job].1.resident_range(token*cols..(token+1)*cols)).collect::<std::result::Result<Vec<_>,_>>()}).collect::<std::result::Result<Vec<_>,_>>()?;
                let mut batch=Vec::with_capacity(launch.len()*m);let (mut words,mut bytes)=(0usize,0usize);
                for (&job,rows) in launch.iter().zip(&views) {
                    let matrix=&self.matrices[jobs[job].0];let Storage::Plain {weight,..}=&matrix.storage else {return Err("row stage plain storage differs".into());};
                    for input in rows {batch.push(spark_gpu_probe::driver::ResidentProjection::Plain {rows:matrix.rows.count,cols:matrix.cols,weight,input});}
                    words+=matrix.rows.count*m;bytes+=weight.len()*4;
                }
                let mut output=b.vector(words)?;b.resident_projection_batch(&batch,&mut output,round)?;drop(batch);
                // A matrix's rows are consecutive jobs: [m,local rows] is one range of the launch's output.
                let output=output.into_resident_shared()?;let mut start=0;
                for &job in launch {let extent=self.matrices[jobs[job].0].rows.count*m;made[job]=Some((output.resident_range(start..start+extent)?,false));start+=extent;}
                self.cost.record(||"projection_mixed".into(),stamp,b,bytes);
            }
        }
        // Block-FP8 projections that read the same rows share ONE input round (two launches less for
        // each further matrix). Each still takes its own launch: the row kernel choice is tuned for
        // one matrix a launch.
        let mut rounded:Vec<(usize,DeviceVector<'d>)>=Vec::new();
        for job in 0..jobs.len() {
            if !fp8(job) {continue;}
            let same=|other:usize|fp8(other) && jobs[other].1.device_address()==jobs[job].1.device_address() && jobs[other].1.len()==jobs[job].1.len();
            let Some(first)=(0..jobs.len()).find(|&other|same(other)) else {continue;};
            if (0..jobs.len()).filter(|&other|same(other)).count()<2 {continue;}
            let stamp=self.cost.start(b);
            if self.lut.is_none() {self.lut=Some(b.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?);}
            if !rounded.iter().any(|(owner,_)|*owner==first) {
                let lanes=b.round_bf16_lanes(jobs[job].1)?;let mut source=b.vector(lanes.len())?;b.copy_prefix(lanes.vector(),&mut source,lanes.len())?;rounded.push((first,source));
            }
            let source=&rounded.iter().find(|(owner,_)|*owner==first).ok_or("row stage lost a rounded input")?.1;
            let matrix=&self.matrices[jobs[job].0];let Storage::Packed {shape,weight,scales}=&matrix.storage else {return Err("row stage FP8 storage differs".into());};
            let output=b.expert_group(&[(Shape {op:Op::Fp8Rows,tokens:m as u32,..*shape},Some((weight,scales)),0)],source,matrix.cols,self.lut.as_ref().ok_or("expert decode table is absent")?)?;
            made[job]=Some((output,true));
            self.cost.settle(b)?;self.cost.record(||"projection_3".into(),stamp,b,(weight.len()+scales.len())*4);
        }
        // Each job's local rows. A whole (not split) output is done. A split one waits for its
        // gather: token-major for the mailbox, row-major for the host link.
        let mut outputs:Vec<Option<DeviceVector<'d>>>=Vec::with_capacity(jobs.len());let mut local:Vec<Option<DeviceVector<'d>>>=Vec::with_capacity(jobs.len());
        for (job,(name,input,round)) in jobs.iter().enumerate() {
            let (output,row_major)=match made[job].take() {Some(local)=>local,None=>self.matrix_rows(name,b,input,m,*round)?};let rows=self.matrices[*name].rows;
            if !rows.split || self.kda_local.contains(*name) {local.push(None);outputs.push(Some(if row_major && m>1 {Self::transpose(b,&output,rows.count,m,false)?} else {output}));continue;}
            let block=if m==1 || row_major!=mailbox.is_some() {output} else if row_major {Self::transpose(b,&output,rows.count,m,false)?} else {Self::transpose(b,&output,m,rows.count,false)?};
            local.push(Some(block.into_resident_shared()?));outputs.push(None);
        }
        let stamp=self.cost.start(b);
        for (held,words,_) in &gathers {
            if let Some(mailbox)=mailbox {
                let mut parts=Vec::with_capacity(held.len()*m);let mut rows=Vec::with_capacity(held.len()*m);
                for &job in held {
                    let matrix_rows=self.matrices[jobs[job].0].rows;let whole=local[job].as_ref().ok_or("row stage lost a local part")?;
                    for token in 0..m {parts.push(whole.resident_range(token*matrix_rows.count..(token+1)*matrix_rows.count)?);rows.push(matrix_rows);}
                }
                projection_stage::checked_rows(&rows)?;
                let joined=gather_joined(b,mailbox,&mut self.sequence,self.topology,&parts.iter().map(|part|(part,false)).collect::<Vec<_>>(),&rows)?;
                // The tokens of one projection are consecutive parts: [m,total rows] is one range of the gather.
                let mut start=0;
                for &job in held {let total=self.matrices[jobs[job].0].rows.total*m;outputs[job]=Some(joined.resident_range(start..start+total)?);start+=total;}
            } else {
                let blocks=held.iter().map(|&job|local[job].as_ref().ok_or("row stage lost a local part")).collect::<std::result::Result<Vec<_>,_>>()?;
                self.gather_mapped_parts(b,&blocks)?;
                // Each projection's rank blocks side by side, [total rows,m], out of the exchange buffer
                // (the next gather writes it again).
                let ranks=self.topology.parts as usize;let mut placed=Vec::with_capacity(held.len()*ranks);let (mut from,mut to)=(0usize,0usize);
                for &job in held {
                    // A projection's gathered rows are one block of `block` words a part of the exchange.
                    let rows=self.matrices[jobs[job].0].rows;let block=rows.count*m;let whole=block*ranks;
                    if rows.total*m!=whole {return Err("row stage rank blocks differ".into());}
                    for rank in 0..ranks {placed.push([rank**words+from,to+rank*block,block]);}
                    from+=block;to+=rows.total*m;
                }
                let mut joined=b.vector(to)?;
                b.resident_scatter_copy(self.exchange.as_ref().ok_or("prompt exchange is absent")?,&mut joined,&placed)?;
                let joined=joined.into_resident_shared()?;let mut start=0;
                for &job in held {
                    let rows=self.matrices[jobs[job].0].rows;let extent=rows.total*m;
                    outputs[job]=Some(Self::transpose(b,&joined.resident_range(start..start+extent)?,rows.total,m,true)?);start+=extent;
                }
            }
        }
        self.cost.record(||"exchange_stage_rows".into(),stamp,b,0);
        outputs.into_iter().map(|output|output.ok_or_else(||"row stage lost an output".into())).collect::<Result<Vec<_>>>().map(Some)
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
