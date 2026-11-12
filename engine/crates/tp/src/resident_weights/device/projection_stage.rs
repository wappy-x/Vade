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
        self.gather_mapped(b,local)?;
        Self::rank_rows(b,self.exchange.as_ref().ok_or("attended exchange is absent")?,tokens,width,self.topology.parts as usize,true)
    }
    /// Several `apply_matrix` calls of one row group as a stage: the split outputs share exchanges.
    /// Two forms, each with the words of the single calls:
    /// - a small group on the device mailbox: each token of each projection is one part of a stage
    ///   gather (the plain step's), so a projection's rows come back token-major with no copy after;
    /// - any other group on the host link: the local blocks lie in order in the exchange buffer, ONE
    ///   link step moves them, one copy puts each projection's rank blocks side by side, and a
    ///   transpose or row placement makes them token-major. Both refuse a word that is not finite.
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
        //
        // DENSE ROWS: a matrix with few rows and long rows that holds the packed BF16 copy takes the plain step's folded
        // batch (`flash_resident_projection_plain_bf16`: producer warps make the rounded products, one lane a weight row
        // adds them in column order). The FP32 job runs products and adds of a row in ONE thread, so its 4,096-term chain
        // set the launch time (the router gate, measured 2026-11-06: 43.7 us as one chain a thread, 16.6 us folded). The
        // words are the same: the same products in the same ascending chain (`plain_warp::rows_fold`), and the widened BF16
        // pattern is bit for bit the FP32 copy's weight. Chosen by shape and storage alone; the shape rule is the FP32
        // entry's own (`plain_geometry`). Other small matrices keep the FP32 jobs.
        //
        // DENSE ROWS 2: with 4, 6 or 8 rows a matrix that holds the packed copy takes the row entry of the ROW STAGE
        // (`flash_resident_projection_bf16_rows`, plain jobs only here): one job reads 3 or 4 source rows, so ONE launch
        // makes a whole stage where the one-row jobs needed two or three (9 jobs a launch: 2 matrices at 4 rows, 1 at
        // 5 to 8). A launch of serial chains costs its longest chain whatever it holds, so fewer launches is the gain.
        // Each word is the one-row body (`plain_warp::row_deep`), one thread a (weight row, source row).
        const FEW:usize=8;const SMALL:usize=8<<20;
        #[derive(Clone,Copy,PartialEq)] enum Form {Fp32,Fold,Rows}
        // Source rows a job of the row entry reads: the rows split in equal parts of 3 or 4 (the entry's contract is 2 to 4).
        let part=match m {4=>4,6=>3,8=>4,_=>0};
        let form=|name:&str,input:&DeviceVector<'_>| {let matrix=&self.matrices[name];
            let Storage::Plain {bf16:Some(copy),..}=&matrix.storage else {return Form::Fp32;};
            // The ordered reader's envelope: whole 16-byte vectors of the packed copy and of every row view of the job.
            let admitted=|step:usize|!copy.is_segmented() && (0..m/step).all(|view|spark_gpu_probe::plain_admission::deep_admitted(matrix.rows.count,matrix.cols,
                copy.device_address() as usize,input.device_address() as usize+view*step*matrix.cols*4));
            if part!=0 && admitted(part) {Form::Rows}
            else if spark_gpu_probe::plain_admission::plain_geometry(matrix.rows.count,matrix.cols).1 as usize==spark_gpu_probe::plain_admission::FOLD_THREADS && admitted(1) {Form::Fold}
            else {Form::Fp32}};
        let small=|round:bool,want:Form|jobs.iter().enumerate().filter(|(_,(name,input,flag))|m<=FEW && *flag==round && self.matrices[*name].cols.is_multiple_of(32)
            && matches!(&self.matrices[*name].storage,Storage::Plain {weight,..} if weight.len()*4<=SMALL) && form(name,input)==want).map(|(job,_)|job).collect::<Vec<_>>();
        let merged=[false,true].into_iter().flat_map(|round|[Form::Fp32,Form::Fold,Form::Rows].into_iter().map(move |form|(round,form))).map(|(round,form)|(round,form,small(round,form))).collect::<Vec<_>>();
        // ROW STAGE: MXFP8 projections that read the same rows take one merged row launch (`row_stage.rs`).
        let mx=self.mx_row_groups(jobs,m);
        // QUANTIZE SHARE: past the row stage, MXFP8 projections that read the same rows share their quantize (`row_stage.rs`).
        let shared=self.mx_share_groups(jobs,m);
        // Nothing to share and nothing to merge: the single calls are the same work.
        let fp8=|job:usize|matches!(&self.matrices[jobs[job].0].storage,Storage::Packed {shape,..} if shape.op!=Op::NvProject);
        let pairs=(0..jobs.len()).any(|a|(a+1..jobs.len()).any(|b|fp8(a) && fp8(b)
            && jobs[a].1.device_address()==jobs[b].1.device_address() && jobs[a].1.len()==jobs[b].1.len() && self.matrices[jobs[a].0].cols==self.matrices[jobs[b].0].cols));
        if (split<2 || gathers.len()==split) && merged.iter().all(|(_,_,list)|list.is_empty()) && !pairs && mx.is_empty() && shared.is_empty() {return Ok(None);}
        // Proof for logs: this stage shares at least one exchange (PROMPT KERNEL COUNTS, SERVED PROMPT COUNTS).
        if split>=2 && gathers.len()!=split {spark_gpu_probe::prompt_launches::record("row_stage");}
        // What a merged launch made for a job: its local rows, and whether they are [local rows,m].
        let mut made:Vec<Option<(DeviceVector<'d>,bool)>>=(0..jobs.len()).map(|_|None).collect();
        for (round,form,list) in &merged {
            // A form with no job sends nothing. (The row entry's part is 0 outside 4, 6 and 8 rows: its list is empty there.)
            if list.is_empty() {continue;}
            // Source rows a job reads, and matrices a launch (9 jobs at most).
            let step=if *form==Form::Rows {part} else {1};let views_a_job=m/step;
            for launch in list.chunks((9/views_a_job).max(1)) {
                let stamp=self.cost.start(b);
                let views=launch.iter().map(|&job| {let cols=self.matrices[jobs[job].0].cols;
                    (0..views_a_job).map(|view|jobs[job].1.resident_range(view*step*cols..(view+1)*step*cols)).collect::<std::result::Result<Vec<_>,_>>()}).collect::<std::result::Result<Vec<_>,_>>()?;
                let mut batch=Vec::with_capacity(launch.len()*views_a_job);let (mut words,mut bytes)=(0usize,0usize);
                for (&job,rows) in launch.iter().zip(&views) {
                    let matrix=&self.matrices[jobs[job].0];let Storage::Plain {weight,bf16,..}=&matrix.storage else {return Err("row stage plain storage differs".into());};
                    let (rows_local,cols)=(matrix.rows.count,matrix.cols);
                    match (form,bf16) {
                        (Form::Fp32,_)=>{for input in rows {batch.push(spark_gpu_probe::driver::ResidentProjection::Plain {rows:rows_local,cols,weight,input});}bytes+=weight.len()*4;},
                        (_,Some(copy))=>{for input in rows {batch.push(spark_gpu_probe::driver::ResidentProjection::PlainBf16 {rows:rows_local,cols,weight:copy,input});}bytes+=copy.len()*4;},
                        (_,None)=>return Err("row stage packed copy is absent".into()),
                    }
                    words+=rows_local*m;
                }
                let mut output=b.vector(words)?;
                if *form==Form::Rows {b.resident_projection_batch_rows(&batch,&mut output,*round,step)?;} else {b.resident_projection_batch(&batch,&mut output,*round)?;}
                drop(batch);
                // A matrix's rows are consecutive jobs, source rows in order: [m,local rows] is one range of the launch's output.
                let output=output.into_resident_shared()?;let mut start=0;
                for &job in launch {let extent=self.matrices[jobs[job].0].rows.count*m;made[job]=Some((output.resident_range(start..start+extent)?,false));start+=extent;}
                self.cost.record(||String::from(match form {Form::Fp32=>"projection_mixed",Form::Fold=>"projection_fold",Form::Rows=>"projection_plain_rows"}),stamp,b,bytes);
            }
        }
        // Block-FP8 projections that read the same rows share ONE input round (two launches less for
        // each further matrix).
        //
        // SHARED ROWS: those of ONE shape and block shape (the shared expert's gate and up) also share ONE launch:
        // `fp8_rows_multi` with a part a matrix, every part on the same rows (`fp8_small::shared_admitted`, 2 to 4
        // matrices). A launch of that kernel costs what its longest warp runs, so two matrices in one launch cost
        // less than two launches (gate and up 1,024 x 4,096, one Spark: 53 us against 2 x 39 at 4 rows, 61 against
        // 2 x 46 at 8 rows), and a launch is saved. Each word is the word of the matrix's own launch: its own
        // ascending chain; only the block that computes it differs. Chosen by shape, storage and addresses alone;
        // a matrix with no partner, or a table the predicate refuses, keeps its own launch.
        let mut rounded:Vec<(usize,DeviceVector<'d>)>=Vec::new();
        for job in 0..jobs.len() {
            if !fp8(job) || made[job].is_some() {continue;}
            let same=|other:usize|fp8(other) && jobs[other].1.device_address()==jobs[job].1.device_address() && jobs[other].1.len()==jobs[job].1.len();
            let Some(first)=(0..jobs.len()).find(|&other|same(other)) else {continue;};
            if (0..jobs.len()).filter(|&other|same(other)).count()<2 {continue;}
            let stamp=self.cost.start(b);
            if self.lut.is_none() {self.lut=Some(b.upload_vector(&spark_gpu_probe::expert_ops::decode_lut())?);}
            if !rounded.iter().any(|(owner,_)|*owner==first) {
                let lanes=b.round_bf16_lanes(jobs[job].1)?;let mut source=b.vector(lanes.len())?;b.copy_prefix(lanes.vector(),&mut source,lanes.len())?;rounded.push((first,source));
            }
            let source=&rounded.iter().find(|(owner,_)|*owner==first).ok_or("row stage lost a rounded input")?.1;
            let matrix=&self.matrices[jobs[job].0];let Storage::Packed {shape,..}=&matrix.storage else {return Err("row stage FP8 storage differs".into());};
            // The table of one launch: this matrix, then the later matrices of this input that have its shape and block shape.
            let part=|other:usize|match &self.matrices[jobs[other].0].storage {
                Storage::Packed {shape,weight,scales}=>Ok((Shape {op:Op::Fp8Rows,tokens:m as u32,..*shape},weight,scales)),
                _=>Err(String::from("row stage FP8 storage differs"))};
            let like=|other:usize|matches!(&self.matrices[jobs[other].0].storage,Storage::Packed {shape:theirs,..}
                if (theirs.rows,theirs.cols,theirs.block_rows,theirs.block_cols)==(shape.rows,shape.cols,shape.block_rows,shape.block_cols));
            let mut group=vec![job];
            group.extend((job+1..jobs.len()).filter(|&other|made[other].is_none() && same(other) && like(other)).take(spark_gpu_probe::fp8_small::SHARED_MAX_PARTS-1));
            let parts=group.iter().map(|&other|part(other)).collect::<Result<Vec<_>>>()?;
            let probe=parts.iter().map(|(shape,weight,scales)|spark_gpu_probe::fp8_rows::Part {rows:shape.rows as usize,cols:shape.cols as usize,
                block_rows:shape.block_rows as usize,block_cols:shape.block_cols as usize,tokens:m,weight:weight.device_address(),scales:scales.device_address()}).collect::<Vec<_>>();
            if !spark_gpu_probe::fp8_small::shared_admitted(&probe,source.device_address()) {group.truncate(1);}
            let table=parts[..group.len()].iter().map(|(shape,weight,scales)|(*shape,Some((*weight,*scales)),0)).collect::<Vec<_>>();
            let bytes=parts[..group.len()].iter().map(|(_,weight,scales)|(weight.len()+scales.len())*4).sum::<usize>();
            let output=b.expert_group(&table,source,matrix.cols,self.lut.as_ref().ok_or("expert decode table is absent")?)?;
            if group.len()==1 {made[job]=Some((output,true));}
            else {
                // A part owns consecutive output units: [local rows,m] of each matrix, in the table's order.
                let extent=shape.rows as usize*m;
                if output.len()!=group.len()*extent {return Err("row stage FP8 group extent differs".into());}
                let output=output.into_resident_shared()?;
                for (at,&other) in group.iter().enumerate() {made[other]=Some((output.resident_range(at*extent..(at+1)*extent)?,true));}
            }
            self.cost.settle(b)?;self.cost.record(||"projection_3".into(),stamp,b,bytes);
        }
        for group in &mx {self.mx_rows_launch(b,jobs,group,m,&mut made)?;}
        for group in &shared {self.mx_rows_shared(b,jobs,group,m,&mut made)?;}
        // Each job's local rows. A whole (not split) output is done. A split one waits for its
        // gather: token-major for the mailbox; the host link keeps each producer's order.
        let mut outputs:Vec<Option<DeviceVector<'d>>>=Vec::with_capacity(jobs.len());let mut local:Vec<Option<DeviceVector<'d>>>=Vec::with_capacity(jobs.len());
        let mut row_orders=Vec::with_capacity(jobs.len());
        for (job,(name,input,round)) in jobs.iter().enumerate() {
            let (output,row_major)=match made[job].take() {Some(local)=>local,None=>self.matrix_rows(name,b,input,m,*round)?};let rows=self.matrices[*name].rows;
            row_orders.push(row_major);
            if !rows.split || self.kda_local.contains(*name) {local.push(None);outputs.push(Some(if row_major && m>1 {Self::transpose(b,&output,rows.count,m,false)?} else {output}));continue;}
            let block=if m>1 && row_major && mailbox.is_some() {Self::transpose(b,&output,rows.count,m,false)?} else {output};
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
                // Each projection's rank blocks side by side, in their producer's layout, out of the exchange buffer
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
                    let packet=joined.resident_range(start..start+extent)?;
                    outputs[job]=Some(if row_orders[job] {Self::transpose(b,&packet,rows.total,m,true)?}
                        else {Self::rank_rows(b,&packet,m,rows.count,ranks,true)?});start+=extent;
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
        // The split envelope holds for the rows a stage can send. A whole (not split) output is a value on this rank at
        // once: the stage keeps it, as it keeps a head-local one, and sends nothing for it.
        let split=rows.iter().copied().filter(|row|row.split).collect::<Vec<_>>();
        let Ok(words)=projection_stage::checked_rows(&split) else {return Ok(None);};
        if projection_stage::checked_messages(counts.len(),words,mailbox.slot_bytes()).is_err() {return Ok(None);}
        Ok(Some((rows,counts)))
    }
    /// PROJ-DELIVERY: whether plain projection stages may fork from now on (the device sets it before each decode step).
    pub fn side_lanes(&mut self,on:bool) {self.side_on=on;}
    /// Stages forked onto the side lane, and stages that would have forked but stayed whole because the step reads a selection.
    pub fn side_stage_counts(&self)->[u64;2] {self.side_stages}
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
        // PROJ-DELIVERY: a stage of plain projection jobs forks: side jobs first, on the side lane, then the main jobs; the gather
        // joins them. A side job is one plain or packed matrix: its launches write only their own fresh outputs and read no
        // backend-wide scratch (an MX matrix or a mixed batch reads the shared flash input). Kernels, inputs and each output's add
        // order are those of the serial order; only the recorded order and the stream differ. Outputs and their round flags are
        // put back in job order.
        // A decode step that reads a selection back keeps every stage on the main stream (side_on false): there the device
        // drains at each selection read, and a fork and its join only waited behind those reads (pair c4158 +0.41 ms a step).
        // SHARED LANE: a stage with routed experts forks its Gated jobs the same way (the shared expert's SwiGLU and its down
        // projection, one plain or packed matrix): they run beside the routed gate and up. Their two inputs were gathered before
        // the stage, the SwiGLU value and the down output are fresh, and only the gather reads the output. The routed job stays on
        // main, so the main lane always has work.
        let plain=jobs.iter().all(|job|matches!(job,Job::Project {..}));let routed=jobs.iter().any(|job|matches!(job,Job::Routed {..}));
        let single=|name:&str|self.matrices.get(name).is_some_and(|m|matches!(m.storage,Storage::Plain {..}|Storage::Packed {..}));
        let eligible=jobs.iter().map(|job|match job {
            Job::Project {input,..}=>plain && input.len()==1 && single(input[0].0),
            Job::Gated {name,..}=>routed && single(name),
            Job::Routed {..}=>false,
        }).collect::<Vec<_>>();
        let side=projection_stage::side_lanes(&eligible.iter().map(|&e|e && self.side_on).collect::<Vec<_>>());
        if side.iter().any(|&s|s) {self.side_stages[0]+=1;} else if projection_stage::side_lanes(&eligible).iter().any(|&s|s) {self.side_stages[1]+=1;}
        let order=(0..jobs.len()).filter(|&at|side[at]).chain((0..jobs.len()).filter(|&at|!side[at])).collect::<Vec<_>>();
        let mut parts:Vec<Option<(Vec<DeviceVector<'d>>,Vec<bool>)>>=(0..jobs.len()).map(|_|None).collect();
        let recorded=(|| {
            for at in order {
                let before=self.rounds.len();
                let mut run=||->Result<Option<Vec<DeviceVector<'d>>>> {
                    Ok(Some(match &jobs[at] {
                        Job::Project {input,round}=>{
                            if input.len()==1 {vec![self.apply(input[0].0,b,input[0].1,*round)?]}
                            else {let Some(values)=self.apply_batch(b,input,*round)? else {return Ok(None);};values}
                        },
                        Job::Routed {steps,ids,input}=>{let Some(values)=self.apply_routed(b,steps,ids,input)? else {return Ok(None);};values},
                        Job::Gated {name,gate,up,shape}=>{
                            let value=b.causal_operation(*shape,[Some(gate),Some(up),None,None],None)?;
                            vec![self.apply(name,b,&value,true)?]
                        },
                    }))
                };
                let Some(values)=(if side[at] {b.resident_side(&mut run)?} else {run()?}) else {return Ok(false);};
                let flags=self.rounds.split_off(before);parts[at]=Some((values,flags));
            }Ok::<bool,String>(true)
        })();
        // Every later command, the gather first, waits for the side jobs, also on a refusal.
        b.resident_side_join();
        if !recorded? {return Ok(None);}
        let (mut output,mut rounds)=(Vec::new(),Vec::new());
        for (values,flags) in parts.into_iter().flatten() {output.extend(values);rounds.extend(flags);}
        drop(guard);drop(defer);self.rounds.clear();
        if output.len()!=rows.len() || output.len()!=rounds.len() || output.iter().zip(&rows).any(|(value,row)|value.len()!=row.count) || rounds.iter().zip(&rows).any(|(round,row)|*round && !row.split) {return Err("projection stage local row extent differs".into());}
        let owned=jobs.iter().flat_map(|job|match job {
            Job::Project {input,..}=>input.iter().map(|(name,_)|self.kda_local.contains(*name)).collect::<Vec<_>>(),
            Job::Routed {steps,ids,..}=>vec![false;steps.len()*ids.len()],Job::Gated {..}=>vec![false],
        }).collect::<Vec<_>>();
        // A whole output is complete here and already rounded (the check above): kept, not sent.
        let owned=owned.into_iter().zip(&rows).map(|(local,row)|local || !row.split).collect::<Vec<_>>();
        let words=output.iter().zip(&owned).filter(|(_,local)|!**local).map(|(v,_)|v.len()).sum();
        let values=gather_stage_selected(b,self.mailbox.ok_or("projection stage mailbox is absent")?,&mut self.sequence,self.topology,output,rounds,rows,owned)?;
        projection_stage::accepted(jobs.len(),words);
        let mut values=values.into_iter();let result=counts.into_iter().map(|count|values.by_ref().take(count).collect::<Vec<_>>()).collect::<Vec<_>>();
        if values.next().is_some() || result.iter().any(Vec::is_empty) {return Err("projection stage result extent differs".into());}Ok(Some(result))
    }
}
