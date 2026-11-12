use super::*;
/// The device part of LOCAL FOLD behind the down projection. `down` holds this rank's rows of every pick (picks x
/// rows.count words, not yet rounded); `shared` is the gathered shared row (rows.total words) when the fold has one.
/// Round the rows as the joined put did, fold them (`fold` with this rank's row count), exchange the folded rows:
/// the fold's whole output, rows.total words. The round and the fold are one launch (the fold with the `FOLD_ROUNDS`
/// mark in `dim`: each row word is rounded as it is read, with the round launch's own cast) where the device has that
/// form; else the round launch, then the fold. Same words either way.
#[allow(clippy::too_many_arguments)]
pub fn fold_then_gather<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,ids:&DeviceVector<'_>,weights:&DeviceVector<'_>,
    down:&DeviceVector<'_>,shared:Option<&DeviceVector<'d>>,fold:spark_gpu_probe::causal_ops::Shape,rows:Rows)->Result<DeviceVector<'d>> {
    let picks=ids.len();
    if picks==0 || weights.len()!=picks || fold.op!=spark_gpu_probe::causal_ops::Op::ExpertFold || fold.rows as usize!=picks || fold.cols as usize!=rows.total || (fold.taps!=0)!=shared.is_some()
        || !rows.split || rows.count==0 || rows.start.checked_add(rows.count).is_none_or(|end|end>rows.total) || picks.checked_mul(rows.count)!=Some(down.len())
        || shared.is_some_and(|shared|shared.len()!=rows.total) {return Err("routed fold extent differs".into());}
    let own=match shared {Some(shared)=>Some(shared.resident_range(rows.start..rows.start+rows.count)?),None=>None};
    let local=spark_gpu_probe::causal_ops::Shape {cols:rows.count.try_into().map_err(|_|"routed fold row count exceeds ABI")?,..fold};
    let fused=spark_gpu_probe::causal_ops::Shape {dim:spark_gpu_probe::causal_ops::FOLD_ROUNDS,..local};
    let folded=if spark_gpu_probe::causal_ops::fast_admitted(fused,spark_gpu_probe::causal_ops::FAST_THREADS) {
        b.causal_operation(fused,[Some(ids),Some(weights),Some(down),own.as_ref()],None)?
    } else {
        let rounded=b.round_bf16_lanes(down)?.into_vector();
        b.causal_operation(local,[Some(ids),Some(weights),Some(&rounded),own.as_ref()],None)?
    };
    let mut gathered=gather_device(b,mailbox,sequence,topology,&[(&folded,false)],&[rows])?;
    if gathered.len()!=1 {return Err("routed fold gather extent differs".into());}
    gathered.pop().ok_or_else(||"routed fold gather is empty".into())
}
/// The routed steps apply_routed takes: one or two ExpertProject or ExpertDown steps of one kind and one weight count.
pub(super) fn routed_shape(steps:&[engine_model::plan::layer::Step],picks:usize)->bool {
    !steps.is_empty() && steps.len()<=2 && picks>0 && matches!(steps[0].kind,Kind::ExpertProject|Kind::ExpertDown)
        && steps.iter().all(|step|step.weights.len()==steps[0].weights.len() && step.kind==steps[0].kind)
}
pub(super) struct Group<'d> {names:Vec<String>,rows:Rows,table:spark_gpu_probe::driver::DeviceRouteTable<'d>,phases:usize,picks:usize,down:bool}
impl<'b,'d> Weights<'b,'d> {
    /// The route table of one routed group, made at its first use (the key is the first weight's name). A group this
    /// path does not admit is remembered as absent. Both routed paths (apply_routed, apply_routed_fold) call it, so a
    /// layer's first step takes the same path as every later step.
    pub(super) fn route_group(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],picks:usize)->Result<()> {
        let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(());};
        let key=&first.name;let down=steps[0].kind==Kind::ExpertDown;
        if !self.routes.contains_key(key) {
            let names=steps.iter().flat_map(|step|step.weights.iter().map(|weight|weight.name.clone())).collect::<Vec<_>>();
            let matrices=names.iter().map(|name|self.matrices.get(name).ok_or_else(||format!("routed projection is not resident: {name}"))).collect::<Result<Vec<_>>>()?;
            let rows=matrices[0].rows;
            let parts=matrices.iter().map(|matrix|match &matrix.storage {Storage::Packed {shape,weight,scales}=>Some((*shape,weight,scales)),_=>None}).collect::<Option<Vec<_>>>();
            let group=if matrices.iter().any(|matrix|matrix.rows!=rows) || rows.split && self.mailbox.is_none() {None} else if let Some(parts)=parts {
                b.device_route_table(&parts,steps[0].weights.len().try_into().map_err(|_|"routed expert table exceeds ABI")?,
                    picks.try_into().map_err(|_|"routed picks exceed ABI")?,steps.len() as u32,down)?
                    .map(|table|Group {names,rows,table,phases:steps.len(),picks,down})
            } else {None};
            self.routes.insert(key.clone(),group);
        }
        Ok(())
    }
    pub fn apply_routed(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],ids:&DeviceVector<'_>,input:&DeviceVector<'_>)->Result<Option<Vec<DeviceVector<'d>>>> {
        let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(None);};
        let key=&first.name;let picks=ids.len();let down=steps[0].kind==Kind::ExpertDown;
        if !routed_shape(steps,picks) {return Ok(None);}
        self.route_group(b,steps,picks)?;
        let Some(group)=self.routes.get(key).and_then(Option::as_ref) else {return Ok(None);};
        if group.phases!=steps.len() || group.picks!=picks || group.down!=down || !group.names.iter().map(String::as_str).eq(steps.iter().flat_map(|step|step.weights.iter().map(|weight|weight.name.as_str()))) {return Ok(None);}
        let stamp=self.cost.start(b);let started=std::time::Instant::now();
        let output=b.device_route_projection(&group.table,ids,input)?;
        // The stage gather or the device gather below does this round inside its joined put.
        let (rows,jobs)=(group.rows,group.picks*group.phases);let owed=rows.split && (self.deferring.get() || !self.collecting.get());
        let output=if owed {output} else {let rounded=b.round_bf16_lanes(&output)?.into_vector();if !b.resident_queue_active() {b.synchronize()?;}rounded};
        self.cost.record(||"projection_route_device".into(),stamp,b,0);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
        let result=if rows.split && !self.collecting.get() {
            gather_device(b,self.mailbox.ok_or("routed projection mailbox is absent")?,&mut self.sequence,self.topology,&[(&output,owed)],&vec![rows;jobs])?
        } else {
            let output=output.into_resident_shared()?;if self.deferring.get() {self.rounds.extend(std::iter::repeat_n(owed,jobs));}
            (0..jobs).map(|job|output.resident_range(job*rows.count..(job+1)*rows.count)).collect::<Result<Vec<_>>>()?
        };
        self.cost.record(||"exchange_device".into(),stamp,b,0);self.timings[2][0]+=jobs as u128;self.timings[2][1]+=compute;self.timings[2][2]+=started.elapsed().as_nanos()-compute;
        Ok(Some(result))
    }    /// LOCAL FOLD: the down projection of the picked experts on this rank's rows, their BF16 round (the round the
    /// joined put gave them), the fold of those rows (`fold` with this rank's row count: the same per-word chain as the
    /// fold of the gathered rows), and one exchange of the folded rows. `shared` is the gathered shared row; this rank
    /// reads its own rows of it. The result is the fold's whole output. None: a shape this path does not take.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_routed_fold(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],ids:&DeviceVector<'_>,input:&DeviceVector<'_>,weights:&DeviceVector<'_>,
        shared:Option<&DeviceVector<'d>>,fold:spark_gpu_probe::causal_ops::Shape)->Result<Option<DeviceVector<'d>>> {
        let picks=ids.len();
        if steps.len()!=1 || steps[0].kind!=Kind::ExpertDown || self.collecting.get() || self.deferring.get() || self.mailbox.is_none() || weights.len()!=picks
            || fold.op!=spark_gpu_probe::causal_ops::Op::ExpertFold || fold.rows as usize!=picks || (fold.taps!=0)!=shared.is_some() {return Ok(None);}
        let Some(key)=steps[0].weights.first().map(|weight|weight.name.clone()) else {return Ok(None);};
        // The table of this down group (made here at the layer's first step); a group without one keeps the two steps.
        self.route_group(b,steps,picks)?;
        let Some(group)=self.routes.get(&key).and_then(Option::as_ref) else {return Ok(None);};
        let rows=group.rows;
        if !group.down || group.phases!=1 || group.picks!=picks || !rows.split || rows.total!=fold.cols as usize || rows.count==0 || rows.start.checked_add(rows.count).is_none_or(|end|end>rows.total)
            || shared.is_some_and(|shared|shared.len()!=rows.total) || !group.names.iter().map(String::as_str).eq(steps[0].weights.iter().map(|weight|weight.name.as_str())) {return Ok(None);}
        let stamp=self.cost.start(b);let started=std::time::Instant::now();
        let output=b.device_route_projection(&group.table,ids,input)?;
        self.cost.record(||"projection_route_fold".into(),stamp,b,0);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
        let gathered=fold_then_gather(b,self.mailbox.ok_or("routed fold mailbox is absent")?,&mut self.sequence,self.topology,ids,weights,&output,shared,fold,rows)?;
        self.cost.record(||"exchange_device".into(),stamp,b,0);self.timings[2][0]+=picks as u128;self.timings[2][1]+=compute;self.timings[2][2]+=started.elapsed().as_nanos()-compute;
        crate::projection_stage::folded();Ok(Some(gathered))
    }
}
