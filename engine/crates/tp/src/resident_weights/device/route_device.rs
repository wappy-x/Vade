use super::*;
pub(super) struct Group<'d> {names:Vec<String>,rows:Rows,table:spark_gpu_probe::driver::DeviceRouteTable<'d>,phases:usize,picks:usize,down:bool}
impl<'b,'d> Weights<'b,'d> {
    pub fn apply_routed(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],ids:&DeviceVector<'_>,input:&DeviceVector<'_>)->Result<Option<Vec<DeviceVector<'d>>>> {
        let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(None);};
        let key=&first.name;let picks=ids.len();let down=steps[0].kind==Kind::ExpertDown;
        if steps.len()>2 || picks==0 || !matches!(steps[0].kind,Kind::ExpertProject|Kind::ExpertDown)
            || steps.iter().any(|step|step.weights.len()!=steps[0].weights.len() || step.kind!=steps[0].kind) {return Ok(None);}
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
        let Some(group)=self.routes.get(key).and_then(Option::as_ref) else {return Ok(None);};
        if group.phases!=steps.len() || group.picks!=picks || group.down!=down || !group.names.iter().map(String::as_str).eq(steps.iter().flat_map(|step|step.weights.iter().map(|weight|weight.name.as_str()))) {return Ok(None);}
        let stamp=self.cost.start(b);let started=std::time::Instant::now();
        let output=b.device_route_projection(&group.table,ids,input)?;
        let output=b.round_bf16_lanes(&output)?.into_vector();if !b.resident_queue_active() {b.synchronize()?;}
        self.cost.record(||"projection_route_device".into(),stamp,b,0);let compute=started.elapsed().as_nanos();let stamp=self.cost.start(b);
        let rows=group.rows;let jobs=group.picks*group.phases;
        let result=if rows.split && !self.collecting.get() {
            gather_device(b,self.mailbox.ok_or("routed projection mailbox is absent")?,&mut self.sequence,self.topology,&output,&vec![rows;jobs])?
        } else {
            let output=output.into_resident_shared()?;
            (0..jobs).map(|job|output.resident_range(job*rows.count..(job+1)*rows.count)).collect::<Result<Vec<_>>>()?
        };
        self.cost.record(||"exchange_device".into(),stamp,b,0);self.timings[2][0]+=jobs as u128;self.timings[2][1]+=compute;self.timings[2][2]+=started.elapsed().as_nanos()-compute;
        Ok(Some(result))
    }
}
