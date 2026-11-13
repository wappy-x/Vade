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
/// ROUTE ROWS: routed layers between two waits of a step on the device path (`apply_routed_rows`). No read drains such a
/// step, so every buffer it drops stays charged until a wait; a wait returns them to the pool and the next layers take
/// them again. Measured on the real model (two Sparks, check tool, 2026-11-12): with no wait a step of 5 to 8 chats freed
/// and allocated 90 to 686 buffers each step and the 8-chat step read 204.8 ms; a wait every 4 layers: 10 to 25 buffers
/// and 187.4 ms (main 191.4). A wait by held bytes (256 MiB, 1 to 3 waits a step) left 43 to 107 freed buffers a step
/// and read 0.4 to 1.5 ms above it. A wait changes no value.
const ROUTE_ROWS_WAIT:usize=4;
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
    /// Rows a ROUTE ROWS launch takes here: a batch step's chats or a small row group, with a device exchange attached.
    pub fn routes_rows(&self,m:usize)->bool {self.mailbox.is_some() && (1..=crate::bounded_decoder::BATCH_ROWS).contains(&m)}
    /// ROUTE ROWS: one layer's routed experts for `m` rows with the picks on the device (`ids`: `m * picks` IDs, row-major;
    /// `input`: one row a token for gate and up, one a (token, pick) for down). The projections are ONE build launch and
    /// ONE project launch for all rows (`Backend::device_route_projection_rows`: each (token, pick) row holds the words
    /// of that token's own one-token route). The rest is `apply_experts` as it stands: gate and up meet in SwiGLU on
    /// this rank's rows, one gather, one placement. The placement table is the identity: the rows are already one a
    /// (token, pick), in that order. None, before any work: a group this path does not take.
    pub fn apply_routed_rows(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],bound:Option<f32>,ids:&DeviceVector<'_>,input:&DeviceVector<'_>,m:usize)->Result<Option<DeviceVector<'d>>> {
        use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};
        let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(None);};
        if !self.routes_rows(m) || ids.is_empty() || !ids.len().is_multiple_of(m) || self.collecting.get() || self.deferring.get() {return Ok(None);}
        let picks=ids.len()/m;let down=steps[0].kind==Kind::ExpertDown;let key=first.name.clone();
        if !routed_shape(steps,picks) || (steps.len()==2)!=bound.is_some() || down && steps.len()!=1 {return Ok(None);}
        self.route_group(b,steps,picks)?;
        let stamp=self.cost.start(b);
        let (output,rows)={
            let Some(group)=self.routes.get(&key).and_then(Option::as_ref) else {return Ok(None);};
            let rows=group.rows;
            if group.phases!=steps.len() || group.picks!=picks || group.down!=down || !rows.split || rows.count==0 || !rows.total.is_multiple_of(rows.count)
                || !group.names.iter().map(String::as_str).eq(steps.iter().flat_map(|step|step.weights.iter().map(|weight|weight.name.as_str()))) {return Ok(None);}
            (b.device_route_projection_rows(&group.table,ids,picks,input,m)?,rows)
        };
        let (pairs,local)=(m*picks,rows.count);
        let output=match bound {
            Some(bound)=>{
                // [gate][pair][local] then [up][pair][local]: the two halves meet word for word, as in `apply_experts`.
                let output=output.into_resident_shared()?;let gate=output.resident_range(0..pairs*local)?;let up=output.resident_range(pairs*local..2*pairs*local)?;
                let s=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::SwiGlu,heads:0,dim:0,taps:0,rows:u32::try_from(pairs*local).map_err(|_|"routed rows exceed ABI")?,cols:0,epsilon:0.0,bound};
                b.causal_operation(s,[Some(&gate),Some(&up),None,None],None)?
            },
            None=>output,
        };
        if output.len()!=pairs*local {return Err("routed rows extent differs".into());}
        self.cost.settle(b)?;self.cost.record(||"projection_2".into(),stamp,b,0);
        let (parts,words)=(rows.total/local,output.len());
        let gathered=self.gather_small(b,&output)?;if gathered.is_none() {self.gather_mapped(b,&output)?;}drop(output);let stamp=self.cost.start(b);
        let exchange=match &gathered {Some(blocks)=>blocks,None=>self.exchange.as_ref().ok_or("prompt exchange is absent")?};
        // Row p of the result is the one row of part p: no earlier rows of its part, one row in it.
        let places=(0..pairs).flat_map(|pair|[pair as f32,1.0,0.0]).collect::<Vec<_>>();
        let tiled=spark_gpu_probe::prompt_ops::place_table(&places);
        let table=b.upload_vector(tiled.as_deref().unwrap_or(&places))?;
        let mut s=RowShape::new(if tiled.is_some() {Row::ExpertPlaceTile} else {Row::ExpertPlace},1).ok_or("prompt expert place shape")?;
        s.rows=u32::try_from(pairs).map_err(|_|"prompt expert rows exceed ABI")?;s.cols=u32::try_from(rows.total).map_err(|_|"prompt expert rows exceed ABI")?;
        s.dim=u32::try_from(local).map_err(|_|"prompt expert rows exceed ABI")?;s.taps=parts as u32;s.slots=u32::try_from(words).map_err(|_|"prompt expert rows exceed ABI")?;s.first=1;
        let out=b.prompt_operation(s,[Some(exchange),Some(&table),None,None,None],None)?;
        self.cost.settle(b)?;self.cost.record(||"exchange_upload".into(),stamp,b,0);
        // A routed layer ends with its down phase: one wait every ROUTE_ROWS_WAIT layers gives the dropped buffers back.
        if down {self.routed_layers+=1;if self.routed_layers>=ROUTE_ROWS_WAIT {self.routed_layers=0;b.synchronize()?;}}
        // One line a row count and kind, at first use: a served log then shows that the route stayed on the device.
        static SEEN:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let bit=1u64<<(m.min(31)+if down {32} else {0});
        if SEEN.fetch_or(bit,std::sync::atomic::Ordering::Relaxed)&bit==0 {eprintln!("ROUTE ROWS rank={} rows={m} picks={picks} down={down} local_rows={local} first use",self.topology.rank);}
        Ok(Some(out))
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
