//! Stable prompt routing stays on the ordered device stream through both products.
use super::*;
use std::rc::Rc;
use spark_gpu_probe::driver::{PromptExpertTable,PromptRoute,PromptRowScales,PromptNativeKeep,PromptOperandSource};
use spark_gpu_probe::causal_ops::Shape as Fold;
use spark_gpu_probe::prompt_ops::{Op as Row,Shape as RowShape};

pub(super) struct Inventory<'d> {table:Rc<PromptExpertTable<'d>>,rows:Rows,names:Vec<String>}
struct Epoch<'d> {routes:Vec<Rc<PromptRoute<'d>>>,down:Rc<PromptRowScales<'d>>,table:Rc<PromptExpertTable<'d>>,rows:Rows,names:Vec<String>,fold:Fold,kept:Vec<PromptNativeKeep<'d>>}
/// The middle's immutable route epoch. It retains IDs, tables, native launch
/// headers and weight owners through the down product and final fold.
pub struct PromptMiddle<'d> {epoch:Rc<Epoch<'d>>,set:usize}
impl PromptMiddle<'_> {
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {
        self.epoch.down.allocations(out);self.epoch.table.allocations(out);
        for route in &self.epoch.routes {route.allocations(out);}for keep in &self.epoch.kept {keep.allocations(out);}
    }
}
fn place_shape(rows:usize,local:usize)->Result<RowShape> {
    let n=|n:usize|u32::try_from(n).map_err(|_|"device route shape exceeds ABI".to_string());
    let mut s=RowShape::new(Row::ExpertPlaceTile,1).ok_or("device route place shape")?;
    s.rows=n(rows)?;s.cols=n(local)?;s.dim=n(local)?;s.taps=1;s.slots=n(rows.checked_mul(local).ok_or("device route extent overflow")?)?;s.first=1;Ok(s)
}
impl<'b,'d> Weights<'b,'d> {
    pub(super) fn warm_prompt_inventory(&mut self,b:&Backend<'d>,step:&engine_model::plan::layer::Step)->Result<()> {
        let Some(first)=step.weights.first() else {return Ok(());};
        if self.prompt_tables.contains_key(&first.name) {return Ok(());}
        let names=step.weights.iter().map(|w|w.name.clone()).collect::<Vec<_>>();
        let matrices=names.iter().map(|n|self.matrices.get(n)).collect::<Option<Vec<_>>>();
        let inventory=if let Some(matrices)=matrices {
            let rows=matrices[0].rows;
            if matrices.iter().any(|m|m.rows!=rows) {None} else {
                let parts=matrices.iter().map(|m|match &m.storage {Storage::Packed {shape,weight,scales}=>Some((*shape,weight,scales)),_=>None}).collect::<Option<Vec<_>>>();
                if let Some(parts)=parts {b.prompt_expert_table(&parts)?.map(|table|Inventory {table:Rc::new(table),rows,names})} else {None}
            }
        } else {None};
        self.prompt_tables.insert(first.name.clone(),inventory);Ok(())
    }
    fn prompt_inventory(&self,step:&engine_model::plan::layer::Step)->Option<&Inventory<'d>> {
        let first=step.weights.first()?;let value=self.prompt_tables.get(&first.name)?.as_ref()?;
        value.names.iter().map(String::as_str).eq(step.weights.iter().map(|w|w.name.as_str())).then_some(value)
    }
    /// Preflight the full gate/up/middle/down/fold pipeline from immutable
    /// geometry before submitting the first dynamic route operation.
    #[allow(clippy::type_complexity)]
    pub fn apply_prompt_native(&mut self,b:&Backend<'d>,steps:&[engine_model::plan::layer::Step],bound:f32,sets:&[(usize,Rc<DeviceVector<'d>>,Rc<DeviceVector<'d>>)],(fold,shared):(Fold,Option<&str>))->Result<Option<Vec<(DeviceVector<'d>,PromptMiddle<'d>)>>> {
        if steps.len()!=3 || sets.is_empty() || sets.len()>spark_gpu_probe::prompt_route::MAX_SETS as usize || !self.folds_groups() || self.collecting.get() || self.deferring.get() {return Ok(None);}
        let (Some(gate),Some(up),Some(down))=(self.prompt_inventory(&steps[0]),self.prompt_inventory(&steps[1]),self.prompt_inventory(&steps[2])) else {return Ok(None);};
        let (local,cols)=gate.table.geometry();let total=gate.rows.total;let (down_local,down_cols)=down.table.geometry();
        if gate.rows!=up.rows || gate.table.geometry()!=up.table.geometry() || gate.table.experts()!=up.table.experts() || gate.table.experts()!=down.table.experts()
            || local==0 || !total.is_multiple_of(local) || !gate.rows.split || !down.rows.split || down_cols!=total || fold.cols as usize!=down.rows.total || fold.rows==0 {return Ok(None);}
        let Some(shared)=shared.and_then(|name|self.matrices.get(name).filter(|_|!self.kda_local.contains(name))) else {return Ok(None);};
        if self.mailbox.is_none() || shared.rows!=down.rows {return Ok(None);}
        let experts=gate.table.experts();let picks=fold.rows as usize;
        for (m,input,ids) in sets {
            let shape=spark_gpu_probe::prompt_route::Shape {rows:u32::try_from(*m).unwrap_or(u32::MAX),picks:fold.rows,experts:experts as u32};
            if !shape.admitted() || ids.len()!=m*picks || input.len()!=m*cols || ids.is_segmented() || input.is_segmented()
                || spark_gpu_probe::expert_ops::packed_middle_words(m*picks,local).is_none()
                || !spark_gpu_probe::prompt_admission::place_admitted(place_shape(m*picks,local)?)
                || group_fold_shapes(down_local,m*picks,fold,*m)?.is_none() {return Ok(None);}
        }
        let (gate_table,up_table,down_table)=(gate.table.clone(),up.table.clone(),down.table.clone());
        let (down_rows,names)=(down.rows,down.names.clone());
        let routes=sets.iter().map(|(m,_,ids)|b.prompt_route(ids.clone(),spark_gpu_probe::prompt_route::Shape {rows:*m as u32,picks:fold.rows,experts:experts as u32}).map(Rc::new)).collect::<Result<Vec<_>>>()?;
        let gate_scales=Rc::new(b.prompt_row_scales(routes.clone(),gate_table.clone())?);
        let up_scales=if gate_table.same_calibration(&up_table) {gate_scales.clone()} else {Rc::new(b.prompt_row_scales(routes.clone(),up_table.clone())?)};
        let down_scales=Rc::new(b.prompt_row_scales(routes.clone(),down_table.clone())?);
        let inputs=||sets.iter().map(|(_,v,_)|PromptOperandSource::Gathered(v.clone())).collect::<Vec<_>>();
        let packed=b.prompt_route_operands(gate_scales.clone(),inputs())?;
        let gate_batch=b.prompt_native_batch(gate_scales,gate_table,false)?;b.prompt_route_product(&gate_batch,&packed)?;
        let up_batch=b.prompt_native_batch(up_scales.clone(),up_table,false)?;
        if Rc::ptr_eq(gate_batch.row_scales(),&up_scales) {b.prompt_route_product(&up_batch,&packed)?;} else {
            let other=b.prompt_route_operands(up_scales,inputs())?;b.prompt_route_product(&up_batch,&other)?;
        }
        let (gate,gate_keep)=gate_batch.into_outputs();let (up,up_keep)=up_batch.into_outputs();drop(packed);
        let epoch=Rc::new(Epoch {routes,down:down_scales,table:down_table,rows:down_rows,names,fold,kept:vec![gate_keep,up_keep]});
        let mut out=Vec::with_capacity(sets.len());let mut first=0;
        for (at,((a,c),(m,_,_))) in gate.into_iter().zip(up).zip(sets).enumerate() {
            let s=Fold {op:spark_gpu_probe::causal_ops::Op::SwiGlu,heads:0,dim:0,taps:0,rows:u32::try_from(a.len()).map_err(|_|"device expert middle exceeds ABI")?,cols:0,epsilon:0.,bound};
            let middle=b.causal_operation(s,[Some(&a),Some(&c),None,None],None)?;
            let scales=epoch.down.values().resident_range(first..first+m*picks)?;first+=m*picks;
            let middle=self.prompt_exchange_packed(b,middle,epoch.routes[at].grouped(),&scales,m*picks,local,total)?;
            out.push((middle,PromptMiddle {epoch:epoch.clone(),set:at}));
        }
        Ok(Some(out))
    }
    fn prompt_exchange_packed(&mut self,b:&Backend<'d>,output:DeviceVector<'d>,table:&DeviceVector<'d>,scales:&DeviceVector<'d>,rows:usize,local:usize,total:usize)->Result<DeviceVector<'d>> {
        let (cw,sw,cr)=spark_gpu_probe::expert_ops::packed_middle_words(rows,local).ok_or("device middle packed shape")?;
        let placed=b.prompt_operation(place_shape(rows,local)?,[Some(&output),Some(table),None,None,None],None)?;
        let (codes,scale_codes)=b.nv_quantize_pack_planes(&placed,local,scales)?;
        self.gather_mapped_parts(b,&[&codes,&scale_codes])?;drop(codes);drop(scale_codes);drop(placed);drop(output);
        let block=rows*(cw+sw);let parts=total/local;let exchange=self.exchange.as_ref().ok_or("device middle exchange is absent")?;
        if exchange.len()!=block*parts {return Err("device middle exchange extent differs".into());}
        let mut packed=Self::rank_rows(b,exchange,cr,cw,parts,false)?;let mut blocks=b.vector(rows*sw*parts)?;
        for rank in 0..parts {b.copy_range(exchange,&mut blocks,rank*block+rows*cw,rank*rows*sw,rows*sw)?;}
        let joined=Self::rank_rows(b,&blocks,rows,sw,parts,false)?;b.copy_range(&joined,&mut packed,0,rows*parts*cw,rows*parts*sw)?;
        crate::exchange_trace::reader();self.exchange_read(b)?;Ok(packed)
    }
    #[allow(clippy::type_complexity)]
    pub fn apply_prompt_native_fold(&mut self,b:&Backend<'d>,down:&engine_model::plan::layer::Step,sets:&[(usize,Rc<DeviceVector<'d>>,&PromptMiddle<'d>,&DeviceVector<'d>,&DeviceVector<'d>,Option<&DeviceVector<'d>>)],fold:Fold)->Result<Vec<DeviceVector<'d>>> {
        let epoch=&sets.first().ok_or("device expert fold has no sets")?.2.epoch;let local=epoch.rows.count;let picks=fold.rows as usize;
        if sets.len()!=epoch.routes.len() || !epoch.names.iter().map(String::as_str).eq(down.weights.iter().map(|w|w.name.as_str()))
            || fold.op!=epoch.fold.op || fold.rows!=epoch.fold.rows || fold.cols!=epoch.fold.cols || fold.taps!=epoch.fold.taps || sets.iter().enumerate().any(|(at,(m,_,marker,ids,weights,shared))|!Rc::ptr_eq(epoch,&marker.epoch) || marker.set!=at
                || *m!=epoch.routes[at].shape().rows as usize || ids.device_address()!=epoch.routes[at].ids().device_address() || ids.len()!=m*picks || weights.len()!=m*picks
                || (fold.taps!=0)!=shared.is_some() || shared.is_some_and(|v|v.len()!=m*local)) {return Err("device expert fold epoch or geometry differs".into());}
        let packed=b.prompt_route_operands(epoch.down.clone(),sets.iter().map(|(_,v,_,_,_,_)|PromptOperandSource::Packed(v.clone())).collect())?;
        // DIRECT PLACE: the down products land in original pair order ([pair][local], the words ExpertPlaceTile with
        // the route's placed table would make), so the fold reads the product output with no placement launch.
        let batch=b.prompt_native_batch(epoch.down.clone(),epoch.table.clone(),true)?;b.prompt_route_product(&batch,&packed)?;
        let (outputs,keep)=batch.into_outputs();drop(packed);let mut out=Vec::with_capacity(sets.len());
        for (output,(m,_,marker,ids,weights,shared)) in outputs.into_iter().zip(sets) {
            let (_,wrapped)=group_fold_shapes(local,m*picks,fold,*m)?.ok_or("device expert fold admission changed")?;
            let folded=b.prompt_operation(wrapped,[Some(ids),Some(weights),Some(&output),*shared,None],None)?;
            let gathered=self.gather_small(b,&folded)?;if gathered.is_none() {self.gather_mapped(b,&folded)?;}drop(folded);drop(output);
            let exchange=match &gathered {Some(v)=>v,None=>self.exchange.as_ref().ok_or("device folded exchange is absent")?};
            out.push(Self::rank_rows(b,exchange,*m,local,epoch.rows.total/local,false)?);
            crate::exchange_trace::reader();if gathered.is_none() {self.exchange_read(b)?;}
            spark_gpu_probe::prompt_launches::record("prompt_fold_first");let _=marker;
        }
        drop(keep);Ok(out)
    }
}
