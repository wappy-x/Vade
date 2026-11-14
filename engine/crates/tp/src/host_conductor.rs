use crate::{Result,layer_driver::LayerBoundary};
use engine_model::plan::{admission::CheckpointAdmission,layer::{Kind,Topology},state::StatePlan};
use spark_gpu_probe::{driver::{Backend,MhcHostLink,TpMailbox},mhc_host::HostBoundary,mhc_host_service::{self,Queue,Worker}};
use std::{sync::Arc,thread::Scope};

pub struct Conductor<'d> {mailbox:TpMailbox<'d>,queue:Arc<Queue>,cores:usize,parameter_bytes:usize}
impl<'d> Conductor<'d> {
    /// The host lane of one-row boundaries (one chat a step).
    pub fn new(backend:&Backend<'d>,checkpoint:&CheckpointAdmission,cores:usize)->Result<Option<Self>> {Self::with_rows(backend,checkpoint,cores,1)}
    /// The host lane of boundaries of up to `rows` rows (1..=8): the chats of one batched decode step.
    /// The mailbox slot holds `rows` residual planes; a boundary with more rows runs on the kernels.
    pub fn with_rows(backend:&Backend<'d>,checkpoint:&CheckpointAdmission,cores:usize,rows:usize)->Result<Option<Self>> {
        if cores==0 {return Err("host conductor requires at least one core".into());}
        if rows==0 || rows>mhc_host_service::ROWS_MAX {return Err("host conductor row count is outside 1..=8".into());}
        let description=checkpoint.description();let topology=Topology {rank:0,parts:1};
        let storage=StatePlan::new(description,topology,1,isize::MAX as usize).map_err(|error|error.to_string())?;
        let (mut words,mut coefficients,mut parameter_bytes)=(0usize,0usize,0usize);
        for boundary in LayerBoundary::each(description,checkpoint.headers(),&storage,1,topology,usize::MAX) {
            let boundary=boundary?;
            for (index,_) in boundary.program().steps.iter().enumerate().filter(|(_,step)|step.kind==Kind::MhcPrepare) {
                let shape=boundary.mhc_shape(index)?;words=words.max(shape.wide()).max(shape.coefficients());coefficients=coefficients.max(shape.coefficients());
                parameter_bytes=parameter_bytes.checked_add(HostBoundary::parameter_bytes(shape,cores)?).ok_or("host conductor parameter byte count overflow")?;
            }
        }
        if words==0 {return Ok(None);}
        let mailbox=backend.tp_mailbox(words.checked_mul(4*rows).ok_or("host conductor mailbox byte count overflow")?)?;
        let queue=Queue::new();queue.reserve_coefficients(coefficients)?;Ok(Some(Self {mailbox,queue,cores,parameter_bytes}))
    }
    pub fn link(&self)->MhcHostLink<'_,'d> {MhcHostLink::new(&self.mailbox,self.queue.clone(),self.cores)}
    pub fn parameter_bytes(&self)->usize {self.parameter_bytes}
    pub fn cores(&self)->usize {self.cores}
    pub fn fail(&self) {self.mailbox.fail();self.queue.stop();}
    pub fn spawn<'scope,'environment>(&'environment self,scope:&'scope Scope<'scope,'environment>)->Result<Worker<'scope>> {
        mhc_host_service::spawn(scope,self.mailbox.take_host().ok_or("host conductor endpoint is taken")?,self.queue.clone(),self.cores-1)
    }
}
