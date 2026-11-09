//! Shared prompt workspace and transactional ownership of every live session charge.
#![forbid(unsafe_code)]
use std::collections::BTreeMap;
use engine_model::plan::admission::CheckpointAdmission;
use crate::Result;
pub struct PromptGroupAdmission<'a> {_checkpoint:&'a CheckpointAdmission,plan:crate::admission_plan::SessionChargePlan,workspaces:[usize;3],context:usize,budget:usize,resident:usize,maximum:usize,group:Option<usize>,workspace:usize,sessions:BTreeMap<u64,usize>}
impl<'a> PromptGroupAdmission<'a> {
    pub fn new(checkpoint:&'a CheckpointAdmission,context:usize,budget:usize,resident:usize,resident_ready:bool,maximum:usize)->Result<Self> {
        if context==0 || resident>budget || ![256,512,1024].contains(&maximum) {return Err("invalid prompt admission geometry or ceiling".into());}
        let plan=crate::admission_plan::SessionChargePlan::new(checkpoint,context)?;
        let mut workspaces=[0;3];for (i,group) in [1024,512,256].into_iter().enumerate() {if group<=maximum {workspaces[i]=crate::row_split::prompt_group_service_charges(checkpoint,context,resident_ready,group)?.1;}}
        Ok(Self {_checkpoint:checkpoint,plan,workspaces,context,budget,resident,maximum,group:None,workspace:0,sessions:BTreeMap::new()})
    }
    /// Install funded_bytes() on both ranks before allocating the reserved state.
    /// The group cannot shrink while any session remains live.
    pub fn reserve(&mut self,id:u64,prompt:usize,tokens:usize)->Result<usize> {
        if tokens>self.context || self.sessions.contains_key(&id) {return Err("duplicate session or context exceeds prompt admission".into());}
        let charge=self.plan.session(prompt,tokens)?;
        let total=self.sessions.values().try_fold(charge,|n,&v|n.checked_add(v).ok_or("live session charge overflow"))?;
        for (i,group) in [1024,512,256].into_iter().enumerate() {
            if group>self.maximum || self.group.is_some_and(|fixed|fixed!=group) {continue;}
            let workspace=self.workspaces[i];
            let funded=self.resident.checked_add(workspace).and_then(|n|n.checked_add(total)).ok_or("live group funding overflow")?;
            if funded>self.budget {continue;}
            self.sessions.insert(id,charge);self.group=Some(group);self.workspace=workspace;return Ok(group);
        }
        Err("all live sessions and their prompt workspace exceed device budget".into())
    }
    /// Call only after the released state and its spare buffers have drained.
    pub fn release(&mut self,id:u64)->Result<()> {
        if self.sessions.remove(&id).is_none() {return Err("session has no prompt admission credit".into());}
        if self.sessions.is_empty() {self.group=None;}Ok(())
    }
    pub fn funded_bytes(&self)->Result<usize> {self.sessions.values().try_fold(self.resident.checked_add(self.workspace).ok_or("group funding overflow")?,|n,&v|n.checked_add(v).ok_or("group funding overflow")).map_err(str::to_owned)}
    pub fn live(&self)->usize {self.sessions.len()}
}
