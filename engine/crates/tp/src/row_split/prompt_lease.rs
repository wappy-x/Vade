//! One synchronous prompt call borrows spare pool bytes. Admission charges stay fixed.
use super::*;
use crate::bounded_decoder::{PROMPT_GROUP,PROMPT_GROUP_MAX};
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    pub(super) fn current_prompt_group(&self)->usize {self.group.max(self.lease_rows.get())}
    /// Fixed checkpoint geometry belongs at startup, never on a request's clock.
    pub(super) fn init_prompt_lease_plan(&mut self)->Result<()> {
        if self.predictor==0 || self.group==PROMPT_GROUP_MAX {return Ok(());}
        let workspace=|group|crate::service_memory::workspace_for_prompt_group(self.checkpoint,self.context,self.workspace,group);
        let base=workspace(self.group)?;
        for (index,rows) in [PROMPT_GROUP_MAX/2,PROMPT_GROUP_MAX].into_iter().enumerate() {
            self.lease_extra[index]=if rows<=self.group {0} else {workspace(rows)?.checked_sub(base).ok_or("prompt lease workspace underflow")?};
        }Ok(())
    }
    fn prompt_lease_extra(&self,rows:usize)->Result<usize> {
        match rows {r if r==self.group=>Ok(0),r if r==PROMPT_GROUP_MAX/2=>Ok(self.lease_extra[0]),r if r==PROMPT_GROUP_MAX=>Ok(self.lease_extra[1]),_=>Err("invalid prompt lease rows".into())}
    }
    /// Same scheduler remainder on both ranks. Geometry can differ; gather the
    /// smaller offer before either rank executes a larger group.
    pub(super) fn agree_prompt_lease(&self,available:usize,live:usize)->Result<()> {
        if self.lease_rows.get()!=0 {return Err("previous prompt lease was not returned".into());}
        let mut offer=self.group;
        if self.predictor>0 && self.memo_admitted && live==1 {
            for rows in [PROMPT_GROUP_MAX,PROMPT_GROUP_MAX/2] {
                if rows>self.group && self.prompt_lease_extra(rows)?<=available {offer=rows;break;}
            }
        }
        let rows=self.negotiate_available_bytes(offer)?;
        if ![PROMPT_GROUP,PROMPT_GROUP_MAX/2,PROMPT_GROUP_MAX].contains(&rows) || rows<self.group || rows>offer {return Err("ranks disagree on prompt lease".into());}
        if rows>self.group {
            let bytes=self.prompt_lease_extra(rows)?;
            self.lease_bytes.set(bytes);self.lease_rows.set(rows);
            eprintln!("TEXT PROMPT LEASE rank={} rows={rows} borrowed_bytes={bytes} available_bytes={available}",self.split.topology.rank);
        }
        Ok(())
    }
    pub(super) fn prepare_prompt_lease(&self,available:usize)->Result<()> {
        // No predictor or already at maximum: preserve the original path.
        if self.predictor==0 || self.group==PROMPT_GROUP_MAX || available==0 {return Ok(());}
        if let Some(error)=self.failure_reason() {return Err(error);}
        self.finish((|| {
            self.split.command_for(25,0,available.try_into().map_err(|_|"prompt lease exceeds wire size")?)?;
            self.agree_prompt_lease(available,self.active.get())?;
            self.split.exchange.borrow_mut().complete()
        })())
    }
    pub(super) fn cancel_prompt_lease(&self)->Result<()> {
        if self.lease_rows.get()==0 {return Ok(());}
        self.finish((|| {
            self.split.command_for(26,0,0)?;
            self.return_prompt_lease()?;
            self.split.exchange.borrow_mut().complete()
        })())
    }
    /// After replacing the old state, on both ranks, on success and error.
    /// rows_keep drains, resets the temporary pool cap and frees real spares.
    pub(super) fn return_prompt_lease(&self)->Result<()> {
        if self.lease_rows.get()==0 {return Ok(());}
        let before=self.split.memory_bytes().map_or(0,|m|m.0);
        self.split.rows_return_lease()?;
        let after=self.split.memory_bytes().map_or(0,|m|m.0);
        let bytes=self.lease_bytes.replace(0);self.lease_rows.set(0);
        eprintln!("TEXT PROMPT RETURN rank={} borrowed_bytes={bytes} before_device_bytes={before} after_device_bytes={after} freed_bytes={} pool={:?}",self.split.topology.rank,before.saturating_sub(after),self.split.vector_pool_counts());
        Ok(())
    }
}

#[cfg(test)] mod tests;
