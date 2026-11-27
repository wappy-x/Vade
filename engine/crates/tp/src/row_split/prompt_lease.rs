//! One synchronous prompt call borrows spare pool bytes. Admission charges stay fixed.
use super::*;
use crate::bounded_decoder::{PROMPT_GROUP,PROMPT_GROUP_MAX,PROMPT_GROUP_LEASE};
/// The groups a prompt call can borrow for, largest first: the four lease-only sizes, then the startup sizes above
/// `PROMPT_GROUP`. A service borrows only for a size above its startup group.
pub(super) const SIZES:[usize;6]=[PROMPT_GROUP_LEASE[0],PROMPT_GROUP_LEASE[1],PROMPT_GROUP_LEASE[2],PROMPT_GROUP_LEASE[3],PROMPT_GROUP_MAX,PROMPT_GROUP_MAX/2];
/// The largest group a service offers to borrow for: every lease size. Each keeps runs of at most 1,024 rows that
/// share the routed expert product launches. 4,096 rows are charged by the two phases of a wide pass where every
/// latent layer is absorbed and no predictor expands (`wide_phase_workspace`, bounded_decoder/rows/wide_charge.rs), and as a whole
/// group otherwise; 3,072 and 2,048 rows are charged as whole groups. An offer needs exactly one live session and,
/// on both ranks, the whole difference of its size and of every smaller size above the startup group
/// (`agree_prompt_lease`: the price list is not monotone).
pub(super) const SERVED:usize=PROMPT_GROUP_LEASE[0];
/// A group size the ranks may agree on: the startup sizes and the lease sizes.
fn agreed(rows:usize)->bool {rows==PROMPT_GROUP || SIZES.contains(&rows)}
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    pub(super) fn current_prompt_group(&self)->usize {self.group.max(self.lease_rows.get())}
    /// Fixed checkpoint geometry belongs at startup, never on a request's clock.
    /// The extra bytes of each size are the workspace formula of that group (`workspace_for_prompt_group`) less
    /// this service's own: every buffer that grows with the rows, the attention planes, the exchange and the state
    /// copies, as a startup group of that size would be charged; above 3,072 rows the two-phase charge of a wide
    /// pass where it applies (`wide_phase_workspace`).
    pub(super) fn init_prompt_lease_plan(&mut self)->Result<()> {
        let workspace=|group|crate::service_memory::workspace_for_prompt_group_with(self.checkpoint,self.context,self.workspace,group,self.latent_rows);
        let base=workspace(self.group)?;
        for (index,rows) in SIZES.into_iter().enumerate() {
            self.lease_extra[index]=if rows<=self.group {0} else {workspace(rows)?.checked_sub(base).ok_or("prompt lease workspace underflow")?};
        }
        eprintln!("TEXT PROMPT LEASE PLAN rank={} startup_rows={} sizes={SIZES:?} extra_bytes={:?}",self.split.topology.rank,self.group,self.lease_extra);
        Ok(())
    }
    fn prompt_lease_extra(&self,rows:usize)->Result<usize> {
        if rows==self.group {return Ok(0);}
        SIZES.iter().position(|&size|size==rows).map(|index|self.lease_extra[index]).ok_or_else(||"invalid prompt lease rows".into())
    }
    /// Same scheduler remainder on both ranks. Geometry can differ; gather the
    /// smaller offer before either rank executes a larger group.
    ///
    /// The offer is the top of the run of sizes above the startup group that the spare bytes fund, smallest first.
    /// The peer may answer with any smaller size, and the price list is not monotone: 4,096 rows at the phase charge
    /// cost less than 3,072 rows at the whole-group charge. So a size is offered only when every smaller size is
    /// funded too, and whatever the ranks agree on is funded here. With a monotone list this is the largest funded
    /// size, as before.
    pub(super) fn agree_prompt_lease(&self,available:usize,live:usize)->Result<()> {
        if self.lease_rows.get()!=0 {return Err("previous prompt lease was not returned".into());}
        let mut offer=self.group;
        if self.memo_admitted && live==1 {
            for rows in SIZES.into_iter().rev() {
                if rows<=self.group {continue;}
                if rows>SERVED || self.prompt_lease_extra(rows)?>available {break;}
                offer=rows;
            }
        }
        let rows=self.negotiate_available_bytes(offer)?;
        if !agreed(rows) || rows<self.group || rows>offer {return Err("ranks disagree on prompt lease".into());}
        if rows>self.group {
            let bytes=self.prompt_lease_extra(rows)?;
            // The agreed size is this rank's offer or a smaller size, and the offer funds each of them.
            if bytes>available {return Err("agreed prompt lease exceeds the spare bytes".into());}
            self.lease_bytes.set(bytes);self.lease_rows.set(rows);
            eprintln!("TEXT PROMPT LEASE rank={} rows={rows} borrowed_bytes={bytes} available_bytes={available}",self.split.topology.rank);
        }
        Ok(())
    }
    pub(super) fn prepare_prompt_lease(&self,available:usize)->Result<()> {
        // Nothing spare: the startup group, and no command on the wire. With or without the predictor: the borrow
        // and its return do not read it.
        if available==0 {return Ok(());}
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
