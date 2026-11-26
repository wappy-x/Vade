//! Return the spare vectors added by a widened, contended prompt.
use super::*;

pub(super) const SNAPSHOT: u32 = 27;

impl<C: Compute, E: Exchange> Factory<'_, C, E> {
    pub(super) fn snapshot_prompt_spares(&self, id: u64) {
        if let Some(bytes) = self.split.compute.prompt_spare_bytes() {
            self.prompt_spares.borrow_mut().entry(id).or_insert(bytes);
        }
    }

    pub(super) fn begin_prompt_spares(&self, id: u64) -> Result<()> {
        if let Some(error) = self.failure_reason() { return Err(error); }
        if self.prompt_spares.borrow().contains_key(&id) {
            return Ok(());
        }
        let started = std::time::Instant::now();
        self.snapshot_prompt_spares(id);
        let Some(target) = self.prompt_spares.borrow().get(&id).copied() else { return Ok(()); };
        self.finish(self.split.command_for(SNAPSHOT, 0, id)
            .and_then(|()| self.split.exchange.borrow_mut().complete()))?;
        eprintln!("TEXT CONTENDED SPARES BEGIN rank={} session={id} target_bytes={target} elapsed_ms={:.3}",
            self.split.topology.rank, started.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }

    pub(super) fn return_prompt_spares(&self, id: u64) -> Result<()> {
        let Some(target) = self.prompt_spares.borrow_mut().remove(&id) else { return Ok(()); };
        let started = std::time::Instant::now();
        let (before, after) = self.split.compute.return_prompt_spares(target)?;
        eprintln!("TEXT CONTENDED SPARES RETURN rank={} session={id} target_bytes={target} before_bytes={before} after_bytes={after} elapsed_ms={:.3}",
            self.split.topology.rank, started.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }
}
