//! Process-local cache affinity. Loaded checkpoints and reconciled plans stay
//! immutable and borrowed for the entire snapshot lifetime; reloads miss.
use engine_model::{plan::ModelPlan, split::SplitPlan};

#[derive(Clone, Debug)]
pub struct Identity<'p> {
    checkpoint: usize,
    plan: &'p ModelPlan,
    arithmetic: &'static str,
    ranks: u32,
    rank: u32,
    split: SplitPlan,
    group: Vec<u64>,
    epoch: u64,
}
impl PartialEq for Identity<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.checkpoint == other.checkpoint && std::ptr::eq(self.plan, other.plan)
            && self.arithmetic == other.arithmetic && self.ranks == other.ranks
            && self.rank == other.rank && self.split == other.split
            && self.group == other.group && self.epoch == other.epoch
    }
}
impl Eq for Identity<'_> {}
impl<'p> Identity<'p> {
    pub fn new(checkpoint: usize, plan: &'p ModelPlan, arithmetic: &'static str,
        split: &SplitPlan, rank: u32) -> Result<Self, String> {
        if checkpoint == 0 || arithmetic.is_empty() { return Err("invalid hybrid cache identity".into()); }
        if split.ranks as usize != split.owners.len() || rank >= split.ranks { return Err("cache rank ownership absent".into()); }
        Ok(Self { checkpoint, plan, arithmetic, ranks: split.ranks, rank, split: split.clone(), group: Vec::new(), epoch: 0 })
    }
    /// Ordered immutable rank membership and incarnation; no live migration.
    pub fn with_group(mut self, group: &[u64], epoch: u64) -> Result<Self, String> {
        if epoch == 0 || group.len() != self.ranks as usize || group.contains(&0)
            || group.iter().collect::<std::collections::BTreeSet<_>>().len() != group.len() {
            return Err("invalid hybrid rank group".into());
        }
        self.group = group.to_vec(); self.epoch = epoch; Ok(self)
    }
}

/// Replaying a tail after a nonzero entry needs every recurrent/history snapshot.
pub fn replay_entry(position: usize, tail: usize, snapshots: bool) -> Result<usize, String> {
    let entry = position.checked_sub(tail).ok_or("hybrid tail exceeds committed position")?;
    if entry != 0 && tail != 0 && !snapshots { return Err("hybrid recurrent replay snapshot absent".into()); }
    Ok(entry)
}
