//! Keep the first decode of a bounded prompt group together.
#[derive(Default)]
pub(super) struct Cohorts { groups: Vec<Vec<u64>>, serial: Vec<u64> }
impl Cohorts {
    /// New arrivals cannot extend a group that already started its prompt.
    pub fn begin(&mut self, ids: impl Iterator<Item = u64>) {
        let new: Vec<_> = ids.filter(|id| !self.holds(*id) && !self.serial(*id)).collect();
        if new.len() > 1 { self.groups.push(new); }
    }
    /// Missing members finished, failed, or cancelled. They cannot hold a peer.
    pub fn refresh(&mut self, live: impl Iterator<Item = (u64, bool)>) {
        let live: Vec<_> = live.collect();
        for group in &mut self.groups { group.retain(|id| live.iter().any(|(active, _)| id == active)); }
        self.groups.retain(|group| group.iter().any(|id| live.iter().any(|(active, ready)| id == active && !ready)));
        self.serial.retain(|id| live.iter().any(|(active, ready)| id == active && !ready));
        // Latch the old sliced path at arrival, even if the older decoder ends
        // before this prompt does. Completed peers held by a fresh group do not decode.
        if self.decoding(live.iter().copied()) {
            for &(id, ready) in &live {
                if !ready && !self.holds(id) && !self.serial(id) { self.serial.push(id); }
            }
        }
    }
    pub fn holds(&self, id: u64) -> bool { self.groups.iter().any(|group| group.contains(&id)) }
    pub fn serial(&self, id: u64) -> bool { self.serial.contains(&id) }
    pub fn serial_ids(&self) -> Vec<u64> { self.serial.clone() }
    pub fn decoding(&self, mut live: impl Iterator<Item = (u64, bool)>) -> bool {
        live.any(|(id, ready)| ready && !self.holds(id))
    }
}

/// Share a funded row bound across every ready short prompt. Fixed canonical
/// tiles keep their own size; only independent, variable-size groups split here.
pub(super) fn row_cap(budget: usize, ready: usize, independent: bool, tile: usize) -> usize {
    if independent && tile == 0 { (budget / ready.max(1)).max(1) } else { budget }
}

#[cfg(test)]
mod tests {
    use super::{Cohorts, row_cap};
    #[test]
    fn prompt_peers_join_decode_together_without_holding_older_chats() {
        let mut c = Cohorts::default(); c.begin([2, 3, 4].into_iter());
        c.refresh([(1, true), (2, true), (3, false), (4, true)].into_iter());
        assert!(!c.holds(1)); assert!(c.holds(2) && c.holds(4));
        c.begin([3, 5, 6].into_iter());
        c.refresh([(1, true), (2, true), (3, true), (4, true), (5, false), (6, false)].into_iter());
        assert!(!c.holds(2) && !c.holds(3) && !c.holds(4));
        assert!(c.holds(5) && c.holds(6));
        // A cancelled prompt must not hold its completed peer.
        c.refresh([(1, true), (5, true)].into_iter()); assert!(!c.holds(5));
    }
    #[test]
    fn all_short_prompts_fit_the_funded_pass_without_changing_fixed_tiles() {
        assert_eq!(row_cap(1024, 8, true, 0), 128);
        assert_eq!(row_cap(1024, 4, true, 0), 256);
        assert_eq!(row_cap(1024, 8, false, 0), 1024);
        assert_eq!(row_cap(1024, 8, true, 256), 1024);
        assert_eq!(row_cap(4, 8, true, 0), 1);
    }
    #[test]
    fn late_prompts_keep_serial_turns_after_the_older_decoder_ends() {
        let mut c = Cohorts::default();
        c.refresh([(1, true), (2, false), (3, false)].into_iter());
        assert!(c.serial(2) && c.serial(3)); c.begin([2, 3].into_iter());
        assert!(!c.holds(2) && !c.holds(3));
        c.refresh([(2, false), (3, false)].into_iter()); assert!(c.serial(2) && c.serial(3));
        c.refresh([(2, true), (3, true)].into_iter()); assert!(!c.serial(2) && !c.serial(3));
        c.refresh([].into_iter()); c.begin([4, 5].into_iter());
        c.refresh([(4, true), (5, false)].into_iter());
        assert!(!c.decoding([(4, true), (5, false)].into_iter())); assert!(!c.serial(5));
    }
}
