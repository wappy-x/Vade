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

pub(super) fn prompt_cohort_candidate(remaining: usize, position: usize, limit: usize, chunk_rows: usize) -> bool {
    remaining <= limit || chunk_rows != 0 && position.is_multiple_of(chunk_rows)
}

/// A lone long prompt uses its wider, serial prefill groups.
pub(super) fn prompt_uses_cohort(remaining: usize, position: usize, limit: usize, chunk_rows: usize, ready: usize) -> bool {
    prompt_cohort_candidate(remaining, position, limit, chunk_rows) && (remaining <= limit || ready >= 2)
}

/// What the whole-read rules read of one runnable job this tick.
#[derive(Clone, Copy)]
pub(super) struct WholeJob {
    pub rows: usize,
    pub read: usize,
    pub open: bool,
    pub serial: bool,
}

/// Use the unsliced read only when another arrival has no admission slot and
/// no admitted prompt would be left behind. Ordinary prefill handles the rest.
pub(super) fn whole_fits(
    cap: usize,
    chats: usize,
    slots: usize,
    limit: usize,
    jobs: impl ExactSizeIterator<Item = WholeJob>,
) -> bool {
    if jobs.len() != slots || !(crate::MIN_WHOLE_PREFILL_COHORT..=chats).contains(&slots) {
        return false;
    }
    let mut budget = cap;
    for job in jobs {
        if !job.open || job.serial || job.read != 0 || !(1..=limit.min(budget)).contains(&job.rows) {
            return false;
        }
        budget -= job.rows;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{Cohorts, WholeJob, row_cap, whole_fits};

    #[test]
    fn whole_read_requires_all_live_slots_and_all_rows() {
        let jobs = [105, 105, 105, 104, 114, 114, 91, 91].map(|rows| WholeJob {
            rows, read: 0, open: true, serial: false,
        });
        assert!(whole_fits(829, 8, 8, 256, jobs.into_iter()));
        assert!(!whole_fits(828, 8, 8, 256, jobs.into_iter()));
        assert!(!whole_fits(1024, 7, 8, 256, jobs.into_iter()));
        assert!(!whole_fits(1024, 8, 8, 256, jobs[..7].iter().copied()));
        let card = jobs.map(|job| WholeJob { rows: 132, ..job });
        assert!(!whole_fits(1024, 8, 8, 256, card.into_iter()));
        assert!(!whole_fits(1024, 8, 8, 256, jobs[..1].iter().copied()));
    }

    #[test]
    fn whole_read_does_not_leave_a_decoder_or_serial_prompt_behind() {
        let cold = WholeJob { rows: 100, read: 0, open: true, serial: false };
        for blocked in [
            WholeJob { read: 100, ..cold },
            WholeJob { read: 1, ..cold },
            WholeJob { serial: true, ..cold },
            WholeJob { open: false, ..cold },
            WholeJob { rows: 257, ..cold },
        ] {
            let mut jobs = [cold; 8];
            jobs[7] = blocked;
            assert!(!whole_fits(1024, 8, 8, 256, jobs.into_iter()));
        }
    }

    #[test]
    fn a_cancelled_slot_does_not_group_late_arrivals_beside_a_decoder() {
        let cold = WholeJob { rows: 100, read: 0, open: true, serial: false };
        let mut jobs = [cold; 8];
        assert!(!whole_fits(1024, 8, 8, 256, jobs[..7].iter().copied()));
        let mut cohorts = Cohorts::default();
        cohorts.refresh((0..8).map(|id| (id, id == 0)));
        jobs[0].read = 100;
        for (id, job) in jobs.iter_mut().enumerate() {
            job.serial = cohorts.serial(id as u64);
        }
        assert!(!whole_fits(1024, 8, 8, 256, jobs.into_iter()));
        // The decoder leaves. Already admitted late prompts still use the serial lane.
        cohorts.refresh((1..=8).map(|id| (id, false)));
        assert!(!cohorts.serial(8));
        let remaining = (1u8..=8).map(|id| WholeJob { serial: cohorts.serial(u64::from(id)), ..cold });
        assert!(!whole_fits(1024, 8, 8, 256, remaining));
    }
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
