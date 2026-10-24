//! Opt-in whole-cohort depth-one cost policy. No per-chat cost attribution.
#[derive(Default)]
struct Cost {
    seen: usize,
    ms: f64,
    useful: f64,
}
impl Cost {
    fn add(&mut self, ms: f64, useful: usize) {
        self.seen += 1;
        // First use may build a graph or verifier workspace. Measure a second
        // cycle before deciding between the two steady execution paths.
        if self.seen == 1 { return; }
        let weight = if self.seen == 2 { 1.0 } else { 0.25 };
        self.ms += weight * (ms - self.ms);
        self.useful += weight * (useful as f64 - self.useful);
    }
    fn per_token(&self) -> f64 { self.ms / self.useful }
}

const MIN_DWELL: usize = 8;
/// Plain must be this much cheaper per token before robust mode leaves draft.
/// 0.92 is an 8% gap: wider than try40 2-X round 1's 0.423% miss (plain lock
/// of 59 ticks) and inside the healthy sample-32 draft lead (23%+ tok/s).
const ROBUST_LEAVE: f64 = 0.92;
/// Once plain is preferred, draft wins back at a 2% per-token deficit or better.
const ROBUST_RETURN: f64 = 1.02;
/// Draft-preferred probes. A 256-token cohort ends near sample 160, so the
/// 64-sample plain probe (16 ticks on the healthy A rounds) does not fire.
const ROBUST_PROBE: usize = 256;

#[derive(Default)]
pub(crate) struct Controller {
    ids: Vec<u64>,
    plain: Cost,
    draft: Cost,
    prefer_draft: bool,
    samples: usize,
    last_probe: usize,
    survived: bool,
    head_miss: bool,
    tick: usize,
    choice: Option<bool>,
    switched_at: usize,
    plain_until: usize,
    robust: bool,
}
impl Controller {
    /// Measured-benefit admission. Legacy `Default` keeps the live 2% latch.
    pub(crate) fn robust() -> Self {
        Self { robust: true, prefer_draft: true, ..Self::default() }
    }
    pub(crate) fn is_robust(&self) -> bool { self.robust }
    pub(crate) fn reset(&mut self) {
        let robust = self.robust;
        *self = Self::default();
        self.robust = robust;
        if robust { self.prefer_draft = true; }
    }
    /// An incomplete tick is not a new membership. Preserve measured costs,
    /// and require a stable plain interval before touching pooled storage again.
    pub(crate) fn suspend(&mut self) {
        self.tick = self.tick.saturating_add(1);
        self.plain_until = self.tick.saturating_add(MIN_DWELL);
        self.apply(false, true);
    }
    /// Fairness changes which rows run, not the cohort's selected path. Do not
    /// learn from a partial tick, force a switch, or authorize storage release.
    pub(crate) fn fairness_skip(&mut self, ids: &[u64]) -> bool {
        if self.ids != ids { self.suspend(); return false; }
        self.tick = self.tick.saturating_add(1);
        self.choice.unwrap_or(false)
    }
    fn apply(&mut self, wanted: bool, safety: bool) -> bool {
        if self.choice.is_some_and(|current| current != wanted) && !safety
            && self.tick.saturating_sub(self.switched_at) < MIN_DWELL {
            return self.choice.unwrap();
        }
        if self.choice != Some(wanted) {
            self.choice = Some(wanted);
            self.switched_at = self.tick;
            eprintln!("cohort_path tick={} draft={wanted} safety={safety}", self.tick);
        }
        wanted
    }
    /// IDs cover every participating ready decoder. Only a real membership
    /// change resets costs; transient fairness/prefill/socket skips suspend.
    pub(crate) fn select(&mut self, ids: &[u64], head_miss: bool) -> bool {
        self.tick = self.tick.saturating_add(1);
        if self.ids != ids {
            let dwell = (self.tick, self.choice, self.switched_at, self.plain_until);
            self.reset();
            (self.tick, self.choice, self.switched_at, self.plain_until) = dwell;
            self.ids.extend_from_slice(ids);
            self.head_miss = head_miss;
        }
        // Coverage is sampled at membership entry from every live canonical
        // next token, including already-created speculators.
        if self.head_miss || self.tick < self.plain_until {
            return self.apply(false, true);
        }
        let interval = if self.robust && self.prefer_draft && self.survived { ROBUST_PROBE }
            else if self.survived { 64 }
            else { 2048 };
        let probe = self.draft.seen >= 2 && self.plain.seen >= 2
            && self.samples.saturating_sub(self.last_probe) >= interval;
        let wanted = if self.draft.seen < 2 { true }
            else if self.plain.seen < 2 { false }
            else if probe { !self.prefer_draft }
            else { self.prefer_draft };
        let chosen = self.apply(wanted, false);
        if probe && chosen == wanted { self.last_probe = self.samples; }
        chosen
    }
    /// Inclusive prepare + proposal + target + rollback + publication time.
    /// With depth one, useful outputs are exactly chats plus accepted first
    /// proposals. Reject incomplete/terminal/mixed-width samples at the caller.
    pub(crate) fn observe(&mut self, drafted: bool, ms: f64, committed: usize,
        proposed: usize, accepted: usize) -> bool
    {
        let chats = self.ids.len();
        if !(6..=8).contains(&chats) || !ms.is_finite() || ms <= 0.0
            || proposed > chats || accepted > proposed
            || committed != chats + accepted
            || !drafted && (proposed != 0 || accepted != 0)
        { return false; }
        if drafted { self.draft.add(ms, chats + accepted); }
        else { self.plain.add(ms, chats); }
        self.survived |= accepted != 0;
        self.samples += 1;
        if self.plain.seen >= 2 && self.draft.seen >= 2 {
            let (plain, draft) = (self.plain.per_token(), self.draft.per_token());
            if self.robust {
                if self.prefer_draft {
                    if plain < draft * ROBUST_LEAVE { self.prefer_draft = false; }
                } else if draft < plain * ROBUST_RETURN {
                    self.prefer_draft = true;
                }
            } else if draft < plain * 0.98 {
                // Require a 2% advantage before changing the preferred path.
                self.prefer_draft = true;
            } else if plain < draft * 0.98 {
                self.prefer_draft = false;
            }
        }
        self.samples <= 4 || self.samples.is_multiple_of(32)
    }
    pub(crate) fn rates(&self) -> (usize, f64, f64) {
        (self.samples, 1000.0 / self.plain.per_token(), 1000.0 / self.draft.per_token())
    }
    pub(crate) fn head_miss(&self) -> bool { self.head_miss }
}

#[cfg(test)]
mod tests {
    use super::*;
    const IDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    fn step(c: &mut Controller, draft_ms: f64, accepted: usize) -> bool {
        let draft = c.select(&IDS, false);
        c.observe(draft, if draft { draft_ms } else { 56. },
            8 + if draft { accepted } else { 0 }, if draft { 8 } else { 0 },
            if draft { accepted } else { 0 });
        draft
    }
    fn warm(c: &mut Controller, draft_ms: f64, accepted: usize) {
        for i in 0..2*MIN_DWELL { assert_eq!(step(c, draft_ms, accepted), i < MIN_DWELL); }
    }
    #[test]
    fn whole_cycle_cost_and_useful_outputs_decide_after_dwell() {
        let mut fast = Controller::default(); warm(&mut fast, 96., 7);
        assert!(fast.select(&IDS, false));
        let mut slow = Controller::default(); warm(&mut slow, 120., 7);
        assert!(!slow.select(&IDS, false));
        let mut rejected = Controller::default(); warm(&mut rejected, 96., 1);
        assert!(!rejected.select(&IDS, false));
    }
    #[test]
    fn normal_changes_and_probe_paths_dwell_for_at_least_eight_ticks() {
        let mut c = Controller::default(); let mut last = None; let mut changed = 0;
        for tick in 0..320 {
            let selected = step(&mut c, 96., 7);
            if last.is_some_and(|v| v != selected) { assert!(tick - changed >= MIN_DWELL); changed = tick; }
            last = Some(selected);
        }
        assert!(c.plain.seen > MIN_DWELL); // periodic remeasurement occurred
    }
    #[test]
    fn fairness_skips_every_two_to_four_ticks_keep_the_selected_path() {
        for period in 2..=4 {
            let mut c = Controller::default(); warm(&mut c, 96., 7);
            assert!(c.select(&IDS, false)); let samples = c.samples;
            for tick in 0..32 {
                if tick % period == 0 { assert!(c.fairness_skip(&IDS)); }
                else { assert!(step(&mut c, 96., 7)); }
            }
            assert!(c.samples > samples); assert!(c.draft.seen >= MIN_DWELL);
            // A genuinely ineligible request still suspends safely.
            c.suspend();
            for _ in 1..MIN_DWELL { assert!(!step(&mut c, 96., 7)); }
            assert!((0..MIN_DWELL).any(|_| step(&mut c, 96., 7)));
            let mut plain=Controller::default(); warm(&mut plain,120.,7);
            assert!(!plain.fairness_skip(&IDS));
        }
    }
    #[test]
    fn membership_resets_costs_and_includes_existing_speculator_head_misses() {
        let mut c = Controller::default(); warm(&mut c, 96., 7);
        assert!(!c.select(&[1,2,3,4,5,6,7], true));
        assert_eq!(c.samples, 0); assert!(c.head_miss());
        for _ in 0..4096 { assert!(!c.select(&[1,2,3,4,5,6,7], false)); }
        assert!(c.select(&IDS, false)); assert!(!c.head_miss());
    }
    #[test]
    fn zero_survival_defers_new_probes_until_2048_completed_samples() {
        let mut c = Controller::default(); warm(&mut c, 96., 0);
        while c.samples < 2048 { assert!(!step(&mut c, 96., 0)); }
        assert!(step(&mut c, 96., 1)); assert!(c.survived);
        for _ in 1..MIN_DWELL { assert!(step(&mut c, 96., 1)); }
        assert!(!step(&mut c, 96., 1));
    }
    /// Safety plain (what a real cohort does before its first draft) fills plain.seen.
    fn after_safety(c: &mut Controller, draft_ms: f64, accepted: usize) {
        c.suspend();
        for _ in 0..MIN_DWELL { let _ = step(c, draft_ms, accepted); }
    }
    #[test]
    fn robust_keeps_a_sub_percent_miss_on_draft_and_still_leaves_a_clear_plain_win() {
        // 105.525 ms / 15 tokens is 0.5% worse per token than plain 56 ms / 8.
        // Legacy stays plain: prefer_draft starts false and 0.5% is inside 2%.
        let mut legacy = Controller::default();
        after_safety(&mut legacy, 105.525, 7);
        let mut saw_draft = false;
        let mut back_to_plain = false;
        for _ in 0..24 {
            let draft = step(&mut legacy, 105.525, 7);
            if draft { saw_draft = true; }
            else if saw_draft { back_to_plain = true; }
        }
        assert!(saw_draft && back_to_plain);
        assert!(!legacy.select(&IDS, false));
        // Robust returns to draft after the safety window and does not take the 64-sample probe.
        let mut tie = Controller::robust();
        after_safety(&mut tie, 105.525, 7);
        let mut drafted = 0;
        for _ in 0..80 {
            if step(&mut tie, 105.525, 7) { drafted += 1; }
        }
        assert!(drafted >= 70, "drafted {drafted}");
        assert!(tie.is_robust());
        // 120 ms / 15 = 8.0 vs plain 7.0: plain is 12.5% cheaper, past the 8% leave margin.
        let mut bad = Controller::robust();
        after_safety(&mut bad, 120., 7);
        let mut left = false;
        for _ in 0..24 {
            if !step(&mut bad, 120., 7) { left = true; }
        }
        assert!(left);
        assert!(!bad.select(&IDS, false));
    }
    #[test]
    fn robust_membership_reset_keeps_the_hysteresis() {
        let mut c = Controller::robust();
        after_safety(&mut c, 105.525, 7);
        for _ in 0..MIN_DWELL { assert!(step(&mut c, 105.525, 7)); }
        let ids = [11, 12, 13, 14, 15, 16, 17, 18];
        c.select(&ids, false);
        c.suspend();
        for _ in 0..MIN_DWELL { let _ = step_ids(&mut c, &ids, 105.525, 7); }
        let mut drafted = 0;
        for _ in 0..40 {
            let draft = c.select(&ids, false);
            c.observe(draft, if draft { 105.525 } else { 56. },
                8 + if draft { 7 } else { 0 }, if draft { 8 } else { 0 }, if draft { 7 } else { 0 });
            if draft { drafted += 1; }
        }
        assert!(drafted >= 30, "drafted {drafted}");
        assert!(c.is_robust());
    }
    fn step_ids(c: &mut Controller, ids: &[u64], draft_ms: f64, accepted: usize) -> bool {
        let draft = c.select(ids, false);
        c.observe(draft, if draft { draft_ms } else { 56. },
            8 + if draft { accepted } else { 0 }, if draft { 8 } else { 0 },
            if draft { accepted } else { 0 });
        draft
    }
    #[test]
    fn reject_partial_samples_and_exclude_first_setup_cost() {
        let mut c = Controller::default(); assert!(c.select(&IDS, false));
        for (ms, committed, proposed, accepted) in [(f64::NAN,8,0,0),(0.,8,0,0),(56.,7,0,0),(56.,16,16,8),(56.,9,0,1)] {
            c.observe(false, ms, committed, proposed, accepted); assert_eq!(c.samples, 0);
        }
        c.observe(false,4000.,8,0,0); c.observe(false,56.,8,0,0);
        c.observe(true,4000.,15,8,7); c.observe(true,96.,15,8,7);
        assert_eq!(c.plain.ms,56.); assert_eq!(c.draft.ms,96.);
    }
}
