//! Host reduction for vocabulary candidates and the sticky routing verdict.
type Result<T> = std::result::Result<T, String>;
pub(crate) const ROUTING_ERROR: &str = "Flash MoE routing rejected non-finite or out-of-range selection";
pub(crate) const ROUTING_RANK_MISMATCH: &str = "ROUTING_RANK_MISMATCH: Flash routing or survivor bitmap differs between ranks";
/// Output failure is recoverable on any rank; only u32::MAX is a routing vote.
pub(crate) const NONFINITE_LOGIT: u32 = u32::MAX - 1;
pub const VOCABULARY: usize = 248_320;

/// One opaque word in each rank's existing output gather. Decode has at most
/// sixteen fixed slots: low bits reject rows, high bits identify live slots.
/// Padding never changes vocabulary offsets, graph keys or collective shapes.
#[derive(Clone, Copy)]
pub(crate) struct RoutingVerdict {
    pub active: u16,
    pub rejected: u16,
}
impl RoutingVerdict {
    pub fn slots(rows: usize) -> Result<u16> {
        if !(1..=16).contains(&rows) { return Err("invalid Flash routing bitmap width".into()); }
        Ok((u16::MAX as u32 >> (16 - rows)) as u16)
    }
    pub fn word(self) -> u32 { (u32::from(self.active) << 16) | u32::from(self.rejected & self.active) }
    pub fn validate_survivors(rows: usize, active: u16, mtp_depth: usize) -> Result<()> {
        let slots = Self::slots(rows)?;
        if active & !slots != 0 { return Err("invalid Flash survivor mask".into()); }
        // Padded rows skip observe_mtp and its gather. Require the full cohort
        // before any state mutation so MTP keeps the same collective schedule.
        if mtp_depth != 0 && active != slots {
            return Err("Flash MTP requires a full survivor mask".into());
        }
        Ok(())
    }
    /// Complete the gather before inspecting any verdict. Every rank sees the
    /// same words and fails the entire step on disagreement, including a live
    /// slot mismatch. No rank can commit a survivor while its peer aborts.
    pub fn agree(words: impl IntoIterator<Item = u32>, rows: usize) -> Result<u16> {
        let slots = Self::slots(rows)?;
        let mut agreed = None;
        for word in words {
            let (active, rejected) = ((word >> 16) as u16, word as u16);
            if active & !slots != 0 || rejected & !active != 0 || agreed.is_some_and(|v| v != word) {
                return Err(ROUTING_RANK_MISMATCH.into());
            }
            agreed = Some(word);
        }
        let word = agreed.ok_or("missing Flash routing rank verdict")?;
        Ok((word >> 16) as u16 & !(word as u16))
    }
}

/// Every local winner carries its global vocabulary coordinate. Equal logits
/// select the lower global token, matching a full-vocabulary greedy reduction.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub token: u32,
    pub logit: f32,
}
impl Candidate {
    pub fn reduce(candidates: impl IntoIterator<Item = Self>) -> Result<Self> {
        let mut winner: Option<Self> = None;
        let mut invalid = false;
        let mut nonfinite = false;
        let (mut rejected, mut ranks) = (0, 0);
        for candidate in candidates {
            ranks += 1;
            if candidate.token == u32::MAX { rejected += 1; continue; }
            if candidate.token == NONFINITE_LOGIT { nonfinite = true; continue; }
            if !candidate.logit.is_finite() || candidate.token as usize >= VOCABULARY {
                invalid = true;
                continue;
            }
            if winner.is_none_or(|best| candidate.logit > best.logit
                || (candidate.logit == best.logit && candidate.token < best.token)) {
                winner = Some(candidate);
            }
        }
        if rejected != 0 {
            return Err(if rejected == ranks { ROUTING_ERROR } else { ROUTING_RANK_MISMATCH }.into());
        }
        if nonfinite { return Err(ROUTING_ERROR.into()); }
        if invalid { return Err("invalid Flash vocabulary candidate".into()); }
        winner.ok_or_else(|| "no Flash vocabulary candidate".into())
    }
}

#[cfg(test)]
#[test]
fn routing_rejection_survives_candidate_gather() {
    for tokens in [[0, u32::MAX], [u32::MAX, 0]] { assert_eq!(Candidate::reduce(tokens.map(|token| Candidate { token, logit: f32::NAN })).unwrap_err(), ROUTING_RANK_MISMATCH); }
    assert_eq!(Candidate::reduce([Candidate { token: u32::MAX, logit: 0.0 }; 2]).unwrap_err(), ROUTING_ERROR);
    for tokens in [[0, NONFINITE_LOGIT], [NONFINITE_LOGIT, 0], [NONFINITE_LOGIT; 2]] { assert_eq!(Candidate::reduce(tokens.map(|token| Candidate { token, logit: 0.0 })).unwrap_err(), ROUTING_ERROR); }
    for rows in 1..=16 { let full = RoutingVerdict::slots(rows).unwrap(); for depth in 0..=3 { assert!(RoutingVerdict::validate_survivors(rows, full, depth).is_ok()); assert_eq!(RoutingVerdict::validate_survivors(rows, full & !1, depth).is_err(), depth != 0); } }
}

#[cfg(test)]
#[test]
fn padded_row_clear_preserves_sixteen_slot_survivors() {
    // Host coverage of the production byte range and routing word, not GPU math.
    let width = 4 * 2560;
    let full = RoutingVerdict::slots(16).unwrap();
    let clean = (0..16 * width).flat_map(|i| (i as u32 ^ 0x7fc0_0011).to_le_bytes()).collect::<Vec<_>>();
    let mut padded = clean.clone();
    assert_eq!(RoutingVerdict::agree([RoutingVerdict { active: full, rejected: 0 }.word(); 2], 16).unwrap(), full);
    let active = RoutingVerdict::agree([RoutingVerdict { active: full, rejected: 1 << 3 }.word(); 2], 16).unwrap();
    assert_eq!(active, 0xfff7);
    let range = spark_gpu_probe::vector_byte_range(16 * width, 3 * width..4 * width).unwrap();
    padded[range.clone()].fill(0);
    assert_eq!(RoutingVerdict::agree([RoutingVerdict { active, rejected: 1 << 3 }.word(); 2], 16).unwrap(), active);
    assert_eq!(&padded[..range.start], &clean[..range.start]);
    assert_eq!(&padded[range.end..], &clean[range.end..]);
    assert!(padded[range].iter().all(|&b| b == 0));
    for range in [0..16 * width + 1, usize::MAX..usize::MAX] {
        assert!(spark_gpu_probe::vector_byte_range(16 * width, range).is_err());
    }
}
