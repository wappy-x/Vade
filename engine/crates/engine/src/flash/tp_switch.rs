//! VADE_W2_TP: the switch, wire kinds and split geometry of the two-rank tensor split
//! (flash::next_batched::tp), shared with the host-only pipeline protocol build.

/// A verify pass posted to rank 1: it records its GDN tape (pipeline KIND_VERIFY) and acks
/// without rows, like an asynchronous prompt tile (pipeline KIND_ROWS_ASYNC).
pub const KIND_VERIFY: u32 = 7;
/// Request kind bit: the pass's residual rows arrive in rank 1's in-graph mailbox; the
/// request's rows are zeros of the same length.
pub const MAIL: u32 = 1 << 30;
/// A handoff request: rank 1 swaps the named chats' session state with rank 0 (raw words over
/// the split's TCP link) after their prompt ran as pipeline tiles, and acks without rows.
pub const KIND_HANDOFF: u32 = 8;
/// Request kind bit: a prompt tile that runs as a pipeline tile (rank 1: layers 24..48).
pub const TILE: u32 = 1 << 29;

/// VADE_W2_TP (unset = 0): 1 selects the tensor split at world 2. Read once.
pub fn from_env() -> Result<bool, String> {
    static ON: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| parse(std::env::var("VADE_W2_TP").ok().as_deref())).clone()
}
fn parse(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("VADE_W2_TP must be 0 or 1".into()),
    }
}

/// The split's geometry, derived from the checkpoint: residual width (HC branches x hidden),
/// hidden width, routed-expert width (gate/up rows, down K), routed experts per token and the
/// expert count. Rank r owns half of the expert width and half of the hidden width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpLayout {
    pub hidden: usize,
    pub residual: usize,
    pub expert_width: usize,
    pub top_k: usize,
    pub experts: usize,
}
impl TpLayout {
    /// Rows a pass may have for its experts to split (decode, verify, short prompt tiles).
    pub const ROWS: usize = 16;
    pub fn new(hidden: u64, branches: u64, expert_width: u64, top_k: u64, experts: u64) -> Result<Self, String> {
        let n = |v: u64| usize::try_from(v).map_err(|_| "tensor split geometry overflow".to_string());
        let layout = Self { hidden: n(hidden)?, residual: n(hidden.checked_mul(branches).ok_or("tensor split geometry overflow")?)?,
            expert_width: n(expert_width)?, top_k: n(top_k)?, experts: n(experts)? };
        // Halves of whole K16 tiles and BF16 pairs; the finish covers 256 columns per CTA.
        if layout.hidden == 0 || !layout.hidden.is_multiple_of(32) || layout.expert_width == 0 || !layout.expert_width.is_multiple_of(32)
            || layout.top_k == 0 || layout.experts == 0 || branches == 0 {
            return Err(format!("tensor split needs even 16-column halves: hidden {hidden}, expert width {expert_width}, top-k {top_k}"));
        }
        Ok(layout)
    }
    /// Gate/up output columns and down output columns per rank.
    pub fn mid_cols(&self) -> usize { self.expert_width / 2 }
    pub fn out_cols(&self) -> usize { self.hidden / 2 }
    /// One BF16 middle row in 32-bit words (BF16 pairs).
    pub fn mid_words(&self) -> usize { self.expert_width / 2 }
    /// Gate output (BF16 pairs): rank r writes middle columns from mid_cols * r.
    pub fn mid(&self, rank: usize) -> u64 { (rank * self.mid_cols() * 2) as u64 }
    /// Down output and the finish's partials: rank r's columns from out_cols * r of each BF16
    /// slot row (a slot row is `hidden` FP32-sized words, its BF16 half the first hidden / 2).
    /// Rank r also writes the FP32 shared row this many bytes lower, so the shifted finish
    /// reads shared column c + out_cols for its column c (inside the shared slot's stride and
    /// the previous slot's unused upper half).
    pub fn routed(&self, rank: usize) -> u64 { (rank * self.out_cols() * 2) as u64 }
    /// Finish output: rank r's columns from out_cols * r of each FP32 MoE output row.
    pub fn out(&self, rank: usize) -> u64 { (rank * self.out_cols() * 4) as u64 }
    /// A world-one launch of `whole` CTAs over `full` output columns, scaled to `own` columns.
    pub fn share(&self, whole: u32, own: usize, full: usize) -> u32 { (whole as usize * own).div_ceil(full).max(1) as u32 }
    /// Finish CTAs (256 columns each) over one rank's columns.
    pub fn finish_ctas(&self) -> u32 { self.out_cols().div_ceil(256) as u32 }
    /// Rank 1's descriptor shift: gate/up start at row mid_cols (K = hidden), down at row
    /// out_cols (K = expert_width); FP4 nibbles take k / 2 bytes per row, E4M3 scales k / 16.
    pub fn gate_shift(&self) -> (usize, usize) { (self.mid_cols() * self.hidden / 2, self.mid_cols() * (self.hidden / 16)) }
    pub fn down_shift(&self) -> (usize, usize) { (self.out_cols() * self.expert_width / 2, self.out_cols() * (self.expert_width / 16)) }
    /// The largest mailbox message: ROWS rows of the residual (the layer-2 post).
    pub fn slot_bytes(&self) -> usize { Self::ROWS * self.residual * 4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// flash_next_k16_finish's addressing (slot rows of `hidden` FP32-sized words, routed BF16
    /// at u16 index row*2*(top_k+1)*hidden + rank*2*hidden + col, shared FP32 at word
    /// (row*(top_k+1)+top_k)*hidden + col, output at row*hidden + col) and fp4_bw_down's
    /// (u32 word slot*hidden + col/2) with rank r's offsets: the finish reads exactly the
    /// columns rank r wrote, and rank 1's shifted shared row stays clear of routed columns.
    /// Run for the shipped checkpoint geometry and a different one, so nothing is fixed.
    #[test]
    fn rank_offsets_address_each_ranks_columns_for_any_geometry() {
        for layout in [TpLayout::new(2560, 4, 640, 10, 512).unwrap(), TpLayout::new(4096, 2, 1536, 8, 128).unwrap()] {
            let (h, k) = (layout.hidden, layout.top_k);
            let slots = k + 1;
            for rank in 0..2 {
                let (routed_u16, routed_words, out_words) = (layout.routed(rank) as usize / 2, layout.routed(rank) as usize / 4, layout.out(rank) as usize / 4);
                assert_eq!(layout.routed(rank) % 4, 0);
                for row in [0usize, 3, TpLayout::ROWS - 1] {
                    for slot in 0..k {
                        for c in [0usize, 1, 255, layout.out_cols() - 2] {
                            let pair = ((row * slots + slot) * h + c / 2) * 2 + c % 2 + routed_u16;
                            assert_eq!(pair, (row * slots + slot) * 2 * h + rank * layout.out_cols() + c);
                            assert_eq!(routed_u16 + row * 2 * slots * h + slot * 2 * h + c, pair);
                            assert!(rank * layout.out_cols() + c < h, "inside the slot's BF16 half");
                        }
                    }
                    for c in [0usize, 7, layout.out_cols() - 1] {
                        let read = routed_words + row * slots * h + c + k * h;
                        assert_eq!(read + routed_words - (row * slots + k) * h, rank * layout.out_cols() + c);
                        assert_eq!(out_words + row * h + c, row * h + rank * layout.out_cols() + c);
                    }
                    let first = (row * slots + k) * h - routed_words;
                    assert!(rank == 0 || first >= (row * slots + k - 1) * h + h / 2);
                }
                assert_eq!(layout.mid(rank) as usize / 4, rank * layout.mid_cols() / 2);
            }
            assert!(layout.finish_ctas() as usize * 256 >= layout.out_cols());
            assert_eq!(layout.share(40, layout.mid_cols(), layout.expert_width), 20);
            assert_eq!(layout.share(160, layout.out_cols(), layout.hidden), 80);
            assert_eq!(layout.share(5, 1, 3), 2);
            assert_eq!(layout.gate_shift().0, layout.mid_cols() * h / 2);
            assert_eq!(layout.down_shift().1, layout.out_cols() * layout.expert_width / 16);
            assert_eq!(layout.slot_bytes(), TpLayout::ROWS * layout.residual * 4);
        }
        assert!(TpLayout::new(2560, 4, 650, 10, 512).is_err());
        assert!(TpLayout::new(2570, 4, 640, 10, 512).is_err());
        assert!(TpLayout::new(2560, 0, 640, 10, 512).is_err());
    }
    #[test]
    fn switch_accepts_only_zero_one_or_unset() {
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("0")), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert!(parse(Some("yes")).is_err());
        assert!(parse(Some("")).is_err());
    }
}
