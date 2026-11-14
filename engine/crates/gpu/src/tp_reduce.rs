//! The REDUCE of the pair split, device side: one exchange a sublayer. The put sends this owner's partial sum with
//! the finite put's kernels. The take waits for the peer's partial and adds the two in owner order
//! (`tp_reduce_take`; CPU twin `tc_twin_fold::reduce_word`). Both owners then hold the same words.
//! It runs on both mailbox forms. The engine's mailbox (two writes a message): the rows start at the slot's first
//! row word and the sequence word is the header's (SEND_SEQ here, RECV_SEQ at the peer), as every gather of today.
//! A mailbox of the INLINE wire (tp_mailbox_host::TRAILER, one write a message): the rows end at the slot's trailer.
//! In both the wire writes the receive sequence word after every row, which is all the take's kernel needs.
use super::*;
use crate::pair_split::{REDUCE_LOWER, REDUCE_ROUND, REDUCE_SCALE, THREADS};
use crate::tp_mailbox_host::TRAILER;

impl<'a> Backend<'a> {
    /// The words of a REDUCE message, checked: a whole contiguous vector of this Backend, a multiple of four words on
    /// a 16-byte line (the take moves four words at a time), that fits a slot of this mailbox.
    fn reduce_words(&self, m: &TpMailbox<'_>, partial: &DeviceVector<'_>) -> Result<usize, String> {
        if let Some(error) = self.fatal_error() { return Err(error); }
        self.check_vector(&m.vector, m.vector.len).and_then(|()| self.check_vector(partial, partial.len()))
            .map_err(|_| "tensor split mailbox or partial sum belongs to another backend".to_string())?;
        let words = partial.len();
        if partial.is_segmented() || words == 0 || !words.is_multiple_of(4) || !partial.device_address().is_multiple_of(16) || words > m.slot_bytes() / 4 {
            return Err("REDUCE partial sum outside the slot".into());
        }
        Ok(words)
    }
    /// Queue the put of REDUCE message `seq`: every word of `partial` goes to the send slot (at its first row word; on
    /// an inline mailbox ending at its trailer).
    /// A nonfinite word sets the sticky wire error and the message is not published (the finite put's rule).
    pub fn tp_reduce_put(&self, m: &TpMailbox<'_>, partial: &DeviceVector<'_>, seq: u64) -> Result<(), String> {
        let words = self.reduce_words(m, partial)?;
        let slot = seq as usize % m.layout.slots;
        let rows = m.device(m.layout.send_slot(slot) + if m.inline { m.layout.slot_words - TRAILER - words } else { m.layout.frame });
        if crate::tp_put_wide::admitted(1, words) && !PUT_NARROW.load(Ordering::Relaxed) {
            let most = match m.words()[TAKE_CTAS].load(Ordering::Relaxed) { 0 => 96, n => n.min(96) };
            let mut args = [partial.device_address(), words as u64, rows, m.device(SEND_LEN + slot), m.device(SEND_SEQ + slot),
                (seq as u32) as u64 | m.wide() << 32, m.device(ERROR), m.done.device_address()];
            self.tp_mailbox_launch("tp_mailbox_put_wide", crate::tp_put_wide::blocks(words, most), 1, crate::tp_put_wide::THREADS, &mut args)?;
        } else {
            let mut args = [partial.device_address(), 1, words as u64, words as u64, 0, rows, m.device(SEND_LEN + slot), m.device(SEND_SEQ + slot),
                seq as u32 as u64, m.device(ERROR)];
            self.tp_mailbox_launch("tp_mailbox_put_finite", 1, 1, 1024, &mut args)?;
        }
        self.resident_mailbox_queued(&m.bell, seq);
        Ok(())
    }
    /// Queue the take of REDUCE message `seq` from receive channel `channel`: wait (at most `limit_ns`) for the
    /// peer's partial sum, then every word of `output` is fl(lower + higher) of the two partials, where `lower` says
    /// that `partial` (this owner's) is the lower owner's; then times `scale` when given; then rounded to BF16 when
    /// `round`. A failed take writes no word of `output` and sets the sticky wire error.
    #[allow(clippy::too_many_arguments)]
    pub fn tp_reduce_take(&self, m: &TpMailbox<'_>, output: &mut DeviceVector<'_>, partial: &DeviceVector<'_>, lower: bool, scale: Option<f32>, round: bool,
        channel: usize, seq: u64, limit_ns: u64) -> Result<(), String> {
        let words = self.reduce_words(m, partial)?;
        self.check_vector(output, output.len())?;
        let inside = crate::scatter_copy::channel_inside(channel, m.layout.world);
        if output.is_segmented() || output.len() != words || !output.device_address().is_multiple_of(16) || !inside { return Err("REDUCE output or channel differs".into()); }
        let ends = |v: &DeviceVector<'_>| v.device_address().checked_add((v.len() as u64).checked_mul(4)?);
        let (output_end, partial_end) = (ends(output).ok_or("REDUCE output address overflow")?, ends(partial).ok_or("REDUCE partial address overflow")?);
        if partial.device_address() < output_end && output.device_address() < partial_end { return Err("REDUCE output and partial sum overlap".into()); }
        let slot = seq as usize % m.layout.slots;
        let end = m.layout.recv_slot(channel, slot) + m.layout.slot_words;
        // The peer's rows and the word the wire writes after them.
        let (peer, seq_word) = if m.inline { (end - TRAILER - words, end - 1) } else { (m.layout.recv_slot(channel, slot) + m.layout.frame, m.layout.recv_seq(channel, slot)) };
        if !m.device(peer).is_multiple_of(16) { return Err("REDUCE receive rows are not on a 16-byte line".into()); }
        let limit = limit_ns.min((1 << 47) - 1) | u64::from(m.words()[TAKE_PAUSE].load(Ordering::Relaxed) & 0xffff) << 48;
        let mode = if lower { REDUCE_LOWER } else { 0 } | if round { REDUCE_ROUND } else { 0 }
            | scale.map_or(0, |scale| REDUCE_SCALE | u64::from(scale.to_bits()) << 32);
        let mut args = [output.device_address(), partial.device_address(), words as u64, m.device(peer), m.device(seq_word),
            (seq as u32) as u64, m.device(ERROR), limit, mode];
        let most = match m.words()[TAKE_CTAS].load(Ordering::Relaxed) { 0 => 96, n => n.min(96) as usize };
        crate::tp_mailbox_host::REDUCE_TAKES.fetch_add(1, Ordering::Relaxed);
        self.tp_mailbox_launch("tp_reduce_take", crate::pair_split::blocks(words).clamp(1, most) as u32, 1, THREADS, &mut args)
    }
}
