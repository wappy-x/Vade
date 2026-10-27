//! The tensor split: a GDN layer's mixer for one tensor split
//! rank's heads. Rank r owns value heads [r V/2, (r + 1) V/2) and key heads [r K/2, (r + 1) K/2)
//! (the value/key repeat is unchanged). The kernels already run on a rank-local head range
//! (key_start, value_start over compact operands), so a half runs today's launches, per head the
//! same words, over its own qkv rows [q | k | v], its conv weights and a state holding only its
//! heads. That state lives in the chat's existing GdnState vectors as a prefix (resized within
//! their storage): to_half compacts a whole state, to_whole rebuilds it from both ranks' halves.
//! Split from gdn.rs (file-size ratchet).
use super::{bf16, chunk_form, Backend, DeviceVector, Gdn, GdnState, PhantomData, Result, Weights, CHUNK, DIM, HIDDEN};

/// A whole state's runs as (words per head, heads): S, then in chunk form the open chunk's key
/// rows, value rows and gamma (gdn_params::chunk_state_floats). Each run splits at heads / 2.
fn state_runs(heads: usize, keys: usize) -> Vec<(usize, usize)> {
    let mut runs = vec![(DIM * DIM, heads)];
    if chunk_form() && heads != 0 { runs.extend([(CHUNK * DIM, keys), (CHUNK * DIM, heads), (CHUNK, heads)]); }
    runs
}
/// The conv history [channels][4] (and the conv weight, [channels][4] BF16): q, k, v channel runs.
pub(in crate::flash) fn channel_runs(heads: usize, keys: usize, per_channel: usize) -> [(usize, usize); 3] {
    [(DIM * per_channel, keys), (DIM * per_channel, keys), (DIM * per_channel, heads)]
}
/// Rank `rank`'s share of each run as (whole offset, half offset, words).
pub(in crate::flash) fn shares(runs: &[(usize, usize)], rank: usize) -> Vec<(usize, usize, usize)> {
    let (mut whole, mut half) = (0, 0);
    runs.iter().map(|&(per, n)| {
        let words = per * (n / 2);
        let share = (whole + rank * words, half, words);
        whole += per * n;
        half += words;
        share
    }).collect()
}
fn total(runs: &[(usize, usize)]) -> usize { runs.iter().map(|&(per, n)| per * n).sum() }

/// Grow `v` to `len` words within its storage, or give it fresh storage (an entry snapshot
/// allocated while its chat was a half holds only a half's words).
fn grow<'a>(b: &Backend<'a>, v: &mut DeviceVector<'a>, len: usize) -> Result<()> {
    if v.resize_within_capacity(len).is_err() { *v = b.vector(len)?; }
    Ok(())
}

impl<'a> GdnState<'a> {
    pub(in crate::flash) fn is_half(&self) -> bool { self.half }
    /// Words of this rank's half (state, history) of a layer of `heads` value and `keys` key heads.
    pub(in crate::flash) fn half_words(heads: usize, keys: usize) -> [usize; 2] {
        [total(&state_runs(heads, keys)) / 2, total(&channel_runs(heads, keys, 4)) / 2]
    }
    /// Whole -> this rank's half, compacted to the front of the same vectors through `temp`.
    /// `copy` false (an entry snapshot, rewritten by the next verify before any use) only reshapes.
    pub(in crate::flash) fn to_half(&mut self, b: &Backend<'a>, rank: usize, heads: usize, keys: usize,
        temp: &mut DeviceVector<'a>, copy: bool) -> Result<()> {
        if self.half { return Ok(()); }
        let runs = [state_runs(heads, keys), channel_runs(heads, keys, 4).to_vec()];
        for (v, runs) in [&mut self.state, &mut self.history].into_iter().zip(&runs) {
            if v.len() != total(runs) { return Err("GDN state is not whole for the head split".into()); }
            let shares = shares(runs, rank);
            let words = total(runs) / 2;
            if copy {
                for &(whole, half, n) in &shares { b.copy_range(v, temp, whole, half, n)?; }
                b.copy_range(temp, v, 0, 0, words)?;
            }
            v.resize_within_capacity(words)?;
        }
        self.half = true;
        Ok(())
    }
    /// This rank's half -> whole: its own heads from the vectors (through `temp`) and the peer
    /// rank's from `peer` (the peer's half state words, then its half history words). `peer` None
    /// only reshapes (an entry snapshot, or a chat restarting from position 0, zeroed by the caller).
    pub(in crate::flash) fn to_whole(&mut self, b: &Backend<'a>, rank: usize, heads: usize, keys: usize,
        temp: &mut DeviceVector<'a>, peer: Option<&[f32]>) -> Result<()> {
        if !self.half { return Ok(()); }
        let runs = [state_runs(heads, keys), channel_runs(heads, keys, 4).to_vec()];
        let mut at = 0usize;
        for (v, runs) in [&mut self.state, &mut self.history].into_iter().zip(&runs) {
            let words = total(runs) / 2;
            if v.len() != words { return Err("GDN half state shape".into()); }
            let Some(peer) = peer else { grow(b, v, total(runs))?; continue };
            let theirs = peer.get(at..at + words).ok_or("GDN peer half shorter than the state")?;
            at += words;
            b.copy_range(v, temp, 0, 0, words)?;
            grow(b, v, total(runs))?;
            for &(whole, half, n) in &shares(runs, rank) { b.copy_range(temp, v, half, whole, n)?; }
            for &(whole, half, n) in &shares(runs, 1 - rank) { b.write_vector_range(v, whole, &theirs[half..half + n])?; }
        }
        if peer.is_some_and(|p| p.len() != at) { return Err("GDN peer half longer than the state".into()); }
        self.half = false;
        Ok(())
    }
}

impl<'a, 'w> Gdn<'a, 'w> {
    /// This layer's mixer for rank `rank`'s heads over a half state (GdnState::to_half); `conv`
    /// is the rank's compact conv weight ([own channels][4] BF16, channel_runs order). The tensor
    /// split launches its in-projections into the scratch at this half's row widths (qkv, z, a
    /// and b stay 0 here); `out` is the whole out projection, read over the gathered rows.
    pub(in crate::flash) fn load_half(w: &'w Weights<'_>, layer: usize, out: u64, conv: u64, rank: usize,
        cache: GdnState<'a>) -> Result<Self> {
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        let [state, history] = GdnState::half_words(heads, keys);
        if !cache.half || heads % 2 != 0 || keys % 2 != 0 || cache.layer != layer || cache.rank != w.plan.rank
            || cache.state.len() != state.max(1) || cache.history.len() != history.max(1) {
            return Err("GDN half state does not belong to this layer and rank".into());
        }
        let (h, k) = (heads / 2, keys / 2);
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        let own = |part: &str| bf16(w, &format!("{p}.{part}"), &[heads as u64], heads).map(|a| a + (rank * h * 2) as u64);
        Ok(Self { hidden: HIDDEN, precision: false, parameter_precision: 0, norm: 0, epsilon: 1e-6,
            layer, rank: w.plan.rank, heads: h, keys: k, key_start: w.plan.key_heads.start as usize + rank * k,
            value_start: w.plan.value_heads.start as usize + rank * h, qkv: 0, z: 0, a: 0, b: 0, conv,
            alog: own("A_log")?, dt: own("dt_bias")?, out,
            state: cache.state, history: cache.history, cached: cache.cached, half: true, _weights: PhantomData })
    }
    /// Per-row replay tape widths [qkv, a, b] of this mixer's heads.
    pub(in crate::flash) fn tape_widths(&self) -> [usize; 3] { [(2 * self.keys + self.heads) * DIM, self.heads, self.heads] }
    pub(in crate::flash) fn is_half(&self) -> bool { self.half }
    /// Rank `rank`'s conv weights ([own q, k, v channels][4] BF16, the order load_half reads) from
    /// the whole layer's stored bytes `whole` ([channels][1][4] BF16) of `heads` value and `keys`
    /// key heads.
    pub(in crate::flash) fn conv_half(whole: &[u8], heads: usize, keys: usize, rank: usize) -> Result<Vec<u8>> {
        let runs = channel_runs(heads, keys, 8);
        if whole.len() != total(&runs) || heads % 2 != 0 || keys % 2 != 0 { return Err("GDN conv weight shape for the head split".into()); }
        Ok(shares(&runs, rank).into_iter().flat_map(|(at, _, n)| whole[at..at + n].iter().copied()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The two ranks' shares tile every run exactly once, and each half packs its runs in order.
    #[test]
    fn shares_tile_the_runs() {
        let runs = [(4usize, 6usize), (3, 2), (2, 6)];
        let mut seen = vec![0u8; total(&runs)];
        for rank in 0..2 {
            let mut next = 0;
            for (whole, half, n) in shares(&runs, rank) {
                assert_eq!(half, next);
                next += n;
                seen[whole..whole + n].iter_mut().for_each(|s| *s += 1);
            }
            assert_eq!(next, total(&runs) / 2);
        }
        assert!(seen.iter().all(|&s| s == 1));
        assert_eq!(shares(&runs, 1)[1], (24 + 3, 12, 3));
    }
    /// to_half's compaction of each rank and to_whole's rebuild from both halves, over the real
    /// state and history runs of a small head layout: every word returns to its place, and each
    /// rank's half holds exactly its own heads' words in run order.
    #[test]
    fn whole_to_halves_to_whole_restores_every_word() {
        let (heads, keys) = (6, 2);
        for runs in [state_runs(heads, keys), channel_runs(heads, keys, 4).to_vec()] {
            let whole: Vec<f32> = (0..total(&runs)).map(|i| i as f32).collect();
            let halves: Vec<Vec<f32>> = (0..2).map(|rank| {
                let mut half = vec![f32::NAN; total(&runs) / 2];
                for (at, to, n) in shares(&runs, rank) { half[to..to + n].copy_from_slice(&whole[at..at + n]); }
                half
            }).collect();
            assert!(halves.iter().all(|h| h.iter().all(|w| !w.is_nan())));
            // Rank 0's first run share is its own value heads of S: heads [0, 3).
            assert_eq!(halves[0][0], 0.0);
            assert_eq!(halves[1][0], (runs[0].0 * runs[0].1 / 2) as f32);
            for rank in 0..2 {
                let mut rebuilt = vec![f32::NAN; total(&runs)];
                for (at, from, n) in shares(&runs, rank) { rebuilt[at..at + n].copy_from_slice(&halves[rank][from..from + n]); }
                for (at, from, n) in shares(&runs, 1 - rank) { rebuilt[at..at + n].copy_from_slice(&halves[1 - rank][from..from + n]); }
                assert_eq!(rebuilt, whole);
            }
        }
        assert_eq!(GdnState::half_words(heads, keys), [total(&state_runs(heads, keys)) / 2, total(&channel_runs(heads, keys, 4)) / 2]);
    }
}
