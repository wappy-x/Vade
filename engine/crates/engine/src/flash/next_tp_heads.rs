//! The GDN mixers of one-chat split passes run by heads (gdn_half.rs). Rank r
//! launches the in-projection rows of its own q, k and v heads and its z, a and b rows (one
//! grouped decode launch at the half's row widths plus one for a and b; each row keeps the whole
//! launch's chains and merge), runs the layer's conv, recurrence and output gate on its heads'
//! state, then the ranks swap their gated rows as raw bits (one message) and split the out
//! projection by row halves (tp_out_project, one message): two messages per GDN layer and no sum
//! crosses ranks. A chat's GDN states hold its rank's heads once rank 0's Policy switches it while
//! it runs one-chat split passes; tp_gdn_frame converts them at a pass's start, rank 1 following
//! rank 0's decision (the request's HEADS bit).
use super::*;

/// KIND_HANDOFF requests' per-chat length field (tile handoffs send 1): rebuild these chats' GDN
/// states whole; or run the Policy probe exchange (once per process).
pub(super) const GDN_WHOLE: u32 = 2;
pub(super) const GDN_PROBE: u32 = 3;

/// Level 3 operands and buffers of this rank.
pub(super) struct GdnHeads<'a> {
    /// Per layer: this rank's conv weights (Gdn::conv_half), on the split GDN layers only.
    conv: Vec<Option<DeviceVector<'a>>>,
    /// Staging of one half state for GdnState::to_half / to_whole.
    temp: Option<DeviceVector<'a>>,
    /// The gathered gated rows [ROWS][value heads x dim] the out projection reads.
    gated: Option<DeviceVector<'a>>,
    heads: usize,
    keys: usize,
    /// Rank 0: when a chat's states become halves.
    policy: Policy,
}

impl<'a> GdnHeads<'a> {
    pub(super) fn new(model: &NextModel<'a>, b: &Backend<'a>, rank: usize) -> Result<Self> {
        let (w, c) = (&model.weights, &model.checkpoint.config);
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        if keys == 0 || heads % keys != 0 || keys % 2 != 0 || c.linear_key_dim != c.linear_value_dim {
            return Err(format!("tensor split: GDN heads {heads} / keys {keys} have no head halves"));
        }
        let mut conv = Vec::with_capacity(LAYERS);
        for li in 0..LAYERS {
            if li < SPLIT || li % 4 == 3 { conv.push(None); continue; }
            let t = w.tensor(&format!("model.language_model.layers.{li}.linear_attn.conv1d.weight"))?;
            let mut whole = vec![0u8; t.bytes];
            w.read_arena(t.offset, &mut whole)?;
            let half = gdn::Gdn::conv_half(&whole, heads, keys, rank)?;
            let words: Vec<f32> = half.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let mut v = b.vector(words.len())?;
            b.write_vector(&mut v, &words)?;
            conv.push(Some(v));
        }
        let staging = gdn::GdnState::half_words(heads, keys).into_iter().max().unwrap_or(1);
        let value = (c.linear_value_heads * c.linear_value_dim) as usize;
        Ok(Self { conv, temp: Some(b.vector(staging)?), gated: Some(b.vector(TpLayout::ROWS * value)?), heads, keys, policy: Policy::default() })
    }
}

impl<'a> NextModel<'a> {
    /// A layer's mixer over a chat's state: rank `rank`'s heads when the state holds a half.
    pub(in crate::flash) fn gdn_load<'w>(&'w self, b: &Backend<'a>, li: usize, state: Option<gdn::GdnState<'a>>) -> Result<gdn::Gdn<'a, 'w>> {
        match state {
            Some(state) if state.is_half() => {
                let guard = self.tp.borrow();
                let s = guard.as_ref().ok_or("GDN half state without the tensor split")?;
                let conv = s.heads.as_ref().and_then(|h| h.conv.get(li)?.as_ref()).ok_or("GDN half conv weights missing")?;
                gdn::Gdn::load_half(&self.weights, li, self.gdn_proj[li][4], conv.device_address(), s.rank, state)
            }
            state => gdn::Gdn::load_next(b, &self.weights, li, self.gdn_proj[li], state),
        }
    }
    /// Level 3 in-projections of rank `rank`'s heads into the scratch at the half's row widths:
    /// q, k and v rows side by side in each qkv row, then z, a and b.
    pub(super) fn tp_heads_project(&self, b: &Backend<'a>, li: usize, rank: usize, s: &mut gdn::GdnScratch<'a>,
        input: &DeviceVector<'a>, count: usize) -> Result<()> {
        let c = &self.checkpoint.config;
        let (kd, vd) = ((c.linear_key_heads * c.linear_key_dim) as usize, (c.linear_value_heads * c.linear_value_dim) as usize);
        let (hk, hv, h, k) = (kd / 2, vd / 2, c.linear_value_heads as usize / 2, c.hidden as usize);
        let width = 2 * hk + hv;
        let [qkv, z, a, gb, _] = self.gdn_proj[li];
        let row = |w: u64, start: usize| w + (start * k * 2) as u64;
        let (x, y) = (input.device_address(), s.qkv.device_address());
        blocks::next_rows_group_at(b, x, k, count, [(row(qkv, rank * hk), y, hk, width), (row(qkv, kd + rank * hk), y + (hk * 4) as u64, hk, width),
            (row(qkv, 2 * kd + rank * hv), y + (2 * hk * 4) as u64, hv, width), (row(z, rank * hv), s.z.device_address(), hv, hv)])?;
        blocks::next_rows_group_at(b, x, k, count, [(row(a, rank * h), s.a.device_address(), h, h),
            (row(gb, rank * h), s.b.device_address(), h, h), (0, 0, 0, 0), (0, 0, 0, 0)])
    }
    /// The GDN layer's out projection into `block` on a split pass: by heads, the ranks swap their gated
    /// rows first; then both split it by row halves (tp_out_project). False: the
    /// caller finishes whole.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::flash) fn tp_gdn_finish(&self, b: &Backend<'a>, li: usize, gdn: &gdn::Gdn<'a, '_>, s: &gdn::GdnScratch<'a>,
        block: &mut DeviceVector<'a>, count: usize, chats: usize) -> Result<bool> {
        if !gdn.is_half() {
            return gdn.out_operands(s).map_or(Ok(false), |o| self.tp_out_project(b, li, o, block, count, chats));
        }
        let moe = self.tp_moe(li, count).ok_or("GDN head split outside a split pass")?;
        let n = gdn.out_cols();
        let mut gated = self.tp.borrow_mut().as_mut().and_then(|s| s.heads.as_mut()).and_then(|h| h.gated.take()).ok_or("GDN gathered rows missing")?;
        let swapped = b.copy_strided(&s.gated, &mut gated, 0, moe.rank * n, n, 2 * n, n, count)
            .and_then(|()| moe.halves(b, &mut gated, count, 2 * n));
        let x = gated.device_address();
        if let Some(h) = self.tp.borrow_mut().as_mut().and_then(|s| s.heads.as_mut()) { h.gated = Some(gated); }
        swapped?;
        if !self.tp_out_project(b, li, (self.gdn_proj[li][4], x, 2 * n), block, count, chats)? {
            return Err("the GDN head split needs the split out projection".into());
        }
        Ok(true)
    }
    /// Level 3, both ranks at a pass's start (rank 0 after tp_part, rank 1 before its layers; the
    /// split serves no pooled steps). First a chat restarting from position 0 gets whole, zeroed
    /// states on both ranks (a reused slot may hold an earlier chat's halves). Then rank 0 decides
    /// whether this pass runs the split GDN layers by heads (one chat at <= ROWS rows, by Policy)
    /// and posts it as the request's HEADS bit; rank 1 follows that bit. By heads: the chat's
    /// states (live and MTP entry) become halves by device copies. Otherwise every chat holding
    /// halves rebuilds them from both ranks' halves over the split link: rank 0 first posts them
    /// as a KIND_HANDOFF request marked GDN_WHOLE (rank 1 reaches a pass only after its post),
    /// rank 1 rebuilds only from that request (`count` usize::MAX).
    pub(in crate::flash) fn tp_gdn_frame<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], count: usize) -> Result<()> {
        let (rank, mut temp, heads, keys, mut policy, bit) = {
            let mut guard = self.tp.borrow_mut();
            let Some(s) = guard.as_mut() else { return Ok(()) };
            let h = s.heads.as_mut().ok_or("GDN head operands missing")?;
            (s.rank, h.temp.take().ok_or("GDN conversion staging missing")?, h.heads, h.keys, std::mem::take(&mut h.policy), s.heads_pass)
        };
        let result = self.gdn_frame(b, sessions, count, (rank, heads, keys), &mut temp, &mut policy, bit);
        if let Some(s) = self.tp.borrow_mut().as_mut() {
            if let Some(h) = s.heads.as_mut() { h.temp = Some(temp); h.policy = policy; }
            if rank == 0 { s.heads_pass = *result.as_ref().unwrap_or(&false); }
        }
        if result.is_err() { sessions.iter_mut().for_each(|chat| chat.valid = false); }
        result.map(|_| ())
    }
    /// Rank 1, a KIND_HANDOFF request marked GDN_PROBE: the probe exchange (Policy).
    pub(in crate::flash) fn tp_gdn_probe(&self) -> Result<()> {
        let guard = self.tp.borrow();
        let s = guard.as_ref().ok_or("tensor split missing")?;
        let h = s.heads.as_ref().ok_or("GDN head operands missing")?;
        let n = probe_bytes(h.heads, h.keys);
        exchange(&s.link, &vec![0u8; n], n).map(|_| ())
    }
    /// Rank 0: post `meta` as a KIND_HANDOFF request (rank 1 answers it before the pass).
    fn tp_heads_request(&self, meta: Vec<[u32; 4]>) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let posted = pp.drain().and_then(|()| { pp.tp_handoff = true; pp.post(&meta, vec![0.0; meta.len() * W], None) });
        posted.map_err(|e| pp.fail(e))
    }
    /// gdn_frame's work; returns whether this pass runs by heads.
    #[allow(clippy::too_many_arguments)]
    fn gdn_frame<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], count: usize,
        (rank, heads, keys): (usize, usize, usize), temp: &mut DeviceVector<'a>, policy: &mut Policy, bit: bool) -> Result<bool> {
        let layers: Vec<usize> = (SPLIT..LAYERS).filter(|li| li % 4 != 3).collect();
        let now = Instant::now();
        let mut converted = false;
        for chat in sessions.iter_mut().filter(|chat| chat.pos == 0) {
            for &li in &layers {
                if let Some(g) = chat.gdn_state[li].as_mut().filter(|g| g.is_half()) {
                    g.to_whole(b, rank, heads, keys, temp, None)?;
                    g.reset(b)?;
                    converted = true;
                }
                if let Some(e) = chat.rollback.as_mut().and_then(|r| r.gdn[li].as_mut()) { e.to_whole(b, rank, heads, keys, temp, None)?; }
            }
        }
        let half = |chat: &NextSession<'a, 'w>| layers.iter().any(|&li| chat.gdn_state[li].as_ref().is_some_and(|g| g.is_half()));
        let one = self.tp_split_rows(count) && sessions.len() == 1;
        let by_heads = if rank != 0 { bit && count != usize::MAX } else {
            policy.tick(now);
            for chat in sessions.iter().filter(|chat| !one || chat.pos == 0) { policy.reset(chat.slot); }
            let mut by_heads = one && half(sessions[0]);
            if one && !by_heads {
                if policy.probe_ms.is_none() && policy.half.1 > 0 {
                    // The rebuild cost: one chat's halves exchanged over the split link, timed.
                    let chat = &sessions[0];
                    self.tp_heads_request(vec![[chat.slot, chat.capacity as u32, chat.pos as u32, GDN_PROBE]])?;
                    let started = Instant::now();
                    let n = probe_bytes(heads, keys);
                    { let guard = self.tp.borrow(); exchange(&guard.as_ref().ok_or("tensor split missing")?.link, &vec![0u8; n], n)?; }
                    policy.probe_ms = Some(started.elapsed().as_secs_f64() * 1e3);
                    eprintln!("flashnext tp gdn heads probe bytes={n} ms={:.1}", policy.probe_ms.unwrap_or(0.0));
                    converted = true;
                }
                by_heads = policy.due(sessions[0].slot);
                if by_heads { policy.reset(sessions[0].slot); }
            }
            by_heads
        };
        if by_heads && !one { return Err("the GDN head split runs one-chat passes of at most ROWS rows only".into()); }
        if by_heads {
            for chat in sessions.iter_mut() {
                for &li in &layers {
                    if let Some(g) = chat.gdn_state[li].as_mut().filter(|g| !g.is_half()) {
                        g.to_half(b, rank, heads, keys, temp, true)?;
                        converted = true;
                    }
                    if let Some(e) = chat.rollback.as_mut().and_then(|r| r.gdn[li].as_mut()) { e.to_half(b, rank, heads, keys, temp, false)?; }
                }
            }
        } else {
            converted |= self.gdn_whole(b, sessions, count, (rank, heads, keys), temp, &layers)?;
        }
        // Only an interval free of conversions, probes and rebuilds measures a pass.
        if rank == 0 { policy.last = (one && !converted).then(|| (now, sessions[0].slot, by_heads)); }
        Ok(by_heads)
    }
    /// Rebuild the whole states of every chat holding halves (all past position 0 here).
    fn gdn_whole<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], count: usize,
        (rank, heads, keys): (usize, usize, usize), temp: &mut DeviceVector<'a>, layers: &[usize]) -> Result<bool> {
        let half = |chat: &NextSession<'a, 'w>| layers.iter().any(|&li| chat.gdn_state[li].as_ref().is_some_and(|g| g.is_half()));
        let owed: Vec<usize> = (0..sessions.len()).filter(|&i| half(sessions[i])).collect();
        if owed.is_empty() { return Ok(false); }
        if rank != 0 && count != usize::MAX { return Err("tensor split GDN halves owed without the front's request".into()); }
        if rank == 0 {
            self.tp_heads_request(owed.iter().map(|&i| [sessions[i].slot, sessions[i].capacity as u32, sessions[i].pos as u32, GDN_WHOLE]).collect())?;
        }
        let started = Instant::now();
        let mut out: Vec<u8> = Vec::new();
        let mut floats: Vec<f32> = Vec::new();
        for &i in &owed {
            for &li in layers {
                let Some(g) = sessions[i].gdn_state[li].as_mut().filter(|g| g.is_half()) else { continue };
                let (state, history, _) = g.handoff();
                for v in [&*state, &*history] {
                    floats.resize(v.len(), 0.0);
                    b.read_vector_range_into(v, 0, &mut floats)?;
                    out.extend_from_slice(pp::f32_bytes(&floats));
                }
            }
        }
        // Both ranks hold the same halves' shapes, so the peer's message is exactly as long.
        let arrived = { let guard = self.tp.borrow(); exchange(&guard.as_ref().ok_or("tensor split missing")?.link, &out, out.len())? };
        let words: Vec<f32> = arrived.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let per: usize = gdn::GdnState::half_words(heads, keys).iter().sum();
        let mut at = 0usize;
        for &i in &owed {
            for &li in layers {
                if let Some(g) = sessions[i].gdn_state[li].as_mut().filter(|g| g.is_half()) {
                    let peer = words.get(at..at + per).ok_or("tensor split GDN halves message too short")?;
                    g.to_whole(b, rank, heads, keys, temp, Some(peer))?;
                    at += per;
                }
                if let Some(e) = sessions[i].rollback.as_mut().and_then(|r| r.gdn[li].as_mut()) { e.to_whole(b, rank, heads, keys, temp, None)?; }
            }
        }
        if at != words.len() { return Err("tensor split GDN halves message has extra words".into()); }
        b.synchronize()?;
        eprintln!("flashnext tp gdn heads whole chats={} bytes={} ms={:.1}", owed.len(), out.len(), started.elapsed().as_secs_f64() * 1e3);
        Ok(true)
    }
}

/// Bytes of one chat's GDN halves on the split layers (the probe's size: one rebuild's message).
fn probe_bytes(heads: usize, keys: usize) -> usize {
    (SPLIT..LAYERS).filter(|li| li % 4 != 3).count() * gdn::GdnState::half_words(heads, keys).iter().sum::<usize>() * 4
}

/// Rank 0's switch to halves, as ski rental over measured times (no fixed pass count): a whole
/// chat running one-chat split passes forgoes the head split's saving each pass ("rent"); its
/// halves cost one rebuild when a pass of other shape comes ("buy"). It switches once the saving
/// it forwent reaches one rebuild: solo / T_whole x (T_whole - T_half) >= rebuild, where solo is
/// its time in such passes since its last pass of other shape, rebuild the probe's time and
/// T_whole, T_half the mean intervals between the starts of one-chat split passes run whole or by
/// heads. Before any head-split pass was measured the first eligible chat switches at once (to
/// measure); when the measured passes show no saving, chats stay whole.
#[derive(Default)]
pub(super) struct Policy {
    probe_ms: Option<f64>,
    whole: (f64, u32),
    half: (f64, u32),
    /// (slot, ms of one-chat split passes run whole since its last pass of other shape).
    solo: Vec<(u32, f64)>,
    /// The previous pass start, when it was a one-chat split pass without conversion.
    last: Option<(Instant, u32, bool)>,
}
impl Policy {
    fn tick(&mut self, now: Instant) {
        let Some((at, slot, by_heads)) = self.last.take() else { return };
        let ms = now.duration_since(at).as_secs_f64() * 1e3;
        let mean = if by_heads { &mut self.half } else { &mut self.whole };
        mean.0 += ms;
        mean.1 += 1;
        if !by_heads {
            match self.solo.iter_mut().find(|s| s.0 == slot) { Some(s) => s.1 += ms, None => self.solo.push((slot, ms)) }
        }
    }
    fn reset(&mut self, slot: u32) { self.solo.retain(|s| s.0 != slot); }
    fn due(&self, slot: u32) -> bool {
        let solo = self.solo.iter().find(|s| s.0 == slot).map_or(0.0, |s| s.1);
        switch_due(self.probe_ms.unwrap_or(f64::INFINITY), self.whole, self.half, solo)
    }
}
/// Policy's rule over (sum ms, passes) of whole and head-split one-chat passes.
fn switch_due(rebuild_ms: f64, whole: (f64, u32), half: (f64, u32), solo_ms: f64) -> bool {
    if half.1 == 0 { return true; }
    if whole.1 == 0 { return false; }
    let (tw, th) = (whole.0 / f64::from(whole.1), half.0 / f64::from(half.1));
    tw > th && solo_ms * (tw - th) >= rebuild_ms * tw
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The first eligible chat switches at once; afterwards a chat switches exactly when the saving
    /// it forwent reaches one rebuild, and never when the head-split passes measured no faster.
    #[test]
    fn switch_is_ski_rental_over_measured_times() {
        assert!(switch_due(100.0, (0.0, 0), (0.0, 0), 0.0));
        assert!(!switch_due(100.0, (0.0, 0), (38.0, 1), 1e9), "no whole pass measured yet");
        // T_whole 40 ms, T_half 38 ms: 2 ms forgone per 40 ms of solo time; rebuild 100 ms -> 2000 ms.
        assert!(!switch_due(100.0, (400.0, 10), (380.0, 10), 1999.0));
        assert!(switch_due(100.0, (400.0, 10), (380.0, 10), 2000.0));
        assert!(!switch_due(100.0, (400.0, 10), (400.0, 10), 1e9));
        assert!(!switch_due(f64::INFINITY, (400.0, 10), (380.0, 10), 1e9), "an unmeasured rebuild never pays");
    }
    /// Intervals land in the mean of the pass kind they measured; solo time counts whole passes of
    /// its slot only and a reset (a pass of other shape, a restart) drops it.
    #[test]
    fn policy_accounts_solo_time_per_slot() {
        let mut p = Policy::default();
        let t = Instant::now();
        p.last = Some((t, 7, false));
        p.tick(t + std::time::Duration::from_millis(40));
        assert_eq!((p.whole.1, p.half.1), (1, 0));
        assert!((p.solo[0].1 - 40.0).abs() < 1.0 && p.solo[0].0 == 7);
        p.last = Some((t, 7, true));
        p.tick(t + std::time::Duration::from_millis(30));
        assert_eq!((p.whole.1, p.half.1, p.solo.len()), (1, 1, 1));
        p.reset(7);
        assert!(p.solo.is_empty());
        p.tick(t);
        assert_eq!(p.whole.1 + p.half.1, 2, "no previous pass, no sample");
    }
}
