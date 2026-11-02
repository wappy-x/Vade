//! The split over more ranks, after a prompt ran as chain tiles: rank r ran only its tile layers
//! (tile_layers) for those chats, and every rank needs the session state of every layer from
//! SPLIT on. Each rank puts its tile layers' state and takes every other rank's, as raw FP32
//! words through the relayed mailbox (an all-gather; no host touches the state words). First one
//! head message per rank: per chat and own layer the head word (QSA past or the GDN cached flag)
//! and the words of each state vector, which the receiver installs and checks against its own.
//! Then the state pieces: message i of every rank carries its piece i (or one pad word once its
//! pieces end), and each rank puts message i + 1 before it takes message i from every other rank,
//! so no rank runs more than one message ahead of the others' takes (the mailbox's lockstep).
use super::*;

/// The layers rank `rank` of `world` runs for a chain tile.
pub(super) fn tile_layers(rank: usize, world: usize) -> std::ops::Range<usize> {
    LAYERS * rank / world..LAYERS * (rank + 1) / world
}
/// The layers whose state rank `rank` holds after a chain tile (none before SPLIT cross).
fn held(rank: usize, world: usize) -> std::ops::Range<usize> {
    let layers = tile_layers(rank, world);
    layers.start.max(SPLIT)..layers.end.max(SPLIT)
}

impl<'a> NextModel<'a> {
    /// Every rank, for the same chats in the same order (rank 0 after posting KIND_HANDOFF).
    /// `push`: rank 0 holds every layer and the other ranks hold none (a sequence split's state).
    pub(super) fn tp_gather_state<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], push: bool) -> Result<()> {
        let started = Instant::now();
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let channels = mailbox.layout();
        let slot = mailbox.slot_bytes() / 4;
        let (rank, world, origins, limit) = self.tp.borrow().as_ref().map(|s| (s.rank, s.layout.world, s.origins(), s.limit_ns))
            .ok_or("tensor split missing")?;
        let held = |o: usize| if !push { held(o, world) } else if o == 0 { SPLIT..LAYERS } else { SPLIT..SPLIT };
        // This rank's head words and pieces.
        let (mut head, mut sends) = (Vec::new(), Vec::new());
        for (c, chat) in sessions.iter_mut().enumerate() {
            for li in held(rank) {
                let (flag, words) = layer_words(chat, li, None)?;
                for w in std::iter::once(flag).chain(words) {
                    head.push(f32::from_bits(u32::try_from(w).map_err(|_| "tensor split gather: a state size exceeds 32 bits")?));
                }
                for (k, n) in words.into_iter().enumerate() { pieces(&mut sends, (c, li, k), n, slot); }
            }
        }
        // Head messages, one per rank, of a length every rank knows (one pad word for no layer).
        let lens: Vec<usize> = (0..world).map(|o| (sessions.len() * held(o).len() * 4).max(1)).collect();
        if head.is_empty() { head.push(0.0); }
        if lens.iter().any(|&n| n > slot) || head.len() != lens[rank] {
            return Err("tensor split gather: the head does not fit one mailbox message".into());
        }
        let mut heads: Vec<DeviceVector<'a>> = lens.iter().map(|&n| b.vector(n)).collect::<Result<_>>()?;
        b.write_vector_range(&mut heads[rank], 0, &head)?;
        {
            let mut guard = self.tp.borrow_mut();
            let s = guard.as_mut().ok_or("tensor split missing")?;
            s.sent += 1;
            b.tp_mailbox_put(mailbox, &heads[rank], 1, lens[rank], lens[rank], 0, s.sent)?;
            for &o in &origins {
                let seq = s.take(o);
                b.tp_mailbox_take_from(mailbox, &mut heads[o], 1, lens[o], lens[o], 0, channels.channel(o, rank), seq, limit)?;
            }
        }
        b.synchronize()?;
        if self.tp_failed() { return Err("tensor split mailbox timed out or its proxy failed during the gather".into()); }
        // Install each origin's head words; its pieces follow in the same order.
        let mut recvs: Vec<Vec<Piece>> = (0..world).map(|_| Vec::new()).collect();
        for &o in &origins {
            let mut words = vec![0f32; lens[o]];
            b.read_vector_range_into(&heads[o], 0, &mut words)?;
            let mut fields = words.iter().map(|w| w.to_bits() as usize);
            for (c, chat) in sessions.iter_mut().enumerate() {
                for li in held(o) {
                    let mut field = || fields.next().ok_or("tensor split gather head too short");
                    let flag = field()?;
                    let peer = [field()?, field()?, field()?];
                    // The installed rows need their KV pages first.
                    if li % 4 == 3 { chat.fit_kv(b, flag)?; }
                    let (_, words) = layer_words(chat, li, Some(flag))?;
                    if words != peer {
                        return Err(format!("tensor split gather: rank {o}'s state sizes differ from this rank's"));
                    }
                    for (k, n) in words.into_iter().enumerate() { pieces(&mut recvs[o], (c, li, k), n, slot); }
                }
            }
        }
        let messages = recvs.iter().map(Vec::len).chain(std::iter::once(sends.len())).max().unwrap_or(0);
        let (pad, mut pad_in) = (b.vector(1)?, b.vector(1)?);
        let put = |sessions: &mut [&mut NextSession<'a, 'w>], i: usize| -> Result<()> {
            let mut guard = self.tp.borrow_mut();
            let s = guard.as_mut().ok_or("tensor split missing")?;
            s.sent += 1;
            match sends.get(i) {
                Some(&Piece { at: (c, li, k), from, words }) => {
                    let v = state_vector(sessions[c], li, k)?;
                    b.tp_mailbox_put(mailbox, v, 1, v.len(), words, from, s.sent)
                }
                None => b.tp_mailbox_put(mailbox, &pad, 1, 1, 1, 0, s.sent),
            }
        };
        let mut gathered = if messages > 0 { put(sessions, 0) } else { Ok(()) };
        for i in 0..messages {
            if gathered.is_err() { break; }
            if i + 1 < messages { gathered = put(sessions, i + 1); }
            for &o in &origins {
                if gathered.is_err() { break; }
                let (seq, channel) = (self.tp.borrow_mut().as_mut().map(|s| s.take(o)).ok_or("tensor split missing")?, channels.channel(o, rank));
                gathered = match recvs[o].get(i) {
                    Some(&Piece { at: (c, li, k), from, words }) => state_vector(sessions[c], li, k)
                        .and_then(|v| { let width = v.len(); b.tp_mailbox_take_from(mailbox, v, 1, width, words, from, channel, seq, limit) }),
                    None => b.tp_mailbox_take_from(mailbox, &mut pad_in, 1, 1, 1, 0, channel, seq, limit),
                };
            }
        }
        gathered?;
        b.synchronize()?;
        if self.tp_failed() { return Err("tensor split mailbox timed out or its proxy failed during the gather".into()); }
        let bytes = |p: &[Piece]| p.iter().map(|p| p.words * 4).sum::<usize>();
        eprintln!("flashnext tp gather chats={} sent_bytes={} received_bytes={} messages={messages} ms={:.1}", sessions.len(), bytes(&sends),
            recvs.iter().map(|r| bytes(r)).sum::<usize>(), started.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Chain tiles cover every layer once; the held layers start at SPLIT and cover the rest once,
    /// on the pair as the halves it always cut (24) and on four ranks.
    #[test]
    fn tile_layers_cover_every_layer_once() {
        for world in [2, 3, 4] {
            let all: Vec<usize> = (0..world).flat_map(|r| tile_layers(r, world)).collect();
            assert_eq!(all, (0..LAYERS).collect::<Vec<_>>());
            let held_all: Vec<usize> = (0..world).flat_map(|r| held(r, world)).collect();
            assert_eq!(held_all, (SPLIT..LAYERS).collect::<Vec<_>>());
        }
        assert_eq!((tile_layers(1, 2), held(0, 2)), (LAYERS / 2..LAYERS, SPLIT..LAYERS / 2));
        assert_eq!((tile_layers(2, 4), held(0, 4)), (24..36, SPLIT..12));
    }
}
