//! VADE_W2_TP=1: the two-rank tensor split of the FP4-Local layer stack (world 2, opt-in).
//! Rank 0 runs the world-one engine (frontend, every layer, head, drafting). Rank 1 mirrors
//! every main-model pass from layer SPLIT on through the pipeline back's session protocol
//! (rank 0 posts the pass before layer 0 and never waits for rows). On passes of at most
//! TpLayout::ROWS rows both ranks split every routed expert by output columns: rank r computes
//! gate/up columns [mid_cols r, mid_cols (r + 1)) and down columns [out_cols r, out_cols (r + 1))
//! of every routed pair with the unchanged fp4_bw entries (each column keeps its CTA, K16 order
//! and rounding); the ranks swap the BF16 middle halves as raw bits, each finishes its half of
//! the output columns (flash_next_k16_finish is per column: same slot order, same shared add),
//! and they swap those halves. Router, routes and the shared expert run on both ranks from the
//! same bits. No sum is split, so every output element is world one's. Every exchange (the
//! layer-SPLIT residual and both halves) rides the in-graph mailbox (spark-gpu-probe
//! TpMailbox), so split passes keep their captured verify segments. All widths come from
//! the checkpoint (flash_tp_switch::TpLayout).
use super::*;
use crate::flash_tp_switch::TpLayout;
use std::time::Instant;

/// The first layer rank 1 runs: rank 0 keeps the frontend and layer 1's PLE (pp::Stage).
pub(super) const SPLIT: usize = 2;
pub(super) use crate::flash_tp_switch::from_env;

/// The checkpoint's split geometry.
pub(super) fn layout(config: &engine_model::flash::Config) -> Result<TpLayout> {
    TpLayout::new(config.hidden, config.branches, config.expert_width, config.top_k, config.experts)
}

/// Rank 1's expert descriptors: the same matrices from row mid_cols (gate, up) and row
/// out_cols (down), appended after the world-one descriptors of each layer.
pub(super) fn shifted(descriptors: &mut Vec<spark_gpu_probe::driver::fp4_abi::Expert>, layout: &TpLayout) {
    use spark_gpu_probe::driver::fp4_abi::Matrix;
    let shift = |m: Matrix, (weights, scales): (usize, usize)| Matrix {
        weights: m.weights + weights as u64,
        scales: m.scales + scales as u64,
        global_scale: m.global_scale,
    };
    let base = descriptors.len();
    for i in 0..base {
        let d = descriptors[i];
        descriptors.push(spark_gpu_probe::driver::fp4_abi::Expert {
            gate: shift(d.gate, layout.gate_shift()),
            up: shift(d.up, layout.gate_shift()),
            down: shift(d.down, layout.down_shift()),
            ..d
        });
    }
}

/// This rank's half and its mailbox counters (NextModel::tp_box holds the mailbox; the proxy
/// thread owns the RDMA wire). Messages count from 1 per direction in u64.
pub(super) struct Swap {
    rank: usize,
    layout: TpLayout,
    sent: u64,
    taken: u64,
    /// A take waits at most this long (VADE_W2_PP_TIMEOUT_MS), then fails the pass.
    limit_ns: u64,
    /// Rank 1: this pass's residual arrives in the mailbox (the request's MAIL bit).
    mail_pass: bool,
    /// Rank 1: this request is a pipeline prompt tile (layers 24..48 here; TILE bit).
    tile_pass: bool,
    /// The split's TCP link carries state handoffs; rank 0: chats whose prompt ran as
    /// pipeline tiles and that owe a handoff before their next split pass.
    link: std::net::TcpStream,
    handoff: Vec<u32>,
    /// Rank 0: the current pass runs as a pipeline tile (tp_part).
    front_tile: bool,
}

/// One routed MoE layer's split on this rank (Workspace::moe_at offsets and swaps).
#[derive(Clone, Copy)]
pub(super) struct TpMoe<'s, 'a> {
    rank: usize,
    layout: TpLayout,
    swap: &'s RefCell<Option<Swap>>,
    mailbox: &'s spark_gpu_probe::driver::TpMailbox<'a>,
}
impl<'a> TpMoe<'_, 'a> {
    /// Bytes past the layer's descriptors: rank 1's shifted table.
    pub(super) fn table(self) -> u64 {
        (self.rank * self.layout.experts * std::mem::size_of::<spark_gpu_probe::driver::fp4_abi::Expert>()) as u64
    }
    pub(super) fn mid(self) -> u64 { self.layout.mid(self.rank) }
    pub(super) fn routed(self) -> u64 { self.layout.routed(self.rank) }
    pub(super) fn out(self) -> u64 { self.layout.out(self.rank) }
    pub(super) fn finish_ctas(self) -> u32 { self.layout.finish_ctas() }
    /// World-one gate / down CTAs scaled to this rank's share of the output columns.
    pub(super) fn gate_ctas(self, whole: u32) -> u32 { self.layout.share(whole, self.layout.mid_cols(), self.layout.expert_width) }
    pub(super) fn down_ctas(self, whole: u32) -> u32 { self.layout.share(whole, self.layout.out_cols(), self.layout.hidden) }
    pub(super) fn swap_mid(self, b: &Backend<'a>, mid: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        self.halves(b, mid, count * self.layout.top_k, self.layout.mid_words())
    }
    pub(super) fn swap_out(self, b: &Backend<'a>, block: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        self.halves(b, block, count, self.layout.hidden)
    }
    /// Publish this rank's half of each of `rows` rows of `width` words and take the peer's,
    /// both queued in the stream.
    fn halves(self, b: &Backend<'a>, vector: &mut DeviceVector<'a>, rows: usize, width: usize) -> Result<()> {
        let mut guard = self.swap.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let half = width / 2;
        s.sent += 1;
        s.taken += 1;
        b.tp_mailbox_put(self.mailbox, vector, rows, width, half, s.rank * half, s.sent)?;
        b.tp_mailbox_take(self.mailbox, vector, rows, width, half, (1 - s.rank) * half, s.taken, s.limit_ns)
    }
}

impl<'a> NextModel<'a> {
    /// Both ranks, before serving: this rank's half and the mailbox. The proxy thread then takes
    /// the mailbox's one host endpoint (tp_mailbox_host).
    pub fn set_tp(&self, b: &Backend<'a>, rank: usize, link: std::net::TcpStream) -> Result<()> {
        if rank > 1 || !from_env()? {
            return Err("the tensor split needs VADE_W2_TP=1 and rank 0 or 1".into());
        }
        let layout = layout(&self.checkpoint.config)?;
        // The world-one MoE path this split reuses sizes its buffers from the same geometry.
        if layout.hidden != H || layout.residual != W || layout.experts != EXPERTS {
            return Err(format!("tensor split geometry {layout:?} differs from the loaded world-one layout"));
        }
        // VADE_W2_TP_HEAD: before the wire thread's key swap on the same link (the proxy's attach wait).
        crate::flash_head_options::agree_split(&link, rank, crate::flash_head_options::split_from_env()?, std::time::Duration::from_secs(600))?;
        if self.tp_box.set(b.tp_mailbox(layout.slot_bytes())?).is_err() {
            return Err("the tensor split mailbox is already set".into());
        }
        *self.tp.borrow_mut() = Some(Swap { rank, layout, sent: 0, taken: 0, limit_ns: pp::timeout_from_env()?.as_nanos() as u64,
            mail_pass: false, tile_pass: false, link, handoff: Vec::new(), front_tile: false });
        eprintln!("flashnext tp rank={rank} split={SPLIT} rows<={} layout={layout:?}", TpLayout::ROWS);
        Ok(())
    }
    pub(super) fn tp_on(&self) -> bool {
        self.tp.borrow().is_some()
    }
    /// The proxy thread's endpoint of this rank's mailbox (handed out once).
    pub fn tp_mailbox_host(&self) -> Option<spark_gpu_probe::driver::TpMailboxHost<'_>> {
        self.tp_box.get().and_then(|m| m.take_host())
    }
    /// A mailbox take timed out or the proxy failed (sticky; read after a pass completed).
    pub(super) fn tp_failed(&self) -> bool {
        self.tp_box.get().is_some_and(|m| m.failed())
    }
    /// A split-mode pass wider than the mailbox (more than ROWS rows): its residual crosses
    /// with the request, it splits no expert and runs without captured verify segments.
    pub(super) fn tp_host_pass(&self, count: usize) -> bool {
        self.tp_on() && count > TpLayout::ROWS
    }
    /// Rank 1, per request: the residual arrives in the mailbox (MAIL bit), or the request is
    /// a pipeline tile for layers 24..48 (TILE bit).
    pub(super) fn tp_mail_next(&self, kind: u32) {
        let mail = kind & crate::flash_tp_switch::MAIL != 0;
        let tile = kind & crate::flash_tp_switch::TILE != 0;
        if let Some(s) = self.tp.borrow_mut().as_mut() { s.mail_pass = mail; s.tile_pass = tile; }
    }
    /// Rank 1: the first layer of the current request (24 for a pipeline tile, else SPLIT).
    pub(super) fn tp_back_split(&self) -> Option<usize> {
        self.tp.borrow().as_ref().filter(|s| s.tile_pass).map(|_| LAYERS / 2)
    }
    /// Rank 0, before a pass. Mailbox mode: a prompt tile of more than ROWS rows runs as a
    /// pipeline tile (Part::Front at layer 24; returns Some(24)) and its chats owe a state
    /// handoff; any other pass first settles the handoffs its chats owe (KIND_HANDOFF).
    pub(super) fn tp_part<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], spans: &[std::ops::Range<usize>], verify: bool) -> Result<Option<usize>> {
        if self.tp_rank() != Some(0) {
            return Ok(None);
        }
        // A prompt pass runs as a pipeline tile when every chat's span fills at least half of
        // the engine's prompt tile (the pipeline's unit of work, an operator setting, not a
        // checkpoint figure); shorter spans stay in the lockstep and owe no state handoff.
        let tile = !verify && spans.iter().all(|span| 2 * span.len() >= crate::flash_next_rows::prompt_tile());
        if let Some(link) = self.pp.borrow_mut().as_mut() { link.tp_tile = tile; }
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        s.front_tile = tile;
        if tile {
            for chat in sessions.iter() { if !s.handoff.contains(&chat.slot) { s.handoff.push(chat.slot); } }
            return Ok(Some(LAYERS / 2));
        }
        // A slot that a new chat reuses from position 0 owes nothing (its prompt ran split).
        let owed: Vec<usize> = (0..sessions.len()).filter(|&i| sessions[i].pos > 0 && s.handoff.contains(&sessions[i].slot)).collect();
        s.handoff.retain(|slot| !sessions.iter().any(|chat| chat.slot == *slot && chat.pos == 0));
        if owed.is_empty() {
            return Ok(None);
        }
        let link = s.link.try_clone().map_err(err)?;
        drop(guard);
        let meta: Vec<[u32; 4]> = owed.iter().map(|&i| [sessions[i].slot, sessions[i].capacity as u32, sessions[i].pos as u32, 1]).collect();
        {
            let mut guard = self.pp.borrow_mut();
            let pp = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
            let posted = pp.drain().and_then(|()| { pp.tp_handoff = true; pp.post(&meta, vec![0.0; meta.len() * W], None) });
            posted.map_err(|e| pp.fail(e))?;
        }
        let mut chats: Vec<&mut NextSession<'a, 'w>> = Vec::new();
        for (i, chat) in sessions.iter_mut().enumerate() { if owed.contains(&i) { chats.push(chat); } }
        // The debt is settled only by a whole exchange: a failure leaves the chats' state partly
        // installed, so they stop being reusable and the link stops (the back sees it too).
        if let Err(error) = self.tp_swap_state(b, &mut chats, SPLIT..LAYERS / 2, LAYERS / 2..LAYERS, link) {
            chats.iter_mut().for_each(|chat| chat.valid = false);
            return Err(self.pp.borrow_mut().as_mut().map_or(error.clone(), |pp| pp.fail(error)));
        }
        let settled: Vec<u32> = chats.iter().map(|chat| chat.slot).collect();
        if let Some(s) = self.tp.borrow_mut().as_mut() { s.handoff.retain(|slot| !settled.contains(slot)); }
        Ok(None)
    }
    /// Rank 1, KIND_HANDOFF: its layers 24..48 of these chats for rank 0's layers 2..24.
    pub(super) fn tp_handoff<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>]) -> Result<()> {
        let link = self.tp.borrow().as_ref().map(|s| s.link.try_clone()).ok_or("tensor split missing")?.map_err(err)?;
        let swapped = self.tp_swap_state(b, sessions, LAYERS / 2..LAYERS, SPLIT..LAYERS / 2, link);
        if swapped.is_err() { sessions.iter_mut().for_each(|chat| chat.valid = false); }
        swapped
    }
    /// Both ranks: send this rank's session state of `send` layers and install the peer's of
    /// `recv` layers, as raw FP32 words (GDN recurrent state, conv history and cached flag;
    /// QSA keys, values and raw index rows 0..past, and past), chats in the given order.
    fn tp_swap_state<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], send: std::ops::Range<usize>,
        recv: std::ops::Range<usize>, mut link: std::net::TcpStream) -> Result<()> {
        use std::io::{Read, Write};
        let started = Instant::now();
        let mut out: Vec<u8> = Vec::new();
        let mut floats: Vec<f32> = Vec::new();
        let mut put = |b: &Backend<'a>, v: &DeviceVector<'a>, n: usize, out: &mut Vec<u8>| -> Result<()> {
            floats.resize(n, 0.0);
            b.read_vector_range_into(v, 0, &mut floats)?;
            out.extend_from_slice(pp::f32_bytes(&floats));
            Ok(())
        };
        for chat in sessions.iter_mut() {
            for li in send.clone() {
                if li % 4 == 3 {
                    let q = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the handoff")?;
                    let (vectors, widths, past) = q.handoff()?;
                    let rows = *past;
                    out.extend_from_slice(&(rows as u64).to_le_bytes());
                    for (v, w) in vectors.into_iter().zip(widths) { if w != 0 { put(b, v, rows * w, &mut out)?; } }
                } else {
                    let g = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the handoff")?;
                    let (state, history, cached) = g.handoff();
                    out.extend_from_slice(&u64::from(*cached).to_le_bytes());
                    let (ns, nh) = (state.len(), history.len());
                    put(b, state, ns, &mut out)?;
                    put(b, history, nh, &mut out)?;
                }
            }
        }
        link.set_nonblocking(false).map_err(err)?;
        link.set_read_timeout(Some(pp::timeout_from_env()?)).map_err(err)?;
        let mut writer = link.try_clone().map_err(err)?;
        let mut arrived = Vec::new();
        std::thread::scope(|scope| -> Result<()> {
            let sender = scope.spawn(|| -> std::io::Result<()> { writer.write_all(&(out.len() as u64).to_le_bytes())?; writer.write_all(&out) });
            let mut len = [0u8; 8];
            link.read_exact(&mut len).map_err(err)?;
            arrived.resize(u64::from_le_bytes(len) as usize, 0);
            link.read_exact(&mut arrived).map_err(err)?;
            sender.join().map_err(|_| "tensor split handoff sender panicked")?.map_err(err)
        })?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let bytes = arrived.get(at..at + n).ok_or("tensor split handoff message too short")?;
            at += n;
            Ok(bytes)
        };
        let words = |bytes: &[u8]| -> Vec<f32> { bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect() };
        for chat in sessions.iter_mut() {
            for li in recv.clone() {
                let head = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
                if li % 4 == 3 {
                    let q = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the handoff")?;
                    let (vectors, widths, past) = q.handoff()?;
                    for (v, w) in vectors.into_iter().zip(widths) {
                        if w != 0 && head != 0 { let values = words(take(head * w * 4)?); b.write_vector_range(v, 0, &values)?; }
                    }
                    *past = head;
                } else {
                    let g = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the handoff")?;
                    let (state, history, cached) = g.handoff();
                    let (ns, nh) = (state.len(), history.len());
                    let values = words(take(ns * 4)?); b.write_vector_range(state, 0, &values)?;
                    let values = words(take(nh * 4)?); b.write_vector_range(history, 0, &values)?;
                    *cached = head != 0;
                }
            }
        }
        if at != arrived.len() {
            return Err("tensor split handoff message has extra bytes".into());
        }
        b.synchronize()?;
        eprintln!("flashnext tp handoff chats={} sent_bytes={} received_bytes={} ms={:.1}", sessions.len(), out.len(), arrived.len(),
            started.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    /// Rank 1, before layer SPLIT: the pass's residual rows from the request, or (MAIL) a take
    /// of rank 0's in-stream post.
    pub(super) fn tp_residual_in(&self, b: &Backend<'a>, residual: &mut DeviceVector<'a>, input: &[f32], count: usize) -> Result<()> {
        let mut guard = self.tp.borrow_mut();
        let Some(s) = guard.as_mut().filter(|s| s.mail_pass) else {
            return b.write_vector_range(residual, 0, &input[..count * W]);
        };
        s.mail_pass = false;
        s.taken += 1;
        let width = s.layout.residual;
        b.tp_mailbox_take(self.tp_box.get().ok_or("tensor split mailbox missing")?, residual, count, width, width, 0, s.taken, s.limit_ns)
    }
    fn tp_rank(&self) -> Option<usize> {
        self.tp.borrow().as_ref().map(|s| s.rank)
    }
    /// VADE_W2_TP_HEAD=1: `rows` rows of `width` words from rank 1 to rank 0, queued in the
    /// stream (rank 1 puts `vector`'s rows, rank 0 takes them into `vector`); both ranks call it
    /// for the same passes (execution::head_split).
    pub(super) fn tp_rows_to_front(&self, b: &Backend<'a>, vector: &mut DeviceVector<'a>, rows: usize, width: usize) -> Result<()> {
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        if s.rank == 1 {
            s.sent += 1;
            b.tp_mailbox_put(mailbox, vector, rows, width, width, 0, s.sent)
        } else {
            s.taken += 1;
            b.tp_mailbox_take(mailbox, vector, rows, width, width, 0, s.taken, s.limit_ns)
        }
    }
    /// Layers whose MoE output is formed without the deferred finish: the front's last layer
    /// before the post (its residual crosses) and every split layer (its halves cross).
    pub(super) fn tp_plain(&self, li: usize, count: usize) -> bool {
        self.tp_on() && (li + 1 == SPLIT || li >= SPLIT && count <= TpLayout::ROWS)
    }
    pub(super) fn tp_moe(&self, li: usize, count: usize) -> Option<TpMoe<'_, 'a>> {
        let (rank, layout) = self.tp.borrow().as_ref().map(|s| (s.rank, s.layout))?;
        let mailbox = self.tp_box.get()?;
        (li >= SPLIT && count <= TpLayout::ROWS).then_some(TpMoe { rank, layout, swap: &self.tp, mailbox })
    }
    /// Rank 0: post the pass to rank 1 (the previous pass's ack first). Prompt and plain rows
    /// ride KIND_ROWS_ASYNC, verify rows KIND_VERIFY. Up to ROWS rows: the request (MAIL bit,
    /// zero rows) before layer 0, then the residual in-stream at layer SPLIT. Wider passes: at
    /// the top of layer SPLIT, with the residual rows read back into the request.
    pub(super) fn tp_post<'w>(&self, b: &Backend<'a>, sessions: &[&mut NextSession<'a, 'w>], spans: &[std::ops::Range<usize>],
        residual: &DeviceVector<'a>, count: usize, verify: bool, li: usize) -> Result<()> {
        let mail = !self.tp_host_pass(count);
        // A pipeline prompt tile (tp_part) posts through the pipeline's own front path.
        if self.tp.borrow().as_ref().is_some_and(|s| s.front_tile) { return Ok(()); }
        if self.tp_rank() != Some(0) || li != if mail { 0 } else { SPLIT } {
            if mail && li == SPLIT && self.tp_rank() == Some(0) {
                let mut guard = self.tp.borrow_mut();
                let s = guard.as_mut().ok_or("tensor split missing")?;
                s.sent += 1;
                let width = s.layout.residual;
                return b.tp_mailbox_put(self.tp_box.get().ok_or("tensor split mailbox missing")?, residual, count, width, width, 0, s.sent);
            }
            return Ok(());
        }
        let meta: Vec<[u32; 4]> = sessions.iter().zip(spans)
            .map(|(s, span)| [s.slot, s.capacity as u32, s.pos as u32, span.len() as u32]).collect();
        let mut rows = vec![0.0; count * W];
        if !mail {
            b.read_vector_range_into(residual, 0, &mut rows)?;
        }
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let posted = link.drain().and_then(|()| {
            link.tp_verify = verify;
            link.tp_mail = mail;
            link.post(&meta, rows, None)
        });
        posted.map_err(|e| link.fail(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Rank 1's descriptors start at its half's rows, for the shipped geometry and another.
    #[test]
    fn shifted_descriptors_start_at_the_rank_one_rows() {
        use spark_gpu_probe::driver::fp4_abi::{Expert, Matrix};
        for layout in [TpLayout::new(2560, 4, 640, 10, 512).unwrap(), TpLayout::new(4096, 2, 1536, 8, 128).unwrap()] {
            let m = |w: u64| Matrix { weights: w, scales: w + 1, global_scale: w + 2 };
            let mut d = vec![Expert { gate: m(1000), up: m(2000), down: m(3000), width: layout.expert_width as u32, reserved: 0 }; 2];
            shifted(&mut d, &layout);
            assert_eq!(d.len(), 4);
            assert_eq!(d[2].gate.weights, 1000 + (layout.mid_cols() * layout.hidden / 2) as u64);
            assert_eq!(d[2].gate.scales, 1001 + (layout.mid_cols() * layout.hidden / 16) as u64);
            assert_eq!(d[3].down.weights, 3000 + (layout.out_cols() * layout.expert_width / 2) as u64);
            assert_eq!(d[3].down.scales, 3001 + (layout.out_cols() * layout.expert_width / 16) as u64);
            assert_eq!(d[3].down.global_scale, 3002);
            assert_eq!(d[2].width, layout.expert_width as u32);
        }
    }
}
