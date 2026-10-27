//! The two-rank tensor split of the FP4-Local layer stack: the only 2-Spark path (world 2).
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
#[path = "next_tp_heads.rs"]
mod heads;

pub(super) use crate::flash_tp_switch::SPLIT;

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
pub(super) struct Swap<'a> {
    rank: usize,
    layout: TpLayout,
    sent: u64,
    taken: u64,
    /// A take waits at most this long (pp::LINK_TIMEOUT), then fails the pass.
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
    /// This rank's row halves of the projections' MXFP8 twins (copied from the loaded twins, so
    /// the same bytes): the GDN in-projections (tp_gdn_project), the mixers' out projections
    /// (tp_out_project) and the shared expert's down projection (TpMoe::shared_rows). The GDN
    /// mixers of one-chat split passes run by heads (next_tp_heads.rs).
    _twins: Vec<blocks::Mx8Twin<'a>>,
    /// This rank's GDN head operands and buffers.
    heads: Option<heads::GdnHeads<'a>>,
    /// The current pass runs its GDN layers by heads (rank 0 decides in tp_gdn_frame and
    /// posts the HEADS bit; rank 1 reads it from the request).
    heads_pass: bool,
    /// The current pass verifies (rank 0: tp_part's flag; rank 1: the request kind), so it splits
    /// up to TpLayout::ROWS rows instead of TpLayout::PROMPT_ROWS.
    verify_pass: bool,
}

/// One routed MoE layer's split on this rank (Workspace::moe_at offsets and swaps).
#[derive(Clone, Copy)]
pub(super) struct TpMoe<'s, 'a> {
    rank: usize,
    layout: TpLayout,
    swap: &'s RefCell<Option<Swap<'a>>>,
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
    /// The shared expert's down projection ([hidden][k] BF16 at `weight`, output `y` as moe_at
    /// lays it out): this rank's output rows [out_cols r, out_cols (r + 1)) as (weight, y, rows),
    /// the only columns its finish reads.
    pub(super) fn shared_rows(self, weight: u64, y: u64, k: usize) -> (u64, u64, usize) {
        let start = self.rank * self.layout.out_cols();
        (weight + (start * k * 2) as u64, y + (start * 4) as u64, self.layout.out_cols())
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
    /// `halves` for two vectors of one launch: both puts, then both takes, so the second
    /// message travels while the first is awaited. With the pass's residual message still
    /// unread by the peer, three are in flight, below the mailbox's four slots.
    fn halves_pair(self, b: &Backend<'a>, [(x, xw), (y, yw)]: [(&mut DeviceVector<'a>, usize); 2], rows: usize) -> Result<()> {
        let mut guard = self.swap.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        let (sent, taken, rank) = (s.sent, s.taken, s.rank);
        s.sent += 2;
        s.taken += 2;
        b.tp_mailbox_put(self.mailbox, x, rows, xw, xw / 2, rank * xw / 2, sent + 1)?;
        b.tp_mailbox_put(self.mailbox, y, rows, yw, yw / 2, rank * yw / 2, sent + 2)?;
        b.tp_mailbox_take(self.mailbox, x, rows, xw, xw / 2, (1 - rank) * xw / 2, taken + 1, s.limit_ns)?;
        b.tp_mailbox_take(self.mailbox, y, rows, yw, yw / 2, (1 - rank) * yw / 2, taken + 2, s.limit_ns)
    }
}

impl<'a> NextModel<'a> {
    /// Both ranks, before serving: this rank's half and the mailbox. The proxy thread then takes
    /// the mailbox's one host endpoint (tp_mailbox_host).
    pub fn set_tp(&self, b: &Backend<'a>, rank: usize, link: std::net::TcpStream) -> Result<()> {
        if rank > 1 || !crate::flash_tp_switch::on() {
            return Err("the tensor split needs world 2 and rank 0 or 1".into());
        }
        let layout = layout(&self.checkpoint.config)?;
        // The world-one MoE path this split reuses sizes its buffers from the same geometry.
        if layout.hidden != H || layout.residual != W || layout.experts != EXPERTS {
            return Err(format!("tensor split geometry {layout:?} differs from the loaded world-one layout"));
        }
        if self.tp_box.set(b.tp_mailbox(layout.slot_bytes())?).is_err() {
            return Err("the tensor split mailbox is already set".into());
        }
        let twins = self.tp_half_twins(b, rank)?;
        let heads = Some(heads::GdnHeads::new(self, b, rank)?);
        *self.tp.borrow_mut() = Some(Swap { rank, layout, sent: 0, taken: 0, limit_ns: pp::LINK_TIMEOUT.as_nanos() as u64,
            mail_pass: false, tile_pass: false, link, handoff: Vec::new(), front_tile: false, _twins: twins, heads, heads_pass: false,
            verify_pass: false });
        eprintln!("flashnext tp rank={rank} split={SPLIT} rows<={} (verify) <={} (other) layout={layout:?}", TpLayout::ROWS,
            TpLayout::PROMPT_ROWS);
        Ok(())
    }
    /// This rank's row half [rank * n / 2, (rank + 1) * n / 2) of every split layer's GDN z
    /// in-projection, GDN and QSA out projection, QSA q/gate projection and shared expert down
    /// projection twin, the qkv twin's rows of this rank's q, k and v heads, and the qkv twin's
    /// row half. Rows are 64-row blocks first in both the packed BF16 order
    /// and the twin's bytes, and the twin's scales are [row / 8][K / 32][8], so a row range is two
    /// contiguous ranges of the loaded twin, registered under its BF16 rows' address.
    fn tp_half_twins(&self, b: &Backend<'a>, rank: usize) -> Result<Vec<blocks::Mx8Twin<'a>>> {
        let c = &self.checkpoint.config;
        let hidden = c.hidden as usize;
        let value = (c.linear_value_heads * c.linear_value_dim) as usize;
        let keys = (c.linear_key_heads * c.linear_key_dim) as usize;
        let rows = [2 * keys + value, value];
        // The split also cuts BF16 matrices of `hidden` rows at their row half without a twin.
        if !(hidden / 2).is_multiple_of(64) { return Err(format!("tensor split: hidden {hidden} has no 64-row halves")); }
        let mut twins = Vec::new();
        for li in SPLIT..self.gdn_proj.len() {
            // (BF16 matrix, rows, K, first row, rows of this rank); absent matrices are 0.
            let half = |w: u64, n: usize, k: usize| (w, n, k, rank * n / 2, n / 2);
            let (qkv, n) = (self.gdn_proj[li][0], rows[0]);
            let split = [(qkv, n, hidden, rank * keys / 2, keys / 2), (qkv, n, hidden, keys + rank * keys / 2, keys / 2),
                (qkv, n, hidden, 2 * keys + rank * value / 2, value / 2), half(self.gdn_proj[li][1], value, hidden),
                // Passes before a chat's states become halves run the qkv row half.
                half(qkv, n, hidden),
                half(self.gdn_proj[li][4], hidden, value), half(self.qsa_proj[li][4], hidden, (c.query_heads * c.head_dim) as usize),
                half(self.qsa_proj[li][0], (c.query_heads * 2 * c.head_dim) as usize, hidden),
                half(self.moe[li].shared[2], hidden, c.shared_width as usize)];
            for (w, n, k, start, part) in split {
                if w == 0 { continue; }
                let Some((full, r, cols)) = crate::flash_mx8_registry::get(w, n).filter(|&(_, r, cols)| (r, cols) == (n, k)) else { continue };
                if start % 64 != 0 || part % 64 != 0 || k % 32 != 0 {
                    return Err(format!("tensor split: projection [{r}][{cols}] has no 64-row ranges at row {start}"));
                }
                let source = self.mx8.iter().find(|t| t.twin.device_address() == full).ok_or("tensor split: loaded twin missing")?;
                let mut twin = b.vector(blocks::mx8_twin_bytes(part, k) / 4)?;
                let scales = |rows: usize| (rows * k).div_ceil(256) * 256 / 4;
                let scale_words = |rows: usize| rows / 8 * (k / 32) * 2;
                b.copy_range(&source.twin, &mut twin, start * k / 4, 0, part * k / 4)?;
                b.copy_range(&source.twin, &mut twin, scales(n) + scale_words(start), scales(part), scale_words(part))?;
                twins.push(blocks::Mx8Twin::register(w + (start * k * 2) as u64, twin, part, k));
            }
        }
        b.synchronize()?;
        Ok(twins)
    }
    /// A split layer at <= ROWS rows of one chat: each rank runs its row half of the GDN
    /// qkv and z in-projections (same per-row chains and merge as the whole launch) into its
    /// columns of the full outputs, a and b whole, then the ranks swap the halves as raw bits.
    /// False: not split here (the caller runs the whole projection).
    pub(super) fn tp_gdn_project(&self, b: &Backend<'a>, li: usize, gdn: &gdn::Gdn<'a, '_>, s: &mut gdn::GdnScratch<'a>,
        input: &DeviceVector<'a>, count: usize, chats: usize) -> Result<bool> {
        // One chat only: a pass of several chats carries more rows, and its larger halves cost
        // more to exchange than the split saves. Both ranks run the same chats, so both agree.
        if chats != 1 { return Ok(false); }
        let Some(moe) = self.tp_moe(li, count) else { return Ok(false) };
        if gdn.is_half() { return self.tp_heads_project(b, li, moe.rank, s, input, count).map(|()| true); }
        let ([qkv, z, a, bb], k) = gdn.projection();
        let half = |(w, n): (u64, usize), y: u64| (w + (moe.rank * n / 2 * k * 2) as u64, y + (moe.rank * n / 2 * 4) as u64, n / 2, n);
        blocks::next_rows_group_at(b, input.device_address(), k, count, [half(qkv, s.qkv.device_address()), half(z, s.z.device_address()),
            (a.0, s.a.device_address(), a.1, a.1), (bb.0, s.b.device_address(), bb.1, bb.1)])?;
        moe.halves_pair(b, [(&mut s.qkv, qkv.1), (&mut s.z, z.1)], count)?;
        Ok(true)
    }
    /// A split layer at <= ROWS rows of one chat: each rank runs its row half
    /// [rank * n / 2, (rank + 1) * n / 2) of the mixer's out projection ([n][k] BF16 at `weight`
    /// over the `x` rows, n = hidden; same per-row chains and merge as the whole launch) into its
    /// columns of `output`, then the ranks swap the halves as raw bits (one message per layer).
    /// False: not split here (the caller projects whole).
    pub(super) fn tp_out_project(&self, b: &Backend<'a>, li: usize, (weight, x, k): (u64, u64, usize), output: &mut DeviceVector<'a>,
        count: usize, chats: usize) -> Result<bool> {
        // One chat only, as tp_gdn_project: wider passes exchange larger halves.
        if chats != 1 { return Ok(false); }
        let Some(moe) = self.tp_moe(li, count) else { return Ok(false) };
        let (n, half) = (moe.layout.hidden, moe.layout.out_cols());
        blocks::next_rows_strided(b, weight + (moe.rank * half * k * 2) as u64, x, output.device_address() + (moe.rank * half * 4) as u64,
            half, k, count, n)?;
        moe.halves(b, output, count, n)?;
        Ok(true)
    }
    /// A split layer at <= ROWS rows of one chat: each rank runs its row half of
    /// the served QSA q/gate projection (Qsa::project_batch_half), then the ranks swap the halves
    /// as raw bits. False: the caller projects whole.
    pub(super) fn tp_qsa_project(&self, b: &Backend<'a>, li: usize, q: &Qsa<'a, '_>, batch: &mut crate::flash::qsa::QsaBatch<'a>,
        input: &DeviceVector<'a>, count: usize, chats: usize) -> Result<bool> {
        if chats != 1 { return Ok(false); }
        let Some(moe) = self.tp_moe(li, count) else { return Ok(false) };
        let Some(n) = q.project_batch_half(b, batch, input, count, moe.rank)? else { return Ok(false) };
        moe.halves(b, Qsa::batch_qg(batch), count, n)?;
        Ok(true)
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
    /// A split-mode pass wider than its kind's bound (TpLayout::split_rows) or a pipeline prompt
    /// tile: its residual crosses with the request, it splits no expert and runs without
    /// captured verify segments.
    pub(super) fn tp_host_pass(&self, count: usize) -> bool {
        self.tp_on() && !self.tp_split_rows(count)
    }
    /// The pass's rows cross in the mailbox and its layers split: at most TpLayout::split_rows
    /// rows (verify passes 32, others 16), and not a pipeline prompt tile (rank 0's front_tile,
    /// rank 1's TILE bit), whose layers run whole on one rank each.
    fn tp_split_rows(&self, count: usize) -> bool {
        self.tp.borrow().as_ref().is_some_and(|s| count <= TpLayout::split_rows(s.verify_pass) && !s.front_tile && !s.tile_pass)
    }
    /// Rank 1, per request: the residual arrives in the mailbox (MAIL bit), or the request is
    /// a pipeline tile for layers 24..48 (TILE bit).
    pub(super) fn tp_mail_next(&self, kind: u32) {
        let mail = kind & crate::flash_tp_switch::MAIL != 0;
        let tile = kind & crate::flash_tp_switch::TILE != 0;
        let heads = kind & crate::flash_tp_switch::HEADS != 0;
        let flags = crate::flash_tp_switch::MAIL | crate::flash_tp_switch::TILE | crate::flash_tp_switch::HEADS;
        let verify = kind & !flags == crate::flash_tp_switch::KIND_VERIFY;
        if let Some(s) = self.tp.borrow_mut().as_mut() { s.mail_pass = mail; s.tile_pass = tile; s.heads_pass = heads; s.verify_pass = verify; }
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
        let mut guard = self.tp.borrow_mut();
        let s = guard.as_mut().ok_or("tensor split missing")?;
        // A prompt pass runs as a pipeline tile when every chat's span fills at least half of
        // the engine's prompt tile (the pipeline's unit of work, an operator setting, not a
        // checkpoint figure), or continues a prompt that ran as tiles (the chat owes a handoff
        // and its last pass gave no logits): so the prompt's last rows and its first token run
        // on the pipeline path, and the handoff waits for the chat's next pass. Shorter
        // prompts stay in the lockstep and owe no state handoff.
        let tile = tile_pass(verify, crate::flash_next_rows::prompt_tile(), sessions.iter().zip(spans)
            .map(|(chat, span)| (span.len(), chat.pos > 0 && !chat.logits_ready && s.handoff.contains(&chat.slot))));
        if let Some(link) = self.pp.borrow_mut().as_mut() { link.tp_tile = tile; }
        s.front_tile = tile;
        s.verify_pass = verify;
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
    pub(super) fn tp_handoff<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], lens: &[usize]) -> Result<()> {
        // Rank 0 asks for these chats' GDN whole rebuild (next_tp_heads.rs).
        if !lens.is_empty() && lens.iter().all(|&n| n == heads::GDN_WHOLE as usize) { return self.tp_gdn_frame(b, sessions, usize::MAX); }
        if !lens.is_empty() && lens.iter().all(|&n| n == heads::GDN_PROBE as usize) { return self.tp_gdn_probe(); }
        let link = self.tp.borrow().as_ref().map(|s| s.link.try_clone()).ok_or("tensor split missing")?.map_err(err)?;
        let swapped = self.tp_swap_state(b, sessions, LAYERS / 2..LAYERS, SPLIT..LAYERS / 2, link);
        if swapped.is_err() { sessions.iter_mut().for_each(|chat| chat.valid = false); }
        swapped
    }
    /// Both ranks: send this rank's session state of `send` layers and install the peer's of
    /// `recv` layers, as raw FP32 words (GDN recurrent state, conv history and cached flag;
    /// QSA keys, values and raw index rows 0..past, and past), chats in the given order. The
    /// split link carries four head words per layer (QSA past or the GDN cached flag, then the
    /// words of each state vector, which the receiver checks against its own); the state words
    /// cross device to device in the mailbox, in pieces of at most one slot, and no host
    /// touches them. Message i carries this rank's piece i (or one pad word once its pieces
    /// end) and both ranks put message i + 1 before they take message i, so neither runs more
    /// than one message ahead of the peer's takes (the mailbox's lockstep).
    fn tp_swap_state<'w>(&self, b: &Backend<'a>, sessions: &mut [&mut NextSession<'a, 'w>], send: std::ops::Range<usize>,
        recv: std::ops::Range<usize>, link: std::net::TcpStream) -> Result<()> {
        let started = Instant::now();
        let mailbox = self.tp_box.get().ok_or("tensor split mailbox missing")?;
        let slot = mailbox.slot_bytes() / 4;
        let (mut head, mut sends, mut recvs) = (Vec::new(), Vec::new(), Vec::new());
        for (c, chat) in sessions.iter_mut().enumerate() {
            for li in send.clone() {
                let (flag, words) = layer_words(chat, li, None)?;
                head.extend(std::iter::once(flag).chain(words).flat_map(|w| (w as u64).to_le_bytes()));
                for (k, n) in words.into_iter().enumerate() { pieces(&mut sends, (c, li, k), n, slot); }
            }
        }
        let heads = sessions.len() * recv.len() * 32;
        let arrived = exchange(&link, &head, heads)?;
        if arrived.len() != heads {
            return Err("tensor split handoff head has the wrong length".into());
        }
        let mut fields = arrived.as_chunks::<8>().0.iter().map(|w| usize::try_from(u64::from_le_bytes(*w)).unwrap_or(usize::MAX));
        for (c, chat) in sessions.iter_mut().enumerate() {
            for li in recv.clone() {
                let mut field = || fields.next().ok_or("tensor split handoff head too short");
                let flag = field()?;
                let peer = [field()?, field()?, field()?];
                let (_, words) = layer_words(chat, li, Some(flag))?;
                if words != peer {
                    return Err("tensor split handoff: the peer's state sizes differ from this rank's".into());
                }
                for (k, n) in words.into_iter().enumerate() { pieces(&mut recvs, (c, li, k), n, slot); }
            }
        }
        let (pad, mut pad_in) = (b.vector(1)?, b.vector(1)?);
        let (mut sent, mut taken, limit) = self.tp.borrow().as_ref().map(|s| (s.sent, s.taken, s.limit_ns)).ok_or("tensor split missing")?;
        let messages = sends.len().max(recvs.len());
        let put = |sessions: &mut [&mut NextSession<'a, 'w>], i: usize, sent: &mut u64| -> Result<()> {
            *sent += 1;
            match sends.get(i) {
                Some(&Piece { at: (c, li, k), from, words }) => {
                    let v = state_vector(sessions[c], li, k)?;
                    b.tp_mailbox_put(mailbox, v, 1, v.len(), words, from, *sent)
                }
                None => b.tp_mailbox_put(mailbox, &pad, 1, 1, 1, 0, *sent),
            }
        };
        let mut swapped = if messages > 0 { put(sessions, 0, &mut sent) } else { Ok(()) };
        for i in 0..messages {
            if swapped.is_err() { break; }
            if i + 1 < messages { swapped = put(sessions, i + 1, &mut sent); }
            taken += 1;
            swapped = swapped.and_then(|()| match recvs.get(i) {
                Some(&Piece { at: (c, li, k), from, words }) => {
                    let v = state_vector(sessions[c], li, k)?;
                    let width = v.len();
                    b.tp_mailbox_take(mailbox, v, 1, width, words, from, taken, limit)
                }
                None => b.tp_mailbox_take(mailbox, &mut pad_in, 1, 1, 1, 0, taken, limit),
            });
        }
        if let Some(s) = self.tp.borrow_mut().as_mut() { (s.sent, s.taken) = (sent, taken); }
        swapped?;
        b.synchronize()?;
        if self.tp_failed() {
            return Err("tensor split mailbox timed out or its proxy failed during the handoff".into());
        }
        let bytes = |p: &[Piece]| p.iter().map(|p| p.words * 4).sum::<usize>();
        eprintln!("flashnext tp handoff chats={} sent_bytes={} received_bytes={} messages={messages} ms={:.1}", sessions.len(), bytes(&sends),
            bytes(&recvs), started.elapsed().as_secs_f64() * 1e3);
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
    /// The split head: `rows` rows of `width` words from rank 1 to rank 0, queued in the
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
        self.tp_on() && (li + 1 == SPLIT || li >= SPLIT && self.tp_split_rows(count))
    }
    pub(super) fn tp_moe(&self, li: usize, count: usize) -> Option<TpMoe<'_, 'a>> {
        let (rank, layout) = self.tp.borrow().as_ref().map(|s| (s.rank, s.layout))?;
        let mailbox = self.tp_box.get()?;
        (li >= SPLIT && self.tp_split_rows(count)).then_some(TpMoe { rank, layout, swap: &self.tp, mailbox })
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
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("tensor split link failed earlier; the front is stopping")?;
        let heads = self.tp.borrow().as_ref().is_some_and(|s| s.heads_pass);
        let posted = link.drain().and_then(|()| {
            // A row buffer the sender thread returned (pages already mapped): a new vector per
            // pass can get new pages from the allocator, and the readback then pays their faults
            // (5 MB in about 45 ms against under 1 ms, 2026-10-26).
            let mut rows = link.spare(count * W);
            if mail { rows.fill(0.0); } else { b.read_vector_range_into(residual, 0, &mut rows)?; }
            link.tp_verify = verify;
            link.tp_mail = mail;
            link.tp_heads = heads;
            link.post(&meta, rows, None)
        });
        posted.map_err(|e| link.fail(e))
    }
}

/// tp_part's tile test over each chat's (span rows, continues a tiled prompt): not a verify
/// pass, and every span fills at least half of the prompt tile `tile` or continues.
fn tile_pass(verify: bool, tile: usize, chats: impl IntoIterator<Item = (usize, bool)>) -> bool {
    !verify && chats.into_iter().all(|(rows, continues)| 2 * rows >= tile || continues)
}
/// Both ranks: send `out` (length-prefixed) on the split link while reading the peer's message
/// (refused above `limit` bytes before anything is allocated).
fn exchange(link: &std::net::TcpStream, out: &[u8], limit: usize) -> Result<Vec<u8>> {
    use std::io::{Read, Write};
    let mut link = link.try_clone().map_err(err)?;
    link.set_nonblocking(false).map_err(err)?;
    link.set_read_timeout(Some(pp::LINK_TIMEOUT)).map_err(err)?;
    let mut writer = link.try_clone().map_err(err)?;
    let mut arrived = Vec::new();
    std::thread::scope(|scope| -> Result<()> {
        let sender = scope.spawn(|| -> std::io::Result<()> { writer.write_all(&(out.len() as u64).to_le_bytes())?; writer.write_all(out) });
        let mut len = [0u8; 8];
        link.read_exact(&mut len).map_err(err)?;
        arrived.resize(handoff_len(len, limit)?, 0);
        link.read_exact(&mut arrived).map_err(err)?;
        sender.join().map_err(|_| "tensor split handoff sender panicked")?.map_err(err)
    })?;
    Ok(arrived)
}
/// One handoff message's words: `words` words from word `from` of state vector `at` (chat,
/// layer, vector: QSA keys, values, raw index; GDN state, conv history).
struct Piece { at: (usize, usize, usize), from: usize, words: usize }
/// `words` words of state vector `at` as pieces of at most `slot` words, in order.
fn pieces(out: &mut Vec<Piece>, at: (usize, usize, usize), words: usize, slot: usize) {
    out.extend((0..words).step_by(slot).map(|from| Piece { at, from, words: slot.min(words - from) }));
}
/// A chat's layer `li` in the handoff: its head word (QSA past, the GDN cached flag) and the words
/// of each state vector that cross (QSA keys, values, raw index rows 0..past; GDN state, conv
/// history). `install`: the peer's head word, set here, and the words to receive for it.
fn layer_words(chat: &mut NextSession<'_, '_>, li: usize, install: Option<usize>) -> Result<(usize, [usize; 3])> {
    if li % 4 == 3 {
        let (vectors, widths, past) = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the handoff")?.handoff()?;
        let rows = install.unwrap_or(*past);
        let mut words = [0; 3];
        for (k, (v, w)) in vectors.into_iter().zip(widths).enumerate() {
            if w != 0 && rows != 0 { words[k] = handoff_words(rows, w, v.len())?; }
        }
        *past = rows;
        Ok((rows, words))
    } else {
        let (state, history, cached) = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the handoff")?.handoff();
        if let Some(flag) = install { *cached = flag != 0; }
        Ok((usize::from(*cached), [state.len(), history.len(), 0]))
    }
}
/// State vector `k` of a chat's layer `li` (the order tp_swap_state's pieces name).
fn state_vector<'s, 'a>(chat: &'s mut NextSession<'a, '_>, li: usize, k: usize) -> Result<&'s mut DeviceVector<'a>> {
    if li % 4 == 3 {
        let (vectors, _, _) = chat.qsa_state[li].as_mut().ok_or("QSA state missing for the handoff")?.handoff()?;
        vectors.into_iter().nth(k).ok_or_else(|| "QSA handoff vector missing".into())
    } else {
        let (state, history, _) = chat.gdn_state[li].as_mut().ok_or("GDN state missing for the handoff")?.handoff();
        Ok(if k == 0 { state } else { history })
    }
}
/// A handoff message's length field, refused above `limit` (the receiving chats' whole state)
/// before anything is allocated.
fn handoff_len(field: [u8; 8], limit: usize) -> Result<usize> {
    usize::try_from(u64::from_le_bytes(field)).ok().filter(|&n| n <= limit)
        .ok_or_else(|| "tensor split handoff message longer than the chats' state".into())
}
/// Words of `rows` QSA rows of `width`, refused past the state vector's `len` words.
fn handoff_words(rows: usize, width: usize, len: usize) -> Result<usize> {
    rows.checked_mul(width).filter(|&n| n <= len).ok_or_else(|| "tensor split handoff QSA rows exceed the state".into())
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
    /// A peer's handoff length beyond the receiving state, or one that does not fit usize, and QSA
    /// rows past the state's capacity (or whose word count overflows) are refused, not allocated.
    #[test]
    fn handoff_bounds_refuse_lengths_and_rows_past_the_state() {
        assert_eq!(handoff_len(64u64.to_le_bytes(), 64).unwrap(), 64);
        assert!(handoff_len(65u64.to_le_bytes(), 64).is_err());
        assert!(handoff_len(u64::MAX.to_le_bytes(), usize::MAX - 1).is_err());
        assert_eq!(handoff_words(3, 128, 4 * 128).unwrap(), 384);
        assert!(handoff_words(5, 128, 4 * 128).is_err());
        assert!(handoff_words(1 << 62, 128, 4 * 128).is_err(), "the row count's product must not wrap");
    }
    /// A state vector crosses as whole slots and one shorter last piece, in order; none when empty.
    #[test]
    fn handoff_pieces_cover_each_vector_once_in_order() {
        let mut out = Vec::new();
        pieces(&mut out, (0, 3, 1), 10, 4);
        pieces(&mut out, (0, 3, 2), 0, 4);
        pieces(&mut out, (1, 4, 0), 8, 4);
        let got: Vec<_> = out.iter().map(|p| (p.at, p.from, p.words)).collect();
        assert_eq!(got, [((0, 3, 1), 0, 4), ((0, 3, 1), 4, 4), ((0, 3, 1), 8, 2), ((1, 4, 0), 0, 4), ((1, 4, 0), 4, 4)]);
    }
    /// A long prompt's tail (rows under half the tile, the chat owes a handoff) stays a pipeline
    /// tile; a short prompt and every verify pass stay split; one short new chat splits the pass.
    #[test]
    fn prompt_tail_of_a_tiled_prompt_stays_a_pipeline_tile() {
        assert!(tile_pass(false, 8192, [(8192, false)]));
        assert!(tile_pass(false, 8192, [(4096, false)]));
        assert!(!tile_pass(false, 8192, [(4095, false)]));
        for rows in [1, 16, 17, 63, 4095] { assert!(tile_pass(false, 8192, [(rows, true)]), "{rows}"); }
        assert!(!tile_pass(true, 8192, [(8192, false)]));
        assert!(!tile_pass(true, 8192, [(4, true)]));
        assert!(tile_pass(false, 8192, [(32, true), (8192, false)]));
        assert!(!tile_pass(false, 8192, [(32, true), (100, false)]));
    }
}
